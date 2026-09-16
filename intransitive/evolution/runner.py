"""Serial, journaled search using the paired official-game harness from #54."""
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import random
import signal
import threading
import time

from ..heuristics.config import SearchConfig
from ..heuristics.tuning import BOUNDS, Genome, saturation_report
from ..tournament.report import report
from ..tournament.runner import FINAL, atomic_json, play_match, replay
from ..tournament.spec import candidate, digest, manifest, unpack
from .strategy import Settings, diversity, initialize, next_population
from .engines import EnginePool

SCHEMA = 'intransitive-evolution-v1'


def genome(data):
    return Genome.from_json(json.dumps(data, allow_nan=False))


def frozen(g, role='population'):
    return candidate(g.config_hash, genome=g.to_dict(), role=role)


def implementation():
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(Path(__file__).parent.glob('*.py'))})


def prepare(settings, positions, limits, *, revision):
    """Freeze bounds, opponents, corpus split and complete compute protocol."""
    incumbent = Genome.from_genes()
    archive = Genome.from_genes({'pressure': 10.})
    # This also replays the entire corpus and validates backend/search inputs.
    check = manifest([frozen(incumbent), frozen(archive, 'archive')], positions, [limits],
                     position_limit=settings.search_positions)
    # Fresh validation starts AND generation-line clusters for each generation.
    validation, seen = [], set()
    for p in sorted(positions, key=lambda p: p['sha256']):
        if p['pool'] == 'validation' and p['seed'] not in seen:
            seen.add(p['seed'])
            validation.append(p['sha256'])
    required = settings.generations * settings.validation_positions
    if len(validation) < required:
        raise ValueError(f'Need {required} distinct validation lines; generate a larger frozen corpus')
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError('Implementation revision is required')
    result = dict(schema=SCHEMA, settings=settings.to_dict(), positions=positions,
                  limits=limits, bounds={k: list(v) for k, v in BOUNDS.items()},
                  incumbent=incumbent.to_dict(), archive=archive.to_dict(),
                  search=check['selected_positions'], validation=validation[:required],
                  implementation=implementation(), backend=check['candidates'][0]['backend_version'],
                  revision=revision,
                  policy=dict(opponents='all contemporaries, fixed default, fixed pressure-10 archive, '
                              'up to hall_size previously validation-eligible winners; deduplicate identities',
                              selection='harness eligibility then wins/scheduled then stable config identity; '
                              'saturation/preflight failures are ineligible',
                              validation='one apparent winner versus default and frozen history on fresh '
                              'validation lines each generation; default receives the same-size slate',
                              hall='append only validation-eligible winners; FIFO, unique, bounded',
                              budget='serial; reserve worst-case logical work before each attempt, including '
                              'two 10000-work warmups; never refund attempts, including interrupted work',
                              mutation='per-gene Bernoulli; zero toggle/uniform activation; Gaussian in log '
                              'space for positive scales, clipped to contract bounds; uniform crossover; '
                              '100 unsuccessful proposals trigger random immigrants',
                              evidence='cache reuse is not a new independent observation; heldout is never evaluated'))
    result['policy']['processes'] = ('bounded LRU of isolated immutable candidates; reset Python/NumPy seed '
                                    'and handshake each game; fresh search/evaluation state each move')
    result['sha256'] = digest(result)
    return result


def validate(spec):
    rebuilt = prepare(Settings(**spec['settings']), spec['positions'], spec['limits'], revision=spec['revision'])
    if rebuilt != spec:
        raise ValueError('Changed/stale optimizer manifest, code, backend, or corpus')


def smoke_receipt(path, spec):
    path = Path(path)
    smoke = json.loads((path / 'manifest.json').read_text())
    receipt = json.loads((path / 'report.json').read_text())
    validate(smoke)
    if (receipt['manifest'] != smoke['sha256'] or receipt['status'] != 'complete'
            or any(receipt['outcomes'][s] for s in ('crash', 'illegal_move', 'infrastructure_timeout'))
            or receipt['unique_matches'] < 2
            or smoke['backend'] != spec['backend'] or smoke['implementation'] != spec['implementation']
            or smoke['limits']['mode'] != spec['limits']['mode']):
        raise ValueError('A completed smoke without operational failures on this backend/code/protocol mode is required')
    return dict(manifest=smoke['sha256'], report=digest(receipt))


def batch(spec, population, hall, generation, phase):
    """Use one common opening batch and one opponent mixture for the population."""
    settings = Settings(**spec['settings'])
    ids = spec['search'] if phase == 'search' else spec['validation'][
        generation*settings.validation_positions:(generation+1)*settings.validation_positions]
    items = {g.config_hash: frozen(g) for g in population}
    for g, role in [(genome(spec['incumbent']), 'incumbent'), (genome(spec['archive']), 'archive')] + [
            (genome(g), 'archive') for g in hall]:
        items.setdefault(g.config_hash, frozen(g, role))
    # Each batch has only its selected positions; the run manifest owns the
    # complete frozen corpus and verifies all splits before scheduling.
    positions = [p for p in spec['positions'] if p['sha256'] in ids]
    return manifest(list(items.values()), positions, [spec['limits']], pool=phase)


def jobs(spec):
    """Canonical pair manifests make journals reusable across changing populations.

    Identity covers both effective candidates/backend, full start/history,
    opening seed, colours, and every search/safety/ranking setting. Names, roles,
    task indices and the surrounding population cannot create new evidence.
    """
    items = {c['sha256']: c for c in spec['candidates']}
    starts = {p['sha256']: p for p in spec['positions']}
    for task in spec['tasks']:
        pair = [frozen(genome(items[h]['genome'])) for h in task['candidates']]
        start = starts[task['position']]
        canonical = manifest(pair, [start], [spec['protocols'][task['protocol']]], pool=start['pool'])
        local = next(t for t in canonical['tasks'] if t['colours'] == task['colours'])
        key = digest(dict(manifest=canonical['sha256'], task=local['id']))
        yield task, key, canonical, local


def tuple_tree(value):
    return tuple(map(tuple_tree, value)) if isinstance(value, list) else value


class Stopped(Exception):
    pass


class Search:
    def __init__(self, spec, output, cancelled, *, match_player=play_match, preflight=saturation_report, engine_pool=None):
        self.spec, self.output, self.cancelled = spec, Path(output), cancelled
        self.settings = Settings(**spec['settings'])
        self.match_player, self.preflight = match_player, preflight
        self.engine_pool = engine_pool
        self.path = self.output / 'checkpoint.json'
        self.rng = random.Random(self.settings.seed)
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state['manifest'] != spec['sha256']:
                raise ValueError('Checkpoint belongs to a different run')
            self.rng.setstate(tuple_tree(self.state['rng']))
            # A hard crash leaves a lease. Count all elapsed time conservatively,
            # including downtime, instead of silently resetting the wall budget.
            if self.state['wall_lease'] is not None:
                self.state['wall_seconds'] += max(0., time.time()-self.state['wall_lease'])
        else:
            excluded = {genome(spec[k]).config_hash for k in ('incumbent', 'archive')}
            population = initialize(self.rng, self.settings, excluded)
            self.state = dict(manifest=spec['sha256'], generation=0, phase='search',
                              population=[g.to_dict() for g in population], hall=[], ledger={},
                              preflights={}, history=[], seen={}, exports=[],
                              games_reserved=0, nodes_reserved=0, wall_seconds=0., wall_lease=None,
                              stop_reason=None, rng=None)
        self.start = time.monotonic()
        self.previous_wall = self.state['wall_seconds']

    def save(self, *, release=False):
        self.state['rng'] = self.rng.getstate()
        self.state['wall_seconds'] = self.previous_wall + time.monotonic()-self.start
        self.state['wall_lease'] = None if release else time.time()
        atomic_json(self.path, self.state)

    def remaining(self):
        return self.settings.max_seconds - self.previous_wall - (time.monotonic()-self.start)

    def check(self):
        if self.remaining() <= 0:
            raise Stopped('wall_budget')
        if self.cancelled.is_set():
            raise Stopped('cancelled')

    def reserve(self, nodes, games=0):
        self.check()
        if self.state['games_reserved'] + games > self.settings.max_games:
            raise Stopped('game_budget')
        if self.state['nodes_reserved'] + nodes > self.settings.max_nodes:
            raise Stopped('node_budget')
        self.state['games_reserved'] += games
        self.state['nodes_reserved'] += nodes
        self.save()  # Durable reservation BEFORE starting expensive work.

    def preflights(self, candidates):
        starts = [p for p in self.spec['positions'] if p['sha256'] in self.spec['search']]
        for item in candidates:
            g = genome(item['genome'])
            identity = g.config_hash
            self.state['seen'][identity] = g.to_dict()
            if identity in self.state['preflights']:
                continue
            limits = SearchConfig(**self.spec['limits']['search'])
            self.reserve(limits.node_limit * len(starts))
            # Preflight uses only the search pool. A failure or saturation makes
            # the candidate ineligible, never a source of positive fitness.
            from dataclasses import replace
            limits = replace(limits, time_limit=min(limits.time_limit, max(.001, self.remaining()/len(starts))))
            try:
                result = self.preflight(g, [unpack(p['state']) for p in starts], base=limits)
            except Exception as exc:
                result = dict(flagged=True, error=f'{type(exc).__name__}: {exc}')
            self.state['preflights'][identity] = result
            self.save()
            self.check()

    def evaluate(self, schedule):
        if self.engine_pool is not None:
            self.engine_pool.retain(schedule['candidates'], self.spec['limits'])
        self.preflights(schedule['candidates'])
        rows, keys, hits = [], [], 0
        for task, key, canonical, local in jobs(schedule):
            self.check()
            folder = self.output / 'matches' / key
            path = folder / 'game.json'
            row = json.loads(path.read_text()) if path.exists() else None
            if row is not None:
                if row['task'] != local or row['manifest_sha256'] != canonical['sha256'] or row['colours'] != local['colours']:
                    raise ValueError('Cached match identity mismatch')
                replay(canonical['positions'][0], row)
                previous = self.state['ledger'].get(key)
                if previous and previous['final'] and previous['record_sha256'] != digest(row):
                    raise ValueError('Cached final match changed')
            if row is None or row['status'] not in FINAL:
                limits = canonical['protocols'][0]
                self.reserve(20_000 + limits['max_plies']*limits['search']['node_limit'], 1)
                atomic_json(folder / 'manifest.json', canonical)
                row = self.match_player(canonical, local, path, self.cancelled)
                # The harness fsyncs each move and final status before returning.
            else:
                hits += 1
            self.state['ledger'][key] = dict(record=str(path.relative_to(self.output)),
                                           record_sha256=digest(row), final=row['status'] in FINAL,
                                           status=row['status'], trajectory=row['trajectory_sha256'])
            self.save()
            if row['status'] not in FINAL:
                self.check()
                raise Stopped('cancelled')
            adapted = deepcopy(row)
            adapted.update(task=task, manifest_sha256=schedule['sha256'])
            rows.append(adapted)
            keys.append(key)
        result = report(schedule, rows)
        for entry in result['leaderboards'][self.spec['limits']['mode']]:
            item = next(c for c in schedule['candidates'] if c['sha256'] == entry['candidate'])
            identity = genome(item['genome']).config_hash
            entry['config_hash'] = identity
            entry['preflight'] = self.state['preflights'][identity]
            entry['depth_violations'] = sum(
                m['result']['completed_depth'] < self.settings.min_completed_depth
                and m['result']['stop_reason'] != 'proven_result'
                for row in rows for m in row['moves'] if m['candidate'] == item['sha256'])
            entry['eligible'] = (entry['eligible'] and not entry['preflight']['flagged']
                                 and entry['depth_violations'] == 0)
        board = result['leaderboards'][self.spec['limits']['mode']]
        board.sort(key=lambda r: (not r['eligible'], -r['win_points_lower'], r['config_hash']))
        result.update(match_keys=keys, cache_hits=hits)
        return result

    def board(self, result):
        return result['leaderboards'][self.spec['limits']['mode']]

    def loop(self):
        while self.state['generation'] < self.settings.generations:
            self.check()
            generation = self.state['generation']
            population = [genome(g) for g in self.state['population']]
            if self.state['phase'] == 'search':
                schedule = batch(self.spec, population, self.state['hall'], generation, 'search')
                result = self.evaluate(schedule)
                ranking = [r['config_hash'] for r in self.board(result) if r['role'] == 'population']
                self.state['current'] = dict(generation=generation, search=result, ranking=ranking,
                                             diversity=diversity(population), population=self.state['population'],
                                             hall_before=deepcopy(self.state['hall']))
                self.state['phase'] = 'validation'
                self.save()
            current = self.state['current']
            winner = next(g for g in population if g.config_hash == current['ranking'][0])
            if self.state['phase'] == 'validation':
                # Include unchanged defaults as a population entry so both face
                # one another and exactly the same archived opponent slate.
                references = [g for g in self.state['hall'] if genome(g).config_hash != winner.config_hash]
                schedule = batch(self.spec, [winner, genome(self.spec['incumbent'])], references,
                                 generation, 'validation')
                result = self.evaluate(schedule)
                current['validation'] = result
                validated = next(r for r in self.board(result) if r['config_hash'] == winner.config_hash)
                trained = next(r for r in self.board(current['search']) if r['config_hash'] == winner.config_hash)
                eligible = validated['eligible'] and trained['eligible']
                export = dict(genome=winner.to_dict(), config=winner.to_config(SearchConfig(**self.spec['limits']['search'])).to_dict(),
                              manifest=winner.manifest(SearchConfig(**self.spec['limits']['search']),
                                                       implementation_revision=self.spec['revision']+':'+self.spec['backend']),
                              run_manifest=self.spec['sha256'], generation=generation,
                              search=trained, validation=validated,
                              validation_match_keys=result['match_keys'], eligible=eligible,
                              acceptance='UNACCEPTED: requires separate held-out validation (#56)')
                self.state['exports'].append(export)
                if eligible:
                    self.state['hall'] = [g for g in self.state['hall'] if genome(g).config_hash != winner.config_hash]
                    self.state['hall'].append(winner.to_dict())
                    self.state['hall'] = self.state['hall'][-self.settings.hall_size:]
                current.update(unique_candidates=len(self.state['seen']), games_reserved=self.state['games_reserved'],
                               nodes_reserved=self.state['nodes_reserved'], wall_seconds=self.state['wall_seconds'])
                self.state['history'].append(current)
                self.state['generation'] += 1
                if self.state['generation'] < self.settings.generations:
                    excluded = {genome(self.spec[k]).config_hash for k in ('incumbent', 'archive')}
                    population = next_population(population, current['ranking'], self.rng, self.settings, excluded)
                    self.state['population'] = [g.to_dict() for g in population]
                self.state['phase'] = 'search'
                del self.state['current']
                self.save()
        self.state['stop_reason'] = 'complete'

    def summary(self):
        ledger = self.state['ledger']
        rows = [json.loads((self.output / entry['record']).read_text()) for entry in ledger.values()]
        moves = [m for r in rows for m in r['moves']] + [r['failed_response'] for r in rows if r.get('failed_response')]
        return dict(schema=SCHEMA, manifest=self.spec['sha256'], settings=self.settings.to_dict(),
                    status=self.state['stop_reason'], generations=self.state['generation'],
                    unique_candidates=len(self.state['seen']), unique_matches=len(ledger),
                    unique_trajectories=len({r['trajectory_sha256'] for r in rows}),
                    failures=sum(r['status'] in FINAL-{'win', 'unfinished'} for r in rows),
                    outcomes={s: sum(r['status'] == s for r in rows) for s in sorted(FINAL | {'cancelled'})},
                    games_reserved=self.state['games_reserved'], nodes_reserved=self.state['nodes_reserved'],
                    measured_search_work=sum(m['result']['work'] for m in moves),
                    known_child_cpu_seconds=sum(m['cpu_seconds'] for m in moves)+sum(s['cpu_seconds'] for r in rows for s in r['startups']),
                    resident_engine_limit=self.settings.resident_engines,
                    reused_process_handshakes=sum(s.get('reused_process', False) for r in rows for s in r['startups']),
                    child_peak_rss_bytes=max((s['peak_rss_bytes'] for r in rows for s in r['startups']+r['moves']), default=0),
                    wall_seconds=self.state['wall_seconds'], convergence=self.state['history'],
                    eligible_exports=sum(e['eligible'] for e in self.state['exports']),
                    note='Convergence reuses common matches; generations are not independent samples. '
                         'Reserved work bounds include unobserved/killed work and warmup. No strength '
                         'or superiority claim follows from ineligible or unfinished games.')


def run(spec, output, *, cancelled=None, smoke=None, match_player=play_match, preflight=saturation_report):
    begin = time.monotonic()
    validate(spec)
    settings = Settings(**spec['settings'])
    short = (settings.max_seconds <= 180 and settings.max_games <= 16
             and settings.max_nodes <= 3_000_000 and settings.generations == 1 and settings.population == 2)
    # Tests can inject an engine-free match implementation. Every real run
    # beyond the explicit smoke envelope needs a successful smoke receipt.
    receipt = None
    if match_player is play_match and not short:
        if smoke is None:
            raise ValueError('Run a bounded smoke first and pass its directory as smoke=')
        receipt = smoke_receipt(smoke, spec)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cancelled = cancelled if cancelled is not None else threading.Event()
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output / 'manifest.json'
        if path.exists() and json.loads(path.read_text()) != spec:
            raise ValueError('Output contains an incompatible run')
        atomic_json(path, spec)
        if receipt is not None:
            atomic_json(output / 'smoke.json', receipt)
        pool = EnginePool(settings.resident_engines) if match_player is play_match else None
        if pool is not None:
            def pooled_match(manifest, task, path, cancelled):
                return play_match(manifest, task, path, cancelled, engine_factory=pool.acquire)
            match_player = pooled_match
        search = Search(spec, output, cancelled, match_player=match_player, preflight=preflight, engine_pool=pool)
        search.start = begin  # Charge input validation/initialization to the run.
        previous = {}
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, lambda *_: cancelled.set())
        timer = threading.Timer(max(0., search.remaining()), cancelled.set)
        timer.daemon = True
        timer.start()
        try:
            search.save()
            search.loop()
        except Stopped as exc:
            search.state['stop_reason'] = str(exc)
        finally:
            timer.cancel()
            if pool is not None:
                pool.close()
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            search.save(release=True)
        result = search.summary()
        atomic_json(output / 'report.json', result)
        atomic_json(output / 'candidates.json', search.state['exports'])
        return result
