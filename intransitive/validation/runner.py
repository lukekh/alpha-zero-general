"""Execute a frozen acceptance plan: parity, tactics, held-out matches, cost.

Every stage writes its result durably before the next one starts, so a run can
be interrupted and resumed without repeating a game or a measurement. The
matches go through the unchanged #54 harness, which owns journals, replay
verification, colour pairing, eligibility and uncertainty; this module only
decides what to schedule and records what came back.
"""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import platform
import signal
import sys
import threading
import time

from ..evolution.engines import EnginePool
from ..heuristics.config import SearchConfig
from ..tournament.report import report as match_report
from ..tournament.runner import atomic_json, peak_rss_bytes, play_match, replay, validate_manifest
from ..tournament.spec import digest, runtime_versions
from . import cost as cost_module
from . import parity as parity_module
from . import report as report_module
from . import tactics as tactics_module
from .spec import SCHEMA, Design, configs, search_config, validate

STAGES = ('parity', 'tactics', 'matches', 'cost')


class Stopped(Exception):
    """Cooperative stop: the plan keeps its completed stages and can resume."""


def search_states(plan, count):
    """Search-pool positions, used for every measurement that is not a match.

    Held-out starts are played once, by the matches. Saturation, determinism,
    native parity and the cost ladder ask questions about an evaluator rather
    than about a position, so they use the pool the optimizer already saw and
    leave the held-out set with exactly one use.
    """
    rows = sorted((p for p in plan['corpus'] if p['pool'] == 'search'),
                  key=lambda p: (p['stage'] != 'official', p['sha256']))
    if len(rows) < count:
        raise ValueError(f'Need {count} search-pool positions; the corpus has {len(rows)}')
    return rows[:count]


def match_limits(plan):
    """The shared search settings the fixed-depth protocol gives every entrant."""
    for item in plan['experiments']:
        limits = item['manifest']['protocols'][0]
        if limits['mode'] == 'depth':
            return SearchConfig(**limits['search'])
    return SearchConfig(**plan['experiments'][0]['manifest']['protocols'][0]['search'])


class Run:
    def __init__(self, plan, output, *, cancelled=None, max_seconds=None,
                 workers=1, skip_native=False, stages=STAGES, match_player=None):
        validate(plan)
        unknown = sorted(set(stages) - set(STAGES))
        if unknown:
            raise ValueError(f'Unknown stages: {unknown}')
        if type(workers) is not int or not 1 <= workers <= 8:
            raise ValueError('workers must be in [1, 8]')
        self.plan, self.output = plan, Path(output)
        self.stages = tuple(s for s in STAGES if s in stages)
        self.workers, self.skip_native = workers, skip_native
        # Tests inject an engine-free match implementation; a real run always
        # uses the harness's own player behind the resident engine pool.
        self.match_player = match_player
        self.cancelled = cancelled if cancelled is not None else threading.Event()
        self.max_seconds = max_seconds
        self.design = Design.from_dict(plan['design'])
        self.path = self.output / 'state.json'
        self.begin = time.monotonic()
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state['plan'] != plan['sha256']:
                raise ValueError('State belongs to a different acceptance plan')
        else:
            self.state = dict(plan=plan['sha256'], parity={}, tactics={}, experiments={},
                              cost={}, stop_reason=None, elapsed_seconds=0.)
        self.previous = self.state['elapsed_seconds']

    def save(self):
        self.state['elapsed_seconds'] = self.previous + time.monotonic() - self.begin
        atomic_json(self.path, self.state)

    def check(self):
        if self.cancelled.is_set():
            raise Stopped('cancelled')
        if self.max_seconds is not None and self.remaining() <= 0:
            raise Stopped('wall_budget')

    def remaining(self):
        if self.max_seconds is None:
            return float('inf')
        return self.max_seconds - self.previous - (time.monotonic() - self.begin)

    # -- stages ---------------------------------------------------------
    def parity(self):
        """Re-derive and check every entrant configuration in the plan."""
        positions = search_states(self.plan, self.design.cost_positions)
        states = cost_module.observations(positions)
        limits = match_limits(self.plan)
        for name, row in configs(self.plan).items():
            self.check()
            if name in self.state['parity']:
                continue
            self.state['parity'][name] = parity_module.evaluate(
                row['genome'], held_states=states, search_positions=positions,
                limits=limits, revision=self.plan['revision'], skip_native=self.skip_native)
            self.save()

    def tactics(self):
        """Both tactical profiles for every entrant configuration."""
        profiles = dict(certified=tactics_module.CERTIFIED_LIMITS,
                        shipped=tactics_module.SHIPPED_LIMITS)
        cases = tactics_module.puzzle_cases() + tactics_module.game_cases()
        if self.design.tactical_cases:
            cases = cases[:self.design.tactical_cases]
        for name, row in configs(self.plan).items():
            for profile, limits in sorted(profiles.items()):
                self.check()
                if self.state['tactics'].get(name, {}).get(profile):
                    continue
                result = tactics_module.evaluate(search_config(row), limits=limits, cases=cases)
                self.state['tactics'].setdefault(name, {})[profile] = result
                self.save()

    def play(self, spec, folder):
        """The #54 harness, with the #55 engine pool in front of it.

        Every game still gets a fresh player, evaluator and transposition table
        for every move; only compiled machine code and the immutable candidate
        survive between games. Paying eleven seconds of Numba startup per game
        would have cost this run more time than the games themselves, and the
        optimizer already measured fresh-versus-reused parity on finished games.
        """
        validate_manifest(spec)
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            target = folder / 'manifest.json'
            if target.exists() and json.loads(target.read_text()) != spec:
                raise ValueError('Output already contains a different manifest')
            atomic_json(target, spec)
            begin = time.perf_counter()
            # An equal-move-time protocol measures what a search finishes in a
            # given second. Running two of them at once would measure CPU
            # contention instead, so timing experiments are always serial
            # however many workers the fixed-depth ones are given.
            workers = 1 if spec['protocols'][0]['mode'] == 'wall' else self.workers
            # Two resident engines per concurrent match, plus headroom so an
            # opponent stays warm between games instead of being evicted.
            capacity = min(24, max(2 * workers + 2, len(spec['candidates']) + 1))
            pool = EnginePool(capacity, workers=workers) if self.match_player is None else None

            def execute(task):
                path = folder / 'matches' / (task['id'] + '.json')
                if self.cancelled.is_set() and not path.exists():
                    return None
                if pool is None:
                    return self.match_player(spec, task, path, self.cancelled)
                return play_match(spec, task, path, self.cancelled,
                                  engine_factory=pool.acquire)
            try:
                if workers > 1:
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        rows = [row for row in executor.map(execute, spec['tasks'])
                                if row is not None]
                else:
                    rows = [row for row in map(execute, spec['tasks']) if row is not None]
            finally:
                if pool is not None:
                    pool.close()
            summary = match_report(spec, rows)
            invocation = dict(workers=workers, requested_workers=self.workers,
                              resident_engine_limit=capacity if pool is not None else 0,
                              wall_seconds=time.perf_counter() - begin,
                              parent_peak_rss_bytes=peak_rss_bytes(),
                              reused_process_handshakes=sum(s.get('reused_process', False)
                                                            for row in rows
                                                            for s in row['startups']),
                              cancelled=self.cancelled.is_set())
            history = folder / 'invocations.json'
            invocations = json.loads(history.read_text()) if history.exists() else []
            invocations.append(invocation)
            atomic_json(history, invocations)
            summary['invocations'] = invocations
            atomic_json(folder / 'report.json', summary)
            return summary

    def matches(self):
        """Run each frozen experiment through the unchanged #54 harness."""
        for item in self.plan['experiments']:
            self.check()
            existing = self.state['experiments'].get(item['name'])
            if existing and existing.get('complete'):
                continue
            folder = self.output / 'experiments' / item['name']
            begin = time.monotonic()
            summary = self.play(item['manifest'], folder)
            complete = summary['final_matches'] == summary['scheduled_matches']
            self.state['experiments'][item['name']] = dict(
                report=summary, complete=complete, directory=str(folder),
                manifest_sha256=item['manifest']['sha256'],
                wall_seconds=time.monotonic() - begin)
            self.save()
            if not complete:
                raise Stopped('cancelled')

    def cost(self):
        """Price the candidates and their comparisons, one process each."""
        positions = search_states(self.plan, self.design.cost_positions)
        wanted = [row['name'] for row in self.plan['candidates']]
        wanted += [row['name'] for row in self.plan['references']]
        available = configs(self.plan)
        for name in wanted:
            self.check()
            if name in self.state['cost']:
                continue
            self.state['cost'][name] = cost_module.measure(
                available[name]['config'], positions, self.design.to_dict())
            self.save()

    # -- orchestration --------------------------------------------------
    def execute(self):
        for stage in self.stages:
            getattr(self, stage)()
        self.state['stop_reason'] = 'complete'

    def results(self):
        return dict(parity=self.state['parity'], tactics=self.state['tactics'],
                    experiments=self.state['experiments'], cost=self.state['cost'])

    def summary(self):
        results = self.results()
        rows = [row for item in self.state['experiments'].values()
                for row in [item['report']]]
        return dict(schema=SCHEMA, plan_sha256=self.plan['sha256'],
                    revision=self.plan['revision'], runtime=runtime_versions(),
                    platform=dict(python=sys.version, platform=platform.platform()),
                    status=self.state['stop_reason'], stages=list(self.stages),
                    elapsed_seconds=self.state['elapsed_seconds'],
                    parent_peak_rss_bytes=peak_rss_bytes(),
                    experiments={name: dict(
                        complete=item['complete'], wall_seconds=item['wall_seconds'],
                        scheduled=item['report']['scheduled_matches'],
                        final=item['report']['final_matches'],
                        distinct_starts=item['report']['distinct_starts'],
                        distinct_trajectories=item['report']['distinct_trajectories'],
                        duplicate_trajectories=item['report']['duplicate_trajectories'],
                        known_child_cpu_seconds=item['report']['known_child_cpu_seconds'],
                        child_peak_rss_bytes=item['report']['child_peak_rss_bytes'],
                        warnings=item['report']['warnings'])
                        for name, item in sorted(self.state['experiments'].items())},
                    total_scheduled=sum(r['scheduled_matches'] for r in rows),
                    total_final=sum(r['final_matches'] for r in rows),
                    total_child_cpu_seconds=sum(r['known_child_cpu_seconds'] for r in rows),
                    tactical_profiles=sorted({p for row in self.state['tactics'].values() for p in row}),
                    configurations=sorted(configs(self.plan)),
                    note='Held-out starts are played once, by these experiments. Parity, '
                         'saturation and cost measurements use search-pool positions. No '
                         'default, generator or trainer is changed by this run.')


def run(plan, output, *, workers=1, cancelled=None, max_seconds=None, skip_native=False,
        stages=STAGES, match_player=None):
    """Run or resume an acceptance plan, then decide it against its own thresholds."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = output / 'plan.json'
        if target.exists() and json.loads(target.read_text()) != plan:
            raise ValueError('Output already contains a different acceptance plan')
        atomic_json(target, plan)
        session = Run(plan, output, cancelled=cancelled, max_seconds=max_seconds,
                      workers=workers, skip_native=skip_native, stages=stages,
                      match_player=match_player)
        handlers = {}
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                handlers[signum] = signal.signal(signum, lambda *_: session.cancelled.set())
        timer = None
        if max_seconds is not None:
            timer = threading.Timer(max(0., session.remaining()), session.cancelled.set)
            timer.daemon = True
            timer.start()
        try:
            session.execute()
        except Stopped as exc:
            session.state['stop_reason'] = str(exc)
        finally:
            if timer is not None:
                timer.cancel()
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
            session.save()
        summary = session.summary()
        results = session.results()
        decision = report_module.verdict(plan, results)
        decision['defects'] = report_module.defects(plan, results)
        atomic_json(output / 'summary.json', summary)
        atomic_json(output / 'decision.json', decision)
        atomic_json(output / 'evidence.json', results)
        return dict(summary=summary, decision=decision)


def verify(plan, output):
    """Replay every recorded game and re-check every artifact digest."""
    validate(plan)
    output = Path(output)
    stored = json.loads((output / 'plan.json').read_text())
    if stored != plan:
        raise ValueError('The recorded plan is not this plan')
    checked, games = {}, 0
    for item in plan['experiments']:
        folder = output / 'experiments' / item['name']
        if not folder.exists():
            continue
        spec = json.loads((folder / 'manifest.json').read_text())
        if spec != item['manifest']:
            raise ValueError(f'{item["name"]}: recorded manifest differs from the plan')
        starts = {p['sha256']: p for p in spec['positions']}
        count = 0
        for task in spec['tasks']:
            path = folder / 'matches' / (task['id'] + '.json')
            if not path.exists():
                continue
            row = json.loads(path.read_text())
            if (row['task'] != task or row['manifest_sha256'] != spec['sha256']
                    or row['colours'] != task['colours']):
                raise ValueError(f'{item["name"]}: mismatched record identity')
            replay(starts[task['position']], row)
            count += 1
        checked[item['name']] = dict(verified=count, scheduled=len(spec['tasks']))
        games += count
    return dict(plan_sha256=plan['sha256'], experiments=checked, verified_games=games,
                evidence_sha256=digest(json.loads((output / 'evidence.json').read_text()))
                if (output / 'evidence.json').exists() else None)
