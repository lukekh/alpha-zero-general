"""Generate an independent, disk-backed extension of a running teacher corpus.

The current trainer and its immutable shards are read-only inputs. A READY file
marks a complete combined corpus; this program never stops or changes training.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import signal
import sqlite3
import subprocess
import time
import traceback
import zlib

from .supervised_minimax import (
    SETTINGS, STAGES, TEACHER, GameProcesses, label_job, write_json, valid_teacher_record,
    acknowledge_label_result,
)


class Catalog:
    """Compressed positions, exact orbit deduplication, and family isolation."""

    def __init__(self, path, target):
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA cache_size=-16384')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY, orbit TEXT NOT NULL UNIQUE,
                family TEXT NOT NULL, split TEXT NOT NULL, stage TEXT NOT NULL,
                origin TEXT NOT NULL, source TEXT NOT NULL, record BLOB NOT NULL);
            CREATE INDEX IF NOT EXISTS families ON positions(family);
            CREATE INDEX IF NOT EXISTS splits ON positions(split);
            CREATE INDEX IF NOT EXISTS stages ON positions(stage);
            CREATE INDEX IF NOT EXISTS origins ON positions(origin);
            CREATE TABLE IF NOT EXISTS blocked_families (family TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS imports (file TEXT PRIMARY KEY, sha256 TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        ''')
        self.target = target
        self.quotas = {s: target // 3 + int(i < target % 3) for i, s in enumerate(STAGES)}
        self.stage_counts = Counter(dict(self.db.execute('SELECT stage,count(*) FROM positions GROUP BY stage')))

    def exclude_extra_family(self, family):
        removed = self.db.execute('SELECT stage,count(*) FROM positions WHERE family=? AND origin="extra" GROUP BY stage',
                                  (family,)).fetchall()
        self.db.execute('DELETE FROM positions WHERE family=? AND origin="extra"', (family,))
        for stage, count in removed:
            self.stage_counts[stage] -= count
        self.db.execute('INSERT OR IGNORE INTO blocked_families VALUES (?)', (family,))

    def get(self, key, default=None):
        row = self.db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key, value):
        self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', (key, json.dumps(value)))

    def counts(self):
        result = {}
        for field in ('stage', 'split', 'origin', 'source'):
            result[field] = dict(self.db.execute(f'SELECT {field},count(*) FROM positions GROUP BY {field}'))
        result['total'] = sum(result['stage'].values())
        return result

    def insert(self, records, origin):
        """Called inside the caller's transaction, one label family at a time."""
        if not records:
            return 0
        if origin not in ('primary', 'extra'):
            raise ValueError('Unknown origin')
        family = records[0]['family']
        split = records[0]['split']
        if any(r['family'] != family or r['split'] != split for r in records):
            raise ValueError('A label family must have exactly one split')
        if origin == 'extra' and self.db.execute(
                'SELECT 1 FROM blocked_families WHERE family=?', (family,)).fetchone():
            return 0
        conflicts = []
        for record in records:
            if not valid_teacher_record(record):
                raise ValueError('Incomplete depth-five label')
            if not record['legal'][record['action']]:
                raise ValueError('Illegal teacher action')
            if record['provenance']['stage'] not in STAGES:
                raise ValueError('Invalid stage')
            old = self.db.execute('SELECT family,split,origin FROM positions WHERE orbit=?',
                                  (record['orbit'],)).fetchone()
            if old and old[1] != record['split']:
                conflicts.append(old)
        if origin == 'extra' and conflicts:
            # Exclude the whole incoming family, including any previous batch.
            self.exclude_extra_family(family)
            return 0
        for old_family, _, old_origin in conflicts:
            if old_origin == 'extra':
                # Primary data may already have trained the live model. It wins.
                self.exclude_extra_family(old_family)
        counts = self.stage_counts
        accepted = 0
        for record in records:
            stage = record['provenance']['stage']
            old = self.db.execute('SELECT origin,stage FROM positions WHERE orbit=?', (record['orbit'],)).fetchone()
            if old:
                if origin == 'primary' and old[0] == 'extra':
                    self.db.execute('DELETE FROM positions WHERE orbit=?', (record['orbit'],))
                    counts[old[1]] -= 1
                else:
                    continue
            if origin == 'extra' and counts[stage] >= self.quotas[stage]:
                continue
            source = 'ablation' if 'ablation' in record else record['provenance'].get('source', 'synthetic_root')
            self.db.execute('''INSERT INTO positions
                (orbit,family,split,stage,origin,source,record) VALUES (?,?,?,?,?,?,?)''',
                (record['orbit'],family,split,stage,origin,source,
                 zlib.compress(pickle.dumps(record, protocol=5), level=1)))
            counts[stage] += 1
            accepted += 1
        return accepted

    def import_primary(self, directory):
        manifest_path = directory / 'dataset-manifest.json'
        if not manifest_path.exists():
            return
        manifest = json.loads(manifest_path.read_text())
        for shard in manifest['shards']:
            old = self.db.execute('SELECT sha256 FROM imports WHERE file=?', (shard['file'],)).fetchone()
            if old:
                if old[0] != shard['sha256']:
                    raise ValueError('Primary shard changed after import')
                continue
            path = (directory / shard['file']).resolve()
            if not path.is_relative_to(directory.resolve() / 'dataset'):
                raise ValueError('Shard path outside primary dataset')
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != shard['sha256']:
                raise ValueError('Primary shard checksum mismatch')
            records = pickle.loads(gzip.decompress(raw))
            if len(records) != shard['positions']:
                raise ValueError('Primary shard count mismatch')
            families = {}
            for record in records:
                families.setdefault(record['family'], []).append(record)
            with self.db:
                for family in families.values():
                    self.insert(family, 'primary')
                self.db.execute('INSERT INTO imports VALUES (?,?)', (shard['file'], shard['sha256']))

    def close(self):
        self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        self.db.close()


def primary_active(directory):
    """Verify process identity rather than trust a stale status file or PID."""
    launch = json.loads((directory / 'launch.json').read_text())
    result = subprocess.run(['ps', '-p', str(int(launch['pid'])), '-o', 'command='],
                            capture_output=True, text=True, check=False)
    return result.returncode == 0 and str(directory / 'run.py') in result.stdout


SEARCH_OPTIMIZATION_FIELDS = frozenset((
    'pvs_enabled','aspiration_enabled','aspiration_window','ordering_enabled',
    'compiled_ordering_enabled','depth_replacement_enabled','table_entries','pressure_cache_entries',
))


def upgrade_config(saved, current, *, allow_depth_change=False, allow_search_optimization=False):
    """Explicit search-budget migration, preserving corpus identity and labels."""
    before = json.loads(json.dumps(saved))
    after = json.loads(json.dumps(current))
    previous_teacher = before['settings'].pop('teacher')
    next_teacher = after['settings'].pop('teacher')
    # This operational setting changes durability, never dataset identity.
    before['settings'].pop('persist_label_results', None)
    after['settings'].pop('persist_label_results', None)
    if before != after or (not (allow_depth_change or allow_search_optimization)
                          and next_teacher['max_depth'] <= previous_teacher['max_depth']):
        raise ValueError('Only an explicit teacher-depth upgrade is supported')
    if next_teacher['max_depth'] < 5:
        raise ValueError('Teacher depth must be at least five')
    from .heuristics import SearchConfig
    old = SearchConfig(**previous_teacher).to_dict()
    new = SearchConfig(**next_teacher).to_dict()
    allowed = set(SEARCH_OPTIMIZATION_FIELDS) if allow_search_optimization else set()
    if allow_depth_change or not allow_search_optimization:
        allowed.update(('max_depth','time_limit','node_limit'))
    for key in old.keys() | new.keys():
        if key not in allowed and old.get(key) != new.get(key):
            raise ValueError('Teacher upgrade cannot change evaluation semantics')
    return dict(previous_teacher=previous_teacher, next_teacher=next_teacher,
                changed_epoch=time.time(), existing_records_relabelled=False,
                search_optimization_only=allow_search_optimization and not allow_depth_change)


def upgrade_backend(saved, current):
    """Backend-only migration: no implicit depth, scoring or dataset changes."""
    before=json.loads(json.dumps(saved))
    after=json.loads(json.dumps(current))
    old=before['settings'].pop('teacher_backend', {'implementation':'python'})
    new=after['settings'].pop('teacher_backend', {'implementation':'python'})
    if before != after or old == new or new.get('implementation') != 'rust-v1':
        raise ValueError('Backend migration must preserve teacher and dataset settings')
    from .heuristics import SearchConfig
    from .rust_teacher.adapter import validate_backend
    validate_backend(SearchConfig(**after['settings']['teacher']),new)
    return dict(previous_backend=old,next_backend=new,changed_epoch=time.time(),
        existing_records_relabelled=False,backend_only=True,
        budget_units_changed='Python charged work -> native search/proof node visits; partial labels still rejected')


def run(output, primary, target, deadline=None, *, teacher_config=None, allow_teacher_upgrade=False,
        allow_teacher_change=False, allow_search_optimization=False, teacher_backend=None,
        allow_backend_change=False, generation_workers=None):
    # Execution parallelism is not part of the corpus/teacher identity.
    if generation_workers is not None:
        from .greedy_process import validate_worker_count
        validate_worker_count(generation_workers)
    output, primary = Path(output).resolve(), Path(primary).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # A nonblocking lock prevents two resumed coordinators sharing a catalog.
    import fcntl
    lock = (output / 'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    catalog = Catalog(output / 'corpus.sqlite3', target)
    teacher_config = dict(TEACHER if teacher_config is None else teacher_config)
    from .heuristics import SearchConfig
    SearchConfig(**teacher_config)
    if teacher_config['max_depth'] < 5:
        raise ValueError('Teacher depth must be at least five')
    settings = dict(SETTINGS, dataset_seed=3026091500, teacher=teacher_config,
                    output=str(output), promoted_roots=2, ablations=True, persist_label_results=True,
                    recorded_games=str(primary / 'rps2-games.ndjson'),
                    recorded_offsets=str(primary / 'recorded-offsets.json'))
    if teacher_backend:
        from .rust_teacher.adapter import validate_backend
        validate_backend(SearchConfig(**teacher_config),teacher_backend)
        settings['teacher_backend']=dict(teacher_backend)
    config = dict(primary=str(primary), target_positions=target, settings=settings)
    saved = catalog.get('config')
    migration = None
    if saved is not None and saved != config:
        if allow_backend_change:
            migration=upgrade_backend(saved, config)
        elif not (allow_teacher_upgrade or allow_teacher_change or allow_search_optimization):
            raise ValueError('Resume configuration differs from existing corpus')
        else:
            migration = upgrade_config(saved, config, allow_depth_change=allow_teacher_change,
                                       allow_search_optimization=allow_search_optimization)
        migration.update(positions=catalog.counts()['total'],next_seed=catalog.get('next_seed'))
    with catalog.db:
        if migration is not None:
            catalog.set('teacher_history', catalog.get('teacher_history', []) + [migration])
        catalog.set('config', config)
    started = time.time()
    pool = None
    workers = 0
    label_stream = None
    phase = 'initializing'
    reason = None

    def status(state='running', **extra):
        counts = catalog.counts()
        write_json(output / 'status.json', dict(state=state, phase=phase, reason=reason,
            pid=os.getpid(), worker_pids=pool.pids if pool else [], workers=workers,
            started_epoch=started, updated_epoch=time.time(), deadline_epoch=deadline,
            target_positions=target, unique_positions=counts['total'],
            primary_positions=counts['origin'].get('primary', 0),
            extra_positions=counts['origin'].get('extra', 0),
            augmented_examples=12 * counts['total'], by_stage=counts['stage'],
            by_split=counts['split'], sources=counts['source'],
            generation_teacher=teacher_config,
            generation_backend=teacher_backend or {'implementation':'python'},
            next_seed=catalog.get('next_seed', settings['dataset_seed']),
            pending_seeds=catalog.get('pending_seeds', []), save_mode='per_worker_completion',
            batch=catalog.get('batch', 0), training_untouched=True,
            restart_training='manual_after_READY', **extra))

    def stop(*_):
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
        signal.signal(sig, stop)
    if deadline is not None:
        if deadline <= time.time():
            raise ValueError('Deadline already passed')
        signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
    state = 'stopped'
    try:
        while deadline is None or time.time() < deadline:
            phase = 'importing_primary'
            catalog.import_primary(primary)
            counts = catalog.counts()
            if counts['total'] >= target:
                reason, state = 'target_reached', 'ready'
                write_json(output / 'READY.json', dict(unique_positions=counts['total'],
                    catalog=str(output / 'corpus.sqlite3'), by_split=counts['split'],
                    augmented_examples=12 * counts['total'], training_restart_required=True))
                break
            if shutil.disk_usage(output).free < 2 * 1024**3:
                reason = 'low_disk_space'
                break
            desired_workers = generation_workers if generation_workers is not None else (
                2 if primary_active(primary) else 4)
            if pool is None or (workers != desired_workers and label_stream is None):
                if pool is not None:
                    pool.close()
                phase = 'starting_workers'
                workers = desired_workers
                status()
                pool = GameProcesses(workers, snapshot=primary / 'seed.pt',
                                     settings=settings, _task=label_job)
            batch = catalog.get('batch', 0)
            pending = catalog.get('pending_seeds', [])
            if label_stream is None:
                if not pending:
                    first = catalog.get('next_seed', settings['dataset_seed'])
                    pending = list(range(first, first+4*workers))
                    with catalog.db:
                        catalog.set('pending_seeds', pending)
                        catalog.set('next_seed', first+len(pending))
                label_stream = pool.iter_results(primary/'seed.pt',pending,True,deadline=deadline)
            phase = f'depth_{teacher_config["max_depth"]}_labelling'
            status()
            (records, row), timing = next(label_stream)
            if row['seed'] not in pending:
                raise ValueError('Worker returned an unreserved seed')
            # Import the latest live-training shards before accepting extra rows.
            catalog.import_primary(primary)
            with catalog.db:
                catalog.insert(records, 'extra')
                pending.remove(row['seed'])
                catalog.set('pending_seeds', pending)
                catalog.set('batch', batch+1)
            acknowledge_label_result(settings,row['seed'])
            if not pending:
                assert next(label_stream,None) is None
                label_stream = None
            phase = 'worker_result_committed'
            status(label_seconds=timing['seconds'])
            print(json.dumps(dict(batch=batch+1, **catalog.counts())), flush=True)
        else:
            reason = 'deadline_reached'
    except (KeyboardInterrupt, TimeoutError):
        reason = 'deadline_reached' if deadline is not None and time.time() >= deadline else 'signal'
    except BaseException:
        state, reason = 'failed', traceback.format_exc()
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if pool is not None:
            pool.close()
            pool = None
        phase = 'complete'
        status(state)
        catalog.close()
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--target', type=int, default=1_000_000)
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--workers', type=int, dest='generation_workers',
                        help='Fixed generation workers (1-64); default: automatic 2/4')
    parser.add_argument('--teacher-depth', type=int, default=5)
    parser.add_argument('--teacher-seconds', type=float)
    parser.add_argument('--allow-teacher-upgrade', action='store_true')
    parser.add_argument('--allow-teacher-change', action='store_true')
    parser.add_argument('--allow-search-optimization', action='store_true')
    args = parser.parse_args()
    if args.target < 3:
        parser.error('--target must be at least three')
    run(args.output, args.primary, args.target, args.deadline,
        teacher_config=dict(TEACHER,max_depth=args.teacher_depth,
            time_limit=args.teacher_seconds if args.teacher_seconds is not None else (120. if args.teacher_depth<=5 else 1800.)),
        allow_teacher_upgrade=args.allow_teacher_upgrade, allow_teacher_change=args.allow_teacher_change,
        allow_search_optimization=args.allow_search_optimization,
        generation_workers=args.generation_workers)
