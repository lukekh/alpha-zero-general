"""Paired fixed-depth native before/after benchmark with exact result guards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
from statistics import median
import time

from . import RustTeacher


PROFILES = {'material': (4, 0.), 'pressure7': (3, 10.), 'pressure9': (4, 10.)}
SIGNATURE = ('action', 'score', 'completed_depth', 'target_depth', 'complete',
             'stop_reason', 'nodes', 'proof_nodes', 'pv', 'tt_hits', 'table_entries')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows, repeats):
    trials = []
    for trial in range(repeats):
        subset = [r for r in rows if r['trial'] == trial]
        before = sum(r['before']['wall_seconds'] for r in subset)
        after = sum(r['after']['wall_seconds'] for r in subset)
        before_cpu = sum(r['before']['cpu_seconds'] for r in subset)
        after_cpu = sum(r['after']['cpu_seconds'] for r in subset)
        trials.append(dict(trial=trial, before_seconds=before, after_seconds=after,
                           speedup=before / after, before_cpu_seconds=before_cpu,
                           after_cpu_seconds=after_cpu, cpu_speedup=before_cpu / after_cpu))
    return dict(pairs=len(rows), full_depth_pairs=sum(r['full_depth'] for r in rows),
                trials=trials,
                median_before_seconds=median(r['before_seconds'] for r in trials),
                median_after_seconds=median(r['after_seconds'] for r in trials),
                median_trial_speedup=median(r['speedup'] for r in trials),
                median_before_cpu_seconds=median(r['before_cpu_seconds'] for r in trials),
                median_after_cpu_seconds=median(r['after_cpu_seconds'] for r in trials),
                median_trial_cpu_speedup=median(r['cpu_speedup'] for r in trials),
                trial_cpu_speedup_range=[min(r['cpu_speedup'] for r in trials),
                                         max(r['cpu_speedup'] for r in trials)],
                trial_speedup_range=[min(r['speedup'] for r in trials),
                                     max(r['speedup'] for r in trials)],
                stages={stage: dict(
                    median_before_seconds=median(r['before']['wall_seconds'] for r in rows if r['stage'] == stage),
                    median_after_seconds=median(r['after']['wall_seconds'] for r in rows if r['stage'] == stage),
                    median_paired_speedup=median(r['speedup'] for r in rows if r['stage'] == stage),
                    median_cpu_speedup=median(r['before']['cpu_seconds'] / r['after']['cpu_seconds'] for r in rows if r['stage'] == stage))
                    for stage in sorted({r['stage'] for r in rows})})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--positions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--seconds', type=float, default=120.)
    parser.add_argument('--profiles', nargs='+', choices=PROFILES, default=list(PROFILES))
    args = parser.parse_args()
    paths = sorted(args.positions.glob('position-*.bin'))
    if not paths or args.repeats < 1 or args.depth < 1 or args.output.exists():
        parser.error('Need saved positions, positive repeats/depth, and a fresh output directory')
    args.output.mkdir(parents=True)
    os.nice(10)
    binaries = dict(before=args.before.resolve(), after=args.after.resolve())
    report = dict(platform=platform.platform(), started_unix=time.time(),
                  binaries={name: dict(path=str(path), sha256=digest(path)) for name, path in binaries.items()},
                  depth=args.depth, repeats=args.repeats, seconds=args.seconds, rows=[],
                  settings=dict(proof_depth=2, proof_nodes=64, table_entries=50000,
                                node_limit=1000000000, reuse=False),
                  notes=['Single search worker; existing background jobs are left running.',
                         'Same saved positions, fresh search table, alternating engine order per pair.',
                         'Full results and node counts must match exactly; incomplete runs fail.',
                         'Wall time includes process startup/exit, IPC and root decoding; compilation is excluded.',
                         'Each request owns a fresh process so reaped-child user+system CPU time is measurable.',
                         'CPU time separates compute cost from scheduling contention; wall time is the observed latency.',
                         'A separate depth-three warmup precedes each measured pair for both engines.',
                         'Proof-complete roots are identified separately from full-depth roots.',
                         'Totals exclude sampling, ablations and corpus I/O.'])
    states = []
    for path in paths:
        data = path.read_bytes()
        if len(data) != 6804:
            raise ValueError(f'Invalid state size: {path}')
        (args.output / path.name).write_bytes(data)
        index = int(path.stem.split('-')[-1])
        states.append((index, data.hex(), digest(path)))

    def save():
        target = args.output / 'results.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(target)

    def search(binary, state_hex, radius, weight, depth):
        cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        started = time.perf_counter()
        with RustTeacher(binary) as client:
            result = client.request(
                f'search {depth} {int(args.seconds*1000)} 1000000000 {radius} {weight} 2 64 50000 {state_hex}',
                timeout=args.seconds + 10.)
        result['wall_seconds'] = time.perf_counter() - started
        cpu_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        result['cpu_seconds'] = (cpu_after.ru_utime - cpu_before.ru_utime
                                 + cpu_after.ru_stime - cpu_before.ru_stime)
        if not result['complete']:
            raise RuntimeError(f'Incomplete benchmark search: {result}')
        return result

    save()
    for trial in range(args.repeats):
        profiles = args.profiles if trial % 2 == 0 else args.profiles[::-1]
        for profile in profiles:
            radius, weight = PROFILES[profile]
            for index, state_hex, state_hash in states:
                order = ('before', 'after') if (index + trial) % 2 == 0 else ('after', 'before')
                for name in order:
                    search(binaries[name], state_hex, radius, weight, min(3, args.depth))
                row = dict(profile=profile, radius=radius, weight=weight, trial=trial,
                           index=index, stage=('opening', 'midgame', 'endgame')[index % 3],
                           state_sha256=state_hash, order=order)
                for name in order:
                    row[name] = search(binaries[name], state_hex, radius, weight, args.depth)
                row['exact_match'] = all(row['before'][key] == row['after'][key] for key in SIGNATURE)
                # JSON floats preserve roundtrip binary64; compare signed zero too.
                row['exact_match'] &= float(row['before']['score']).hex() == float(row['after']['score']).hex()
                row['full_depth'] = row['before']['completed_depth'] == args.depth
                row['speedup'] = row['before']['wall_seconds'] / row['after']['wall_seconds']
                report['rows'].append(row)
                save()
                print(json.dumps({key: row[key] for key in ('profile', 'trial', 'index', 'exact_match', 'full_depth', 'speedup')}), flush=True)
                if not row['exact_match']:
                    raise AssertionError('Before/after results or search work differ; benchmark rejected')
    report['summary'] = {profile: summarize([r for r in report['rows'] if r['profile'] == profile], args.repeats)
                         for profile in args.profiles}
    report['all_results_identical'] = all(r['exact_match'] for r in report['rows'])
    report['finished_unix'] = time.time()
    save()
    print(json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
