"""Bounded serial/parallel smoke with sampled aggregate process-tree RSS.

Run: python -m intransitive.benchmarks.tournament.reproduce --output /tmp/matches
No worker-count recommendation is inferred from CPU count.
"""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


def tree_rss(pid):
    output = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,rss='], text=True)
    rows = [tuple(map(int, line.split())) for line in output.splitlines() if line.strip()]
    children = {pid}
    while True:
        expanded = children | {p for p, parent, _ in rows if parent in children}
        if expanded == children:
            break
        children = expanded
    return sum(rss * 1024 for p, _, rss in rows if p in children)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / 'input.json'
    if manifest.exists():
        raise ValueError('Benchmark requires a fresh output to avoid measuring resume')
    subprocess.run([sys.executable, '-m', 'intransitive.tournament', 'prepare',
                    '--output', str(manifest), '--positions', '2', '--max-plies', '4'], check=True)
    measurements = []
    for workers in (1, 2):
        output = args.output / f'workers-{workers}'
        start, peak = time.perf_counter(), 0
        with (args.output / f'workers-{workers}.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '-m', 'intransitive.tournament', 'run',
                '--manifest', str(manifest), '--output', str(output), '--workers', str(workers)],
                stdout=log, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    peak = max(peak, tree_rss(process.pid))
                    time.sleep(.2)
                if process.returncode:
                    raise RuntimeError(f'Tournament exited {process.returncode}; see log')
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)
        elapsed = time.perf_counter() - start
        subprocess.run([sys.executable, '-m', 'intransitive.tournament', 'verify',
                        '--manifest', str(manifest), '--output', str(output)], check=True)
        report = json.loads((output / 'report.json').read_text())
        measurements.append(dict(workers=workers, wall_seconds=elapsed,
                                 games_per_second=report['final_matches'] / elapsed,
                                 sampled_process_tree_peak_rss_bytes=peak,
                                 child_high_water_rss_bytes=report['child_peak_rss_bytes'],
                                 known_child_cpu_seconds=report['known_child_cpu_seconds'],
                                 final_matches=report['final_matches']))
    spec = json.loads(manifest.read_text())
    def depth_traces(workers):
        return {t['id']: json.loads((args.output / f'workers-{workers}' / 'matches' / (t['id'] + '.json')).read_text())['trajectory_sha256']
                for t in spec['tasks'] if spec['protocols'][t['protocol']]['mode'] == 'depth'}
    same = depth_traces(1) == depth_traces(2)
    result = dict(platform=platform.platform(), python=sys.version, measurements=measurements,
                  fixed_depth_trajectories_equal=same,
                  methodology='Sequential worker-count runs, one run each; 200ms process-tree RSS samples. '
                              'Includes startup and compilation, OS filesystem caches may be warm. '
                              'Wall-mode trajectories are timing-dependent. Tiny smoke is not strength evidence.')
    (args.output / 'measurements.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not same:
        raise AssertionError('Fixed-depth trajectories changed with worker count')


if __name__ == '__main__':
    main()
