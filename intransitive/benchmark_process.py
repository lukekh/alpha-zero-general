"""Matched 1/2/4-process benchmarks using a committed v2 continuation fixture."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import torch
import onnxruntime
from .greedy_process import GameProcesses, cpu_seconds, make_net
from .greedy_training import checkpoint, iteration, load_bundle, publish


def source_hashes():
    root = Path(__file__).resolve().parent.parent
    paths = [root/'GenericNNetWrapper.py', root/'MCTS.py'] + sorted((root/'intransitive').glob('*.py'))
    return {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


class MemorySampler:
    """Sample aggregate RSS of this process tree (shared pages double-counted)."""
    def __init__(self):
        self.stop = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            raw = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,rss='], text=True)
            rows = [list(map(int, line.split())) for line in raw.splitlines()]
            owned = {os.getpid()}
            while True:
                added = {pid for pid, parent, _ in rows if parent in owned}
                if added <= owned:
                    break
                owned |= added
            self.samples.append(dict(epoch=time.time(), rss_mib=sum(
                rss for pid, _, rss in rows if pid in owned)/1024, processes=len(owned)))
            self.stop.wait(.25)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()


def fingerprint(net):
    h = hashlib.sha256()
    for key, value in sorted(net.items()):
        h.update(key.encode())
        h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def trial(args):
    setup = time.perf_counter()
    bundle = load_bundle(args.fixture)
    # Only benchmark copies get a fresh measurement deadline. Production resume
    # always retains the absolute deadline stored in its continuation file.
    bundle['deadline_epoch'] = time.time() + args.timeout
    bundle['settings'] = dict(bundle['settings'], seed=bundle['settings']['seed'] + args.seed_offset)
    parent_cpu = cpu_seconds()
    before_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    with MemorySampler() as memory, tempfile.TemporaryDirectory(prefix='process-benchmark-') as folder:
        net = make_net(bundle['settings'])
        net.load_network(bundle['current'])
        snapshot = checkpoint(net, folder)
        with GameProcesses(args.workers, snapshot=snapshot, settings=bundle['settings']) as pool:
            startup = pool.startup
            setup_seconds = time.perf_counter()-setup
            # Matched untrained selection probe isolates evaluation throughput.
            fixed, selection_metrics = pool.collect(snapshot, range(800000, 800020), False,
                                                    deadline=bundle['deadline_epoch'])
            start, start_cpu = time.perf_counter(), cpu_seconds()
            candidate, report = iteration(bundle, pool, folder)
            commit_start = time.perf_counter()
            target = Path(folder)/'continuation.pt'
            publish(target, candidate)
            commit_seconds = time.perf_counter()-commit_start
            total_seconds = time.perf_counter()-start
            phase_cpu = sum(v.get('worker_cpu_seconds', 0) for v in report['phases'].values()
                            if isinstance(v, dict)) + cpu_seconds()-start_cpu
            reload_start = time.perf_counter()
            restored = load_bundle(target)
            assert fingerprint(restored['current']['state_dict']) == fingerprint(candidate['current']['state_dict'])
            reload_seconds = time.perf_counter()-reload_start
            assert len(report['training_games']) == 8
            assert len(report['selection_games']) == 20, 'Fixture must precede a scheduled selection'
            pids = pool.pids
        assert all(not p.is_alive() for p in pool.initial_workers)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        child_cpu = children.ru_utime+children.ru_stime-before_children.ru_utime-before_children.ru_stime
        result = dict(workers=args.workers, seed_offset=args.seed_offset, source_sha256=source_hashes(),
            fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
            input_iteration=bundle['iteration'], input_weights_sha256=fingerprint(bundle['current']['state_dict']),
            output_weights_sha256=fingerprint(candidate['current']['state_dict']),
            output_replay_sha256=candidate['replay_sha256'], selection=candidate['selection'],
            best_iteration=candidate['best_iteration'], optimizer_updates=candidate['optimizer_updates'],
            setup_seconds=setup_seconds, pool_startup=startup,
            frozen_selection=selection_metrics, frozen_selection_games=[row for _, row in fixed],
            commit_seconds=commit_seconds, reload_validation_seconds=reload_seconds,
            continuation_bytes=target.stat().st_size,
            complete_iteration_seconds=total_seconds,
            complete_iteration_games_per_second=28/total_seconds,
            complete_iterations_per_hour=3600/total_seconds,
            complete_iteration_cpu_percent=100*phase_cpu/total_seconds,
            total_cpu_seconds=child_cpu+cpu_seconds()-parent_cpu,
            children_reaped=True, worker_pids=pids, **report)
    result['whole_trial_seconds'] = time.perf_counter()-setup
    result['memory'] = dict(interval_seconds=.25, peak_aggregate_rss_mib=max(
        s['rss_mib'] for s in memory.samples), samples=memory.samples,
        definition='Parent plus descendant RSS; shared pages counted per process; excludes competing training job')
    for phase in ('rollouts', 'selection'):
        row = result['phases'][phase]
        row['games_per_second'] = row['games']/row['seconds']
    result['frozen_selection']['games_per_second'] = 20/selection_metrics['seconds']
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('workers','seed_offset','complete_iteration_seconds','setup_seconds')}), flush=True)


def suite(args):
    args.output.mkdir(parents=True, exist_ok=True)
    workloads = subprocess.check_output(['ps','-axo','pid,ppid,%cpu,rss,command'],text=True)
    manifest = dict(platform=platform.platform(), machine=platform.machine(), cpus=os.cpu_count(),
        python=sys.version, torch=torch.__version__, numpy=np.__version__, onnxruntime=onnxruntime.__version__,
        started_epoch=time.time(), workload_before=workloads, source_sha256=source_hashes(),
        fixture=str(args.fixture), fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        protocol='Three matched offsets; Latin-square worker order; 8 training/20 selection, 32 full sims, 1 optimizer thread',
        startup='New spawned interpreters, per-worker load/export/session and MCTS/greedy warmup; existing Numba disk caches allowed',
        default_workers=1)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    rows = []
    for offset, order in zip((0,10000,20000), ((1,2,4),(2,4,1),(4,1,2))):
        for workers in order:
            path = args.output/f'w{workers}-seed{offset}.json'
            launched_seconds = None
            if not path.exists():
                command=[sys.executable,'-m','intransitive.benchmark_process','--trial','--fixture',str(args.fixture),
                    '--workers',str(workers),'--seed-offset',str(offset),'--output',str(path),'--timeout',str(args.timeout)]
                with path.with_suffix('.log').open('w') as log:
                    launched = time.perf_counter()
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = process.wait(timeout=args.timeout+120)
                        launched_seconds = time.perf_counter()-launched
                        if code:
                            raise subprocess.CalledProcessError(code, command)
                    except BaseException:
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                            process.wait(timeout=10)
                        except ProcessLookupError:
                            pass
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise
            rows.append(json.loads(path.read_text()))
            if launched_seconds is not None:
                rows[-1]['spawned_command_seconds'] = launched_seconds
                rows[-1]['parent_bootstrap_and_exit_seconds'] = launched_seconds-rows[-1]['whole_trial_seconds']
                path.write_text(json.dumps(rows[-1],indent=2)+'\n')
            assert rows[-1]['source_sha256'] == manifest['source_sha256'], 'Existing result uses different sources; use a fresh output directory'
            assert rows[-1]['fixture_sha256'] == manifest['fixture_sha256']
            print(f'Completed seed offset {offset}, workers {workers}: {rows[-1]["complete_iteration_seconds"]:.3f}s',flush=True)
        matched = rows[-3:]
        for row in matched[1:]:
            for key in ('input_weights_sha256','output_weights_sha256','output_replay_sha256',
                        'optimizer_updates','selection','best_iteration'):
                assert row[key] == matched[0][key], (offset,key)
            for key in ('training_games','selection_games','frozen_selection_games'):
                fields=('seed','model_side','actions','trajectory_sha256','replay_sha256','outcome','reason','examples')
                assert [[r[f] for f in fields] for r in row[key]] == [
                    [r[f] for f in fields] for r in matched[0][key]], (offset,key)
    manifest.update(finished_epoch=time.time(), matched_trajectory_targets_weights_optimizer_selection=True,
        workload_after=subprocess.check_output(['ps','-axo','pid,ppid,%cpu,rss,command'],text=True))
    # Keep workload inventory relevant and avoid committing unrelated command lines.
    for key in ('workload_before','workload_after'):
        manifest[key]='\n'.join(line for line in manifest[key].splitlines()
                               if 'python' in line.lower() or 'PID' in line)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',type=Path,default=Path(__file__).parent/'benchmarks/process/fixture.pt')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,choices=(1,2,4),default=1)
    parser.add_argument('--seed-offset',type=int,default=0)
    parser.add_argument('--timeout',type=float,default=1800)
    parser.add_argument('--trial',action='store_true')
    args=parser.parse_args()
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    trial(args) if args.trial else suite(args)


if __name__ == '__main__':
    main()
