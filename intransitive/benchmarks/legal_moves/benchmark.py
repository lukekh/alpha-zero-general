"""Paired before/after node-throughput benchmark, with warmup and exact guards.

Run --help for inputs. Baseline must be a clean checkout of the recorded commit.
Uses only standard library imports until a worker selects its source checkout.
"""
import argparse
import json
import hashlib
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time


def worker(args):
    sys.path.insert(0, str(args.repo.resolve()))
    import numpy as np
    from intransitive.heuristics import AlphaBetaPlayer, SearchConfig
    from intransitive.heuristics.position import SearchPosition
    from intransitive.IntransitiveLogicNumba import raw_movement_mask
    from intransitive.rust_teacher import RustTeacher
    rows = []
    config = SearchConfig(max_depth=3, time_limit=120., node_limit=10**9, table_entries=50000)
    if args.backend == "python":
        AlphaBetaPlayer(config=config)._prepare()
    states = [(p.stem, np.frombuffer(p.read_bytes(), dtype=np.int8).reshape(9, 9, 84))
              for p in sorted(args.positions.glob('*.bin'))]
    if not states:
        raise ValueError('No .bin corpus states found')
    for name, state in states:
        if args.backend == 'python':
            pos = SearchPosition(state)
            # Compile both update signatures and all query kernels before timing.
            actions = pos.legal()
            if len(actions):
                pos.push(int(actions[0])); pos.pop()
            if hasattr(pos, 'masks'):
                from intransitive.heuristics.moves import legal_actions, has_move
                legal = lambda: legal_actions(pos.masks, pos.side)
                exists = lambda: has_move(pos.masks, pos.side)
            else:
                legal = lambda: np.flatnonzero(raw_movement_mask(pos.pieces, pos.side))
                exists = lambda: raw_movement_mask(pos.pieces, pos.side).any()
            def update():
                pos.push(int(actions[0])); pos.pop()
            micro, micro_cpu = {}, {}
            for label, fn in [('legal', legal), ('has_move', exists), ('push_pop', update)]:
                fn()
                cpu_start = time.process_time()
                start = time.perf_counter()
                for _ in range(10000):
                    fn()
                micro[label] = (time.perf_counter() - start) / 10000
                micro_cpu[label] = (time.process_time() - cpu_start) / 10000
            def search(depth, seconds):
                from dataclasses import replace
                player = AlphaBetaPlayer(config=replace(config, max_depth=depth, time_limit=seconds))
                cpu = time.process_time()
                start = time.perf_counter()
                r = player.analyze(state)
                return dict(action=int(r.action), score=r.score, completed_depth=r.completed_depth,
                            nodes=r.nodes, proof_nodes=r.proof_nodes, pv=list(map(int, r.pv)),
                            complete=not r.stopped, seconds=time.perf_counter()-start,
                            cpu_seconds=time.process_time()-cpu)
            depth = 4
        else:
            micro, micro_cpu = {}, {}
            def search(depth, seconds):
                cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
                start = time.perf_counter()
                with RustTeacher(args.repo / "intransitive/rust_teacher/target/release/intransitive-rust-teacher") as engine:
                    r = engine.analyze(state, depth=depth, seconds=seconds, weight=0.)
                used = resource.getrusage(resource.RUSAGE_CHILDREN)
                return {**{k:r[k] for k in ('action','score','completed_depth','nodes','proof_nodes','pv','complete')},
                        'seconds':time.perf_counter()-start,
                        'cpu_seconds':used.ru_utime + used.ru_stime - cpu.ru_utime - cpu.ru_stime}
            depth = 6
        search(min(depth, 2), 120.)
        fixed = search(depth, 120.)
        assert fixed['complete'], (name, fixed)
        timed = search(20, .25)
        rows.append(dict(name=name, micro=micro, micro_cpu=micro_cpu, fixed=fixed, timed=timed))
    print(json.dumps(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path)
    parser.add_argument('--after', type=Path, default=Path.cwd())
    parser.add_argument('--positions', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--repo', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--backend', choices=['python', 'rust'])
    args = parser.parse_args()
    if args.repo:
        worker(args)
        return
    if not args.before or not args.output or args.repeats < 1:
        parser.error('--before, --output and positive --repeats are required')
    report = dict(platform=platform.platform(), machine=platform.machine(), python=sys.version,
                  baseline_commit=subprocess.check_output(['git','-C',str(args.after),'rev-parse','HEAD'],text=True).strip(),
                  repeats=args.repeats, trials=[], settings=dict(python_depth=4,rust_depth=6,
                  time_budget=.25,proof_depth=2,proof_nodes=64,table_entries=50000,weight=0))
    sources = ['intransitive/heuristics/' + name for name in
               ('position.py', 'moves.py', 'proof.py', 'kernels.py', 'config.py', 'search.py', 'ordering.py')] + [
               'intransitive/rust_teacher/src/lib.rs', 'intransitive/rust_teacher/src/moves.rs',
               'intransitive/rust_teacher/target/release/intransitive-rust-teacher']
    report['sha256'] = {variant: {name: hashlib.sha256((getattr(args, variant)/name).read_bytes()).hexdigest()
                                for name in sources if (getattr(args, variant)/name).exists()}
                        for variant in ('before', 'after')}
    report['corpus_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(args.positions.glob('*.bin'))}
    backends = [args.backend] if args.backend else ['python', 'rust']
    for backend in backends:
        for trial in range(args.repeats):
            row = dict(backend=backend, trial=trial)
            for variant in (['before','after'] if trial % 2 == 0 else ['after','before']):
                cmd = [sys.executable, str(Path(__file__).resolve()), '--repo', str(getattr(args,variant).resolve()),
                       '--positions', str(args.positions.resolve()), '--backend', backend]
                row[variant] = json.loads(subprocess.check_output(cmd, text=True))
            assert len(row['before']) == len(row['after'])
            for a,b in zip(row['before'], row['after']):
                assert a['name'] == b['name']
                for key in ('action','score','completed_depth','nodes','proof_nodes','pv','complete'):
                    assert a['fixed'][key] == b['fixed'][key], (backend,a['name'],key,a,b)
            row['speedup'] = sum(r['fixed']['seconds'] for r in row['before']) / sum(r['fixed']['seconds'] for r in row['after'])
            row['cpu_speedup'] = sum(r['fixed']['cpu_seconds'] for r in row['before']) / sum(r['fixed']['cpu_seconds'] for r in row['after'])
            report['trials'].append(row)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(backend, trial, row['speedup'], row['cpu_speedup'], flush=True)
    report['summary'] = {}
    for backend in backends:
        trials = [r for r in report['trials'] if r['backend'] == backend]
        report['summary'][backend] = dict(median_speedup=statistics.median(r['speedup'] for r in trials),
            median_cpu_speedup=statistics.median(r['cpu_speedup'] for r in trials))
        for variant in ['before','after']:
            times = [sum(p['fixed']['seconds'] for p in r[variant]) for r in trials]
            cpu_times = [sum(p['fixed']['cpu_seconds'] for p in r[variant]) for r in trials]
            nodes = sum(p['fixed']['nodes'] for p in trials[0][variant])
            report['summary'][backend][variant] = dict(seconds=statistics.median(times), nodes=nodes,
                                                      nodes_per_second=nodes/statistics.median(times),
                                                      cpu_seconds=statistics.median(cpu_times),
                                                      nodes_per_cpu_second=nodes/statistics.median(cpu_times))
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))


if __name__ == '__main__':
    main()
