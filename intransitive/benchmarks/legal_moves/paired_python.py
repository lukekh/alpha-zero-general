"""Same-process Python confirmation: alternate engines for every position.

The baseline is imported under a separate package name from a cache-free copy.
Both implementations therefore use the same interpreter/thread, reducing core
placement and long-gap scheduling differences between independent workers.
"""
import argparse
import importlib
import hashlib
import platform
import json
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--positions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--before-python-ordering', action='store_true',
                        help='Force uncompiled ranking in the baseline for a compilation-only comparison')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    import numpy as np
    import intransitive.heuristics as after
    with tempfile.TemporaryDirectory(prefix='issue62-python-') as temp:
        shutil.copytree(args.before / 'intransitive', Path(temp)/'before_intransitive',
                        ignore=shutil.ignore_patterns('__pycache__', 'target', '*.gz', '*.pt', '*.npz', 'evidence'))
        sys.path.insert(0, temp)
        before = importlib.import_module('before_intransitive.heuristics')
        engines = {'before':before, 'after':after}
        configs = {k:m.SearchConfig(max_depth=4, time_limit=120., node_limit=10**9, table_entries=50000,
                   **({'compiled_ordering_enabled':False} if k == 'before' and args.before_python_ordering else {}))
                   for k,m in engines.items()}
        for k,m in engines.items():
            m.AlphaBetaPlayer(config=configs[k])._prepare()
        states = [(p.stem, np.frombuffer(p.read_bytes(), dtype=np.int8).reshape(9,9,84))
                  for p in sorted(args.positions.glob('*.bin'))]
        assert states and args.repeats > 0
        def search(k, state):
            player = engines[k].AlphaBetaPlayer(config=configs[k])
            cpu, wall = time.process_time(), time.perf_counter()
            r = player.analyze(state)
            return dict(cpu=time.process_time()-cpu, wall=time.perf_counter()-wall,
                        action=int(r.action), score=r.score, nodes=r.nodes,
                        proof_nodes=r.proof_nodes, work=r.work, depth=r.completed_depth,
                        pv=list(map(int, r.pv)), complete=not r.stopped)
        # Warm actual search paths in both implementations before timing.
        for _, state in states:
            for k in engines:
                search(k, state)
        report = dict(repeats=args.repeats, depth=4, rows=[], trials=[],
                      platform=platform.platform(), python=sys.version,
                      settings={k:c.to_dict() for k,c in configs.items()})
        source_names = ('config.py', 'search.py', 'position.py', 'proof.py', 'kernels.py', 'moves.py', 'ordering.py')
        report['sha256'] = {k:{name:hashlib.sha256((repo/'intransitive/heuristics'/name).read_bytes()).hexdigest()
                              for name in source_names}
                            for k,repo in [('before', args.before), ('after', root)]}
        for trial in range(args.repeats):
            for index, (name, state) in enumerate(states):
                row = dict(trial=trial, name=name)
                for k in (['before','after'] if (trial+index)%2 == 0 else ['after','before']):
                    row[k] = search(k, state)
                for key in ('action','score','nodes','proof_nodes','work','depth','pv','complete'):
                    assert row['before'][key] == row['after'][key], (name, key, row)
                assert row['before']['complete']
                report['rows'].append(row)
            rows = [r for r in report['rows'] if r['trial']==trial]
            result = {k:{clock:sum(r[k][clock] for r in rows) for clock in ('cpu','wall')}
                      for k in engines}
            result['cpu_speedup'] = result['before']['cpu']/result['after']['cpu']
            result['wall_speedup'] = result['before']['wall']/result['after']['wall']
            report['trials'].append(result)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(trial, result, flush=True)
        report['summary'] = {clock:statistics.median(r[clock+'_speedup'] for r in report['trials'])
                             for clock in ('cpu','wall')}
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(report['summary'])


if __name__ == '__main__':
    main()
