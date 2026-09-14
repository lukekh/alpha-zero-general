"""Independent PVS/aspiration comparisons against the exact archived baseline."""
import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import tarfile
from time import perf_counter
import tracemalloc
import types

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from intransitive.heuristics import search
from intransitive.heuristics.budget import Budget
from intransitive.record import load_record, state_hash
from intransitive.tests.reference_rules import Position
from intransitive.tests.test_game_blunders import DATA, meets_objective
from intransitive.tests.tactical_oracle import move_name

EVIDENCE = Path(__file__).with_name('evidence')


def baseline():
    manifest = json.loads((EVIDENCE/'baseline-manifest.json').read_text())
    archive_path = EVIDENCE/'baseline.tar.gz'
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == manifest['archive_sha256']
    changed = {'intransitive/heuristics/search.py', 'intransitive/heuristics/config.py',
               'intransitive/heuristics/budget.py'}
    for path, digest in manifest['files'].items():
        if path.endswith('.py') and path not in changed:
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    modules = {}
    with tarfile.open(archive_path) as archive:
        for name in ('config', 'budget', 'search'):
            module_name = 'intransitive.heuristics._issue41_baseline_' + name
            module = types.ModuleType(module_name)
            module.__package__ = 'intransitive.heuristics'
            sys.modules[module_name] = module
            source = archive.extractfile('intransitive/heuristics/' + name + '.py').read()
            exec(compile(source, '<baseline_' + name + '>', 'exec'), module.__dict__)
            modules[name] = module
    # Keep the original per-operation budget too, including its exception class.
    modules['search'].Budget = modules['budget'].Budget
    modules['search'].BudgetExpired = modules['budget'].BudgetExpired
    # Proof/evaluator calls use current BudgetExpired; both exceptions must be
    # catchable by baseline search. Budget implementation otherwise unchanged.
    modules['budget'].BudgetExpired = search.BudgetExpired
    modules['search'].BudgetExpired = search.BudgetExpired
    modules['search'].SearchConfig = modules['config'].SearchConfig
    return modules['search'], manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seconds', type=float, default=1.)
    parser.add_argument('--depth', type=int, default=4)
    args = parser.parse_args()
    start = perf_counter()
    record = load_record((ROOT/'intransitive/benchmarks/search_budget/game79.pgn').read_text())
    archived, manifest = baseline()
    search.AlphaBetaPlayer(config=record.config)._prepare()
    common_setup = perf_counter() - start
    variants = {'archived': (archived, False, False), 'full': (search, False, False),
                'pvs': (search, True, False), 'aspiration': (search, False, True),
                'combined': (search, True, True)}
    configs = {}
    for name, (module, pvs, aspiration) in variants.items():
        for mode in ('fixed', 'time'):
            fields = record.config.to_dict()
            fields.update(max_depth=args.depth if mode == 'fixed' else 20,
                          node_limit=10**9, time_limit=600. if mode == 'fixed' else args.seconds,
                          pvs_enabled=pvs, aspiration_enabled=aspiration, aspiration_window=25.)
            if name == 'archived':
                for key in ('pvs_enabled', 'aspiration_enabled', 'aspiration_window'):
                    fields.pop(key)
            configs[name, mode] = module.SearchConfig(**fields)
    cases = {case['ply']: case for case in DATA['original_positions']}
    runs, memory = [], []
    # Warm all game/proof signatures and leaf caches before any measured runs.
    for ply in (0, 27, 35, 61):
        search.AlphaBetaPlayer(config=replace(record.config, max_depth=1,
            time_limit=600, node_limit=10**9)).analyze(record.states[ply])
    for repeat in range(args.repeats):
        for ply in (0, 27, 35, 61):
            state = record.states[ply]
            board = Position.from_storage(state)
            names = list(variants)
            # Rotate order across paired repetitions; every search starts fresh.
            names = names[repeat % len(names):] + names[:repeat % len(names)]
            for name in names:
                module = variants[name][0]
                for mode in ('fixed', 'time'):
                    config = configs[name, mode]
                    player = module.AlphaBetaPlayer(config=config)
                    result = player.analyze(state)
                    fields = asdict(result)
                    fields.pop('explanation')
                    fields.update(main_nodes=result.nodes-result.proof_nodes,
                        move=move_name(result.action),
                        tactical_objective_met=(meets_objective(board, result.action, cases[ply])
                                                if ply in cases else None),
                        timeout_overshoot=max(0., result.elapsed-config.time_limit),
                        material_cache=player.evaluator.material.values.cache_info()._asdict())
                    for key in ('pvs_probes', 'pvs_researches', 'aspiration_researches',
                                'aspiration_fail_highs', 'aspiration_fail_lows'):
                        fields.setdefault(key, 0)
                    fields['material_array_bytes'] = fields['material_cache']['currsize'] * 1024
                    fields['main_nodes_per_second'] = fields['main_nodes'] / result.elapsed
                    # Validate every returned PV using the independent rules.
                    cursor = board
                    for action in result.pv:
                        assert action in cursor.legal(), (name, ply, result.pv)
                        cursor, _ = cursor.move(action)
                    runs.append(dict(repeat=repeat+1, ply=ply, variant=name, mode=mode,
                                     state_sha256=state_hash(state), **fields))
                    print(f'{repeat+1} ply={ply} {mode} {name}: {result.elapsed:.4f}s '
                          f'{result.completed_depth}/{result.selected_depth} {result.score}', file=sys.stderr)
    # Run allocation sampling separately so tracing cannot contaminate timings.
    for ply in (0, 27, 35, 61):
        for name, (module, _, _) in variants.items():
            player = module.AlphaBetaPlayer(config=configs[name, 'fixed'])
            gc.collect()
            tracemalloc.start()
            player.analyze(record.states[ply])
            retained, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory.append(dict(ply=ply, variant=name, traced_peak_bytes=peak,
                               traced_retained_bytes=retained))
    for ply in (0, 27, 35, 61):
        rows = [r for r in runs if r['mode'] == 'fixed' and r['ply'] == ply]
        for row in rows:
            for key in ('score', 'completed_depth', 'selected_depth'):
                assert row[key] == rows[0][key], (ply, row['variant'], key, row[key], rows[0][key])
            if row['variant'] in ('full', 'archived'):
                for key in ('action', 'pv', 'nodes', 'proof_nodes', 'work'):
                    assert row[key] == rows[0][key], (ply, key)
            # Tie ordering may legitimately change. Check that each selected
            # move has the same full-window value at this exact horizon.
            player = search.AlphaBetaPlayer(config=configs['full', 'fixed'])
            player._prepare()
            child, _ = player.game.getNextState(record.states[ply],
                int(record.states[ply][:, :, 82:84].flat[1]), row['action'])
            value, _ = player._search(child, args.depth-1, -float('inf'), float('inf'), 0,
                                      Budget(10**9, 600))
            assert search.from_table(-value, 1) == row['score'], (ply, row['variant'], value)
    report = dict(schema=1, baseline_revision=manifest['revision'],
                  python=sys.version, platform=platform.platform(),
                  common_setup_seconds=common_setup,
                  window_setup_seconds=0.,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                                 (1 if sys.platform == 'darwin' else 1024),
                  configs={f'{name}/{mode}': asdict(c) for (name, mode), c in configs.items()},
                  implementation_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (ROOT/'intransitive/heuristics').glob('*.py')},
                  repeats=args.repeats, runs=runs, memory=memory)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
