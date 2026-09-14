"""Compare archived #39 baseline with incremental material at fixed/equal time."""
import argparse
import cProfile
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
from intransitive.heuristics.material import count_pieces, warm_material_kernels
from intransitive.record import load_record, state_hash
from intransitive.tests.reference_rules import Position
from intransitive.tests.test_game_blunders import DATA, meets_objective
from intransitive.tests.tactical_oracle import move_name


def baseline():
    modules = []
    evidence = Path(__file__).with_name('evidence')
    manifest = json.loads((evidence/'baseline-manifest.json').read_text())
    archive_path = evidence/'baseline.tar.gz'
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == manifest['archive_sha256']
    changed = {'intransitive/heuristics/evaluation.py', 'intransitive/heuristics/search.py'}
    for path, digest in manifest['files'].items():
        if path.endswith('.py') and path not in changed:
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    with tarfile.open(archive_path) as archive:
        for name in ('evaluation', 'search'):
            module_name = 'intransitive.heuristics._issue39_baseline_' + name
            module = types.ModuleType(module_name)
            module.__package__ = 'intransitive.heuristics'
            sys.modules[module_name] = module
            source = archive.extractfile('intransitive/heuristics/' + name + '.py').read()
            exec(compile(source, '<baseline_' + name + '>', 'exec'), module.__dict__)
            modules.append(module)
    modules[1].Evaluator = modules[0].Evaluator
    return modules[1]


def allocation_sample(call):
    gc.collect()
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    profile = cProfile.Profile()
    profile.enable()
    result = call()
    profile.disable()
    current, peak = tracemalloc.get_traced_memory()
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    constructors = {}
    for row in profile.getstats():
        code = row.code
        if not isinstance(code, str) and code.co_name == '__init__':
            for name, suffix in (('Geometry', '/heuristics/geometry.py'), ('Counter', '/collections/__init__.py')):
                if code.co_filename.endswith(suffix):
                    constructors[name] = constructors.get(name, 0) + row.callcount
            if code.co_filename == '<string>':
                # Generated dataclass initializers: only Piece has these fields.
                if 'goal_distances' in code.co_varnames:
                    constructors['Piece'] = constructors.get('Piece', 0) + row.callcount
    differences = after.compare_to(before, 'lineno')
    return result, dict(traced_peak_bytes=peak, traced_retained_bytes=current,
                       net_retained_blocks=sum(d.count_diff for d in differences),
                       constructor_calls=constructors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seconds', type=float, default=1.)
    parser.add_argument('--evaluations', type=int, default=2000)
    args = parser.parse_args()
    start = perf_counter()
    record = load_record((ROOT/'intransitive/benchmarks/search_budget/game79.pgn').read_text())
    archived = baseline()
    archived.AlphaBetaPlayer(config=record.config)._prepare()
    common_setup = perf_counter() - start
    start = perf_counter()
    warm_material_kernels()
    material_setup = perf_counter() - start
    variants = {'baseline': archived, 'incremental': search}
    configs = {mode: replace(record.config, max_depth=4 if mode == 'fixed' else 20,
                             node_limit=10**9, time_limit=600. if mode == 'fixed' else args.seconds)
               for mode in ('fixed', 'time')}
    cases = {case['ply']: case for case in DATA['original_positions']}
    runs, core, memory = [], [], []
    for repeat in range(args.repeats):
        for ply in (0, 27, 35, 61):
            state = record.states[ply]
            board = Position.from_storage(state)
            order = list(variants) if repeat % 2 == 0 else list(reversed(variants))
            for name in order:
                module = variants[name]
                evaluator = module.Evaluator(module.AlphaBetaPlayer().game, configs['fixed'])
                counts = count_pieces(state)
                budget = Budget(10**12, 600)
                def evaluate():
                    return evaluator.score(state, int(state[:, :, 32].flat[1]), budget,
                        proof={'status': 'unknown'}, **({'counts': counts} if name == 'incremental' else {}))
                start = perf_counter()
                evaluate()
                first_evaluation = perf_counter() - start
                start = perf_counter()
                for _ in range(args.evaluations):
                    score = evaluate()
                core.append(dict(repeat=repeat+1, ply=ply, variant=name, score=score,
                                 evaluations=args.evaluations, seconds=perf_counter()-start,
                                 first_evaluation_seconds=first_evaluation))
                if repeat == 0:
                    _, allocation = allocation_sample(lambda: [evaluate() for _ in range(100)])
                    memory.append(dict(kind='core_100', ply=ply, variant=name, **allocation))
                for mode, config in configs.items():
                    player = module.AlphaBetaPlayer(config=config)
                    result = player.analyze(state)
                    fields = asdict(result)
                    fields.pop('explanation')
                    fields['main_nodes'] = result.nodes - result.proof_nodes
                    fields['main_nodes_per_second'] = fields['main_nodes'] / result.elapsed
                    fields['timeout_overshoot'] = max(0., result.elapsed - config.time_limit)
                    fields['move'] = move_name(result.action)
                    fields['tactical_objective_met'] = (meets_objective(board, result.action, cases[ply])
                                                       if ply in cases else None)
                    fields['material_cache'] = (player.evaluator.material.values.cache_info()._asdict()
                                                if name == 'incremental' else None)
                    fields['material_array_bytes'] = (fields['material_cache']['currsize'] * 2 * 64 * 8
                                                       if fields['material_cache'] else 0)
                    runs.append(dict(repeat=repeat+1, ply=ply, variant=name, mode=mode,
                                     state_sha256=state_hash(state), **fields))
                    print(f'{repeat+1} ply={ply} {mode} {name}: {result.elapsed:.4f}s', file=sys.stderr)
                    if repeat == 0 and mode == 'fixed':
                        measured_player = module.AlphaBetaPlayer(config=config)
                        _, allocation = allocation_sample(lambda: measured_player.analyze(state))
                        memory.append(dict(kind='search_depth4', ply=ply, variant=name, **allocation))
    for ply in (0, 27, 35, 61):
        rows = [r for r in runs if r['mode'] == 'fixed' and r['ply'] == ply]
        for row in rows[1:]:
            for key in ('score', 'action', 'pv', 'nodes', 'proof_nodes', 'work', 'completed_depth',
                        'selected_depth', 'tactical_objective_met'):
                assert row[key] == rows[0][key], (ply, key, row[key], rows[0][key])
        scores = [r['score'].hex() for r in core if r['ply'] == ply]
        assert len(set(scores)) == 1, (ply, scores)
    report = dict(schema=1, baseline_revision='4a007f709ef3f8d81d2cc27d7a9c54efa2f69d2c',
                  python=sys.version, platform=platform.platform(),
                  common_setup_seconds=common_setup, material_setup_seconds=material_setup,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                                 (1 if sys.platform == 'darwin' else 1024),
                  configs={mode: asdict(config) for mode, config in configs.items()},
                  implementation_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (ROOT/'intransitive/heuristics').glob('*.py')},
                  repeats=args.repeats, runs=runs, core=core, memory=memory)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
