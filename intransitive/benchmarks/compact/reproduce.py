"""Reproduce #40 archive parity, transition/boundary costs and search outcomes."""
import argparse
import ast
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
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from intransitive.heuristics import search
from intransitive.heuristics.position import SearchPosition, warm_position_kernels
from intransitive.heuristics.proof import warm_compact_proof_kernel
from intransitive.record import load_record, state_hash
from intransitive.tests.reference_rules import Position
from intransitive.tests.test_game_blunders import DATA, meets_objective
from intransitive.tests.tactical_oracle import move_name


def baseline():
    evidence = Path(__file__).with_name('evidence')
    manifest = json.loads((evidence/'baseline-manifest.json').read_text())
    archive_path = evidence/'baseline.tar.gz'
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == manifest['archive_sha256']
    modules = []
    with tarfile.open(archive_path) as archive:
        for path, digest in manifest['files'].items():
            assert hashlib.sha256(archive.extractfile(path).read()).hexdigest() == digest, path
            # Shared rules/config/replay and evaluator kernels must be identical.
            if (path.startswith('intransitive/') and '/tests/' not in path
                    and path not in {'intransitive/heuristics/search.py',
                                     'intransitive/heuristics/evaluation.py',
                                     'intransitive/heuristics/proof.py'}):
                assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
        # The reference compiled proof functions are reused, not rewritten.
        # Refuse a comparison if any archived function changes in the live file.
        proof_path = 'intransitive/heuristics/proof.py'
        original = ast.parse(archive.extractfile(proof_path).read())
        current = {node.name: ast.dump(node) for node in ast.parse((ROOT/proof_path).read_text()).body
                   if isinstance(node, ast.FunctionDef)}
        for node in original.body:
            if isinstance(node, ast.FunctionDef):
                assert current[node.name] == ast.dump(node), node.name
        for name in ('evaluation', 'search'):
            module_name = 'intransitive.heuristics._issue40_baseline_' + name
            module = types.ModuleType(module_name)
            module.__package__ = 'intransitive.heuristics'
            sys.modules[module_name] = module
            source = archive.extractfile('intransitive/heuristics/' + name + '.py').read()
            exec(compile(source, '<baseline_' + name + '>', 'exec'), module.__dict__)
            modules.append(module)
    evaluation, archived = modules
    archived.Evaluator = evaluation.Evaluator
    archived.terminal_value = evaluation.terminal_value
    original_player = archived.AlphaBetaPlayer

    class ArchivedPlayer(original_player):
        def analyze(self, *args, **kwargs):
            # The archived evaluator imports .search.prove lazily for root
            # diagnostics. Keep that import on its original implementation too.
            with patch.object(search, 'prove', archived.prove):
                return super().analyze(*args, **kwargs)

    archived.AlphaBetaPlayer = ArchivedPlayer
    return archived, manifest['revision']


def allocations(call):
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
    calls = {}
    for row in profile.getstats():
        code = row.code
        if not isinstance(code, str) and code.co_name in ('getNextState', 'push', 'pop', 'export', '__init__'):
            if code.co_name != '__init__' or code.co_filename.endswith('/heuristics/position.py'):
                label = code.co_filename.split('/')[-1] + ':' + code.co_name
                calls[label] = calls.get(label, 0) + row.callcount
    return result, dict(traced_peak_bytes=peak, traced_retained_bytes=current,
        net_retained_blocks=sum(d.count_diff for d in after.compare_to(before, 'lineno')),
        calls=calls)


def time_calls(call, count):
    start = perf_counter()
    for _ in range(count):
        call()
    return perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seconds', type=float, default=1.)
    parser.add_argument('--transitions', type=int, default=2000)
    parser.add_argument('--depth', type=int, default=4)
    args = parser.parse_args()
    start = perf_counter()
    record = load_record((ROOT/'intransitive/benchmarks/search_budget/game79.pgn').read_text())
    archived, revision = baseline()
    archived.AlphaBetaPlayer(config=record.config)._prepare()
    common_setup = perf_counter() - start
    start = perf_counter()
    warm_position_kernels()
    position_setup = perf_counter() - start
    start = perf_counter()
    warm_compact_proof_kernel()
    proof_setup = perf_counter() - start
    variants = {'baseline': archived, 'compact': search}
    configs = {mode: replace(record.config, max_depth=args.depth if mode == 'fixed' else 20,
        node_limit=10**9, time_limit=600. if mode == 'fixed' else args.seconds) for mode in ('fixed', 'time')}
    cases = {case['ply']: case for case in DATA['original_positions']}
    runs, transitions, memory = [], [], []
    for repeat in range(args.repeats):
        for ply in (0, 27, 35, 61):
            state = record.states[ply]
            board = Position.from_storage(state)
            game = search.AlphaBetaPlayer().game
            node = SearchPosition(state)
            action = int(node.legal()[0])
            def public_transition():
                return game.getNextState(state, board.player, action)
            def compact_transition():
                node.push(action)
                node.pop()
            calls = {'public_make': public_transition, 'compact_make_unmake': compact_transition,
                     'import': lambda: SearchPosition(state), 'export': node.export,
                     'roundtrip': lambda: SearchPosition(state).export()}
            order = list(variants) if repeat % 2 == 0 else list(reversed(variants))
            for kind, call in calls.items():
                seconds = time_calls(call, args.transitions)
                transitions.append(dict(repeat=repeat+1, ply=ply, kind=kind,
                                        count=args.transitions, seconds=seconds))
                if repeat == 0:
                    _, measured = allocations(lambda: time_calls(call, 100))
                    memory.append(dict(kind=kind+'_100', ply=ply, **measured))
            assert node.export().tobytes() == state.tobytes()
            for name in order:
                module = variants[name]
                for mode, config in configs.items():
                    player = module.AlphaBetaPlayer(config=config)
                    result = player.analyze(state)
                    fields = asdict(result)
                    fields.pop('explanation')
                    fields['main_nodes'] = result.nodes - result.proof_nodes
                    fields['main_nodes_per_second'] = fields['main_nodes'] / result.elapsed
                    fields['nodes_per_second'] = result.nodes / result.elapsed
                    fields['timeout_overshoot'] = max(0., result.elapsed - config.time_limit)
                    fields['move'] = move_name(result.action)
                    fields['tactical_objective_met'] = (meets_objective(board, result.action, cases[ply]) if ply in cases else None)
                    runs.append(dict(repeat=repeat+1, ply=ply, variant=name, mode=mode,
                                     state_sha256=state_hash(state), **fields))
                    print(f'{repeat+1} ply={ply} {mode} {name}: {result.elapsed:.4f}s depth={result.completed_depth}',file=sys.stderr)
                    if repeat == 0 and mode == 'fixed':
                        measured_player = module.AlphaBetaPlayer(config=config)
                        _, measured = allocations(lambda: measured_player.analyze(state))
                        memory.append(dict(kind='search_fixed', ply=ply, variant=name, **measured))
    for ply in (0, 27, 35, 61):
        rows = [r for r in runs if r['mode'] == 'fixed' and r['ply'] == ply]
        for row in rows[1:]:
            for key in ('score', 'action', 'pv', 'nodes', 'proof_nodes', 'work', 'completed_depth',
                        'selected_depth', 'root_moves_completed', 'tt_hits', 'tactical_objective_met'):
                assert row[key] == rows[0][key], (ply, key, row[key], rows[0][key])
    report = dict(schema=1, baseline_revision=revision, python=sys.version, platform=platform.platform(),
        setup_seconds=dict(common=common_setup, compact_position=position_setup, compact_proof=proof_setup),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
        configs={mode: asdict(config) for mode, config in configs.items()},
        implementation_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'intransitive/heuristics').glob('*.py')},
        repeats=args.repeats, runs=runs, transitions=transitions, memory=memory)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
