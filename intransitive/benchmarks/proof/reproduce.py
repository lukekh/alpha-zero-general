"""Four proof ablations at identical depth and repeated equal wall time."""
import argparse
from dataclasses import asdict, replace
from functools import partial
import json
import hashlib
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from intransitive.heuristics import search
from intransitive.heuristics import proof
from intransitive.record import load_record, state_hash
from intransitive.tests.reference_rules import Position
from intransitive.tests.test_game_blunders import DATA, meets_objective
from intransitive.tests.tactical_oracle import move_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seconds', type=float, default=1.)
    args = parser.parse_args()
    start = perf_counter()
    record = load_record((ROOT/'intransitive/benchmarks/search_budget/game79.pgn').read_text())
    search.warm_search_kernels()
    rules_setup = perf_counter() - start
    start = perf_counter()
    proof.warm_proof_kernel()
    proof_setup = perf_counter() - start
    native = proof.native_proof
    current = search.prove
    variants = ('reference', 'ordering', 'compiled', 'specialised')
    configs = {mode: replace(record.config, max_depth=4 if mode == 'fixed' else 20,
                             node_limit=10**9, time_limit=600. if mode == 'fixed' else args.seconds)
               for mode in ('fixed', 'time')}
    cases = {case['ply']: case for case in DATA['original_positions']}
    runs = []
    # Rotate variant order to reduce consistent warm/load ordering bias.
    for repeat in range(args.repeats):
        for ply in (0, 27, 35, 61):
            state = record.states[ply]
            board = Position.from_storage(state)
            for mode, config in configs.items():
                order = variants[repeat % 4:] + variants[:repeat % 4]
                for variant in order:
                    implementation = (search.prove_reference if variant == 'reference' else
                                      partial(search.prove_reference, compiled_order=True) if variant == 'ordering'
                                      else partial(current, specialised=variant != 'compiled'))
                    with patch.object(search, 'prove', implementation):
                        result = search.AlphaBetaPlayer(config=config).analyze(state)
                    fields = asdict(result)
                    fields.pop('explanation')
                    fields['main_nodes'] = result.nodes - result.proof_nodes
                    fields['timeout_overshoot'] = max(0., result.elapsed - config.time_limit)
                    fields['move'] = move_name(result.action)
                    fields['tactical_objective_met'] = (meets_objective(board, result.action, cases[ply])
                                                        if ply in cases else None)
                    runs.append(dict(repeat=repeat + 1, ply=ply, variant=variant, mode=mode,
                                     state_sha256=state_hash(state), **fields))
                    print(f'{repeat + 1} ply={ply} {mode} {variant}: {result.elapsed:.4f}s', file=sys.stderr)
    # Fail before publishing if performance was bought with different work or
    # a changed fixed-depth decision/certificate.
    for ply in (0, 27, 35, 61):
        fixed = [row for row in runs if row['mode'] == 'fixed' and row['ply'] == ply]
        reference = fixed[0]
        for row in fixed[1:]:
            for name in ('score', 'action', 'pv', 'nodes', 'proof_nodes', 'work',
                         'completed_depth', 'selected_depth', 'tactical_objective_met'):
                if row[name] != reference[name]:
                    raise AssertionError((ply, row['variant'], name, row[name], reference[name]))
    # Observe individual cancellation interval, separately from search timing.
    intervals = []
    for ply in (27, 35, 61):
        for depth in (2, 8):
            for _ in range(100):
                start = perf_counter()
                _, _, counts = native(record.states[ply], depth, 64, 10**9, True)
                intervals.append(dict(ply=ply, depth=depth, seconds=perf_counter()-start,
                                      nodes=int(counts[1]), work=int(counts[0])))
    report = dict(schema=1, baseline_revision='216cdc2d3580381fc72755cd42a0fb15286855c1',
                  python=sys.version, platform=platform.platform(),
                  rules_setup_seconds=rules_setup, proof_setup_seconds=proof_setup,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                                 (1 if sys.platform == 'darwin' else 1024),
                  configs={mode: asdict(config) for mode, config in configs.items()},
                  implementation_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (ROOT/"intransitive/heuristics").glob("*.py")},
                  repeats=args.repeats, runs=runs, native_intervals=intervals)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
