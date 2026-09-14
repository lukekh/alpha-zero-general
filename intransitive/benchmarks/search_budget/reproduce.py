"""Repeat issue #37's work-first/time-first searches with fresh tables."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import resource
from statistics import mean, median
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.heuristics.config import TIME_FIRST_LIMITS
from intransitive.heuristics.search import AlphaBetaPlayer
from intransitive.record import load_record, state_hash


FIXTURE = Path(__file__).with_name('game79.pgn')
BASELINE_REVISION = '3bdf9333329b1499bc7c35bc625e6a314d0084b3'


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == 'darwin' else value * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seconds', type=float, default=5.)
    parser.add_argument('--work-cap', type=int, default=200_000)
    parser.add_argument('--time-first-cap', type=int,
                        default=TIME_FIRST_LIMITS['node_limit'])
    args = parser.parse_args()
    if args.repeats < 1 or args.seconds <= 0 or args.work_cap < 1 or args.time_first_cap < 1:
        parser.error('Use positive repeats, seconds and work caps')

    record = load_record(FIXTURE.read_text())
    game = IntransitiveGame()
    config = replace(record.config, max_depth=20, time_limit=args.seconds)
    setup_start = perf_counter()
    warm = AlphaBetaPlayer(game, replace(config, max_depth=1, node_limit=10_000))
    warm._prepare()
    for ply in (35, 61):
        side = int(record.states[ply][:, :, 82:84].flat[1])
        game.getValidMoves(record.states[ply], side)
    setup_elapsed = perf_counter() - setup_start

    runs = []
    policies = (('work_cap', args.work_cap), ('time_first', args.time_first_cap))
    for repeat in range(1, args.repeats + 1):
        for ply in (35, 61):
            state = record.states[ply]
            for policy, work_limit in policies:
                result = AlphaBetaPlayer(
                    game, replace(config, node_limit=work_limit)).analyze(state)
                runs.append(dict(
                    repeat=repeat, ply=ply, state_sha256=state_hash(state),
                    policy=policy, configured_time=args.seconds,
                    configured_work=work_limit,
                    **{name: value for name, value in asdict(result).items()
                       if name in ('elapsed', 'work', 'nodes', 'proof_nodes', 'table_bytes',
                                   'completed_depth', 'selected_depth', 'partial_depth',
                                   'root_moves_completed', 'root_moves_total',
                                   'selection_source', 'stop_reason', 'diagnostics_status')},
                ))

    summary = []
    for ply in (35, 61):
        for policy, work_limit in policies:
            rows = [row for row in runs if row['ply'] == ply and row['policy'] == policy]
            summary.append(dict(
                ply=ply, policy=policy, configured_time=args.seconds,
                configured_work=work_limit,
                elapsed_mean=mean(row['elapsed'] for row in rows),
                elapsed_median=median(row['elapsed'] for row in rows),
                work_mean=mean(row['work'] for row in rows),
                completed_depths=[row['completed_depth'] for row in rows],
                selected_depths=[row['selected_depth'] for row in rows],
                partial_depths=[row['partial_depth'] for row in rows],
                selection_sources=[row['selection_source'] for row in rows],
                stop_reasons=[row['stop_reason'] for row in rows],
            ))

    report = dict(
        schema=1,
        issue='https://github.com/lukekh/alpha-zero-general/issues/37',
        baseline_revision=BASELINE_REVISION,
        fixture=str(FIXTURE.relative_to(ROOT)),
        fixture_final_state_sha256=record.tags['StateSHA256'],
        environment=dict(python=sys.version.split()[0], platform=platform.platform()),
        protocol=dict(
            repeats=args.repeats, plies=[35, 61], max_depth=20,
            time_limit=args.seconds, work_cap=args.work_cap,
            time_first_cap=args.time_first_cap, proof_depth=config.proof_depth,
            proof_nodes=config.proof_nodes, table_entries=config.table_entries,
            evaluator_version=config.evaluator_version,
            note='Fresh player/table per run; compilation warmed before timed Budget creation.',
        ),
        setup_elapsed=setup_elapsed,
        peak_rss_bytes=peak_rss_bytes(),
        summary=summary,
        runs=runs,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
