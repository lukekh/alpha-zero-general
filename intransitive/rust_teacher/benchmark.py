"""Same-position, same-score comparisons with the optimized Python teacher."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import time

from . import BINARY, HERE, RustTeacher
from ..heuristics import AlphaBetaPlayer, SearchConfig
from ..heuristics.budget import Budget
from ..heuristics.position import SearchPosition
from ..supervised_minimax import generate_position, write_json
from ..IntransitiveDisplay import move_to_str


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--positions', type=int, default=6)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--weight', type=float, default=10.)
    parser.add_argument('--seed', type=int, default=2026091600)
    args = parser.parse_args()
    if args.output.exists() or args.positions < 1 or args.repeats < 1:
        parser.error('Use a fresh output directory and positive counts')
    args.output.mkdir(parents=True)
    os.nice(10)
    config = SearchConfig(max_depth=args.depth, node_limit=1_000_000_000,
        time_limit=args.seconds, pressure_enabled=True, pressure_radius=3, pressure_weight=args.weight,
        pvs_enabled=True, aspiration_enabled=True, ordering_enabled=True,
        compiled_ordering_enabled=True, depth_replacement_enabled=True, table_entries=50000)
    begin = time.perf_counter()
    AlphaBetaPlayer(config=config)._prepare()
    report = dict(platform=platform.platform(), rustc=subprocess.check_output(['rustc','--version'],text=True).strip(),
        binary_sha256=hashlib.sha256(BINARY.read_bytes()).hexdigest(), config=config.to_dict(),
        rust_source_sha256={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((HERE/'src').glob('*.rs'))},
        python_warmup_seconds=time.perf_counter()-begin, rows=[],
        notes=['Single worker, low priority, background training/generation remain running.',
               'Both teachers use material + 7x7 pressure, weight specified in config, depth-2/64-node terminal proofs.',
               'Rust PVS/FIFO table/full ordered-history identity differs from Python aspiration/depth-aware table/occurrence identity.',
               'Scores and chosen actions are checked; equivalent tied moves may differ.',
               'Native node_limit counts visits, not Python logical-work units; incomplete searches are excluded.',
               'No promoted roots, ablations or dataset I/O in search-throughput measurements.'])
    with RustTeacher() as rust:
        for index in range(args.positions):
            stage = ('opening','midgame','endgame')[index%3]
            state, provenance = generate_position(args.seed+index, stage)
            (args.output/f'position-{index}.bin').write_bytes(state.tobytes())
            for trial in range(args.repeats):
                row = dict(index=index, trial=trial, stage=stage, provenance=provenance,
                    state_sha256=hashlib.sha256(state.tobytes()).hexdigest())
                player = AlphaBetaPlayer(config=config)
                # The second engine changes every trial/position to reduce order bias.
                for engine in (('rust','python') if (index+trial)%2==0 else ('python','rust')):
                    begin = time.perf_counter()
                    if engine=='rust':
                        row['rust'] = rust.analyze(state,depth=args.depth,seconds=args.seconds,weight=args.weight)
                    else:
                        result = player.analyze(state)
                        row['python'] = dict(action=result.action, score=result.score,
                            completed_depth=result.completed_depth,
                            complete=result.completed_depth==args.depth or result.stop_reason=='proven_result',
                            seconds=result.elapsed, nodes=result.nodes, stop_reason=result.stop_reason)
                    row[engine]['wall_seconds'] = time.perf_counter()-begin
                    row[engine]['move'] = move_to_str(row[engine]['action']) if row[engine]['action'] is not None else None
                native, reference = row['rust'], row['python']
                row['both_complete'] = native['complete'] and reference['complete']
                if row['both_complete']:
                    row['score_matches'] = abs(native['score']-reference['score']) < 1e-8
                    row['same_action'] = native['action']==reference['action']
                    if not row['same_action']:
                        child = SearchPosition(state)
                        child.push(native['action'])
                        score, _ = player._search(child,reference['completed_depth']-1,-float('inf'),float('inf'),1,
                                                 Budget(1_000_000_000,args.seconds))
                        row['native_action_python_score'] = -score
                        row['native_action_optimal'] = abs(-score-reference['score']) < 1e-8
                    else:
                        row['native_action_optimal'] = True
                    row['speedup'] = reference['wall_seconds']/native['wall_seconds']
                report['rows'].append(row)
                write_json(args.output/'results.json', report)
                print(json.dumps({k:v for k,v in row.items() if k!='provenance'}),flush=True)
                if row['both_complete']:
                    assert row['score_matches'] and row['native_action_optimal'], 'Native semantic mismatch; benchmark rejected'
    complete = [r for r in report['rows'] if r['both_complete']]
    report['summary'] = dict(paired_complete=len(complete), attempted=len(report['rows']),
        all_scores_match=all(r['score_matches'] and r['native_action_optimal'] for r in complete))
    if complete:
        report['summary'].update(median_paired_speedup=median(r['speedup'] for r in complete),
            python_total_seconds=sum(r['python']['wall_seconds'] for r in complete),
            rust_total_seconds=sum(r['rust']['wall_seconds'] for r in complete))
    write_json(args.output/'results.json', report)
    print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':
    main()
