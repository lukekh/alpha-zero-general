"""Forced opening alternatives and independently certified tactical controls."""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import json
import time

from intransitive.benchmarks.pressure.reproduce import opening_state, write
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveDisplay import parse_move, move_to_str
from intransitive.heuristics import AlphaBetaPlayer, SearchConfig
from intransitive.tests.test_tactics import CASES
from intransitive.tests.tactical_oracle import certify, load_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--weights',type=float,nargs='+',default=[0.,5.,10.,25.])
    parser.add_argument('--radius',type=int,choices=(3,4),default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh evidence path')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if hasattr(os,'nice'):
        os.nice(10)
    report = dict(pressure_radius=args.radius,opening_candidates=[],tactics=[],notes=[
        'Each forced move plus completed depth four is a five-ply comparison.',
        'The baseline and pressure scores have different units; compare moves within each weight.',
        'Tactical answers certified with an independent rules/search implementation before scoring.',
        'Known fixtures are calibration controls, not a held-out strength evaluation.'])
    root = Path(__file__).resolve().parents[2]
    report['source_sha256'] = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root/'heuristics').glob('*.py'))}
    game,state = IntransitiveGame(),opening_state()
    side = int(state[:,:,82:84].flat[1])
    for weight in args.weights:
        config = SearchConfig(max_depth=4,node_limit=1_000_000_000,time_limit=120.,
                              pressure_enabled=bool(weight),pressure_weight=weight,pressure_radius=args.radius)
        for move in ('F6 E5','F6 F5'):
            child,_ = game.getNextState(state,side,parse_move(move))
            result = AlphaBetaPlayer(config=config).analyze(child)
            row = dict(weight=weight,move=move.replace(' ','-'),score_for_red=-result.score,
                child_completed_depth=result.completed_depth,stop_reason=result.stop_reason,
                seconds=result.elapsed,pv=[move_to_str(a) for a in result.pv])
            report['opening_candidates'].append(row)
            write(args.output,report)
            print(json.dumps(row),flush=True)
    fixtures = []
    for case in CASES:
        certify(case)
        reference,expected = load_case(case)
        fixtures.append((case,reference.storage(),expected))
    for weight in args.weights:
        config = SearchConfig(max_depth=3,node_limit=1_000_000_000,time_limit=30.,
                              pressure_enabled=bool(weight),pressure_weight=weight,pressure_radius=args.radius)
        rows = []
        started = time.perf_counter()
        for case,state,expected in fixtures:
            result = AlphaBetaPlayer(config=config).analyze(state)
            complete = result.completed_depth >= 3 or result.stop_reason == 'proven_result'
            rows.append(dict(id=case['id'],category=case['category'],expected=case['expected'],
                selected=move_to_str(result.action),passed=bool(result.action == expected and complete),
                completed_depth=result.completed_depth,stop_reason=result.stop_reason))
        summary = dict(weight=weight,passed=sum(row['passed'] for row in rows),total=len(rows),
            failures_by_category=dict(Counter(row['category'] for row in rows if not row['passed'])),
            seconds=time.perf_counter()-started,results=rows)
        report['tactics'].append(summary)
        write(args.output,report)
        print(json.dumps({key:value for key,value in summary.items() if key != 'results'}),flush=True)


if __name__=='__main__':
    main()
