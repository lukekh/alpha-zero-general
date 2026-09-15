"""Interleaved 7x7 versus 9x9 timing and tactical checks; no live-job writes."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
import time

from intransitive.benchmarks.pressure.reproduce import opening_state, timing, write
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveDisplay import parse_move, move_to_str
from intransitive.heuristics import AlphaBetaPlayer, SearchConfig
from intransitive.heuristics.budget import Budget
from intransitive.heuristics.evaluation import Evaluator
from intransitive.heuristics.pressure import pressure_totals, warm_pressure_kernel
from intransitive.tests.test_tactics import CASES
from intransitive.tests.tactical_oracle import certify, load_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--seconds',type=float,default=120.)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        parser.error('Use a fresh output path and at least one repeat')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if hasattr(os,'nice'):
        os.nice(10)
    base = SearchConfig(max_depth=5,node_limit=1_000_000_000,time_limit=args.seconds)
    variants = [('baseline',base),
                ('9x9',replace(base,pressure_enabled=True,pressure_weight=10.,pressure_radius=4)),
                ('7x7',replace(base,pressure_enabled=True,pressure_weight=10.,pressure_radius=3))]
    state = opening_state()
    start = time.perf_counter()
    warm_pressure_kernel()
    for _,config in variants:
        AlphaBetaPlayer(config=config)._prepare()
    root = Path(__file__).resolve().parents[2]
    report = dict(platform=platform.platform(),warmup_seconds=time.perf_counter()-start,
        configurations={name:asdict(config) for name,config in variants},
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted((root/'heuristics').glob('*.py'))},
        micro=[],searches=[],candidates=[],tactics=[],notes=[
            'Live jobs left running; low-priority process; timings subject to contention.',
            '7x7 retains weights 2, 1, 0.5; only distance-four contributions are removed.',
            'Root searches use fresh players and rotate variant order across repeats.',
            'Known opening and tactical controls do not measure general playing strength.'])
    evaluators = {name:Evaluator(IntransitiveGame(),config) for name,config in variants}
    budget = Budget(10**12,3600.)
    pieces = state[:,:,0].copy()
    for trial in range(args.repeats):
        order = variants[trial%len(variants):]+variants[:trial%len(variants)]
        for name,config in order:
            evaluate = lambda: evaluators[name].score(state,1,budget,proof=dict(status='unknown'))
            evaluate()
            row = dict(trial=trial,variant=name,evaluator=timing(evaluate))
            if config.pressure_enabled:
                row['kernel'] = timing(lambda:pressure_totals(pieces,config.pressure_radius))
            report['micro'].append(row)
        write(args.output,report)
    for trial in range(args.repeats):
        order = variants[trial%len(variants):]+variants[:trial%len(variants)]
        for name,config in order:
            result = AlphaBetaPlayer(config=config).analyze(state)
            row = dict(trial=trial,variant=name,move=move_to_str(result.action),score=result.score,
                completed_depth=result.completed_depth,stop_reason=result.stop_reason,
                seconds=result.elapsed,nodes=result.nodes,work=result.work,
                pressure_seconds=result.module_seconds.get('local_pressure',0.),
                avoids_reported_blunder=result.action != parse_move('F6 E5'))
            report['searches'].append(row)
            write(args.output,report)
            print(json.dumps(row),flush=True)
    for name,config in variants:
        for move in ('F6 E5','F6 F5'):
            child,_ = IntransitiveGame().getNextState(state,1,parse_move(move))
            result = AlphaBetaPlayer(config=replace(config,max_depth=4)).analyze(child)
            report['candidates'].append(dict(variant=name,move=move.replace(' ','-'),
                score_for_red=-result.score,child_completed_depth=result.completed_depth,
                stop_reason=result.stop_reason))
        write(args.output,report)
    fixtures = []
    for case in CASES:
        certify(case)
        reference,expected = load_case(case)
        fixtures.append((case,reference.storage(),expected))
    for name,config in variants:
        rows = []
        for case,state,expected in fixtures:
            result = AlphaBetaPlayer(config=replace(config,max_depth=3)).analyze(state)
            complete = result.completed_depth >= 3 or result.stop_reason == 'proven_result'
            rows.append(dict(id=case['id'],category=case['category'],
                passed=bool(result.action == expected and complete),selected=move_to_str(result.action),
                expected=case['expected'],completed_depth=result.completed_depth,stop_reason=result.stop_reason))
        report['tactics'].append(dict(variant=name,passed=sum(row['passed'] for row in rows),
            total=len(rows),failures_by_category=dict(Counter(row['category'] for row in rows if not row['passed'])),
            results=rows))
        write(args.output,report)
    report['summary'] = {name:dict(
        median_search_seconds=median(row['seconds'] for row in report['searches'] if row['variant']==name),
        median_evaluator_microseconds=median(row['evaluator']['median_microseconds'] for row in report['micro'] if row['variant']==name),
        all_searches_complete=all(row['completed_depth']==5 for row in report['searches'] if row['variant']==name))
        for name,_ in variants}
    write(args.output,report)
    print(json.dumps(dict(summary=report['summary'],candidates=report['candidates'],
                         tactics=[{k:v for k,v in row.items() if k!='results'} for row in report['tactics']])),flush=True)


if __name__=='__main__':
    main()
