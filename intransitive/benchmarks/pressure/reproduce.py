"""Opt-in pressure calibration; never reads/writes live training artifacts."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
import time
import numpy as np

from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveDisplay import parse_move, move_to_str
from intransitive.heuristics import AlphaBetaPlayer, SearchConfig
from intransitive.heuristics.budget import Budget
from intransitive.heuristics.evaluation import Evaluator
from intransitive.heuristics.pressure import pressure_totals, warm_pressure_kernel
from intransitive.tests.test_heuristics import position


def opening_state():
    game = IntransitiveGame(modelling_draws=False)
    state,side = game.getInitBoard(),0
    states = [state]
    for move in ('D3 E4','F6 E5','C3 D3','E5 E4','D3 E4',
                 'G7 F6','C4 D5','F6 E5','D5 E5','G6 F6'):
        state,side = game.getNextState(state,side,parse_move(move))
        states.append(state)
    assert hashlib.sha256(state.tobytes()).hexdigest() == 'a1d39de802f4ce506604d952b9d48cb49956963323397bb18298ae5e922e688b'
    return states[7]


def cases():
    return [('reported_opening',opening_state()),
            ('retreat_from_paper',position({'E5':1,'F5':-3})),
            ('win_before_saving_piece',position({'H8':2,'D4':-1,'E4':-1,'F4':-1}))]


def timing(function, iterations=2000):
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(iterations):
            function()
        samples.append((time.perf_counter()-start)/iterations)
    return dict(median_microseconds=median(samples)*1e6,samples_microseconds=[s*1e6 for s in samples])


def write(path, value):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--weights',type=float,nargs='+',default=[0.,5.,10.,25.])
    parser.add_argument('--depth',type=int,default=5)
    parser.add_argument('--seconds',type=float,default=120.)
    parser.add_argument('--radius',type=int,choices=(3,4),default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory to preserve prior evidence')
    args.output.mkdir(parents=True)
    if hasattr(os,'nice'):
        os.nice(10)
    fixtures = cases()
    base = SearchConfig(max_depth=args.depth,node_limit=1_000_000_000,time_limit=args.seconds,
                        pressure_radius=args.radius)
    warm_start = time.perf_counter()
    warm_pressure_kernel()
    AlphaBetaPlayer(config=base)._prepare()
    warm_seconds = time.perf_counter()-warm_start
    state = fixtures[0][1]
    quiet = dict(status='unknown')
    micro = {}
    # Give timing loops their own generous non-search work allowance.
    for enabled in (False,True):
        evaluator = Evaluator(IntransitiveGame(),replace(base,pressure_enabled=enabled,pressure_weight=10.))
        budget = Budget(10**12,3600.)
        evaluator.score(state,0,budget,proof=quiet)
        micro['pressure' if enabled else 'baseline'] = timing(lambda:evaluator.score(state,0,budget,proof=quiet))
    pieces = state[:,:,0].copy()
    micro['kernel'] = timing(lambda:pressure_totals(pieces,args.radius))
    report = dict(platform=platform.platform(),processor=platform.processor(),
        startup_warm_seconds=warm_seconds,weights=args.weights,base_config=asdict(base),
        micro=micro,results=[],notes=[
            'One low-priority benchmark process; live training/generation left running.',
            'Micro timings exclude terminal proof search, include terminal checks and work accounting.',
            'Different weights change search ordering/visited trees; elapsed ratios are not pure kernel overhead.',
            'Avoiding the reported move is not proof that another selected move is optimal.',
            'No strength or tuned-weight claim from this small, known fixture set.'])
    source_root = Path(__file__).resolve().parents[2]
    report['source_sha256'] = {str(p.relative_to(source_root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((source_root/'heuristics').glob('*.py'))}
    write(args.output/'results.json',report)
    for name,state in fixtures:
        side = int(state[:,:,82:84].flat[1])
        for weight in args.weights:
            config = replace(base,pressure_enabled=bool(weight),pressure_weight=weight)
            result = AlphaBetaPlayer(config=config).analyze(state)
            row = dict(case=name,weight=weight,move=move_to_str(result.action),
                score=result.score,completed_depth=result.completed_depth,stop_reason=result.stop_reason,
                elapsed=result.elapsed,work=result.work,nodes=result.nodes,
                pressure_calls=result.module_calls.get('local_pressure',0),
                pressure_seconds=result.module_seconds.get('local_pressure',0.),
                pv=[move_to_str(a) for a in result.pv])
            if name == 'reported_opening':
                row['avoids_reported_blunder'] = result.action != parse_move('F6 E5')
            if name == 'retreat_from_paper':
                child,opponent = IntransitiveGame().getNextState(state,side,result.action)
                row['avoids_immediate_capture'] = all(
                    np.count_nonzero(IntransitiveGame().getNextState(child,opponent,int(action))[0][:,:,0] > 0) == 1
                    for action in np.flatnonzero(IntransitiveGame().getValidMoves(child,opponent)))
            if name == 'win_before_saving_piece':
                child,opponent = IntransitiveGame().getNextState(state,side,result.action)
                row['wins_immediately'] = bool(IntransitiveGame().getGameEnded(child,opponent)[side] == 1)
            report['results'].append(row)
            write(args.output/'results.json',report)
            print(json.dumps(row),flush=True)


if __name__=='__main__':
    main()
