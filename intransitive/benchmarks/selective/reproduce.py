"""Bounded held-out measurements; run as a module from repository root."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import numpy as np
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveLogicNumba import search_observation
from intransitive.IntransitiveSymmetries import transform_state
from intransitive.heuristics import AlphaBetaPlayer, SearchConfig
from intransitive.rust_teacher import RustTeacher
from intransitive.tests.test_heuristics import position

MODES={'baseline':{},'nmp':{'nmp_enabled':True},'futility':{'futility_enabled':True},
       'both':{'nmp_enabled':True,'futility_enabled':True}}
HERE=Path(__file__).resolve().parent

def fixtures():
    game=IntransitiveGame(modelling_draws=False)
    state=game.getInitBoard();side=0
    rng=np.random.default_rng(600917)
    for _ in range(24):
        action=int(rng.choice(np.flatnonzero(game.getValidMoves(state,side))))
        state,side=game.getNextState(state,side,action)
    return [('opening',game.getInitBoard()),('midgame',state),
            ('endgame',position({'D4':1,'F6':-2}))]

def wilson(wins,losses):
    n=wins+losses
    if not n:return [0.,1.]
    p=wins/n;z=1.96
    mid=(p+z*z/(2*n))/(1+z*z/n)
    half=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/(1+z*z/n)
    return [mid-half,mid+half]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',action='store_true')
    parser.add_argument('--archived-source',help='Exact git archive revision, when outside a checkout')
    parser.add_argument('--backend',choices=('python','rust'),default='rust')
    args=parser.parse_args()
    plan=json.loads((HERE/'plan.json').read_text())
    config=SearchConfig(max_depth=3,time_limit=600,node_limit=10**9,proof_nodes=0,
                        pvs_enabled=True,aspiration_enabled=True,
                        ordering_enabled=True,compiled_ordering_enabled=True)
    report=dict(plan=plan,backend=args.backend,platform=platform.platform(),
                revision=args.archived_source or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                source_diff_sha256=None if args.archived_source else hashlib.sha256(subprocess.check_output(['git','diff','HEAD'])).hexdigest(),
                baseline_config=config.to_dict(),fixed=[],paired=[])
    start=perf_counter();AlphaBetaPlayer(config=config)._prepare()
    report['python_warmup_seconds']=perf_counter()-start
    if args.reference:
        for stage,state in fixtures():
            for depth in (1,2,3):
                player=AlphaBetaPlayer(config=replace(config,max_depth=depth))
                for reuse in (False,True):
                    result=player.analyze(state)
                    report['fixed'].append(dict(stage=stage,depth=depth,reuse=reuse,
                        **{k:getattr(result,k) for k in ('action','score','completed_depth','pv','nodes','work','tt_hits')}))
        args.output.write_text(json.dumps(report,indent=2)+'\n');return
    start=perf_counter();native=RustTeacher()
    native.inspect(fixtures()[0][1]);report['native_startup_seconds']=perf_counter()-start
    def analyze(state,mode,depth,seconds):
        if args.backend=='rust':
            return native.analyze(search_observation(state),depth=depth,seconds=seconds,
                proof_nodes=0,weight=0.,table_entries=10000,**MODES[mode])
        return asdict(AlphaBetaPlayer(config=replace(config,max_depth=depth,time_limit=seconds,
                                                   **MODES[mode])).analyze(search_observation(state)))
    for stage,state in fixtures():
        for symmetry in (0,6):
            board=transform_state(state,symmetry)
            for depth in plan['depths']:
                for mode in MODES:
                    row=analyze(board,mode,depth,plan['fixed_depth_seconds_cap'])
                    report['fixed'].append(dict(stage=stage,symmetry=symmetry,depth=depth,mode=mode,
                        state_sha256=hashlib.sha256(board.tobytes()).hexdigest(),result=row))
                    args.output.write_text(json.dumps(report,indent=2)+'\n')
    game=IntransitiveGame(modelling_draws=False)
    for mode in ('nmp','futility','both'):
        for stage,start_state in fixtures():
            for candidate_side in (0,1):
                state=transform_state(start_state,6);moves=[];winner=None
                for ply in range(plan['paired_plies_cap']):
                    side=int(state[:,:,82:84].flat[1])
                    terminal=game.getGameEnded(state,side)
                    if terminal.any():winner=int(np.argmax(terminal));break
                    row=analyze(state,mode if side==candidate_side else 'baseline',20,plan['paired_seconds_per_move'])
                    action=row['action']
                    if action is None:action=int(np.flatnonzero(game.getValidMoves(state,side))[0])
                    assert game.getValidMoves(state,side)[action]
                    moves.append(dict(action=action,side=side,depth=row['completed_depth'],nodes=row['nodes']))
                    state,_=game.getNextState(state,side,action)
                terminal=game.getGameEnded(state,int(state[:,:,82:84].flat[1]))
                if terminal.any():winner=int(np.argmax(terminal))
                report['paired'].append(dict(mode=mode,stage=stage,candidate_side=candidate_side,
                    outcome='unfinished' if winner is None else 'win' if winner==candidate_side else 'loss',moves=moves))
                args.output.write_text(json.dumps(report,indent=2)+'\n')
    report['paired_summary']={}
    for mode in ('nmp','futility','both'):
        rows=[r for r in report['paired'] if r['mode']==mode]
        counts={outcome:sum(r['outcome']==outcome for r in rows) for outcome in ('win','loss','unfinished')}
        report['paired_summary'][mode]=dict(counts,decisive_wilson_95=wilson(counts['win'],counts['loss']))
    report['divergences'] = []
    baseline = {(r['stage'],r['symmetry'],r['depth']):r['result'] for r in report['fixed'] if r['mode']=='baseline'}
    for row in report['fixed']:
        if row['mode']=='baseline': continue
        result=row['result']; reference=baseline[row['stage'],row['symmetry'],row['depth']]
        if result['completed_depth'] != reference['completed_depth']:
            category='different_completed_depth_under_cap'
        elif result['score'] != reference['score']:
            category='selective_score_divergence'
        elif result['action'] != reference['action']:
            category='same_reported_score_different_action'
        else: continue
        report['divergences'].append(dict(stage=row['stage'],symmetry=row['symmetry'],depth=row['depth'],
            mode=row['mode'],classification=category,baseline_score=reference['score'],selective_score=result['score'],
            baseline_action=reference['action'],selective_action=result['action']))
    report['adoption']='Both flags remain off: bounded evidence cannot establish strength non-regression.'
    native.close();args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
