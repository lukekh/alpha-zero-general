"""Opt-in consensus adjudication of saved, time/ply-capped match prefixes.

This is a separate analysis: original journals, official outcomes, optimizer
fitness and acceptance eligibility are never rewritten.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
import hashlib
import math
from pathlib import Path
import time

from ...IntransitiveGame import IntransitiveGame
from ...heuristics.budget import Budget, BudgetExpired
from ...heuristics.evaluation import Evaluator
from ...heuristics.tuning import Genome
from ...tournament.runner import atomic_json, replay, validate_manifest
from ...tournament.spec import backend_version, digest


def consensus(scores):
    """Both scores are player-zero minus player-one; exact ties abstain."""
    if len(scores) != 2 or any(s is None or not math.isfinite(s) for s in scores):
        return None
    if all(s > 0 for s in scores):
        return 0
    if all(s < 0 for s in scores):
        return 1
    return None


def adjudicate(spec, row, *, seconds=2., nodes=5_000_000):
    if (row['manifest_sha256'] != spec['sha256'] or row['task'] not in spec['tasks']
            or row['colours'] != row['task']['colours']):
        raise ValueError('Journal/manifest mismatch')
    start = next(p for p in spec['positions'] if p['sha256'] == row['task']['position'])
    state = replay(start, row)
    result = dict(match_key=digest(dict(manifest=spec['sha256'], task=row['task']['id'])),
                  original_status=row['status'], official_winner=row['winner'],
                  final_state_sha256=row['final_state_sha256'], scores=[],
                  adjudicated_winner=None, effective_winner=row['winner'],
                  outcome='official' if row['status'] == 'win' else 'excluded')
    # A cancelled overall run may leave a valid played prefix. Empty prefixes,
    # infrastructure failures and incomplete requested searches never qualify.
    if row['status'] not in ('unfinished', 'cancelled') or not row['moves']:
        return result
    result['outcome'] = 'inconclusive'
    by_id = {c['sha256']: c for c in spec['candidates']}
    for identity in row['colours']:
        item = by_id[identity]
        if item['backend'] != 'python':
            raise ValueError('This analysis currently supports Python genomes only')
        genome = Genome.from_json(json.dumps(item['genome']))
        config = replace(genome.to_config(), proof_depth=0, proof_nodes=0)
        budget = Budget(nodes, seconds)
        try:
            # Static weighted board evaluation, with no tree/proof adjudicator.
            value = Evaluator(IntransitiveGame(modelling_draws=False), config).score(
                state, 0, budget, proof={'status': 'unknown'})
            budget.check()
            score = dict(candidate=identity, config_hash=genome.config_hash,
                         score=float(value), status='complete')
        except BudgetExpired as exc:
            score = dict(candidate=identity, config_hash=genome.config_hash,
                         score=None, status=exc.reason)
        score.update(work=budget.work, seconds=budget.clock()-budget.start)
        result['scores'].append(score)
    winner = consensus([s['score'] for s in result['scores']])
    if winner is not None:
        result.update(adjudicated_winner=winner, effective_winner=winner, outcome='consensus')
    return result


def analyze(folder, *, seconds=2., nodes=5_000_000, deadline=None):
    folder = Path(folder)
    if (folder/'checkpoint.json').exists():
        checkpoint = json.loads((folder/'checkpoint.json').read_text())
        paths = [folder/e['record'] for e in checkpoint['ledger'].values()]
    else:
        paths = sorted((folder/'matches').glob('*.json'))
    rows=[];board=defaultdict(Counter);manifests={}
    for path in paths:
        if deadline is not None and time.monotonic() >= deadline:
            break
        spec_path = path.parent/'manifest.json' if path.name == 'game.json' else folder/'manifest.json'
        if spec_path not in manifests:
            spec=json.loads(spec_path.read_text());validate_manifest(spec);manifests[spec_path]=spec
        spec=manifests[spec_path];row=json.loads(path.read_text())
        allowance=seconds if deadline is None else max(.001,min(seconds,(deadline-time.monotonic())/2))
        result=adjudicate(spec,row,seconds=allowance,nodes=nodes);rows.append(result)
        by_id={c['sha256']:c for c in spec['candidates']}
        for side,identity in enumerate(row['colours']):
            stats=board[Genome.from_json(json.dumps(by_id[identity]['genome'])).config_hash];stats['scheduled']+=1
            if result['effective_winner'] is not None:
                won=result['effective_winner']==side
                stats['wins' if won else 'losses']+=1
                stats[('official_' if result['outcome']=='official' else 'consensus_')+('wins' if won else 'losses')]+=1
            else:stats['inconclusive' if result['outcome']=='inconclusive' else 'excluded']+=1
    ranking=[]
    for identity,stats in board.items():
        decided=stats['wins']+stats['losses'];ranking.append(dict(config_hash=identity,**dict(stats),
            decided=decided,win_fraction_decided=stats['wins']/decided if decided else None,
            wins_over_scheduled=stats['wins']/stats['scheduled']))
    ranking.sort(key=lambda r:(-(r['win_fraction_decided'] if r['win_fraction_decided'] is not None else -1.),-r['decided'],r['config_hash']))
    return dict(folder=str(folder),policy='Both competing static weighted scores from player-zero perspective must have the same strict nonzero sign. Proof/tree search disabled. Official outcomes preserved; unfinished and cancelled nonempty legal prefixes eligible. Other failures excluded. Inconclusive games excluded from decided fraction; also show conservative wins/scheduled.',
                seconds_per_evaluator=seconds,nodes_per_evaluator=nodes,
                completed=len(rows)==len(paths),records=len(rows),scheduled_records=len(paths),
                outcomes=dict(Counter(r['outcome'] for r in rows)),ranking=ranking,matches=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders',type=Path,nargs='+');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-seconds',type=float,default=120.)
    args=parser.parse_args()
    if not math.isfinite(args.max_seconds) or args.max_seconds<=0:parser.error('Positive finite time cap required')
    start=time.monotonic();deadline=start+args.max_seconds;results=[]
    for folder in args.folders:
        if time.monotonic()>=deadline:break
        results.append(analyze(folder,deadline=deadline))
    atomic_json(args.output,dict(schema='intransitive-consensus-analysis-v1',wall_seconds=time.monotonic()-start,
        requested_folders=[str(p) for p in args.folders],runs=results,
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),backend_version=backend_version()))
    print(json.dumps(dict(output=str(args.output),runs=len(results),records=sum(r['records'] for r in results),wall_seconds=time.monotonic()-start)))


if __name__=='__main__':main()
