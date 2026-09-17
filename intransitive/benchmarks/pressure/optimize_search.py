"""Isolate search optimisations without changing the 7x7 heuristic."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
import time

from intransitive.benchmarks.pressure.reproduce import opening_state,write
from intransitive.heuristics import AlphaBetaPlayer,SearchConfig
from intransitive.IntransitiveDisplay import move_to_str
from intransitive.tests.test_tactics import CASES
from intransitive.tests.tactical_oracle import certify,load_case


def variants(base):
    result = dict(baseline=base,
        pvs=replace(base,pvs_enabled=True),
        aspiration=replace(base,aspiration_enabled=True),
        windows=replace(base,pvs_enabled=True,aspiration_enabled=True),
        compiled=replace(base,compiled_ordering_enabled=True),
        ordering=replace(base,ordering_enabled=True),
        native_ordering=replace(base,ordering_enabled=True,compiled_ordering_enabled=True),
        table50k=replace(base,table_entries=50000,depth_replacement_enabled=True),
        table100k=replace(base,table_entries=100000,depth_replacement_enabled=True),
        cache=replace(base,pressure_cache_entries=8192),
        all=replace(base,pvs_enabled=True,aspiration_enabled=True,ordering_enabled=True,
            compiled_ordering_enabled=True,table_entries=50000,depth_replacement_enabled=True,
            pressure_cache_entries=8192))
    result['all_no_cache'] = replace(result['all'],pressure_cache_entries=0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variants',nargs='+')
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--seconds',type=float,default=120.)
    parser.add_argument('--position',choices=('opening','midgame','late_game'),default='opening')
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        parser.error('Use a fresh output path and positive repeats')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if hasattr(os,'nice'):
        os.nice(10)
    base = SearchConfig(compiled_ordering_enabled=False,
                        pressure_enabled=True,pressure_radius=3,pressure_weight=10.,
                        max_depth=5,node_limit=1_000_000_000,time_limit=args.seconds)
    configs = variants(base)
    if args.variants:
        configs = {name:configs[name] for name in args.variants}
    state = opening_state()
    if args.position != 'opening':
        from intransitive.tests.test_game_blunders import DATA
        from intransitive.record import load_record
        state = load_record(DATA['record_pgn']).states[35 if args.position == 'midgame' else 61]
    start = time.perf_counter()
    for config in configs.values():
        AlphaBetaPlayer(config=config)._prepare()
    root = Path(__file__).resolve().parents[2]
    report = dict(platform=platform.platform(),position=args.position,
        state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),startup_seconds=time.perf_counter()-start,
        configurations={name:asdict(config) for name,config in configs.items()},searches=[],tactics=[],
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted((root/'heuristics').glob('*.py'))},
        notes=['One known position: performance screen, not general strength evidence.',
               'Every variant has identical evaluation weights, proof horizon and modelling draw rules.',
               'Exact scores must agree when depth five completes; tied actions/PVs can differ.',
               'Live training/generation left running; low-priority benchmark subject to contention.',
               'Compilation covers ranking/sorting, not the recursive Python controller.',
               'table_bytes is a shallow table estimate, not full process/native/cache memory.'])
    names = list(configs)
    expected = None
    for trial in range(args.repeats):
        order = names[trial%len(names):]+names[:trial%len(names)]
        for name in order:
            result = AlphaBetaPlayer(config=configs[name]).analyze(state)
            complete = result.completed_depth == 5
            if complete:
                if expected is None:
                    expected = result.score
                assert result.score == expected,(name,result.score,expected)
            row = dict(trial=trial,variant=name,seconds=result.elapsed,score=result.score,
                move=move_to_str(result.action),completed_depth=result.completed_depth,
                stop_reason=result.stop_reason,nodes=result.nodes,work=result.work,
                table_bytes=result.table_bytes,tt_hits=result.tt_hits,
                pvs_probes=result.pvs_probes,pvs_researches=result.pvs_researches,
                aspiration_researches=result.aspiration_researches,
                pressure_calls=result.module_calls.get('local_pressure',0),
                pressure_cache_hits=result.module_calls.get('pressure_cache_hit',0),
                pressure_seconds=result.module_seconds.get('local_pressure',0),
                pv=[move_to_str(a) for a in result.pv])
            report['searches'].append(row)
            write(args.output,report)
            print(json.dumps(row),flush=True)
    fixtures = []
    for case in CASES:
        certify(case)
        reference,expected_move = load_case(case)
        fixtures.append((case,reference.storage(),expected_move))
    for name,config in configs.items():
        rows = []
        for case,state,expected_move in fixtures:
            result = AlphaBetaPlayer(config=replace(config,max_depth=3)).analyze(state)
            complete = result.completed_depth >= 3 or result.stop_reason == 'proven_result'
            rows.append(dict(id=case['id'],passed=bool(complete and result.action==expected_move),
                selected=move_to_str(result.action),expected=case['expected'],
                completed_depth=result.completed_depth,stop_reason=result.stop_reason))
        report['tactics'].append(dict(variant=name,passed=sum(r['passed'] for r in rows),total=len(rows),results=rows))
        write(args.output,report)
    report['summary'] = {name:dict(median_seconds=median(r['seconds'] for r in report['searches'] if r['variant']==name),
        all_complete=all(r['completed_depth']==5 for r in report['searches'] if r['variant']==name)) for name in names}
    write(args.output,report)
    print(json.dumps(dict(summary=report['summary'],tactics=[{k:v for k,v in r.items() if k!='results'} for r in report['tactics']])),flush=True)


if __name__=='__main__':
    main()
