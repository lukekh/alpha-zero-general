"""Create isolated experimental policy-only shards with native search labels."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

from . import BINARY, RustTeacher
from ..IntransitiveGame import IntransitiveGame
from ..supervised_minimax import (generate_position, orbit_key, split_for,
    save_shard, write_json, valid_teacher_record)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--positions',type=int,default=100)
    parser.add_argument('--depth',type=int,default=6)
    parser.add_argument('--seconds',type=float,default=60.)
    parser.add_argument('--weight',type=float,default=10.)
    parser.add_argument('--seed',type=int,default=2026091700)
    args=parser.parse_args()
    if args.output.exists() or args.positions<1 or not 5<=args.depth<=32 or args.seconds<=0:
        parser.error('Use a fresh output directory, positive counts/seconds and depth 5..32')
    args.output.mkdir(parents=True)
    os.nice(10)
    profile=dict(implementation='experimental-rust-teacher-v1',depth=args.depth,
        count_weight=100.,advantage_weight=25.,pressure_radius=3,pressure_weight=args.weight,
        proof_depth=2,proof_nodes=64,seconds=args.seconds,
        binary_sha256=hashlib.sha256(BINARY.read_bytes()).hexdigest())
    manifest=dict(profile=profile,target_positions=args.positions,unique_positions=0,
        augmented_examples=0,shards=[],rejected={},stage_counts={},
        isolation='Experimental only; not imported by live trainer or million-position catalog',
        expansion='12 lazy symmetries via supervised_minimax.SymmetryExamples',
        sampling='Existing Python reachable-position sampler; no ablations/promoted roots',
        started_epoch=time.time())
    seen=set(); pending=[]; rejected=Counter(); stages=Counter()
    label_seconds=0.; sampling_seconds=0.

    def commit():
        if pending:
            name=f'shard-{len(manifest["shards"]):06d}.pkl.gz'
            save_shard(args.output/name,pending)
            manifest['shards'].append(dict(path=name,positions=len(pending),
                sha256=hashlib.sha256((args.output/name).read_bytes()).hexdigest()))
            pending.clear()
        manifest.update(unique_positions=len(seen),augmented_examples=12*len(seen),
            rejected=dict(rejected),stage_counts=dict(stages),label_seconds=label_seconds,
            sampling_seconds=sampling_seconds,updated_epoch=time.time())
        write_json(args.output/'manifest.json',manifest)

    commit()
    try:
        with RustTeacher() as rust:
            for index in range(10*args.positions):
                if len(seen)>=args.positions:
                    break
                stage=('opening','midgame','endgame')[len(seen)%3]
                begin=time.perf_counter()
                state, provenance=generate_position(args.seed+index,stage)
                sampling_seconds+=time.perf_counter()-begin
                begin=time.perf_counter()
                label=rust.analyze(state,depth=args.depth,seconds=args.seconds,weight=args.weight)
                label_seconds+=time.perf_counter()-begin
                if not label['complete']:
                    rejected[label['stop_reason']]+=1
                    commit()
                    continue
                game=IntransitiveGame()
                legal=game.getValidMoves(state,0)
                assert legal[label['action']]
                key=orbit_key(state,label['action'],legal)
                if key in seen:
                    rejected['duplicate_orbit']+=1
                    continue
                record=dict(state=state,legal=legal,action=label['action'],orbit=key,
                    family=key,split=split_for(key),provenance=dict(provenance,source='synthetic_root'),
                    teacher_depth=label['completed_depth'],teacher_target_depth=args.depth,
                    teacher_reason=label['stop_reason'],teacher_seconds=label['seconds'],
                    teacher_score=label['score'],teacher_nodes=label['nodes'],teacher_profile=profile)
                assert valid_teacher_record(record)
                seen.add(key);pending.append(record);stages[stage]+=1
                if len(pending)>=16:
                    commit()
                    print(json.dumps(dict(positions=len(seen),label_seconds=label_seconds)),flush=True)
        manifest['state']='complete' if len(seen)==args.positions else 'partial'
    except BaseException:
        manifest['state']='interrupted_or_failed'
        raise
    finally:
        commit()
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':
    main()
