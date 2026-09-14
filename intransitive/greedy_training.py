"""Checkpoint-safe continuation for greedy-opponent modelling rollouts.

Only a complete iteration (including scheduled selection) is published. An
interruption rolls back to that commit; deterministic seeds replay pending work.
"""
import argparse
from collections import deque
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import signal
import tempfile
import time
import uuid

import torch
from .greedy_process import (SETTINGS, GameProcesses, make_net, seed_all, selection)


FORMAT = 1


def checkpoint(net, folder):
    path = Path(folder) / ('snapshot-' + uuid.uuid4().hex + '.pt')
    net.save_checkpoint(str(path.parent), path.name)
    return path


def replay_digest(replay):
    h = hashlib.sha256()
    for batch in replay:
        h.update(len(batch).to_bytes(8, 'little'))
        for blob in batch:
            h.update(len(blob).to_bytes(8, 'little'))
            h.update(blob)
    return h.hexdigest()


def validate(bundle):
    if bundle.get('format') != FORMAT:
        raise ValueError('Unknown continuation format')
    if bundle['iteration'] != bundle['replay_iteration']:
        raise ValueError('Checkpoint/replay iteration mismatch')
    if replay_digest(bundle['replay']) != bundle['replay_sha256']:
        raise ValueError('Checkpoint/replay digest mismatch')
    settings = bundle['settings']
    if len(bundle['replay']) != min(bundle['iteration'], settings['replay_iterations']):
        raise ValueError('Wrong number of replay generations')
    if any(len(b) > settings['replay_max_examples'] for b in bundle['replay']):
        raise ValueError('Replay exceeds configured bound')
    if not (bundle['started_epoch'] < bundle['deadline_epoch']):
        raise ValueError('Invalid absolute deadline')
    if not (0 <= bundle['best_iteration'] <= bundle['iteration']):
        raise ValueError('Invalid best iteration')
    expected = bundle['iteration'] if (bundle['iteration'] == 1 or
        bundle['iteration'] % settings['evaluation_interval'] == 0) else (
        max(1, bundle['iteration']//settings['evaluation_interval'] * settings['evaluation_interval']))
    if bundle['iteration'] == 0:
        expected = 0
    if bundle['selection_iteration'] != expected:
        raise ValueError('Selection is incomplete for committed iteration')
    for key in ('current', 'best'):
        net = make_net(settings)
        net.load_network(bundle[key])
    if bundle['optimizer_updates'] != bundle['current'].get('optimizer_updates', 0):
        raise ValueError('Optimizer update count mismatch')


def load_bundle(path):
    # Bundles/checkpoints are trusted local pickle artifacts, like existing .pt files.
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    validate(bundle)
    return bundle


def publish(path, bundle, boundary=lambda stage: None):
    validate(bundle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, pending = tempfile.mkstemp(prefix='.' + path.name, suffix='.pending', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(bundle, stream)
            stream.flush()
            os.fsync(stream.fileno())
        boundary('before_replace')
        os.replace(pending, path)
        boundary('after_replace')
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(pending):
            os.unlink(pending)


def initial_bundle(net, settings, baseline, started, deadline, folder):
    data = torch.load(checkpoint(net, folder), map_location='cpu', weights_only=False)
    return dict(format=FORMAT, settings=dict(settings), iteration=0, replay_iteration=0,
        replay=[], replay_sha256=replay_digest([]), current=data, best=data,
        optimizer_updates=net.optimizer_updates, best_score=baseline['score'],
        best_iteration=0, selection_iteration=0, selection=baseline,
        started_epoch=started, deadline_epoch=deadline)


def iteration(bundle, pool, folder, boundary=lambda stage: None):
    settings = bundle['settings']
    number = bundle['iteration'] + 1
    deadline = bundle['deadline_epoch']
    if time.time() >= deadline:
        raise TimeoutError('Original absolute deadline has passed')
    start = time.perf_counter()
    phases = {}
    net = make_net(settings)
    # AdamW.load_state_dict may reuse CPU tensors; isolate the committed
    # moments as well as weights before any in-place optimizer step.
    net.load_network(copy.deepcopy(bundle['current']))
    snapshot = checkpoint(net, folder)
    phases['checkpoint_distribution_seconds'] = time.perf_counter()-start
    games, phases['rollouts'] = pool.collect(snapshot,
        [settings['seed'] + number * 100 + i for i in range(settings['episodes_per_iteration'])],
        True, deadline=deadline)
    boundary('rollouts')
    transfer = time.perf_counter()
    batch = deque(maxlen=settings['replay_max_examples'])
    for examples, _ in games:
        batch.extend(examples)
    replay = deque(bundle['replay'], maxlen=settings['replay_iterations'])
    replay.append(list(batch))
    examples = [e for b in replay for e in b]
    phases['replay_ordering_seconds'] = time.perf_counter()-transfer
    seed_all(settings['seed'] + number)
    train_start = time.perf_counter()
    net.train(examples)
    if not all(torch.isfinite(p).all() for p in net.nnet.parameters()):
        raise RuntimeError('Nonfinite trained parameters')
    phases['optimization_seconds'] = time.perf_counter()-train_start
    boundary('optimization')
    save_start = time.perf_counter()
    candidate = checkpoint(net, folder)
    current = torch.load(candidate, map_location='cpu', weights_only=False)
    phases['candidate_checkpoint_seconds'] = time.perf_counter()-save_start
    result = dict(bundle, current=current, iteration=number, replay_iteration=number,
        replay=list(replay), replay_sha256=replay_digest(replay), optimizer_updates=net.optimizer_updates)
    evaluation_rows = []
    if number == 1 or number % settings['evaluation_interval'] == 0:
        evaluated, phases['selection'] = pool.collect(candidate,
            range(800000, 800000 + settings['evaluation_games']), False, deadline=deadline)
        evaluation_rows = [row for _, row in evaluated]
        score = selection(evaluation_rows)
        result.update(selection=score, selection_iteration=number)
        if score['score'] > bundle['best_score']:
            result.update(best=current, best_iteration=number, best_score=score['score'])
    boundary('selection')
    if time.time() >= deadline:
        raise TimeoutError('Original absolute deadline reached before commit')
    return result, dict(phases=phases, training_games=[row for _, row in games],
        selection_games=evaluation_rows, examples=len(examples),
        updates=net.optimizer_updates-bundle['optimizer_updates'])



def capture_legacy(source, destination):
    """Read a provably consistent legacy rollout boundary; never control its job.

    Legacy optimization writes replay before weights and selection writes best
    later still. Those phases cannot establish a full commit and are rejected.
    """
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError('Refusing to overwrite an existing continuation')
    before = json.loads((source/'status.json').read_text())
    number = before['completed_iterations']
    if before['phase'] != 'opponent_rollouts' or before['iteration'] != number+1:
        raise ValueError('Legacy snapshot requires an opponent_rollouts boundary; retry later')
    names = ('latest.pt','best.pt','replay.pkl','settings.json','progress.json',
             'training-games.jsonl','evaluations.jsonl')
    raw = {name:(source/name).read_bytes() for name in names}
    settings = json.loads(raw['settings.json'])
    after = json.loads((source/'status.json').read_text())
    if any(before[key] != after[key] for key in ('phase','iteration','completed_iterations',
                                                'started_epoch','deadline_epoch')):
        raise ValueError('Legacy run crossed a publication boundary; retry snapshot')
    # The legacy script replaces replay after logging the last rollout, before
    # changing phase to optimization. Reject both sides of that status window.
    if max(before.get('episodes_completed', 0), after.get('episodes_completed', 0)) >= settings['episodes_per_iteration']:
        raise ValueError('Legacy last rollout may be publishing replay; retry at the next iteration')
    current = torch.load(io.BytesIO(raw['latest.pt']),map_location='cpu',weights_only=False)
    best = torch.load(io.BytesIO(raw['best.pt']),map_location='cpu',weights_only=False)
    progress = json.loads(raw['progress.json'])
    if current['run_iteration'] != number or progress['iteration'] != number:
        raise ValueError('Legacy checkpoint/progress iteration mismatch')
    if progress['optimizer_updates'] != current['optimizer_updates']:
        raise ValueError('Legacy optimizer/progress mismatch')
    replay = [list(batch) for batch in pickle.loads(raw['replay.pkl'])]
    games = [json.loads(line) for line in raw['training-games.jsonl'].splitlines()]
    first = max(1, number-settings['replay_iterations']+1)
    if len(replay) != number-first+1:
        raise ValueError('Legacy replay generation count mismatch')
    for batch, generation in zip(replay,range(first,number+1)):
        rows = [row for row in games if row['iteration']==generation]
        seeds = [settings['seed']+generation*100+i for i in range(settings['episodes_per_iteration'])]
        if [row['seed'] for row in rows] != seeds or [row['model_side'] for row in rows] != [i%2 for i in range(len(seeds))]:
            raise ValueError('Legacy replay game quota, order or seeds mismatch')
        if len(batch) != min(settings['replay_max_examples'],sum(row['examples'] for row in rows)):
            raise ValueError('Legacy replay/game log mismatch')
    if sum(map(len,replay)) != progress['examples']:
        raise ValueError('Legacy replay/progress example count mismatch')
    evaluations = [json.loads(line) for line in raw['evaluations.jsonl'].splitlines()]
    latest = evaluations[-1]
    for data in (current,best):
        data.pop('full_model',None)
    bundle = dict(format=FORMAT,settings=settings,iteration=number,replay_iteration=number,
        replay=replay,replay_sha256=replay_digest(replay),current=current,best=best,
        optimizer_updates=current['optimizer_updates'],best_score=best['selection']['score'],
        best_iteration=best['run_iteration'],selection_iteration=latest['iteration'],
        selection={k:v for k,v in latest.items() if k not in ('iteration','promoted')},
        started_epoch=before['started_epoch'],deadline_epoch=before['deadline_epoch'])
    publish(destination,bundle)
    provenance = dict(source=str(source),captured_epoch=time.time(),iteration=number,
        sha256={name:hashlib.sha256(blob).hexdigest() for name,blob in raw.items()},
        status_before=before,status_after=after,replay_lengths=list(map(len,replay)))
    destination.with_suffix('.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--initialize-from', type=Path)
    parser.add_argument('--capture-legacy-run', type=Path,
                        help='Capture a consistent legacy boundary and exit without starting training')
    parser.add_argument('--duration', type=float, default=86400,
                        help='Used only for a new run; resume preserves stored deadline')
    parser.add_argument('--workers', type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args()
    if args.capture_legacy_run:
        if args.initialize_from:
            parser.error('capture and initialize are mutually exclusive')
        capture_legacy(args.capture_legacy_run, args.state)
        return
    def interrupted(*_):
        raise KeyboardInterrupt
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
        signal.signal(sig, interrupted)
    with tempfile.TemporaryDirectory(prefix='greedy-continuation-') as folder:
        if args.initialize_from:
            if args.state.exists():
                raise ValueError('Refusing to overwrite an existing continuation')
            started = time.time()
            deadline = started + args.duration
            if args.duration <= 0:
                raise ValueError('duration must be positive')
            signal.setitimer(signal.ITIMER_REAL, args.duration)
            net = make_net()
            net.load_checkpoint(str(args.initialize_from.parent), args.initialize_from.name)
            snapshot = checkpoint(net, folder)
            with GameProcesses(args.workers, snapshot=snapshot) as pool:
                rows, _ = pool.collect(snapshot, range(800000, 800020), False, deadline=deadline)
                bundle = initial_bundle(net, SETTINGS, selection([r for _, r in rows]),
                                        started, deadline, folder)
                publish(args.state, bundle)
        bundle = load_bundle(args.state)
        remaining = bundle['deadline_epoch']-time.time()
        if remaining <= 0:
            print('Original absolute deadline has passed; no work started.')
            return
        signal.setitimer(signal.ITIMER_REAL, remaining)
        net = make_net(bundle['settings'])
        net.load_network(bundle['current'])
        snapshot = checkpoint(net, folder)
        try:
            with GameProcesses(args.workers, snapshot=snapshot, settings=bundle['settings']) as pool:
                while time.time() < bundle['deadline_epoch']:
                    # Clean immutable snapshots after the pool has completed all work.
                    with tempfile.TemporaryDirectory(dir=folder) as generation:
                        result, report = iteration(bundle, pool, generation)
                        publish(args.state, result)
                    bundle = result
                    print(json.dumps(dict(iteration=bundle['iteration'],
                        optimizer_updates=bundle['optimizer_updates'], best_score=bundle['best_score'],
                        phases=report['phases'])), flush=True)
        except (KeyboardInterrupt, TimeoutError):
            print('Stopped; continuation contains the last fully committed iteration.')
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
