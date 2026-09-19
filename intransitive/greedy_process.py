"""Independent, spawn-safe modelling rollouts against the reference greedy player.

The defaults and episode semantics reproduce the frozen 20260914 continuation.
Official-play rules are deliberately outside this module.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import random
import resource
import signal
import traceback
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from MCTS import MCTS
from .IntransitiveGame import IntransitiveGame
from .IntransitivePlayers import GreedyPlayer, ReferenceGreedyPlayer
from .NNet import NNetWrapper

SETTINGS = dict(duration_seconds=86400, seed=2026091422, opponent='ReferenceGreedyPlayer',
    episodes_per_iteration=8, replay_iterations=2, replay_max_examples=57600,
    evaluation_games=20, evaluation_interval=5, numMCTSSims=32,
    prob_fullMCTS=1., ratio_fullMCTS=1, universes=0, cpuct=1.25, fpu=0.,
    forced_playouts=False, no_mem_optim=False, dirichletAlpha=0.,
    nn_version=2, dropout=0., epochs=1, batch_size=64, learn_rate=.0001,
    no_compression=False, q_weight=0., persist_optimizer=True,
    policy_targets='MCTS visits on model turns only',
    value_targets='actual model-vs-greedy rollout outcome; no search-Q mixing',
    selection='strictly higher win-plus-half-draw score on 20 fixed selection seeds',
    model_draws='threefold or 80 noncapture plies; modelling rollouts only',
    official_play_rules_changed=False)
NN_KEYS = ('nn_version', 'dropout', 'epochs', 'batch_size', 'learn_rate',
           'no_compression', 'q_weight', 'persist_optimizer')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_net(settings=SETTINGS):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return NNetWrapper(IntransitiveGame(), {k: settings[k] for k in NN_KEYS})


def make_opponent(game, seed, settings):
    name = settings.get('opponent', 'ReferenceGreedyPlayer')
    if name == 'ReferenceGreedyPlayer':
        return ReferenceGreedyPlayer(game, seed=seed + 100)
    if name == 'GreedyPlayer':
        return GreedyPlayer(game)
    if name == 'FlybrainPlayer':
        from .flybrain import FlybrainPlayer
        return FlybrainPlayer(game, bank_path=settings.get('flybrain_bank'), seed=seed + 100)
    if name == 'AlphaBetaPlayer':
        from .heuristics import AlphaBetaPlayer, SearchConfig
        if settings.get('opponent_backend'):
            from .rust_teacher.adapter import RustAlphaBetaPlayer
            return RustAlphaBetaPlayer(game, SearchConfig(**settings['opponent_search']),
                                       settings['opponent_backend'])
        return AlphaBetaPlayer(game, SearchConfig(**settings['opponent_search']))
    raise ValueError(f'Unknown training opponent: {name}')


def game_tasks(snapshot, seeds, training, settings, *, evaluation_stage=None):
    opponents = settings.get('worker_opponents') if training else None
    evaluation = settings.get('evaluation_opponents') if not training else None
    schedule = []
    if evaluation:
        for block in evaluation:
            count = block.get('games')
            if type(count) is not int or count <= 0 or count % 2:
                raise ValueError('Each evaluation opponent needs a positive even game quota')
            if block.get('opponent') not in ('FlybrainPlayer','AlphaBetaPlayer',
                                              'GreedyPlayer','ReferenceGreedyPlayer'):
                raise ValueError('Unknown evaluation opponent')
            overrides = {k:v for k,v in block.items() if k != 'games'}
            if set(overrides) - {'opponent','opponent_search','numMCTSSims'}:
                raise ValueError('Evaluation blocks may only change the opponent/search or MCTS simulations')
            if 'numMCTSSims' in overrides and (type(overrides['numMCTSSims']) is not int or
                                               overrides['numMCTSSims'] < 2):
                raise ValueError('Evaluation MCTS simulations must be an integer of at least two')
            selected = dict(settings, **overrides)
            if selected['opponent'] != 'AlphaBetaPlayer':
                selected.pop('opponent_search', None)
            schedule.extend([selected] * count)
        if len(schedule) != settings['evaluation_games']:
            raise ValueError('Evaluation quota does not match the configured suite')
        if evaluation_stage is not None:
            if type(evaluation_stage) is not int or not 0 <= evaluation_stage < len(evaluation):
                raise ValueError('Invalid evaluation stage')
            start = sum(b['games'] for b in evaluation[:evaluation_stage])
            schedule = schedule[start:start + evaluation[evaluation_stage]['games']]
        if len(schedule) != len(seeds):
            raise ValueError('Evaluation quota does not match the requested stage')
    elif evaluation_stage is not None:
        raise ValueError('Evaluation stages require an evaluation-only opponent schedule')
    if opponents and len(seeds) % (2 * len(opponents)):
        raise ValueError('Mixed games require both colours for every opponent')
    tasks = []
    for index, seed in enumerate(seeds):
        selected = settings
        side = index % 2
        if schedule:
            selected = schedule[index]
        elif opponents:
            selected = dict(settings, **opponents[index % len(opponents)])
            side = (index // len(opponents)) % 2
        tasks.append((index, str(Path(snapshot).resolve()), selected, seed, side, training))
    return tasks


def worker_status(settings, **data):
    path = settings.get('_status_path')
    if path:
        path = Path(path)
        pending = path.with_suffix('.pending')
        pending.write_text(json.dumps(dict(data, pid=os.getpid(), updated_epoch=time.time())))
        os.replace(pending, path)


def episode(net, seed, model_side, training, settings=SETTINGS):
    seed_all(seed)
    game = IntransitiveGame(modelling_draws=True)
    opponent = make_opponent(game, seed, settings)
    try:
        return _episode(net, seed, model_side, training, settings, game, opponent)
    finally:
        if hasattr(opponent, 'close'):
            opponent.close()


def _episode(net, seed, model_side, training, settings, game, opponent):
    seed_all(seed)
    rng = np.random.default_rng(seed)
    search = MCTS(game, net, argparse.Namespace(**settings))
    search.rng = np.random.default_rng(seed)
    board, player = game.getInitBoard(), 0
    trajectory, actions = [], []
    search_depths = []
    trace = hashlib.sha256(board.tobytes())
    while True:
        result = game.getGameEnded(board, player)
        if result.any():
            reason = game.board.get_terminal_reason()
            break
        canonical = game.getCanonicalForm(board, player)
        valid = game.getValidMoves(canonical, 0)
        worker_status(settings, phase='model_move' if player == model_side else 'opponent_move',
                      seed=seed, model_side=model_side, ply=len(actions)+1,
                      opponent=settings.get('opponent'), opponent_search=settings.get('opponent_search'),
                      numMCTSSims=settings['numMCTSSims'], training=training,
                      last_search=search_depths[-1] if search_depths else None)
        if player == model_side:
            pi, q, _ = search.getActionProb(canonical, temp=1 if training else 0,
                                           force_full_search=True)
            pi = np.asarray(pi, dtype=np.float32)
            assert np.isfinite(pi).all() and np.isclose(pi.sum(), 1.)
            assert not np.any(pi[~valid])
            action = int(rng.choice(len(pi), p=pi / pi.sum())) if training else int(np.argmax(pi))
            if training:
                for state, policy, mask in game.getSymmetries(canonical, pi, valid):
                    trajectory.append((state, policy, mask))
        else:
            action = opponent.play(canonical)
            if settings.get('opponent') == 'AlphaBetaPlayer':
                result = opponent.last_result
                target = settings['opponent_search']['max_depth']
                if result.completed_depth < target and result.stop_reason != 'proven_result':
                    raise RuntimeError(f'Minimax did not complete requested depth {target}: {result.stop_reason}')
                search_depths.append(dict(ply=len(actions)+1, completed_depth=result.completed_depth,
                    stop_reason=result.stop_reason, seconds=result.elapsed))
        assert valid[action], (len(actions), action)
        actions.append(action)
        parent = board
        before = board.tobytes()
        board, player = game.getNextState(board, player, action)
        assert parent.tobytes() == before
        trace.update(action.to_bytes(2, "little"))
        trace.update(board.tobytes())
        if len(actions) > 2000:
            raise RuntimeError('Unexpected unbounded modelling rollout')
    relative = np.roll(result, -model_side).astype(np.float32)
    examples = [zlib.compress(pickle.dumps((b, pi, relative, mask,
                 np.zeros(2, dtype=np.float32))), level=1) for b, pi, mask in trajectory]
    reward = float(result[model_side])
    row = dict(seed=seed, model_side=model_side, plies=len(actions), actions=actions,
               outcome='win' if reward == 1 else 'loss' if reward == -1 else 'model_draw',
               reason=reason, examples=len(examples), modelling_only=True,
               trajectory_sha256=trace.hexdigest(),
               replay_sha256=hashlib.sha256(b"".join(examples)).hexdigest())
    row.update(opponent=settings.get('opponent', 'ReferenceGreedyPlayer'),
               numMCTSSims=settings['numMCTSSims'],
               opponent_search=settings.get('opponent_search'), opponent_searches=search_depths)
    if settings.get('opponent') == 'AlphaBetaPlayer' and settings.get('opponent_backend'):
        row['opponent_backend'] = settings['opponent_backend']
    return examples, row


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


_NET = None
_SNAPSHOT = None


def _load(snapshot, settings):
    global _NET, _SNAPSHOT
    if _SNAPSHOT != snapshot:
        _NET = make_net(settings)
        _NET.load_checkpoint(str(Path(snapshot).parent), Path(snapshot).name)
        _NET.switch_target('inference')
        _SNAPSHOT = snapshot
    return _NET


def _job(task):
    index, snapshot, settings, seed, side, training = task
    if settings.get('worker_status_dir'):
        settings = dict(settings, _status_path=str(Path(settings['worker_status_dir']) / f'worker-{os.getpid()}.json'))
    start, cpu = time.perf_counter(), cpu_seconds()
    net = _load(snapshot, settings)
    loaded = time.perf_counter()
    examples, row = episode(net, seed, side, training, settings)
    row.update(index=index, pid=os.getpid(), worker_seconds=time.perf_counter()-start,
               snapshot_load_seconds=loaded-start, cpu_seconds=cpu_seconds()-cpu)
    worker_status(settings, phase='game_completed', **row)
    return examples, row


def _worker_loop(snapshot, settings, connection, job):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    if hasattr(signal, 'pthread_sigmask'):
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM, signal.SIGALRM})
    start = time.perf_counter()
    try:
        net = _load(snapshot, settings)
        game = IntransitiveGame()
        board = game.getInitBoard()
        MCTS(game, net, argparse.Namespace(**settings)).getActionProb(board, force_full_search=True)
        ReferenceGreedyPlayer(game, seed=0).play(board)
        connection.send(('ready', (os.getpid(), time.perf_counter()-start, None)))
        while True:
            task = connection.recv()
            connection.send(('result', job(task)))
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def validate_worker_count(workers):
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError('workers must be an integer between 1 and 64')
    return workers


class GameProcesses:
    """Independent workers with private pipes, ordered replay and bounded cleanup.

    Snapshot paths must be immutable and unique for each weight generation.
    Each worker has at most one outstanding episode. Any failure discards the
    batch; no shared multiprocessing queue locks survive a killed worker.
    """
    def __init__(self, workers=1, *, snapshot, settings=SETTINGS, timeout=600, _task=_job):
        validate_worker_count(workers)
        self.settings = dict(settings)
        opponents = self.settings.get('worker_opponents')
        if opponents and len(opponents) != workers:
            raise ValueError('One configured opponent is required per worker')
        self.pool = True
        self.pids = []
        self.connections = []
        self.initial_workers = []
        ctx = mp.get_context('spawn')
        start = time.perf_counter()
        try:
            for _ in range(workers):
                parent, child = ctx.Pipe()
                process = ctx.Process(target=_worker_loop, args=(
                    str(Path(snapshot).resolve()), self.settings, child, _task))
                # Defer cancellation across spawn/registration so cleanup always
                # owns the new PID. Spawned workers restore their signal mask.
                blocked = {signal.SIGINT, signal.SIGTERM, signal.SIGALRM}
                previous = signal.pthread_sigmask(signal.SIG_BLOCK, blocked) if hasattr(signal, 'pthread_sigmask') else None
                try:
                    process.start()
                    self.connections.append(parent)
                    self.initial_workers.append(process)
                finally:
                    child.close()
                    if previous is not None:
                        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
            pending = set(range(workers))
            rows = []
            while pending:
                self._check_workers()
                if time.perf_counter()-start > timeout:
                    raise TimeoutError('Worker startup timed out')
                for index in list(pending):
                    if self.connections[index].poll():
                        kind, row = self.connections[index].recv()
                        if kind != 'ready':
                            raise RuntimeError(f'Worker startup failed: {row}')
                        rows.append(row)
                        pending.remove(index)
                time.sleep(.01)
            self.pids = [p.pid for p in self.initial_workers]
            self.startup = dict(seconds=time.perf_counter()-start, workers=rows)
        except BaseException:
            self.close()
            raise

    def _check_workers(self):
        if any(p.exitcode is not None for p in self.initial_workers):
            raise RuntimeError('Game worker exited unexpectedly')

    def collect(self, snapshot, seeds, training, *, deadline=None, evaluation_stage=None):
        if self.pool is None:
            raise RuntimeError('Pool is closed')
        seeds = list(seeds)
        if not seeds or len(seeds) % 2:
            raise ValueError('Game quota must be positive and even for colour balance')
        start = time.perf_counter()
        if not training and self.settings.get('evaluation_gate') and evaluation_stage is None:
            raise ValueError('Gated selection must dispatch opponent stages separately')
        tasks = game_tasks(snapshot, seeds, training, self.settings, evaluation_stage=evaluation_stage)
        ordered = [None] * len(tasks)
        active = {}
        next_task = completed = 0
        affinity = bool(training and self.settings.get('worker_opponents'))
        queues = [list(range(i, len(tasks), len(self.connections))) for i in range(len(self.connections))]
        send_seconds = receive_seconds = 0.
        try:
            while completed < len(tasks):
                self._check_workers()
                if deadline is not None and time.time() >= deadline:
                    raise TimeoutError('Absolute deadline reached during game batch')
                for worker, connection in enumerate(self.connections):
                    if worker in active and connection.poll():
                        received = time.perf_counter()
                        kind, result = connection.recv()
                        receive_seconds += time.perf_counter()-received
                        if kind != 'result':
                            raise RuntimeError(f'Game worker failed: {result}')
                        index = active.pop(worker)
                        if result[1]['index'] != index:
                            raise RuntimeError('Worker returned wrong episode index')
                        ordered[index] = result
                        completed += 1
                    task_index = (queues[worker][0] if queues[worker] else None) if affinity else (
                        next_task if next_task < len(tasks) else None)
                    if worker not in active and task_index is not None:
                        sent = time.perf_counter()
                        connection.send(tasks[task_index])
                        send_seconds += time.perf_counter()-sent
                        active[worker] = task_index
                        if affinity:
                            queues[worker].pop(0)
                        else:
                            next_task += 1
                if completed < len(tasks):
                    time.sleep(.01)
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError('Absolute deadline reached during game batch')
            return ordered, dict(seconds=time.perf_counter()-start,
                result_bytes=sum(len(pickle.dumps(r)) for r in ordered),
                parent_send_seconds=send_seconds, parent_receive_decode_seconds=receive_seconds,
                games=len(ordered), worker_seconds=sum(r[1]['worker_seconds'] for r in ordered),
                worker_cpu_seconds=sum(r[1]['cpu_seconds'] for r in ordered))
        except BaseException:
            self.close()
            raise

    def iter_results(self, snapshot, seeds, training, *, deadline=None):
        """Yield completed tasks immediately, replenishing that worker first.

        Unlike colour-balanced match collection, a recovered label queue may
        contain an odd number of pending tasks. Callers persist their seed queue.
        """
        if self.pool is None:
            raise RuntimeError('Pool is closed')
        tasks = game_tasks(snapshot, list(seeds), training, self.settings)
        if not tasks:
            return
        active, next_task, completed = {}, 0, 0
        affinity = bool(training and self.settings.get('worker_opponents'))
        queues = [list(range(i,len(tasks),len(self.connections))) for i in range(len(self.connections))]
        started = time.perf_counter()
        def send(worker):
            nonlocal next_task
            index = (queues[worker][0] if queues[worker] else None) if affinity else (
                next_task if next_task < len(tasks) else None)
            if index is None:
                return
            self.connections[worker].send(tasks[index])
            active[worker] = index
            if affinity:
                queues[worker].pop(0)
            else:
                next_task += 1
        try:
            for worker in range(len(self.connections)):
                send(worker)
            while completed < len(tasks):
                self._check_workers()
                if deadline is not None and time.time() >= deadline:
                    raise TimeoutError('Absolute deadline reached during streamed labels')
                received = False
                for worker, connection in enumerate(self.connections):
                    if worker not in active or not connection.poll():
                        continue
                    kind, result = connection.recv()
                    if kind != 'result':
                        raise RuntimeError(f'Game worker failed: {result}')
                    if result[1]['index'] != active.pop(worker):
                        raise RuntimeError('Worker returned wrong task index')
                    completed += 1
                    received = True
                    send(worker)
                    yield result, dict(seconds=result[1].get('worker_seconds',time.perf_counter()-started),
                                       completed=completed,tasks=len(tasks))
                if not received:
                    time.sleep(.01)
        except GeneratorExit:
            if completed < len(tasks):
                self.close()
            raise
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.pool is not None:
            backend = self.settings.get('teacher_backend') or self.settings.get('opponent_backend')
            native_pids = []
            if backend:
                from .rust_teacher.adapter import native_children, reap_native
                native_pids = native_children([p.pid for p in self.initial_workers], backend)
            for process in self.initial_workers:
                if process.is_alive():
                    process.terminate()
            for process in self.initial_workers:
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
                if process.is_alive():
                    raise RuntimeError(f'Could not reap game worker {process.pid}')
            for connection in self.connections:
                connection.close()
            if backend:
                reap_native(native_pids, backend)
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def selection(rows):
    wins = sum(r['outcome'] == 'win' for r in rows)
    losses = sum(r['outcome'] == 'loss' for r in rows)
    draws = len(rows) - wins - losses
    result = dict(wins=wins, losses=losses, model_draws=draws,
                  score=(wins + .5 * draws) / len(rows), games=len(rows))
    names = {r.get('opponent') for r in rows}
    if len(names) > 1:
        result['by_opponent'] = {name:selection([r for r in rows if r.get('opponent') == name])
                                 for name in sorted(names, key=str)}
    return result
