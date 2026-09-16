"""Bounded candidate processes and fsync'd, replay-verified per-game journals."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import resource
import signal
import sys
import threading
import time
import traceback

import numpy as np

from ..IntransitiveGame import IntransitiveGame
from ..heuristics.search import AlphaBetaPlayer
from ..record import state_hash
from .spec import digest, effective_config, manifest, unpack

FINAL = {'win', 'unfinished', 'crash', 'illegal_move', 'infrastructure_timeout', 'depth_incomplete'}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def peak_rss_bytes():
    size = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(size if sys.platform == 'darwin' else size * 1024)


def engine_worker(connection, item, limits, seed):
    """Only this process owns this candidate's search and evaluation caches."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    begin = time.perf_counter()
    cpu = time.process_time()
    try:
        random.seed(seed)
        np.random.seed(seed % 2**32)
        import numba
        numba.set_num_threads(1)
        config = effective_config(item, limits)
        engine = AlphaBetaPlayer(config=config)
        engine._prepare()
        # Execute one bounded search as well: jitclass dispatch and geometry
        # compilation must not be charged to the first measured move.
        from dataclasses import replace
        warm = AlphaBetaPlayer(config=replace(config, max_depth=1, time_limit=30., node_limit=10000))
        warm.analyze(IntransitiveGame().getInitBoard())
        connection.send(dict(kind='ready', config=config.to_dict(), candidate=item['sha256'],
                             startup_seconds=time.perf_counter() - begin,
                             cpu_seconds=time.process_time() - cpu, peak_rss_bytes=peak_rss_bytes()))
        while True:
            state = connection.recv()
            if state is None:
                return
            start, cpu = time.perf_counter(), time.process_time()
            # A fresh engine makes interrupted-game resume independent of the
            # searches that preceded it. Compiled code is read-only and shared.
            engine = AlphaBetaPlayer(config=config)
            result = engine.analyze(state)
            connection.send(dict(kind='move', result=asdict(result),
                                 latency_seconds=time.perf_counter() - start,
                                 cpu_seconds=time.process_time() - cpu,
                                 peak_rss_bytes=peak_rss_bytes()))
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send(dict(kind='error', traceback=traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class MatchFailure(Exception):
    def __init__(self, status, reason):
        self.status, self.reason = status, reason
        super().__init__(reason)


class EngineProcess:
    def __init__(self, item, limits, seed, *, target=engine_worker):
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=target, args=(child, item, limits, seed))
        try:
            self.process.start()
        except BaseException:
            self.connection.close()
            child.close()
            raise
        child.close()

    def receive(self, deadline, cancelled):
        while True:
            if cancelled.is_set():
                raise MatchFailure('cancelled', 'Cancellation requested')
            if time.perf_counter() >= deadline:
                raise MatchFailure('infrastructure_timeout', 'Child response deadline exceeded')
            if self.connection.poll(.025):
                try:
                    message = self.connection.recv()
                except (EOFError, OSError) as exc:
                    raise MatchFailure('crash', f'Child disconnected: {exc}') from exc
                if message.get('kind') == 'error':
                    raise MatchFailure('crash', message['traceback'])
                return message
            if not self.process.is_alive():
                raise MatchFailure('crash', f'Child exited: {self.process.exitcode}')

    def close(self):
        # Always reap both normal and failed children, including a hung engine.
        self.connection.close()
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(1.)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        self.process.close()


def replay(start, row):
    """Verify every physical transition; no evaluator participates in outcomes."""
    game = IntransitiveGame(modelling_draws=False)
    state = unpack(start['state'])
    for move in row['moves']:
        side = int(state[:, :, 82:84].flat[1])
        action = move['result']['action']
        if (move['side'] != side or move['candidate'] != row['colours'][side]
                or move['before'] != state_hash(state)
                or move['observation_sha256'] != state_hash(game.getCanonicalForm(game.getSearchObservation(state), side))
                or type(action) is not int or not 0 <= action < game.getActionSize()
                or not game.getValidMoves(state, side)[action]):
            raise ValueError('Illegal or mismatched recorded move')
        state, _ = game.getNextState(state, side, action)
        if move['after'] != state_hash(state):
            raise ValueError('Recorded transition hash mismatch')
    if row['final_state_sha256'] != state_hash(state):
        raise ValueError('Final state hash mismatch')
    outcome = game.getGameEnded(state, int(state[:, :, 82:84].flat[1]))
    winner = int(np.argmax(outcome)) if outcome.any() else None
    if row['status'] == 'win' and (winner is None or row['winner'] != winner):
        raise ValueError('Recorded official win does not replay')
    if row['status'] != 'win' and (row['winner'] is not None or (row['status'] in FINAL and winner is not None)):
        raise ValueError('Non-win status disagrees with physical outcome')
    if row.get('trajectory_sha256') != digest([start['sha256'], [m['result']['action'] for m in row['moves']]]):
        raise ValueError('Trajectory hash mismatch')
    return state


def play_match(spec, task, path, cancelled, *, engine_factory=EngineProcess):
    start = next(p for p in spec['positions'] if p['sha256'] == task['position'])
    limits = spec['protocols'][task['protocol']]
    items = {c['sha256']: c for c in spec['candidates']}
    path = Path(path)
    if path.exists():
        row = json.loads(path.read_text())
        if row['task'] != task or row['manifest_sha256'] != spec['sha256'] or row['colours'] != task['colours']:
            raise ValueError('Journal does not belong to this immutable schedule')
        if row['status'] not in FINAL | {'running', 'cancelled'}:
            raise ValueError('Unknown journal status')
        state = replay(start, row)
        if row['status'] in FINAL:
            return row
    else:
        state = unpack(start['state'])
        row = dict(task=task, manifest_sha256=spec['sha256'], colours=task['colours'],
                   status='running', winner=None, reason=None, moves=[], startups=[],
                   startup_wait_seconds=0., active_seconds=0., final_state_sha256=state_hash(state))
    game = IntransitiveGame(modelling_draws=False)
    engines, active_start = [], None

    def save():
        row['final_state_sha256'] = state_hash(state)
        row['trajectory_sha256'] = digest([start['sha256'], [m['result']['action'] for m in row['moves']]])
        atomic_json(path, row)

    def terminal():
        side = int(state[:, :, 82:84].flat[1])
        outcome = game.getGameEnded(state, side)
        if outcome.any():
            row.update(status='win', winner=int(np.argmax(outcome)), reason=game.board.get_terminal_reason())
            return True
        return False

    row.update(status='running', reason=None)
    save()
    try:
        if terminal():
            return row
        if cancelled.is_set():
            raise MatchFailure('cancelled', 'Cancellation requested')
        for colour, identity in enumerate(task['colours']):
            row['responsible_colour'] = colour
            begin = time.perf_counter()
            try:
                engine = engine_factory(items[identity], limits, task['seed'])
                engines.append(engine)
                ready = engine.receive(begin + limits['startup_seconds'], cancelled)
            finally:
                row['startup_wait_seconds'] += time.perf_counter() - begin
            if ready['kind'] != 'ready' or ready['candidate'] != identity or ready['config'] != effective_config(items[identity], limits).to_dict():
                raise MatchFailure('crash', 'Child candidate/configuration handshake mismatch')
            row['startups'].append(dict(colour=colour, parent_seconds=time.perf_counter() - begin, **ready))
            save()
        active_start = time.perf_counter()
        previous_seconds = row['active_seconds']
        while not terminal():
            row['active_seconds'] = previous_seconds + time.perf_counter() - active_start
            if len(row['moves']) >= limits['max_plies'] or row['active_seconds'] >= limits['game_seconds']:
                row.update(status='unfinished', reason='safety_ply_limit' if len(row['moves']) >= limits['max_plies'] else 'safety_time_limit')
                break
            if cancelled.is_set():
                raise MatchFailure('cancelled', 'Cancellation requested')
            side = int(state[:, :, 82:84].flat[1])
            row['responsible_colour'] = side
            # Canonicalization swaps player labels/goals, never coordinates.
            observation = game.getSearchObservation(state)
            canonical = game.getCanonicalForm(observation, side)
            engines[side].connection.send(canonical)
            begin = time.perf_counter()
            move_deadline = begin + limits['search']['time_limit'] + limits['timeout_grace']
            game_deadline = active_start + limits['game_seconds'] - previous_seconds
            try:
                response = engines[side].receive(min(move_deadline, game_deadline), cancelled)
            except MatchFailure as exc:
                if exc.status == 'infrastructure_timeout' and game_deadline <= move_deadline:
                    raise MatchFailure('unfinished', 'safety_time_limit') from exc
                raise
            if response['kind'] != 'move':
                raise MatchFailure('crash', 'Unexpected child response')
            result = response['result']
            action = result['action']
            if type(action) is not int or not 0 <= action < game.getActionSize() or not game.getValidMoves(state, side)[action]:
                row['failed_response'] = response
                raise MatchFailure('illegal_move', f'Illegal action {action!r}')
            # A partial-depth choice never enters the completed-depth board.
            # Exact proven results may stop early and are explicitly visible.
            if limits['mode'] == 'depth' and (result['stopped'] or result['selection_source'] != 'completed_iteration'
                    or (result['completed_depth'] < limits['search']['max_depth'] and result['stop_reason'] != 'proven_result')):
                row['failed_response'] = response
                raise MatchFailure('depth_incomplete', 'Requested depth did not complete')
            before = state_hash(state)
            state, _ = game.getNextState(state, side, action)
            row['moves'].append(dict(side=side, candidate=task['colours'][side], before=before,
                                     after=state_hash(state), observation_sha256=state_hash(canonical),
                                     parent_latency_seconds=time.perf_counter() - begin, **response))
            row['active_seconds'] = previous_seconds + time.perf_counter() - active_start
            save()
    except MatchFailure as exc:
        row.update(status=exc.status, reason=exc.reason)
    except (BrokenPipeError, EOFError, OSError) as exc:
        row.update(status='crash', reason=repr(exc))
    except Exception:
        row.update(status='crash', reason=traceback.format_exc())
    finally:
        if active_start is not None:
            row['active_seconds'] = previous_seconds + time.perf_counter() - active_start
        for engine in engines:
            engine.close()
        save()
    return row


def validate_manifest(spec):
    rebuilt = manifest(spec['candidates'], spec['positions'], spec['protocols'], pool=spec['pool'],
                       position_limit=len(spec['selected_positions']))
    if rebuilt != spec:
        raise ValueError('Manifest is changed, invalid or from a different backend version')


def run(spec, output, *, workers=1, cancelled=None):
    """One locked output, fixed quotas and durable match IDs; resume is automatic."""
    import fcntl
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('workers must be in [1, 8] (two child processes per match)')
    validate_manifest(spec)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cancelled = cancelled if cancelled is not None else threading.Event()
    begin = time.perf_counter()
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = output / 'manifest.json'
        if target.exists() and json.loads(target.read_text()) != spec:
            raise ValueError('Output already contains a different manifest')
        atomic_json(target, spec)
        previous_handlers = {}
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.signal(signum, lambda *_: cancelled.set())
        try:
            def execute(task):
                path = output / 'matches' / (task['id'] + '.json')
                if cancelled.is_set() and not path.exists():
                    return None
                return play_match(spec, task, path, cancelled)

            # Executor.map returns in manifest order regardless of completion;
            # match-level files make every completed game durable immediately.
            with ThreadPoolExecutor(max_workers=workers) as executor:
                try:
                    rows = [row for row in executor.map(execute, spec['tasks']) if row is not None]
                except BaseException:
                    cancelled.set()
                    raise
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        from .report import report
        summary = report(spec, rows)
        elapsed = time.perf_counter() - begin
        invocation = dict(workers=workers, child_process_limit=workers * 2, wall_seconds=elapsed,
                          parent_peak_rss_bytes=peak_rss_bytes(), cancelled=cancelled.is_set())
        history = output / 'invocations.json'
        invocations = json.loads(history.read_text()) if history.exists() else []
        invocations.append(invocation)
        atomic_json(history, invocations)
        summary['invocations'] = invocations
        atomic_json(output / 'report.json', summary)
        return summary
