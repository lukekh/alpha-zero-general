"""Practical cost: static evaluation, achieved depth, and teacher label throughput.

Optimized fitness is not evidence that a configuration is affordable. These
measurements price the evaluation itself, the depth it can actually finish
inside a move budget, and how long one training label takes at the teacher's
own depth. Each candidate is profiled in its own spawned process so startup,
compilation and peak memory belong to that candidate and not to whichever one
happened to run first.
"""
from dataclasses import replace
import multiprocessing as mp
import statistics
import time

import numpy as np

from ..IntransitiveGame import IntransitiveGame
from ..heuristics.budget import Budget
from ..heuristics.config import SearchConfig
from ..heuristics.evaluation import Evaluator
from ..heuristics.search import AlphaBetaPlayer
from ..supervised_minimax import TEACHER
from ..tournament.runner import peak_rss_bytes
from ..tournament.spec import unpack


def observations(positions):
    """Canonical current-player observations, exactly as a match would search."""
    game = IntransitiveGame(modelling_draws=False)
    rows = []
    for item in positions:
        state = unpack(item['state'])
        side = int(state[:, :, 82:84].flat[1])
        rows.append((item['sha256'],
                     game.getCanonicalForm(game.getSearchObservation(state), side)))
    return rows


def static_cost(config, states, *, repeats=50):
    """Time one leaf evaluation with proof and tree search removed.

    This is the per-node price every deeper search multiplies, which is why a
    candidate can win a fixed-depth board and still lose at equal time.
    """
    config = replace(config, proof_depth=0, proof_nodes=0)
    evaluator = Evaluator(IntransitiveGame(modelling_draws=False), config)
    rows = []
    for identity, state in states:
        side = int(state[:, :, 82:84].flat[1])
        budget = Budget(10**15, 3600.)
        evaluator.score(state, side, budget, proof={'status': 'unknown'})  # Warm caches.
        begin, before = time.perf_counter(), budget.work
        for _ in range(repeats):
            evaluator.score(state, side, budget, proof={'status': 'unknown'})
        elapsed = time.perf_counter() - begin
        rows.append(dict(position=identity, evaluations=repeats,
                         seconds_per_evaluation=elapsed / repeats,
                         work_per_evaluation=(budget.work - before) / repeats))
    return dict(positions=rows, repeats=repeats,
                median_seconds=statistics.median(r['seconds_per_evaluation'] for r in rows),
                median_work=statistics.median(r['work_per_evaluation'] for r in rows))


def depth_ladder(config, states, depths, *, seconds, node_limit=10**12):
    """Price each fixed depth, stopping a position once a depth runs out of time.

    A depth that cannot finish inside the cap is reported as unfinished with the
    work it reached. Deeper entries for that position are not attempted: their
    cost is bounded below by the one that already failed.
    """
    rows = []
    for identity, state in states:
        for depth in sorted(depths):
            limits = replace(config, max_depth=depth, time_limit=seconds, node_limit=node_limit)
            engine = AlphaBetaPlayer(config=limits)
            engine._prepare()
            begin = time.perf_counter()
            result = engine.analyze(state)
            elapsed = time.perf_counter() - begin
            rows.append(dict(position=identity, depth=depth, seconds=elapsed,
                             completed_depth=int(result.completed_depth), work=int(result.work),
                             nodes=int(result.nodes), stopped=bool(result.stopped),
                             stop_reason=result.stop_reason,
                             complete=bool(not result.stopped
                                           or result.stop_reason == 'proven_result')))
            if not rows[-1]['complete']:
                break
    ladder = {}
    for depth in sorted(depths):
        matching = [r for r in rows if r['depth'] == depth]
        attempted = len(matching)
        complete = [r for r in matching if r['complete']]
        ladder[str(depth)] = dict(
            attempted=attempted, completed=len(complete),
            median_seconds=statistics.median(r['seconds'] for r in complete) if complete else None,
            median_work=statistics.median(r['work'] for r in complete) if complete else None,
            unfinished_seconds=[r['seconds'] for r in matching if not r['complete']],
            skipped=len(states) - attempted)
    return dict(cap_seconds=seconds, by_depth=ladder, searches=rows)


def equal_time(config, states, budgets):
    """Depth actually reached when every candidate is given the same move time."""
    rows = []
    for identity, state in states:
        for seconds in sorted(budgets):
            limits = replace(config, max_depth=64, time_limit=seconds, node_limit=10**12)
            engine = AlphaBetaPlayer(config=limits)
            engine._prepare()
            begin = time.perf_counter()
            result = engine.analyze(state)
            rows.append(dict(position=identity, seconds=seconds,
                             elapsed=time.perf_counter() - begin,
                             completed_depth=int(result.completed_depth),
                             selected_depth=int(result.selected_depth),
                             work=int(result.work), stop_reason=result.stop_reason))
    return dict(by_budget={str(seconds): dict(
                    median_completed_depth=statistics.median(
                        r['completed_depth'] for r in rows if r['seconds'] == seconds),
                    min_completed_depth=min(r['completed_depth'] for r in rows if r['seconds'] == seconds),
                    median_work=statistics.median(r['work'] for r in rows if r['seconds'] == seconds))
                    for seconds in sorted(budgets)},
                searches=rows)


def label_throughput(ladder, *, depth, teacher=TEACHER):
    """Training labels per hour at the teacher's own depth, from measured cost.

    The teacher searches one position per label, so the depth ladder already
    contains the measurement; reporting it separately avoids paying for the
    same searches twice and keeps the protocol identical.
    """
    row = ladder['by_depth'].get(str(depth))
    if row is None:
        raise ValueError(f'The ladder does not include depth {depth}')
    seconds = row['median_seconds']
    return dict(teacher_protocol=dict(teacher), depth=depth,
                completed=row['completed'], attempted=row['attempted'],
                median_seconds_per_label=seconds,
                labels_per_hour=(3600. / seconds) if seconds else None,
                note='Measured at the cost ladder cap, which is shorter than the teacher\'s own '
                     'time limit; an unfinished depth means no label at that depth, not a slow one.')


def shipped_budget(config, states):
    """What the shipped default limits actually deliver for this evaluation.

    `SearchConfig()` ships `max_depth` 3 with a work cap sized for the core
    evaluator. A candidate whose modules cost far more per node keeps the depth
    setting and loses the depth, silently, in every default-configured bot.
    """
    defaults = SearchConfig()
    rows = []
    for identity, state in states:
        limits = replace(config, max_depth=defaults.max_depth, node_limit=defaults.node_limit,
                         time_limit=defaults.time_limit)
        engine = AlphaBetaPlayer(config=limits)
        engine._prepare()
        begin = time.perf_counter()
        result = engine.analyze(state)
        rows.append(dict(position=identity, seconds=time.perf_counter() - begin,
                         completed_depth=int(result.completed_depth), work=int(result.work),
                         stopped=bool(result.stopped), stop_reason=result.stop_reason))
    return dict(limits=dict(max_depth=defaults.max_depth, node_limit=defaults.node_limit,
                            time_limit=defaults.time_limit),
                median_completed_depth=statistics.median(r['completed_depth'] for r in rows),
                reached_requested_depth=sum(r['completed_depth'] >= defaults.max_depth for r in rows),
                positions=len(rows), searches=rows)


def profile(config, positions, design):
    """One candidate's complete cost profile, measured in this process."""
    config = SearchConfig(**config) if isinstance(config, dict) else config
    states = observations(positions)
    begin = time.perf_counter()
    warm = AlphaBetaPlayer(config=replace(config, max_depth=1, time_limit=60., node_limit=10**7))
    warm._prepare()
    warm.analyze(states[0][1])
    startup = time.perf_counter() - begin
    ladder = depth_ladder(config, states, design['cost_depths'], seconds=design['cost_seconds'])
    return dict(startup_seconds=startup,
                static=static_cost(config, states),
                ladder=ladder,
                equal_time=equal_time(config, states, design['equal_time_budgets']),
                shipped=shipped_budget(config, states),
                labels=label_throughput(ladder, depth=design['teacher_depth']),
                peak_rss_bytes=peak_rss_bytes(),
                cpu_seconds=time.process_time(),
                positions=[identity for identity, _ in states])


def _child(connection, config, positions, design):
    try:
        import numba
        numba.set_num_threads(1)
        np.random.seed(0)
        connection.send(dict(kind='profile', result=profile(config, positions, design)))
    except BaseException as error:  # Report, never hang the parent.
        import traceback
        connection.send(dict(kind='error', traceback=traceback.format_exc(), error=repr(error)))
    finally:
        connection.close()


def measure(config, positions, design, *, timeout=None):
    """Profile one candidate in a fresh process, so startup and RSS are its own."""
    context = mp.get_context('spawn')
    parent, child = context.Pipe()
    process = context.Process(target=_child, args=(child, config, positions, design))
    process.start()
    child.close()
    # Every ladder and equal-time search is individually capped; this only
    # bounds the total, including compilation, and never shortens a search.
    if timeout is None:
        timeout = 600. + design['cost_seconds'] * len(design['cost_depths']) * design['cost_positions']
    try:
        if not parent.poll(timeout):
            raise TimeoutError(f'Cost profile exceeded {timeout} seconds')
        message = parent.recv()
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
        process.join(5.)
        if process.is_alive():
            process.kill()
            process.join()
    if message['kind'] != 'profile':
        raise RuntimeError(message.get('traceback', message.get('error', 'unknown child failure')))
    return message['result']


def compare(candidate, baseline, thresholds):
    """Cost verdict against the predeclared depth and throughput allowances."""
    rows = {}
    for seconds, row in candidate['equal_time']['by_budget'].items():
        other = baseline['equal_time']['by_budget'][seconds]
        rows[seconds] = dict(candidate=row['median_completed_depth'],
                             baseline=other['median_completed_depth'],
                             loss=other['median_completed_depth'] - row['median_completed_depth'])
    worst = max((row['loss'] for row in rows.values()), default=0)
    ours, theirs = candidate['labels']['labels_per_hour'], baseline['labels']['labels_per_hour']
    ratio = (ours / theirs) if ours and theirs else None
    return dict(equal_time_depth=rows, worst_median_depth_loss=worst,
                depth_within_allowance=worst <= thresholds['max_median_depth_loss'],
                static_seconds_ratio=(candidate['static']['median_seconds']
                                      / baseline['static']['median_seconds']),
                label_throughput_ratio=ratio,
                teacher_throughput_acceptable=(ratio is not None
                                               and ratio >= thresholds['min_label_throughput_ratio']),
                shipped_median_depth=candidate['shipped']['median_completed_depth'],
                baseline_shipped_median_depth=baseline['shipped']['median_completed_depth'],
                shipped_reaches_default_depth=(candidate['shipped']['reached_requested_depth']
                                               == candidate['shipped']['positions']))
