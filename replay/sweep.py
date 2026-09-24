"""Score search configurations against a recorded pool.

Every configuration is re-run from a fresh tree on each recorded root, so the
comparison between configurations is fair. The action space here is deliberately
*compute allocation only* -- simulations, cpuct, FPU, forced playouts,
universes. Dirichlet noise and the move-selection temperature are excluded on
purpose: they trade per-move quality for data diversity, and an objective scored
on per-move agreement would drive both to zero and quietly destroy training.
"""

import itertools

import numpy as np

from MCTS import MCTS
from utils import dotdict

from .bench import wall_clock_per_move
from .simulator import ReplayNet

# `prob_fullMCTS` is pinned to 1 and every root is searched with
# `force_full_search`, so a configuration's simulation count is exactly
# `numMCTSSims` and no RNG is consulted.
BASE_ARGS = dict(
    numMCTSSims=800,
    cpuct=1.25,
    fpu=0.0,
    universes=1,
    forced_playouts=False,
    prob_fullMCTS=1.0,
    ratio_fullMCTS=5,
    no_mem_optim=True,
    dirichletAlpha=0.0,
    temperature=[1.0, 0.1, 1.1],
)

SWEEPABLE = ('numMCTSSims', 'cpuct', 'fpu', 'universes', 'forced_playouts')


def make_args(base, overrides):
    merged = dict(BASE_ARGS)
    merged.update(base or {})
    known = set(merged)
    unknown = set(overrides or {}) - known
    if unknown:
        raise ValueError(f'unknown search settings {sorted(unknown)}; '
                         f'known settings are {sorted(known)}')
    merged.update(overrides or {})
    return dotdict(merged)


def expand_grid(grid):
    """{'numMCTSSims': [100, 400]} -> [{'numMCTSSims': 100}, {'numMCTSSims': 400}]"""
    keys = sorted(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def _accumulator():
    return dict(positions=0, agreement=0, tv=0.0, value_error=0.0, evals=0,
                reference_evals=0, hits=0, misses=0, seconds=0.0, reference_seconds=0.0)


def _finalise(totals, beta1):
    count = max(totals['positions'], 1)
    lookups = max(totals['hits'] + totals['misses'], 1)
    evals_per_move = totals['evals'] / count
    reference_per_move = totals['reference_evals'] / count
    agreement = totals['agreement'] / count
    # Cost is modelled wall clock when a measured latency curve is available,
    # and evaluation count otherwise. On the measured curve for this repository
    # -- a single-threaded ONNX session, flat within 3% from batch 1 to 64 --
    # the two are proportional, so the collection size cannot change the
    # ordering. That is the answer to Dream-RSI's parallelism bonus here, and
    # it is a measurement rather than an assumption: re-run `replay bench`
    # after any change to the session's thread budget.
    if totals['reference_seconds'] > 0:
        cost_ratio = totals['seconds'] / totals['reference_seconds']
    elif reference_per_move:
        cost_ratio = evals_per_move / reference_per_move
    else:
        cost_ratio = float('nan')
    return dict(
        positions=totals['positions'],
        agreement=agreement,
        policy_tv=totals['tv'] / count,
        value_mae=totals['value_error'] / count,
        evals_per_move=evals_per_move,
        reference_evals_per_move=reference_per_move,
        seconds_per_move=totals['seconds'] / count if totals['seconds'] else None,
        cost_ratio=cost_ratio,
        miss_rate=totals['misses'] / lookups,
        value=agreement - beta1 * cost_ratio,
    )


def evaluate_config(game, store, positions, args, *, on_miss='uniform', fallback=None,
                    references=None, beta1=0.0, latency_curve=None, collection=1):
    """Replay one configuration over every recorded root.

    `references` maps a pool key to the action an external reference chose; when
    omitted, the recorded full-search decision is the reference.
    """
    net = ReplayNet(store, game, on_miss=on_miss, fallback=fallback)
    splits, skipped = {}, 0

    for position in positions:
        reference_action = position['reference_action']
        if references is not None:
            reference_action = references.get(position['key'])
            if reference_action is None:
                skipped += 1
                continue
        board = position['state']
        mask = np.asarray(game.getValidMoves(board, 0)).astype(bool)
        before_hits, before_misses = net.hits, net.misses
        mcts = MCTS(game, net, args, dirichlet_noise=False)
        pi, q, _ = mcts.getActionProb(board, temp=1.0, force_full_search=True)
        pi = np.asarray(pi, dtype=np.float64)
        used = (net.hits - before_hits) + (net.misses - before_misses)

        totals = splits.setdefault(position['split'], _accumulator())
        totals['positions'] += 1
        totals['agreement'] += int(np.argmax(pi) == reference_action)
        reference_pi = np.asarray(position['reference_pi'], dtype=np.float64)
        if reference_pi.size == int(mask.sum()):
            totals['tv'] += 0.5 * float(np.abs(pi[mask] - reference_pi).sum())
        totals['value_error'] += abs(float(q[0]) - float(position['reference_q'][0]))
        totals['evals'] += used
        totals['reference_evals'] += position['reference_evals']
        if latency_curve is not None:
            totals['seconds'] += wall_clock_per_move(used, collection, latency_curve)
            totals['reference_seconds'] += wall_clock_per_move(
                position['reference_evals'], collection, latency_curve)
        totals['hits'] += net.hits - before_hits
        totals['misses'] += net.misses - before_misses

    combined = _accumulator()
    for totals in splits.values():
        for key in combined:
            combined[key] += totals[key]
    report = {name: _finalise(totals, beta1) for name, totals in sorted(splits.items())}
    report['all'] = _finalise(combined, beta1)
    report['skipped_unlabelled'] = skipped
    report['distinct_misses'] = net.distinct_misses
    return report


def sweep(game, store, positions, configs, *, base=None, on_miss='uniform', fallback=None,
          references=None, beta1=0.0, latency_curve=None, collection=1, progress=None):
    results = []
    for index, overrides in enumerate(configs):
        args = make_args(base, overrides)
        report = evaluate_config(game, store, positions, args, on_miss=on_miss,
                                 fallback=fallback, references=references, beta1=beta1,
                                 latency_curve=latency_curve, collection=collection)
        results.append(dict(config=dict(overrides), report=report))
        if progress is not None:
            progress(index + 1, len(configs), overrides, report)
    results.sort(key=lambda row: row['report']['all']['value'], reverse=True)
    return results


def self_check(game, store, positions, base):
    """Replay the recording configuration; it must reproduce it exactly.

    Only meaningful for a pool recorded without tree reuse: with reuse, a
    recorded decision depends on statistics inherited from earlier plies, which
    a fresh-tree replay cannot and should not reproduce.
    """
    args = make_args(base, {})
    report = evaluate_config(game, store, positions, args, on_miss='uniform')
    checks = report['all']
    return dict(
        agreement=checks['agreement'],
        miss_rate=checks['miss_rate'],
        policy_tv=checks['policy_tv'],
        exact=bool(checks['agreement'] == 1.0 and checks['miss_rate'] == 0.0
                   and checks['policy_tv'] < 1e-12),
    )
