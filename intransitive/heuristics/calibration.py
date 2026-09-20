"""Measure what one quiet ply is worth, so a margin can be derived not guessed.

`selective.margin` charges the module terms at their per-side feature ranges.
Those bound the whole evaluation and not one ply of it, which is why the default
allowance prunes nothing at evolved scales: it is roughly an order of magnitude
larger than any quiet move can actually gain. This samples the real quantity so
`futility_margin` can be set from a measurement of the genome in hand, rather
than carried over from a genome it was never calibrated for (issue #66).
"""
from . import selective
from .budget import Budget
from .evaluation import Evaluator
from .position import SearchPosition


def quiet_ply_gains(config, states, *, game=None, depth=1):
    """Signed static change across every quiet move the guard and test admit.

    Conditioned exactly where futility runs: positions `guarded` accepts, moves
    `quiet` accepts. At depth one the skipped child is evaluated statically, so
    this static delta is precisely the quantity the allowance has to bound. From
    depth two the child gets a search that may capture, which is why material
    and advantage are charged from that ply and are outside this measurement.
    """
    if game is None:
        from ..IntransitiveGame import IntransitiveGame
        game = IntransitiveGame()
    evaluator = Evaluator(game, config)

    def score(position):
        return evaluator.score(position, position.side, Budget(config.node_limit, config.time_limit),
                               proof={'status': 'unknown'}, counts=position.counts)

    gains, positions = [], 0
    for state in states:
        position = SearchPosition(state)
        if not selective.guarded(position, depth, Budget(config.node_limit, config.time_limit)):
            continue
        positions += 1
        context = selective.quiet_context(position)
        before = score(position)
        for action in map(int, position.legal()):
            if not selective.quiet(position, action, context):
                continue
            position.push(action)
            try:
                # The child scores from the opponent's view; negate for ours.
                gains.append(-score(position) - before)
            finally:
                position.pop()
    return dict(positions=positions, moves=len(gains), gains=gains)


def calibrate(config, states, *, game=None, depth=1, quantile=1.):
    """The `futility_margin` that covers `quantile` of measured quiet-ply gains.

    A quantile is not a bound. Futility is a heuristic and skipping a move in
    the tail is the risk it exists to take; this reports which fraction of
    observed gains a multiplier covers, and never that a multiplier is safe.
    Only positive gains matter: a move that loses ground cannot be the move the
    allowance was protecting.
    """
    if not 0 < quantile <= 1:
        raise ValueError('quantile must be in (0, 1]')
    sample = quiet_ply_gains(config, states, game=game, depth=depth)
    gains = sorted(sample['gains'])
    if not gains:
        raise ValueError('No quiet move in any guarded position; nothing to calibrate')
    index = min(len(gains) - 1, int(round(quantile * (len(gains) - 1))))
    covered = max(0., gains[index])
    allowance = selective.margin(config, depth) / config.futility_margin
    return dict(sample, quantile=quantile, covered_gain=covered,
                mean_gain=sum(gains) / len(gains), max_gain=gains[-1],
                allowance_at_unit_margin=allowance,
                futility_margin=covered / allowance if allowance else None)
