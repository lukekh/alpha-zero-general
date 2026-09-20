"""Static exchange evaluation for the cyclic capture rule; see EXCHANGE.md.

The capture cycle gives the series no attacker-selection freedom: the kind that
may take on a square is forced, and same-kind attackers are interchangeable
because the valuation reads only per-side, per-kind counts. What remains is the
stopping decision, resolved by the usual backward induction.

The series terminates on the attacker pool, never on the kind cycle, which
repeats with period three. Every step consumes one piece standing next to the
target square and permanently vacates it, so at most eight steps follow the
first. Nothing here is a certificate: a swing is a heuristic ranking.
"""
from functools import lru_cache

import numpy as np
from numba import njit

from ..IntransitiveConstants import DIRECTIONS
from .material import BASE, variable_material_total

# One victim plus the at most eight neighbours that can recapture.
MAX_STEPS = 9


def neighbours(square):
    """The at most eight board squares a king step from `square`."""
    x, y = square % 9, square // 9
    return [(y + dy) * 9 + x + dx
            for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if (dx or dy) and 0 <= x + dx < 9 and 0 <= y + dy < 9]


@njit(cache=True)
def material_edge(counts, side, count_weight, variable, linear):
    """The evaluator's own material term for these counts, from `side`'s view."""
    if variable:
        own = variable_material_total(counts, side, linear) / BASE
        enemy = variable_material_total(counts, 1 - side, linear) / BASE
    else:
        own = float(counts[3*side] + counts[3*side+1] + counts[3*side+2])
        enemy = float(counts[3*(1-side)] + counts[3*(1-side)+1] + counts[3*(1-side)+2])
    return count_weight * (own - enemy)


def material_edge_reference(counts, side, count_weight, variable, linear):
    """Readable twin of `material_edge`, sharing the material formula itself.

    The valuation is deliberately not re-derived here: `variable_material_total`
    already has its own reference and tests, and reusing it keeps the two
    exchange implementations exactly equal in floating point, so a parity
    failure can only mean the series or the induction diverged.
    """
    if variable:
        counts = np.asarray(counts, dtype=np.int64)
        own = variable_material_total(counts, side, linear) / BASE
        enemy = variable_material_total(counts, 1 - side, linear) / BASE
    else:
        own = float(sum(counts[3*side:3*side+3]))
        enemy = float(sum(counts[3*(1-side):3*(1-side)+3]))
    return count_weight * (own - enemy)


@njit(cache=True)
def exchange_swing(pieces, counts, side, source, target, count_weight, variable, linear):
    """`(swing, gain)` for the capture `source -> target`, in evaluator units.

    `swing` allows either side to break off; `gain` is the first step alone.
    A move that is not a legal capture for `side` scores zero on both.
    """
    mover = int(pieces[source // 9, source % 9])
    victim = int(pieces[target // 9, target % 9])
    if mover == 0 or victim == 0 or int(mover < 0) != side or int(victim < 0) == side:
        return 0., 0.
    attacker_kind, victim_kind = abs(mover) - 1, abs(victim) - 1
    if (attacker_kind + 1) % 3 != victim_kind:
        return 0., 0.
    # Only pieces already next to the target can ever capture on it, and the
    # series never puts one back, so this pool is the whole exchange.
    pool = np.zeros((2, 3), dtype=np.int64)
    tx, ty = target % 9, target // 9
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            nx, ny = tx + dx, ty + dy
            if (dx == 0 and dy == 0) or not (0 <= nx < 9 and 0 <= ny < 9) or ny*9 + nx == source:
                continue
            code = int(pieces[ny, nx])
            if code:
                pool[int(code < 0), abs(code) - 1] += 1
    removed_side = np.empty(MAX_STEPS + 1, dtype=np.int64)
    removed_kind = np.empty(MAX_STEPS + 1, dtype=np.int64)
    occupant_side, occupant_kind = 1 - side, victim_kind
    next_side, next_kind = side, attacker_kind
    steps = 0
    while True:
        steps += 1
        removed_side[steps], removed_kind[steps] = occupant_side, occupant_kind
        occupant_side, occupant_kind = next_side, next_kind
        next_side, next_kind = 1 - occupant_side, (occupant_kind + 2) % 3
        if pool[next_side, next_kind] <= 0:
            break
        pool[next_side, next_kind] -= 1
    values = np.empty(steps + 1, dtype=np.float64)
    work = counts.copy()
    values[0] = material_edge(work, side, count_weight, variable, linear)
    for i in range(1, steps + 1):
        work[3*removed_side[i] + removed_kind[i]] -= 1
        values[i] = material_edge(work, side, count_weight, variable, linear)
    # Only the first step is compulsory: it is the move being evaluated. After
    # an even number of steps it is the mover's turn again.
    best = values[steps]
    for i in range(steps - 1, 0, -1):
        if i % 2 == 0:
            if values[i] > best:
                best = values[i]
        elif values[i] < best:
            best = values[i]
    return best - values[0], values[1] - values[0]


def exchange_series(pieces, side, source, target):
    """The forced series for `source -> target`, as the `(owner, kind)` removed.

    Empty when the move is not a legal capture for `side`. The list is at most
    eight long: every step spends one piece standing next to the target and
    never puts one back, which is the whole termination argument, since the
    kinds themselves cycle with period three and never run out.
    """
    mover = int(pieces[source // 9, source % 9])
    victim = int(pieces[target // 9, target % 9])
    if not mover or not victim or int(mover < 0) != side or int(victim < 0) == side:
        return []
    attacker_kind, victim_kind = abs(mover) - 1, abs(victim) - 1
    if (attacker_kind + 1) % 3 != victim_kind:
        return []
    pool = {}
    for square in neighbours(target):
        code = int(pieces[square // 9, square % 9])
        if square != source and code:
            key = (int(code < 0), abs(code) - 1)
            pool[key] = pool.get(key, 0) + 1
    removed = [(1 - side, victim_kind)]   # step one takes the victim,
    occupant = (side, attacker_kind)      # leaving the mover on the square
    while True:
        attacker = (1 - occupant[0], (occupant[1] + 2) % 3)
        if not pool.get(attacker):
            break
        pool[attacker] -= 1
        removed.append(occupant)
        occupant = attacker
    return removed


def exchange_swing_reference(pieces, counts, side, source, target, count_weight, variable, linear):
    """Readable definition of EXCHANGE.md; `exchange_swing` is its compiled twin."""
    removed = exchange_series(pieces, side, source, target)
    if not removed:
        return 0., 0.
    work = list(counts)
    values = [material_edge_reference(work, side, count_weight, variable, linear)]
    for owner, kind in removed:
        work[3*owner + kind] -= 1
        values.append(material_edge_reference(work, side, count_weight, variable, linear))
    best = values[-1]
    for i in range(len(values) - 2, 0, -1):
        best = max(values[i], best) if i % 2 == 0 else min(values[i], best)
    return best - values[0], values[1] - values[0]


@njit(cache=True)
def exchange_swings(pieces, counts, side, actions, count_weight, variable, linear):
    """Per-action swing and gain, and how many of the actions were captures."""
    scores = np.zeros(len(actions), dtype=np.float64)
    gains = np.zeros(len(actions), dtype=np.float64)
    captures = 0
    for i in range(len(actions)):
        action = int(actions[i])
        source = action // 8
        dx, dy = DIRECTIONS[action % 8]
        x, y = source % 9 + dx, source // 9 + dy
        if pieces[y, x] != 0:
            captures += 1
            scores[i], gains[i] = exchange_swing(pieces, counts, side, source, 9*y + x,
                                                 count_weight, variable, linear)
    return scores, gains, captures


def exchange_swings_reference(pieces, counts, side, actions, count_weight, variable, linear):
    scores = np.zeros(len(actions), dtype=np.float64)
    gains = np.zeros(len(actions), dtype=np.float64)
    captures = 0
    for i, action in enumerate(map(int, actions)):
        source = action // 8
        dx, dy = DIRECTIONS[action % 8]
        x, y = source % 9 + dx, source // 9 + dy
        if pieces[y, x]:
            captures += 1
            scores[i], gains[i] = exchange_swing_reference(
                pieces, counts, side, source, 9*y + x, count_weight, variable, linear)
    return scores, gains, captures


def survey(pieces, counts, side, actions, config):
    """Swings and gains for every candidate action under this configuration."""
    kernel = exchange_swings if config.compiled_see_enabled else exchange_swings_reference
    return kernel(pieces, np.asarray(counts, dtype=np.int64), side,
                  np.asarray(actions, dtype=np.int64), float(config.count_weight),
                  config.variable_material_enabled, config.variable_material_linear)


def stalemate_safe(pieces, side):
    """Whether no single reply by `side` can stalemate the opponent.

    Reuses the two counting arguments the selective guards already rely on: a
    side with more than two disjoint (piece, empty neighbour) pairs, or more
    than one piece with more than one empty neighbour, still has a move after
    one ply. Either one suffices; both failing only means neither argument
    applies, so pruning is refused for the node.
    """
    from .kernels import crowded_side_has_moves, open_side_has_moves
    return bool(crowded_side_has_moves(pieces, 1 - side, 1)
                or open_side_has_moves(pieces, 1 - side, 1))


def delta_allowance(config):
    """Evaluator-unit allowance for the non-material swing of one capture.

    Built from the same per-module scales as `selective.margin`, without its
    material term: delta pruning already accounts for material exactly. This is
    a heuristic allowance, not a bound on what the evaluator can do.
    """
    advantage_scale = (max(abs(config.predator_zero_bonus), abs(config.predator_scarcity_bonus))
                       + abs(config.prey_bonus))
    allowance = abs(config.advantage_weight) * advantage_scale
    for name, scale in (('attack', 3), ('defence', 4), ('overload', 2),
                        ('pressure', 8), ('runner', 3)):
        if getattr(config, name + '_enabled'):
            allowance += scale * abs(getattr(config, name + '_weight'))
    return config.delta_margin * allowance


@lru_cache(maxsize=1)
def warm_exchange_kernels():
    counts = np.ones(6, dtype=np.int64)
    actions = np.empty(0, dtype=np.int64)
    # Both board layouts the search passes: the compact copy and a state plane.
    for board in (np.zeros((9, 9), dtype=np.int8), np.zeros((9, 9, 84), dtype=np.int8)[:, :, 0]):
        for variable in (False, True):
            exchange_swing(board, counts, 0, 0, 1, 100., variable, False)
            exchange_swings(board, counts, 0, actions, 100., variable, False)
