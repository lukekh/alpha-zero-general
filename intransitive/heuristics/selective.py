"""Conservative Intransitive guards shared conceptually with native search."""
from contextlib import contextmanager
import numpy as np
from ..IntransitiveConstants import action_destination
from ..IntransitiveLogicNumba import raw_movement_mask


def supported(config):
    # Evolved/route scales require an explicit experimental opt-in.
    # Otherwise diagnostics report that selection is disabled.
    if config.selective_evaluator_enabled:
        return True
    return (not config.variable_material_enabled and config.count_weight == 100 and config.advantage_weight == 25
            and config.predator_zero_bonus == 1 and config.predator_scarcity_bonus == .5
            and config.prey_bonus == .25
            and not (config.attack_enabled or config.defence_enabled or config.overload_enabled)
            and (not config.pressure_enabled or 0 <= config.pressure_weight <= 20))


def distance(a, b):
    return max(abs(a % 9 - b % 9), abs(a // 9 - b // 9))


def guarded(position, depth, budget):
    budget.charge(81 + 2 * 648)
    pieces = position.pieces.ravel()
    occupied = np.flatnonzero(pieces)
    if position.clock + depth + 1 >= 80 or any(n > 1 for n in position.occurrences.values()):
        return False
    if min(sum(position.counts[:3]), sum(position.counts[3:])) < 4:
        return False
    for square in occupied:
        side = int(pieces[square] < 0)
        if distance(int(square), 80 if side == position.a1 else 0) <= max(3, (depth + 1)//2):
            return False
    for side in (0, 1):
        actions = np.flatnonzero(raw_movement_mask(position.pieces, side))
        if len(actions) < 8:
            return False
        for action in actions:
            x, y = action_destination(int(action))
            if position.pieces[y, x]:
                return False  # includes forced captures and threatened retreats
    return True


def blocks(square, piece, goal):
    """Whether `square` lies on some shortest king route from `piece` to `goal`.

    King distance is additive exactly along such a route, so the equality test
    is the whole predicate. An occupied route square is a genuine obstruction
    here: `guarded()` has already refused the node if any capture is available,
    so the runner cannot simply take the blocker.
    """
    return distance(piece, square) + distance(square, goal) == distance(piece, goal)


def quiet(position, action):
    """Whether a move may be discarded by a margin test rather than searched.

    Reviewed for issue #68 against corner-threat creation and prevention,
    forced defence and races. Threat creation, captures and the mover's own
    fastest runners were already excluded. Added: a move that vacates, or
    occupies, a square on a shortest enemy route to the enemy's corner is
    corner-threat prevention, so it is never quiet. This is deliberately
    stricter than the issue #60 version; it can only refuse pruning.
    """
    pieces = position.pieces.ravel()
    source = action // 8
    x, y = action_destination(action)
    target = y * 9 + x
    if pieces[target]:
        return False
    side = position.side
    goal = 80 if side == position.a1 else 0
    enemy_goal = 80 - goal
    own = [int(s) for s in np.flatnonzero(pieces) if int(pieces[s] < 0) == side]
    # Keep every fastest runner, including moves away from its goal.
    if distance(source, goal) <= min(distance(s, goal) for s in own):
        return False
    for square in np.flatnonzero(pieces):
        square = int(square)
        if int(pieces[square] < 0) == side:
            continue
        if min(distance(square, source), distance(square, target)) <= 2:
            return False
        if blocks(source, square, enemy_goal) or blocks(target, square, enemy_goal):
            return False  # leaving or taking a blocking square is defence
    return distance(target, goal) > 3


def allowance(config, position=None):
    """One ply of evaluator units, in this configuration's own scales.

    Every margin in the selective family is a multiple of this quantity, so a
    re-tuned genome moves all of them together instead of silently invalidating
    a constant. It is a local heuristic scale, never a bound on future gains.
    """
    if config.selective_evaluator_enabled:
        # A local heuristic allowance, not a bound on future evaluation changes.
        piece_scale = abs(config.count_weight)
        if config.variable_material_enabled:
            if position is None:
                raise ValueError('Variable-material pruning needs current piece counts')
            from .material import variable_piece_values, BASE
            for side in (0, 1):
                own = position.counts[side*3:side*3+3]
                enemy = position.counts[(1-side)*3:(1-side)*3+3]
                values = variable_piece_values(own, enemy)
                piece_scale = max([piece_scale] + [abs(config.count_weight)*v/BASE
                                  for n, v in zip(own, values) if n])
        advantage_scale = max(abs(config.predator_zero_bonus),
                              abs(config.predator_scarcity_bonus)) + abs(config.prey_bonus)
        total = piece_scale/2 + abs(config.advantage_weight)*advantage_scale
        for name, scale in (('attack', 3), ('defence', 4), ('overload', 2), ('pressure', 8)):
            if getattr(config, name+'_enabled'):
                total += scale*abs(getattr(config, name+'_weight'))
        return total

    weight = config.pressure_weight if config.pressure_enabled else 0.
    return config.count_weight / 2 + config.advantage_weight + 8 * weight


def margin(config, depth, position=None):
    """Forward futility: how much a skipped quiet child is allowed to gain."""
    return depth * config.futility_margin * allowance(config, position)


def razor_margin(config, depth, position=None):
    """Razoring: how far below alpha a node may stand before dropping to
    quiescence. Deliberately the most generous of the three, because a razored
    node abandons its whole full-width search rather than one child."""
    return depth * config.razoring_margin * allowance(config, position)


def reverse_margin(config, depth, position=None):
    """Reverse futility: how much the opponent is allowed to claw back from a
    static score already above beta. Unlike null-move pruning it makes no
    hypothetical pass, so zugzwang is not one of its failure modes."""
    return depth * config.reverse_futility_margin * allowance(config, position)


@contextmanager
def unpruned(player, *, null=False):
    old = player._selective_disabled, player._null_context, player.use_table
    player._selective_disabled = True
    player._null_context = null
    player.use_table = False
    try:
        yield
    finally:
        player._selective_disabled, player._null_context, player.use_table = old
