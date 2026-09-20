"""Conservative Intransitive guards shared conceptually with native search."""
from contextlib import contextmanager
import numpy as np
from ..IntransitiveConstants import ACTION_SIZE, action_destination
from ..IntransitiveLogicNumba import raw_movement_mask
from .config import TECHNIQUE_COUNTERS, activation, blocked, unsupported_scales  # noqa: F401

_ROWS, _COLS = np.arange(81) // 9, np.arange(81) % 9
# Chebyshev distances, precomputed once: these guards run at every candidate
# node and every candidate move, so the scans below must not be Python loops.
SQUARE_DISTANCE = np.maximum(np.abs(_COLS[:, None] - _COLS), np.abs(_ROWS[:, None] - _ROWS))
GOAL_DISTANCE = {goal: SQUARE_DISTANCE[goal].copy() for goal in (0, 80)}


def _destinations():
    """Destination square of every action, or -1 where it leaves the board."""
    table = np.full(ACTION_SIZE, -1, dtype=np.int64)
    for action in range(ACTION_SIZE):
        x, y = action_destination(action)
        if 0 <= x < 9 and 0 <= y < 9:
            table[action] = y * 9 + x
    return table


DESTINATIONS = _destinations()


def supported(config):
    """Whether the conservative margins are calibrated for these scales.

    Evolved/route scales require the explicit `selective_evaluator_enabled`
    opt-in. Enabling NMP or futility without it is a configuration error raised
    by `SearchConfig`, never a silently disabled technique.
    """
    return not unsupported_scales(config)


def distance(a, b):
    return max(abs(a % 9 - b % 9), abs(a // 9 - b // 9))


def guarded(position, depth, budget, *, ignore_own_captures=False):
    """The shared board exclusions. Same charge and same answer as the loops it
    replaces; the scans are vectorised because this runs at every eligible node.

    `ignore_own_captures` drops one clause, and only for a null probe. The guard
    refuses any position where either side has a capture available; that clause
    exists to protect threatened retreats and forced capture and defence
    situations, which is a futility concern, because futility skips one quiet
    move while a tactic is pending. A null probe's risk is the opposite one:
    it is unsound where passing beats every legal move, and a side holding a
    capture is precisely a side that is not in zugzwang. The opponent's captures
    still refuse the probe, because those are the threats a pass declines to
    answer.
    """
    budget.charge(81 + 2 * 648)
    if position.clock + depth + 1 >= 80 or any(n > 1 for n in position.occurrences.values()):
        return False
    if min(sum(position.counts[:3]), sum(position.counts[3:])) < 4:
        return False
    pieces = position.pieces.ravel()
    occupied = np.flatnonzero(pieces)
    owners = pieces[occupied] < 0
    reach = np.where(owners == bool(position.a1), GOAL_DISTANCE[80][occupied],
                     GOAL_DISTANCE[0][occupied])
    if (reach <= max(3, (depth + 1)//2)).any():
        return False
    occupancy = pieces != 0
    for side in (0, 1):
        actions = np.flatnonzero(raw_movement_mask(position.pieces, side))
        if len(actions) < 8:
            return False
        if ignore_own_captures and side == position.side:
            continue
        # Any available capture; this also covers forced captures and
        # threatened retreats.
        if occupancy[DESTINATIONS[actions]].any():
            return False
    return True


def quiet_context(position):
    """The part of `quiet` every candidate move in one position shares.

    Recomputing the fastest own runner and the enemy squares per move made the
    test quadratic in a node's move count for no reason; hoist it instead.
    """
    pieces = position.pieces.ravel()
    occupied = np.flatnonzero(pieces)
    goal = 80 if position.side == position.a1 else 0
    mine = (pieces[occupied] < 0) == bool(position.side)
    return goal, int(GOAL_DISTANCE[goal][occupied[mine]].min()), occupied[~mine]


def blocks(square, piece, goal):
    """Whether `square` lies on some shortest king route from `piece` to `goal`.

    King distance is additive exactly along such a route, so the equality test
    is the whole predicate. An occupied route square is a genuine obstruction
    here: `guarded()` has already refused the node if any capture is available,
    so the runner cannot simply take the blocker.
    """
    return distance(piece, square) + distance(square, goal) == distance(piece, goal)


def quiet(position, action, context=None):
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
    goal, nearest, enemies = context if context is not None else quiet_context(position)
    enemy_goal = 80 - goal
    # Keep every fastest runner, including moves away from its goal.
    if GOAL_DISTANCE[goal][source] <= nearest:
        return False
    for square in enemies:
        square = int(square)
        if min(distance(square, source), distance(square, target)) <= 2:
            return False
        if blocks(source, square, enemy_goal) or blocks(target, square, enemy_goal):
            return False  # leaving or taking a blocking square is defence
    return GOAL_DISTANCE[goal][target] > 3


def allowance(config, position=None, *, material=True, positional=True):
    """One ply of evaluator units, in this configuration's own scales.

    Every margin in the selective family is a multiple of this quantity, so a
    re-tuned genome moves all of them together instead of silently invalidating
    a constant. It is a local heuristic scale, never a bound on future gains.

    The two components are separable because forward futility needs them apart:
    it skips a *quiet* move at a node where `guarded` has already refused every
    available capture, so across that first ply the piece counts cannot change
    and the material term contributes nothing to it. No other margin in the
    family has that guarantee -- razoring and reverse futility are statements
    about a node, not about one quiet child -- so they take the whole quantity.
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
        total = (piece_scale/2 + abs(config.advantage_weight)*advantage_scale) if material else 0.
        if positional:
            for name, scale in (('attack', 3), ('defence', 4), ('overload', 2), ('pressure', 8)):
                if getattr(config, name+'_enabled'):
                    total += scale*abs(getattr(config, name+'_weight'))
        return total

    weight = config.pressure_weight if config.pressure_enabled else 0.
    total = (config.count_weight / 2 + config.advantage_weight) if material else 0.
    return total + (8 * weight if positional else 0.)


def margin(config, depth, position=None):
    """Forward futility: how much a skipped quiet child is allowed to gain.

    Material is charged only from the second ply. The first skipped ply is a
    quiet move at a node where no capture is available to either side, so the
    piece counts across it are fixed and the material term cannot describe any
    part of its change. The original non-experimental branch keeps the whole
    quantity every ply, because its margins are what issue #60 validated.
    """
    if config.selective_evaluator_enabled:
        return config.futility_margin * (
            depth * allowance(config, position, material=False)
            + max(0, depth - 1) * allowance(config, position, positional=False))
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
    """Suspend pruning for a probe or a verification search.

    A null probe searches a hypothetical position with the modelling draw rules
    suspended, so nothing it computes may be stored at all. A verification
    search is ordinary alpha-beta on the real position: keep its results out of
    the selective namespace, where a pruned ancestor could read them, but let
    repeated verifications reuse each other's work instead of re-expanding a
    subtree that was, in the #60 measurements, a quarter of the whole tree.
    """
    old = (player._selective_disabled, player._null_context, player.use_table,
           player._table_namespace)
    player._selective_disabled = True
    player._null_context = null
    player.use_table = player.use_table and not null
    if not null:
        player._table_namespace = b'verified-v1\0'
    try:
        yield
    finally:
        (player._selective_disabled, player._null_context, player.use_table,
         player._table_namespace) = old
