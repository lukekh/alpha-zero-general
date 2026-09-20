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


def quiet(position, action, context=None):
    pieces = position.pieces.ravel()
    source = action // 8
    x, y = action_destination(action)
    target = y * 9 + x
    if pieces[target]:
        return False
    goal, nearest, enemies = context if context is not None else quiet_context(position)
    # Keep every fastest runner, including moves away from its goal.
    if GOAL_DISTANCE[goal][source] <= nearest:
        return False
    if enemies.size and (np.minimum(SQUARE_DISTANCE[enemies, source],
                                    SQUARE_DISTANCE[enemies, target]) <= 2).any():
        return False
    return GOAL_DISTANCE[goal][target] > 3


def margin(config, depth, position=None):
    """The allowance a skipped quiet child is credited with over `depth` plies.

    The experimental branch separates the terms a quiet ply can move from the
    terms it cannot. `guarded` excludes every position where either side has a
    capture available and `quiet` excludes capture moves, so across the first
    skipped ply the piece counts are fixed: material and advantage contribute
    exactly nothing to that ply's change, and are charged only from the second
    ply, where the skipped subtree can capture. The module terms remain the
    conservative per-side feature ranges, which are a bound on the evaluation
    and not on one ply of it, so `futility_margin` is still the tuning knob.
    The original branch is unchanged: its margins are what #60 validated.
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
        material = piece_scale/2 + abs(config.advantage_weight)*advantage_scale
        positional = 0.
        for name, scale in (('attack', 3), ('defence', 4), ('overload', 2), ('pressure', 8)):
            if getattr(config, name+'_enabled'):
                positional += scale*abs(getattr(config, name+'_weight'))
        return config.futility_margin*(depth*positional + max(0, depth - 1)*material)

    weight = config.pressure_weight if config.pressure_enabled else 0.
    return depth * config.futility_margin * (config.count_weight / 2 + config.advantage_weight + 8 * weight)


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
