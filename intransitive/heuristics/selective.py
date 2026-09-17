"""Conservative Intransitive guards shared conceptually with native search."""
from contextlib import contextmanager
import numpy as np
from ..IntransitiveConstants import action_destination
from ..IntransitiveLogicNumba import raw_movement_mask


def supported(config):
    # Evolved/route scales are intentionally unsupported in v1. Configuration
    # remains loadable, but diagnostics report that selection is disabled.
    return (config.count_weight == 100 and config.advantage_weight == 25
            and config.predator_zero_bonus == 1 and config.predator_scarcity_bonus == .5
            and config.prey_bonus == .25
            and not (config.attack_enabled or config.defence_enabled or config.overload_enabled)
            and (not config.pressure_enabled or config.pressure_weight <= 20))


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


def quiet(position, action):
    pieces = position.pieces.ravel()
    source = action // 8
    x, y = action_destination(action)
    target = y * 9 + x
    if pieces[target]:
        return False
    side = position.side
    goal = 80 if side == position.a1 else 0
    own = [int(s) for s in np.flatnonzero(pieces) if int(pieces[s] < 0) == side]
    # Keep every fastest runner, including moves away from its goal.
    if distance(source, goal) <= min(distance(s, goal) for s in own):
        return False
    for square in np.flatnonzero(pieces):
        if int(pieces[square] < 0) != side and min(distance(int(square), source), distance(int(square), target)) <= 2:
            return False
    return distance(target, goal) > 3


def margin(config, depth):
    weight = config.pressure_weight if config.pressure_enabled else 0.
    return depth * config.futility_margin * (config.count_weight / 2 + config.advantage_weight + 8 * weight)


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
