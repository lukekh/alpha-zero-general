"""Search-owned counts and bounded, order-exact ordinary material reuse.

Counts are immutable: quiet children share the parent tuple; captures replace
one entry. The reference Counter sums in first-occurrence board order, which
can change on a quiet move. Cache the six possible sums for each side rather
than rounding away those observable floating-point differences.
"""
from functools import lru_cache, partial
from itertools import permutations

import numpy as np
from numba import njit

from ..IntransitiveConstants import STATE_SHAPE


BASE = 100.
REG = .25


@njit(cache=True)
def variable_piece_values(own, enemy, linear=False):
    """Per-piece values in ROCK, SCISSORS, PAPER order.

    Counts come from one player's remaining army and its opponent. REG keeps
    extinct predator/prey and absent own types finite; absent types contribute
    nothing when the army total multiplies these values by actual counts.

    `linear` drops the square root from the own-scarcity factor, so a type held
    at half the average army size is worth twice the base rather than about 1.41
    times it. That is a different valuation, not an approximation of the same
    one: it prices concentration far more aggressively, and the two must be
    compared by playing strength rather than assumed interchangeable.
    """
    target = (own[0] + own[1] + own[2]) / 3.
    values = np.empty(3, dtype=np.float64)
    for kind in range(3):
        prey, predator = enemy[(kind + 1) % 3], enemy[(kind + 2) % 3]
        scarcity = (target + REG) / (own[kind] + REG)
        if not linear:
            scarcity = np.sqrt(scarcity)
        values[kind] = BASE * (prey + REG) / (predator + REG) * scarcity
    return values


@njit(cache=True)
def variable_material_total(counts, side, linear=False):
    own, enemy = counts[:3], counts[3:]
    if side:
        own, enemy = enemy, own
    values = variable_piece_values(own, enemy, linear)
    return (own[0] * values[0] + own[1] * values[1]) + own[2] * values[2]


@njit(cache=True)
def count_pieces(state):
    counts = np.zeros(6, dtype=np.int64)
    for y in range(9):
        for x in range(9):
            code = int(state[y, x, 0])
            if code:
                counts[3 * int(code < 0) + abs(code) - 1] += 1
    return (counts[0], counts[1], counts[2], counts[3], counts[4], counts[5])


def after_capture(counts, code):
    if not code:
        return counts
    index = 3 * int(code < 0) + abs(code) - 1
    return counts[:index] + (counts[index] - 1,) + counts[index + 1:]


@njit(cache=True)
def ordered_score(state, side, counts, advantages, count_weight, advantage_weight,
                  variable_material=False, linear_material=False):
    # Encode first occurrences in base four, exactly matching Geometry/Counter.
    first, second, seen = 0, 0, 0
    for y in range(9):
        for x in range(9):
            code = int(state[y, x, 0])
            if not code:
                continue
            kind = abs(code)
            bit = 1 << (3 * int(code < 0) + kind - 1)
            if not seen & bit:
                seen |= bit
                if code > 0:
                    first = 4 * first + kind
                else:
                    second = 4 * second + kind
    own_count = float(counts[0] + counts[1] + counts[2])
    enemy_count = float(counts[3] + counts[4] + counts[5])
    if variable_material:
        own_count = variable_material_total(counts, 0, linear_material) / BASE
        enemy_count = variable_material_total(counts, 1, linear_material) / BASE
    own_adv, enemy_adv = advantages[0, first], advantages[1, second]
    if side:
        own_count, enemy_count = enemy_count, own_count
        own_adv, enemy_adv = enemy_adv, own_adv
    # Keep the reference operation sequence, including zero terms. No fastmath.
    total = 0.
    total += count_weight * (own_count - enemy_count)
    total += 0.
    total += advantage_weight * (own_adv - enemy_adv)
    total += 0.
    total += 0.
    total += -0.
    return total


class MaterialCache:
    def __init__(self, config):
        # Frozen SearchConfig is the complete configuration identity. The cache
        # belongs to this evaluator, never a process-global mutable position.
        self.config = config
        self.values = lru_cache(maxsize=256)(partial(self._values, config=config))

    @staticmethod
    def _values(counts, *, config):
        values = np.zeros((2, 64), dtype=np.float64)
        for side in (0, 1):
            own, enemy = counts[side * 3:side * 3 + 3], counts[(1-side) * 3:(1-side) * 3 + 3]
            contributions = {}
            for kind, count in enumerate(own, 1):
                if count:
                    predator, prey = enemy[(kind + 1) % 3], enemy[kind % 3]
                    contributions[kind] = count * (
                        config.predator_zero_bonus * (predator == 0)
                        + config.predator_scarcity_bonus / (1 + predator)
                        + config.prey_bonus * prey / (1 + prey))
            for order in permutations(contributions):
                key = 0
                for kind in order:
                    key = 4 * key + kind
                values[side, key] = sum(contributions[kind] for kind in order)
        return values

    def score(self, state, side, counts):
        return ordered_score(state, side, counts, self.values(counts),
                             float(self.config.count_weight), float(self.config.advantage_weight),
                             self.config.variable_material_enabled,
                             self.config.variable_material_linear)


@lru_cache(maxsize=1)
def warm_material_kernels():
    state = np.zeros(STATE_SHAPE, dtype=np.int8)
    counts = count_pieces(state)
    ordered_score(state, 0, counts, np.zeros((2, 64)), 100., 25.)
    ordered_score(state, 0, counts, np.zeros((2, 64)), 100., 25., True)
    ordered_score(state, 0, counts, np.zeros((2, 64)), 100., 25., True, True)
