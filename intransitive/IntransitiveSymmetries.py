"""The 12 continuation equivalences, with stable IDs k + 3*d + 6*e.

C cycles rock -> scissors -> paper -> rock. D transposes coordinates.
E reflects in the other diagonal and swaps absolute player labels.
These are not alternative official starting positions or canonicalization.
"""

import numpy as np
from numba import njit

from .IntransitiveConstants import (
    ACTION_SIZE, DIRECTIONS, METADATA_PLANE, META_HISTORY_LENGTH,
    META_HISTORY_PLAYERS, META_NEXT_PLAYER,
)
from .IntransitiveLogicNumba import validate_state

NUM_SYMMETRIES = 12
IDENTITY, C, D, E = 0, 1, 3, 6


@njit(cache=True)
def symmetry_components(symmetry):
    if symmetry != int(symmetry) or not 0 <= symmetry < NUM_SYMMETRIES:
        raise ValueError("Symmetry must be an integer in [0, 12)")
    symmetry = int(symmetry)
    return symmetry % 3, (symmetry // 3) % 2, symmetry // 6


@njit(cache=True)
def symmetry_id(k, d, e):
    """Encode exponents, reducing modulo the generator orders."""
    if k != int(k) or d != int(d) or e != int(e):
        raise ValueError("Symmetry exponents must be integers")
    return int(k) % 3 + 3 * (int(d) % 2) + 6 * (int(e) % 2)


@njit(cache=True)
def compose_symmetries(first, second):
    """Return the ID for applying first, then second."""
    k, d, e = symmetry_components(first)
    l, f, g = symmetry_components(second)
    return symmetry_id(k + l, d + f, e + g)


@njit(cache=True)
def inverse_symmetry(symmetry):
    k, d, e = symmetry_components(symmetry)
    return symmetry_id(-k, d, e)


@njit(cache=True)
def transform_coordinate(x, y, symmetry):
    """Map coordinates, including off-board action destinations."""
    _, d, e = symmetry_components(symmetry)
    if d:
        x, y = y, x
    if e:
        x, y = 8 - y, 8 - x
    return x, y


def _action_tables():
    forward = np.empty((NUM_SYMMETRIES, ACTION_SIZE), dtype=np.int64)
    for symmetry in range(NUM_SYMMETRIES):
        d, e = (symmetry // 3) % 2, symmetry // 6
        for action in range(ACTION_SIZE):
            square, direction = divmod(action, 8)
            y, x = divmod(square, 9)
            dx, dy = DIRECTIONS[direction]
            if d:
                x, y, dx, dy = y, x, dy, dx
            if e:
                x, y, dx, dy = 8 - y, 8 - x, -dy, -dx
            forward[symmetry, action] = 8 * (9 * y + x) + DIRECTIONS.index((dx, dy))
    inverse = np.argsort(forward, axis=1)
    forward.flags.writeable = False
    inverse.flags.writeable = False
    return forward, inverse


# forward[s, a] is the transformed slot; inverse[s, transformed_a] is a.
ACTION_PERMUTATIONS, INVERSE_ACTION_PERMUTATIONS = _action_tables()


@njit(cache=True)
def transform_action(action, symmetry):
    symmetry_components(symmetry)
    if action != int(action) or not 0 <= action < ACTION_SIZE:
        raise ValueError("Action must be an integer in [0, 648)")
    return ACTION_PERMUTATIONS[int(symmetry), int(action)]


@njit(cache=True)
def transform_action_vector(values, symmetry):
    """Map a 648-slot policy or mask by out[p[a]] = values[a]."""
    symmetry_components(symmetry)
    if values.shape != (ACTION_SIZE,):
        raise ValueError("Expected a vector of 648 action slots")
    out = values.copy()
    for action in range(ACTION_SIZE):
        out[ACTION_PERMUTATIONS[int(symmetry), action]] = values[action]
    return out


@njit(cache=True)
def transform_player_vector(values, symmetry):
    """Map absolute [Blue, Red] value or Q vectors; not relative targets."""
    _, _, e = symmetry_components(symmetry)
    if values.shape != (2,):
        raise ValueError("Expected a vector of two absolute player values")
    out = values.copy()
    if e:
        out[0], out[1] = values[1], values[0]
    return out


@njit(cache=True)
def transform_state(state, symmetry):
    """Return an owned full-state transform, preserving history order/padding."""
    k, d, e = symmetry_components(symmetry)
    validate_state(state)
    out = state.copy()
    meta = out[:, :, METADATA_PLANE:]
    length = int(meta.flat[META_HISTORY_LENGTH])
    for plane in range(length + 1):
        for y in range(9):
            for x in range(9):
                tx, ty = x, y
                if d:
                    tx, ty = ty, tx
                if e:
                    tx, ty = 8 - ty, 8 - tx
                piece = int(state[y, x, plane])
                if piece:
                    sign = 1 if piece > 0 else -1
                    piece = sign * (1 + (abs(piece) - 1 + k) % 3)
                    if e:
                        piece = -piece
                out[ty, tx, plane] = piece
    if e:
        meta.flat[META_NEXT_PLAYER] = 1 - meta.flat[META_NEXT_PLAYER]
        for i in range(length):
            meta.flat[META_HISTORY_PLAYERS + i] = 1 - meta.flat[META_HISTORY_PLAYERS + i]
    # D fixes both defended corners. E exchanges them AND their player labels:
    # new A1 defender = swap(old I9 defender) = 1 - (1 - old A1 defender).
    return out
