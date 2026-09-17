"""Incremental 81-square legality masks stored in two Numba uint64 lanes.

Lane 0 holds squares 0..63; lane 1 holds 64..80. Public rule masks remain
independent and serve as the reference implementation.
"""
import numpy as np
from numba import njit
from ..IntransitiveConstants import DIRECTIONS

EDGES = np.zeros((8, 2), dtype=np.uint64)
for _d, (_dx, _dy) in enumerate(DIRECTIONS):
    for _s in range(81):
        if 0 <= _s % 9 + _dx < 9 and 0 <= _s // 9 + _dy < 9:
            EDGES[_d, _s // 64] |= np.uint64(1) << np.uint64(_s % 64)


@njit(cache=True)
def masks_from_board(pieces):
    masks = np.zeros((6, 2), dtype=np.uint64)
    for s in range(81):
        code = int(pieces.flat[s])
        if code:
            masks[3 * int(code < 0) + abs(code) - 1, s // 64] |= np.uint64(1) << np.uint64(s % 64)
    return masks


@njit(cache=True)
def move_board(pieces, masks, source, target, captured, undo=False):
    mover = int(pieces.flat[target if undo else source])
    kind = 3 * int(mover < 0) + abs(mover) - 1
    masks[kind, source // 64] ^= np.uint64(1) << np.uint64(source % 64)
    masks[kind, target // 64] ^= np.uint64(1) << np.uint64(target % 64)
    if captured:
        kind = 3 * int(captured < 0) + abs(captured) - 1
        masks[kind, target // 64] ^= np.uint64(1) << np.uint64(target % 64)
    pieces.flat[source] = mover if undo else 0
    pieces.flat[target] = captured if undo else mover


@njit(cache=True, inline='always')
def empty_squares(masks):
    low, high = np.uint64(0), np.uint64(0)
    for k in range(6):
        low |= masks[k, 0]
        high |= masks[k, 1]
    return ~low, ~high & np.uint64((1 << 17) - 1)


@njit(cache=True, inline='always')
def sources(masks, side, d, low, high):
    result0, result1 = np.uint64(0), np.uint64(0)
    shift = int(DIRECTIONS[d][0]) + 9 * int(DIRECTIONS[d][1])
    for k in range(3):
        prey = 3 * (1 - side) + (k + 1) % 3
        a, b = low | masks[prey, 0], high | masks[prey, 1]
        # Inverse-shift targets to retain sources and direction IDs.
        if shift > 0:
            a, b = (a >> shift) | (b << (64 - shift)), b >> shift
        else:
            shift_left = -shift
            a, b = a << shift_left, (b << shift_left) | (a >> (64 - shift_left))
        result0 |= a & masks[3 * side + k, 0] & EDGES[d, 0]
        result1 |= b & masks[3 * side + k, 1] & EDGES[d, 1]
    return result0, result1


@njit(cache=True)
def has_move(masks, side):
    low, high = empty_squares(masks)
    for d in range(8):
        a, b = sources(masks, side, d, low, high)
        if a or b:
            return True
    return False


@njit(cache=True, inline='always')
def population(bits):
    count = 0
    while bits:
        bits &= bits - np.uint64(1)
        count += 1
    return count


@njit(cache=True)
def legal_actions(masks, side):
    low, high = empty_squares(masks)
    by_direction = np.empty((8, 2), dtype=np.uint64)
    union0, union1 = np.uint64(0), np.uint64(0)
    size = 0
    for d in range(8):
        a, b = sources(masks, side, d, low, high)
        size += population(a) + population(b)
        by_direction[d, 0], by_direction[d, 1] = a, b
        union0 |= a
        union1 |= b
    # Preserve the reference's source-then-direction action ordering.
    actions = np.empty(size, dtype=np.int64)
    count = 0
    for s in range(81):
        bit = np.uint64(1) << np.uint64(s % 64)
        if (union0 if s < 64 else union1) & bit:
            for d in range(8):
                if by_direction[d, s // 64] & bit:
                    actions[count] = 8 * s + d
                    count += 1
    return actions[:count]
