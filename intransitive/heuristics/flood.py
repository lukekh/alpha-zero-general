"""King-move route maps as a bitboard flood fill.

The board is 81 squares in two uint64 lanes, the same layout `moves.py` already
maintains incrementally: lane 0 holds squares 0..63, lane 1 holds 64..80. A BFS
ring is then one dilation — eight masked shifts of the frontier — intersected
with the squares this code may enter and the squares not yet reached. That
replaces a per-square queue whose inner loop re-derives passability for every
neighbour it looks at.

Passability depends only on the moving code, so the six masks are built once per
position and every piece of a code shares them.
"""
import numpy as np
from numba import njit

from ..IntransitiveConstants import DIRECTIONS

LANE_HIGH = np.uint64((1 << 17) - 1)
UNREACHED = np.int16(99)

# EDGES[d] marks the squares from which direction d stays on the board, so a
# shifted frontier can never wrap across a row.
EDGES = np.zeros((8, 2), dtype=np.uint64)
SHIFTS = np.zeros(8, dtype=np.int64)
for _d, (_dx, _dy) in enumerate(DIRECTIONS):
    SHIFTS[_d] = _dx + 9 * _dy
    for _s in range(81):
        if 0 <= _s % 9 + _dx < 9 and 0 <= _s // 9 + _dy < 9:
            EDGES[_d, _s // 64] |= np.uint64(1) << np.uint64(_s % 64)

# De Bruijn sequence for 64-bit trailing-zero count: each isolated low bit maps
# to a distinct top-six-bit product, so one multiply and shift names the square.
_DEBRUIJN = 0x03F79D71B4CB0A89
DEBRUIJN = np.uint64(_DEBRUIJN)
DEBRUIJN_INDEX = np.zeros(64, dtype=np.int64)
for _i in range(64):
    # Built with Python ints: the product wraps by design, which numpy would
    # report as an overflow.
    DEBRUIJN_INDEX[(((1 << _i) * _DEBRUIJN) & ((1 << 64) - 1)) >> 58] = _i
for _table in (EDGES, SHIFTS, DEBRUIJN_INDEX):
    _table.flags.writeable = False


@njit(cache=True, inline='always')
def _shift(low, high, amount):
    """Move a two-lane square set by `amount` squares, without wrapping."""
    if amount > 0:
        places = np.uint64(amount)
        return (low << places,
                ((high << places) | (low >> (np.uint64(64) - places))) & LANE_HIGH)
    places = np.uint64(-amount)
    return ((low >> places) | (high << (np.uint64(64) - places)), high >> places)


@njit(cache=True, inline='always')
def dilate(low, high, edges, shifts):
    """Every square one king step from the given set."""
    out_low, out_high = np.uint64(0), np.uint64(0)
    for direction in range(8):
        a, b = _shift(low & edges[direction, 0], high & edges[direction, 1],
                      shifts[direction])
        out_low |= a
        out_high |= b
    return out_low, out_high


@njit(cache=True)
def passable_masks(pieces):
    """Squares each code may enter: empty, or holding the prey it captures."""
    occupied_low, occupied_high = np.uint64(0), np.uint64(0)
    by_slot = np.zeros((6, 2), dtype=np.uint64)
    for y in range(9):
        for x in range(9):
            code = int(pieces[y, x])
            if not code:
                continue
            square = y * 9 + x
            bit = np.uint64(1) << np.uint64(square % 64)
            lane = square // 64
            by_slot[3 * int(code < 0) + abs(code) - 1, lane] |= bit
            if lane:
                occupied_high |= bit
            else:
                occupied_low |= bit
    empty_low = ~occupied_low
    empty_high = (~occupied_high) & LANE_HIGH
    masks = np.empty((6, 2), dtype=np.uint64)
    for slot in range(6):
        side, kind = slot // 3, slot % 3 + 1
        # This code captures the opposing type one step around the cycle.
        prey = 3 * (1 - side) + kind % 3
        masks[slot, 0] = empty_low | by_slot[prey, 0]
        masks[slot, 1] = empty_high | by_slot[prey, 1]
    return masks


@njit(cache=True)
def flood_into(distance, passable_low, passable_high, source_low, source_high,
               edges, shifts, index):
    """Ring-by-ring distances into a caller-owned row. 99 = unreachable.

    Writing in place keeps `Geometry` from allocating a fresh array per piece
    and copying it into its block, which measured as a third of the fill cost.
    """
    for square in range(81):
        distance[square] = UNREACHED
    frontier_low, frontier_high = source_low, source_high
    seen_low, seen_high = source_low, source_high
    step = np.int16(0)
    while frontier_low or frontier_high:
        bits = frontier_low
        while bits:
            lowest = bits & (~bits + np.uint64(1))
            distance[index[(lowest * DEBRUIJN) >> np.uint64(58)]] = step
            bits ^= lowest
        bits = frontier_high
        while bits:
            lowest = bits & (~bits + np.uint64(1))
            distance[64 + index[(lowest * DEBRUIJN) >> np.uint64(58)]] = step
            bits ^= lowest
        next_low, next_high = dilate(frontier_low, frontier_high, edges, shifts)
        next_low &= passable_low & ~seen_low
        next_high &= passable_high & ~seen_high
        seen_low |= next_low
        seen_high |= next_high
        frontier_low, frontier_high = next_low, next_high
        step += np.int16(1)


@njit(cache=True)
def flood(passable_low, passable_high, source_low, source_high,
          edges, shifts, index):
    """Allocating form, kept for callers that want an owned row."""
    distance = np.empty(81, dtype=np.int16)
    flood_into(distance, passable_low, passable_high, source_low, source_high,
               edges, shifts, index)
    return distance


@njit(cache=True, inline='always')
def _source_bits(square):
    bit = np.uint64(1) << np.uint64(square % 64)
    if square >= 64:
        return np.uint64(0), bit
    return bit, np.uint64(0)


@njit(cache=True)
def route_into(distance, masks, slot, source, removed):
    """Distances for one piece. `removed` frees a square, or is 81 for none."""
    passable_low, passable_high = masks[slot, 0], masks[slot, 1]
    if removed < 81:
        bit = np.uint64(1) << np.uint64(removed % 64)
        if removed // 64:
            passable_high |= bit
        else:
            passable_low |= bit
    low, high = _source_bits(source)
    flood_into(distance, passable_low, passable_high, low, high,
               EDGES, SHIFTS, DEBRUIJN_INDEX)


@njit(cache=True)
def route_distances(masks, slot, source, removed):
    """Allocating form of `route_into`, kept as the tested reference."""
    distance = np.empty(81, dtype=np.int16)
    route_into(distance, masks, slot, source, removed)
    return distance


@njit(cache=True)
def nearest_into(distance, masks, slot, sources, count):
    """Distances from the whole set of pieces of one code, in one fill."""
    low, high = np.uint64(0), np.uint64(0)
    for i in range(count):
        a, b = _source_bits(int(sources[i]))
        low |= a
        high |= b
    flood_into(distance, masks[slot, 0], masks[slot, 1], low, high,
               EDGES, SHIFTS, DEBRUIJN_INDEX)


@njit(cache=True)
def nearest_distances(masks, slot, sources, count):
    """Allocating form of `nearest_into`, kept as the tested reference."""
    distance = np.empty(81, dtype=np.int16)
    nearest_into(distance, masks, slot, sources, count)
    return distance


def warm_flood_kernels():
    board = np.zeros((9, 9, 84), dtype=np.int8)[:, :, 0]
    masks = passable_masks(board)
    row = np.empty(81, dtype=np.int16)
    route_distances(masks, 0, 0, 81)
    route_distances(masks, 0, 0, 40)
    nearest_distances(masks, 0, np.zeros(1, dtype=np.int16), 1)
    route_into(row, masks, 0, 0, 81)
    route_into(row, masks, 0, 0, 40)
    nearest_into(row, masks, 0, np.zeros(1, dtype=np.int16), 1)
