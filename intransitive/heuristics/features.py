"""Compiled positional features over Geometry's stacked route blocks.

`evaluation` keeps the reference implementations; these kernels reproduce them
exactly and `tests/test_features.py` asserts that over a corpus. Two structural
changes make the compiled versions cheaper than a transliteration:

* **Attack only ever matches at distance one.** Both of its tests — a safe step
  on a shortest route, and a safe adjacent capture — require `distances == 1`,
  so walking the eight neighbours answers both and the shortest-path DAG and
  the enemy list never need scanning. The capture half also collapses to a
  boolean: `opportunities` holds one per distinct square under `min(1., sum)`,
  so it is one exactly when some piece has a safe adjacent capture.
* **Defence shares each runner's route across every defender.** The squares,
  deadlines and arrival plies depend only on the runner, and the whole
  defender-by-runner matrix is one pass instead of a Python call per pair.

Both kernels return the work their reference would have charged, so the budget
still sees the same totals. The charge lands once per kernel call rather than
once per square, so a budget that expires mid-module now expires at its end;
the total is unchanged.
"""
import numpy as np
from numba import njit

from ..IntransitiveConstants import DIRECTIONS

# NEIGHBOURS[square, direction] is the square one king step away, or -1 off board.
NEIGHBOURS = np.full((81, 8), -1, dtype=np.int16)
for _square in range(81):
    for _direction, (_dx, _dy) in enumerate(DIRECTIONS):
        _x, _y = _square % 9 + _dx, _square // 9 + _dy
        if 0 <= _x < 9 and 0 <= _y < 9:
            NEIGHBOURS[_square, _direction] = _y * 9 + _x
NEIGHBOURS.flags.writeable = False


@njit(cache=True, inline='always')
def _arrival(moves, side, turn):
    if moves == 0:
        return 0
    return 2 * moves - (1 if side == turn else 0)


@njit(cache=True, inline='always')
def _captures(attacker, defender):
    return attacker * defender < 0 and abs(defender) == abs(attacker) % 3 + 1


@njit(cache=True)
def attack_value(sides, codes, slots, distances, goal_distances, goal_distance,
                 squares, threat, board, neighbours, side, turn):
    """`attacking_position` for one side, with the work its reference charges.

    Returns (value, charge). Charge counts the shortest-route squares the
    reference walks per own piece, so the budget is unaffected by the fact that
    this version only visits the eight neighbours that can ever match.
    """
    progress = 0.
    capture = 0.
    charge = 0
    deadline = _arrival(1, side, turn) + 1
    for piece in range(len(squares)):
        if sides[piece] != side:
            continue
        source = squares[piece]
        total = goal_distance[piece]
        if total != 99:
            # The reference charges one per square of the shortest-path DAG.
            for square in range(81):
                if square != source and distances[piece, square] + goal_distances[piece, square] == total:
                    charge += 1
        stepped = False
        for direction in range(8):
            target = neighbours[source, direction]
            if target < 0 or distances[piece, target] != 1:
                continue
            if threat[slots[piece], target] <= deadline:
                continue
            if total != 99 and goal_distances[piece, target] == total - 1:
                stepped = True
            # Distance one into an occupied square means a capturable enemy:
            # the route map only enters squares this code may take.
            if board[target // 9, target % 9] != 0:
                capture = 1.
        if stepped:
            progress += 1. / (1. + total)
    if progress > 2.:
        progress = 2.
    return progress + capture, charge


@njit(cache=True)
def interception_plies(sides, codes, slots, distances, goal_distances, goal_distance,
                       squares, threat, goals, cached, side, turn):
    """Earliest ply each `side` defender answers each enemy runner, or -1.

    One pass over (runner, defender) pairs replaces a Python call per pair.
    Each runner's interception squares and deadlines are built once and reused
    by every defender that can answer it. Returns (plies, charge); charge skips
    pairs already memoized, matching the reference's per-pair cache.
    """
    count = len(squares)
    plies = np.full((count, count), -1, dtype=np.int16)
    charge = 0
    route = np.empty(82, dtype=np.int16)
    deadlines = np.empty(82, dtype=np.int32)
    for runner in range(count):
        if sides[runner] == side:
            continue
        total = goal_distance[runner]
        if total == 99:
            continue
        runner_side = sides[runner]
        goal = goals[runner_side]
        # The runner's own square first, then its shortest-path DAG in square
        # order, exactly as `route_squares` yields them.
        route[0] = squares[runner]
        deadlines[0] = _arrival(1, runner_side, turn) - 1
        length = 1
        for square in range(81):
            if square == squares[runner]:
                continue
            if distances[runner, square] + goal_distances[runner, square] != total:
                continue
            route[length] = square
            runner_ply = _arrival(distances[runner, square], runner_side, turn)
            deadlines[length] = runner_ply + (1 if square != goal else -1)
            length += 1
        block_deadline = (_arrival(1, runner_side, turn) - 1 if goal == squares[runner]
                          else _arrival(distances[runner, goal], runner_side, turn) - 1)
        for defender in range(count):
            if sides[defender] != side:
                continue
            capture = _captures(codes[defender], codes[runner])
            if not capture and abs(codes[defender]) != abs(codes[runner]):
                continue
            row = slots[defender]
            best = -1
            if capture:
                if not cached[defender, runner]:
                    charge += length
                for index in range(length):
                    square = route[index]
                    limit = deadlines[index]
                    ply = _arrival(distances[defender, square], sides[defender], turn)
                    if ply <= limit and threat[row, square] > (ply if ply > limit else limit):
                        if best < 0 or ply < best:
                            best = ply
            else:
                if not cached[defender, runner]:
                    charge += 1
                ply = _arrival(distances[defender, goal], sides[defender], turn)
                limit = block_deadline
                if ply <= limit and threat[row, goal] > (ply if ply > limit else limit):
                    best = ply
            plies[runner, defender] = best
    return plies, charge


@njit(cache=True)
def defence_value(plies, sides, codes, slots, at, threat, goals, side):
    """`defensive_position` for one side, read straight from the matrix.

    The reference builds a dict of dicts of reply lists and sums it in Python.
    Every scoring caller wants only the earliest ply per defender, so the whole
    reduction happens here and no dictionary is built. Runners and defenders are
    visited in piece order, which is the order the reference's dictionaries were
    inserted in, so the floating-point sum is identical rather than merely close.
    """
    count = len(sides)
    total = 0.
    for runner in range(count):
        if sides[runner] == side:
            continue
        covered = 0.
        for defender in range(count):
            ply = plies[runner, defender]
            if ply >= 0:
                covered += 1. / (1. + ply)
        total += covered if covered < 1. else 1.
    # A safe same-type occupant can close the actual goal completely.
    goal = goals[1 - side]
    blocker = at[goal]
    if blocker >= 0 and sides[blocker] == side and threat[slots[blocker], goal] > 2:
        for index in range(count):
            if sides[index] != side and abs(codes[index]) == abs(codes[blocker]):
                total += 1.
                break
    return total if total < 4. else 4.


@njit(cache=True)
def advantage_value(codes, sides, side, zero_bonus, scarcity_bonus, prey_bonus):
    """`piece_advantage` for one side, from counts alone.

    The reference sums over a Counter, whose order is each kind's first
    appearance on the board, and that order is observable in the last bits of
    the result. This reproduces it rather than summing 1, 2, 3.
    """
    own = np.zeros(4, dtype=np.int64)
    enemy = np.zeros(4, dtype=np.int64)
    order = np.empty(3, dtype=np.int64)
    kinds = 0
    seen = 0
    for index in range(len(codes)):
        kind = abs(codes[index])
        if sides[index] != side:
            enemy[kind] += 1
            continue
        own[kind] += 1
        bit = 1 << kind
        if not seen & bit:
            seen |= bit
            order[kinds] = kind
            kinds += 1
    total = 0.
    for position in range(kinds):
        kind = order[position]
        predator = enemy[(kind + 1) % 3 + 1]
        prey = enemy[kind % 3 + 1]
        total += own[kind] * (zero_bonus * (1. if predator == 0 else 0.)
                              + scarcity_bonus / (1. + predator)
                              + prey_bonus * prey / (1. + prey))
    return total


def warm_feature_kernels():
    """Compile both kernels against the shapes Geometry actually passes."""
    sides = np.zeros(1, dtype=np.int8)
    codes = np.ones(1, dtype=np.int8)
    slots = np.zeros(1, dtype=np.int8)
    distances = np.full((1, 81), 99, dtype=np.int16)
    goal_distances = np.full((1, 81), 99, dtype=np.int16)
    goal_distance = np.full(1, 99, dtype=np.int16)
    squares = np.zeros(1, dtype=np.int16)
    threat = np.zeros((6, 81), dtype=np.int64)
    board = np.zeros((9, 9, 84), dtype=np.int8)[:, :, 0]
    cached = np.zeros((1, 1), dtype=np.bool_)
    goals = np.array([80, 0], dtype=np.int64)
    at = np.full(81, -1, dtype=np.int8)
    attack_value(sides, codes, slots, distances, goal_distances, goal_distance,
                 squares, threat, board, NEIGHBOURS, 0, 0)
    plies, _ = interception_plies(sides, codes, slots, distances, goal_distances,
                                  goal_distance, squares, threat, goals, cached, 0, 0)
    defence_value(plies, sides, codes, slots, at, threat, goals, 0)
    advantage_value(codes, sides, 0, 1., .5, .25)
