"""Forced corner-run certificate: a sufficient condition, never an estimate.

The bounded terminal search in `proof` sees two plies. A runner that cannot be
stopped often wins further out than that, and saying so is cheap — but only if
the claim is airtight, because a proof result replaces the whole weighted sum
with a mate score and prunes the lines that would have refuted it.

The certificate is built from bounds that can only err toward silence:

* **Enemy arrival is bounded below by free-board Chebyshev distance.** Ignoring
  occupancy can only make an enemy look faster than it is, so a square this
  treats as contested may really be safe, never the reverse.
* **Blocking and capturing are treated alike.** Any enemy able to stand on a
  square in time disqualifies it, not only one that could capture there — a
  piece of any type in the way forces a detour.
* **The runner only walks empty squares**, never capturing its way through, so
  its route cannot depend on an exchange going as hoped.
* **The runner must arrive strictly first**, ahead of any enemy reaching their
  own corner, again on their free-board lower bound.
* **The draw clock must not expire** during the run. The run makes no captures,
  so the no-capture counter runs the whole way; repetition cannot occur because
  the runner's distance to the goal strictly decreases every move.

What it deliberately does not do is claim a position is *not* winnable. A
negative answer means only that this argument did not apply.
"""
import numpy as np
from numba import njit

NO_RUN = -1
UNREACHABLE = 1 << 20


@njit(cache=True, inline='always')
def _arrival(moves, side, turn):
    if moves == 0:
        return 0
    return 2 * moves - (1 if side == turn else 0)


@njit(cache=True)
def _soonest_enemy(pieces, side, turn, enemy_goal):
    """Earliest ply any enemy could occupy each square, and reach its own goal.

    Free-board Chebyshev distance, so occupancy — which can only slow an enemy
    down — is ignored in the runner's favour being *denied*, never granted.
    """
    soonest = np.full(81, UNREACHABLE, dtype=np.int64)
    own_goal_race = UNREACHABLE
    enemy_side = 1 - side
    for y in range(9):
        for x in range(9):
            code = int(pieces[y, x])
            if not code or int(code < 0) != enemy_side:
                continue
            for target in range(81):
                distance = max(abs(target // 9 - y), abs(target % 9 - x))
                ply = _arrival(distance, enemy_side, turn)
                if ply < soonest[target]:
                    soonest[target] = ply
            distance = max(abs(enemy_goal // 9 - y), abs(enemy_goal % 9 - x))
            ply = _arrival(distance, enemy_side, turn)
            if ply < own_goal_race:
                own_goal_race = ply
    return soonest, own_goal_race


@njit(cache=True)
def _safe_run(pieces, source, side, turn, goal, soonest, limit):
    """Plies for this runner to force the goal, or NO_RUN.

    A layer at a time: a square joins the frontier only when every enemy's
    earliest arrival is later than the runner's stay there. The runner holds a
    square from its arrival until its next move, so an enemy arriving one ply
    later could still take or block it; the goal is the exception, because
    reaching it ends the game before any reply.
    """
    # The runner stands on its own square until it first moves. When the
    # opponent moves first, they get a ply to capture it where it sits, so the
    # starting square needs the same clearance as every square on the route.
    if soonest[source] <= _arrival(1, side, turn) - 1:
        return NO_RUN
    step = np.full(81, -1, dtype=np.int64)
    frontier = np.empty(81, dtype=np.int64)
    following = np.empty(81, dtype=np.int64)
    step[source] = 0
    frontier[0] = source
    count = 1
    moves = 0
    while count and moves < limit:
        moves += 1
        ply = _arrival(moves, side, turn)
        found = 0
        for index in range(count):
            square = int(frontier[index])
            x, y = square % 9, square // 9
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if not dx and not dy:
                        continue
                    nx, ny = x + dx, y + dy
                    if nx < 0 or nx > 8 or ny < 0 or ny > 8:
                        continue
                    target = ny * 9 + nx
                    if step[target] >= 0 or pieces[ny, nx] != 0:
                        continue
                    deadline = ply if target == goal else ply + 1
                    if soonest[target] <= deadline:
                        continue
                    step[target] = moves
                    following[found] = target
                    found += 1
            if step[goal] >= 0:
                break
        if step[goal] >= 0:
            return _arrival(int(step[goal]), side, turn)
        for index in range(found):
            frontier[index] = following[index]
        count = found
    return NO_RUN


@njit(cache=True)
def certify(pieces, side, turn, a1, clock_left, limit=20):
    """Plies until `side` forces a corner win, or NO_RUN.

    `clock_left` is the number of plies before a modelling draw would end the
    game; the run makes no captures, so it must finish strictly inside that.
    """
    goals0 = 80 if a1 == 0 else 0
    goal = goals0 if side == 0 else 80 - goals0
    enemy_goal = goals0 if side == 1 else 80 - goals0
    soonest, own_goal_race = _soonest_enemy(pieces, side, turn, enemy_goal)
    best = NO_RUN
    for y in range(9):
        for x in range(9):
            code = int(pieces[y, x])
            if not code or int(code < 0) != side:
                continue
            if y * 9 + x == goal:
                # Already there: the position is terminal and not ours to call.
                continue
            plies = _safe_run(pieces, y * 9 + x, side, turn, goal, soonest, limit)
            if plies == NO_RUN:
                continue
            # A run that arrives second is not a win, and one that outlasts the
            # draw clock is not a win either.
            if plies >= own_goal_race or plies >= clock_left:
                continue
            if best == NO_RUN or plies < best:
                best = plies
    return best


def warm_certificate_kernel():
    board = np.zeros((9, 9, 84), dtype=np.int8)[:, :, 0]
    certify(board, 0, 0, 0, 80)
    certify(board, 1, 0, 0, 80)
