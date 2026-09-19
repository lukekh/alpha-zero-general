"""Occupied-board king routes; estimates, never forced-win certificates."""
from dataclasses import dataclass
import numpy as np
from numba import njit
from ..IntransitiveConstants import DIRECTIONS, META_A1_DEFENDER, META_NEXT_PLAYER
from .flood import nearest_into, passable_masks, route_into

# Larger than any reachable arrival (a 99-move distance arrives by ply 198), so
# a piece whose predators are all captured stays safe at every deadline.
NO_PREDATOR = 1 << 30


@njit(cache=True)
def distance_map(pieces, source, piece, removed=-1):
    """Static occupancy, allowing only this mover's captures. 99 = unreachable."""
    distance = np.full(81, 99, dtype=np.int16)
    queue = np.empty(81, dtype=np.int16)
    distance[source] = 0
    queue[0] = source
    head, tail = 0, 1
    while head < tail:
        square = int(queue[head])
        head += 1
        x, y = square % 9, square // 9
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < 9 and 0 <= ny < 9):
                continue
            target = ny * 9 + nx
            occupant = int(pieces[ny, nx]) if target != removed else 0
            if distance[target] != 99 or (occupant and not (
                    occupant * piece < 0 and abs(occupant) == abs(piece) % 3 + 1)):
                continue
            distance[target] = distance[square] + 1
            queue[tail] = target
            tail += 1
    return distance


@njit(cache=True)
def nearest_distance(pieces, sources, count, code):
    """Min moves for ANY piece of `code` to reach each square. 99 = unreachable.

    The traversal rule depends only on the moving code, never on which piece is
    moving, so every piece of that code shares one frontier and the minimum
    over all of them falls out of a single pass. Source squares hold pieces of
    this code, so they are already occupied and are never re-entered; paths
    emanate from each source rather than through its neighbours.
    """
    distance = np.full(81, 99, dtype=np.int16)
    queue = np.empty(81, dtype=np.int16)
    head, tail = 0, 0
    for index in range(count):
        square = int(sources[index])
        if distance[square] == 99:
            distance[square] = 0
            queue[tail] = square
            tail += 1
    while head < tail:
        square = int(queue[head])
        head += 1
        x, y = square % 9, square // 9
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < 9 and 0 <= ny < 9):
                continue
            target = ny * 9 + nx
            occupant = int(pieces[ny, nx])
            if distance[target] != 99 or (occupant and not (
                    occupant * code < 0 and abs(occupant) == abs(code) % 3 + 1)):
                continue
            distance[target] = distance[square] + 1
            queue[tail] = target
            tail += 1
    return distance


def code_slot(code):
    """Dense 0..5 index for a piece code, matching `masks_from_board`."""
    return 3 * int(code < 0) + abs(code) - 1


def predator_code(code):
    """The single code that captures `code`: paper<-rock<-scissors<-paper."""
    return (abs(code) - 1 or 3) * (-1 if code > 0 else 1)


def captures(attacker, defender):
    return attacker * defender < 0 and abs(defender) == abs(attacker) % 3 + 1


def arrival(moves, side, turn):
    return 0 if moves == 0 else 2 * int(moves) - int(side == turn)


@dataclass
class Piece:
    square: int
    code: int
    side: int
    distances: np.ndarray | None
    goal_distances: np.ndarray | None
    goal: int

    @property
    def distance(self):
        return int(self.distances[self.goal])


class Geometry:
    def __init__(self, state, budget, *, routes=True):
        self.state, self.budget = state, budget
        self.board = state[:, :, 0]
        self.turn = int(state[:, :, 82:84].flat[META_NEXT_PLAYER])
        a1 = int(state[:, :, 82:84].flat[META_A1_DEFENDER])
        self.goals = (80, 0) if a1 == 0 else (0, 80)
        occupied = np.flatnonzero(self.board.ravel())
        count = len(occupied)
        # Route maps live in two (pieces, 81) blocks rather than one array per
        # piece. Each Piece keeps a row view, so attribute access is unchanged,
        # and the compiled feature kernels read the blocks with no repacking.
        self.squares = np.empty(count, dtype=np.int16)
        self.codes = np.empty(count, dtype=np.int8)
        self.slots = np.empty(count, dtype=np.int8)
        self.sides = np.empty(count, dtype=np.int8)
        self.distances = np.full((count, 81), 99, dtype=np.int16) if routes else None
        self.goal_distances = np.full((count, 81), 99, dtype=np.int16) if routes else None
        self.goal_distance = np.full(count, 99, dtype=np.int16)
        # Passability depends only on the moving code, so the six masks are
        # built once and every flood fill below shares them.
        self.passable = passable_masks(self.board) if routes else None
        # `at[square]` is the piece index, or -1. An array rather than a dict:
        # the kernels and the scoring path index it, and only the diagnostic
        # paths below ever need `Piece` objects at all.
        self.at = np.full(81, -1, dtype=np.int8)
        for index, square in enumerate(occupied):
            budget.charge(162 if routes else 1)
            square = int(square)
            code = int(self.board.flat[square])
            side = int(code < 0)
            self.squares[index], self.codes[index] = square, code
            self.slots[index], self.sides[index] = code_slot(code), side
            self.at[square] = index
            if routes:
                slot = code_slot(code)
                goal = self.goals[side]
                route_into(self.distances[index], self.passable, slot, square, 81)
                route_into(self.goal_distances[index], self.passable, slot, goal, square)
                self.goal_distance[index] = self.distances[index][goal]
        self._pieces = None
        self._by_square = None
        self._by_side = None
        # A Geometry describes one immutable position. These answers depend only
        # on that position, not on any search bounds or repetition history.
        self._route_squares = {}
        self._interceptions = {}
        # Built on demand, per piece code rather than per piece: see
        # `threat_map` and `profile`. A sparse board never pays for maps or
        # route profiles nothing asks about.
        self._threat = {}
        self._profiles = {}
        self._sources = None
        # One row per piece code, filled on demand; `slots` indexes into it, so
        # a kernel takes the whole block and never rebuilds per-piece copies.
        self._threat_rows = np.zeros((6, 81), dtype=np.int64)
        self._threat_ready = False
        self._scratch = np.empty(81, dtype=np.int16)
        self._plies = {}

    @property
    def pieces(self):
        """Piece objects, built on first use.

        The compiled feature kernels read the stacked blocks directly, so a
        scored leaf never materializes these; the reference implementations,
        attribution and race diagnostics do.
        """
        if self._pieces is None:
            routes = self.distances is not None
            self._pieces = [
                Piece(int(self.squares[i]), int(self.codes[i]), int(self.sides[i]),
                      self.distances[i] if routes else None,
                      self.goal_distances[i] if routes else None,
                      self.goals[int(self.sides[i])])
                for i in range(len(self.squares))]
        return self._pieces

    @property
    def by_square(self):
        if self._by_square is None:
            self._by_square = {p.square: p for p in self.pieces}
        return self._by_square

    @property
    def by_side(self):
        if self._by_side is None:
            self._by_side = tuple([p for p in self.pieces if p.side == side]
                                  for side in (0, 1))
        return self._by_side

    @property
    def index(self):
        """Square to piece index. Prefer `at`, which needs no dict."""
        return {int(square): i for i, square in enumerate(self.squares)}

    def own(self, side):
        return self.by_side[side]

    def threat_map(self, code):
        """Earliest ply a predator of `code` can reach each square.

        One multi-source search per predator type replaces rescanning every
        enemy for every piece, square and deadline. `arrival` is monotone in
        distance, so the minimum over predators of their arrival is the arrival
        of the nearest one, and every piece sharing a code shares this map.
        `NO_PREDATOR` keeps a piece with no surviving predator safe at any ply,
        matching the empty-scan result it replaces.

        Not charged to the budget. At most one map exists per distinct code on
        the board, so this costs at most 81*min(6, pieces) work against the
        162*pieces the piece index has already charged; it replaces the
        per-enemy scans in `safe`, which were themselves never charged. Charging
        it would shrink the search a fixed budget can afford.
        """
        threat = self._threat.get(code)
        if threat is not None:
            return threat
        slot = code_slot(code)
        if self._sources is None:
            self._sources = {}
            for index in range(len(self.squares)):
                self._sources.setdefault(int(self.codes[index]), []).append(
                    int(self.squares[index]))
        squares = self._sources.get(predator_code(code))
        if not squares:
            self._threat_rows[slot] = NO_PREDATOR
        else:
            predator = predator_code(code)
            nearest_into(self._scratch, self.passable, code_slot(predator),
                         np.array(squares, dtype=np.int16), len(squares))
            distances = self._scratch.astype(np.int64)
            # arrival(), vectorised over the whole board.
            self._threat_rows[slot] = np.where(
                distances == 0, 0, 2 * distances - int((1 - int(code < 0)) == self.turn))
        threat = self._threat_rows[slot]
        self._threat[code] = threat
        return threat

    def plies(self, side):
        """Earliest interception ply for every (runner, defender) pair, or -1.

        `coverage` and `defensive_position` both read this instead of calling
        `intercepts` once per pair. It is memoized per defending side and
        charges only pairs `intercepts` has not already memoized, so the budget
        sees the same total either way. The charge lands once rather than once
        per square, so an expiring budget now stops at the end of the pass.
        """
        from .features import interception_plies
        cached = self._plies.get(side)
        if cached is not None:
            self.budget.check()
            return cached
        count = len(self.squares)
        memoized = np.zeros((count, count), dtype=np.bool_)
        for (defender_square, runner_square) in self._interceptions:
            memoized[self.at[defender_square], self.at[runner_square]] = True
        plies, charge = interception_plies(
            self.sides, self.codes, self.slots, self.distances, self.goal_distances,
            self.goal_distance, self.squares, self.threat_block(),
            np.array(self.goals, dtype=np.int64), memoized, side, self.turn)
        self.budget.charge(int(charge)) if charge else self.budget.check()
        self._plies[side] = plies
        return plies

    def threat_block(self):
        """Fill and return every occupied code's row, for the kernels.

        Memoized: the kernels ask for this once per module per side, and
        re-deriving the set of occupied codes each time was measurably more
        expensive than the fills it guards.
        """
        if not self._threat_ready:
            for code in dict.fromkeys(int(value) for value in self.codes):
                self.threat_map(code)
            self._threat_ready = True
        return self._threat_rows

    def safe(self, piece, square, ply):
        # Conservative exposure estimate, including predators elsewhere on board.
        return int(self.threat_map(piece.code)[square]) > ply

    def profile(self, runner):
        """This runner's interception squares, deadlines and arrival plies.

        All three depend only on the runner, so they are built once instead of
        once per defender examining the same route.
        """
        cached = self._profiles.get(runner.square)
        if cached is not None:
            return cached
        squares = [runner.square] + self.route_squares(runner)
        deadlines, plies = [], []
        for square in squares:
            runner_ply = arrival(int(runner.distances[square]), runner.side, self.turn)
            plies.append(runner_ply)
            # Capture may occur on the reply to arrival, except at winning goal.
            deadlines.append(arrival(1, runner.side, self.turn) - 1 if square == runner.square
                             else runner_ply + (1 if square != runner.goal else -1))
        cached = (squares, deadlines, plies)
        self._profiles[runner.square] = cached
        return cached

    def route_squares(self, runner):
        if runner.distance == 99:
            return []
        if runner.square not in self._route_squares:
            self._route_squares[runner.square] = [int(s) for s in np.flatnonzero(
                runner.distances + runner.goal_distances == runner.distance)
                    if int(s) != runner.square]
        return self._route_squares[runner.square]

    def intercepts(self, defender, runner):
        """Timely capture on a shortest route, or a safe same-type goal hold.

        A route intersection is potential coverage, not universal defence. A
        same-type piece earns blocking credit only at the unavoidable goal.
        """
        if runner.distance == 99:
            return []
        capture = captures(defender.code, runner.code)
        if not capture and abs(defender.code) != abs(runner.code):
            return []
        key = (defender.square, runner.square)
        if key in self._interceptions:
            self.budget.check()
            return self._interceptions[key]
        if capture:
            squares, deadlines, plies = self.profile(runner)
        else:
            # A block is only ever credited at the goal, one ply before arrival.
            # The runner-square case cannot arise in a live position, but keep
            # the original conditional so the two paths cannot drift.
            goal = runner.goal
            runner_ply = arrival(int(runner.distances[goal]), runner.side, self.turn)
            deadline = (arrival(1, runner.side, self.turn) - 1 if goal == runner.square
                        else runner_ply - 1)
            squares, deadlines, plies = [goal], [deadline], [runner_ply]
        # `safe` inlined, with the defender's map hoisted out of the loop: the
        # comparison below is exactly safe(defender, square, max(ply, deadline)).
        threat = self.threat_map(defender.code)
        found = []
        for square, deadline, runner_ply in zip(squares, deadlines, plies):
            self.budget.charge()
            defender_ply = arrival(int(defender.distances[square]), defender.side, self.turn)
            if defender_ply <= deadline and int(threat[square]) > max(defender_ply, deadline):
                found.append(dict(square=square, ply=defender_ply,
                                  runner_ply=runner_ply, kind='capture' if capture else 'block'))
        self._interceptions[key] = found
        return found

    def route(self, runner):
        """One illustrative shortest route; scoring uses the whole shortest DAG."""
        if runner.distance == 99:
            return []
        current, result = runner.square, [runner.square]
        while current != runner.goal:
            candidates = [s for s in self.route_squares(runner)
                          if runner.distances[s] == runner.distances[current] + 1
                          and max(abs(s % 9 - current % 9), abs(s // 9 - current // 9)) == 1]
            current = min(candidates)
            result.append(current)
        return result
