"""Occupied-board king routes; estimates, never forced-win certificates."""
from dataclasses import dataclass
import numpy as np
from numba import njit
from ..IntransitiveConstants import DIRECTIONS, META_A1_DEFENDER, META_NEXT_PLAYER


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
        self.pieces = []
        for square in np.flatnonzero(self.board.ravel()):
            budget.charge(162 if routes else 1)
            square = int(square)
            code = int(self.board.flat[square])
            side = int(code < 0)
            goal = self.goals[side]
            distances = distance_map(self.board, square, code) if routes else None
            reverse = distance_map(self.board, goal, code, removed=square) if routes else None
            self.pieces.append(Piece(square, code, side, distances, reverse, goal))
        self.by_square = {p.square: p for p in self.pieces}
        self.by_side = tuple([p for p in self.pieces if p.side == side] for side in (0, 1))
        # A Geometry describes one immutable position. These answers depend only
        # on that position, not on any search bounds or repetition history.
        self._route_squares = {}
        self._safety = {}
        self._interceptions = {}

    def own(self, side):
        return self.by_side[side]

    def safe(self, piece, square, ply):
        # Conservative exposure estimate, including predators elsewhere on board.
        key = (piece.square, square, ply)
        if key not in self._safety:
            self._safety[key] = not any(captures(enemy.code, piece.code)
                and arrival(int(enemy.distances[square]), enemy.side, self.turn) <= ply
                for enemy in self.own(1 - piece.side))
        return self._safety[key]

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
        squares = [runner.square] + self.route_squares(runner) if capture else [runner.goal]
        found = []
        for square in squares:
            self.budget.charge()
            rd = int(runner.distances[square])
            dd = int(defender.distances[square])
            runner_ply = arrival(rd, runner.side, self.turn)
            defender_ply = arrival(dd, defender.side, self.turn)
            # Capture may occur on the reply to arrival, except at winning goal.
            deadline = (arrival(1, runner.side, self.turn) - 1 if square == runner.square
                        else runner_ply + (1 if capture and square != runner.goal else -1))
            if defender_ply <= deadline and self.safe(defender, square, max(defender_ply, deadline)):
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
