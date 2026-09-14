"""Full-width iterative alpha-beta and an independent exhaustive reference.

Search organization reference: Stockfish src/search.cpp (iterative deepening,
TT bounds, mate-distance conversion). No selective/chess-specific pruning.
"""
from dataclasses import dataclass, field
from math import inf
from time import perf_counter
import sys
import numpy as np
from .budget import Budget, BudgetExpired
from .config import SearchConfig
from .evaluation import Evaluator, MATE, MATE_THRESHOLD, terminal_value
from ..IntransitiveConstants import action_destination


def to_table(score, ply):
    return score + ply if score > MATE_THRESHOLD else score - ply if score < -MATE_THRESHOLD else score


def from_table(score, ply):
    return score - ply if score > MATE_THRESHOLD else score + ply if score < -MATE_THRESHOLD else score


@dataclass
class Entry:
    depth: int
    score: float
    bound: str
    best_move: int
    pv: tuple


@dataclass
class SearchResult:
    action: int
    score: float | None
    completed_depth: int
    pv: list
    nodes: int
    work: int
    proof_nodes: int
    elapsed: float
    stopped: bool
    table_bytes: int = 0
    explanation: dict = field(default_factory=dict)
    module_seconds: dict = field(default_factory=dict)
    module_calls: dict = field(default_factory=dict)


class ProofLimit(Exception):
    pass


def prove(game, state, config, budget):
    """Terminal-only bounded adversarial search; exhausted/horizon = unknown.

    Unknown leaves have neutral utility only inside this proof search. They can
    never become exact draws or ordinary heuristic cutoffs. A nonzero mate score
    means every relevant defence has been visited, including counter-races.
    """
    used = 0

    def visit(position, depth, ply, alpha, beta):
        nonlocal used
        if used >= config.proof_nodes:
            raise ProofLimit
        used += 1
        budget.visit(proof=True)
        side = int(position[:, :, 32].flat[1])
        terminal = terminal_value(game, position, side, ply)
        if terminal is not None:
            return terminal, []
        if depth == 0:
            return 0., []
        best, pv = -inf, []
        actions = list(map(int, np.flatnonzero(game.getValidMoves(position, side))))
        goal = 80 if side == int(position[:, :, 32].flat[2]) else 0
        actions.sort(key=lambda a: action_destination(a) != (goal % 9, goal // 9))
        for action in actions:
            budget.charge()
            child, _ = game.getNextState(position, side, action)
            value, line = visit(child, depth - 1, ply + 1, -beta, -alpha)
            value = -value
            if value > best:
                best, pv = value, [action] + line
            alpha = max(alpha, value)
            if alpha >= beta or best == MATE - ply - 1:
                break
        return best, pv

    if config.proof_depth == 0 or config.proof_nodes == 0:
        return {'status': 'unknown', 'reason': 'disabled', 'nodes': 0}
    start = perf_counter()
    try:
        score, pv = visit(state, config.proof_depth, 0, -inf, inf)
        if abs(score) > MATE_THRESHOLD:
            return dict(status='proven', score=score, pv=pv, plies=int(MATE - abs(score)), nodes=used)
        return dict(status='unknown', reason='horizon', nodes=used)
    except ProofLimit:
        return dict(status='unknown', reason='proof budget', nodes=used)
    finally:
        budget.module_seconds['proof'] += perf_counter() - start
        budget.module_calls['proof'] += 1


class AlphaBetaPlayer:
    label = 'Alpha–beta'
    kind = 'alphabeta'

    def __init__(self, game=None, config=None, use_table=True):
        if game is None:
            from ..IntransitiveGame import IntransitiveGame
            game = IntransitiveGame()
        self.game = game
        self.config = config or SearchConfig()
        self.use_table = use_table
        self.table = {}
        self._identity = None
        self.last_result = None

    def reload(self):
        self.table.clear()
        self.last_result = None

    def _prepare(self):
        identity = self.config.identity()
        if identity != self._identity:
            self.table.clear()
            self._identity = identity
        self.evaluator = Evaluator(self.game, self.config)

    def _leaf(self, state, side, ply, budget):
        proof = prove(self.game, state, self.config, budget)
        if proof['status'] == 'proven':
            return from_table(proof['score'], ply), proof['pv']
        explanation = self.evaluator.explain(state, side, budget)
        return explanation['score'], []

    def _ordered(self, state, side, preferred, budget):
        actions = list(map(int, np.flatnonzero(self.game.getValidMoves(state, side))))
        goal = 80 if side == int(state[:, :, 32].flat[2]) else 0
        own_goal = 80 - goal
        threats = []
        for y, x in np.argwhere(state[:, :, 0] * (1 if side == 0 else -1) < 0):
            if max(abs(x - own_goal % 9), abs(y - own_goal // 9)) <= 1:
                threats.append(int(y) * 9 + int(x))
        children = []
        for action in actions:
            budget.charge()
            child, _ = self.game.getNextState(state, side, action)
            win = terminal_value(self.game, child, side) == MATE
            children.append((action, child, win))

        def rank(item):
            action, child, win = item
            budget.charge()
            dx, dy = action_destination(action)
            dest = dy * 9 + dx
            return (win, dest in threats or dest == own_goal,
                    action == preferred, state[dy, dx, 0] != 0,
                    -max(abs(dx - goal % 9), abs(dy - goal // 9)), -action)
        return [(action, child) for action, child, _ in sorted(children, key=rank, reverse=True)]

    def _search(self, state, depth, alpha, beta, ply, budget):
        budget.visit()
        side = int(state[:, :, 32].flat[1])
        terminal = terminal_value(self.game, state, side, ply)
        if terminal is not None:
            return terminal, []
        if depth == 0:
            return self._leaf(state, side, ply, budget)
        # Exact bytes include turn, goal owner, capture clock and every history
        # plane. Different-depth heuristic results are only ordering hints.
        key = state.tobytes()
        entry = self.table.get((key, depth)) if self.use_table else None
        alpha_original, beta_original = alpha, beta
        if entry is not None:
            value = from_table(entry.score, ply)
            if entry.bound == 'exact':
                return value, list(entry.pv)
            if entry.bound == 'lower':
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if alpha >= beta:
                return value, list(entry.pv)
        hint = entry or (self.table.get((key, depth - 1)) if self.use_table else None)
        preferred = hint.best_move if hint else None
        best, pv = -inf, []
        for action, child in self._ordered(state, side, preferred, budget):
            value, line = self._search(child, depth - 1, -beta, -alpha, ply + 1, budget)
            value = -value
            if value > best:
                best, pv = value, [action] + line
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        budget.check()
        if self.use_table and self.config.table_entries:
            bound = 'upper' if best <= alpha_original else 'lower' if best >= beta_original else 'exact'
            if ((key, depth) not in self.table
                    and len(self.table) >= self.config.table_entries):
                self.table.pop(next(iter(self.table)))
            self.table[(key, depth)] = Entry(depth, to_table(best, ply), bound, pv[0], tuple(pv))
        return best, pv

    def analyze(self, state, budget=None):
        self._prepare()
        budget = budget or Budget(self.config.node_limit, self.config.time_limit)
        side = int(state[:, :, 32].flat[1])
        # Validation/one legal fallback is required even for a zero budget.
        if terminal_value(self.game, state, side) is not None:
            raise ValueError('Cannot select a move from a terminal position')
        actions = np.flatnonzero(self.game.getValidMoves(state, side))
        action, score, depth, pv = int(actions[0]), None, 0, [int(actions[0])]
        explanation = {'status': 'budget exhausted before evaluation', 'proof': {'status': 'unknown'}}
        stopped = False
        try:
            explanation = self.evaluator.explain(state, side, budget)
            for target in range(1, self.config.max_depth + 1):
                value, line = self._search(state, target, -inf, inf, 0, budget)
                budget.check()
                score, pv, action, depth = value, line, line[0], target
                if abs(score) > MATE_THRESHOLD:
                    break
        except BudgetExpired:
            stopped = True
        explanation['config'] = self.config.to_dict()
        explanation['pv'] = pv
        explanation['search_score'] = score
        if score is not None and abs(score) > MATE_THRESHOLD:
            explanation['proof'] = dict(status='proven', plies=int(MATE - abs(score)),
                                        winner=side if score > 0 else 1 - side)
        table_bytes = sys.getsizeof(self.table) + sum(
            sys.getsizeof(key) + sys.getsizeof(key[0]) + sys.getsizeof(entry)
            + sys.getsizeof(entry.pv) for key, entry in self.table.items())
        result = SearchResult(action, score, depth, pv, budget.nodes, budget.work,
                              budget.proof_nodes, budget.clock() - budget.start, stopped, table_bytes,
                              explanation, dict(budget.module_seconds), dict(budget.module_calls))
        self.last_result = result
        return result

    def choose(self, state, player):
        if player != int(state[:, :, 32].flat[1]):
            raise ValueError('Player does not match state')
        return self.analyze(state).action

    def play(self, board, nb_moves=0):
        if int(board[:, :, 32].flat[1]) != 0:
            raise ValueError('Expected canonical current player zero')
        return self.analyze(board).action


def exhaustive_minimax(game, state, depth, config=None, budget=None):
    """Independent max/min reference, with the same horizon evaluator/proof."""
    config = config or SearchConfig()
    budget = budget or Budget(10**12, 3600.)
    evaluator = Evaluator(game, config)
    root = int(state[:, :, 32].flat[1])

    def visit(position, remaining, ply):
        budget.visit()
        side = int(position[:, :, 32].flat[1])
        terminal = terminal_value(game, position, root, ply)
        if terminal is not None:
            return terminal, []
        if remaining == 0:
            proof = prove(game, position, config, budget)
            if proof['status'] == 'proven':
                value = from_table(proof['score'], ply)
                return value if side == root else -value, proof['pv']
            return evaluator.explain(position, root, budget)['score'], []
        best, pv = (-inf if side == root else inf), []
        for action in map(int, np.flatnonzero(game.getValidMoves(position, side))):
            budget.charge()
            child, _ = game.getNextState(position, side, action)
            value, line = visit(child, remaining - 1, ply + 1)
            if (value > best if side == root else value < best):
                best, pv = value, [action] + line
        return best, pv

    return visit(state, depth, 0)
