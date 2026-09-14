"""Full-width iterative alpha-beta and an independent exhaustive reference.

Search organization reference: Stockfish src/search.cpp (iterative deepening,
TT bounds, mate-distance conversion). No selective/chess-specific pruning.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from math import inf
from time import perf_counter
import sys
import numpy as np
from .budget import Budget, BudgetExpired
from .config import SearchConfig
from .evaluation import Evaluator, MATE, MATE_THRESHOLD, terminal_value
from .kernels import no_terminal_win_in_horizon, winning_actions, warm_search_kernels
from ..IntransitiveConstants import action_destination


@lru_cache(maxsize=1)
def warm_route_kernels():
    from .geometry import distance_map
    board = np.zeros((9, 9, 33), dtype=np.int8)[:, :, 0]
    distance_map(board, 0, 1)
    distance_map(board, 0, 1, 1)


def position_key(state):
    """Exact future-play identity, independent of move number/history order.

    Threefold depends on counts of board+turn occurrences since the last
    capture, not their order. Keep those counts, goal ownership and the draw
    clock. Never merge by board alone or discard a mere second occurrence.
    Bytes are compared exactly by dict, so hash collisions cannot merge nodes.
    """
    meta = state[:, :, 32].ravel()
    length = int(meta[4])
    # Copy the strided planes once, rather than once per historical board.
    planes = state[:, :, :length + 1].transpose(2, 0, 1).tobytes()
    occurrences = {}
    for index in range(length):
        item = planes[(index + 1) * 81:(index + 2) * 81] + bytes((int(meta[10 + index]),))
        occurrences[item] = occurrences.get(item, 0) + 1
    return (planes[:81] + bytes(map(int, meta[1:4])) +
            b''.join(item + bytes((count,)) for item, count in sorted(occurrences.items())))


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
    selected_depth: int = 0
    partial_depth: int = 0
    root_moves_completed: int = 0
    root_moves_total: int = 0
    selection_source: str = 'completed_iteration'
    score_bound: str | None = None
    tt_hits: int = 0


@dataclass
class RootProgress:
    depth: int
    total: int
    incumbent: int | None
    moves: dict = field(default_factory=dict)
    finished: bool = False

    def record(self, action, score, line, alpha):
        self.moves[action] = dict(action=action, score=score, pv=[action] + line,
                                  depth=self.depth,
                                  bound='upper' if score <= alpha else 'exact')


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
        if ply == 0:
            budget.charge()
            if no_terminal_win_in_horizon(position[:, :, 0], side,
                                           int(position[:, :, 32].flat[2]), depth):
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
        self._hints = {}
        self._identity = None
        self.last_result = None
        self._root_progress = None
        self._root_previous = {}

    def reload(self):
        self.table.clear()
        self._hints.clear()
        self.last_result = None

    def _prepare(self):
        warm_search_kernels()
        if self.config.attack_enabled or self.config.defence_enabled or self.config.overload_enabled:
            warm_route_kernels()
        identity = self.config.identity()
        if identity != self._identity:
            self.table.clear()
            self._hints.clear()
            self._identity = identity
        self.evaluator = Evaluator(self.game, self.config)

    def _leaf(self, state, side, ply, budget):
        proof = prove(self.game, state, self.config, budget)
        if proof['status'] == 'proven':
            return from_table(proof['score'], ply), proof['pv']
        return self.evaluator.score(state, side, budget, proof=proof), []

    def _ordered(self, state, side, preferred, budget, root=False):
        actions = list(map(int, np.flatnonzero(self.game.getValidMoves(state, side))))
        goal = 80 if side == int(state[:, :, 32].flat[2]) else 0
        own_goal = 80 - goal
        threats = []
        for y, x in np.argwhere(state[:, :, 0] * (1 if side == 0 else -1) < 0):
            if max(abs(x - own_goal % 9), abs(y - own_goal // 9)) <= 1:
                threats.append(int(y) * 9 + int(x))
        budget.charge(len(actions))
        wins = winning_actions(state[:, :, 0], np.asarray(actions, dtype=np.int64), side, goal)
        budget.check()

        def rank(item):
            action, win = item
            budget.charge()
            dx, dy = action_destination(action)
            dest = dy * 9 + dx
            prior = self._root_previous.get(action) if root else None
            return (win, action == preferred,
                    prior['score'] if prior else -inf,
                    dest in threats or dest == own_goal, state[dy, dx, 0] != 0,
                    -max(abs(dx - goal % 9), abs(dy - goal // 9)), -action)
        # Construct history states only for children actually visited.
        for action, _ in sorted(zip(actions, wins), key=rank, reverse=True):
            budget.charge()
            child, _ = self.game.getNextState(state, side, action)
            yield action, child

    def _search(self, state, depth, alpha, beta, ply, budget):
        budget.visit()
        side = int(state[:, :, 32].flat[1])
        terminal = terminal_value(self.game, state, side, ply)
        if terminal is not None:
            return terminal, []
        # Post-capture leaves commonly transpose. Quiet leaves usually carry
        # different repetition histories; avoid building a costly cache key
        # where reuse is rare. Internal nodes still use the full draw-safe key.
        if depth == 0 and (not self.use_table or int(state[:, :, 32].flat[4]) != 1):
            return self._leaf(state, side, ply, budget)
        # Fold positions only when all future-play/draw information agrees.
        # Different-depth heuristic scores remain ordering hints, not values.
        key = position_key(state) if self.use_table else b''
        progress = self._root_progress if ply == 0 else None
        entry = self.table.get((key, depth)) if self.use_table else None
        alpha_original, beta_original = alpha, beta
        if entry is not None:
            budget.tt_hits += 1
            value = from_table(entry.score, ply)
            if entry.bound == 'exact':
                return value, list(entry.pv)
            if progress is None:
                if entry.bound == 'lower':
                    alpha = max(alpha, value)
                else:
                    beta = min(beta, value)
                if alpha >= beta:
                    return value, list(entry.pv)
        if depth == 0:
            value, line = self._leaf(state, side, ply, budget)
            self._store(key, depth, value, 'exact', line, ply)
            return value, line
        hint = entry or (self.table.get((key, self._hints.get(key, -1)))
                         if self.use_table else None)
        preferred = hint.best_move if hint else None
        if progress is not None and progress.incumbent is not None:
            preferred = progress.incumbent
        best, pv = -inf, []
        for action, child in self._ordered(state, side, preferred, budget, root=progress is not None):
            value, line = self._search(child, depth - 1, -beta, -alpha, ply + 1, budget)
            value = -value
            if progress is not None:
                # Publish only a fully returned child, never an interrupted
                # subtree or an intermediate optimistic score.
                progress.record(action, value, line, alpha)
            if value > best:
                best, pv = value, [action] + line
            alpha = max(alpha, best)
            if alpha >= beta or best == MATE - ply - 1:
                break
        if progress is not None:
            progress.finished = True
        budget.check()
        bound = 'upper' if best <= alpha_original else 'lower' if best >= beta_original else 'exact'
        self._store(key, depth, best, bound, pv, ply)
        return best, pv

    def _store(self, key, depth, value, bound, pv, ply):
        if self.use_table and self.config.table_entries:
            if ((key, depth) not in self.table
                    and len(self.table) >= self.config.table_entries):
                old_key, old_depth = next(iter(self.table))
                self.table.pop((old_key, old_depth))
                if self._hints.get(old_key) == old_depth:
                    del self._hints[old_key]
            self.table[(key, depth)] = Entry(depth, to_table(value, ply), bound,
                                             pv[0] if pv else -1, tuple(pv))
            if pv and depth >= self._hints.get(key, -1):
                self._hints[key] = depth

    def analyze(self, state, budget=None):
        self._prepare()
        budget = budget or Budget(self.config.node_limit, self.config.time_limit)
        side = int(state[:, :, 32].flat[1])
        # Validation/one legal fallback is required even for a zero budget.
        if terminal_value(self.game, state, side) is not None:
            raise ValueError('Cannot select a move from a terminal position')
        actions = np.flatnonzero(self.game.getValidMoves(state, side))
        action, score, depth, pv = int(actions[0]), None, 0, [int(actions[0])]
        selected_depth, source, score_bound = 0, 'legal_fallback', None
        self._root_previous = {}
        self._root_progress = None
        explanation = {'status': 'diagnostics deferred to prioritise search',
                       'proof': {'status': 'unknown'}}
        stopped = False
        try:
            for target in range(1, self.config.max_depth + 1):
                self._root_progress = RootProgress(target, len(actions), action if depth else None)
                value, line = self._search(state, target, -inf, inf, 0, budget)
                score, pv, action, depth = value, line, line[0], target
                selected_depth, source, score_bound = target, 'completed_iteration', 'exact'
                if self._root_progress.moves:
                    self._root_previous = self._root_progress.moves.copy()
                self._root_progress.finished = True
                budget.check()
                if abs(score) > MATE_THRESHOLD:
                    break
        except BudgetExpired:
            stopped = True
            progress = self._root_progress
            if progress is not None and progress.moves:
                best = max(progress.moves.values(), key=lambda row: row['score'])
                # Re-search the incumbent first, then compare completed siblings
                # at the SAME depth. Never compare a half-searched branch, or
                # substitute a shallow optimistic score for a deeper result.
                if (progress.incumbent is None or progress.incumbent in progress.moves
                        or best['score'] > MATE_THRESHOLD):
                    choice = best
                    source = 'partial_iteration'
                    if progress.finished:
                        depth, source = progress.depth, 'completed_iteration'
                    elif best['score'] < -MATE_THRESHOLD:
                        # A proved loss for one move is not a proved loss for
                        # the root. Prefer an alternative not yet refuted.
                        known = self._root_previous.copy()
                        known.update(progress.moves)
                        remaining = [int(a) for a in actions if a not in progress.moves
                                     and (a not in known or known[a]['score'] >= -MATE_THRESHOLD)]
                        if remaining:
                            alternative = max(remaining, key=lambda a: known.get(a, {}).get('score', -inf))
                            choice = known.get(alternative, dict(action=alternative, score=None,
                                pv=[alternative], depth=0, bound=None))
                            source = 'unrefuted_fallback'
                    action, score, pv = choice['action'], choice['score'], choice['pv']
                    selected_depth, score_bound = choice['depth'], choice['bound']
        # Explanations must not consume the move budget before a single useful
        # branch is searched. Emit them only if budget remains after search.
        try:
            explanation = self.evaluator.explain(state, side, budget, diagnostics=False)
        except BudgetExpired:
            if not depth and score is None:
                stopped = True
        progress = self._root_progress
        partial_depth = progress.depth if progress is not None and not progress.finished else 0
        completed_moves = len(progress.moves) if progress is not None else 0
        self._root_progress = None
        explanation['config'] = self.config.to_dict()
        explanation['pv'] = pv
        explanation['search_score'] = score
        explanation['search_scope'] = 'position' if source == 'completed_iteration' else 'selected_move'
        explanation['search_score_bound'] = score_bound
        if score is not None and abs(score) > MATE_THRESHOLD:
            explanation['proof'] = dict(status='proven', plies=int(MATE - abs(score)),
                                        winner=side if score > 0 else 1 - side,
                                        scope=explanation['search_scope'])
        table_bytes = sys.getsizeof(self.table) + sum(
            sys.getsizeof(key) + sys.getsizeof(key[0]) + sys.getsizeof(entry)
            + sys.getsizeof(entry.pv) for key, entry in self.table.items())
        table_bytes += sys.getsizeof(self._hints)
        result = SearchResult(action, score, depth, pv, budget.nodes, budget.work,
                              budget.proof_nodes, budget.clock() - budget.start, stopped, table_bytes,
                              explanation, dict(budget.module_seconds), dict(budget.module_calls),
                              selected_depth, partial_depth, completed_moves, len(actions),
                              source, score_bound, budget.tt_hits)
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
            return evaluator.explain(position, root, budget, proof=proof)['score'], []
        best, pv = (-inf if side == root else inf), []
        for action in map(int, np.flatnonzero(game.getValidMoves(position, side))):
            budget.charge()
            child, _ = game.getNextState(position, side, action)
            value, line = visit(child, remaining - 1, ply + 1)
            if (value > best if side == root else value < best):
                best, pv = value, [action] + line
        return best, pv

    return visit(state, depth, 0)
