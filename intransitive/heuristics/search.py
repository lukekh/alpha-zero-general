"""Iterative alpha-beta, optional PVS/aspiration and an exhaustive reference.

Search organization reference: Stockfish src/search.cpp (iterative deepening,
TT bounds, mate-distance conversion). Opt-in guarded selective search; no chess-specific eligibility tests.
"""
from dataclasses import dataclass, field
from collections import OrderedDict
from itertools import islice
from functools import lru_cache
from math import inf, isfinite, nextafter
from time import perf_counter
import sys
import numpy as np
from .budget import Budget, BudgetExpired
from . import selective
from .config import SELECTIVE_COUNTERS, SearchConfig
from .evaluation import Evaluator, MATE, MATE_THRESHOLD, terminal_value
from .kernels import (no_capture_in_horizon, no_terminal_win_in_horizon, winning_actions,
                      warm_search_kernels)
from .material import after_capture, count_pieces, warm_material_kernels
from . import exchange
from ..IntransitiveConstants import NO_CAPTURE_LIMIT, action_destination
from ..IntransitiveDisplay import move_to_str
from .position import SearchPosition, warm_position_kernels


@lru_cache(maxsize=1)
def warm_route_kernels():
    from .geometry import distance_map, nearest_distance
    board = np.zeros((9, 9, 84), dtype=np.int8)[:, :, 0]
    distance_map(board, 0, 1)
    distance_map(board, 0, 1, 1)
    nearest_distance(board, np.zeros(1, dtype=np.int16), 1, 1)
    from .features import warm_feature_kernels
    from .flood import warm_flood_kernels
    warm_feature_kernels()
    warm_flood_kernels()


def position_key(state):
    """Exact future-play identity, independent of move number/history order.

    Threefold depends on counts of board+turn occurrences since the last
    capture, not their order. Keep those counts, goal ownership and the draw
    clock. Never merge by board alone or discard a mere second occurrence.
    Bytes are compared exactly by dict, so hash collisions cannot merge nodes.
    """
    if isinstance(state, SearchPosition):
        return state.key()
    meta = state[:, :, 82:84].ravel()
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


def selective_mode_early(config):
    """Whether any opt-in selective feature is on for this config.

    Three deliberate absences. Mate-distance pruning only narrows the window
    to bounds the true value already respects. The certificate cutoff returns
    a proof, and its guard only ever refuses a reduction. None of the three
    makes a result heuristic or withdraws a certificate. The race reduction
    does reduce, so it belongs here.
    """
    return (config.selective_pruning() or config.quiescence_enabled
            or config.lmr_enabled or config.race_reduction_enabled)


def selective_needs_compact(config):
    """Features that read `SearchPosition` state and so need that backend."""
    return config.selective_pruning() or config.quiescence_enabled


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
    stop_reason: str = 'maximum_depth'
    diagnostics_status: str = 'completed'
    effective_limits: dict = field(default_factory=dict)
    pvs_probes: int = 0
    pvs_researches: int = 0
    aspiration_researches: int = 0
    aspiration_fail_highs: int = 0
    aspiration_fail_lows: int = 0
    search_kind: str = 'minimax'
    completed_simulations: int = 0
    requested_simulations: int = 0
    max_tree_depth: int = 0
    selective: dict = field(default_factory=dict)
    certificate: dict = field(default_factory=dict)
    ordering: dict = field(default_factory=dict)
    root_moves: list = field(default_factory=list)
    exact_root: bool = False


@dataclass
class RootProgress:
    depth: int
    total: int
    incumbent: int | None
    moves: dict = field(default_factory=dict)
    finished: bool = False

    def record(self, action, score, line, alpha, beta=inf):
        bound = 'upper' if score <= alpha else 'lower' if score >= beta else 'exact'
        previous = self.moves.get(action)
        # Recovery passes can return a weaker bound on an already verified
        # sibling. Keep its exact value and PV available across retries.
        if previous is not None and previous['bound'] == 'exact' and bound != 'exact':
            return
        self.moves[action] = dict(action=action, score=score, pv=[action] + line,
                                  depth=self.depth, bound=bound)


class ProofLimit(Exception):
    pass


def prove_reference(game, state, config, budget, *, compiled_order=False):
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
        side = position.side if isinstance(position, SearchPosition) else int(position[:, :, 82:84].flat[1])
        terminal = terminal_value(game, position, side, ply)
        if terminal is not None:
            return terminal, []
        if depth == 0:
            return 0., []
        compact = isinstance(position, SearchPosition)
        pieces = position.pieces if compact else position[:, :, 0]
        a1 = position.a1 if compact else int(position[:, :, 82:84].flat[2])
        if ply == 0:
            budget.charge()
            if no_terminal_win_in_horizon(pieces, side, a1, depth):
                return 0., []
        best, pv = -inf, []
        if compact:
            from ..IntransitiveLogicNumba import raw_movement_mask
            mask = raw_movement_mask(pieces, side)
        else:
            mask = game.getValidMoves(position, side)
        goal = 80 if side == a1 else 0
        if compiled_order:
            from .proof import order_legal_actions
            actions = order_legal_actions(mask, side, a1)
        else:
            actions = list(map(int, np.flatnonzero(mask)))
            actions.sort(key=lambda a: action_destination(a) != (goal % 9, goal // 9))
        for action in actions:
            budget.charge()
            if compact:
                position.push(action)
                try:
                    value, line = visit(position, depth - 1, ply + 1, -beta, -alpha)
                finally:
                    position.pop()
            else:
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


def certificate(state, config, budget, stats=None):
    """A forced corner run for either side, or None.

    Returns the score from the side to move's perspective, so a run by the
    opponent reports as a loss. At most one side can certify: each run requires
    the other to be unable to reach its own corner first.

    The race gate answers that question first, from one pass over the board,
    and names the side worth certifying. Most positions stop there, which is
    what makes asking at an interior node affordable; the charge reflects that,
    one board pass for a refusal and a pass per piece for the full argument,
    rather than the flat constant this used to spend either way.
    """
    from .clear_run import NO_RUN, NO_SIDE, certify, race_gate
    compact = isinstance(state, SearchPosition)
    if compact:
        pieces, turn, a1 = state.pieces, state.side, state.a1
        clock = state.clock if state.modelling_draws else 0
    else:
        meta = state[:, :, 82:84].ravel()
        pieces, turn, a1 = state[:, :, 0], int(meta[1]), int(meta[2])
        # Take whichever counter is further along; a shorter allowance can only
        # withhold a certificate, never grant one.
        clock = max(int(meta[3]), max(0, int(meta[4]) - 1))
    clock_left = max(0, NO_CAPTURE_LIMIT - clock)
    start = perf_counter()
    try:
        if stats is not None:
            stats['probes'] += 1
        budget.charge(81)
        side = int(race_gate(pieces, turn, a1, clock_left, config.certificate_plies))
        if side == NO_SIDE:
            if stats is not None:
                stats['gated'] += 1
            return None
        budget.charge(81 * max(1, int(np.count_nonzero(pieces))))
        plies = certify(pieces, side, turn, a1, clock_left, config.certificate_plies)
        if plies == NO_RUN:
            return None
        if stats is not None:
            stats['certified'] += 1
        score = MATE - plies
        return score if side == turn else -score
    finally:
        budget.module_seconds['certificate'] += perf_counter() - start
        budget.module_calls['certificate'] += 1


def prove(game, state, config, budget, *, specialised=True, stats=None):
    """Bounded terminal search, then the run certificate if it is enabled."""
    result = prove_terminal(game, state, config, budget, specialised=specialised)
    if result['status'] == 'proven' or not config.certificate_enabled:
        return result
    score = certificate(state, config, budget, stats)
    if score is None:
        return result
    # The certificate argues that a run cannot be stopped, not which squares
    # it walks, so it publishes no line. Callers read `pv`, so say so in the
    # result rather than leaving the key out.
    return dict(status='proven', score=score, plies=int(MATE - abs(score)), pv=[],
                source='clear-run certificate', nodes=result.get('nodes', 0))


def prove_terminal(game, state, config, budget, *, specialised=True):
    """Use a warmed, bounded native call; preserve reference for larger proofs.

    Each logical visit and traversed edge costs one work, plus the root bound
    query, exactly as in the reference. Specialisation saves physical work;
    it does not rename the counted unit. An interrupted call publishes no proof.
    """
    from . import proof as kernels
    from ..IntransitiveGame import IntransitiveGame
    compact = isinstance(state, SearchPosition)
    ready = kernels.COMPACT_READY if compact else kernels.READY
    if (not ready or (not compact and (not state.flags.c_contiguous or not state.flags.writeable))
            or type(game) is not IntransitiveGame
            or config.proof_nodes > kernels.NATIVE_NODE_LIMIT
            or config.proof_depth == 0 or config.proof_nodes == 0):
        return prove_reference(game, state, config, budget)
    start = perf_counter()
    try:
        budget.check()
        allowance = min(2 * kernels.NATIVE_NODE_LIMIT + 1, max(0, budget.limit - budget.work))
        if compact:
            score, line, counts = kernels.compact_proof(
                state, config.proof_depth, config.proof_nodes, allowance, _borrow=True)
        else:
            score, line, counts = kernels.native_proof(
                state, config.proof_depth, config.proof_nodes, allowance, specialised)
        work, nodes, stop = map(int, counts)
        budget.work += work
        budget.nodes += nodes
        budget.proof_nodes += nodes
        budget.check()
        if stop == 2:
            raise BudgetExpired('work')
        if stop == 1:
            return dict(status='unknown', reason='proof budget', nodes=nodes)
        if abs(score) > MATE_THRESHOLD:
            return dict(status='proven', score=score, pv=list(map(int, line[line >= 0])),
                        plies=int(MATE - abs(score)), nodes=nodes)
        return dict(status='unknown', reason='horizon', nodes=nodes)
    finally:
        budget.module_seconds['proof'] += perf_counter() - start
        budget.module_calls['proof'] += 1


class AlphaBetaPlayer:
    label = 'Alpha–beta'
    kind = 'alphabeta'

    def __init__(self, game=None, config=None, use_table=True, *, use_compact=True):
        if game is None:
            from ..IntransitiveGame import IntransitiveGame
            game = IntransitiveGame()
        self.game = game
        self.config = config or SearchConfig()
        self.use_table = use_table
        self.use_compact = use_compact
        self.table = {}
        self._hints = {}
        self._identity = None
        self.last_result = None
        self._root_progress = None
        self._root_previous = {}
        # Diagnostic only: suppress root alpha raising so every reply returns an
        # exact score instead of an upper bound. Not part of the config identity
        # because it changes cost and reported bounds, never the chosen move.
        self._exact_root = False
        self._killers = np.full((65,2),-1,dtype=np.int64)
        self._history = np.zeros((2,648),dtype=np.int64)
        self._selective_disabled = False
        self._null_context = False
        self._below_horizon = False
        self._table_namespace = b''
        self._selective_stats = {}
        self._certificate_stats = {}

    def reload(self):
        self.table.clear()
        self._hints.clear()
        self.last_result = None

    def _prepare(self, *, warm_proof=True):
        if selective_needs_compact(self.config) and not self.use_compact:
            raise ValueError("Selective search requires the compact Python backend")
        self._selective_stats = dict.fromkeys(SELECTIVE_COUNTERS, 0)
        self._table_namespace = b'selective-v1\0' if selective_mode_early(self.config) else b''
        self._certificate_stats = dict.fromkeys(("probes", "gated", "certified", "cutoffs",
            "guards", "unreduced", "race_probes", "race_quiet", "race_reductions"), 0)
        warm_search_kernels()
        warm_material_kernels()
        warm_position_kernels()
        if self.config.compiled_ordering_enabled or self.config.ordering_enabled or self.config.mvv_lva_enabled:
            from .ordering import warm_ordering
            warm_ordering()
        if self._exchange_active():
            exchange.warm_exchange_kernels()
        if self.config.pressure_enabled and self.config.pressure_weight:
            from .pressure import warm_pressure_kernel
            warm_pressure_kernel()
        if warm_proof:
            from .proof import warm_proof_kernel
            warm_proof_kernel()
            from .proof import warm_compact_proof_kernel
            warm_compact_proof_kernel()
        if (self.config.certificate_enabled or self.config.certificate_cutoff_enabled
                or self.config.certificate_guard_enabled):
            from .clear_run import warm_certificate_kernel
            warm_certificate_kernel()
        if self.config.attack_enabled or self.config.defence_enabled or self.config.overload_enabled:
            warm_route_kernels()
        identity = self.config.identity()
        if identity != self._identity:
            self.table = OrderedDict() if self.config.depth_replacement_enabled else {}
            self._hints.clear()
            self._identity = identity
        self._killers.fill(-1)
        self._history.fill(0)
        self._mvv_lva_nodes = 0
        self._mvv_lva_captures = 0
        self._see_nodes = 0
        self._see_captures = 0
        self._ordering_cutoffs = 0
        self._ordering_first_cutoffs = 0
        self.evaluator = Evaluator(self.game, self.config)
        self._material_root = None
        self._material_counts = None

    def _leaf(self, state, side, ply, budget):
        # A quiescence chain runs a bounded proof at every capture it resolves,
        # not only at the horizon the search would have evaluated anyway. Charge
        # those separately so the chain's real cost is visible (issue #66). Every
        # ordinary leaf reaches this too, so it pays one boolean and no clock.
        if self._below_horizon:
            spent, started = budget.proof_nodes, perf_counter()
            proof = prove(self.game, state, self.config, budget, stats=self._certificate_stats)
            self._selective_stats['quiescence_proof_nodes'] += budget.proof_nodes - spent
            budget.module_seconds['quiescence_proof'] += perf_counter() - started
        else:
            proof = ({'status': 'unknown'} if self._null_context else
                     prove(self.game, state, self.config, budget,
                           stats=self._certificate_stats))
        if proof['status'] == 'proven':
            return from_table(proof['score'], ply), proof['pv']
        return self.evaluator.score(state, side, budget, proof=proof, counts=self._material_counts), []

    def _exchange_active(self):
        """Whether any application site of the exchange evaluation is switched on."""
        cfg = self.config
        return (cfg.see_ordering_enabled or cfg.see_quiescence_ordering_enabled
                or cfg.see_quiescence_pruning_enabled or cfg.delta_pruning_enabled)

    def _exchange_survey(self, pieces, counts, side, actions, budget):
        """Per-action exchange swings and immediate gains; see EXCHANGE.md.

        Charged before and after: the neighbourhood scan is per candidate, the
        bounded series only per capture. Neither is a search.
        """
        budget.charge(8 + 4*len(actions))
        swings, gains, captures = exchange.survey(pieces, counts, side, actions, self.config)
        budget.charge(20*captures)
        self._see_nodes += 1
        self._see_captures += captures
        return swings, gains, captures

    def _see_order_scores(self, state, side, actions, budget):
        if not self.config.see_ordering_enabled:
            return None
        compact = isinstance(state, SearchPosition)
        if not compact:
            budget.charge(81)
        counts = state.counts if compact else count_pieces(state)
        pieces = state.pieces if compact else state[:, :, 0]
        swings, _, _ = self._exchange_survey(pieces, counts, side, actions, budget)
        budget.check()
        return swings

    def _quiesce(self, state, side, alpha, beta, ply, budget, remaining):
        """Resolve captures past the horizon, standing pat on quiet positions.

        Only captures are searched, so the chain is bounded by the pieces on the
        board as well as by `remaining`: every move removes one. Repetition
        cannot arise inside it for the same reason, and each capture resets the
        no-capture clock, so no draw rule can trigger part-way through.

        The stand-pat score is the ordinary leaf, including its proof, so a
        position with no captures costs exactly what it did before.

        The optional exchange and delta filters skip captures, so they can miss
        a resource the full capture list would have found. They never skip a
        capture onto the mover's own goal, and are refused entirely at a node
        where neither counting argument rules out stalemating the opponent.
        """
        budget.visit()
        terminal = terminal_value(self.game, state, side, ply)
        if terminal is not None:
            return terminal, []
        stand_pat, line = self._leaf(state, side, ply, budget)
        if remaining <= 0 or abs(stand_pat) > MATE_THRESHOLD:
            return stand_pat, line
        if stand_pat >= beta:
            return stand_pat, line
        best, best_line = stand_pat, line
        if stand_pat > alpha:
            alpha = stand_pat
        pieces = state.pieces
        cfg = self.config
        actions = state.legal()
        swings = gains = None
        filtering = cfg.see_quiescence_pruning_enabled or cfg.delta_pruning_enabled
        if (cfg.see_quiescence_ordering_enabled or filtering) and len(actions):
            swings, gains, captures = self._exchange_survey(
                pieces, state.counts, side, actions, budget)
            if cfg.see_quiescence_ordering_enabled and captures:
                # Stable, so equal swings keep the generator's ascending order.
                order = np.argsort(-swings, kind='stable')
                actions, swings, gains = actions[order], swings[order], gains[order]
        allowance = 0.
        if filtering and swings is not None:
            budget.charge(81)
            filtering = exchange.stalemate_safe(pieces, side)
            allowance = exchange.delta_allowance(cfg) if cfg.delta_pruning_enabled else 0.
        else:
            filtering = False
        goal = 80 if side == state.a1 else 0
        for index in range(len(actions)):
            action = int(actions[index])
            x, y = action_destination(action)
            if not pieces[y, x]:
                continue  # quiet moves are the caller's business, not ours
            if filtering and 9*y + x != goal:
                if cfg.see_quiescence_pruning_enabled and swings[index] < cfg.see_threshold:
                    self._selective_stats['quiescence_see_skips'] += 1
                    continue
                if (cfg.delta_pruning_enabled
                        and stand_pat + gains[index] + allowance <= alpha):
                    self._selective_stats['quiescence_delta_skips'] += 1
                    continue
            self._selective_stats['quiescence_captures'] += 1
            state.push(action)
            counts = self._material_counts
            self._material_counts = state.counts
            horizon = self._below_horizon
            self._below_horizon = True
            try:
                value, reply = self._quiesce(state, 1 - side, -beta, -alpha,
                                             ply + 1, budget, remaining - 1)
                value = -value
            finally:
                state.pop()
                self._material_counts = counts
                self._below_horizon = horizon
            if value > best:
                best, best_line = value, [action] + reply
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        return best, best_line

    def _quiescence_active(self):
        return (self.config.quiescence_enabled and self.use_compact
                and not self._selective_disabled)

    def _horizon(self, state, side, alpha, beta, ply, budget):
        """The value at depth zero: a static leaf, or a capture search."""
        if self._quiescence_active() and isinstance(state, SearchPosition):
            return self._quiesce(state, side, alpha, beta, ply, budget,
                                 self.config.quiescence_max_plies)
        return self._leaf(state, side, ply, budget)

    def _capture_order_values(self, state, side, actions, budget):
        if not self.config.mvv_lva_enabled:
            return None
        from .ordering import material_order_values
        compact = isinstance(state, SearchPosition)
        budget.charge(12 + 2*len(actions) + (0 if compact else 81))
        counts = state.counts if compact else count_pieces(state)
        pieces = state.pieces if compact else state[:,:,0]
        self._mvv_lva_nodes += 1
        for action in actions:
            x,y = action_destination(int(action))
            self._mvv_lva_captures += bool(pieces[y,x])
        return material_order_values(counts, side, self.config.variable_material_enabled)

    def _ordered(self, state, side, preferred, budget, root=False, ply=0, *, terminal_checked=False):
        compact = isinstance(state, SearchPosition)
        legal = ((state.raw_legal() if terminal_checked else state.legal()) if compact
                 else np.flatnonzero(self.game.getValidMoves(state, side)))
        if self.config.compiled_ordering_enabled or self.config.ordering_enabled:
            from .ordering import ordered_actions
            actions = legal
            pieces = state.pieces if compact else state[:,:,0]
            a1 = state.a1 if compact else int(state[:,:,82:84].flat[2])
            # Same baseline rank-work charge when only compilation is enabled.
            budget.charge(len(actions)*(83 if self.config.ordering_enabled else 2))
            prior = np.full(648,-np.inf)
            if root:
                for action,row in self._root_previous.items():
                    prior[action] = row['score']
            capture_values = self._capture_order_values(state, side, actions, budget)
            see_scores = self._see_order_scores(state, side, actions, budget)
            kernel = ordered_actions if self.config.compiled_ordering_enabled else ordered_actions.py_func
            ordered = kernel(pieces,actions,side,80 if side == a1 else 0,
                -1 if preferred is None else preferred,prior,self._killers[min(ply,64)],
                self._history[side],self.config.ordering_enabled,capture_values,see_scores)
            budget.check()
            for action in ordered:
                budget.charge()
                action = int(action)
                yield action, None if compact else self.game.getNextState(state,side,action)[0]
            return
        actions = list(map(int, legal))
        pieces = state.pieces if compact else state[:, :, 0]
        capture_values = self._capture_order_values(state, side, actions, budget)
        see_scores = self._see_order_scores(state, side, actions, budget)
        swing_of = {} if see_scores is None else dict(zip(actions, map(float, see_scores)))
        a1 = state.a1 if compact else int(state[:, :, 82:84].flat[2])
        goal = 80 if side == a1 else 0
        own_goal = 80 - goal
        threats = []
        for y, x in np.argwhere(pieces * (1 if side == 0 else -1) < 0):
            if max(abs(x - own_goal % 9), abs(y - own_goal // 9)) <= 1:
                threats.append(int(y) * 9 + int(x))
        budget.charge(len(actions))
        wins = winning_actions(pieces, np.asarray(actions, dtype=np.int64), side, goal)
        budget.check()

        def rank(item):
            action, win = item
            budget.charge()
            dx, dy = action_destination(action)
            dest = dy * 9 + dx
            prior = self._root_previous.get(action) if root else None
            return (win, action == preferred,
                    prior['score'] if prior else -inf,
                    dest in threats or dest == own_goal, pieces[dy, dx] != 0,
                    swing_of.get(action, 0.),
                    capture_values[1,abs(int(pieces[dy,dx]))-1] if capture_values is not None and pieces[dy,dx] else 0.,
                    -capture_values[0,abs(int(pieces[action//8//9,action//8%9]))-1] if capture_values is not None and pieces[dy,dx] else 0.,
                    -max(abs(dx - goal % 9), abs(dy - goal // 9)), -action)
        # Construct history states only for children actually visited.
        for action, _ in sorted(zip(actions, wins), key=rank, reverse=True):
            budget.charge()
            if compact:
                yield action, None  # mutation belongs to the recursive try/finally
            else:
                child, _ = self.game.getNextState(state, side, action)
                yield action, child

    def _search(self, state, depth, alpha, beta, ply, budget):
        budget.visit()
        compact = isinstance(state, SearchPosition)
        if compact:
            self._material_counts = state.counts
        elif ply == 0 and self._material_root is not state:
            self._material_counts = count_pieces(state)
        side = state.side if compact else int(state[:, :, 82:84].flat[1])
        terminal = terminal_value(self.game, state, side, ply)
        if terminal is not None:
            return terminal, []
        # Post-capture leaves commonly transpose. Quiet leaves usually carry
        # different repetition histories; avoid building a costly cache key
        # where reuse is rare. Internal nodes still use the full draw-safe key.
        if depth == 0 and (not self.use_table or (len(state.history) if compact else int(state[:, :, 82:84].flat[4])) != 1):
            return self._horizon(state, side, alpha, beta, ply, budget)
        # Fold positions only when all future-play/draw information agrees.
        # Different-depth heuristic scores remain ordering hints, not values.
        key = self._table_namespace + position_key(state) if self.use_table else b''
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
            value, line = self._horizon(state, side, alpha, beta, ply, budget)
            # A quiescence value is bounded by the window it was searched in,
            # so it is not a reusable exact score for this key.
            if not self._quiescence_active():
                self._store(key, depth, value, 'exact', line, ply)
            return value, line
        cfg = self.config
        if cfg.mate_distance_pruning_enabled and ply and not self._selective_disabled:
            # Free, so it runs before the certificate probe pays for its board
            # passes. This node is not terminal, so it cannot already be lost,
            # and the fastest win available is a child that ends the game —
            # exactly the `MATE - ply - 1` the sibling loop already breaks on.
            # Both are bounds the true value satisfies, so narrowing to them is
            # value preserving: it changes how soon a known mate cuts, never
            # which.
            self._selective_stats['mate_distance_eligible'] += 1
            alpha, beta = max(alpha, -MATE + ply), min(beta, MATE - ply - 1)
            if alpha >= beta:
                self._selective_stats['mate_distance_pruned'] += 1
                return alpha, []
        # A certified forced run is a statement about the game rather than
        # about this horizon, so it is worth asking before a single move is
        # generated. The race gate inside `certificate` keeps a node with no
        # runner in range at one board pass. A null-move subtree is excluded:
        # its side-to-move flip never happened, and `certify` reads the turn.
        run = None
        if (ply and not self._null_context
                and ((cfg.certificate_cutoff_enabled and depth >= cfg.certificate_cutoff_min_depth)
                     or (cfg.certificate_guard_enabled and not self._selective_disabled
                         and (cfg.selective_pruning() or cfg.lmr_enabled)))):
            run = certificate(state, cfg, budget, self._certificate_stats)
        if run is not None and cfg.certificate_cutoff_enabled:
            # Proven, and proven without reference to depth: the runner cannot
            # be stopped, so nothing deeper at this node can refute it. What it
            # does not prove is that the win is no quicker, so the node is
            # stored as a bound — lower when the mover runs, upper when the
            # defender does — and never as an exact score.
            self._certificate_stats['cutoffs'] += 1
            value = from_table(run, ply)
            self._store(key, depth, value, 'lower' if run > 0 else 'upper', [], ply)
            return value, []
        hint = entry or (self.table.get((key, self._hints.get(key, -1)))
                         if self.use_table else None)
        preferred = hint.best_move if hint else None
        if progress is not None and progress.incumbent is not None:
            preferred = progress.incumbent
        static = None
        # The game is about to be decided here, so the cheap heuristics do not
        # get to discard a move on a static score and a margin.
        certified = run is not None and cfg.certificate_guard_enabled
        quiet_horizon = not certified and self._quiet_horizon(state, side, depth, ply, budget)
        # One shared eligibility shape for the whole family: non-root, non-PV
        # on entry, finite window strictly inside the evaluator's clipping
        # range, a supported evaluator scale, and the Intransitive board guard.
        active = (cfg.selective_pruning()
                  and compact and not self._selective_disabled and selective.supported(cfg)
                  and ply > 0 and isfinite(alpha) and isfinite(beta)
                  and nextafter(alpha_original, inf) >= beta_original
                  and max(abs(alpha), abs(beta)) < 10000)
        null_move = active and cfg.nmp_enabled and depth >= cfg.nmp_min_depth
        futility = active and cfg.futility_enabled and depth <= cfg.futility_max_depth
        razoring = (active and cfg.razoring_enabled and depth <= cfg.razoring_max_depth
                    and self._quiescence_active())
        reverse = active and cfg.reverse_futility_enabled and depth <= cfg.reverse_futility_max_depth
        move_count = (active and cfg.move_count_pruning_enabled
                      and depth <= cfg.move_count_max_depth)
        candidate = null_move or futility or razoring or reverse or move_count
        if candidate and certified:
            self._certificate_stats['guards'] += 1
        # Only a null probe may relax the own-capture clause, and only when it
        # is the sole reason this node is eligible: the rest of the family are
        # statements about a node, not about one quiet child, so the clause
        # still means to them what it always did.
        safe = candidate and not certified and selective.guarded(
            state, depth, budget,
            ignore_own_captures=(cfg.nmp_relaxed_guard_enabled and null_move
                                 and not (futility or razoring or reverse or move_count)))
        # Move-count pruning asks the evaluator nothing, and futility asks only
        # once a candidate quiet child actually appears, so neither pays here.
        if safe and (null_move or razoring or reverse):
            static = self._static(state, side, budget)
        if safe and reverse:
            self._selective_stats['reverse_futility_eligible'] += 1
            if static - selective.reverse_margin(cfg, depth, state) >= beta:
                # The mirror of a null move without the move: no hypothetical
                # pass is made, so zugzwang is not a failure mode here.
                self._selective_stats['reverse_futility_pruned'] += 1
                self._store(key, depth, beta, 'lower', [], ply)
                return beta, []
        if safe and razoring:
            self._selective_stats['razoring_eligible'] += 1
            if static + selective.razor_margin(cfg, depth, state) <= alpha:
                # Futility discards one quiet child; razoring gives up the
                # node's full-width search and lets quiescence settle the
                # captures instead. A quiescence value that still cannot reach
                # alpha is the fail-low this node was going to report anyway.
                razor_nodes = budget.nodes
                try:
                    value, line = self._quiesce(state, side, alpha, beta, ply, budget,
                                                cfg.quiescence_max_plies)
                finally:
                    self._selective_stats['razoring_nodes'] += budget.nodes - razor_nodes
                if value <= alpha and abs(value) < MATE_THRESHOLD:
                    self._selective_stats['razoring_applied'] += 1
                    self._store(key, depth, value, 'upper', line, ply)
                    return value, line
        if cfg.nmp_enabled:
            if safe and null_move and static >= beta:
                self._selective_stats['nmp_attempts'] += 1
                reduction = self._null_reduction(depth)
                probe_nodes = budget.nodes
                try:
                    with selective.unpruned(self, null=True), state.null_turn():
                        value, _ = self._search(state, depth - 1 - reduction,
                                                -beta, nextafter(-beta, inf), ply + 1, budget)
                finally:
                    self._selective_stats['null_nodes'] += budget.nodes - probe_nodes
                if -value >= beta and abs(value) < MATE_THRESHOLD:
                    self._selective_stats['verification_searches'] += 1
                    verification_nodes = budget.nodes
                    try:
                        with selective.unpruned(self):
                            verified, line = self._search(state, depth - reduction,
                                                         nextafter(beta, -inf), beta, ply, budget)
                    finally:
                        self._selective_stats['verification_nodes'] += budget.nodes - verification_nodes
                    if verified >= beta and abs(verified) < MATE_THRESHOLD:
                        self._selective_stats['nmp_cutoffs'] += 1
                        self._store(key, depth, beta, 'lower', line, ply)
                        return beta, line
                    self._selective_stats['verification_failures'] += 1
            else:
                self._selective_stats['nmp_skips'] += 1
        # Every candidate move in this node shares the runner and enemy scan.
        quiet_context = selective.quiet_context(state) if safe and (futility or move_count) else None
        best, pv = -inf, []
        for index, (action, child) in enumerate(self._ordered(
                state, side, preferred, budget, root=progress is not None, ply=ply, terminal_checked=True)):
            if (safe and move_count and index >= cfg.move_count_base + depth * depth
                    and isfinite(best) and abs(best) < MATE_THRESHOLD):
                # The stronger form of a late-move reduction: skip the child
                # rather than search it shallower. A finite `best` means a
                # child has already returned, so no node can end up with zero
                # searched moves and be mistaken for a stalemate.
                self._selective_stats['move_count_eligible'] += 1
                if selective.quiet(state, action, quiet_context):
                    self._selective_stats['move_count_pruned'] += 1
                    continue
            if safe and futility and index and selective.quiet(state, action, quiet_context):
                self._selective_stats['futility_eligible'] += 1
                if static is None:
                    static = self._static(state, side, budget)
                if static + selective.margin(cfg, depth, state) <= alpha and abs(best) < MATE_THRESHOLD:
                    self._selective_stats['futility_pruned'] += 1
                    continue
            counts = self._material_counts
            capture = False
            if self.config.ordering_enabled:
                x,y = action_destination(action)
                capture = bool(state.pieces[y,x] if compact else state[y,x,0])
            if compact:
                state.push(action)
                child = state
                self._material_counts = state.counts
            elif int(child[:, :, 82:84].flat[3]) == 0:
                x, y = action_destination(action)
                self._material_counts = after_capture(counts, int(state[y, x, 0]))
            try:
                # Scores are binary64, including fractional heuristics. Adjacent
                # representable endpoints leave no possible score between them.
                # nextafter also handles -0 correctly; infinite alpha uses the
                # ordinary path. A probe is a bound until its challenger returns
                # from a full re-search. Never publish an intermediate probe.
                probe_beta = nextafter(alpha, inf)
                scout = self.config.pvs_enabled and index and isfinite(alpha) and probe_beta < beta
                # A reduction applies to whichever narrow search happens first.
                # Making these alternatives, as they once were, meant a scout
                # always won and LMR only ever ran below one (issue #66).
                reduction = self._reduction(depth, index, capture, ply, best,
                                            certified=certified, quiet_horizon=quiet_horizon)
                window = -probe_beta if scout else -beta
                if scout:
                    budget.pvs_probes += 1
                if reduction:
                    # Late, quiet moves get a shallower look first; anything that
                    # beats alpha is re-searched at full depth before it counts.
                    self._selective_stats['lmr_reduced'] += 1
                    value, line = self._search(child, depth - 1 - reduction, window, -alpha, ply + 1, budget)
                    if -value > alpha:
                        self._selective_stats['lmr_researches'] += 1
                        value, line = self._search(child, depth - 1, window, -alpha, ply + 1, budget)
                else:
                    value, line = self._search(child, depth - 1, window, -alpha, ply + 1, budget)
                if scout and alpha < -value < beta:
                    budget.pvs_researches += 1
                    value, line = self._search(child, depth - 1, -beta, -alpha, ply + 1, budget)
            finally:
                if compact:
                    state.pop()
                self._material_counts = counts
            value = -value
            if progress is not None:
                # Publish only a fully returned child, never an interrupted
                # subtree or an intermediate optimistic score.
                progress.record(action, value, line, alpha, beta)
            if value > best:
                best, pv = value, [action] + line
            # Raising alpha at the root turns every later sibling into an upper
            # bound. Analysis can pay for full windows to score them all.
            if progress is None or not self._exact_root:
                alpha = max(alpha, best)
            if alpha >= beta or best == MATE - ply - 1:
                if alpha >= beta:
                    self._ordering_cutoffs += 1
                    self._ordering_first_cutoffs += index == 0
                if self.config.ordering_enabled and not self._selective_disabled and alpha >= beta and not capture:
                    killers = self._killers[min(ply,64)]
                    if killers[0] != action:
                        killers[1],killers[0] = killers[0],action
                    self._history[side,action] = min(32767,int(self._history[side,action])+depth*depth)
                break
        if progress is not None:
            progress.finished = alpha_original < best < beta_original
        budget.check()
        bound = 'upper' if best <= alpha_original else 'lower' if best >= beta_original else 'exact'
        self._store(key, depth, best, bound, pv, ply)
        return best, pv

    def _static(self, state, side, budget):
        """The node's own evaluation, charged and timed as selective overhead."""
        self._selective_stats['static_evaluations'] += 1
        start = perf_counter()
        try:
            return self.evaluator.score(state, side, budget, proof={'status': 'unknown'},
                                        counts=state.counts)
        finally:
            budget.module_seconds['selective_static'] += perf_counter() - start

    def _null_reduction(self, depth):
        """Probe plies to remove. One ply must always survive below the probe."""
        cfg = self.config
        reduction = cfg.nmp_reduction
        if cfg.nmp_depth_divisor:
            reduction += (depth - cfg.nmp_min_depth) // cfg.nmp_depth_divisor
        return max(1, min(reduction, depth - 2))

    def _reducible(self, depth, index, capture, ply, best, *, certified=False,
                   quiet_horizon=False):
        """Whether this child may be searched shallower first.

        Reductions need a move order worth trusting, so they are refused unless
        ordering is on. Captures, the first `lmr_min_index` moves, the root, and
        any node still without a value are always searched in full.

        A certified node is exempt outright: a forced run is being carried out
        or defended here, and no reply to it gets a shallower look. Where the
        race test has shown the branch quiet, the index condition is dropped
        instead — `lmr_min_index` is a guess that late moves matter less, and
        the test has replaced it with an argument that nothing in reach of this
        horizon matters at all. That argument bounds the tactics, not the
        evaluation, so it still licenses a reduction and never a cutoff.
        """
        cfg = self.config
        # The first move of a node is never reduced. After that the index rule
        # qualifies it, or the race test does in the index rule's place.
        late = bool(index) and (index >= cfg.lmr_min_index or quiet_horizon)
        allowed = (cfg.lmr_enabled and not self._selective_disabled
                   and (cfg.ordering_enabled or cfg.compiled_ordering_enabled)
                   and depth >= cfg.lmr_min_depth and late
                   and ply > 0 and not capture and isfinite(best)
                   and abs(best) < MATE_THRESHOLD)
        if certified:
            # Count the reductions actually prevented, not every refusal: a
            # search with reductions off would have refused this anyway.
            self._certificate_stats['unreduced'] += allowed
            return False
        if allowed and quiet_horizon and index < cfg.lmr_min_index:
            self._certificate_stats['race_reductions'] += 1
        return allowed

    def _quiet_horizon(self, state, side, depth, ply, budget):
        """Nothing decisive can happen within the remaining `depth` plies.

        Both tests are free-board Chebyshev arguments, and both hold whatever
        either side plays: `no_terminal_win_in_horizon` rules out a corner win
        and a stalemate, `no_capture_in_horizon` rules out every capture.
        Modelling draws are ruled out separately, since a repetition or the
        eightieth noncapture would end the branch inside the horizon.

        Route distances cannot be used for this. `geometry.distance_map` and
        `nearest_distance` read the board as it stands and allow only the
        mover's own captures, so a route blocked now reports a distance longer
        than the one actually available once the blocker steps aside: an
        overestimate, not an admissible lower bound on arrival. Chebyshev
        distance is admissible, because one king step changes each coordinate
        by at most one whatever occupies the board.
        """
        cfg = self.config
        if not (cfg.race_reduction_enabled and cfg.lmr_enabled and ply
                and depth >= cfg.lmr_min_depth and not self._selective_disabled
                and isinstance(state, SearchPosition)):
            return False
        self._certificate_stats['race_probes'] += 1
        budget.charge(len(state.occurrences) + 1)
        if state.modelling_draws and (state.clock + depth >= NO_CAPTURE_LIMIT
                                      or any(count > 1 for count in state.occurrences.values())):
            return False
        budget.charge(3 * 81)
        if not no_terminal_win_in_horizon(state.pieces, side, state.a1, depth):
            return False
        quiet, pairs = no_capture_in_horizon(state.pieces, depth)
        budget.charge(81 + int(pairs))
        if quiet:
            self._certificate_stats['race_quiet'] += 1
        return bool(quiet)

    def _reduction(self, depth, index, capture, ply, best, *, certified=False,
                   quiet_horizon=False):
        """Plies to remove from this child, or zero to search it in full.

        A fixed single ply makes every reduced child cost one ply less than the
        full search and buys almost nothing; the divisors let the reduction grow
        with remaining depth and with how late the move is. One ply below the
        reduction always survives, so a reduced search is never a static leaf.
        """
        if not self._reducible(depth, index, capture, ply, best,
                               certified=certified, quiet_horizon=quiet_horizon):
            return 0
        cfg = self.config
        reduction = cfg.lmr_reduction
        if cfg.lmr_depth_divisor:
            reduction += (depth - cfg.lmr_min_depth) // cfg.lmr_depth_divisor
        if cfg.lmr_index_divisor:
            reduction += (index - cfg.lmr_min_index) // cfg.lmr_index_divisor
        return max(1, min(reduction, depth - 2))

    def _store(self, key, depth, value, bound, pv, ply):
        if self.use_table and self.config.table_entries:
            if ((key, depth) not in self.table
                    and len(self.table) >= self.config.table_entries):
                if self.config.depth_replacement_enabled:
                    # Prefer deeper/exact entries among eight oldest candidates.
                    # OrderedDict keeps this bounded even after many evictions.
                    old_key,old_depth = min(islice(self.table,8),
                        key=lambda item:(item[1],self.table[item].bound == 'exact'))
                else:
                    old_key, old_depth = next(iter(self.table))
                self.table.pop((old_key, old_depth))
                if self._hints.get(old_key) == old_depth:
                    del self._hints[old_key]
            self.table[(key, depth)] = Entry(depth, to_table(value, ply), bound,
                                             pv[0] if pv else -1, tuple(pv))
            if pv and depth >= self._hints.get(key, -1):
                self._hints[key] = depth

    def analyze(self, state, budget=None, *, exact_root=False):
        self._prepare(warm_proof=budget is None)
        self._exact_root = bool(exact_root)
        budget = budget or Budget(self.config.node_limit, self.config.time_limit)
        from ..IntransitiveGame import IntransitiveGame
        compact = self.use_compact and type(self.game) is IntransitiveGame
        if selective_needs_compact(self.config) and not compact:
            raise ValueError("Selective search requires the Intransitive compact backend")
        # Validate/copy once on entry, including for a zero-budget fallback.
        search_state = (SearchPosition(state, modelling_draws=self.game.board.modelling_draws)
                        if compact else state)
        side = search_state.side if compact else int(state[:, :, 82:84].flat[1])
        if terminal_value(self.game, search_state, side) is not None:
            raise ValueError('Cannot select a move from a terminal position')
        actions = (search_state.legal() if compact else
                   np.flatnonzero(self.game.getValidMoves(state, side)))
        self._material_counts = search_state.counts if compact else count_pieces(state)
        self._material_root = search_state
        action, score, depth, pv = int(actions[0]), None, 0, [int(actions[0])]
        selected_depth, source, score_bound = 0, 'legal_fallback', None
        self._root_previous = {}
        self._root_progress = None
        explanation = {'status': 'diagnostics deferred to prioritise search',
                       'proof': {'status': 'unknown'}}
        stopped = False
        stop_reason = 'maximum_depth'
        try:
            for target in range(1, self.config.max_depth + 1):
                self._root_progress = RootProgress(target, len(actions), action if depth else None)
                low, high = -inf, inf
                width = self.config.aspiration_window
                if self.config.aspiration_enabled and depth and isfinite(score) and not exact_root:
                    low = min(score - width, nextafter(score, -inf))
                    high = max(score + width, nextafter(score, inf))
                retries = 0
                while True:
                    value, line = self._search(search_state, target, low, high, 0, budget)
                    if low < value < high:
                        break
                    budget.aspiration_researches += 1
                    retries += 1
                    width *= 2
                    if value <= low:
                        budget.aspiration_fail_lows += 1
                        low = min(value - width, nextafter(value, -inf))
                    else:
                        budget.aspiration_fail_highs += 1
                        high = max(value + width, nextafter(value, inf))
                    # A bounded number of recovery passes, even for subnormal
                    # widths or mate-distance jumps. The reference always exists.
                    if retries >= 8:
                        low, high = -inf, inf
                score, pv, action, depth = value, line, line[0], target
                selected_depth, source, score_bound = target, 'completed_iteration', 'exact'
                if self._root_progress.moves:
                    self._root_previous = self._root_progress.moves.copy()
                self._root_progress.finished = True
                budget.check()
                # Selective mate-range values are not certificates: finish the
                # requested depth instead of treating a shallow result as proof.
                if abs(score) > MATE_THRESHOLD and (
                        not selective_mode_early(self.config)
                        or target == self.config.max_depth):
                    stop_reason = ('selective_result' if selective_mode_early(self.config)
                                   else 'proven_result')
                    break
        except BudgetExpired as exc:
            stopped = True
            stop_reason = exc.reason
            progress = self._root_progress
            if progress is not None and progress.moves:
                exact = [row for row in progress.moves.values() if row['bound'] == 'exact']
                upper = [row for row in progress.moves.values() if row['bound'] == 'upper']
                # Lower bounds from aspiration fail-highs are not verified
                # challengers. Upper bounds cannot beat an exact sibling either.
                candidates = exact or upper
                best = max(candidates, key=lambda row: row['score']) if candidates else None
                incumbent_row = progress.moves.get(progress.incumbent)
                if (not exact and incumbent_row is not None
                        and incumbent_row['bound'] == 'upper'
                        and incumbent_row['score'] < -MATE_THRESHOLD):
                    # A failed-low pass may have visited other moves only with
                    # upper bounds. They are still unrefuted alternatives to an
                    # incumbent whose upper bound already proves it loses.
                    best = incumbent_row
                # Re-search the incumbent first, then compare completed siblings
                # at the SAME depth. Never compare a half-searched branch, or
                # substitute a shallow optimistic score for a deeper result.
                if (best is not None and (exact or best['score'] < -MATE_THRESHOLD)
                        and (progress.incumbent is None or progress.incumbent in progress.moves
                             or best['score'] > MATE_THRESHOLD)):
                    choice = best
                    source = 'partial_iteration'
                    if progress.finished:
                        depth, source = progress.depth, 'completed_iteration'
                    elif best['score'] < -MATE_THRESHOLD:
                        # A proved loss for one move is not a proved loss for
                        # the root. Prefer an alternative not yet refuted.
                        known = self._root_previous.copy()
                        known.update(progress.moves)
                        remaining = [int(a) for a in actions
                                     if a not in known or known[a]['bound'] == 'lower'
                                     or known[a]['score'] >= -MATE_THRESHOLD]
                        if remaining:
                            alternative = max(remaining, key=lambda a: known.get(a, {}).get('score', -inf))
                            choice = known.get(alternative, dict(action=alternative, score=None,
                                pv=[alternative], depth=0, bound=None))
                            if choice['bound'] == 'lower':
                                choice = dict(action=alternative, score=None, pv=[alternative],
                                              depth=0, bound=None)
                            source = 'unrefuted_fallback'
                    action, score, pv = choice['action'], choice['score'], choice['pv']
                    selected_depth, score_bound = choice['depth'], choice['bound']
        finally:
            self._material_root = None
            self._exact_root = False
        # Explanations must not consume the move budget before a single useful
        # branch is searched. Emit them only if budget remains after search.
        diagnostics_status = 'skipped_budget'
        try:
            explanation = self.evaluator.explain(state, side, budget, diagnostics=False)
            diagnostics_status = 'completed'
        except BudgetExpired:
            pass
        progress = self._root_progress
        partial_depth = progress.depth if progress is not None and not progress.finished else 0
        completed_moves = len(progress.moves) if progress is not None else 0
        # Per-move root scores, for showing which replies the search likes. Rows
        # from different iterations are not comparable, so each keeps the depth
        # and bound it was actually established at; a partial pass never
        # overwrites a sibling already completed deeper.
        rows = dict(self._root_previous)
        for candidate, row in (progress.moves if progress is not None else {}).items():
            if candidate not in rows or row['depth'] > rows[candidate]['depth']:
                rows[candidate] = row
        root_moves = sorted(
            (dict(row, move=move_to_str(row['action']),
                  line=[move_to_str(step) for step in row['pv']],
                  selected=row['action'] == action)
             for row in rows.values()),
            key=lambda row: (-row['score'], row['action']))
        self._root_progress = None
        explanation['config'] = self.config.to_dict()
        explanation['pv'] = pv
        explanation['search_score'] = score
        explanation['search_scope'] = 'position' if source == 'completed_iteration' else 'selected_move'
        explanation['search_score_bound'] = score_bound
        effective_limits = dict(max_depth=self.config.max_depth,
                                time_limit=budget.seconds, work_limit=budget.limit)
        explanation['stop_reason'] = stop_reason
        explanation['diagnostics_status'] = diagnostics_status
        explanation['effective_limits'] = effective_limits
        selective_mode = selective_mode_early(self.config)
        if selective_mode:
            if score_bound is not None:
                score_bound = 'selective_' + score_bound
            explanation['search_score_bound'] = score_bound
            explanation['proof'] = dict(status='unknown', reason='selective search is not a certificate')
        if not selective_mode and score is not None and abs(score) > MATE_THRESHOLD:
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
                              source, score_bound, budget.tt_hits, stop_reason,
                              diagnostics_status, effective_limits,
                              budget.pvs_probes, budget.pvs_researches,
                              budget.aspiration_researches, budget.aspiration_fail_highs,
                              budget.aspiration_fail_lows)
        # Report what this configuration can actually execute, not merely what it
        # declares. Unsupported evaluator scales are now a configuration error,
        # so anything still unreachable here is a depth or window precondition.
        preconditions = selective.activation(self.config, compact=compact)
        declared = sorted(name for name, row in preconditions.items() if row['enabled'])
        unreachable = {name: preconditions[name]['blockers'] for name in declared
                       if preconditions[name]['blockers']}
        counters = dict(self._selective_stats, mvv_lva_captures=self._mvv_lva_captures)
        result.selective = dict(self._selective_stats, enabled=selective_mode,
            effective=selective_mode and not unreachable,
            declared=declared, unreachable=unreachable,
            fired=sorted(name for name in declared
                         if counters[selective.TECHNIQUE_COUNTERS[name]]),
            disabled_reason='; '.join(f'{name}: {"; ".join(reasons)}'
                                      for name, reasons in unreachable.items()) or None,
            mate_distance_enabled=self.config.mate_distance_pruning_enabled,
            depth=selected_depth, identity=self.config.identity())
        result.certificate = dict(self._certificate_stats,
            leaf_enabled=self.config.certificate_enabled,
            cutoff_enabled=self.config.certificate_cutoff_enabled,
            cutoff_min_depth=self.config.certificate_cutoff_min_depth,
            guard_enabled=self.config.certificate_guard_enabled,
            race_reduction_enabled=self.config.race_reduction_enabled,
            plies=self.config.certificate_plies,
            seconds=budget.module_seconds.get('certificate', 0.),
            calls=budget.module_calls.get('certificate', 0))
        quiescing = self.config.quiescence_enabled and compact
        result.ordering = dict(mvv_lva_enabled=self.config.mvv_lva_enabled,
            mvv_lva_nodes=self._mvv_lva_nodes, mvv_lva_captures=self._mvv_lva_captures,
            cutoffs=self._ordering_cutoffs, first_cutoffs=self._ordering_first_cutoffs,
            see_nodes=self._see_nodes, see_captures=self._see_captures,
            compiled_see_enabled=self.config.compiled_see_enabled,
            see_threshold=self.config.see_threshold,
            delta_allowance=(exchange.delta_allowance(self.config)
                             if self.config.delta_pruning_enabled else 0.),
            # An enabled site that cannot run must say so rather than look off.
            see_effective=dict(ordering=self.config.see_ordering_enabled,
                               quiescence_ordering=self.config.see_quiescence_ordering_enabled and quiescing,
                               quiescence_pruning=self.config.see_quiescence_pruning_enabled and quiescing,
                               delta=self.config.delta_pruning_enabled and quiescing))
        result.root_moves = root_moves
        result.exact_root = bool(exact_root)
        self.last_result = result
        return result

    def choose(self, state, player):
        if player != int(state[:, :, 82:84].flat[1]):
            raise ValueError('Player does not match state')
        return self.analyze(state).action

    def play(self, board, nb_moves=0):
        if int(board[:, :, 82:84].flat[1]) != 0:
            raise ValueError('Expected canonical current player zero')
        return self.analyze(board).action


def exhaustive_minimax(game, state, depth, config=None, budget=None):
    """Independent max/min reference, with the same horizon evaluator/proof."""
    config = config or SearchConfig()
    budget = budget or Budget(10**12, 3600.)
    evaluator = Evaluator(game, config)
    root = int(state[:, :, 82:84].flat[1])

    def visit(position, remaining, ply):
        budget.visit()
        side = int(position[:, :, 82:84].flat[1])
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
