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
from .config import SearchConfig
from .evaluation import Evaluator, MATE, MATE_THRESHOLD, terminal_value
from .kernels import no_terminal_win_in_horizon, winning_actions, warm_search_kernels
from .material import after_capture, count_pieces, warm_material_kernels
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

    Internal iterative *reduction* returns a shallower value than the caller
    asked for, so it belongs here. Internal iterative *deepening* only pays
    nodes to find an ordering move and leaves the completed value alone.
    """
    return (config.nmp_enabled or config.futility_enabled
            or config.quiescence_enabled or config.lmr_enabled
            or (config.iir_enabled and config.iir_mode == 'reduce'))


ORDERING_COUNTERS = (
    'ordered_nodes', 'beta_cutoffs', 'first_move_cutoffs', 'cutoff_index_sum',
    'cutoff_from_preferred', 'cutoff_from_killer', 'cutoff_from_counter',
    'cutoff_from_other', 'counter_move_available', 'counter_move_updates',
    'continuation_updates', 'history_updates', 'history_decays',
    'iir_nodes', 'iir_reduced_plies', 'iir_deepen_searches')


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


def certificate(state, config, budget):
    """A forced corner run for either side, or None.

    Returns the score from the side to move's perspective, so a run by the
    opponent reports as a loss. At most one side can certify: each run requires
    the other to be unable to reach its own corner first.
    """
    from .clear_run import NO_RUN, certify
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
        budget.charge(2 * 81)
        for side in (turn, 1 - turn):
            plies = certify(pieces, side, turn, a1, clock_left, config.certificate_plies)
            if plies != NO_RUN:
                score = MATE - plies
                return score if side == turn else -score
        return None
    finally:
        budget.module_seconds['certificate'] += perf_counter() - start
        budget.module_calls['certificate'] += 1


def prove(game, state, config, budget, *, specialised=True):
    """Bounded terminal search, then the run certificate if it is enabled."""
    result = prove_terminal(game, state, config, budget, specialised=specialised)
    if result['status'] == 'proven' or not config.certificate_enabled:
        return result
    score = certificate(state, config, budget)
    if score is None:
        return result
    return dict(status='proven', score=score, plies=int(MATE - abs(score)),
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
        # Refutations indexed by the move that led here, and the continuation
        # tables indexed by that move and (optionally) our own previous one.
        self._counter_moves = np.full((2,648),-1,dtype=np.int64)
        self._continuation = None
        self._zero_continuation = np.zeros(648,dtype=np.int32)
        # Actions on the path from the root; -1 marks a hypothetical null turn.
        self._path = []
        self._selective_disabled = False
        self._null_context = False
        self._selective_stats = {}
        self._ordering_stats = dict.fromkeys(ORDERING_COUNTERS, 0)
        self._cutoffs_by_depth = np.zeros((65,3),dtype=np.int64)

    def reload(self):
        self.table.clear()
        self._hints.clear()
        self.last_result = None

    def _prepare(self, *, warm_proof=True):
        if ((self.config.nmp_enabled or self.config.futility_enabled
             or self.config.quiescence_enabled) and not self.use_compact):
            raise ValueError("Selective search requires the compact Python backend")
        self._selective_stats = dict.fromkeys(("nmp_attempts", "nmp_cutoffs", "nmp_skips",
            "verification_searches", "verification_failures", "futility_eligible",
            "futility_pruned", "static_evaluations", "null_nodes", "verification_nodes",
            "quiescence_captures", "lmr_reduced", "lmr_researches"), 0)
        self._ordering_stats = dict.fromkeys(ORDERING_COUNTERS, 0)
        self._cutoffs_by_depth = np.zeros((65,3),dtype=np.int64)
        warm_search_kernels()
        warm_material_kernels()
        warm_position_kernels()
        if self.config.compiled_ordering_enabled or self.config.ordering_enabled or self.config.mvv_lva_enabled:
            from .ordering import warm_ordering
            warm_ordering(self.config.continuation_enabled and self.config.ordering_enabled)
        if self.config.pressure_enabled and self.config.pressure_weight:
            from .pressure import warm_pressure_kernel
            warm_pressure_kernel()
        if warm_proof:
            from .proof import warm_proof_kernel
            warm_proof_kernel()
            from .proof import warm_compact_proof_kernel
            warm_compact_proof_kernel()
        if self.config.certificate_enabled:
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
        self._counter_moves.fill(-1)
        # Allocated per search, so nothing survives into the next move. Only a
        # configuration that reads the rows pays for them.
        self._continuation = (np.zeros((self.config.continuation_plies,2,648,648),dtype=np.int32)
                              if self.config.continuation_enabled and self.config.ordering_enabled else None)
        self._path.clear()
        self._mvv_lva_nodes = 0
        self._mvv_lva_captures = 0
        self.evaluator = Evaluator(self.game, self.config)
        self._material_root = None
        self._material_counts = None

    def _leaf(self, state, side, ply, budget):
        proof = ({'status': 'unknown'} if self._null_context else
                 prove(self.game, state, self.config, budget))
        if proof['status'] == 'proven':
            return from_table(proof['score'], ply), proof['pv']
        return self.evaluator.score(state, side, budget, proof=proof, counts=self._material_counts), []

    def _quiesce(self, state, side, alpha, beta, ply, budget, remaining):
        """Resolve captures past the horizon, standing pat on quiet positions.

        Only captures are searched, so the chain is bounded by the pieces on the
        board as well as by `remaining`: every move removes one. Repetition
        cannot arise inside it for the same reason, and each capture resets the
        no-capture clock, so no draw rule can trigger part-way through.

        The stand-pat score is the ordinary leaf, including its proof, so a
        position with no captures costs exactly what it did before.
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
        for action in state.legal():
            action = int(action)
            x, y = action_destination(action)
            if not pieces[y, x]:
                continue  # quiet moves are the caller's business, not ours
            self._selective_stats['quiescence_captures'] += 1
            state.push(action)
            counts = self._material_counts
            self._material_counts = state.counts
            try:
                value, reply = self._quiesce(state, 1 - side, -beta, -alpha,
                                             ply + 1, budget, remaining - 1)
                value = -value
            finally:
                state.pop()
                self._material_counts = counts
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

    def _ordering_context(self, side):
        """The counter move and continuation rows for the path that led here.

        `self._path` holds the actions played from the root, so its last entry
        is the opponent's reply being answered and the one before it is this
        side's own previous move. A hypothetical null turn stores -1 and
        therefore supplies no context at all, rather than borrowing its
        grandparent's.
        """
        counter, one, two = -1, None, None
        cfg = self.config
        if not cfg.ordering_enabled:
            return counter, one, two
        previous = self._path[-1] if self._path else -1
        if previous < 0:
            return counter, one, two
        if cfg.counter_move_enabled:
            counter = int(self._counter_moves[side, previous])
            if counter >= 0:
                self._ordering_stats['counter_move_available'] += 1
        if self._continuation is not None:
            one = self._continuation[0, side, previous]
            if cfg.continuation_plies > 1:
                own = self._path[-2] if len(self._path) > 1 else -1
                # A constant row keeps one compiled signature per ply count.
                two = self._continuation[1, side, own] if own >= 0 else self._zero_continuation
        return counter, one, two

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
            # Each additional ordering key is one more lookup per candidate. The
            # charge follows the configuration, never the data, so identical
            # settings always cost identical work.
            keys = (self.config.counter_move_enabled
                    + (self.config.continuation_plies if self._continuation is not None else 0))
            if keys:
                budget.charge(len(actions)*keys)
            prior = np.full(648,-np.inf)
            if root:
                for action,row in self._root_previous.items():
                    prior[action] = row['score']
            capture_values = self._capture_order_values(state, side, actions, budget)
            counter, continuation, continuation2 = self._ordering_context(side)
            kernel = ordered_actions if self.config.compiled_ordering_enabled else ordered_actions.py_func
            ordered = kernel(pieces,actions,side,80 if side == a1 else 0,
                -1 if preferred is None else preferred,prior,self._killers[min(ply,64)],
                self._history[side],self.config.ordering_enabled,capture_values,
                counter,continuation,continuation2)
            budget.check()
            for action in ordered:
                budget.charge()
                action = int(action)
                yield action, None if compact else self.game.getNextState(state,side,action)[0]
            return
        actions = list(map(int, legal))
        pieces = state.pieces if compact else state[:, :, 0]
        capture_values = self._capture_order_values(state, side, actions, budget)
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
        key = position_key(state) if self.use_table else b''
        if self.use_table and selective_mode_early(self.config):
            key = b'selective-v1\0' + key
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
        hint = entry or (self.table.get((key, self._hints.get(key, -1)))
                         if self.use_table else None)
        preferred = hint.best_move if hint else None
        if progress is not None and progress.incumbent is not None:
            preferred = progress.incumbent
        static = None
        cfg = self.config
        # A deep node with no move to try first is searched in whatever order
        # the static keys happen to give. Either buy an ordering move with a
        # shallow pass, or stop paying full depth for an unordered node.
        if (cfg.iir_enabled and preferred is None and ply > 0
                and depth >= cfg.iir_min_depth and not self._selective_disabled):
            self._ordering_stats['iir_nodes'] += 1
            if cfg.iir_mode == 'deepen':
                self._ordering_stats['iir_deepen_searches'] += 1
                # The shallow value is discarded: only its move is used, so the
                # completed value of this node is the one full depth gives.
                _, line = self._search(state, depth - 1 - cfg.iir_reduction,
                                       alpha, beta, ply, budget)
                preferred = line[0] if line else None
            else:
                self._ordering_stats['iir_reduced_plies'] += cfg.iir_reduction
                depth -= cfg.iir_reduction
        active = ((cfg.nmp_enabled or cfg.futility_enabled)
                  and compact and not self._selective_disabled and selective.supported(cfg)
                  and ply > 0 and isfinite(alpha) and isfinite(beta)
                  and nextafter(alpha_original, inf) >= beta_original
                  and max(abs(alpha), abs(beta)) < 10000)
        candidate = active and ((cfg.nmp_enabled and depth >= cfg.nmp_min_depth)
                               or (cfg.futility_enabled and depth <= cfg.futility_max_depth))
        safe = candidate and selective.guarded(state, depth, budget)
        if safe:
            self._selective_stats['static_evaluations'] += 1
            evaluation_start = perf_counter()
            try:
                static = self.evaluator.score(state, side, budget, proof={'status': 'unknown'}, counts=state.counts)
            finally:
                budget.module_seconds['selective_static'] += perf_counter() - evaluation_start
        if cfg.nmp_enabled:
            if safe and depth >= cfg.nmp_min_depth and static >= beta:
                self._selective_stats['nmp_attempts'] += 1
                probe_nodes = budget.nodes
                self._path.append(-1)
                try:
                    with selective.unpruned(self, null=True), state.null_turn():
                        value, _ = self._search(state, depth - 1 - cfg.nmp_reduction,
                                                -beta, nextafter(-beta, inf), ply + 1, budget)
                finally:
                    self._path.pop()
                    self._selective_stats['null_nodes'] += budget.nodes - probe_nodes
                if -value >= beta and abs(value) < MATE_THRESHOLD:
                    self._selective_stats['verification_searches'] += 1
                    verification_nodes = budget.nodes
                    try:
                        with selective.unpruned(self):
                            verified, line = self._search(state, depth - cfg.nmp_reduction,
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
        best, pv = -inf, []
        self._ordering_stats['ordered_nodes'] += 1
        for index, (action, child) in enumerate(self._ordered(
                state, side, preferred, budget, root=progress is not None, ply=ply, terminal_checked=True)):
            if (safe and cfg.futility_enabled and depth <= cfg.futility_max_depth
                    and index and selective.quiet(state, action)):
                self._selective_stats['futility_eligible'] += 1
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
            self._path.append(action)
            try:
                # Scores are binary64, including fractional heuristics. Adjacent
                # representable endpoints leave no possible score between them.
                # nextafter also handles -0 correctly; infinite alpha uses the
                # ordinary path. A probe is a bound until its challenger returns
                # from a full re-search. Never publish an intermediate probe.
                probe_beta = nextafter(alpha, inf)
                if self.config.pvs_enabled and index and isfinite(alpha) and probe_beta < beta:
                    budget.pvs_probes += 1
                    value, line = self._search(child, depth - 1, -probe_beta, -alpha, ply + 1, budget)
                    if alpha < -value < beta:
                        budget.pvs_researches += 1
                        value, line = self._search(child, depth - 1, -beta, -alpha, ply + 1, budget)
                elif self._reducible(depth, index, capture, ply, best):
                    # Late, quiet moves get a shallower look first; anything that
                    # beats alpha is re-searched at full depth before it counts.
                    self._selective_stats['lmr_reduced'] += 1
                    reduction = min(self.config.lmr_reduction, depth - 2)
                    value, line = self._search(child, depth - 1 - reduction,
                                               -beta, -alpha, ply + 1, budget)
                    if -value > alpha:
                        self._selective_stats['lmr_researches'] += 1
                        value, line = self._search(child, depth - 1, -beta, -alpha,
                                                   ply + 1, budget)
                else:
                    value, line = self._search(child, depth - 1, -beta, -alpha, ply + 1, budget)
            finally:
                self._path.pop()
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
                    self._cutoff(side, action, index, depth, ply, capture, preferred)
                break
        if progress is not None:
            progress.finished = alpha_original < best < beta_original
        budget.check()
        bound = 'upper' if best <= alpha_original else 'lower' if best >= beta_original else 'exact'
        self._store(key, depth, best, bound, pv, ply)
        return best, pv

    def _bonus(self, table, index, bonus):
        """One bounded ordering credit.

        Aging uses the usual gravity update, whose fixed point is `history_max`,
        so a statistic can neither run away nor need a rescaling pass. Without
        it the original saturating update is kept exactly.
        """
        if self.config.history_aging_enabled:
            table[index] += bonus - int(table[index]) * bonus // self.config.history_max
        else:
            table[index] = min(32767, int(table[index]) + bonus)

    def _cutoff(self, side, action, index, depth, ply, capture, preferred):
        """Record where the beta cutoff came from, then credit the move.

        Attribution reads the killer and counter tables before they are
        updated, so a move is credited to what actually ordered it first.
        Quiet moves only: a capture is already ordered by its own keys.
        """
        stats = self._ordering_stats
        stats['beta_cutoffs'] += 1
        stats['cutoff_index_sum'] += index
        stats['first_move_cutoffs'] += not index
        row = self._cutoffs_by_depth[min(max(depth, 0), 64)]
        row[0] += 1
        row[1] += not index
        row[2] += index
        cfg = self.config
        killers = self._killers[min(ply, 64)]
        previous = self._path[-1] if self._path else -1
        if preferred is not None and action == preferred:
            stats['cutoff_from_preferred'] += 1
        elif cfg.ordering_enabled and action in (killers[0], killers[1]):
            stats['cutoff_from_killer'] += 1
        elif cfg.counter_move_enabled and previous >= 0 and action == self._counter_moves[side, previous]:
            stats['cutoff_from_counter'] += 1
        else:
            stats['cutoff_from_other'] += 1
        if not cfg.ordering_enabled or self._selective_disabled or capture:
            return
        if killers[0] != action:
            killers[1], killers[0] = killers[0], action
        bonus = depth * depth
        self._bonus(self._history[side], action, bonus)
        stats['history_updates'] += 1
        if previous < 0:
            return
        if cfg.counter_move_enabled:
            self._counter_moves[side, previous] = action
            stats['counter_move_updates'] += 1
        if self._continuation is not None:
            self._bonus(self._continuation[0, side, previous], action, bonus)
            stats['continuation_updates'] += 1
            own = self._path[-2] if len(self._path) > 1 else -1
            if cfg.continuation_plies > 1 and own >= 0:
                self._bonus(self._continuation[1, side, own], action, bonus)
                stats['continuation_updates'] += 1

    def _decay(self):
        """Halve ordering statistics between root iterations.

        Cutoffs found early in a search describe a shallower tree than the one
        about to be searched. Halving keeps their order without letting the
        first iterations outweigh everything the deepest one learns.
        """
        self._history >>= 1
        if self._continuation is not None:
            self._continuation >>= 1
        self._ordering_stats['history_decays'] += 1

    def _reducible(self, depth, index, capture, ply, best):
        """Whether this child may be searched shallower first.

        Reductions need a move order worth trusting, so they are refused unless
        ordering is on. Captures, the first `lmr_min_index` moves, the root, and
        any node still without a value are always searched in full.
        """
        cfg = self.config
        return (cfg.lmr_enabled and not self._selective_disabled
                and (cfg.ordering_enabled or cfg.compiled_ordering_enabled)
                and depth >= cfg.lmr_min_depth and index >= cfg.lmr_min_index
                and ply > 0 and not capture and isfinite(best)
                and abs(best) < MATE_THRESHOLD)

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
        if ((self.config.nmp_enabled or self.config.futility_enabled
             or self.config.quiescence_enabled) and not compact):
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
                if self.config.history_aging_enabled and target > 1:
                    self._decay()
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
        result.selective = dict(self._selective_stats, enabled=selective_mode,
            effective=selective_mode and selective.supported(self.config),
            disabled_reason=None if selective.supported(self.config) else 'unsupported evaluator scales',
            depth=selected_depth, identity=self.config.identity())
        cutoffs = self._ordering_stats['beta_cutoffs']
        result.ordering = dict(self._ordering_stats,
            mvv_lva_enabled=self.config.mvv_lva_enabled,
            mvv_lva_nodes=self._mvv_lva_nodes, mvv_lva_captures=self._mvv_lva_captures,
            ordering_enabled=self.config.ordering_enabled,
            counter_move_enabled=self.config.counter_move_enabled,
            continuation_enabled=self.config.continuation_enabled,
            continuation_plies=self.config.continuation_plies,
            history_aging_enabled=self.config.history_aging_enabled,
            iir_enabled=self.config.iir_enabled, iir_mode=self.config.iir_mode,
            # The quantities that predict whether a reduction schedule is safe.
            # The first move searched has index zero.
            first_move_cutoff_rate=(self._ordering_stats['first_move_cutoffs']/cutoffs
                                    if cutoffs else None),
            mean_cutoff_index=(self._ordering_stats['cutoff_index_sum']/cutoffs
                               if cutoffs else None),
            cutoffs_by_depth={int(level): dict(cutoffs=int(counts[0]),
                                               first_move=int(counts[1]), index_sum=int(counts[2]))
                              for level, counts in enumerate(self._cutoffs_by_depth) if counts[0]})
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
