"""Native terminal-only proofs, with a bounded Python cancellation interval.

A native call visits at most 64 nodes (and charges at most 129 work). Larger
configured proofs retain the Python reference, including its per-operation
cancellation. State transitions use the existing compiled Board API.
"""
from functools import lru_cache
import numpy as np
from numba import njit

from ..IntransitiveConstants import DIRECTIONS, MAX_TOTAL_PLY, HISTORY_CAPACITY, NO_CAPTURE_LIMIT
from ..IntransitiveLogicNumba import (
    Board, _terminal_status, raw_movement_mask, _corner_winner, validate_state,
)
from .evaluation import MATE
from .kernels import no_terminal_win_in_horizon
from .moves import masks_from_board, move_board, has_move, legal_actions

NATIVE_NODE_LIMIT = 64
READY = False

# Stable goal-first order, identical to sorting the ascending legal actions.
GOAL_ORDER = np.array([
    sorted(range(648), key=lambda a: (
        (a // 8 % 9 + DIRECTIONS[a % 8][0],
         a // 8 // 9 + DIRECTIONS[a % 8][1]) != (goal, goal)))
    for goal in (0, 8)], dtype=np.int64)


@njit(cache=True)
def ordered_actions(pieces, side, a1_defender):
    return order_legal_actions(raw_movement_mask(pieces, side), side, a1_defender)


@njit(cache=True)
def order_legal_actions(mask, side, a1_defender):
    order = GOAL_ORDER[int(side == a1_defender)]
    return order[mask[order]]


@njit(cache=True)
def _charge(counts, work_limit):
    if counts[0] >= work_limit:
        counts[2] = 2  # shared work budget
        return False
    counts[0] += 1
    return True


# Recursive code using jitclass transitions is compiled per process: loading
# reloading its disk cache crashed under the pinned Numba version. Warm before
# creating a move deadline; never trigger compilation from prove().
@njit
def _visit(state, depth, ply, alpha, beta, node_limit, work_limit, counts, lines, specialised):
    if counts[1] >= node_limit:
        counts[2] = 1  # local proof limit
        return 0.
    if not _charge(counts, work_limit):
        return 0.
    counts[1] += 1
    if ply == 0:
        validate_state(state)
    side = int(state[:, :, 82:84].flat[1])
    winner, reason = _terminal_status(state)
    if winner >= 0:
        return MATE - ply if winner == side else -MATE + ply
    if reason != 'ongoing' or depth == 0:
        return 0.
    a1 = int(state[:, :, 82:84].flat[2])
    if ply == 0:
        if not _charge(counts, work_limit):
            return 0.
        if no_terminal_win_in_horizon(state[:, :, 0], side, a1, depth):
            return 0.
    best = -np.inf
    board = Board(2, True)
    for action in ordered_actions(state[:, :, 0], side, a1):
        if not _charge(counts, work_limit):
            return 0.
        lines[ply + 1, :] = -1
        if specialised and depth == 1:
            # The last ply needs only official-win detection. Draw and ongoing
            # both have neutral internal utility and remain UNKNOWN publicly.
            # Preserve the public transition's total-ply overflow check.
            board.copy_state(state, False)
            if board.get_total_ply() == MAX_TOTAL_PLY:
                raise ValueError('Total ply overflow')
            if counts[1] >= node_limit:
                counts[2] = 1
                return 0.
            if not _charge(counts, work_limit):
                return 0.
            counts[1] += 1
            pieces = state[:, :, 0].copy()
            source = action // 8
            x, y = source % 9, source // 9
            dx, dy = DIRECTIONS[action % 8]
            pieces[y + dy, x + dx] = pieces[y, x]
            pieces[y, x] = 0
            winner = _corner_winner(pieces, a1)
            if winner < 0 and not raw_movement_mask(pieces, 1 - side).any():
                winner = side
            value = MATE - ply - 1 if winner == side else 0.
        else:
            board.copy_state(state, False)
            board.make_move(action, side)
            value = -_visit(board.state, depth - 1, ply + 1, -beta, -alpha,
                            node_limit, work_limit, counts, lines, specialised)
        if counts[2]:
            return 0.
        if value > best:
            best = value
            lines[ply, :] = lines[ply + 1, :]
            lines[ply, ply] = action
        alpha = max(alpha, value)
        if alpha >= beta or best == MATE - ply - 1:
            break
    return best


@njit
def native_proof(state, depth, node_limit, work_limit, specialised):
    counts = np.zeros(3, dtype=np.int64)  # charged work, visited nodes, stop code
    lines = np.full((10, 9), -1, dtype=np.int64)
    score = _visit(state, depth, 0, -np.inf, np.inf, node_limit, work_limit,
                   counts, lines, specialised)
    return score, lines[0], counts


@lru_cache(maxsize=1)
def warm_proof_kernel():
    global READY
    board = Board(2, True)
    native_proof(board.get_state(), 2, NATIVE_NODE_LIMIT, 128, True)
    READY = True


@njit(cache=True)
def ordered_bitboard_actions(masks, side, a1):
    actions = legal_actions(masks, side)
    ordered = np.empty_like(actions)
    goal = 80 if side == a1 else 0
    count = 0
    for winning in (True, False):
        for action in actions:
            dx, dy = DIRECTIONS[action % 8]
            if (action // 8 + dx + 9 * dy == goal) == winning:
                ordered[count] = action
                count += 1
    return ordered


# Compact proof traversal owns scratch memory, so even native exceptions cannot
# alter the main search position. History rows are append-only along a branch;
# a capture changes the start index, and returning restores the previous range.
@njit
def _visit_compact(pieces, masks, history, start, end, side, a1, total, modelling_draws,
                   depth, ply, alpha, beta, node_limit, work_limit, counts, lines):
    if counts[1] >= node_limit:
        counts[2] = 1
        return 0.
    if not _charge(counts, work_limit):
        return 0.
    counts[1] += 1
    winner = _corner_winner(pieces, a1)
    if winner < 0 and not has_move(masks, side):
        winner = 1 - side
    if winner >= 0:
        return MATE - ply if winner == side else -MATE + ply
    if modelling_draws:
        repetitions = 0
        for i in range(start, end + 1):
            if np.array_equal(history[i], history[end]):
                repetitions += 1
        if repetitions >= 3 or end - start >= NO_CAPTURE_LIMIT:
            return 0.
    if depth == 0:
        return 0.
    if ply == 0:
        if not _charge(counts, work_limit):
            return 0.
        if no_terminal_win_in_horizon(pieces, side, a1, depth):
            return 0.
    best = -np.inf
    for action in ordered_bitboard_actions(masks, side, a1):
        if not _charge(counts, work_limit):
            return 0.
        if total == MAX_TOTAL_PLY:
            raise ValueError('Total ply overflow')
        source = action // 8
        x, y = source % 9, source // 9
        dx, dy = DIRECTIONS[action % 8]
        nx, ny = x + dx, y + dy
        captured = int(pieces[ny, nx])
        target = 9 * ny + nx
        lines[ply + 1, :] = -1
        move_board(pieces, masks, source, target, captured)
        if depth == 1:
            # Same logical visit as the reference, without unused draw history.
            if counts[1] >= node_limit:
                counts[2] = 1
                move_board(pieces, masks, source, target, captured, True)
                return 0.
            if not _charge(counts, work_limit):
                move_board(pieces, masks, source, target, captured, True)
                return 0.
            counts[1] += 1
            winner = _corner_winner(pieces, a1)
            if winner < 0 and not has_move(masks, 1 - side):
                winner = side
            value = MATE - ply - 1 if winner == side else 0.
        else:
            next_start = end + 1 if captured else start
            if not modelling_draws and end - next_start + 1 == HISTORY_CAPACITY:
                next_start += 1
            for square in range(81):
                history[end + 1, square] = pieces.flat[square]
            history[end + 1, 81] = 1 - side
            value = -_visit_compact(pieces, masks, history, next_start, end + 1,
                1 - side, a1, total + 1, modelling_draws, depth - 1, ply + 1,
                -beta, -alpha, node_limit, work_limit, counts, lines)
        move_board(pieces, masks, source, target, captured, True)
        if counts[2]:
            return 0.
        if value > best:
            best = value
            lines[ply, :] = lines[ply + 1, :]
            lines[ply, ply] = action
        alpha = max(alpha, value)
        if alpha >= beta or best == MATE - ply - 1:
            break
    return best


@njit
def native_compact_proof(pieces, history, length, side, a1, total, modelling_draws,
                         depth, node_limit, work_limit, masks=None, counts=None, lines=None):
    if counts is None:
        counts = np.empty(3, dtype=np.int64)
    if lines is None:
        lines = np.empty((10, 9), dtype=np.int64)
    counts[:] = 0
    lines[:] = -1
    if masks is None:
        masks = masks_from_board(pieces)
    score = _visit_compact(pieces, masks, history, 0, length - 1, side, a1, total,
        modelling_draws, depth, 0, -np.inf, np.inf, node_limit, work_limit, counts, lines)
    return score, lines[0], counts


class ProofScratch:
    """Private storage reused by sequential proofs of one search position.

    Every input and result buffer is reset before a call, including after an
    exception. The live position never aliases native traversal storage.
    """
    def __init__(self, capacity):
        self.history = np.empty((capacity, 82), dtype=np.int8)
        self.pieces = np.empty((9, 9), dtype=np.int8)
        self.masks = np.empty((6, 2), dtype=np.uint64)
        self.counts = np.empty(3, dtype=np.int64)
        self.lines = np.empty((10, 9), dtype=np.int64)


def compact_proof(position, depth, node_limit, work_limit, *, _borrow=False):
    """Return owned results unless the search consumes them before the next call."""
    length = len(position.history)
    scratch = position._proof_scratch
    if scratch is None or len(scratch.history) < length + depth:
        scratch = ProofScratch(max(HISTORY_CAPACITY, length) + depth)
        position._proof_scratch = scratch
    scratch.history[:length] = np.frombuffer(
        b''.join(position.history), dtype=np.int8).reshape(length, 82)
    scratch.pieces[:] = position.pieces
    scratch.masks[:] = position.masks
    score, line, counts = native_compact_proof(scratch.pieces, scratch.history, length, position.side,
        position.a1, position.total, position.modelling_draws, depth, node_limit, work_limit,
        scratch.masks, scratch.counts, scratch.lines)
    if _borrow:
        return score, line, counts
    return score, line.copy(), counts.copy()


COMPACT_READY = False


@lru_cache(maxsize=1)
def warm_compact_proof_kernel():
    global COMPACT_READY
    from .position import SearchPosition
    compact_proof(SearchPosition(Board().get_state()), 2, NATIVE_NODE_LIMIT, 128)
    COMPACT_READY = True
