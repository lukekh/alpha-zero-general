"""Native terminal-only proofs, with a bounded Python cancellation interval.

A native call visits at most 64 nodes (and charges at most 129 work). Larger
configured proofs retain the Python reference, including its per-operation
cancellation. State transitions use the existing compiled Board API.
"""
from functools import lru_cache
import numpy as np
from numba import njit

from ..IntransitiveConstants import DIRECTIONS, MAX_TOTAL_PLY
from ..IntransitiveLogicNumba import (
    Board, _terminal_status, raw_movement_mask, _corner_winner, validate_state,
)
from .evaluation import MATE
from .kernels import no_terminal_win_in_horizon

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
    side = int(state[:, :, 32].flat[1])
    winner, reason = _terminal_status(state)
    if winner >= 0:
        return MATE - ply if winner == side else -MATE + ply
    if reason != 'ongoing' or depth == 0:
        return 0.
    a1 = int(state[:, :, 32].flat[2])
    if ply == 0:
        if not _charge(counts, work_limit):
            return 0.
        if no_terminal_win_in_horizon(state[:, :, 0], side, a1, depth):
            return 0.
    best = -np.inf
    board = Board(2)
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
    board = Board(2)
    native_proof(board.get_state(), 2, NATIVE_NODE_LIMIT, 128, True)
    READY = True
