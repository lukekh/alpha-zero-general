"""Bounded board-only queries; the rules engine still applies every searched move."""
from functools import lru_cache
import numpy as np
from numba import njit

from ..IntransitiveConstants import DIRECTIONS
from ..IntransitiveLogicNumba import raw_movement_mask


@njit(cache=True)
def no_terminal_win_in_horizon(pieces, turn, a1_defender, depth):
    """Sufficient condition only: neither corner nor stalemate can be won.

    Every move changes at most two squares. For each side, find more than 2*depth
    disjoint (own piece, empty adjacent square) pairs. At least one pair survives
    any depth-ply continuation untouched, so that side always has a legal move.
    Chebyshev distance separately rules out reaching either winning corner.
    Draws can stop a continuation sooner, but cannot introduce a terminal win.
    """
    counts = np.zeros(2, dtype=np.int64)
    for y in range(9):
        for x in range(9):
            piece = int(pieces[y, x])
            if not piece:
                continue
            side = int(piece < 0)
            counts[side] += 1
            corner = 8 if side == a1_defender else 0
            moves = (depth + int(side == turn)) // 2
            if max(abs(x - corner), abs(y - corner)) <= moves:
                return False
    required = 2 * depth + 1
    if counts[0] < required or counts[1] < required:
        return False
    for side in range(2):
        destinations = np.zeros((9, 9), dtype=np.bool_)
        pairs = 0
        for y in range(9):
            for x in range(9):
                piece = int(pieces[y, x])
                if not piece or int(piece < 0) != side:
                    continue
                for dx, dy in DIRECTIONS:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < 9 and 0 <= ny < 9 and pieces[ny, nx] == 0
                            and not destinations[ny, nx]):
                        destinations[ny, nx] = True
                        pairs += 1
                        break
        if pairs < required:
            return False
    return True


@njit(cache=True)
def winning_actions(pieces, actions, side, goal):
    """Exact immediate corner/stalemate wins, without allocating history states."""
    board = pieces.copy()
    wins = np.zeros(len(actions), dtype=np.bool_)
    for i in range(len(actions)):
        action = int(actions[i])
        source = action // 8
        x, y = source % 9, source // 9
        dx, dy = DIRECTIONS[action % 8]
        nx, ny = x + dx, y + dy
        mover, occupant = board[y, x], board[ny, nx]
        board[y, x], board[ny, nx] = 0, mover
        wins[i] = ny * 9 + nx == goal or not raw_movement_mask(board, 1 - side).any()
        board[y, x], board[ny, nx] = mover, occupant
    return wins


@lru_cache(maxsize=1)
def warm_search_kernels():
    # Match the strided board plane layout used by full game states.
    board = np.zeros((9, 9, 33), dtype=np.int8)[:, :, 0]
    no_terminal_win_in_horizon(board, 0, 0, 2)
    winning_actions(board, np.empty(0, dtype=np.int64), 0, 80)
    # Compile rule queries/transitions before the first move's deadline.
    from ..IntransitiveGame import IntransitiveGame
    game = IntransitiveGame()
    state = game.getInitBoard()
    game.getGameEnded(state, 0)
    action = int(np.flatnonzero(game.getValidMoves(state, 0))[0])
    child, side = game.getNextState(state, 0, action)
    game.getGameEnded(child, side)
