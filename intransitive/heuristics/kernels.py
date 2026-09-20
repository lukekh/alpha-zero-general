"""Bounded board-only queries; the rules engine still applies every searched move."""
from functools import lru_cache
import numpy as np
from numba import njit

from ..IntransitiveConstants import DIRECTIONS
from .moves import masks_from_board, move_board, has_move


@njit(cache=True)
def crowded_side_has_moves(pieces, side, depth):
    """More than 2*depth disjoint (own piece, empty adjacent square) pairs.

    Every move changes at most two squares, so a depth-ply continuation can
    damage at most 2*depth pairs. One survives untouched, and its piece can
    still step into its square.
    """
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
    return pairs > 2 * depth


@njit(cache=True)
def open_side_has_moves(pieces, side, depth):
    """More than `depth` pieces with more than `depth` empty neighbours.

    The counting argument the pair rule cannot make on a sparse board, where
    2*depth+1 pieces a side simply do not exist.

    A position can gain at most one occupied square per ply: captures only
    remove pieces, so the squares newly occupied are never more than the
    squares newly vacated, and each ply vacates only the square it moves from.
    A piece still standing where it started therefore keeps at least one of its
    empty neighbours once it began with more than `depth` of them, and can step
    into it.

    A piece leaves its square only by moving or by being captured. Across
    `depth` plies this side moves at most ceil(depth/2) times and loses at most
    floor(depth/2) pieces, so at most `depth` of these pieces are disturbed and
    more than `depth` of them leaves one untouched.
    """
    spacious = 0
    for y in range(9):
        for x in range(9):
            piece = int(pieces[y, x])
            if not piece or int(piece < 0) != side:
                continue
            empty = 0
            for dx, dy in DIRECTIONS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 9 and 0 <= ny < 9 and pieces[ny, nx] == 0:
                    empty += 1
            if empty > depth:
                spacious += 1
    return spacious > depth


@njit(cache=True)
def no_terminal_win_in_horizon(pieces, turn, a1_defender, depth):
    """Sufficient condition only: neither corner nor stalemate can be won.

    Chebyshev distance rules out reaching either winning corner. Stalemate is
    ruled out by whichever of two independent counting arguments applies: the
    pair rule suits crowded boards, the spacious-piece rule sparse ones, and a
    side is safe if either holds. Draws can stop a continuation sooner, but
    cannot introduce a terminal win.
    """
    for y in range(9):
        for x in range(9):
            piece = int(pieces[y, x])
            if not piece:
                continue
            side = int(piece < 0)
            corner = 8 if side == a1_defender else 0
            moves = (depth + int(side == turn)) // 2
            if max(abs(x - corner), abs(y - corner)) <= moves:
                return False
    for side in range(2):
        if not (crowded_side_has_moves(pieces, side, depth)
                or open_side_has_moves(pieces, side, depth)):
            return False
    return True


@njit(cache=True)
def no_capture_in_horizon(pieces, depth):
    """Sufficient condition only: no capture can occur within `depth` plies.

    Returns the answer and the number of piece pairs examined, so the caller
    charges the work actually done rather than a constant.

    One ply moves one piece one king step, so the Chebyshev distance between
    any two pieces changes by at most one per ply, and pieces are never
    created: every capture in the subtree is between two pieces standing here
    now. A capture landing at ply `t` is a step onto the victim's square, so
    the pair stood one square apart after `t-1` plies and therefore no more
    than `t` squares apart now. Keeping every capturable pair strictly further
    apart than `depth` rules out every capture inside the horizon, whoever
    moves and wherever they move.

    Only ordered predator/prey pairs count. A piece cannot take its own
    colour, its own type, or the type that takes it, so those pairs may stand
    adjacent without threatening anything.
    """
    squares = np.empty(81, dtype=np.int64)
    codes = np.empty(81, dtype=np.int64)
    count = 0
    for y in range(9):
        for x in range(9):
            code = int(pieces[y, x])
            if code:
                squares[count] = y * 9 + x
                codes[count] = code
                count += 1
    pairs = 0
    for i in range(count):
        first = int(codes[i])
        for j in range(i + 1, count):
            pairs += 1
            second = int(codes[j])
            if first * second > 0:
                continue
            if not (abs(second) == abs(first) % 3 + 1
                    or abs(first) == abs(second) % 3 + 1):
                continue
            a, b = int(squares[i]), int(squares[j])
            if max(abs(a // 9 - b // 9), abs(a % 9 - b % 9)) <= depth:
                return False, pairs
    return True, pairs


@njit(cache=True)
def winning_actions(pieces, actions, side, goal):
    """Exact immediate corner/stalemate wins, without allocating history states."""
    board = pieces.copy()
    masks = masks_from_board(board)
    wins = np.zeros(len(actions), dtype=np.bool_)
    for i in range(len(actions)):
        action = int(actions[i])
        source = action // 8
        x, y = source % 9, source // 9
        dx, dy = DIRECTIONS[action % 8]
        nx, ny = x + dx, y + dy
        target = ny * 9 + nx
        occupant = int(board[ny, nx])
        move_board(board, masks, source, target, occupant)
        wins[i] = target == goal or not has_move(masks, 1 - side)
        move_board(board, masks, source, target, occupant, True)
    return wins


@lru_cache(maxsize=1)
def warm_search_kernels():
    # Match the strided board plane layout used by full game states.
    board = np.zeros((9, 9, 84), dtype=np.int8)[:, :, 0]
    no_terminal_win_in_horizon(board, 0, 0, 2)
    no_capture_in_horizon(board, 2)
    winning_actions(board, np.empty(0, dtype=np.int64), 0, 80)
    # Compile rule queries/transitions before the first move's deadline.
    from ..IntransitiveGame import IntransitiveGame
    game = IntransitiveGame()
    state = game.getInitBoard()
    game.getGameEnded(state, 0)
    action = int(np.flatnonzero(game.getValidMoves(state, 0))[0])
    child, side = game.getNextState(state, 0, action)
    game.getGameEnded(child, side)
