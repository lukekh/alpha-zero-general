"""Arena/pit callbacks: play(canonical_state, nb_moves) -> fixed action slot."""

import numpy as np
from numba import njit

from .IntransitiveConstants import action_destination
from .IntransitiveDisplay import parse_move, player_colour
from .IntransitiveLogicNumba import Board


def _legal_actions(game, state):
    # Also enforce the canonical callback contract (next player must be zero).
    if game.getGameEnded(state, 0).any():
        raise ValueError("Cannot select a move from a terminal position")
    actions = np.flatnonzero(game.getValidMoves(state, 0))
    if not len(actions):
        raise ValueError("Cannot select a move from a terminal position")
    return actions


class RandomPlayer:
    def __init__(self, game, seed=None):
        self.game = game
        self.rng = np.random.default_rng(seed)

    def play(self, board, nb_moves=0):
        return int(self.rng.choice(_legal_actions(self.game, board)))


class HumanPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board, nb_moves=0):
        actions = set(_legal_actions(self.game, board))
        colour = player_colour(board, 0)
        while True:
            # EOF and interrupts deliberately propagate so users can leave play.
            text = input(f"{colour} move (source destination, e.g. B5 C5): ")
            try:
                action = parse_move(text)
                if action not in actions:
                    raise ValueError("That move is not legal in this position")
            except ValueError as error:
                print(f"Invalid move: {error}. Try again.")
                continue
            return action


@njit(cache=True)
def _greedy_action(state):
    """Two-ply safety check using the exact history-bearing rule engine."""
    board = Board(2)
    board.copy_state(state, False)
    actions = np.flatnonzero(board.valid_moves(0))
    target = 8 if board.get_a1_defender() == 0 else 0
    best_action = -1
    best_rank = (-1, -1, -10)
    for action in actions:
        board.copy_state(state, False)
        nx, ny = action_destination(action)
        capture = int(state[ny, nx, 0] < 0)
        board.make_move(action, 0, 0)
        result = board.check_end_game(1)
        if result[0] == 1:
            # Ascending actions also break ties between immediate wins.
            return int(action)
        safe = 1
        if not result.any():
            child = board.get_state()
            replies = np.flatnonzero(board.valid_moves(1))
            for reply in replies:
                board.copy_state(child, False)
                board.make_move(reply, 1, 0)
                if board.check_end_game(0)[1] == 1:
                    safe = 0
                    break
        distance = max(abs(target - nx), abs(target - ny))
        rank = (safe, capture, -distance)
        if rank > best_rank:
            best_rank = rank
            best_action = int(action)
    return best_action


class GreedyPlayer:
    """Win, prevent a one-move loss, capture, approach goal, lowest action ID.

    Distance is the moved piece's Chebyshev distance (king moves) to its target.
    Draws have no winning reply; all rewards and termination rules stay intact.
    """

    def __init__(self, game):
        self.game = game

    def play(self, board, nb_moves=0):
        _legal_actions(self.game, board)
        return _greedy_action(board)
