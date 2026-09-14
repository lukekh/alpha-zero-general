"""Fixed-coordinate Game adapter for the shared two-player search pipeline."""

import numpy as np

from Game import Game
from .IntransitiveConstants import (
    ACTION_SIZE, METADATA_PLANE, META_NEXT_PLAYER, NUMBER_PLAYERS, STATE_SHAPE,
)
from .IntransitiveDisplay import move_to_str, print_board
from .IntransitiveLogicNumba import Board, serialize_state
from .IntransitiveSymmetries import (
    NUM_SYMMETRIES, transform_action_vector, transform_state,
)


class IntransitiveGame(Game):
    def __init__(self, modelling_draws=True):
        self.num_players = NUMBER_PLAYERS
        self.board = Board(NUMBER_PLAYERS, modelling_draws)

    def for_play(self):
        """Official game transitions; searches keep their own modelling game."""
        return IntransitiveGame(modelling_draws=False)

    def getSearchObservation(self, state):
        from .IntransitiveLogicNumba import search_observation
        return search_observation(state)

    def getInitBoard(self):
        self.board.init_game()
        return self.board.get_state()

    def getBoardSize(self):
        return STATE_SHAPE

    def getActionSize(self):
        return ACTION_SIZE

    def getNumberOfPlayers(self):
        return NUMBER_PLAYERS

    def getNextState(self, board, player, action, random_seed=0):
        self.board.copy_state(board, True)
        next_player = self.board.make_move(action, player, random_seed)
        return self.board.get_state(), next_player

    def getValidMoves(self, board, player):
        self.board.copy_state(board, False)
        return self.board.valid_moves(player)

    def getGameEnded(self, board, next_player):
        self.board.copy_state(board, False)
        return self.board.check_end_game(next_player)

    def getScore(self, board, player):
        self.board.copy_state(board, False)
        return self.board.get_score(player)

    def getRound(self, board):
        self.board.copy_state(board, False)
        return self.board.get_total_ply()

    def getCanonicalForm(self, board, player):
        self.board.copy_state(board, False)
        if player != self.board.get_next_player():
            raise ValueError("Player does not match state")
        self.board.swap_players(player)
        return self.board.get_state()

    def getSymmetries(self, board, pi, valid_actions):
        """Materialize distinct full triples in Coach's current-player frame.

        E swaps absolute players; colour-only recanonicalization restores the
        mover to 0 without undoing the spatial reflection or action mapping.
        Coach can therefore reuse its relative outcome and Q targets unchanged.
        """
        canonical = self.getCanonicalForm(board, 0)
        policy = np.array(pi, dtype=np.float32, copy=True)
        valid = np.array(valid_actions, dtype=np.bool_, copy=True)
        if policy.shape != (ACTION_SIZE,) or valid.shape != (ACTION_SIZE,):
            raise ValueError("Expected policy and mask vectors of 648 action slots")
        examples, seen = [], set()
        for symmetry in range(NUM_SYMMETRIES):
            transformed = transform_state(canonical, symmetry)
            player = int(transformed[:, :, METADATA_PLANE:].flat[META_NEXT_PLAYER])
            transformed = self.getCanonicalForm(transformed, player)
            transformed_policy = transform_action_vector(policy, symmetry)
            transformed_valid = transform_action_vector(valid, symmetry)
            key = (transformed.tobytes(), transformed_policy.tobytes(),
                   transformed_valid.tobytes())
            if key not in seen:
                seen.add(key)
                examples.append((transformed, transformed_policy, transformed_valid))
        return examples

    def stringRepresentation(self, board):
        # Search identity includes goals, counters, and history, unlike repetition.
        return serialize_state(board)

    def moveToString(self, move, current_player):
        return move_to_str(move, current_player)

    def printBoard(self, numpy_board):
        print_board(numpy_board)
