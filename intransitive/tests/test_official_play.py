"""Official games outlive draw cutoffs; modelling searches still terminate."""

import unittest
from contextlib import redirect_stdout
import io

import numpy as np

from Arena import Arena
from MCTS import MCTS
from intransitive.IntransitiveConstants import E, N, S, W, encode_action
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveLogicNumba import Board, search_observation, validate_state
from intransitive.tests.test_draws import load_history, noncapture_actions, play, sparse_position


class OfficialPlayTests(unittest.TestCase):
    def tearDown(self):
        MCTS.reset_all_search_trees()

    def test_rolling_history_past_80_and_search_copy_isolation(self):
        board = Board(modelling_draws=False)
        load_history(board, [sparse_position()])
        play(board, noncapture_actions())
        at_limit = board.get_state()
        model = Board()
        model.copy_state(at_limit, True)
        self.assertEqual(model.get_terminal_reason(), 'no-capture limit')
        self.assertEqual(board.get_terminal_reason(), 'ongoing')
        self.assertTrue(board.valid_moves(0).any())
        observation = search_observation(at_limit)
        model.copy_state(observation, True)
        self.assertEqual(model.get_terminal_reason(), 'ongoing')
        self.assertEqual(model.get_total_ply(), 80)
        self.assertEqual(model.get_no_capture_count(), 0)
        np.testing.assert_array_equal(observation[:, :, 0], at_limit[:, :, 0])
        np.testing.assert_array_equal(board.get_state(), at_limit)
        board.make_move(encode_action(1, 3, E), 0)
        after = board.get_state()
        validate_state(after)
        self.assertEqual(board.get_total_ply(), 81)
        self.assertEqual(board.get_history_length(), 81)
        for i in range(80):
            np.testing.assert_array_equal(after[:, :, i + 1], at_limit[:, :, i + 2])
        self.assertFalse(board.check_end_game(1).any())
        # A rebased search still draws if a simulated continuation repeats.
        model.copy_state(search_observation(after), True)
        cycle = [encode_action(1, 5, E), encode_action(2, 3, W),
                 encode_action(2, 5, W), encode_action(1, 3, E)]
        play(model, cycle * 2)
        self.assertEqual(model.get_terminal_reason(), 'repetition')

    def test_official_arena_continues_past_modelling_draw_to_win(self):
        pieces = sparse_position()
        pieces[1, 1] = 0
        pieces[7, 6] = 1
        pieces[7, 7] = 0
        pieces[1, 2] = -1
        board = Board()
        load_history(board, [pieces])
        cycle = [encode_action(6, 7, W), encode_action(2, 1, E),
                 encode_action(5, 7, E), encode_action(3, 1, W)]
        play(board, cycle * 2)
        physical = board.get_state()
        calls = []

        def scripted(state, turn):
            self.assertFalse(IntransitiveGame().getGameEnded(state, 0).any())
            calls.append(turn)
            return {9: encode_action(6, 7, E), 10: encode_action(2, 1, E),
                    11: encode_action(7, 7, 1)}[turn]

        modelling = IntransitiveGame()
        arena = Arena(scripted, scripted, modelling, modelling_draws=False)
        encoded = arena.serialize_state(physical, 0, 8)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(arena.playGame(initial_state=encoded), 1)
        self.assertEqual(calls, [9, 10, 11])
        self.assertTrue(modelling.board.modelling_draws)
        with redirect_stdout(io.StringIO()):
            result = Arena(None, None, modelling).playGame(initial_state=encoded)
        self.assertAlmostEqual(result, 1e-4)

    def test_official_wins_still_stop_and_capture_resets_window(self):
        board = Board(modelling_draws=False)
        pieces = sparse_position()
        pieces[8, 8] = 1
        load_history(board, [pieces] * 81)
        self.assertEqual(board.get_terminal_reason(), 'corner')
        self.assertFalse(board.valid_moves(0).any())
        np.testing.assert_array_equal(search_observation(board.get_state()), board.get_state())
        pieces = sparse_position()
        pieces[1, 2] = -2
        load_history(board, [pieces] * 81)
        board.make_move(encode_action(1, 1, E), 0)
        self.assertEqual(board.get_no_capture_count(), 0)
        self.assertEqual(board.get_history_length(), 1)
        validate_state(board.get_state())


if __name__ == '__main__':
    unittest.main()
