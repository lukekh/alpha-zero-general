"""Independent acceptance fixtures for movement, captures, and official wins."""

import unittest

import numpy as np
from numba import njit

from intransitive.IntransitiveConstants import (
    E, N, NE, S, W, encode_action,
)
from intransitive.IntransitiveLogicNumba import Board, raw_movement_mask


def load_position(board, pieces, next_player=0, a1_defender=0):
    """Build a storage-valid synthetic position with a one-entry history."""
    state = board.get_state()
    state[:] = 0
    state[:, :, 0] = pieces
    state[:, :, 1] = pieces
    meta = state[:, :, 32]
    meta.flat[0] = 1
    meta.flat[1] = next_player
    meta.flat[2] = a1_defender
    meta.flat[4] = 1
    meta.flat[10] = next_player
    board.copy_state(state, True)
    return state


@njit
def compiled_rules(board, state, action):
    board.copy_state(state, True)
    legal_before = board.valid_moves(0)
    next_player = board.make_move(action, 0, random_seed=17)
    result = board.check_end_game(next_player)
    return legal_before, board.get_state(), next_player, result


class MovementAndCaptures(unittest.TestCase):
    def setUp(self):
        self.board = Board(2)

    def test_all_type_pairings_for_both_colours(self):
        # RPS type numbers cycle through their sole capturable defender.
        for player, sign in ((0, 1), (1, -1)):
            for attacker in (1, 2, 3):
                for defender in (1, 2, 3):
                    pieces = np.zeros((9, 9), dtype=np.int8)
                    pieces[4, 4] = sign * attacker
                    pieces[4, 5] = -sign * defender
                    mask = raw_movement_mask(pieces, player)
                    self.assertEqual(
                        bool(mask[encode_action(4, 4, E)]),
                        defender == attacker % 3 + 1,
                        (player, attacker, defender),
                    )
                    pieces[4, 5] = sign * defender
                    self.assertFalse(raw_movement_mask(pieces, player)[encode_action(4, 4, E)])

    def test_edges_and_diagonal_clearance(self):
        pieces = np.zeros((9, 9), dtype=np.int8)
        pieces[0, 0] = 1
        mask = raw_movement_mask(pieces, 0)
        expected = {encode_action(0, 0, N), encode_action(0, 0, NE), encode_action(0, 0, E)}
        self.assertEqual(set(np.flatnonzero(mask)), expected)

        pieces = np.zeros((9, 9), dtype=np.int8)
        pieces[3, 3] = 1
        pieces[3, 4] = 3
        pieces[4, 3] = 2
        self.assertTrue(raw_movement_mask(pieces, 0)[encode_action(3, 3, NE)])

    def test_noncapture_and_optional_capture(self):
        pieces = np.zeros((9, 9), dtype=np.int8)
        pieces[4, 4] = 1
        pieces[4, 5] = -2
        pieces[0, 8] = -3
        original = load_position(self.board, pieces)
        legal = self.board.valid_moves(0)
        self.assertTrue(legal[encode_action(4, 4, E)])
        self.assertTrue(legal[encode_action(4, 4, W)])

        next_player = self.board.make_move(encode_action(4, 4, W), 0, 0)
        self.assertEqual(next_player, 1)
        self.assertEqual(self.board.get_next_player(), 1)
        self.assertEqual(self.board.get_total_ply(), 1)
        self.assertEqual(self.board.get_no_capture_count(), 1)
        self.assertEqual(np.count_nonzero(self.board.get_board()), 3)
        np.testing.assert_array_equal(original[:, :, 0], pieces)

    def test_capture_removes_only_defender_and_preserves_attacker(self):
        pieces = np.zeros((9, 9), dtype=np.int8)
        pieces[4, 4] = 1
        pieces[4, 5] = -2
        pieces[7, 7] = -1
        load_position(self.board, pieces)
        self.board.make_move(encode_action(4, 4, E), 0, 0)
        result = self.board.get_board()
        self.assertEqual(result[4, 4], 0)
        self.assertEqual(result[4, 5], 1)
        self.assertEqual(result[7, 7], -1)
        self.assertEqual(np.count_nonzero(result), 2)
        self.assertEqual(self.board.get_no_capture_count(), 0)
        self.assertEqual(self.board.get_history_length(), 1)

    def test_each_colour_executes_its_capture_cycle(self):
        for player, sign in ((0, 1), (1, -1)):
            for attacker in (1, 2, 3):
                pieces = np.zeros((9, 9), dtype=np.int8)
                pieces[4, 4] = sign * attacker
                pieces[4, 5] = -sign * (attacker % 3 + 1)
                load_position(self.board, pieces, next_player=player)
                self.assertEqual(self.board.make_move(encode_action(4, 4, E), player, 0), 1 - player)
                result = self.board.get_board()
                self.assertEqual(result[4, 4], 0)
                self.assertEqual(result[4, 5], sign * attacker)
                self.assertEqual(np.count_nonzero(result), 1)

    def test_public_action_errors_are_atomic(self):
        pieces = np.zeros((9, 9), dtype=np.int8)
        pieces[4, 4] = 1
        pieces[4, 5] = 1
        pieces[5, 4] = -1
        load_position(self.board, pieces)
        expected = self.board.get_state()
        bad_calls = (
            (-1, 0), (648, 0), (1.5, 0),
            (encode_action(0, 0, W), 0),
            (encode_action(3, 3, E), 0),
            (encode_action(4, 4, E), 0),
            (encode_action(4, 4, N), 0),
            (encode_action(4, 4, W), 1),
        )
        for action, player in bad_calls:
            with self.subTest(action=action, player=player), self.assertRaises(ValueError):
                self.board.make_move(action, player, 0)
            np.testing.assert_array_equal(self.board.get_state(), expected)


class OfficialWins(unittest.TestCase):
    def setUp(self):
        self.board = Board(2)

    def test_each_player_wins_by_entering_opponents_corner(self):
        pieces = np.zeros((9, 9), dtype=np.int8)
        pieces[7, 8] = 3
        pieces[1, 1] = -1
        load_position(self.board, pieces, next_player=0)
        next_player = self.board.make_move(encode_action(8, 7, N), 0, 0)
        np.testing.assert_array_equal(self.board.check_end_game(next_player), [1, -1])
        self.assertFalse(self.board.valid_moves(0).any())
        self.assertFalse(self.board.valid_moves(1).any())

        pieces = np.zeros((9, 9), dtype=np.int8)
        pieces[1, 0] = -2
        pieces[7, 7] = 1
        load_position(self.board, pieces, next_player=1)
        next_player = self.board.make_move(encode_action(0, 1, S), 1, 0)
        np.testing.assert_array_equal(self.board.check_end_game(next_player), [-1, 1])

    def test_occupied_goal_requires_legal_capture(self):
        for attacker, defender, allowed in ((1, 2, True), (1, 1, False), (1, 3, False)):
            pieces = np.zeros((9, 9), dtype=np.int8)
            pieces[7, 8] = attacker
            pieces[8, 8] = -defender
            pieces[1, 1] = -1
            load_position(self.board, pieces)
            action = encode_action(8, 7, N)
            self.assertEqual(bool(self.board.valid_moves(0)[action]), allowed)
            if allowed:
                self.board.make_move(action, 0, 0)
                np.testing.assert_array_equal(self.board.check_end_game(1), [1, -1])

    def test_own_corner_can_be_entered_and_left(self):
        pieces = np.zeros((9, 9), dtype=np.int8)
        pieces[1, 0] = 1
        pieces[7, 7] = -1
        load_position(self.board, pieces)
        enter = encode_action(0, 1, S)
        self.assertTrue(self.board.valid_moves(0)[enter])
        self.board.make_move(enter, 0, 0)
        self.assertEqual(self.board.check_end_game(1).tolist(), [0.0, 0.0])
        self.assertTrue(raw_movement_mask(self.board.get_board(), 0)[encode_action(0, 0, N)])
        self.board.make_move(encode_action(7, 7, W), 1, 0)
        self.board.make_move(encode_action(0, 0, N), 0, 0)
        self.assertEqual(self.board.get_board()[1, 0], 1)

    def test_transformed_corner_ownership_is_respected(self):
        pieces = np.zeros((9, 9), dtype=np.int8)
        pieces[0, 0] = 1
        pieces[7, 7] = -1
        load_position(self.board, pieces, next_player=1, a1_defender=1)
        np.testing.assert_array_equal(self.board.check_end_game(1), [1, -1])
        self.assertFalse(self.board.valid_moves(1).any())

    def test_no_pieces_and_blocked_army_are_stalemates(self):
        pieces = np.zeros((9, 9), dtype=np.int8)
        pieces[8, 0] = -1
        load_position(self.board, pieces, next_player=0)
        np.testing.assert_array_equal(self.board.check_end_game(0), [-1, 1])
        self.assertFalse(self.board.valid_moves(1).any())

        pieces = np.zeros((9, 9), dtype=np.int8)
        pieces[3:6, 3:6] = -1
        pieces[4, 4] = 1
        load_position(self.board, pieces, next_player=0)
        np.testing.assert_array_equal(self.board.check_end_game(0), [-1, 1])
        self.assertFalse(self.board.valid_moves(0).any())

    def test_losing_one_type_has_no_special_effect(self):
        pieces = np.zeros((9, 9), dtype=np.int8)
        pieces[4, 4] = 1
        pieces[7, 7] = -1
        load_position(self.board, pieces)
        self.assertTrue(self.board.valid_moves(0).any())
        np.testing.assert_array_equal(self.board.check_end_game(0), [0, 0])
        self.assertEqual(self.board.get_score(0), 1)

    def test_simultaneous_corner_winners_are_rejected_without_mutation(self):
        original = self.board.get_state()
        malformed = original.copy()
        malformed[0, 0, 0] = -1
        malformed[0, 0, 1] = -1
        malformed[8, 8, 0] = 1
        malformed[8, 8, 1] = 1
        with self.assertRaisesRegex(ValueError, "Both players"):
            self.board.copy_state(malformed, True)
        np.testing.assert_array_equal(self.board.get_state(), original)
        np.testing.assert_array_equal(malformed[0, 0, :2], [-1, -1])

    def test_compiled_board_path(self):
        pieces = np.zeros((9, 9), dtype=np.int8)
        pieces[7, 8] = 1
        pieces[1, 1] = -1
        state = load_position(self.board, pieces)
        legal, result_state, next_player, result = compiled_rules(
            self.board, state, encode_action(8, 7, N))
        self.assertTrue(legal[encode_action(8, 7, N)])
        self.assertEqual(next_player, 1)
        self.assertEqual(result_state[8, 8, 0], 1)
        np.testing.assert_array_equal(result, [1, -1])
        self.assertTrue(compiled_rules.nopython_signatures)


if __name__ == "__main__":
    unittest.main()
