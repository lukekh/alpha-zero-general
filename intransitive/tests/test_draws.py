"""Exact physical repetition, modelling limits, and compiled branch isolation."""

import unittest

import numpy as np
from numba import njit

from intransitive.IntransitiveConstants import E, N, S, W, encode_action
from intransitive.IntransitiveLogicNumba import (
    Board, deserialize_state, raw_movement_mask, serialize_state, validate_state,
)
from intransitive.IntransitiveSymmetries import transform_state


def load_history(board, positions, first_player=0, a1_defender=0):
    """Hand-built storage fixture; historical moves need not be reachable."""
    state = np.zeros((9, 9, 33), dtype=np.int8)
    for i, pieces in enumerate(positions):
        state[:, :, i + 1] = pieces
    state[:, :, 0] = positions[-1]
    meta = state[:, :, 32]
    meta.flat[0] = 1
    meta.flat[1] = (first_player + len(positions) - 1) % 2
    meta.flat[2] = a1_defender
    meta.flat[3] = len(positions) - 1
    meta.flat[4] = len(positions)
    meta.flat[5] = len(positions) - 1
    for i in range(len(positions)):
        meta.flat[10 + i] = (first_player + i) % 2
    board.copy_state(state, True)
    return state


def sparse_position():
    pieces = np.zeros((9, 9), dtype=np.int8)
    pieces[1, 1] = 1
    pieces[7, 7] = -1
    return pieces


def noncapture_actions():
    """Two separated, non-revisiting paths: exactly 15 moves per player."""
    blue = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
            (6, 2), (5, 2), (4, 2), (3, 2), (2, 2), (1, 2),
            (1, 3), (2, 3), (3, 3), (4, 3)]
    red = [(7, 7), (6, 7), (5, 7), (4, 7), (3, 7), (2, 7),
           (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (7, 6),
           (7, 5), (6, 5), (5, 5), (4, 5)]
    directions = {(1, 0): E, (-1, 0): W, (0, 1): N, (0, -1): S}
    actions = []
    for i in range(15):
        for path in (blue, red):
            x, y = path[i]
            nx, ny = path[i + 1]
            actions.append(encode_action(x, y, directions[nx - x, ny - y]))
    return actions


def play(board, actions):
    for action in actions:
        board.make_move(action, board.get_next_player())


@njit
def compiled_transition(board, state, action, copy):
    board.copy_state(state, copy)
    player = board.get_next_player()
    board.make_move(action, player, random_seed=17)
    return (board.get_state(), board.check_end_game(1 - player),
            board.get_terminal_reason(), board.get_repetition_count(),
            board.valid_moves(0), board.valid_moves(1))


class ModellingDraws(unittest.TestCase):
    def setUp(self):
        self.board = Board()

    def assert_ongoing(self):
        self.assertEqual(self.board.get_terminal_reason(), "ongoing")
        self.assertFalse(self.board.check_end_game(self.board.get_next_player()).any())
        self.assertTrue(self.board.valid_moves(self.board.get_next_player()).any())

    def assert_draw(self, reason):
        result = self.board.check_end_game(self.board.get_next_player())
        np.testing.assert_array_equal(result, np.array([1e-4, 1e-4], dtype=np.float32))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.any())
        self.assertEqual(self.board.get_terminal_reason(), reason)
        for player in (0, 1):
            self.assertFalse(self.board.valid_moves(player).any())
        # Terminal-aware actions must reject even moves allowed by raw mobility.
        state = self.board.get_state()
        player = self.board.get_next_player()
        action = int(np.flatnonzero(raw_movement_mask(state[:, :, 0], player))[0])
        with self.assertRaisesRegex(ValueError, "Illegal action"):
            self.board.make_move(action, player)
        np.testing.assert_array_equal(self.board.get_state(), state)

    def test_initial_second_and_third_nonconsecutive_occurrence(self):
        initial = self.board.get_board()
        cycle = [encode_action(1, 4, N), encode_action(4, 7, W),
                 encode_action(1, 5, S), encode_action(3, 7, E)]
        self.assertEqual(self.board.get_repetition_count(), 1)
        self.assert_ongoing()
        play(self.board, cycle)
        np.testing.assert_array_equal(self.board.get_board(), initial)
        self.assertEqual(self.board.get_repetition_count(), 2)
        self.assert_ongoing()
        play(self.board, cycle[:-1])
        self.assert_ongoing()
        self.board.make_move(cycle[-1], 1)
        self.assertEqual(self.board.get_repetition_count(), 3)
        self.assertEqual(self.board.get_no_capture_count(), 8)
        self.assert_draw("repetition")

    def test_exact_piece_type_colour_square_and_turn(self):
        current = sparse_position()
        for kind in ("type", "colour", "square", "turn"):
            with self.subTest(kind=kind):
                different = current.copy()
                if kind == "type":
                    different[1, 1] = 2
                elif kind == "colour":
                    different[1, 1] = -1
                elif kind == "square":
                    different[1, 1] = 0
                    different[1, 2] = 1
                else:
                    different[1, 1] = 2
                # Even-indexed history has the current turn. In the turn case
                # the exact board occurs only with the opposite mover before now.
                positions = [different, current, different, current, current]
                if kind != "turn":
                    positions = [different, different, different, different, current]
                load_history(self.board, positions)
                self.assertEqual(self.board.get_repetition_count(), 1)
                self.assert_ongoing()

    def test_same_type_piece_identities_are_interchangeable(self):
        pieces = sparse_position()
        pieces[1, 2] = 1
        load_history(self.board, [pieces])
        # Two Blue rocks exchange B2/C2 via B3/C3, while Red returns to H8.
        exchange = [encode_action(1, 1, N), encode_action(7, 7, W),
                    encode_action(2, 1, W), encode_action(6, 7, E),
                    encode_action(1, 2, E), encode_action(7, 7, W),
                    encode_action(2, 2, S), encode_action(6, 7, E)]
        play(self.board, exchange)
        self.assertEqual(self.board.get_repetition_count(), 2)
        np.testing.assert_array_equal(self.board.get_board(), pieces)
        self.assert_ongoing()
        play(self.board, exchange)
        self.assert_draw("repetition")

    def test_symmetry_equivalent_physical_positions_do_not_repeat(self):
        current = sparse_position()
        current[2, 4] = 3  # Break diagonal symmetry.
        base = load_history(self.board, [current])
        for symmetry in range(1, 12):
            with self.subTest(symmetry=symmetry):
                different = transform_state(base, symmetry)[:, :, 0]
                self.assertFalse(np.array_equal(current, different))
                load_history(self.board, [different, current, different, current, current])
                self.assertEqual(self.board.get_repetition_count(), 1)
                self.assert_ongoing()

    def test_29_and_30_individual_noncaptures(self):
        load_history(self.board, [sparse_position()])
        actions = noncapture_actions()
        for ply, action in enumerate(actions, 1):
            self.assert_ongoing()
            self.board.make_move(action, (ply - 1) % 2)
            self.assertEqual(self.board.get_no_capture_count(), ply)
            self.assertEqual(self.board.get_history_length(), ply + 1)
            self.assertEqual(self.board.get_repetition_count(), 1)
        self.assertEqual(self.board.get_total_ply(), 30)
        self.assert_draw("no-capture limit")
        validate_state(self.board.get_state())

    def test_captures_on_29th_and_30th_move_reset_history(self):
        for capture_ply in (29, 30):
            with self.subTest(capture_ply=capture_ply):
                pieces = sparse_position()
                if capture_ply == 29:
                    pieces[3, 4] = -2  # Blue's last east step captures.
                else:
                    pieces[5, 4] = 2  # Red's last west step captures.
                load_history(self.board, [pieces])
                actions = noncapture_actions()
                play(self.board, actions[:capture_ply - 1])
                before = self.board.get_state()
                self.assertEqual(self.board.get_no_capture_count(), capture_ply - 1)
                self.board.make_move(actions[capture_ply - 1], self.board.get_next_player())
                self.assertEqual(self.board.get_total_ply(), capture_ply)
                self.assertEqual(self.board.get_no_capture_count(), 0)
                self.assertEqual(self.board.get_history_length(), 1)
                self.assertEqual(self.board.get_repetition_count(), 1)
                self.assertEqual(np.count_nonzero(self.board.get_board()), 2)
                after = self.board.get_state()
                np.testing.assert_array_equal(after[:, :, 1], after[:, :, 0])
                np.testing.assert_array_equal(after[:, :, 2:32], 0)
                np.testing.assert_array_equal(after[:, :, 32].ravel()[11:41], 0)
                self.assertEqual(np.count_nonzero(before[:, :, 0]), 3)
                self.assert_ongoing()
                # The first subsequent noncapture counts from the capture result.
                action = int(np.flatnonzero(self.board.valid_moves(self.board.get_next_player()))[0])
                self.board.make_move(action, self.board.get_next_player())
                self.assertEqual(self.board.get_no_capture_count(), 1)
                self.assertEqual(self.board.get_history_length(), 2)
                self.assertEqual(self.board.get_total_ply(), capture_ply + 1)
                validate_state(self.board.get_state())

    def test_repetition_precedes_simultaneous_capture_limit(self):
        pieces = sparse_position()
        different = pieces.copy()
        different[1, 1] = 2
        history = [different.copy() for _ in range(31)]
        for i in (0, 14, 30):
            history[i] = pieces.copy()
        load_history(self.board, history)
        self.assertEqual(self.board.get_repetition_count(), 3)
        self.assertEqual(self.board.get_no_capture_count(), 30)
        self.assert_draw("repetition")

    def test_official_wins_precede_both_draw_conditions(self):
        for reason in ("corner", "stalemate"):
            for winner in (0, 1):
                with self.subTest(reason=reason, winner=winner):
                    pieces = np.zeros((9, 9), dtype=np.int8)
                    if reason == "corner":
                        pieces = sparse_position()
                        pieces[8, 8] = 1 if winner == 0 else 0
                        pieces[0, 0] = -1 if winner == 1 else 0
                    else:
                        pieces[4, 4] = 1 if winner == 0 else -1
                    load_history(self.board, [pieces] * 31, first_player=1 - winner)
                    self.assertEqual(self.board.get_repetition_count(), 16)
                    self.assertEqual(self.board.get_terminal_reason(), reason)
                    expected = [1, -1] if winner == 0 else [-1, 1]
                    np.testing.assert_array_equal(self.board.check_end_game(1 - winner), expected)
                    self.assertFalse(self.board.valid_moves(0).any())
                    self.assertFalse(self.board.valid_moves(1).any())

    def test_corner_on_30th_move_wins(self):
        pieces = sparse_position()
        pieces[7, 7] = 0
        pieces[1, 0] = -1
        # Populate 29 different earlier snapshots to isolate the clock boundary.
        history = []
        for i in range(29):
            earlier = pieces.copy()
            earlier[1, 1] = 0
            earlier[3 + i // 9, i % 9] = 1
            history.append(earlier)
        load_history(self.board, history + [pieces])
        self.assert_ongoing()
        self.board.make_move(encode_action(0, 1, S), 1)
        self.assertEqual(self.board.get_no_capture_count(), 30)
        self.assertEqual(self.board.get_terminal_reason(), "corner")
        np.testing.assert_array_equal(self.board.check_end_game(0), [-1, 1])

    def test_stalemate_on_30th_move_wins(self):
        pieces = np.zeros((9, 9), dtype=np.int8)
        pieces[3:6, 3:6] = -1
        pieces[4, 4] = 1
        pieces[4, 3] = 0  # Blue's sole exit, which Red will close.
        pieces[4, 2] = -1
        history = []
        for y, x in np.argwhere(pieces == 0):
            if (x, y) in ((0, 0), (8, 8)):
                continue
            earlier = pieces.copy()
            earlier[4, 2] = 0
            earlier[y, x] = -1
            history.append(earlier)
            if len(history) == 29:
                break
        load_history(self.board, history + [pieces])
        self.assert_ongoing()
        self.assertTrue(raw_movement_mask(pieces, 0).any())
        self.board.make_move(encode_action(2, 4, E), 1)
        self.assertEqual(self.board.get_no_capture_count(), 30)
        self.assertEqual(self.board.get_terminal_reason(), "stalemate")
        np.testing.assert_array_equal(self.board.check_end_game(0), [-1, 1])

    def test_serialization_and_all_symmetries_preserve_draws(self):
        for draw in ("repetition", "no-capture limit"):
            load_history(self.board, [sparse_position()])
            if draw == "repetition":
                cycle = [encode_action(1, 1, E), encode_action(7, 7, W),
                         encode_action(2, 1, W), encode_action(6, 7, E)]
                play(self.board, cycle + cycle[:-1])
                final_action = cycle[-1]
            else:
                actions = noncapture_actions()
                play(self.board, actions[:-1])
                final_action = actions[-1]
            restored = deserialize_state(serialize_state(self.board.get_state()))
            self.board.copy_state(restored, False)
            self.assert_ongoing()
            self.board.make_move(final_action, self.board.get_next_player())
            terminal = self.board.get_state()
            for symmetry in range(12):
                with self.subTest(draw=draw, symmetry=symmetry):
                    transformed = transform_state(terminal, symmetry)
                    self.board.copy_state(transformed, False)
                    self.assert_draw(draw)
                    np.testing.assert_array_equal(self.board.get_state(), transformed)

    def test_compiled_siblings_keep_independent_draws_and_counters(self):
        pieces = sparse_position()
        pieces[4, 5] = 2  # Red can capture south or draw by moving west.
        load_history(self.board, [pieces])
        actions = noncapture_actions()
        play(self.board, actions[:-1])
        parent = self.board.get_state()
        original = parent.copy()
        for copy in (True, False):
            with self.subTest(copy=copy):
                drawn, result, reason, count, blue, red = compiled_transition(
                    self.board, parent, actions[-1], copy)
                drawn_saved = drawn.copy()
                self.assertEqual(reason, "no-capture limit")
                self.assertEqual(count, 1)
                self.assertTrue(result.any())
                self.assertFalse(blue.any() or red.any())
                captured, result, reason, count, _, _ = compiled_transition(
                    self.board, parent, encode_action(5, 5, S), copy)
                self.assertEqual(reason, "ongoing")
                self.assertFalse(result.any())
                self.assertEqual(count, 1)
                self.assertEqual(captured[:, :, 32].flat[3], 0)
                self.assertEqual(drawn[:, :, 32].flat[3], 30)
                np.testing.assert_array_equal(parent, original)
                np.testing.assert_array_equal(drawn, drawn_saved)
                self.board.copy_state(parent, copy)
                self.assertEqual(self.board.get_no_capture_count(), 29)
                self.assert_ongoing()
        self.assertTrue(compiled_transition.nopython_signatures)

    def test_compiled_repetition_branch_does_not_contaminate_sibling(self):
        cycle = [encode_action(1, 4, N), encode_action(4, 7, W),
                 encode_action(1, 5, S), encode_action(3, 7, E)]
        play(self.board, cycle + cycle[:-1])
        parent = self.board.get_state()
        original = parent.copy()
        for copy in (True, False):
            drawn, result, reason, count, _, _ = compiled_transition(
                self.board, parent, cycle[-1], copy)
            saved = drawn.copy()
            self.assertEqual(reason, "repetition")
            self.assertEqual(count, 3)
            np.testing.assert_array_equal(result, np.full(2, 1e-4, dtype=np.float32))
            ongoing, result, reason, count, _, _ = compiled_transition(
                self.board, parent, encode_action(3, 7, W), copy)
            self.assertEqual(reason, "ongoing")
            self.assertEqual(count, 1)
            self.assertFalse(result.any())
            self.assertEqual(ongoing[:, :, 32].flat[3], 8)
            np.testing.assert_array_equal(parent, original)
            np.testing.assert_array_equal(drawn, saved)


if __name__ == "__main__":
    unittest.main()
