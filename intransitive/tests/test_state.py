"""Independent fixtures for the state contract; run with unittest discovery."""

import unittest

import numpy as np
from numba import njit

from intransitive.IntransitiveConstants import (
    ACTION_SIZE, MAX_TOTAL_PLY, STATE_SHAPE, action_destination,
    action_stays_on_board, decode_action, encode_action, format_coordinate,
    parse_coordinate,
)
from intransitive.IntransitiveLogicNumba import (
    Board, action_size, deserialize_state, observation_size, serialize_state,
    validate_state,
)


@njit
def compiled_branch(board, state):
    board.copy_state(state, True)
    pieces = board.get_board()
    pieces[4, 1] = 0
    pieces[5, 1] = 3
    board.record_position(pieces, 1, False)
    return board.get_state(), board.get_total_ply()


class CoordinatesAndActions(unittest.TestCase):
    def test_coordinates(self):
        for y, row in enumerate("123456789"):
            for x, col in enumerate("ABCDEFGHI"):
                self.assertEqual(parse_coordinate(col + row), (x, y))
                self.assertEqual(parse_coordinate(col.lower() + row), (x, y))
                self.assertEqual(format_coordinate(x, y), col + row)
        for label in ("A0", "J1", "A10", "1A", " A1", "", None):
            with self.subTest(label=label), self.assertRaises(ValueError):
                parse_coordinate(label)
        for x, y in ((-1, 0), (9, 0), (0, 9), (0, -1), (0.5, 0)):
            with self.assertRaises(ValueError):
                format_coordinate(x, y)

    def test_all_actions(self):
        # Independently listed destination offsets in official compass order.
        deltas = [(0, 1), (1, 1), (1, 0), (1, -1),
                  (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        seen = set()
        off_board = 0
        for y in range(9):
            for x in range(9):
                for direction, (dx, dy) in enumerate(deltas):
                    action = encode_action(x, y, direction)
                    seen.add(action)
                    self.assertEqual(action, 8 * (9 * y + x) + direction)
                    self.assertEqual(decode_action(action), (x, y, direction))
                    self.assertEqual(action_destination(action), (x + dx, y + dy))
                    inside = 0 <= x + dx <= 8 and 0 <= y + dy <= 8
                    self.assertEqual(action_stays_on_board(action), inside)
                    off_board += not inside
        self.assertEqual(seen, set(range(648)))
        self.assertEqual(off_board, 104)
        self.assertEqual(ACTION_SIZE, action_size())
        self.assertEqual(decode_action(0), (0, 0, 0))
        self.assertEqual(decode_action(647), (8, 8, 7))

    def test_bad_actions(self):
        for action in (-1, 648, 1.5):
            with self.assertRaises(ValueError):
                decode_action(action)
        for args in ((9, 0, 0), (-1, 0, 0), (0, 9, 0),
                     (0, 0, -1), (0, 0, 8), (0.5, 0, 0)):
            with self.assertRaises(ValueError):
                encode_action(*args)


class StateContract(unittest.TestCase):
    def setUp(self):
        self.board = Board(2)

    def test_official_setup_and_initial_bytes(self):
        # Explicit Red fixture avoids deriving the expected setup by reflection.
        expected = np.zeros((9, 9), dtype=np.int8)
        positions = {
            3: ("B5", "C4", "D3", "E2"),
            1: ("B4", "C3", "D2"),
            2: ("C5", "D4", "E3"),
            -3: ("E8", "F7", "G6", "H5"),
            -1: ("F8", "G7", "H6"),
            -2: ("E7", "F6", "G5"),
        }
        for piece, labels in positions.items():
            for label in labels:
                x, y = parse_coordinate(label)
                self.assertEqual(expected[y, x], 0)
                expected[y, x] = piece
        state = self.board.get_state()
        self.assertEqual(state.shape, (9, 9, 33))
        self.assertEqual(observation_size(), STATE_SHAPE)
        self.assertEqual(state.dtype, np.int8)
        np.testing.assert_array_equal(state[:, :, 0], expected)
        self.assertEqual(np.count_nonzero(state[:, :, 0]), 20)
        for sign in (1, -1):
            for kind, count in ((1, 3), (2, 3), (3, 4)):
                self.assertEqual(np.count_nonzero(expected == sign * kind), count)
        self.assertEqual(expected[0, 0], 0)
        self.assertEqual(expected[8, 8], 0)
        self.assertEqual(self.board.get_next_player(), 0)
        self.assertEqual(self.board.get_a1_defender(), 0)
        self.assertEqual(self.board.get_total_ply(), 0)
        self.assertEqual(self.board.get_no_capture_count(), 0)
        self.assertEqual(self.board.get_history_length(), 1)
        self.assertEqual(self.board.get_history_player(0), 0)
        np.testing.assert_array_equal(self.board.get_history(0), expected)
        np.testing.assert_array_equal(state[:, :, 2:32], 0)
        expected_meta = np.zeros(81, dtype=np.int8)
        expected_meta[[0, 4]] = 1
        np.testing.assert_array_equal(state[:, :, 32].ravel(), expected_meta)
        validate_state(state)

    def test_capacity_and_capture_reset(self):
        initial = self.board.get_state()
        pieces = self.board.get_board()
        # Storage-level fixtures: record_position intentionally does not check moves.
        for ply in range(1, 31):
            self.board.record_position(pieces, ply % 2, False)
        full = self.board.get_state()
        self.assertEqual(self.board.get_history_length(), 31)
        self.assertEqual(self.board.get_no_capture_count(), 30)
        self.assertEqual(self.board.get_total_ply(), 30)
        for index in range(31):
            np.testing.assert_array_equal(self.board.get_history(index), pieces)
            self.assertEqual(self.board.get_history_player(index), index % 2)
        validate_state(full)
        with self.assertRaises(ValueError):
            self.board.record_position(pieces, 1, False)
        np.testing.assert_array_equal(self.board.get_state(), full)
        pieces[7, 4] = 0  # Remove Red paper in this storage fixture.
        self.board.record_position(pieces, 1, True)
        self.assertEqual(self.board.get_history_length(), 1)
        self.assertEqual(self.board.get_no_capture_count(), 0)
        self.assertEqual(self.board.get_total_ply(), 31)
        reset = self.board.get_state()
        np.testing.assert_array_equal(reset[:, :, 2:32], 0)
        np.testing.assert_array_equal(reset[:, :, 32].ravel()[11:], 0)
        np.testing.assert_array_equal(full[:, :, 0], initial[:, :, 0])
        for ply in range(32, 62):
            self.board.record_position(pieces, ply % 2, False)
        self.assertEqual(self.board.get_total_ply(), 61)
        self.assertEqual(self.board.get_history_length(), 31)
        validate_state(self.board.get_state())

    def test_total_ply_survives_127_and_serialization(self):
        # A loaded late-game fixture: total is independent of resettable clock.
        state = self.board.get_state()
        meta = state[:, :, 32]
        meta.flat[5] = 127
        meta.flat[1] = 1
        meta.flat[10] = 1
        meta.flat[2] = 1  # Canonical/transformed goal ownership is serialized too.
        self.board.copy_state(state, True)
        self.board.record_position(self.board.get_board(), 0, False)
        self.assertEqual(self.board.get_total_ply(), 128)
        self.assertEqual(self.board.get_no_capture_count(), 1)
        result = self.board.get_state()
        np.testing.assert_array_equal(result[:, :, 32].ravel()[5:10], [0, 1, 0, 0, 0])
        wire = serialize_state(result)
        self.assertEqual(len(wire), 2673)
        self.assertEqual(wire, result.tobytes(order="C"))
        restored = deserialize_state(wire)
        np.testing.assert_array_equal(restored, result)
        self.board.copy_state(restored, False)
        self.board.record_position(self.board.get_board(), 1, True)
        self.assertEqual(self.board.get_total_ply(), 129)
        self.assertEqual(self.board.get_no_capture_count(), 0)
        self.assertEqual(self.board.get_a1_defender(), 1)
        np.testing.assert_array_equal(restored, result)
        # Decode owns its data even for a mutable wire buffer.
        mutable = bytearray(wire)
        decoded = deserialize_state(mutable)
        mutable[:] = bytes(2673)
        np.testing.assert_array_equal(decoded, result)

    def test_maximum_ply_and_atomic_overflow(self):
        state = self.board.get_state()
        for i in range(5, 10):
            state[:, :, 32].flat[i] = 127
        self.board.copy_state(state, True)
        self.assertEqual(self.board.get_total_ply(), MAX_TOTAL_PLY)
        self.assertEqual(serialize_state(deserialize_state(serialize_state(state))), serialize_state(state))
        with self.assertRaises(ValueError):
            self.board.record_position(self.board.get_board(), 1, True)
        np.testing.assert_array_equal(self.board.get_state(), state)

    def test_ownership_queries_and_branches(self):
        parent = self.board.get_state()
        expected = parent.copy()
        self.board.copy_state(parent, True)
        self.board.state[0, 0, 0] = 1
        np.testing.assert_array_equal(parent, expected)
        self.board.copy_state(parent, False)
        state = self.board.get_state()
        pieces = self.board.get_board()
        history = self.board.get_history(0)
        state[:] = 0
        pieces[:] = 0
        history[:] = 0
        np.testing.assert_array_equal(parent, expected)
        np.testing.assert_array_equal(self.board.get_state(), expected)
        first, total = compiled_branch(self.board, parent)
        self.assertEqual(total, 1)
        self.assertTrue(compiled_branch.nopython_signatures)
        np.testing.assert_array_equal(parent, expected)
        saved = first.copy()
        self.board.copy_state(parent, False)
        self.board.record_position(self.board.get_board(), 1, True)
        np.testing.assert_array_equal(first, saved)
        np.testing.assert_array_equal(parent, expected)
        self.board.init_game()
        np.testing.assert_array_equal(first, saved)
        for index in (-1, 1, 0.5):
            with self.assertRaises(ValueError):
                self.board.get_history(index)
            with self.assertRaises(ValueError):
                self.board.get_history_player(index)

    def test_malformed_state_rejected_without_mutation(self):
        original = self.board.get_state()
        for offset, value in ((0, 2), (1, 2), (2, -1), (3, 1), (4, 0),
                              (4, 32), (5, -1), (10, 1), (11, 1), (41, 1)):
            bad = original.copy()
            bad[:, :, 32].flat[offset] = value
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                self.board.copy_state(bad, True)
            np.testing.assert_array_equal(self.board.get_state(), original)
        for plane in (0, 1, 2):
            bad = original.copy()
            bad[0, 0, plane] = 4
            with self.assertRaises(ValueError):
                serialize_state(bad)
        for bad in (original.astype(np.int16), np.zeros((8, 9, 33), dtype=np.int8)):
            with self.assertRaises(ValueError):
                validate_state(bad)
        for wire in (b"", bytes(2672), bytes(2674), bytes(2673)):
            with self.assertRaises(ValueError):
                deserialize_state(wire)
        with self.assertRaises(ValueError):
            Board(3)
        for pieces, player in ((self.board.get_board(), 0),
                               (np.zeros((8, 9), dtype=np.int8), 1),
                               (np.full((9, 9), 4, dtype=np.int8), 1)):
            with self.assertRaises(ValueError):
                self.board.record_position(pieces, player, False)
            np.testing.assert_array_equal(self.board.get_state(), original)


if __name__ == "__main__":
    unittest.main()
