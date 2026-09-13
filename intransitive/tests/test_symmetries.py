"""Exhaustive algebra/action checks and independent whole-state fixtures."""

import unittest

import numpy as np
from numba import njit

from intransitive.IntransitiveConstants import (
    action_destination, action_stays_on_board, decode_action, encode_action,
)
from intransitive.IntransitiveLogicNumba import Board, validate_state
from intransitive.IntransitiveSymmetries import (
    ACTION_PERMUTATIONS, INVERSE_ACTION_PERMUTATIONS, C, D, E,
    compose_symmetries, inverse_symmetry, symmetry_components, symmetry_id,
    transform_action, transform_action_vector, transform_coordinate,
    transform_player_vector, transform_state,
)


@njit
def compiled_round_trip(state, policy, values, symmetry):
    inverse = inverse_symmetry(symmetry)
    result = transform_state(state, symmetry)
    validate_state(result)
    return (transform_state(result, inverse),
            transform_action_vector(transform_action_vector(policy, symmetry), inverse),
            transform_player_vector(transform_player_vector(values, symmetry), inverse),
            transform_action(transform_action(123, symmetry), inverse),
            compose_symmetries(symmetry, inverse))


def fixture(length, defender):
    rng = np.random.default_rng(417)
    state = Board().get_state()
    state[:, :, :32] = 0
    state[:, :, 1:length + 1] = rng.integers(-3, 4, (9, 9, length), dtype=np.int8)
    # Repeated boards with equal and different movers must retain exact equality.
    if length > 4:
        state[:, :, 3] = state[:, :, 1]
        state[:, :, 4] = state[:, :, 1]
    state[:, :, 0] = state[:, :, length]
    meta = state[:, :, 32]
    meta.flat[1] = (length - 1) % 2
    meta.flat[2] = defender
    meta.flat[3] = length - 1
    meta.flat[4] = length
    meta.flat[5:10] = [7, 3, 2, 1, 1]
    for i in range(length):
        meta.flat[10 + i] = i % 2
    validate_state(state)
    return state


class Symmetries(unittest.TestCase):
    def test_group(self):
        self.assertEqual(compose_symmetries(C, compose_symmetries(C, C)), 0)
        self.assertEqual(compose_symmetries(D, D), 0)
        self.assertEqual(compose_symmetries(E, E), 0)
        for a in range(12):
            self.assertEqual(symmetry_id(*symmetry_components(a)), a)
            self.assertEqual(compose_symmetries(a, inverse_symmetry(a)), 0)
            self.assertEqual(compose_symmetries(a, 0), a)
            for b in range(12):
                ab = compose_symmetries(a, b)
                self.assertIn(ab, range(12))
                self.assertEqual(ab, compose_symmetries(b, a))
                for c in range(12):
                    self.assertEqual(compose_symmetries(ab, c),
                                     compose_symmetries(a, compose_symmetries(b, c)))

    def test_actions_exhaustively(self):
        slots = np.arange(648)
        for s in range(12):
            p, inv = ACTION_PERMUTATIONS[s], INVERSE_ACTION_PERMUTATIONS[s]
            np.testing.assert_array_equal(np.sort(p), slots)
            np.testing.assert_array_equal(inv[p], slots)
            np.testing.assert_array_equal(p[inv], slots)
            np.testing.assert_array_equal(inv, ACTION_PERMUTATIONS[inverse_symmetry(s)])
            for a in slots:
                mapped = transform_action(a, s)
                x, y, _ = decode_action(a)
                dx, dy = action_destination(a)
                # Independent affine fixture includes all off-board destinations.
                if (s // 3) % 2:
                    x, y, dx, dy = y, x, dy, dx
                if s // 6:
                    x, y, dx, dy = 8 - y, 8 - x, 8 - dy, 8 - dx
                self.assertEqual(decode_action(mapped)[:2], (x, y))
                self.assertEqual(action_destination(mapped), (dx, dy))
                self.assertEqual(action_stays_on_board(a), action_stays_on_board(mapped))
            for t in range(12):
                np.testing.assert_array_equal(ACTION_PERMUTATIONS[t, p],
                                              ACTION_PERMUTATIONS[compose_symmetries(s, t)])
        np.testing.assert_array_equal(ACTION_PERMUTATIONS[C], slots)

    def test_printed_coordinate_fixtures(self):
        # B5 -> C5 (east) becomes E2 -> E3 (north), or E8 -> E7 (south).
        self.assertEqual(transform_coordinate(1, 4, D), (4, 1))
        self.assertEqual(transform_coordinate(1, 4, E), (4, 7))
        self.assertEqual(transform_action(encode_action(1, 4, 2), D), encode_action(4, 1, 0))
        self.assertEqual(transform_action(encode_action(1, 4, 2), E), encode_action(4, 7, 4))

    def test_full_state(self):
        for length in (1, 5, 31):
            for defender in (0, 1):
                state = fixture(length, defender)
                original = state.copy()
                outputs = []
                for s in range(12):
                    out = transform_state(state, s)
                    validate_state(out)
                    self.assertFalse(np.shares_memory(out, state))
                    expected = state.copy()
                    for plane in range(length + 1):
                        board = state[:, :, plane].copy()
                        nonempty = board != 0
                        board[nonempty] = np.sign(board[nonempty]) * (1 + (np.abs(board[nonempty]) - 1 + s % 3) % 3)
                        if (s // 3) % 2:
                            board = board.T
                        if s // 6:
                            board = -board.T[::-1, ::-1]
                        expected[:, :, plane] = board
                    if s // 6:
                        meta = expected[:, :, 32]
                        meta.flat[1] = 1 - meta.flat[1]
                        for i in range(length):
                            meta.flat[10 + i] = 1 - meta.flat[10 + i]
                    np.testing.assert_array_equal(out, expected)
                    np.testing.assert_array_equal(transform_state(out, inverse_symmetry(s)), state)
                    outputs.append(out.tobytes())
                    for t in range(12):
                        np.testing.assert_array_equal(transform_state(out, t),
                                                      transform_state(state, compose_symmetries(s, t)))
                self.assertEqual(len(set(outputs)), 12)
                np.testing.assert_array_equal(state, original)

    def test_policy_mask_values_and_compiled_call(self):
        rng = np.random.default_rng(12)
        policy = rng.random(648)
        policy /= policy.sum()
        mask = rng.random(648) > 0.6
        values = np.array([0.25, -0.75])
        state = fixture(5, 1)
        for s in range(12):
            for vector in (policy, mask):
                before = vector.copy()
                out = transform_action_vector(vector, s)
                np.testing.assert_array_equal(out[ACTION_PERMUTATIONS[s]], vector)
                self.assertEqual(out.dtype, vector.dtype)
                self.assertAlmostEqual(float(out.sum()), float(vector.sum()))
                self.assertFalse(np.shares_memory(out, vector))
                np.testing.assert_array_equal(vector, before)
            out = transform_player_vector(values, s)
            np.testing.assert_array_equal(out, values[::-1] if s // 6 else values)
            self.assertFalse(np.shares_memory(out, values))
            result = compiled_round_trip(state, policy, values, s)
            for actual, expected in zip(result, (state, policy, values, 123, 0)):
                np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(values, [0.25, -0.75])
        self.assertTrue(compiled_round_trip.nopython_signatures)

    def test_only_allowed_transforms_and_official_initialization(self):
        state = Board().get_state()
        for s in range(12):
            # Both goal squares stay on the A1/I9 axis: no quarter rotations.
            self.assertIn(transform_coordinate(0, 0, s), ((0, 0), (8, 8)))
            # Every type map preserves the directed capture cycle.
            cycle = [1 + (piece - 1 + s % 3) % 3 for piece in (1, 2, 3)]
            self.assertIn(cycle, ([1, 2, 3], [2, 3, 1], [3, 1, 2]))
            transform_state(state, s)
        fresh = Board()
        np.testing.assert_array_equal(fresh.get_state(), state)
        self.assertEqual(fresh.get_next_player(), 0)
        self.assertEqual(np.count_nonzero(fresh.get_board() == 3), 4)
        self.assertEqual(transform_state(state, E)[:, :, 32].flat[1], 1)
        self.assertEqual(np.count_nonzero(transform_state(state, C)[:, :, 0] == 1), 4)

    def test_invalid_arguments(self):
        state = Board().get_state()
        for s in (-1, 12, 0.5):
            for function, args in ((symmetry_components, (s,)),
                                   (transform_state, (state, s)),
                                   (transform_action, (0, s)),
                                   (transform_action_vector, (np.zeros(648), s)),
                                   (transform_player_vector, (np.zeros(2), s))):
                with self.assertRaises(ValueError):
                    function(*args)
        for action in (-1, 648, 0.5):
            with self.assertRaises(ValueError):
                transform_action(action, 0)
        for function, array in ((transform_action_vector, np.zeros(647)),
                                (transform_player_vector, np.zeros(3))):
            with self.assertRaises(ValueError):
                function(array, 0)
        state[:, :, 32].flat[4] = 0
        with self.assertRaises(ValueError):
            transform_state(state, 0)


if __name__ == '__main__':
    unittest.main()
