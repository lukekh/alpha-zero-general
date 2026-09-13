"""Complete training triples and relative targets through real Coach episodes."""

import pickle
from types import SimpleNamespace
import unittest
import zlib

import numpy as np

from Coach import Coach
from intransitive.IntransitiveConstants import E, N, NE, S, W, encode_action
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveLogicNumba import Board, validate_state
from intransitive.IntransitiveSymmetries import (
    ACTION_PERMUTATIONS, inverse_symmetry, transform_action_vector,
    transform_player_vector, transform_state,
)
from intransitive.tests.test_draws import load_history, play, sparse_position
from intransitive.tests.test_game import relabel
from intransitive.tests.test_symmetries import fixture


def triple_key(triple):
    return tuple(array.tobytes() for array in triple)


def reference_augmentation(state, symmetry):
    """Array-operation oracle in the relative frame, independent of Board swaps."""
    out = state.copy()
    length = int(state[:, :, 32].flat[4])
    planes = state[:, :, :length + 1].copy()
    # Explicit type substitutions for C^0, C^1, C^2.
    types = np.array([[0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]])
    planes = np.sign(planes) * types[symmetry % 3, np.abs(planes)]
    if (symmetry // 3) % 2:
        planes = planes.transpose(1, 0, 2)
    if symmetry >= 6:
        planes = planes.transpose(1, 0, 2)[::-1, ::-1]
        out[:, :, 32].flat[2] = 1 - out[:, :, 32].flat[2]
    out[:, :, :length + 1] = planes
    return out


class TrainingAugmentation(unittest.TestCase):
    def setUp(self):
        self.game = IntransitiveGame()

    def weighted_policy(self, state):
        valid = self.game.getValidMoves(state, 0)
        policy = np.arange(1, 649, dtype=np.float32) * valid
        if policy.any():
            policy /= policy.sum()
        return policy, valid

    def test_all_twelve_full_histories_masks_and_inverse_policies(self):
        for length in (1, 5, 31):
            for defender in (0, 1):
                state = fixture(length, defender)
                policy, valid = self.weighted_policy(state)
                original = tuple(a.copy() for a in (state, policy, valid))
                triples = self.game.getSymmetries(state, policy, valid)
                self.assertEqual(len(triples), 12)
                self.assertEqual(len({triple_key(t) for t in triples}), 12)
                board = Board()
                board.copy_state(state, False)
                count = board.get_repetition_count()
                for symmetry, (b, p, v) in enumerate(triples):
                    with self.subTest(length=length, defender=defender, symmetry=symmetry):
                        validate_state(b)
                        np.testing.assert_array_equal(b, reference_augmentation(state, symmetry))
                        self.assertEqual(b[:, :, 32].flat[1], 0)
                        np.testing.assert_array_equal(v, self.game.getValidMoves(b, 0))
                        np.testing.assert_array_equal(p[ACTION_PERMUTATIONS[symmetry]], policy)
                        np.testing.assert_array_equal(v[ACTION_PERMUTATIONS[symmetry]], valid)
                        np.testing.assert_array_equal(
                            transform_action_vector(p, inverse_symmetry(symmetry)), policy)
                        self.assertAlmostEqual(float(p.sum()), float(policy.sum()), places=6)
                        self.assertFalse(p[~v].any())
                        board.copy_state(b, False)
                        self.assertEqual(board.get_repetition_count(), count)
                        self.assertEqual(board.get_no_capture_count(), length - 1)
                        # Undo in the absolute API, then return to the original mover frame.
                        restored = transform_state(b, inverse_symmetry(symmetry))
                        if restored[:, :, 32].flat[1]:
                            restored = relabel(restored)
                        np.testing.assert_array_equal(restored, state)
                for actual, saved in zip((state, policy, valid), original):
                    np.testing.assert_array_equal(actual, saved)

    def test_e_reflection_retains_square_actions_and_changes_goal(self):
        pieces = sparse_position()
        pieces[1, 1], pieces[4, 1] = 0, 1
        state = load_history(Board(), [pieces])
        policy = np.zeros(648, dtype=np.float32)
        policy[encode_action(1, 4, E)] = 1  # B5 -> C5
        valid = self.game.getValidMoves(state, 0)
        triples = self.game.getSymmetries(state, policy, valid)
        self.assertEqual(len(triples), 12)
        reflected, p, v = triples[6]
        self.assertEqual(reflected[7, 4, 0], 1)  # E8, same relative colour
        self.assertEqual(reflected[:, :, 32].flat[2], 1)  # mover now defends I9
        self.assertEqual(np.argmax(p), encode_action(4, 7, S))  # E8 -> E7
        self.assertTrue(v[np.argmax(p)])
        child, _ = self.game.getNextState(reflected, 0, int(np.argmax(p)))
        self.assertEqual(child[6, 4, 0], 1)
        self.assertEqual(child[7, 4, 0], 0)

    def test_deduplication_uses_history_policy_and_mask(self):
        pieces = sparse_position()  # D-invariant board
        state = load_history(Board(), [pieces])
        valid = self.game.getValidMoves(state, 0)
        uniform = valid.astype(np.float32) / valid.sum()
        triples = self.game.getSymmetries(state, uniform, valid)
        self.assertEqual(len(triples), 6)
        # D fixes this policy, but an extra asymmetric legal slot in the supplied
        # mask must keep both examples. This tests mask identity independently.
        diagonal = np.zeros(648, dtype=np.float32)
        diagonal[encode_action(1, 1, NE)] = 1
        partial_mask = diagonal.astype(bool)
        partial_mask[encode_action(1, 1, E)] = True
        self.assertEqual(len(self.game.getSymmetries(state, diagonal, partial_mask)), 12)
        weighted, _ = self.weighted_policy(state)
        policy_triples = self.game.getSymmetries(state, weighted, valid)
        self.assertEqual(len(policy_triples), 12)
        np.testing.assert_array_equal(policy_triples[0][0], policy_triples[3][0])
        self.assertFalse(np.array_equal(policy_triples[0][1], policy_triples[3][1]))
        previous = pieces.copy()
        previous[1, 1], previous[1, 2] = 0, 1
        history = load_history(Board(), [previous, previous, pieces])
        history_triples = self.game.getSymmetries(history, uniform, valid)
        self.assertEqual(len(history_triples), 12)
        np.testing.assert_array_equal(history_triples[0][0][:, :, 0],
                                      history_triples[3][0][:, :, 0])
        self.assertFalse(np.array_equal(history_triples[0][0], history_triples[3][0]))
        # A later call must still emit the example; deduplication is per input.
        self.assertEqual([triple_key(t) for t in triples],
                         [triple_key(t) for t in self.game.getSymmetries(state, uniform, valid)])

    def test_owned_outputs_and_invalid_shapes_or_mover(self):
        state = fixture(5, 0)
        policy, valid = self.weighted_policy(state)
        triples = self.game.getSymmetries(state, policy.tolist(), valid)
        saved = [tuple(a.copy() for a in t) for t in triples]
        for b, p, v in triples:
            self.assertEqual(b.dtype, np.int8)
            self.assertEqual(p.dtype, np.float32)
            self.assertEqual(v.dtype, np.bool_)
            for output, original in zip((b, p, v), (state, policy, valid)):
                self.assertFalse(np.shares_memory(output, original))
        for i, triple in enumerate(triples):
            for actual, expected in zip(triple, saved[i]):
                np.testing.assert_array_equal(actual, expected)
                actual[:] = 0
        for p, v in ((policy[:-1], valid), (policy, valid[:-1]),
                     (policy.reshape(81, 8), valid), (policy, valid.reshape(81, 8))):
            with self.assertRaises(ValueError):
                self.game.getSymmetries(state, p, v)
        with self.assertRaises(ValueError):
            self.game.getSymmetries(relabel(state), policy, valid)

    def test_exact_repetitions_survive_recanonicalization(self):
        board = Board()
        cycle = [encode_action(1, 4, N), encode_action(4, 7, W),
                 encode_action(1, 5, S), encode_action(3, 7, E)]
        for repetitions in (1, 2, 3):
            if repetitions > 1:
                play(board, cycle)
            state = board.get_state()
            policy, valid = self.weighted_policy(state)
            for b, _, _ in self.game.getSymmetries(state, policy, valid):
                probe = Board()
                probe.copy_state(b, False)
                self.assertEqual(probe.get_repetition_count(), repetitions)
                self.assertEqual(probe.get_no_capture_count(), (repetitions - 1) * 4)
                self.assertEqual(probe.get_terminal_reason(),
                                 "repetition" if repetitions == 3 else "ongoing")

    def test_absolute_targets_swap_only_for_e(self):
        for absolute in (np.array([1, -1]), np.array([-1, 1]), np.array([0.7, -0.2])):
            for symmetry in range(12):
                expected = absolute[::-1] if symmetry >= 6 else absolute
                np.testing.assert_array_equal(transform_player_vector(absolute, symmetry), expected)


class RecordingGame(IntransitiveGame):
    """Record the real episode's physical trajectory and augmented inputs."""
    def __init__(self, initial=None):
        super().__init__()
        self.initial = initial
        self.physical, self.batches = [], []

    def getInitBoard(self):
        state = super().getInitBoard() if self.initial is None else self.initial.copy()
        self.physical.append((state.copy(), 0))
        return state

    def getNextState(self, state, player, action, random_seed=0):
        child, next_player = super().getNextState(state, player, action, random_seed)
        self.physical.append((child.copy(), next_player))
        return child, next_player

    def getSymmetries(self, state, policy, valid):
        triples = super().getSymmetries(state, policy, valid)
        self.batches.append((state.copy(), triples))
        return triples


class ScriptedSearch:
    """Deterministic legal policies; Coach and all Game transitions are real."""
    def __init__(self, game, actions):
        self.game, self.actions, self.queries = game, iter(actions), []

    def getActionProb(self, canonical, temp):
        action = next(self.actions)
        assert self.game.getValidMoves(canonical, 0)[action]
        policy = np.zeros(648, dtype=np.float32)
        policy[action] = 1
        # Distinct nonzero targets expose accidental E swaps or physical-player rolls.
        value = (len(self.queries) + 1) / 20
        q = [value, -value]
        self.queries.append((canonical.copy(), q))
        return policy, q, True


class CoachAugmentation(unittest.TestCase):
    def check_episode(self, game, actions, expected, compressed=False):
        coach = Coach.__new__(Coach)
        coach.game = game
        coach.args = SimpleNamespace(no_compression=not compressed,
                                     temperature=[1., 1.], tempThreshold=10)
        search = ScriptedSearch(game, actions)
        coach.mcts = search
        examples = coach.executeEpisode()
        if compressed:
            self.assertTrue(all(isinstance(e, bytes) for e in examples))
            examples = [pickle.loads(zlib.decompress(e)) for e in examples]
        self.assertEqual(len(search.queries), len(actions))
        np.testing.assert_array_equal(game.getGameEnded(*game.physical[-1]), expected)
        offset = 0
        for turn, ((canonical, triples), (_, q)) in enumerate(zip(game.batches, search.queries)):
            physical, player = game.physical[turn]
            self.assertEqual(player, turn % 2)
            np.testing.assert_array_equal(canonical, relabel(physical) if player else physical)
            for b, p, v in triples:
                example = examples[offset]
                offset += 1
                for actual, target in zip(example, (b, p, expected if player == 0 else expected[::-1], v, q)):
                    np.testing.assert_array_equal(actual, target)
                self.assertEqual(example[0][:, :, 32].flat[1], 0)
        self.assertEqual(offset, len(examples))
        return examples

    def test_official_blue_first_selfplay_and_compressed_draw_targets(self):
        game = RecordingGame()
        cycle = [encode_action(1, 4, N), encode_action(4, 7, W),
                 encode_action(1, 5, S), encode_action(3, 7, E)]
        self.check_episode(game, cycle * 2, np.full(2, 1e-4, dtype=np.float32), compressed=True)
        np.testing.assert_array_equal(game.physical[0][0], Board().get_state())
        # Augmenting the entire episode cannot change a later official reset.
        np.testing.assert_array_equal(game.getInitBoard(), Board().get_state())

    def test_both_winners_keep_relative_outcome_and_q_order(self):
        pieces = np.zeros((9, 9), dtype=np.int8)
        pieces[8, 7], pieces[0, 1] = 1, -1  # H9 and B1, near opposite goals
        initial = load_history(Board(), [pieces])
        for winner, actions in (
            (0, [encode_action(7, 8, S), encode_action(1, 0, N), encode_action(7, 7, NE)]),
            (1, [encode_action(7, 8, S), encode_action(1, 0, W)]),
        ):
            expected = np.array([1, -1] if winner == 0 else [-1, 1], dtype=np.float32)
            game = RecordingGame(initial)
            self.check_episode(game, actions, expected)
            self.assertTrue(all(len(triples) == 12 for _, triples in game.batches))


if __name__ == "__main__":
    unittest.main()
