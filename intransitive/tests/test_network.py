"""Feature oracles and real shared-wrapper optimization/inference coverage."""

import tempfile
import unittest

import numpy as np
import torch

from intransitive.IntransitiveConstants import MAX_TOTAL_PLY
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveLogicNumba import Board
from intransitive.IntransitiveNNet import IntransitiveNNet, PIECE_CODES
from intransitive.NNet import NNetWrapper
from intransitive.tests.test_draws import load_history, sparse_position
from intransitive.tests.test_game import relabel, with_ply
from intransitive.tests.test_symmetries import fixture


def args(version=2):
    return dict(nn_version=version, learn_rate=0.001, epochs=1, batch_size=2,
                no_compression=True, q_weight=0.5)


class HistoryNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()

    def setUp(self):
        torch.manual_seed(10)
        self.wrapper = NNetWrapper(self.game, args())
        self.net = self.wrapper.nnet.eval()

    def features(self, *states):
        return self.net.extract_features(torch.tensor(np.stack(states), dtype=torch.float32))

    def test_categorical_goals_and_scalar_oracle_all_history_lengths(self):
        for length in (1, 5, 31, 81):
            for defender in (0, 1):
                state = fixture(length, defender)
                current, history = self.features(state)
                self.assertEqual(tuple(current.shape), (1, 11, 9, 9))
                self.assertEqual(tuple(history.shape), (1, 81, 8, 9, 9))
                for i, code in enumerate(PIECE_CODES):
                    np.testing.assert_array_equal(current[0, i], state[:, :, 0] == code)
                    for slot in range(length):
                        np.testing.assert_array_equal(history[0, slot, i], state[:, :, slot + 1] == code)
                own_corner = (0, 0) if defender == 0 else (8, 8)
                other_corner = (8, 8) if defender == 0 else (0, 0)
                self.assertEqual(current[0, 6].sum(), 1)
                self.assertEqual(current[0, 7].sum(), 1)
                self.assertEqual(current[0, 6, *own_corner], 1)
                self.assertEqual(current[0, 7, *other_corner], 1)
                np.testing.assert_array_equal(history[0, :length, 6], 1)
                for slot in range(length):
                    np.testing.assert_array_equal(history[0, slot, 7], slot % 2)
                self.assertEqual(history[0, length:].count_nonzero(), 0)
                board = Board()
                board.copy_state(state, True)
                np.testing.assert_allclose(current[0, 8], (length - 1) / 80)
                np.testing.assert_allclose(current[0, 9], board.get_repetition_count() / 3)
                np.testing.assert_allclose(current[0, 10],
                    np.log1p(board.get_total_ply()) / np.log1p(MAX_TOTAL_PLY), rtol=1e-6)

    def test_same_board_distinguishes_history_order_turns_goals_and_clock(self):
        pieces = sparse_position()
        a, b = pieces.copy(), pieces.copy()
        a[1, 1], a[1, 2] = 0, 1
        b[7, 7], b[7, 6] = 0, -1
        first = load_history(Board(), [a, pieces, b, pieces, pieces])
        second = load_history(Board(), [b, pieces, a, pieces, pieces])
        current, history = self.features(first, second)
        torch.testing.assert_close(current[0], current[1])
        self.assertFalse(torch.equal(history[0], history[1]))
        with torch.no_grad():
            encoded = self.net.encode_history(history)
        self.assertFalse(torch.equal(encoded[0], encoded[1]))
        torch.testing.assert_close(encoded[0, :4], encoded[1, 8:12])
        # Compare same slot with the same board but a different historical mover.
        turns = first.copy()
        turns[:, :, 82:84].flat[10] = 1
        c, h = self.features(first, turns)
        self.assertFalse(torch.equal(h[0], h[1]))
        goals = first.copy()
        goals[:, :, 82:84].flat[2] = 1
        counters = first.copy()
        counters[:, :, 82:84].flat[3] = 20
        c, _ = self.features(first, goals, counters, with_ply(first, 128))
        self.assertFalse(torch.equal(c[0, 6:8], c[1, 6:8]))
        self.assertAlmostEqual(c[2, 8, 0, 0].item(), 20 / 80, places=6)
        self.assertNotEqual(c[0, 10, 0, 0], c[3, 10, 0, 0])

    def test_repetition_includes_turn_and_ignores_padding(self):
        pieces = sparse_position()
        for length, count in ((1, 1), (3, 2), (5, 3)):
            state = load_history(Board(), [pieces] * length)
            current, _ = self.features(state)
            self.assertAlmostEqual(current[0, 9, 0, 0].item(), count / 3, places=6)
        state = load_history(Board(), [pieces] * 5)
        changed = state.copy()
        changed[:, :, 82:84].flat[10] = 1
        current, _ = self.features(changed)
        self.assertAlmostEqual(current[0, 9, 0, 0].item(), 2 / 3, places=6)

    def test_padding_and_reserved_bytes_do_not_leak_through_encoder_bias(self):
        state = fixture(5, 0)
        noisy = state.copy()
        noisy[:, :, 6:82] = 3
        noisy[:, :, 82:84].flat[15:91] = 1
        noisy[:, :, 82:84].flat[91:] = 99
        c, h = self.features(state, noisy)
        torch.testing.assert_close(c[0], c[1])
        torch.testing.assert_close(h[0], h[1])
        with torch.no_grad():
            self.net.history_encoder[0].bias.fill_(4)
            encoded = self.net.encode_history(h)
        self.assertEqual(encoded[:, 5 * 4:].count_nonzero(), 0)
        torch.testing.assert_close(encoded[0], encoded[1])

    def test_square_major_action_mapping_exhaustive(self):
        directions = torch.empty(2, 8, 9, 9)
        for y in range(9):
            for x in range(9):
                for d in range(8):
                    directions[:, d, y, x] = 8 * (9 * y + x) + d
        logits = IntransitiveNNet.square_major_logits(directions)
        torch.testing.assert_close(logits, torch.arange(648).float().expand(2, -1))

    def test_finite_gradients_parameter_update_and_legal_mask(self):
        state = self.game.getInitBoard()
        valid = self.game.getValidMoves(state, 0)
        boards = torch.tensor(np.stack([state, state]), dtype=torch.float32)
        masks = torch.tensor(np.stack([valid, valid]))
        policy = masks.float() / masks.sum(1, keepdim=True)
        targets_v = torch.tensor([[1., -1.], [-1., 1.]])
        targets_q = torch.tensor([[0.6, -0.2], [-0.3, 0.7]])
        self.net.train()
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=0.001)
        before = self.net.policy_head.weight.detach().clone()
        pi, v = self.net(boards, masks)
        self.assertEqual(tuple(v.shape), (2, 2))
        self.assertTrue((v.abs() <= 1).all())
        torch.testing.assert_close(pi.exp().sum(1), torch.ones(2))
        self.assertEqual(pi.exp()[~masks].count_nonzero(), 0)
        loss = self.wrapper.loss_pi(policy, pi) + 0.25 * self.wrapper.loss_v(targets_v, targets_q, v)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in self.net.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(self.net.history_encoder[0].weight.grad.abs().sum(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, self.net.policy_head.weight))
        pi, v = self.net(boards, torch.zeros_like(masks))
        self.assertTrue(torch.isfinite(pi).all() and torch.isfinite(v).all())
        self.assertEqual(pi.exp().count_nonzero(), 0)

    def test_canonical_features_and_value_q_label_order(self):
        physical = fixture(5, 0)
        red_turn = relabel(physical)
        canonical = self.game.getCanonicalForm(red_turn, 1)
        c, h = self.features(physical, canonical)
        torch.testing.assert_close(c[0], c[1])
        torch.testing.assert_close(h[0], h[1])
        # Coach rolls physical Red's winning outcome/Q into mover-first order.
        absolute_v, absolute_q = np.array([-1., 1.]), np.array([-0.2, 0.6])
        v = torch.tensor(np.roll(absolute_v, -1)[None], dtype=torch.float32)
        q = torch.tensor(np.roll(absolute_q, -1)[None], dtype=torch.float32)
        expected = (v + 0.5 * q) / 1.5
        self.assertEqual(self.wrapper.loss_v(v, q, expected), 0)
        self.assertGreater(self.wrapper.loss_v(v, q, expected.flip(1)), 0)

    def test_shared_training_checkpoint_placeholder_and_onnx_dynamic_batch(self):
        states = [fixture(1, 0), fixture(5, 1), fixture(31, 0)]
        masks = [self.game.getValidMoves(s, 0) for s in states]
        initial = self.game.getInitBoard()
        legal = self.game.getValidMoves(initial, 0)
        example = (initial, legal.astype(np.float32) / legal.sum(),
                   np.array([1., -1.]), legal, np.array([0.5, -0.5]))
        before = self.net.policy_head.weight.detach().clone()
        self.wrapper.train([example] * 4)
        self.assertFalse(torch.equal(before, self.net.policy_head.weight))
        self.wrapper.device['inference'] = 'cpu'
        expected = [self.wrapper.predict(s, m) for s, m in zip(states, masks)]
        with tempfile.TemporaryDirectory() as directory:
            self.wrapper.save_checkpoint(directory, 'network.pt')
            for version in (2, -1):
                loaded = NNetWrapper(self.game, args(version))
                checkpoint = loaded.load_checkpoint(directory, 'network.pt')
                self.assertIsNotNone(checkpoint)
                self.assertEqual(loaded.nnet.version, 2)
                self.assertEqual(checkpoint['intransitive_config'], loaded.nnet.feature_config)
                # Actual wrapper predict exports the same forward graph.
                pi, v = loaded.predict(states[0], masks[0])
                np.testing.assert_allclose(pi, expected[0][0], atol=2e-6)
                np.testing.assert_allclose(v, expected[0][1], atol=2e-6)
                batch = loaded.ort_session.run(None, {
                    'board': np.stack(states).astype(np.float32),
                    'valid_actions': np.stack(masks),
                })
                np.testing.assert_allclose(np.exp(batch[0]), np.stack([e[0] for e in expected]), atol=2e-6)
                np.testing.assert_allclose(batch[1], np.stack([e[1] for e in expected]), atol=2e-6)

    def test_unsupported_version_fails_at_construction(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            NNetWrapper(self.game, args(1))

    def test_version_one_replay_has_a_clear_migration_error(self):
        with self.assertRaisesRegex(ValueError, 'migrate version-1 replay'):
            self.net.extract_features(torch.zeros(1, 9, 9, 33))


if __name__ == '__main__':
    unittest.main()
