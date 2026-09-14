"""Masked teacher targets, provenance-safe data and resumable continuation."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.NNet import NNetWrapper
from intransitive.teacher_learning import (
    DATA_SCHEMA, TacticalHybridPlayer, augment_records, orbit_hash,
    sample_mix, training_tuple, validate_dataset,
)
from intransitive.tests.tactical_oracle import load_case
from intransitive.tests.test_network import args as basic_args
from intransitive.tests.test_tactics import CASES


class FixedPolicy:
    def __init__(self, action):
        self.action = action

    def predict(self, state, legal):
        policy = np.zeros_like(legal, dtype=np.float32)
        policy[self.action] = 1
        return policy, np.zeros(2, dtype=np.float32)


class TeacherLearningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(33)
        self.game = IntransitiveGame()

    def example(self, extended=False):
        state = self.game.getInitBoard()
        legal = self.game.getValidMoves(state, 0)
        policy = legal.astype(np.float32) / legal.sum()
        base = (state, policy, np.array([1., -1.], dtype=np.float32), legal,
                np.array([.4, -.4], dtype=np.float32))
        return base + (np.ones(2, dtype=np.bool_), np.zeros(2, dtype=np.bool_)) \
            if extended else base

    def test_value_and_q_masks_do_not_fabricate_teacher_targets(self):
        wrapper = NNetWrapper(self.game, basic_args())
        output = torch.tensor([[.2, -.3], [.5, -.7]], requires_grad=True)
        value = torch.tensor([[1., -1.], [-1., 1.]])
        q = torch.tensor([[-.8, .8], [.6, -.6]])
        value_mask = torch.tensor([[True, True], [False, False]])
        q_mask = torch.tensor([[False, False], [True, True]])
        loss = wrapper.loss_v(value, q, output, value_mask, q_mask)
        expected = torch.cat(((value[0]-output[0])**2, (q[1]-output[1])**2)).mean()
        torch.testing.assert_close(loss, expected)

        policy_only = wrapper.loss_v(value, q, output,
                                     torch.zeros_like(value_mask), torch.zeros_like(q_mask))
        policy_only.backward()
        torch.testing.assert_close(output.grad, torch.zeros_like(output))

    def test_mixed_legacy_and_teacher_records_train_together(self):
        settings = dict(basic_args(), batch_size=2, batches_per_epoch=1)
        wrapper = NNetWrapper(self.game, settings)
        before = wrapper.nnet.policy_head.weight.detach().clone()
        wrapper.train([self.example(), self.example(extended=True)])
        self.assertFalse(torch.equal(before, wrapper.nnet.policy_head.weight))
        self.assertEqual(wrapper.optimizer_updates, 1)

    def test_persistent_optimizer_round_trip_continues_moments_and_count(self):
        settings = dict(basic_args(), batch_size=2, batches_per_epoch=1,
                        persist_optimizer=True)
        wrapper = NNetWrapper(self.game, settings)
        examples = [self.example(extended=True)] * 2
        wrapper.train(examples)
        first_steps = [state['step'].item() for state in wrapper.optimizer.state.values()]
        with tempfile.TemporaryDirectory() as folder:
            wrapper.save_checkpoint(folder, 'phase.pt')
            saved = torch.load(Path(folder) / 'phase.pt', weights_only=False)
            self.assertEqual(saved['optimizer_updates'], 1)
            self.assertEqual(saved['intransitive_checkpoint']['optimizer_state'], 'persistent')
            loaded = NNetWrapper(self.game, settings)
            loaded.load_checkpoint(folder, 'phase.pt')
            self.assertIsNone(loaded.optimizer)
            self.assertIsNotNone(loaded._pending_optimizer_state)
            loaded.train(examples)
            self.assertEqual(loaded.optimizer_updates, 2)
            self.assertTrue(all(step == 1 for step in first_steps))
            self.assertTrue(all(state['step'].item() == 2
                                for state in loaded.optimizer.state.values()))
            # nn_version=-1 is an inference-only load and deliberately ignores
            # optimizer state while retaining the validated weights.
            inference = NNetWrapper(self.game, dict(settings, nn_version=-1,
                                                     persist_optimizer=False))
            inference.load_checkpoint(folder, 'phase.pt')
            self.assertIsNone(inference._pending_optimizer_state)

            bad = dict(saved)
            del bad['optim_state']
            torch.save(bad, Path(folder) / 'bad.pt')
            original = loaded.nnet
            with self.assertRaisesRegex(ValueError, 'missing optim_state'):
                loaded.load_checkpoint(folder, 'bad.pt')
            self.assertIs(loaded.nnet, original)

    def test_all_symmetries_and_split_orbits_are_isolated(self):
        records = []
        for index, split in zip((0, 40, 80), ('train', 'validation', 'test')):
            case = CASES[index]
            reference, expected = load_case(case)
            state = self.game.getCanonicalForm(reference.storage(), case['player'])
            legal = self.game.getValidMoves(state, 0)
            policy = np.zeros(648, dtype=np.float32)
            policy[expected] = 1
            records.append(dict(state=state, policy=policy, legal=legal,
                outcome=np.array([1., -1.], dtype=np.float32), split=split,
                trajectory_id=f'trajectory-{split}', ply=1, source='test',
                state_sha256='unused', orbit_sha256=orbit_hash(self.game, state)))
        data = dict(schema=DATA_SCHEMA, examples=augment_records(self.game, records))
        self.assertTrue(validate_dataset(data))
        for split, rows in data['examples'].items():
            self.assertEqual({row['symmetry'] for row in rows}, set(range(12)))
            self.assertTrue(all(not row['q_mask'].any() for row in rows))
            self.assertTrue(all(len(training_tuple(row)) == 7 for row in rows))

    def test_annealed_mix_records_effective_sampling_weight(self):
        teacher = [self.example(extended=True)] * 10
        selfplay = [self.example()] * 12
        mixed, report = sample_mix(teacher, selfplay, .25, 9)
        self.assertEqual((report['teacher_examples'], report['selfplay_examples']), (4, 12))
        self.assertEqual(report['effective_teacher_fraction'], .25)
        self.assertEqual(len(mixed), 16)

    def test_hybrid_uses_exact_win_and_loss_filter_without_mutation(self):
        immediate = CASES[0]
        reference, expected = load_case(immediate)
        state = self.game.getCanonicalForm(reference.storage(), immediate['player'])
        legal = np.flatnonzero(self.game.getValidMoves(state, 0))
        wrong = next(int(action) for action in legal if action != expected)
        before = state.copy()
        self.assertEqual(TacticalHybridPlayer(self.game, FixedPolicy(wrong)).play(state), expected)
        np.testing.assert_array_equal(state, before)

        defence = CASES[40]
        reference, expected = load_case(defence)
        state = self.game.getCanonicalForm(reference.storage(), defence['player'])
        legal = np.flatnonzero(self.game.getValidMoves(state, 0))
        wrong = next(int(action) for action in legal if action != expected)
        before = state.copy()
        self.assertEqual(TacticalHybridPlayer(self.game, FixedPolicy(wrong)).play(state), expected)
        np.testing.assert_array_equal(state, before)


if __name__ == '__main__':
    unittest.main()
