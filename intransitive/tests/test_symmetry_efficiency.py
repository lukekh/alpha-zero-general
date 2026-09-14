"""Guard real optimization budgets and the controlled comparison's data contract."""

import copy
import unittest

from intransitive import symmetry_efficiency as experiment
import numpy as np
import torch

from intransitive.benchmark_training import Metrics, MeasuredNet
from intransitive.NNet import NNetWrapper
from intransitive.tests.test_augmentation import reference_augmentation
from intransitive.tests.test_symmetries import fixture


class SymmetryEfficiency(unittest.TestCase):
    def test_enabled_covers_all_ids_and_disabled_is_owned_identity(self):
        enabled = experiment.baseline.BaselineGame()
        disabled = experiment.IdentityGame()
        enabled.metrics = Metrics()
        disabled.metrics = Metrics()
        for length in (1, 5, 31):
            for defender in (0, 1):
                state = fixture(length, defender)
                mask = enabled.getValidMoves(state, 0)
                policy = np.arange(1, 649, dtype=np.float32) * mask
                policy /= policy.sum()
                original = [a.copy() for a in (state, policy, mask)]
                on = enabled.getSymmetries(state, policy, mask)
                off = disabled.getSymmetries(state, policy, mask)
                self.assertEqual(len(on), 12)
                self.assertEqual(len(off), 1)
                for symmetry, triple in enumerate(on):
                    np.testing.assert_array_equal(triple[0], reference_augmentation(state, symmetry))
                for actual, expected, initial in zip(off[0], on[0], original):
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(actual, initial)
                    self.assertFalse(np.shares_memory(actual, expected))
                off[0][0].fill(0)
                np.testing.assert_array_equal(state, original[0])

    def test_actual_optimizer_steps_are_independent_of_replay_length(self):
        game = experiment.IdentityGame()
        state = game.getInitBoard()
        mask = game.getValidMoves(state, 0)
        example = (state, mask.astype(np.float32)/mask.sum(), np.zeros(2, np.float32),
                   mask, np.zeros(2, np.float32))
        for explicit, expected in ((None, 2), (5, 5)):
            with self.subTest(batches_per_epoch=explicit):
                args = dict(nn_version=2, learn_rate=.0003, epochs=1, batch_size=2,
                            no_compression=True, q_weight=.5)
                if explicit:
                    args['batches_per_epoch'] = explicit
                net = MeasuredNet(game, args)
                net.metrics = Metrics()
                before = experiment.parameter_digest(net)
                net.train([example] * 4)
                self.assertEqual(net.training[0]['updates'], expected)
                self.assertNotEqual(before, experiment.parameter_digest(net))
                self.assertTrue(all(torch.isfinite(p).all() for p in net.nnet.parameters()))

    def test_invalid_update_budgets_and_short_replay_fail(self):
        for batches in (0, -1, 1.5, True, 2):
            net = NNetWrapper(experiment.IdentityGame(), dict(nn_version=2,
                learn_rate=.0003, epochs=1, batch_size=64, batches_per_epoch=batches))
            with self.assertRaises(ValueError):
                net.train([])

    def test_pairing_rejects_changed_seeds_and_accounts_for_draws(self):
        on = [dict(opponent='random', index=i, model_colour='Blue', seeds=[i],
                   physical_first_player='Blue', outcome=result)
              for i, result in enumerate(('wins', 'draws', 'losses'))]
        off = copy.deepcopy(on)
        for row in off:
            row['outcome'] = 'draws'
        result = experiment.score_difference(on, off)
        self.assertEqual(result['on_minus_off'], 0.)
        self.assertEqual(result['games'], 3)
        self.assertLess(result['difference_95_hoeffding'][0], 0.)
        off[1]['seeds'] = [99]
        with self.assertRaises(RuntimeError):
            experiment.score_difference(on, off)

    def test_protocol_caps_equal_original_positions_and_fixed_updates(self):
        on, off = [experiment.budget_for(a) for a in experiment.ARMS]
        self.assertEqual(on['settings']['maxlenOfQueue'] // 12, off['settings']['maxlenOfQueue'])
        self.assertEqual(on['comparison']['updates_per_iteration'], (60, 105, 101, 107))
        self.assertEqual(on['comparison']['fixed_earlier_sha256'], off['comparison']['fixed_earlier_sha256'])


if __name__ == '__main__':
    unittest.main()
