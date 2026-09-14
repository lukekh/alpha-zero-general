"""Checkpoint compatibility, resumed optimization, exported inference and play."""

import copy
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
import torch

from Arena import Arena
from MCTS import MCTS
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveNNet import FEATURE_CONFIG
from intransitive.NNet import NNetWrapper
from intransitive.tests.test_network import args
from intransitive.tests.test_symmetries import fixture


ROOT = Path(__file__).resolve().parents[2]
ATOL = 2e-6
RTOL = 1e-5


class Checkpoints(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.game = IntransitiveGame()
        self.net = NNetWrapper(self.game, args())
        self.net.device['inference'] = 'cpu'
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = self.directory.name
        self.path = Path(self.folder, 'network.pt')
        # Include long histories, both canonical goal assignments and terminal
        # masks; synthetic storage fixtures deliberately exercise large ply IDs.
        self.states = [self.game.getInitBoard(), fixture(5, 0), fixture(5, 1),
                       fixture(29, 1), fixture(31, 0)]
        self.masks = [self.game.getValidMoves(s, 0) for s in self.states]

    def save(self, **settings):
        self.net.save_checkpoint(self.folder, self.path.name, settings)
        return torch.load(self.path, map_location='cpu', weights_only=False)

    def load(self, version=-1):
        net = NNetWrapper(self.game, args(version))
        checkpoint = net.load_checkpoint(self.folder, self.path.name)
        return net, checkpoint

    def assert_predictions(self, actual, expected):
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want, atol=ATOL, rtol=RTOL)

    def test_round_trip_reconstructs_architecture_features_and_settings(self):
        expected = [self.net.predict(s, m) for s, m in zip(self.states, self.masks)]
        settings = dict(cpuct=[1.25], temperature=[1., 0.5, 10], numMCTSSims=2)
        saved = self.save(**settings)
        self.assertEqual(saved['nn_args'], args())
        self.assertEqual(saved['intransitive_config'], FEATURE_CONFIG)
        self.assertEqual(saved['intransitive_checkpoint'], dict(
            format_version=2, game='intransitive', board_size=(9, 9, 84),
            action_size=648, num_players=2, optimizer_state='recreated',
            scheduler_state='recreated'))
        for key, value in settings.items():
            self.assertEqual(saved[key], value)
        # Full-model pickle is retained for existing consumers, but reconstruction
        # must use the validated state_dict and configuration, not that object.
        del saved['full_model']
        torch.save(saved, self.path)
        for version in (2, -1):
            with self.subTest(version=version):
                loaded, _ = self.load(version)
                loaded.device['inference'] = 'cpu'
                self.assertEqual(loaded.nnet.board_size, (9, 9, 84))
                self.assertEqual(loaded.nnet.action_size, 648)
                self.assertEqual(loaded.nnet.version, 2)
                self.assertEqual(loaded.number_params(), self.net.number_params())
                for key, value in self.net.nnet.state_dict().items():
                    torch.testing.assert_close(loaded.nnet.state_dict()[key], value, rtol=0, atol=0)
                for state, mask, want in zip(self.states, self.masks, expected):
                    self.assert_predictions(loaded.predict(state, mask), want)
                x = torch.tensor(np.stack(self.states), dtype=torch.float32)
                for actual, original in zip(loaded.nnet.extract_features(x), self.net.nnet.extract_features(x)):
                    torch.testing.assert_close(actual, original, rtol=0, atol=0)

    def test_legacy_feature_config_checkpoint_is_supported(self):
        saved = self.save()
        for key in ('intransitive_checkpoint', 'nn_args', 'nn_version'):
            del saved[key]
        # This reproduces the keys written by the issue #10 wrapper.
        torch.save(saved, self.path)
        loaded, _ = self.load()
        loaded.device['inference'] = 'cpu'
        self.assert_predictions(loaded.predict(self.states[0], self.masks[0]),
                                self.net.predict(self.states[0], self.masks[0]))

    def test_incompatible_metadata_and_weights_fail_without_replacing_model(self):
        saved = self.save()
        cases = [
            ('intransitive_checkpoint', 'format_version', 1),
            ('intransitive_checkpoint', 'format_version', 99),
            ('intransitive_checkpoint', 'game', 'other'),
            ('intransitive_checkpoint', 'board_size', (9, 9, 1)),
            ('intransitive_checkpoint', 'action_size', 81),
            ('intransitive_checkpoint', 'num_players', 3),
            ('intransitive_config', 'state_version', 99),
            ('intransitive_config', 'network_version', 99),
            ('intransitive_config', 'history_order', 'newest first'),
            ('intransitive_config', 'current_order', ('noncapture / 60',)),
            ('intransitive_config', 'action_order', 'direction first'),
            ('intransitive_config', 'trunk_channels', 32),
            ('intransitive_config', 'total_ply_max', 127),
            ('state_dict', 'ply_scale', torch.tensor(1.)),
            ('state_dict', 'policy_head.weight', torch.zeros(1)),
        ]
        original = self.net.nnet
        expected = self.net.predict(self.states[0], self.masks[0])
        for section, key, value in cases:
            with self.subTest(section=section, key=key):
                bad = copy.deepcopy(saved)
                bad[section][key] = value
                torch.save(bad, self.path)
                with self.assertRaisesRegex(ValueError, key):
                    self.net.load_checkpoint(self.folder, self.path.name)
                self.assertIs(self.net.nnet, original)
                self.assertFalse(self.net.requestKnowledgeTransfer)
                self.assert_predictions(self.net.predict(self.states[0], self.masks[0]), expected)

    def test_missing_metadata_weights_and_unreadable_files_fail_clearly(self):
        saved = self.save()
        for key, message in (('intransitive_config', 'feature configuration'),
                             ('state_dict', 'state_dict')):
            bad = dict(saved)
            del bad[key]
            torch.save(bad, self.path)
            with self.assertRaisesRegex(ValueError, message):
                self.load()
        self.path.write_bytes(b'not a checkpoint')
        with self.assertRaisesRegex(ValueError, 'Cannot load checkpoint.*network.pt'):
            self.load()
        self.path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'network.pt'):
            self.load()

    def test_reload_invalidates_existing_onnx_session(self):
        expected = self.net.predict(self.states[0], self.masks[0])
        self.save()
        loaded, _ = self.load()
        loaded.predict(self.states[0], self.masks[0])
        previous_session = loaded.ort_session
        with torch.no_grad():
            self.net.nnet.value_head[-1].bias.add_(0.5)
        changed = self.net.predict(self.states[0], self.masks[0])
        self.assertFalse(np.allclose(expected[1], changed[1]))
        self.save()
        loaded.load_checkpoint(self.folder, self.path.name)
        self.assertIsNone(loaded.ort_session)
        self.assert_predictions(loaded.predict(self.states[0], self.masks[0]), changed)
        self.assertIsNot(loaded.ort_session, previous_session)

    def test_resumed_training_recreates_optimizer_and_uses_caller_settings(self):
        state, mask = self.states[0], self.masks[0]
        example = (state, mask.astype(np.float32) / mask.sum(),
                   np.array([1., -1.]), mask, np.array([0.4, -0.4]))
        optimizers = []
        adamw = torch.optim.AdamW

        def record_optimizer(*positional, **keywords):
            optimizer = adamw(*positional, **keywords)
            self.assertFalse(optimizer.state)
            optimizers.append(optimizer)
            return optimizer

        with patch('GenericNNetWrapper.optim.AdamW', side_effect=record_optimizer):
            self.net.train([example] * 2)  # Exactly one update before saving.
            saved = self.save()
            self.assertNotIn('optim_state', saved)
            loaded, _ = self.load()
            loaded.args['learn_rate'] = 0.0002
            # Exercise ONNX -> resumed PyTorch training -> fresh ONNX inference.
            loaded.predict(state, mask)
            old_session = loaded.ort_session
            before = loaded.nnet.policy_head.weight.detach().clone()
            loaded.train([example] * 2)
            self.assertEqual(loaded.args['learn_rate'], 0.0002)
            self.assertIsNone(loaded.ort_session)
            self.assertFalse(torch.equal(before, loaded.nnet.policy_head.weight))
            for parameter in loaded.nnet.parameters():
                self.assertTrue(torch.isfinite(parameter).all())
                self.assertTrue(torch.isfinite(parameter.grad).all())
            after = loaded.predict(state, mask)
            self.assertIsNot(loaded.ort_session, old_session)
            loaded.device['inference'] = 'cpu'
            self.assert_predictions(after, loaded.predict(state, mask))
            self.assertEqual(len(optimizers), 2)
            self.assertIsNot(optimizers[0], optimizers[1])
            for optimizer in optimizers:
                self.assertTrue(optimizer.state)
                self.assertTrue(all(s['step'].item() == 1 for s in optimizer.state.values()))
            # Resaving a model loaded via nn_version=-1 records its real version.
            loaded.save_checkpoint(self.folder, 'resumed.pt')
            resaved = torch.load(Path(self.folder, 'resumed.pt'), weights_only=False)
            self.assertEqual(resaved['nn_args']['nn_version'], 2)
            self.assertEqual(resaved['nn_args']['learn_rate'], 0.0002)

    def test_dynamic_batch_and_single_onnx_match_pytorch_and_masks(self):
        self.save()
        loaded, _ = self.load()
        for state, mask in zip(self.states, self.masks):
            self.assert_predictions(loaded.predict(state, mask), self.net.predict(state, mask))
        self.check_session(loaded.ort_session)

    def check_session(self, session):
        for size in (1, 2, 5):
            states = np.stack(self.states[:size]).astype(np.float32)
            masks = np.stack(self.masks[:size])
            pi, value = session.run(None, {'board': states, 'valid_actions': masks})
            policy = np.exp(pi)
            self.assertEqual(policy.shape, (size, 648))
            self.assertEqual(value.shape, (size, 2))
            self.assertTrue(np.isfinite(pi).all() and np.isfinite(value).all())
            self.assertFalse(policy[~masks].any())
            np.testing.assert_allclose(policy.sum(1), masks.any(1).astype(float), atol=ATOL, rtol=RTOL)
            expected = [self.net.predict(s, m) for s, m in zip(states, masks)]
            self.assert_predictions((policy, value), tuple(np.stack(x) for x in zip(*expected)))
            self.net.nnet.eval()
            with torch.no_grad():
                batch_pi, batch_v = self.net.nnet(torch.from_numpy(states), torch.from_numpy(masks))
            self.assert_predictions((policy, value), (batch_pi.exp().numpy(), batch_v.numpy()))

    def test_standalone_export_cli_dynamic_batch_parity(self):
        self.save()
        output = Path(self.folder, 'network.onnx')
        result = subprocess.run([sys.executable, 'chkpt_to_onnx.py', '-i', str(self.path), '-o', str(output)],
                                cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        onnx.checker.check_model(onnx.load(output))
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.check_session(ort.InferenceSession(str(output), sess_options=options,
                                               providers=['CPUExecutionProvider']))

    def test_standalone_converter_rejects_incompatible_features(self):
        from chkpt_to_onnx import load_checkpoint

        saved = self.save()
        saved['intransitive_config']['state_version'] = 99
        torch.save(saved, self.path)
        with self.assertRaisesRegex(ValueError, 'state_version'):
            load_checkpoint(str(self.path))

    def test_unrelated_game_shared_checkpoint_and_export_regression(self):
        from chkpt_to_onnx import export_onnx, load_checkpoint
        from GameSwitcher import import_game

        Game, Net, _, _ = import_game('santorini')
        game = Game()
        original = Net(game, args(88))
        original.device['inference'] = 'cpu'
        state = game.getCanonicalForm(game.getInitBoard(), 0)
        mask = game.getValidMoves(state, 0)
        expected = original.predict(state, mask)
        original.save_checkpoint(self.folder, 'santorini.pt')
        loaded = Net(game, args(-1))
        loaded.load_checkpoint(self.folder, 'santorini.pt')
        self.assert_predictions(loaded.predict(state, mask), expected)
        output = Path(self.folder, 'santorini.onnx')
        export_onnx(load_checkpoint(str(Path(self.folder, 'santorini.pt'))), str(output))
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(output), sess_options=options, providers=['CPUExecutionProvider'])
        pi, value = session.run(None, {'board': state[None].astype(np.float32), 'valid_actions': mask[None]})
        self.assert_predictions((np.exp(pi[0]), value[0]), expected)
        with self.assertRaises(FileNotFoundError):
            loaded.load_checkpoint(self.folder, 'missing.pt')

    def test_pit_checkpoint_opponents_play_complete_games_as_both_colours(self):
        import pit

        pit_args = SimpleNamespace(game='intransitive', numMCTSSims=2, fpu=None, cpuct=None)
        # Verify both bare wrapper saves and the list-valued Coach search settings.
        for settings in ({}, dict(cpuct=[1.25], temperature=[1., 0.5, 10], numMCTSSims=2)):
            self.save(**settings)
            with patch.object(pit, 'game', None):
                model = pit.create_player(str(self.path), pit_args)
                baseline = pit.create_player('random', pit_args)
                counts = [0, 0]

                def checked(callback, index):
                    def choose(state, turn):
                        self.assertLessEqual(turn, 1600)
                        untouched = state.copy()
                        action = callback(state, turn)
                        np.testing.assert_array_equal(state, untouched)
                        self.assertTrue(self.game.getValidMoves(state, 0)[action])
                        counts[index] += 1
                        return action
                    return choose

                arena = Arena(checked(model, 0), checked(baseline, 1), pit.game)
                for swapped in (False, True):
                    try:
                        result = arena.playGame(other_way=swapped)
                        self.assertIn(result, (1., -1., np.float32(1e-4)))
                    finally:
                        MCTS.reset_all_search_trees()
                self.assertGreater(counts[0], 0)
                self.assertGreater(counts[1], 0)


if __name__ == '__main__':
    unittest.main()
