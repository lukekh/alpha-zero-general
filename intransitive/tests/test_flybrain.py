"""Response sampling, perspective, legality, and player entry points."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from intransitive.flybrain import DEFAULT_BANK, FEATURE_ORDER, FlybrainPlayer, UPSTREAM_COMMIT
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveConstants import decode_action, action_destination
from intransitive.reference_greedy import features


class FlybrainTests(unittest.TestCase):
    def setUp(self):
        self.game = IntransitiveGame()

    def test_real_bank_and_seeded_legal_moves_for_both_colours(self):
        first, second = (FlybrainPlayer(self.game, seed=7) for _ in range(2))
        self.assertEqual(first.metadata['neurons'], 165122)
        self.assertFalse(first.responses.flags.writeable)
        state, side = self.game.getInitBoard(), 0
        for _ in range(40):
            if self.game.getGameEnded(state, side).any():
                break
            before = state.copy()
            a = first.choose(state, side)
            self.assertEqual(a, second.choose(state, side))
            self.assertTrue(self.game.getValidMoves(state, side)[a])
            np.testing.assert_array_equal(state, before)
            state, side = self.game.getNextState(state, side, a)

    def test_invalid_bank_is_rejected_and_reload_preserves_working_bank(self):
        with np.load(DEFAULT_BANK, allow_pickle=False) as saved:
            original = {key: saved[key] for key in saved.files}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bank.npz'
            np.savez(path, **original)
            player = FlybrainPlayer(self.game, path)
            responses = player.responses.copy()
            wrong = json.loads(str(original['metadata'].item()))
            for data in (dict(original, wired=np.zeros((32, 6))),
                         dict(original, wired=np.full((32, 12), np.nan)),
                         dict(original, patterns=original['patterns'][::-1]),
                         dict(original, metadata=json.dumps(dict(wrong, features=list(reversed(FEATURE_ORDER)))))):
                np.savez(path, **data)
                with self.assertRaises(ValueError):
                    player.reload()
                np.testing.assert_array_equal(player.responses, responses)
            with self.assertRaises(FileNotFoundError):
                FlybrainPlayer(self.game, Path(folder) / 'absent.npz')

    def test_only_held_out_responses_are_sampled(self):
        with np.load(DEFAULT_BANK, allow_pickle=False) as saved:
            original = {key: saved[key] for key in saved.files}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bank.npz'
            wired = np.zeros((32, 12))
            # Opposite preferences in fitting vs held-out halves.
            wired[:, :6] = -100 * original['patterns'][:, 2, None]
            wired[:, 6:] = 100 * original['patterns'][:, 2, None]
            np.savez(path, **dict(original, wired=wired))
            player = FlybrainPlayer(self.game, path, seed=3)
            state = self.game.getInitBoard()
            for _ in range(20):
                self.assertEqual(features(state, player.play(state))['progress'], 1)

    def test_terminal_and_noncanonical_positions_are_rejected(self):
        from intransitive.tests.test_rules import load_position
        from intransitive.IntransitiveLogicNumba import Board
        pieces = np.zeros((9, 9), dtype=np.int8)
        pieces[8, 8], pieces[1, 1] = 1, -1
        terminal = load_position(Board(), pieces)
        player = FlybrainPlayer(self.game)
        with self.assertRaises(ValueError):
            player.play(terminal)
        state = self.game.getInitBoard()
        action = player.play(state)
        state, _ = self.game.getNextState(state, 0, action)
        with self.assertRaises(ValueError):
            player.play(state)

    def test_browser_and_pit_entry_points(self):
        from intransitive.play import GameSession, OpponentFactory
        import pit
        factory = OpponentFactory()
        self.assertIn('flybrain', factory.choices)
        session = GameSession(factory.create('flybrain'), human_player=1, opponent_factory=factory)
        result = session.update('ai', dict(revision=0))
        self.assertEqual(result['ply'], 1)
        self.assertEqual(result['opponent'], 'flybrain')
        with patch.object(pit, 'game', None):
            player = pit.create_player('flybrain', SimpleNamespace(game='intransitive', flybrain_seed=4))
            state = self.game.getInitBoard()
            self.assertTrue(self.game.getValidMoves(state, 0)[player(state, 0)])

    def test_upstream_features_and_exact_selection_code(self):
        """Optional source parity check; no simulator/build imports required."""
        source = Path(__file__).parents[1] / 'data/flybrain-intransitive'
        if not (source / 'experiment.py').exists():
            self.skipTest('Upstream checkout needed for source parity check')
        spec = importlib.util.spec_from_file_location('flybrain_reference_rules', source / 'rps2.py')
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        # Execute the two unmodified upstream selection functions without
        # importing its pandas/pyarrow simulation dependencies into our runtime.
        tree = ast.parse((source / 'experiment.py').read_text())
        funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('pattern', 'brain_player')]
        self.assertEqual(len(funcs), 2)
        namespace = dict(np=np, FEATURES=upstream.FEATURES, REPEATS=12, pick_best=upstream.pick_best)
        exec(compile(ast.Module(body=funcs, type_ignores=[]), str(source / 'experiment.py'), 'exec'), namespace)
        player = FlybrainPlayer(self.game, seed=23)
        self.assertEqual(player.metadata['upstream_commit'], UPSTREAM_COMMIT)
        bank = {key: dict(wired=player.responses[i]) for i, key in enumerate(np.ndindex(2,2,2,2,2))}
        choose = namespace['brain_player'](bank, 'wired')
        rng = np.random.default_rng(23)
        trajectory_rng = np.random.default_rng(17)
        state, side = self.game.getInitBoard(), 0
        checked = 0
        for _ in range(100):
            if self.game.getGameEnded(state, side).any():
                state, side = self.game.getInitBoard(), 0
            canonical = self.game.getCanonicalForm(state, side)
            legal = np.flatnonzero(self.game.getValidMoves(canonical, 0))
            reference = upstream.Game()
            reference.board = {(int(x), int(y)): ('blue' if state[y,x,0] > 0 else 'red',
                               {1:'R',2:'S',3:'P'}[abs(int(state[y,x,0]))])
                               for y,x in np.argwhere(state[:,:,0])}
            reference.turn = 'blue' if side == 0 else 'red'
            moves = [(decode_action(int(a))[:2], action_destination(int(a))) for a in legal]
            self.assertEqual(set(reference.legal_moves()), set(moves))
            for action, move in zip(legal, moves):
                self.assertEqual(features(canonical, int(action)), reference.features(move))
                checked += 1
            # Order candidates identically to compare identical RNG draws.
            # Legal order has no effect on the distribution of upstream play.
            view = SimpleNamespace(legal_moves=lambda: moves, features=reference.features)
            expected = choose(view, rng)
            actual = player.play(canonical)
            self.assertEqual((decode_action(actual)[:2], action_destination(actual)), expected)
            state, side = self.game.getNextState(state, side, int(trajectory_rng.choice(legal)))
        self.assertGreater(checked, 1000)


if __name__ == '__main__':
    unittest.main()
