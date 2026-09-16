"""Cross-language semantics, completed-depth labels and tactical parity."""
from dataclasses import replace
import unittest

import numpy as np

from intransitive.rust_teacher import BINARY, RustTeacher
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.heuristics import AlphaBetaPlayer, SearchConfig
from intransitive.heuristics.budget import Budget
from intransitive.heuristics.evaluation import Evaluator
from intransitive.heuristics.pressure import pressure_totals
from intransitive.heuristics.position import SearchPosition
from intransitive.tests.test_tactics import CASES
from intransitive.tests.tactical_oracle import load_case


@unittest.skipUnless(BINARY.exists(), 'Build rust_teacher with cargo build --release first')
class NativeTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rust = RustTeacher()
        cls.config = SearchConfig(max_depth=3, time_limit=30., node_limit=1_000_000_000,
            pressure_enabled=True, pressure_weight=10., pressure_radius=3,
            pvs_enabled=True, aspiration_enabled=True, ordering_enabled=True,
            compiled_ordering_enabled=True, depth_replacement_enabled=True, table_entries=50000)
        AlphaBetaPlayer(config=cls.config)._prepare()

    @classmethod
    def tearDownClass(cls):
        cls.rust.close()

    def test_legal_transitions_pressure_and_material_on_random_trajectories(self):
        rng = np.random.default_rng(20260916)
        game = IntransitiveGame(modelling_draws=False)
        state, side = game.getInitBoard(), 0
        for i in range(160):
            if game.getGameEnded(state, side).any():
                state, side = game.getInitBoard(), 0
            for radius in (3, 4):
                for weight in (0., 10.):
                    row = self.rust.inspect(state, radius, weight)
                    model = IntransitiveGame()
                    legal = np.flatnonzero(model.getValidMoves(state, side)).tolist()
                    self.assertEqual(row['legal'], legal)
                    np.testing.assert_array_equal(row['pressure'], pressure_totals(state[:, :, 0], radius))
                    if not model.getGameEnded(state, side).any():
                        config = replace(self.config, pressure_radius=radius, pressure_weight=weight)
                        expected = Evaluator(model, config).score(state, side, Budget(1000000, 10), proof={'status':'unknown'})
                        self.assertAlmostEqual(row['score'], expected, places=10)
            action = int(rng.choice(np.flatnonzero(game.getValidMoves(state, side))))
            native = self.rust.apply(state, action, modelling=False)
            state, side = game.getNextState(state, side, action)
            np.testing.assert_array_equal(state, native)

    def test_all_100_certified_tactics(self):
        for case in CASES:
            with self.subTest(case=case['id']):
                reference, expected = load_case(case)
                state = reference.storage()
                rust = self.rust.analyze(state, depth=3)
                python = AlphaBetaPlayer(config=self.config).analyze(state)
                self.assertTrue(rust['complete'])
                self.assertEqual(rust['action'], expected)
                self.assertEqual(rust['action'], python.action)
                self.assertAlmostEqual(rust['score'], python.score, places=8)

    def test_random_shallow_search_and_tied_actions(self):
        from intransitive.supervised_minimax import generate_position
        for index in range(9):
            stage = ('opening', 'midgame', 'endgame')[index%3]
            state, _ = generate_position(2026091600+index, stage)
            for depth in (1, 2, 3):
                with self.subTest(index=index, depth=depth):
                    rust = self.rust.analyze(state, depth=depth)
                    player = AlphaBetaPlayer(config=replace(self.config, max_depth=depth))
                    python = player.analyze(state)
                    self.assertTrue(rust['complete'])
                    self.assertAlmostEqual(rust['score'], python.score, places=8)
                    if rust['action'] != python.action:
                        child = SearchPosition(state)
                        child.push(rust['action'])
                        score, _ = player._search(child, python.completed_depth-1, -float('inf'), float('inf'), 1, Budget(1000000000, 30))
                        self.assertAlmostEqual(-score, python.score, places=8)

    def test_repetition_and_clock_are_modelling_only(self):
        from intransitive.tests.test_heuristics import position
        from intransitive.IntransitiveDisplay import parse_move
        game = IntransitiveGame(modelling_draws=False)
        state = position({'C3':1, 'G7':-1})
        side = 0
        moves = ('C3 C4', 'G7 G6', 'C4 C3', 'G6 G7')
        for i in range(84):
            row = self.rust.inspect(state)
            if i>=8:
                self.assertEqual(row['reason'], 'repetition')
                self.assertEqual(row['official_reason'], 'ongoing')
                self.assertFalse(self.rust.analyze(state, depth=2)['complete'])
            action = parse_move(moves[i%4])
            native = self.rust.apply(state, action, modelling=False)
            state, side = game.getNextState(state, side, action)
            np.testing.assert_array_equal(state, native)

    def test_incomplete_search_and_invalid_inputs_are_not_labels(self):
        state = IntransitiveGame().getInitBoard()
        for kwargs in (dict(seconds=0.), dict(node_limit=1)):
            row = self.rust.analyze(state, depth=6, **kwargs)
            self.assertFalse(row['complete'])
            self.assertEqual(row['completed_depth'], 0)
        for command in ('search bad', 'inspect 5 10 '+state.tobytes().hex(), 'inspect 3 10 00'):
            with self.assertRaises(ValueError):
                self.rust.request(command)
        with self.assertRaises(ValueError):
            self.rust.apply(state, 0)

    def test_no_capture_boundary_and_malformed_state_rejection(self):
        from intransitive.tests.test_heuristics import position
        state = position({'C3':1, 'G7':-1})
        meta = state[:, :, 82:84]
        meta.flat[3], meta.flat[4], meta.flat[5] = 80, 81, 80
        # Storage-valid distinct historical boards isolate the draw clock;
        # this fixture is not claimed to be a reachable synthetic trajectory.
        for i in range(81):
            state[:, :, i+1] = 0
            state[i//9, i%9, i+1] = 1
            meta.flat[10+i] = i%2
        state[:, :, 81] = state[:, :, 0]
        row = self.rust.inspect(state)
        self.assertEqual(row['reason'], 'no-capture limit')
        self.assertEqual(row['official_reason'], 'ongoing')
        bad = state.copy()
        bad[:, :, 82:84].flat[4] = 82
        with self.assertRaises(ValueError):
            self.rust.request('inspect 3 10 '+bad.tobytes().hex())


if __name__ == '__main__':
    unittest.main()
