"""Per-square attribution must sum to the evaluator's own module terms.

Attribution recomputes what `evaluation` sums, so the two implementations can
drift apart. These tests are the guard: any module whose squares stop adding up
to its term fails here rather than quietly painting a wrong heatmap.
"""
from dataclasses import replace
import unittest

import numpy as np

from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.heuristics.attribution import (
    DEFAULT_MODULES, attribute, module_active, module_weights,
)
from intransitive.heuristics.budget import Budget
from intransitive.heuristics.config import SearchConfig
from intransitive.heuristics.evaluation import MODULES, Evaluator
from intransitive.heuristics.pressure import pressure_squares, pressure_totals
from intransitive.tests.test_heuristics import position, unlimited


ALL_ON = dict(attack_enabled=True, defence_enabled=True, overload_enabled=True,
              pressure_enabled=True, runner_enabled=True, proof_depth=0)
CONFIGURATIONS = {
    'core': SearchConfig(proof_depth=0),
    'all': SearchConfig(**ALL_ON),
    'variable': SearchConfig(variable_material_enabled=True, count_weight=5., **ALL_ON),
    'signed': SearchConfig(count_weight=-40., advantage_weight=-12., attack_weight=60.,
                           defence_weight=-30., overload_weight=-7., pressure_weight=-3.,
                           runner_weight=-11., pressure_radius=3, **ALL_ON),
}


def random_positions(games=6, plies=25, seed=11):
    """Reachable positions from random legal play, including captures."""
    game, generator, states = IntransitiveGame(), np.random.default_rng(seed), []
    for _ in range(games):
        state, player = game.getInitBoard(), 0
        for _ in range(plies):
            if game.getGameEnded(state, player).any():
                break
            legal = np.flatnonzero(game.getValidMoves(state, player))
            if not legal.size:
                break
            state, player = game.getNextState(state, player, int(generator.choice(legal)))
            states.append(state.copy())
    return states


class AttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()
        cls.states = random_positions()

    def reports(self, config, states=None, modules=MODULES):
        for state in states or self.states:
            for side in (0, 1):
                budget = unlimited()
                explanation = Evaluator(self.game, config).explain(
                    state, side, budget, diagnostics=False)
                if explanation.get('terminal') or explanation['proof'].get('status') == 'proven':
                    continue
                yield state, side, explanation, attribute(
                    self.game, state, side, budget, config,
                    modules=modules, explanation=explanation)

    def test_every_module_sums_to_its_own_term(self):
        for label, config in CONFIGURATIONS.items():
            for state, side, _, report in self.reports(config):
                for name, module in report['modules'].items():
                    if module['residual'] is None:
                        continue
                    with self.subTest(configuration=label, module=name, side=side):
                        self.assertAlmostEqual(module['attributed'], module['term'],
                                               delta=1e-9 * max(1., abs(module['term'])))

    def test_scoring_modules_sum_to_the_unclipped_score(self):
        for label, config in CONFIGURATIONS.items():
            for state, side, explanation, report in self.reports(config):
                with self.subTest(configuration=label, side=side):
                    self.assertFalse(explanation['clipped'])
                    self.assertAlmostEqual(report['attributed_total'], explanation['raw_score'],
                                           delta=1e-9 * max(1., abs(explanation['raw_score'])))

    def test_attribution_is_antisymmetric_between_perspectives(self):
        config = CONFIGURATIONS['all']
        for state in self.states[:20]:
            budgets = (unlimited(), unlimited())
            blue = attribute(self.game, state, 0, budgets[0], config, modules=MODULES)
            red = attribute(self.game, state, 1, budgets[1], config, modules=MODULES)
            if not (blue['decomposed'] and red['decomposed']):
                continue
            np.testing.assert_allclose(blue['total'], [-value for value in red['total']], atol=1e-9)

    def test_inactive_modules_are_previews_without_a_residual(self):
        config = SearchConfig(attack_enabled=False, defence_enabled=False,
                              overload_enabled=False, pressure_enabled=False, proof_depth=0)
        _, _, _, report = next(self.reports(config, self.states[:1]))
        active = module_active(config)
        for name, module in report['modules'].items():
            with self.subTest(module=name):
                self.assertEqual(module['active'], active[name] or name == 'clear_run')
                if module['preview']:
                    self.assertIsNone(module['term'])
                    self.assertIsNone(module['residual'])
        # A preview still uses the configured coefficient, so it can be nonzero.
        self.assertEqual(report['modules']['attacking_position']['weight'],
                         config.attack_weight)

    def test_previews_are_excluded_from_the_board_total(self):
        config = SearchConfig(attack_enabled=False, defence_enabled=False,
                              overload_enabled=False, pressure_enabled=False, proof_depth=0)
        for _, _, explanation, report in self.reports(config, self.states[:12]):
            self.assertAlmostEqual(report['attributed_total'], explanation['raw_score'],
                                   delta=1e-9 * max(1., abs(explanation['raw_score'])))

    def test_material_attributes_one_weight_to_every_piece(self):
        config = SearchConfig(proof_depth=0)
        state = self.states[0]
        report = attribute(self.game, state, 0, unlimited(), config, modules=('piece_count',))
        squares = np.array(report['modules']['piece_count']['squares'])
        board = state[:, :, 0].ravel()
        np.testing.assert_allclose(squares[board > 0], config.count_weight)
        np.testing.assert_allclose(squares[board < 0], -config.count_weight)
        np.testing.assert_allclose(squares[board == 0], 0.)

    def test_variable_material_values_each_type_separately(self):
        config = SearchConfig(variable_material_enabled=True, count_weight=5., proof_depth=0)
        state = self.states[0]
        report = attribute(self.game, state, 0, unlimited(), config, modules=('piece_count',))
        module = report['modules']['piece_count']
        values = module['own']['detail']['piece_values']
        self.assertEqual(sorted(values), ['paper', 'rock', 'scissors'])
        squares = np.array(module['squares'])
        board = state[:, :, 0].ravel()
        for kind, name in enumerate(('rock', 'scissors', 'paper'), 1):
            held = squares[board == kind]
            if held.size:
                np.testing.assert_allclose(held, config.count_weight * values[name] / 100.)

    def test_a_binding_cap_is_reported_with_its_raw_feature(self):
        # Ten pieces a side: defence exceeds its ceiling of four in the opening.
        report = attribute(self.game, self.game.getInitBoard(), 0, unlimited(),
                           SearchConfig(**ALL_ON), modules=('defensive_position',))
        module = report['modules']['defensive_position']
        binding = [cap for cap in module['caps'] if cap['binding']]
        self.assertTrue(binding, 'the opening should saturate the defensive cap')
        for cap in binding:
            self.assertGreater(cap['raw'], cap['used'])
            self.assertEqual(cap['used'], cap['cap'])
        self.assertAlmostEqual(module['own']['used'], 4.)

    def test_capture_opportunity_lands_on_the_threatened_square(self):
        # Blue scissors on D4 can take Red paper on E5 and nothing answers it.
        state = position({'D4': 2, 'E5': -3, 'A2': 1, 'I8': -1})
        report = attribute(self.game, state, 0, unlimited(),
                           SearchConfig(**ALL_ON), modules=('attacking_position',))
        squares = np.array(report['modules']['attacking_position']['squares'])
        target = ord('E') - ord('A') + 9 * 4
        self.assertGreater(squares[target], 0.,
                           'the capture credit belongs to the square being attacked')

    def test_overload_spreads_over_its_distinct_defenders(self):
        config = SearchConfig(**ALL_ON)
        for state in self.states[:15]:
            budget = unlimited()
            report = attribute(self.game, state, 0, budget, config, modules=('overload',))
            module = report['modules'].get('overload')
            if module is None or not module['own']['used']:
                continue
            squares = np.array(module['squares'])
            self.assertAlmostEqual(float(squares.sum()), module['attributed'], places=9)
            break

    def test_terminal_and_proven_positions_carry_no_decomposition(self):
        # Blue paper one step from I9 with nothing able to answer it.
        state = position({'H8': 3, 'A2': -1}, turn=0)
        config = SearchConfig(proof_depth=2, proof_nodes=64)
        report = attribute(self.game, state, 0, unlimited(), config, modules=MODULES)
        if report['proof'].get('status') == 'proven' or report['terminal']:
            self.assertFalse(report['decomposed'])
            self.assertIn('note', report)
            self.assertEqual(report['total'], [0.] * 81)
        else:
            self.skipTest('this fixture was not proven within the configured budget')

    def test_unknown_modules_and_perspectives_are_rejected(self):
        state = self.states[0]
        with self.assertRaises(ValueError):
            attribute(self.game, state, 0, unlimited(), SearchConfig(), modules=('nonsense',))
        with self.assertRaises(ValueError):
            attribute(self.game, state, 2, unlimited(), SearchConfig())
        self.assertNotIn('overload', DEFAULT_MODULES)

    def test_weights_match_the_evaluator_including_the_subtracted_overload(self):
        config = CONFIGURATIONS['signed']
        weights = module_weights(config)
        self.assertEqual(weights['overload'], -config.overload_weight)
        self.assertEqual(weights['piece_count'], config.count_weight)
        self.assertEqual(weights['clear_run'], 0.)


class PressureSquareTests(unittest.TestCase):
    def test_square_credit_reproduces_the_pair_totals(self):
        generator = np.random.default_rng(5)
        for _ in range(60):
            board = np.zeros((9, 9), dtype=np.int8)
            for square in generator.choice(81, size=int(generator.integers(0, 20)), replace=False):
                board.flat[square] = int(generator.integers(1, 4)) * (1 if generator.random() < .5 else -1)
            flat = board.ravel()
            for radius in (3, 4):
                blue, red = pressure_totals(board, radius)
                attack = pressure_squares(board, radius)
                self.assertAlmostEqual(float(attack[flat > 0].sum()), blue, places=12)
                self.assertAlmostEqual(float(attack[flat < 0].sum()), red, places=12)
                self.assertEqual(float(attack[flat == 0].sum()), 0.)

    def test_radius_is_validated(self):
        with self.assertRaises(ValueError):
            pressure_squares(np.zeros((9, 9), dtype=np.int8), 2)


if __name__ == '__main__':
    unittest.main()
