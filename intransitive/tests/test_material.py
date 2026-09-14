"""Exact material equivalence, cache separation and search branch ownership."""
from dataclasses import replace
from itertools import product
from math import inf
import unittest
from unittest.mock import patch

import numpy as np

from intransitive.IntransitiveConstants import METADATA_PLANE, NO_CAPTURE_LIMIT, action_destination
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveLogicNumba import Board
from intransitive.tests.test_draws import clock_history, load_history
from intransitive.IntransitiveSymmetries import transform_state
from intransitive.IntransitiveDisplay import parse_move
from intransitive.heuristics import AlphaBetaPlayer, SearchConfig
from intransitive.heuristics.budget import BudgetExpired
from intransitive.heuristics.evaluation import Evaluator, MATE
from intransitive.heuristics.material import MaterialCache, after_capture, count_pieces, warm_material_kernels
from intransitive.tests.test_heuristics import position, unlimited

UNKNOWN = {'status': 'unknown'}


def recount(state):
    return tuple(int(np.count_nonzero(state[:, :, 0] == code)) for code in (1, 2, 3, -1, -2, -3))


class MaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warm_material_kernels()
        cls.game = IntransitiveGame()
        cls.config = SearchConfig(proof_nodes=0, max_depth=3, time_limit=60, node_limit=10**9)

    def assert_score(self, evaluator, state, counts):
        self.assertEqual(counts, recount(state))
        for side in (0, 1):
            expected = evaluator.explain(state, side, unlimited(), proof=UNKNOWN, diagnostics=False)['score']
            actual = evaluator.score(state, side, unlimited(), proof=UNKNOWN, counts=counts)
            self.assertEqual(actual.hex(), expected.hex())

    def test_generated_games_branching_and_all_symmetries(self):
        rng = np.random.default_rng(39)
        evaluator = Evaluator(self.game, self.config)
        for game in range(5):
            state = self.game.getInitBoard()
            counts = count_pieces(state)
            for ply in range(80):
                before = state.tobytes()
                self.assert_score(evaluator, state, counts)
                side = int(state[:, :, METADATA_PLANE:].flat[1])
                actions = np.flatnonzero(self.game.getValidMoves(state, side))
                if not len(actions):
                    break
                # Visit siblings without mutating the parent material tuple.
                for action in rng.choice(actions, min(3, len(actions)), replace=False):
                    x, y = action_destination(int(action))
                    child_counts = after_capture(counts, int(state[y, x, 0]))
                    child, _ = self.game.getNextState(state, side, int(action))
                    self.assert_score(evaluator, child, child_counts)
                    if not state[y, x, 0]:
                        self.assertIs(child_counts, counts)
                self.assertEqual(state.tobytes(), before)
                if ply % 11 == 0:
                    for symmetry in range(12):
                        transformed = transform_state(state, symmetry)
                        self.assert_score(evaluator, transformed, count_pieces(transformed))
                        turn = int(transformed[:, :, METADATA_PLANE:].flat[1])
                        canonical = self.game.getCanonicalForm(transformed, turn)
                        self.assert_score(evaluator, canonical, count_pieces(canonical))
                state, counts = child, child_counts

    def test_all_capture_pairings_and_type_extinction(self):
        evaluator = Evaluator(self.game, self.config)
        for side, attacker, defender in product((0, 1), (1, 2, 3), (1, 2, 3)):
            sign = 1 if side == 0 else -1
            state = position({'D4': sign * attacker, 'E5': -sign * defender,
                              'H7': -sign * (defender % 3 + 1)}, turn=side)
            counts = count_pieces(state)
            action = parse_move('D4 E5')
            if defender != attacker % 3 + 1:
                self.assertFalse(self.game.getValidMoves(state, side)[action])
                continue
            child, _ = self.game.getNextState(state, side, action)
            updated = after_capture(counts, -sign * defender)
            self.assertEqual(sum(updated), sum(counts) - 1)
            self.assertEqual(updated[(1-side)*3 + defender - 1], 0)
            self.assert_score(evaluator, child, updated)
            self.assertEqual(counts, recount(state))

    def test_quiet_reordering_preserves_bit_exact_reference_sum(self):
        # Random weights and first-occurrence orders expose sub-ULP tie changes.
        rng = np.random.default_rng(3901)
        different_orders = set()
        for _ in range(100):
            config = replace(self.config, predator_zero_bonus=float(rng.random()),
                             predator_scarcity_bonus=float(rng.random()), prey_bonus=float(rng.random()))
            evaluator = Evaluator(self.game, config)
            state = self.game.getInitBoard()
            counts = count_pieces(state)
            for __ in range(10):
                self.assert_score(evaluator, state, counts)
                side = int(state[:, :, METADATA_PLANE:].flat[1])
                quiet = [int(a) for a in np.flatnonzero(self.game.getValidMoves(state, side))
                         if state[action_destination(int(a))[1], action_destination(int(a))[0], 0] == 0]
                state, _ = self.game.getNextState(state, side, int(rng.choice(quiet)))
                different_orders.add(tuple(dict.fromkeys(int(c) for c in state[:, :, 0].ravel() if c)))
            self.assertEqual(evaluator.material.values.cache_info().misses, 1)
        self.assertGreater(len(different_orders), 6)

    def test_configuration_identity_clamping_and_bounded_cache(self):
        state = position({'D4': 1, 'E4': 3, 'F6': -2})
        evaluator = Evaluator(self.game, self.config)
        self.assert_score(evaluator, state, count_pieces(state))
        old = evaluator.material
        for name in self.config.to_dict():
            value = getattr(self.config, name)
            if name == 'evaluator_version':
                continue
            changed = not value if type(value) is bool else value + 1
            evaluator.config = replace(self.config, **{name: changed})
            self.assert_score(evaluator, state, count_pieces(state))
        evaluator.config = replace(self.config, count_weight=1e9)
        self.assert_score(evaluator, state, count_pieces(state))
        self.assertIsNot(old, evaluator.material)
        cache = MaterialCache(self.config)
        for counts in product(range(4), repeat=6):
            cache.values(counts)
        self.assertEqual(cache.values.cache_info().currsize, 256)

    def test_quiet_positions_draws_wins_proofs_and_optional_modules_are_not_cached(self):
        state = position({'H8': 3, 'D4': -1})
        evaluator = Evaluator(self.game, self.config)
        counts = count_pieces(state)
        self.assert_score(evaluator, state, counts)
        child, _ = self.game.getNextState(state, 0, parse_move('H8 I9'))
        self.assertEqual(evaluator.score(child, 0, unlimited(), counts=counts), MATE)
        repeated = load_history(Board(), [state[:, :, 0]] * 5)
        self.assertEqual(evaluator.score(repeated, 0, unlimited(), counts=counts), 0.)
        history = clock_history(state[:, :, 0], NO_CAPTURE_LIMIT - 1)
        before_limit = load_history(Board(), history, first_player=1)
        for move, expected in (('H8 I9', MATE), ('H8 G8', 0.)):
            child, _ = self.game.getNextState(before_limit, 0, parse_move(move))
            self.assertEqual(evaluator.score(child, 0, unlimited(), counts=counts), expected)
        for score in (MATE - 3, -MATE + 3):
            self.assertEqual(evaluator.score(state, 0, unlimited(),
                proof={'status': 'proven', 'score': score}, counts=counts), score)
        for flags in product((False, True), repeat=3):
            config = replace(self.config, attack_enabled=flags[0], defence_enabled=flags[1], overload_enabled=flags[2])
            evaluator = Evaluator(self.game, config)
            state = position({'D4': 1, 'G7': -2, 'H7': 3})
            child, _ = self.game.getNextState(state, 0, parse_move('H7 H8'))
            for item in (state, child):
                self.assert_score(evaluator, item, count_pieces(item))
        with (patch('intransitive.heuristics.evaluation.Geometry', side_effect=AssertionError),
              patch('intransitive.heuristics.evaluation.Counter', side_effect=AssertionError)):
            Evaluator(self.game, self.config).score(state, 0, unlimited(), proof=UNKNOWN)

    def test_search_counts_restore_after_cutoff_interrupt_error_and_resume(self):
        state = position({'D4': 1, 'E5': -2, 'F6': 3, 'G7': -1})
        original = state.tobytes()
        root_counts = recount(state)
        for stop_ply in (1, 2, 3):
            for error in (BudgetExpired('time'), BudgetExpired('work'), RuntimeError('injected')):
                player = AlphaBetaPlayer(config=self.config)
                player._prepare()
                search = player._search
                def checked(item, depth, alpha, beta, ply, budget):
                    if ply:
                        self.assertEqual(player._material_counts, recount(item))
                    if ply == stop_ply:
                        raise error
                    return search(item, depth, alpha, beta, ply, budget)
                with patch.object(player, '_search', checked):
                    with self.assertRaises(type(error)):
                        player._search(state, 3, -inf, inf, 0, unlimited())
                self.assertEqual(player._material_counts, root_counts)
                self.assertEqual(state.tobytes(), original)
                resumed = player._search(state, 2, -inf, inf, 0, unlimited())
                fresh = AlphaBetaPlayer(config=self.config)
                fresh._prepare()
                self.assertEqual(resumed, fresh._search(state, 2, -inf, inf, 0, unlimited()))
                player._search(state, 3, -1., 1., 0, unlimited())
                self.assertEqual(player._material_counts, root_counts)
        for work in (0, 1, 100, 500, 1000, 3000):
            player = AlphaBetaPlayer(config=replace(self.config, node_limit=work))
            result = player.analyze(state)
            self.assertLessEqual(result.work, work)
            self.assertEqual(player._material_counts, root_counts)
            self.assertEqual(state.tobytes(), original)

    def test_root_counts_initialized_once_and_no_warm_compilation(self):
        # Integer-valued JSON weights must also use the warmed float signature.
        player = AlphaBetaPlayer(config=replace(self.config, max_depth=2, count_weight=100, advantage_weight=25))
        player._prepare()
        from intransitive.heuristics.material import ordered_score
        with (patch('intransitive.heuristics.search.count_pieces', wraps=count_pieces) as count,
              patch.object(count_pieces, 'compile', side_effect=AssertionError('cold count')),
              patch.object(ordered_score, 'compile', side_effect=AssertionError('cold score'))):
            result = player.analyze(self.game.getInitBoard())
        self.assertEqual(count.call_count, 1)
        self.assertEqual(result.completed_depth, 2)
        self.assertEqual(player._material_counts, (3, 3, 4, 3, 3, 4))


if __name__ == '__main__':
    unittest.main()
