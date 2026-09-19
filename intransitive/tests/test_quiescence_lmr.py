"""Opt-in quiescence and late move reductions.

Both are selective, like null-move and futility: they change which move comes
back, so they must declare themselves as non-certificates and must be exactly
inert when switched off. Neither claims to preserve the minimax value.
"""
import unittest

import numpy as np

from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.heuristics import AlphaBetaPlayer, SearchConfig
from intransitive.heuristics.evaluation import MATE_THRESHOLD
from intransitive.tests.test_attribution import random_positions
from intransitive.tests.test_heuristics import position


BASE = dict(max_depth=4, time_limit=120., node_limit=50_000_000, ordering_enabled=True)


class SettingsTests(unittest.TestCase):
    def test_both_are_off_by_default(self):
        config = SearchConfig()
        self.assertFalse(config.quiescence_enabled)
        self.assertFalse(config.lmr_enabled)

    def test_reduction_must_leave_a_ply_to_search(self):
        SearchConfig(lmr_enabled=True, lmr_min_depth=3, lmr_reduction=1)
        with self.assertRaises(ValueError):
            SearchConfig(lmr_enabled=True, lmr_min_depth=3, lmr_reduction=2)

    def test_bounds_are_validated(self):
        for bad in (dict(quiescence_max_plies=0), dict(quiescence_max_plies=33),
                    dict(lmr_min_depth=1), dict(lmr_min_index=0), dict(lmr_reduction=0)):
            with self.subTest(**bad):
                with self.assertRaises(ValueError):
                    SearchConfig(**bad)

    def test_they_require_the_compact_backend(self):
        game = IntransitiveGame()
        with self.assertRaises(ValueError):
            AlphaBetaPlayer(game, SearchConfig(quiescence_enabled=True),
                            use_compact=False).analyze(game.getInitBoard())


class InertWhenOffTests(unittest.TestCase):
    """With both off, the search must be the one that shipped before them."""

    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()
        cls.states = random_positions(games=3, plies=40, seed=91)[:12]

    def test_disabled_settings_change_nothing(self):
        for state in self.states:
            plain = AlphaBetaPlayer(self.game, SearchConfig(**BASE)).analyze(state)
            # Values set but not enabled must not reach the search.
            dressed = AlphaBetaPlayer(self.game, SearchConfig(
                quiescence_max_plies=32, lmr_min_depth=3, lmr_min_index=1,
                lmr_reduction=1, **BASE)).analyze(state)
            with self.subTest(state=id(state)):
                self.assertEqual(plain.action, dressed.action)
                self.assertEqual(plain.score, dressed.score)
                self.assertEqual(plain.nodes, dressed.nodes)
                self.assertEqual(plain.score_bound, 'exact')


class DeclaredSelectiveTests(unittest.TestCase):
    def setUp(self):
        self.game = IntransitiveGame()
        self.state = self.game.getInitBoard()

    def test_each_marks_the_result_as_not_a_certificate(self):
        for name in ('quiescence_enabled', 'lmr_enabled'):
            with self.subTest(feature=name):
                result = AlphaBetaPlayer(self.game, SearchConfig(**{name: True}, **BASE)).analyze(self.state)
                self.assertEqual(result.score_bound, 'selective_exact')
                self.assertTrue(result.selective['enabled'])
                self.assertEqual(result.explanation['proof']['status'], 'unknown')
                self.assertIn('not a certificate', result.explanation['proof']['reason'])


class QuiescenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()

    def test_captures_at_the_horizon_are_resolved(self):
        """Over a corpus, not one fixture.

        A single position proves little: with ordering on, the first children
        usually fail high and a fail-high is entitled to skip the capture loop.
        What matters is that captures beyond the horizon do get resolved when
        the window makes them relevant.
        """
        config = SearchConfig(quiescence_enabled=True, max_depth=3, time_limit=120.,
                              node_limit=50_000_000, ordering_enabled=True)
        states = random_positions(games=3, plies=60, seed=103)[:8]
        resolved = sum(AlphaBetaPlayer(self.game, config).analyze(s)
                       .selective['quiescence_captures'] for s in states)
        self.assertGreater(resolved, 0)

    def test_resolving_captures_can_change_the_chosen_move(self):
        """The point of it: a static horizon and a resolved one differ."""
        plain = SearchConfig(max_depth=3, time_limit=120., node_limit=50_000_000,
                             ordering_enabled=True)
        quiet = SearchConfig(quiescence_enabled=True, max_depth=3, time_limit=120.,
                             node_limit=50_000_000, ordering_enabled=True)
        states = random_positions(games=3, plies=60, seed=107)[:12]
        differing = sum(AlphaBetaPlayer(self.game, plain).analyze(s).action
                        != AlphaBetaPlayer(self.game, quiet).analyze(s).action
                        for s in states)
        self.assertGreater(differing, 0, 'quiescence never changed a decision')

    def test_a_position_without_captures_costs_nothing_extra(self):
        # Pieces far apart: quiescence stands pat immediately everywhere.
        state = position({'A1': 1, 'B1': 2, 'I9': -1, 'I8': -2})
        plain = AlphaBetaPlayer(self.game, SearchConfig(**BASE)).analyze(state)
        quiet = AlphaBetaPlayer(self.game, SearchConfig(quiescence_enabled=True, **BASE)).analyze(state)
        self.assertEqual(quiet.selective['quiescence_captures'], 0)
        self.assertEqual(plain.score, quiet.score)

    def test_the_chain_is_bounded_by_its_ply_ceiling(self):
        states = random_positions(games=2, plies=50, seed=97)[:6]
        for state in states:
            for ceiling in (1, 3):
                config = SearchConfig(quiescence_enabled=True, quiescence_max_plies=ceiling,
                                      max_depth=3, time_limit=60., node_limit=20_000_000,
                                      ordering_enabled=True)
                result = AlphaBetaPlayer(self.game, config).analyze(state)
                with self.subTest(ceiling=ceiling):
                    self.assertFalse(result.stopped, 'a bounded chain must not exhaust the budget')


class LateMoveReductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()
        cls.states = random_positions(games=3, plies=60, seed=101)[:8]

    def test_reductions_need_a_move_order_to_trust(self):
        # Without ordering the reduction is refused, so nothing is reduced.
        config = SearchConfig(lmr_enabled=True, ordering_enabled=False,
                              compiled_ordering_enabled=False, max_depth=4,
                              time_limit=60., node_limit=20_000_000)
        result = AlphaBetaPlayer(self.game, config).analyze(self.game.getInitBoard())
        self.assertEqual(result.selective['lmr_reduced'], 0)

    def test_reductions_fire_and_are_re_searched_when_they_beat_alpha(self):
        config = SearchConfig(lmr_enabled=True, max_depth=5, time_limit=120.,
                              node_limit=100_000_000, ordering_enabled=True)
        reduced = researched = 0
        for state in self.states[:4]:
            result = AlphaBetaPlayer(self.game, config).analyze(state)
            reduced += result.selective['lmr_reduced']
            researched += result.selective['lmr_researches']
        self.assertGreater(reduced, 0)
        self.assertLessEqual(researched, reduced)

    def test_reductions_search_fewer_nodes(self):
        plain = SearchConfig(max_depth=5, time_limit=120., node_limit=100_000_000,
                             ordering_enabled=True)
        reduced = SearchConfig(lmr_enabled=True, max_depth=5, time_limit=120.,
                               node_limit=100_000_000, ordering_enabled=True)
        a = sum(AlphaBetaPlayer(self.game, plain).analyze(s).nodes for s in self.states[:4])
        b = sum(AlphaBetaPlayer(self.game, reduced).analyze(s).nodes for s in self.states[:4])
        self.assertLess(b, a)

    def test_an_immediate_win_is_still_found(self):
        # Blue paper one step from I9 with Blue to move: no reduction may hide it.
        state = position({'H8': 3, 'A1': 1, 'E5': -1, 'D4': -2}, turn=0)
        config = SearchConfig(lmr_enabled=True, max_depth=5, time_limit=60.,
                              node_limit=20_000_000, ordering_enabled=True)
        result = AlphaBetaPlayer(self.game, config).analyze(state)
        self.assertGreater(result.score, MATE_THRESHOLD)


if __name__ == '__main__':
    unittest.main()
