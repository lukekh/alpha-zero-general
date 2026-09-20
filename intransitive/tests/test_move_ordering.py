"""Counter moves, continuation history, history aging and IIR.

These are ordering mechanisms, so the standard they are held to is the one the
existing ordering options meet: exactly inert when switched off, never able to
drop a legal move, never able to reach the bounded proof or the run
certificate, and — apart from internal iterative *reduction*, which deliberately
returns a shallower value — unable to change a completed unpruned score.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import unittest

import numpy as np

from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.heuristics import AlphaBetaPlayer, SearchConfig
from intransitive.heuristics.budget import Budget
from intransitive.heuristics.evaluation import MATE_THRESHOLD
from intransitive.heuristics.ordering import ordered_actions, warm_ordering
from intransitive.heuristics.position import SearchPosition
from intransitive.heuristics.search import certificate, prove, selective_mode_early
from intransitive.IntransitiveDisplay import parse_move
from intransitive.tests.test_attribution import random_positions
from intransitive.tests.test_heuristics import position


BASE = dict(max_depth=4, time_limit=300., node_limit=10**12, ordering_enabled=True)
DEEP = dict(BASE, max_depth=5)


def analyze(config, state, game=None):
    return AlphaBetaPlayer(game or IntransitiveGame(), config).analyze(state)


def search(**flags):
    return SearchConfig(**dict(BASE, **flags))


class SettingsTests(unittest.TestCase):
    def test_every_mechanism_is_off_by_default(self):
        config = SearchConfig()
        self.assertFalse(config.counter_move_enabled)
        self.assertFalse(config.continuation_enabled)
        self.assertFalse(config.history_aging_enabled)
        self.assertFalse(config.iir_enabled)
        self.assertEqual((config.continuation_plies, config.iir_mode), (1, 'reduce'))

    def test_bounds_are_validated(self):
        for bad in (dict(continuation_plies=0), dict(continuation_plies=3),
                    dict(history_max=255), dict(iir_min_depth=2), dict(iir_reduction=0),
                    dict(iir_mode='shallow'), dict(counter_move_enabled=1)):
            with self.subTest(**bad):
                with self.assertRaises(ValueError):
                    SearchConfig(ordering_enabled=True, **bad)

    def test_a_reduction_must_leave_a_ply_below_it(self):
        SearchConfig(iir_enabled=True, iir_min_depth=3, iir_reduction=1)
        with self.assertRaises(ValueError):
            SearchConfig(iir_enabled=True, iir_min_depth=3, iir_reduction=2)

    def test_an_enabled_mechanism_may_not_be_silently_inert(self):
        """#66: a technique that cannot run must say so, not quietly do nothing."""
        for name in ('counter_move_enabled', 'continuation_enabled'):
            with self.subTest(mechanism=name):
                with self.assertRaises(ValueError):
                    SearchConfig(**{name: True}, ordering_enabled=False)
                SearchConfig(**{name: True}, ordering_enabled=True)

    def test_only_the_reduction_declares_itself_selective(self):
        self.assertTrue(selective_mode_early(search(iir_enabled=True, iir_mode='reduce')))
        self.assertFalse(selective_mode_early(search(iir_enabled=True, iir_mode='deepen')))
        for name in ('counter_move_enabled', 'continuation_enabled', 'history_aging_enabled'):
            self.assertFalse(selective_mode_early(search(**{name: True})), name)

    def test_each_mechanism_changes_the_search_identity(self):
        base = search()
        for flags in (dict(counter_move_enabled=True), dict(continuation_enabled=True),
                      dict(continuation_enabled=True, continuation_plies=2),
                      dict(history_aging_enabled=True), dict(history_max=4096),
                      dict(iir_enabled=True), dict(iir_enabled=True, iir_mode='deepen')):
            with self.subTest(**flags):
                self.assertNotEqual(base.identity(), replace(base, **flags).identity())


class InertWhenOffTests(unittest.TestCase):
    """With the flags off, this must be the search that shipped before them."""

    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()
        cls.states = random_positions(games=3, plies=40, seed=91)[:8]

    def test_unset_parameters_never_reach_the_search(self):
        dressed = search(continuation_plies=2, history_max=4096,
                         iir_min_depth=3, iir_reduction=1, iir_mode='deepen')
        for state in self.states:
            plain = analyze(search(), state, self.game)
            other = analyze(dressed, state, self.game)
            with self.subTest(state=id(state)):
                self.assertEqual(plain.action, other.action)
                self.assertEqual(plain.score, other.score)
                self.assertEqual(plain.nodes, other.nodes)
                self.assertEqual(plain.work, other.work)
                self.assertEqual(plain.score_bound, 'exact')

    def test_an_empty_continuation_row_is_indistinguishable_from_none(self):
        """The new keys must be constant columns, not a re-ranking, when unused."""
        state = position({'D4': 1, 'F4': 2, 'C6': 3, 'B2': 3, 'E4': -2, 'G4': -3, 'D6': -1})
        board = state[:, :, 0]
        actions = np.flatnonzero(IntransitiveGame().getValidMoves(state, 0)).astype(np.int64)
        history = np.arange(648, dtype=np.int64) % 7
        killers = np.array([int(actions[3]), int(actions[1])], dtype=np.int64)
        args = (board, actions, 0, 80, -1, np.full(648, -np.inf), killers, history, True)
        empty = np.zeros(648, dtype=np.int32)
        for kernel in (ordered_actions, ordered_actions.py_func):
            with self.subTest(kernel=kernel):
                reference = kernel(*args)
                np.testing.assert_array_equal(reference, kernel(*args, None, None, -1, None, None))
                np.testing.assert_array_equal(reference, kernel(*args, None, None, -1, empty, empty))


class OrderingKeyTests(unittest.TestCase):
    """The kernel's own priorities, isolated from the search that feeds it.

    Every candidate leaves the same central square, so the winning, preferred,
    threat, capture, safety and escape keys are identical across them and only
    the keys under test can decide the order.
    """

    @classmethod
    def setUpClass(cls):
        warm_ordering(True)
        # A lone central runner with no enemy anywhere near its 3x3 ring.
        cls.state = position({'E5': 1, 'B2': 2, 'I1': -1, 'H1': -2})
        legal = np.flatnonzero(IntransitiveGame().getValidMoves(cls.state, 0))
        cls.actions = np.array([a for a in map(int, legal)
                                if a // 8 == 4 * 9 + 4], dtype=np.int64)
        assert len(cls.actions) == 8, cls.actions

    def keys(self, **changes):
        defaults = dict(preferred=-1, killers=np.full(2, -1, dtype=np.int64),
                        history=np.zeros(648, dtype=np.int64), counter=-1,
                        continuation=None, continuation2=None)
        defaults.update(changes)
        return (self.state[:, :, 0], self.actions, 0, 80, defaults['preferred'],
                np.full(648, -np.inf), defaults['killers'], defaults['history'], True, None, None,
                defaults['counter'], defaults['continuation'], defaults['continuation2'])

    def test_the_candidates_are_separated_by_the_tested_keys_alone(self):
        """Any one of them can be ordered first by history, so the keys above
        history — wins, the preferred move, threats, captures, safety and
        escapes — are tied across the whole set."""
        for action in map(int, self.actions):
            history = np.zeros(648, dtype=np.int64)
            history[action] = 1
            for kernel in (ordered_actions, ordered_actions.py_func):
                with self.subTest(action=action, kernel=kernel):
                    self.assertEqual(int(kernel(*self.keys(history=history))[0]), action)

    def test_a_counter_move_outranks_history_but_yields_to_a_killer(self):
        counter, rich, killer = (int(self.actions[i]) for i in (7, 1, 3))
        history = np.zeros(648, dtype=np.int64)
        history[rich] = 5000
        for kernel in (ordered_actions, ordered_actions.py_func):
            with self.subTest(kernel=kernel):
                order = list(map(int, kernel(*self.keys(counter=counter, history=history))))
                self.assertEqual(order[0], counter)
                self.assertEqual(order[1], rich)
                order = list(map(int, kernel(*self.keys(
                    counter=counter, killers=np.array([killer, -1], dtype=np.int64)))))
                self.assertEqual(order[:2], [killer, counter])

    def test_continuation_rows_add_to_the_flat_history_score(self):
        low, high = int(self.actions[2]), int(self.actions[6])
        history = np.zeros(648, dtype=np.int64)
        history[low] = 100
        continuation = np.zeros(648, dtype=np.int32)
        continuation[high] = 90
        second = np.zeros(648, dtype=np.int32)
        second[high] = 20
        for kernel in (ordered_actions, ordered_actions.py_func):
            with self.subTest(kernel=kernel):
                one = list(map(int, kernel(*self.keys(history=history, continuation=continuation))))
                self.assertEqual(one[0], low, '90 < 100 keeps the flat history order')
                both = list(map(int, kernel(*self.keys(history=history, continuation=continuation,
                                                       continuation2=second))))
                self.assertEqual(both[0], high, '90 + 20 > 100 overtakes it')

    def test_no_legal_move_is_ever_dropped_or_duplicated(self):
        game, state = IntransitiveGame(), position(
            {'D4': 1, 'F4': 2, 'C6': 3, 'B2': 3, 'E4': -2, 'G4': -3, 'D6': -1})
        legal = np.flatnonzero(game.getValidMoves(state, 0)).astype(np.int64)
        rows = np.arange(648, dtype=np.int32) % 13
        killers = np.array([int(legal[2]), int(legal[5])], dtype=np.int64)
        for kernel in (ordered_actions, ordered_actions.py_func):
            for enhanced in (False, True):
                for extra in ({}, dict(counter=int(legal[4])), dict(continuation=rows),
                              dict(continuation=rows, continuation2=rows[::-1].copy())):
                    args = (state[:, :, 0], legal, 0, 80, -1, np.full(648, -np.inf), killers,
                            np.arange(648, dtype=np.int64) % 5, enhanced, None, None,
                            extra.get('counter', -1), extra.get('continuation'),
                            extra.get('continuation2'))
                    order = list(map(int, kernel(*args)))
                    with self.subTest(kernel=kernel, enhanced=enhanced, keys=sorted(extra)):
                        self.assertEqual(sorted(order), sorted(map(int, legal)))


class CounterMoveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()
        cls.states = random_positions(games=3, plies=50, seed=101)[:6]

    def test_refutations_are_recorded_and_then_used(self):
        totals = dict(counter_move_updates=0, counter_move_available=0, cutoff_from_counter=0)
        for state in self.states:
            ordering = analyze(search(counter_move_enabled=True), state, self.game).ordering
            for name in totals:
                totals[name] += ordering[name]
        self.assertGreater(totals['counter_move_updates'], 0, 'no refutation was ever stored')
        self.assertGreater(totals['counter_move_available'], 0, 'a stored refutation was never offered')
        self.assertGreater(totals['cutoff_from_counter'], 0, 'a counter move never caused a cutoff')

    def test_the_table_is_empty_without_the_flag(self):
        ordering = analyze(search(), self.states[0], self.game).ordering
        self.assertEqual(ordering['counter_move_updates'], 0)
        self.assertEqual(ordering['counter_move_available'], 0)
        self.assertEqual(ordering['cutoff_from_counter'], 0)

    def test_it_changes_the_tree_without_changing_the_value(self):
        changed = 0
        for state in self.states:
            plain = analyze(search(), state, self.game)
            counter = analyze(search(counter_move_enabled=True), state, self.game)
            with self.subTest(state=id(state)):
                self.assertEqual(plain.score, counter.score, 'a completed unpruned value must not move')
            changed += plain.nodes != counter.nodes
        self.assertGreater(changed, 0, 'counter moves never changed the tree')


class ContinuationHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()
        cls.states = random_positions(games=3, plies=50, seed=107)[:6]

    def test_both_context_depths_accumulate_and_the_deeper_one_records_more(self):
        one = two = 0
        for state in self.states:
            one += analyze(search(continuation_enabled=True), state, self.game).ordering['continuation_updates']
            two += analyze(search(continuation_enabled=True, continuation_plies=2),
                           state, self.game).ordering['continuation_updates']
        self.assertGreater(one, 0)
        self.assertGreater(two, one)

    def test_the_tables_are_only_allocated_when_they_are_read(self):
        player = AlphaBetaPlayer(self.game, search())
        player._prepare()
        self.assertIsNone(player._continuation)
        player = AlphaBetaPlayer(self.game, search(continuation_enabled=True, continuation_plies=2))
        player._prepare()
        self.assertEqual(player._continuation.shape, (2, 2, 648, 648))
        self.assertFalse(player._continuation.any())

    def test_it_changes_the_tree_without_changing_the_value(self):
        for state in self.states:
            plain = analyze(search(), state, self.game)
            follow = analyze(search(continuation_enabled=True, continuation_plies=2), state, self.game)
            with self.subTest(state=id(state)):
                self.assertEqual(plain.score, follow.score)


class HistoryAgingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()
        cls.states = random_positions(games=2, plies=50, seed=113)[:4]

    def test_gravity_bounds_every_statistic(self):
        config = search(counter_move_enabled=True, continuation_enabled=True,
                        continuation_plies=2, history_aging_enabled=True, history_max=1024)
        player = AlphaBetaPlayer(self.game, config)
        for state in self.states:
            player.analyze(state)
            with self.subTest(state=id(state)):
                self.assertLessEqual(int(player._history.max()), config.history_max)
                self.assertLessEqual(int(player._continuation.max()), config.history_max)

    def test_each_root_iteration_after_the_first_ages_the_statistics(self):
        result = analyze(search(history_aging_enabled=True), self.states[0], self.game)
        self.assertEqual(result.ordering['history_decays'], result.completed_depth - 1)
        self.assertEqual(analyze(search(), self.states[0], self.game).ordering['history_decays'], 0)

    def test_it_changes_the_tree_without_changing_the_value(self):
        for state in self.states:
            plain = analyze(search(), state, self.game)
            aged = analyze(search(history_aging_enabled=True), state, self.game)
            with self.subTest(state=id(state)):
                self.assertEqual(plain.score, aged.score)


class InternalIterativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()
        cls.states = random_positions(games=3, plies=50, seed=127)[:6]

    def test_both_modes_fire_on_ordinary_positions(self):
        for mode in ('reduce', 'deepen'):
            fired = sum(analyze(SearchConfig(**DEEP, iir_enabled=True, iir_mode=mode),
                                state, self.game).ordering['iir_nodes'] for state in self.states)
            with self.subTest(mode=mode):
                self.assertGreater(fired, 0)

    def test_it_is_refused_where_a_move_is_already_known(self):
        shallow = analyze(SearchConfig(**dict(BASE, max_depth=3), iir_enabled=True,
                                       iir_min_depth=32), self.states[0], self.game)
        self.assertEqual(shallow.ordering['iir_nodes'], 0)

    def test_deepening_keeps_the_completed_value_and_the_exact_label(self):
        for state in self.states:
            plain = analyze(SearchConfig(**DEEP), state, self.game)
            deepened = analyze(SearchConfig(**DEEP, iir_enabled=True, iir_mode='deepen'),
                               state, self.game)
            with self.subTest(state=id(state)):
                self.assertEqual(plain.score, deepened.score)
                self.assertEqual(deepened.score_bound, 'exact')
                self.assertNotIn('not a certificate', str(deepened.explanation['proof']))

    def test_reduction_declares_itself_a_non_certificate(self):
        result = analyze(SearchConfig(**DEEP, iir_enabled=True, iir_mode='reduce'),
                         self.states[0], self.game)
        self.assertEqual(result.score_bound, 'selective_exact')
        self.assertTrue(result.selective['enabled'])
        self.assertIn('not a certificate', result.explanation['proof']['reason'])
        self.assertGreater(result.ordering['iir_reduced_plies'], 0)

    def test_an_immediate_win_survives_both_modes(self):
        state = position({'H8': 3, 'A1': 1, 'E5': -1, 'D4': -2}, turn=0)
        for mode in ('reduce', 'deepen'):
            result = analyze(SearchConfig(**DEEP, iir_enabled=True, iir_mode=mode), state, self.game)
            with self.subTest(mode=mode):
                self.assertGreater(result.score, MATE_THRESHOLD)


class OrderingStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()
        states = random_positions(games=1, plies=30, seed=131)
        cls.result = analyze(search(counter_move_enabled=True, continuation_enabled=True),
                             states[min(12, len(states) - 1)], cls.game)

    def test_the_reported_rates_agree_with_their_own_counters(self):
        ordering = self.result.ordering
        cutoffs = ordering['beta_cutoffs']
        self.assertGreater(cutoffs, 0)
        self.assertLessEqual(ordering['first_move_cutoffs'], cutoffs)
        self.assertLessEqual(cutoffs, ordering['ordered_nodes'])
        self.assertAlmostEqual(ordering['first_move_cutoff_rate'],
                               ordering['first_move_cutoffs'] / cutoffs)
        self.assertAlmostEqual(ordering['mean_cutoff_index'],
                               ordering['cutoff_index_sum'] / cutoffs)

    def test_every_cutoff_is_attributed_to_exactly_one_source(self):
        ordering = self.result.ordering
        self.assertEqual(ordering['beta_cutoffs'],
                         sum(ordering['cutoff_from_' + name]
                             for name in ('preferred', 'killer', 'counter', 'other')))

    def test_the_depth_distribution_sums_to_the_totals(self):
        rows = self.result.ordering['cutoffs_by_depth'].values()
        self.assertEqual(sum(row['cutoffs'] for row in rows), self.result.ordering['beta_cutoffs'])
        self.assertEqual(sum(row['first_move'] for row in rows),
                         self.result.ordering['first_move_cutoffs'])
        self.assertEqual(sum(row['index_sum'] for row in rows),
                         self.result.ordering['cutoff_index_sum'])

    def test_no_position_is_left_without_a_first_move_cutoff_rate(self):
        self.assertIsNotNone(self.result.ordering['first_move_cutoff_rate'])
        self.assertGreaterEqual(self.result.ordering['first_move_cutoff_rate'], 0.)
        self.assertLessEqual(self.result.ordering['first_move_cutoff_rate'], 1.)


ALL_ON = dict(counter_move_enabled=True, continuation_enabled=True, continuation_plies=2,
              history_aging_enabled=True, iir_enabled=True, iir_mode='deepen')


def isolated(payload):
    """One move by one fresh engine, as the match harness runs a candidate."""
    state, flags = payload
    result = analyze(search(**flags), state)
    return int(result.action), result.score, result.nodes, result.work


class SearchStateTests(unittest.TestCase):
    """The tables belong to one search and must not outlive it."""

    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()
        cls.states = random_positions(games=2, plies=40, seed=137)[:4]

    def test_preparing_a_search_clears_every_table(self):
        player = AlphaBetaPlayer(self.game, search(**ALL_ON))
        player.analyze(self.states[0])
        self.assertTrue((player._history != 0).any(), 'nothing was learned to begin with')
        player._prepare()
        self.assertFalse(player._history.any())
        self.assertFalse(player._continuation.any())
        self.assertTrue((player._counter_moves == -1).all())
        self.assertEqual(player._path, [])
        self.assertEqual(player._ordering_stats['beta_cutoffs'], 0)

    def test_a_reused_engine_matches_a_fresh_one_move_for_move(self):
        """Ordering state is per search; the transposition table deliberately
        is not, which is why the match harness builds a fresh engine per move.
        Reuse may therefore save nodes, but it must never move the value."""
        shared = AlphaBetaPlayer(self.game, search(**ALL_ON))
        cleared = AlphaBetaPlayer(self.game, search(**ALL_ON))
        for state in self.states:
            fresh = analyze(search(**ALL_ON), state, self.game)
            cleared.reload()
            with self.subTest(state=id(state)):
                reset = cleared.analyze(state)
                self.assertEqual((reset.action, reset.score, reset.nodes, reset.work),
                                 (fresh.action, fresh.score, fresh.nodes, fresh.work))
                warm = shared.analyze(state)
                self.assertEqual((warm.action, warm.score), (fresh.action, fresh.score))

    def test_the_path_unwinds_even_when_the_budget_expires(self):
        player = AlphaBetaPlayer(self.game, search(**ALL_ON))
        player._prepare()
        player.analyze(self.states[0], Budget(4000, 30.))
        self.assertEqual(player._path, [])

    def test_concurrent_candidates_get_the_same_moves_as_isolated_ones(self):
        """The match harness plays candidates side by side in separate processes."""
        work = [(state, flags) for state in self.states for flags in (ALL_ON, {})]
        serial = [isolated(item) for item in work]
        with ProcessPoolExecutor(max_workers=2) as pool:
            concurrent = list(pool.map(isolated, work))
        self.assertEqual(serial, concurrent)


class ProofIsolationTests(unittest.TestCase):
    """Ordering must never reach the bounded proof or the run certificate."""

    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()
        cls.states = random_positions(games=2, plies=45, seed=149)[:6]

    def test_the_bounded_proof_is_unchanged_by_every_mechanism(self):
        for state in self.states:
            plain = SearchConfig(proof_depth=2, proof_nodes=64)
            dressed = replace(plain, ordering_enabled=True, **ALL_ON)
            budgets = (Budget(10**12, 600.), Budget(10**12, 600.))
            results = [prove(self.game, SearchPosition(state), config, budget)
                       for config, budget in zip((plain, dressed), budgets)]
            with self.subTest(state=id(state)):
                self.assertEqual(results[0], results[1])
                self.assertEqual(budgets[0].proof_nodes, budgets[1].proof_nodes)

    def test_the_run_certificate_is_unchanged_by_every_mechanism(self):
        for state in self.states:
            plain = SearchConfig(certificate_enabled=True)
            dressed = replace(plain, ordering_enabled=True, **ALL_ON)
            self.assertEqual(certificate(SearchPosition(state), plain, Budget(10**12, 600.)),
                             certificate(SearchPosition(state), dressed, Budget(10**12, 600.)))

    def test_a_proven_result_still_reports_as_proven(self):
        state = position({'H8': 3, 'A1': 1, 'E5': -1, 'D4': -2}, turn=0)
        result = analyze(search(**dict(ALL_ON, iir_enabled=False)), state, self.game)
        self.assertGreater(result.score, MATE_THRESHOLD)
        self.assertEqual(result.explanation['proof']['status'], 'proven')


class CompletedValueTests(unittest.TestCase):
    """Every sound mechanism together must leave the unpruned value alone."""

    def test_the_whole_ladder_agrees_with_plain_alpha_beta(self):
        game = IntransitiveGame()
        for state in random_positions(games=3, plies=45, seed=151)[:8]:
            plain = analyze(search(), state, game)
            dressed = analyze(search(**ALL_ON), state, game)
            with self.subTest(state=id(state)):
                self.assertEqual(plain.score, dressed.score)
                self.assertEqual(plain.completed_depth, dressed.completed_depth)
                self.assertEqual(dressed.score_bound, 'exact')

    def test_a_known_tactical_choice_is_unchanged(self):
        game = IntransitiveGame()
        state = position({'I8': 1, 'D4': 1, 'E4': -2, 'F6': -3})
        result = analyze(search(**ALL_ON), state, game)
        self.assertEqual(result.action, parse_move('I8 I9'))


if __name__ == '__main__':
    unittest.main()
