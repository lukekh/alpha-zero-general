"""Deterministic deadline interruptions and draw-safe transposition folding."""
from dataclasses import replace
from math import inf
import unittest
from unittest.mock import patch

import numpy as np

from intransitive.heuristics import AlphaBetaPlayer, SearchConfig, exhaustive_minimax
from intransitive.heuristics.budget import Budget, BudgetExpired
from intransitive.heuristics.evaluation import MATE
from intransitive.heuristics.search import from_table, position_key
from intransitive.heuristics.position import SearchPosition
from intransitive.tests.reference_rules import Position, position


class Clock:
    now = 0.

    def __call__(self):
        return self.now


class ScriptedChildren(AlphaBetaPlayer):
    """Real root search/ordering/budget, controlled completed child values."""
    def __init__(self, state, stop_index=1, midway=False, losing=False, finish=False):
        super().__init__(config=SearchConfig(max_depth=2, proof_nodes=0,
                                             node_limit=10**7, time_limit=60))
        self.clock = Clock()
        self._prepare()
        ordered = list(self._ordered(state, 0, None, Budget(10**7, 60)))
        self.actions = [a for a, _ in ordered]
        self.indices = {child.tobytes(): i for i, (_, child) in enumerate(ordered)}
        self.stop_index = len(ordered) - 1 if finish else stop_index
        self.midway, self.losing = midway, losing
        self.visited = []

    def _search(self, state, depth, alpha, beta, ply, budget):
        if ply != 1:
            return super()._search(state, depth, alpha, beta, ply, budget)
        snapshot = state.export() if isinstance(state, SearchPosition) else state
        index = self.indices[snapshot.tobytes()]
        self.visited.append((depth, index))
        stop = depth == 1 and index == self.stop_index
        if stop and self.midway:
            self.clock.now = 2.
        budget.visit()
        value = ((10 if index == 0 else 8 if index == 1 else 0) if depth == 0
                 else (1 if index == 0 else 20 if index == 1 else 0))
        if self.losing and depth == 1 and index == 0:
            value = -MATE + 4
        if stop:
            self.clock.now = 2.
        return -value, []


class AnytimeSearchTests(unittest.TestCase):
    def setUp(self):
        self.reference = Position.fixture(position({'D4': 1, 'F6': -2}))
        self.state = self.reference.storage()

    def run_script(self, **kwargs):
        player = ScriptedChildren(self.state, **kwargs)
        budget = Budget(10**7, 1, clock=player.clock)
        result = player.analyze(self.state, budget)
        self.assertTrue(result.stopped)
        self.assertEqual(result.stop_reason, 'time')
        self.assertEqual(result.diagnostics_status, 'skipped_budget')
        self.assertEqual(result.effective_limits,
                         {'max_depth': 2, 'time_limit': 1, 'work_limit': 10**7})
        self.assertEqual(result.completed_depth, 1)
        self.assertNotIn((position_key(self.state), 2), player.table)
        return player, result

    def test_timeout_keeps_better_fully_searched_sibling(self):
        player, result = self.run_script()
        self.assertEqual(result.action, player.actions[1])
        self.assertEqual(result.score, 20)
        self.assertEqual(result.selected_depth, 2)
        self.assertEqual(result.partial_depth, 2)
        self.assertEqual(result.root_moves_completed, 2)
        self.assertEqual(result.selection_source, 'partial_iteration')
        self.assertEqual(result.explanation['search_scope'], 'selected_move')
        self.assertEqual([i for depth, i in player.visited if depth == 1], [0, 1])

    def test_interrupted_child_is_not_published_or_mixed_with_shallow_scores(self):
        player, result = self.run_script(midway=True)
        self.assertEqual(result.action, player.actions[0])
        self.assertEqual(result.score, 1)  # not the old depth-one score of 10
        self.assertEqual(result.selected_depth, 2)
        self.assertEqual(result.root_moves_completed, 1)

    def test_proved_losing_incumbent_yields_to_an_unrefuted_alternative(self):
        player, result = self.run_script(stop_index=0, losing=True)
        self.assertEqual(result.action, player.actions[1])
        self.assertEqual(result.selected_depth, 1)
        self.assertEqual(result.selection_source, 'unrefuted_fallback')
        self.assertEqual(result.score_bound, 'upper')
        self.assertNotEqual(result.explanation['proof']['status'], 'proven')

    def test_iteration_finished_at_deadline_is_not_discarded(self):
        player = ScriptedChildren(self.state, finish=True)
        result = player.analyze(self.state, Budget(10**7, 1, clock=player.clock))
        self.assertTrue(result.stopped)
        self.assertEqual(result.completed_depth, 2)
        self.assertEqual(result.selected_depth, 2)
        self.assertEqual(result.partial_depth, 0)
        self.assertEqual(result.action, player.actions[1])
        self.assertEqual(result.selection_source, 'completed_iteration')

    def test_diagnostics_follow_move_search(self):
        player = ScriptedChildren(self.state)
        from intransitive.heuristics.evaluation import Evaluator
        original = Evaluator.explain

        def explain(evaluator, *args, **kwargs):
            self.assertTrue(player.visited)
            return original(evaluator, *args, **kwargs)

        with patch.object(Evaluator, 'explain', explain):
            player.analyze(self.state, Budget(10**7, 1, clock=player.clock))

    def test_budget_reasons_and_simultaneous_limit_precedence(self):
        clock = Clock()
        with self.assertRaises(BudgetExpired) as expired:
            Budget(0, 1, clock=clock).charge()
        self.assertEqual(expired.exception.reason, 'work')
        with self.assertRaises(BudgetExpired) as expired:
            Budget(0, 0, clock=clock).charge()
        self.assertEqual(expired.exception.reason, 'time')

        work = AlphaBetaPlayer(config=SearchConfig(
            max_depth=2, proof_nodes=0, node_limit=0, time_limit=60)).analyze(self.state)
        elapsed = AlphaBetaPlayer(config=SearchConfig(
            max_depth=2, proof_nodes=0, node_limit=10**7, time_limit=0)).analyze(self.state)
        self.assertEqual((work.stop_reason, elapsed.stop_reason), ('work', 'time'))
        self.assertEqual((work.selection_source, elapsed.selection_source),
                         ('legal_fallback', 'legal_fallback'))

    def test_natural_stop_reasons_and_diagnostics_are_independent(self):
        config = SearchConfig(max_depth=1, proof_nodes=0, node_limit=10**7, time_limit=60)
        completed = AlphaBetaPlayer(config=config).analyze(self.state)
        self.assertFalse(completed.stopped)
        self.assertEqual(completed.stop_reason, 'maximum_depth')
        self.assertEqual(completed.diagnostics_status, 'completed')

        winning = Position.fixture(position({'H8': 3, 'C4': -2})).storage()
        proven = AlphaBetaPlayer(config=config).analyze(winning)
        self.assertFalse(proven.stopped)
        self.assertEqual(proven.stop_reason, 'proven_result')

        from intransitive.heuristics.evaluation import Evaluator
        with patch.object(Evaluator, 'explain', side_effect=BudgetExpired('work')):
            skipped = AlphaBetaPlayer(config=config).analyze(self.state)
        self.assertFalse(skipped.stopped)
        self.assertEqual(skipped.stop_reason, 'maximum_depth')
        self.assertEqual(skipped.diagnostics_status, 'skipped_budget')
        self.assertEqual(skipped.explanation['stop_reason'], 'maximum_depth')
        self.assertEqual(skipped.explanation['diagnostics_status'], 'skipped_budget')

    def test_move_number_does_not_prevent_exact_cache_reuse(self):
        other = replace(self.reference, ply=100).storage()
        self.assertNotEqual(self.state.tobytes(), other.tobytes())
        self.assertEqual(position_key(self.state), position_key(other))
        player = AlphaBetaPlayer(config=SearchConfig(proof_nodes=0))
        player._prepare()
        value, line = player._search(self.state, 2, -inf, inf, 0, Budget(10**7, 60))
        budget = Budget(10**7, 60)
        with patch.object(player, '_leaf', side_effect=AssertionError('already searched')):
            cached = player._search(other, 2, -inf, inf, 0, budget)
        self.assertEqual(cached, (value, line))
        self.assertEqual(budget.tt_hits, 1)
        self.assertEqual(budget.nodes, 1)

    def test_history_order_folds_but_repetition_counts_do_not(self):
        a = self.reference.pieces
        b = position({'D5': 1, 'F6': -2})
        c = position({'D5': 1, 'F7': -2})
        d = position({'E5': 1, 'F7': -2})
        first = Position.fixture(a, history=(a, b, c, d, a))
        reordered = Position.fixture(a, history=(c, d, a, b, a))
        different = Position.fixture(a, history=(a, b, a, d, a))
        self.assertEqual(position_key(first.storage()), position_key(reordered.storage()))
        self.assertNotEqual(position_key(first.storage()), position_key(different.storage()))
        self.assertEqual(first.terminal()[0], 'ongoing')  # second occurrence
        self.assertEqual(different.terminal()[0], 'repetition')  # third occurrence
        for action in first.legal():
            x, _ = first.move(action)
            y, _ = reordered.move(action)
            self.assertEqual(position_key(x.storage()), position_key(y.storage()))
            self.assertEqual(x.terminal(), y.terminal())
        player = AlphaBetaPlayer(config=SearchConfig(proof_nodes=0))
        player._prepare()
        player._search(first.storage(), 1, -inf, inf, 0, Budget(10**7, 60))
        value, line = player._search(different.storage(), 1, -inf, inf, 0, Budget(10**7, 60))
        self.assertEqual((value, line), (0., []))

    def test_leaf_reuse_and_full_depth_score_match_exhaustive(self):
        config = SearchConfig(max_depth=2, proof_nodes=0, node_limit=10**7, time_limit=60)
        player = AlphaBetaPlayer(config=config)
        player._prepare()
        first = player._search(self.state, 0, -inf, inf, 0, Budget(10**7, 60))
        with patch.object(player, '_leaf', side_effect=AssertionError('leaf should be cached')):
            second = player._search(self.state, 0, -inf, inf, 0, Budget(10**7, 60))
        self.assertEqual(first, second)
        expected, _ = exhaustive_minimax(player.game, self.state, 2, config, Budget(10**7, 60))
        before = self.state.copy()
        result = player.analyze(self.state)
        self.assertEqual(result.score, expected)
        self.assertEqual(result.selected_depth, result.completed_depth)
        np.testing.assert_array_equal(self.state, before)

    def test_work_limited_real_search_publishes_only_valid_completed_values(self):
        state = Position.fixture(position({'D4': 1, 'E5': -2, 'G7': -3})).storage()
        saw_partial = False
        for work in (100, 500, 1000, 3000):
            config = SearchConfig(max_depth=3, proof_nodes=0, node_limit=work, time_limit=60)
            player = AlphaBetaPlayer(config=config)
            result = player.analyze(state)
            self.assertLessEqual(result.work, work)
            saw_partial |= result.selection_source == 'partial_iteration'
            if result.selected_depth and result.score_bound == 'exact':
                child, _ = player.game.getNextState(state, 0, result.action)
                value, _ = exhaustive_minimax(player.game, child, result.selected_depth - 1,
                                              config, Budget(10**7, 60))
                self.assertAlmostEqual(result.score, from_table(-value, 1))
        self.assertTrue(saw_partial)

    def test_deeper_entry_orders_moves_without_replacing_shallower_score(self):
        player = AlphaBetaPlayer(config=SearchConfig(proof_nodes=0, table_entries=100))
        player._prepare()
        _, deep_line = player._search(self.state, 2, -inf, inf, 0, Budget(10**7, 60))
        ordered = player._ordered
        hints = []

        def observe(state, side, preferred, budget, **kwargs):
            if position_key(state) == position_key(self.state):
                hints.append(preferred)
            yield from ordered(state, side, preferred, budget, **kwargs)

        with patch.object(player, '_ordered', observe):
            value, _ = player._search(self.state, 1, -inf, inf, 0, Budget(10**7, 60))
        expected, _ = exhaustive_minimax(player.game, self.state, 1, player.config, Budget(10**7, 60))
        self.assertEqual(hints, [deep_line[0]])
        self.assertEqual(value, expected)
        self.assertLessEqual(len(player.table), 100)
        self.assertTrue(all((key, depth) in player.table for key, depth in player._hints.items()))
        player.reload()
        self.assertFalse(player._hints)


if __name__ == '__main__':
    unittest.main()
