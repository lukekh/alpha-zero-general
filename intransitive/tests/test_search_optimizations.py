"""Optimisations preserve scores, legal candidates, history and interruption."""
from dataclasses import replace
from itertools import product
import unittest
from unittest.mock import patch
import numpy as np

from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.heuristics import AlphaBetaPlayer, SearchConfig, exhaustive_minimax
from intransitive.heuristics.budget import Budget, BudgetExpired
from intransitive.heuristics.evaluation import Evaluator
from intransitive.heuristics.position import SearchPosition
from intransitive.heuristics.search import position_key
from intransitive.tests.test_heuristics import position, unlimited
from intransitive.tests.test_draws import load_history
from intransitive.IntransitiveLogicNumba import Board


def optimized(**kwargs):
    base = dict(pressure_enabled=True,pressure_weight=10.,pressure_radius=3,
                pressure_cache_entries=128,ordering_enabled=True,compiled_ordering_enabled=True,
                depth_replacement_enabled=True,pvs_enabled=True,aspiration_enabled=True,
                max_depth=2,proof_nodes=0,time_limit=60.,node_limit=10**9,table_entries=1000)
    base.update(kwargs)
    return SearchConfig(**base)


class SearchOptimizationTests(unittest.TestCase):
    def test_compiled_legacy_order_matches_original_on_reached_positions(self):
        game = IntransitiveGame()
        state = game.getInitBoard()
        original = AlphaBetaPlayer(config=SearchConfig(compiled_ordering_enabled=False))
        compiled = AlphaBetaPlayer(config=SearchConfig(compiled_ordering_enabled=True))
        original._prepare()
        compiled._prepare()
        rng = np.random.default_rng(923)
        for _ in range(30):
            side = int(state[:,:,82:84].flat[1])
            actions = np.flatnonzero(game.getValidMoves(state,side))
            if not len(actions):
                break
            preferred = int(actions[-1])
            previous = {int(a):dict(score=float(i%3)/7) for i,a in enumerate(actions[:5])}
            original._root_previous = compiled._root_previous = previous
            for root in (False,True):
                first,second = unlimited(),unlimited()
                a = [a for a,_ in original._ordered(state,side,preferred,first,root=root)]
                b = [a for a,_ in compiled._ordered(SearchPosition(state),side,preferred,second,root=root)]
                self.assertEqual(a,b)
                self.assertEqual(first.work,second.work)
            state,_ = game.getNextState(state,side,int(rng.choice(actions)))

    def test_enhanced_python_and_native_order_match_and_include_every_move(self):
        state = position({'D4':1,'E5':-2,'F5':-3,'C3':3})
        first = AlphaBetaPlayer(config=optimized(compiled_ordering_enabled=False))
        second = AlphaBetaPlayer(config=optimized())
        for player in (first,second):
            player._prepare()
            player._history[0] = np.arange(648)
        a = [a for a,_ in first._ordered(state,0,None,unlimited())]
        b = [a for a,_ in second._ordered(SearchPosition(state),0,None,unlimited())]
        self.assertEqual(a,b)
        self.assertEqual(set(a),set(np.flatnonzero(IntransitiveGame().getValidMoves(state,0))))

    def test_all_flag_combinations_match_exhaustive_score(self):
        state = position({'D4':1,'F5':-3,'H7':2})
        expected,_ = exhaustive_minimax(IntransitiveGame(),state,2,optimized(),unlimited())
        before = state.tobytes()
        for pvs,aspiration,ordering,native,replacement,cache in product((False,True),repeat=6):
            config = optimized(pvs_enabled=pvs,aspiration_enabled=aspiration,ordering_enabled=ordering,
                compiled_ordering_enabled=native,depth_replacement_enabled=replacement,
                pressure_cache_entries=128 if cache else 0)
            result = AlphaBetaPlayer(config=config).analyze(state)
            self.assertEqual(result.completed_depth,2)
            self.assertEqual(result.score,expected)
            self.assertEqual(state.tobytes(),before)

    def test_public_and_compact_search_agree_with_all_optimisations(self):
        state = position({'D4':1,'F5':-3,'H7':2})
        config = optimized(max_depth=3)
        first = AlphaBetaPlayer(config=config,use_compact=False).analyze(state)
        second = AlphaBetaPlayer(config=config).analyze(state)
        self.assertEqual(first.completed_depth,3)
        self.assertEqual((first.action,first.score),(second.action,second.score))

    def test_depth_replacement_preserves_deeper_old_entries_and_valid_hints(self):
        player = AlphaBetaPlayer(config=optimized(table_entries=3))
        player._prepare()
        for key,depth in ((b'deep',5),(b'shallow',1),(b'middle',3),(b'new',2)):
            player._store(key,depth,0.,'exact',[1],0)
        self.assertIn((b'deep',5),player.table)
        self.assertNotIn((b'shallow',1),player.table)
        self.assertEqual(len(player.table),3)
        self.assertTrue(all((k,d) in player.table for k,d in player._hints.items()))

    def test_configuration_change_clears_table_and_learning_hints(self):
        player = AlphaBetaPlayer(config=optimized())
        player._prepare()
        player._store(b'old',1,0.,'exact',[1],0)
        player._killers[0,0] = 123
        player._history[0,0] = 42
        player.config = replace(player.config,pressure_radius=4)
        player._prepare()
        self.assertFalse(player.table)
        self.assertFalse(player._hints)
        self.assertEqual(player._killers[0,0],-1)
        self.assertEqual(player._history[0,0],0)

    def test_pressure_cache_is_bounded_exact_and_radius_separated(self):
        evaluator = Evaluator(IntransitiveGame(),optimized(pressure_cache_entries=2))
        state = position({'E5':1,'I5':-2})
        first,second = unlimited(),unlimited()
        value = evaluator.score(state,0,first)
        with patch('intransitive.heuristics.pressure.pressure_totals',side_effect=AssertionError('cache miss')):
            self.assertEqual(evaluator.score(state,0,second),value)
        self.assertEqual(first.work,second.work)
        self.assertEqual(second.module_calls['pressure_cache_hit'],1)
        evaluator.config = replace(evaluator.config,pressure_radius=4)
        self.assertNotEqual(evaluator.score(state,0,unlimited()),value)
        evaluator.score(position({'E5':1,'G5':-2}),0,unlimited())
        self.assertEqual(len(evaluator.pressure_cache),2)

    def test_cached_pressure_never_caches_draw_or_terminal_outcome(self):
        state = position({'E5':1,'G5':-2})
        repeated = load_history(Board(),[state[:,:,0]]*5)
        evaluator = Evaluator(IntransitiveGame(),optimized())
        self.assertNotEqual(evaluator.score(state,0,unlimited()),0.)
        self.assertEqual(evaluator.score(repeated,0,unlimited()),0.)
        self.assertNotEqual(position_key(state),position_key(repeated))

    def test_work_limits_restore_board_and_leave_legal_fallback(self):
        state = position({'D4':1,'F5':-3,'H7':2})
        before = state.tobytes()
        for work in (0,1,50,500,2000):
            result = AlphaBetaPlayer(config=optimized()).analyze(state,Budget(work,60))
            self.assertLessEqual(result.work,work)
            self.assertTrue(IntransitiveGame().getValidMoves(state,0)[result.action])
            self.assertEqual(state.tobytes(),before)

    def test_compiled_cancellation_and_search_error_unwind_mutation(self):
        state = SearchPosition(position({'D4':1,'F5':-3,'H7':2}))
        before = state.export().tobytes()
        for error in (BudgetExpired('time'),RuntimeError('injected')):
            player = AlphaBetaPlayer(config=optimized())
            player._prepare()
            with patch.object(player,'_leaf',side_effect=error):
                with self.assertRaises(type(error)):
                    player._search(state,2,-np.inf,np.inf,0,unlimited())
            self.assertEqual(state.export().tobytes(),before)
            self.assertFalse(state.stack)


if __name__=='__main__':
    unittest.main()
