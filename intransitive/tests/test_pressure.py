"""Independent square-window oracle and experimental evaluator regressions."""
from dataclasses import replace
from itertools import product
import json
import unittest
from unittest.mock import patch
import numpy as np

from intransitive.heuristics import AlphaBetaPlayer, SearchConfig, exhaustive_minimax
from intransitive.heuristics.budget import Budget, BudgetExpired
from intransitive.heuristics.evaluation import Evaluator, HEURISTIC_LIMIT, MATE_THRESHOLD
from intransitive.heuristics.position import SearchPosition
from intransitive.heuristics.pressure import pressure_totals, warm_pressure_kernel
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveSymmetries import transform_state
from intransitive.tests.test_heuristics import position, unlimited


def reference_pressure(board, side=0, window_radius=4):
    """Literal per-piece 9x9 scan, independent of production pair accounting."""
    def defenders(y,x,radius):
        victim = int(board[y,x])
        kind = abs(victim)%3+1
        return sum(board[dy,dx]*victim > 0 and abs(int(board[dy,dx])) == kind
                   for dy in range(max(0,y-radius),min(9,y+radius+1))
                   for dx in range(max(0,x-radius),min(9,x+radius+1)))
    total = 0.
    for y in range(9):
        for x in range(9):
            piece = int(board[y,x])
            if not piece or int(piece < 0) != side:
                continue
            for dy in range(-window_radius,window_radius+1):
                for dx in range(-window_radius,window_radius+1):
                    ny,nx = y+dy,x+dx
                    if not (0 <= ny < 9 and 0 <= nx < 9):
                        continue
                    enemy = int(board[ny,nx])
                    if piece*enemy >= 0 or abs(piece) == abs(enemy):
                        continue
                    radius = max(abs(dy),abs(dx))
                    weight = 2.**(2-radius)
                    if abs(piece)%3+1 == abs(enemy):
                        total += weight*2.**-defenders(ny,nx,radius)
                    else:
                        total -= weight*2.**-defenders(y,x,radius)
    return total


class PressureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warm_pressure_kernel()

    def totals(self, entries):
        board = np.zeros((9,9),dtype=np.int8)
        for square,code in entries.items():
            board[int(square[1])-1,ord(square[0])-65] = code
        return pressure_totals(board)

    def test_all_four_rings_and_diagonals_have_exact_weights(self):
        for radius,weight in enumerate((2.,1.,.5,.25),1):
            for dy,dx in ((0,radius),(radius,0),(radius,radius),(-radius,-radius)):
                board = np.zeros((9,9),dtype=np.int8)
                board[4,4],board[4+dy,4+dx] = 1,-2
                self.assertEqual(pressure_totals(board),(weight,0.))

    def test_seven_by_seven_drops_only_outer_ring_without_rescaling(self):
        for distance,expected in ((1,2.),(2,1.),(3,.5),(4,0.)):
            board = np.zeros((9,9),dtype=np.int8)
            board[4,4],board[4,4+distance] = 1,-2
            self.assertEqual(pressure_totals(board,3),(expected,0.))
        with self.assertRaises(ValueError):
            pressure_totals(board,2)

    def test_both_sizes_match_reference_on_random_boards(self):
        rng = np.random.default_rng(777)
        for _ in range(40):
            board = np.zeros((9,9),dtype=np.int8)
            board.flat[rng.choice(81,size=20,replace=False)] = rng.choice([-3,-2,-1,1,2,3],20)
            for radius in (3,4):
                blue,red = pressure_totals(board,radius)
                self.assertEqual(blue-red,reference_pressure(board,window_radius=radius))

    def test_seven_by_seven_evaluator_and_configuration_identity(self):
        state = position({'E5':2,'D5':3,'H5':-1,'A5':-1})
        config = SearchConfig(pressure_enabled=True,pressure_weight=10.,pressure_radius=3,proof_nodes=0)
        evaluator = Evaluator(IntransitiveGame(),config)
        report = evaluator.explain(state,0,unlimited(),diagnostics=False)
        self.assertEqual(report['terms']['local_pressure'],10.*reference_pressure(state[:,:,0],window_radius=3))
        self.assertEqual(evaluator.score(state,0,unlimited()),report['score'])
        self.assertEqual(evaluator.score(SearchPosition(state),0,unlimited()),report['score'])
        self.assertNotEqual(config.identity(),replace(config,pressure_radius=4).identity())
        self.assertEqual(SearchConfig(**json.loads(config.identity())),config)
        for radius in (True,2,5,3.,'3'):
            with self.assertRaises(ValueError):
                replace(config,pressure_radius=radius)

    def test_padding_does_not_wrap_and_radius_five_is_excluded(self):
        self.assertEqual(self.totals({'A1':1,'I9':-2}),(0.,0.))
        self.assertEqual(self.totals({'A1':1,'F1':-2}),(0.,0.))
        self.assertEqual(self.totals({'A1':1,'E5':-2}),(.25,0.))
        self.assertEqual(self.totals({'A1':1,'B2':-2}),(2.,0.))

    def test_rps_cycles_and_same_type_neutrality(self):
        for kind in (1,2,3):
            self.assertEqual(self.totals({'E5':kind,'F5':-kind}),(0.,0.))
            self.assertEqual(self.totals({'E5':kind,'F5':-(kind%3+1)}),(2.,0.))
            self.assertEqual(self.totals({'E5':kind,'F5':-((kind+1)%3+1)}),(0.,2.))

    def test_adjacent_defender_halves_every_ring(self):
        for radius,weight in enumerate((2.,1.,.5,.25),1):
            # Blue scissors defended by Blue paper against Red rock.
            self.assertEqual(self.totals({'E5':2,'D5':3,f'{chr(69+radius)}5':-1})[1],weight/2)

    def test_outer_defender_only_discounts_outer_threats(self):
        for radius,weight in enumerate((2.,1.,.5,.25),1):
            totals = self.totals({'E5':2,'A5':3,f'{chr(69+radius)}5':-1})
            self.assertEqual(totals[1],weight/(2 if radius == 4 else 1))

    def test_multiple_defenders_compound_and_boundary_is_inclusive(self):
        self.assertEqual(self.totals({'E5':2,'D5':3,'E4':3,'F5':-1})[1],.5)
        self.assertEqual(self.totals({'E5':2,'C5':3,'G5':-1})[1],.5)
        self.assertEqual(self.totals({'E5':2,'B5':3,'G5':-1})[1],1.)

    def test_defender_count_is_unconditional_and_blockers_ignored(self):
        # Red scissors threaten the supporting paper; it still defends Blue scissors.
        self.assertEqual(self.totals({'E5':2,'D5':3,'D4':-2,'F5':-1})[1],3.)
        self.assertEqual(self.totals({'E5':2,'G5':-1})[1],1.)
        # Same-type Red blocker contributes no new predator/prey pair.
        self.assertEqual(self.totals({'E5':2,'F5':-2,'G5':-1})[1],1.)

    def test_attack_reward_respects_defenders_around_victim_not_attacker(self):
        # Victim is D5 scissors; A5 paper is inside its radius-4 ring but
        # outside the attacking H5 rock's window.
        self.assertEqual(self.totals({'H5':1,'D5':-2,'A5':-3})[0],.125)

    def test_random_boards_match_literal_reference_and_colour_antisymmetry(self):
        rng = np.random.default_rng(7285)
        for _ in range(80):
            board = np.zeros((9,9),dtype=np.int8)
            board.flat[rng.choice(81,size=20,replace=False)] = rng.choice([-3,-2,-1,1,2,3],20)
            blue,red = pressure_totals(board)
            self.assertEqual(blue-red,reference_pressure(board,0))
            self.assertEqual(red-blue,reference_pressure(board,1))
            self.assertEqual(pressure_totals(-board),(red,blue))

    def test_all_twelve_game_symmetries(self):
        state = position({'D4':1,'F5':-2,'G5':-3,'C6':3,'F7':2,'B3':-1})
        for radius in (3,4):
            original = pressure_totals(state[:,:,0],radius)
            for symmetry in range(12):
                expected = original if symmetry < 6 else original[::-1]
                self.assertEqual(pressure_totals(transform_state(state,symmetry)[:,:,0],radius),expected)

    def test_compact_public_and_explanation_agree_without_routes(self):
        state = position({'D4':1,'F5':-2,'G5':-3,'C6':3})
        config = SearchConfig(pressure_enabled=True,pressure_weight=10.,proof_nodes=0)
        evaluator = Evaluator(IntransitiveGame(),config)
        original = state.tobytes()
        with patch('intransitive.heuristics.evaluation.Geometry',side_effect=AssertionError('no routes')):
            budget = unlimited()
            score = evaluator.score(state,0,budget)
            compact = evaluator.score(SearchPosition(state),0,unlimited())
        explanation = evaluator.explain(state,0,unlimited(),diagnostics=False)
        self.assertEqual(score,compact)
        self.assertEqual(score,explanation['score'])
        self.assertEqual(budget.module_calls['local_pressure'],1)
        self.assertEqual(budget.module_calls['routes'],0)
        self.assertEqual(explanation['terms']['local_pressure'],10.*reference_pressure(state[:,:,0]))
        self.assertEqual(state.tobytes(),original)
        json.dumps(explanation,allow_nan=False)

    def test_disabled_and_zero_weight_do_no_pressure_work(self):
        state = position({'D4':1,'F5':-2})
        base = SearchConfig(proof_nodes=0)
        expected = Evaluator(IntransitiveGame(),base).score(state,0,unlimited())
        with patch('intransitive.heuristics.pressure.pressure_totals',side_effect=AssertionError('disabled')):
            for config in (base,replace(base,pressure_enabled=True,pressure_weight=0.)):
                budget = unlimited()
                self.assertEqual(Evaluator(IntransitiveGame(),config).score(state,0,budget),expected)
                self.assertEqual(budget.module_calls['local_pressure'],0)

    def test_budget_and_configuration_validation(self):
        config = SearchConfig(pressure_enabled=True,pressure_weight=10.)
        self.assertEqual(SearchConfig(**json.loads(config.identity())),config)
        self.assertNotEqual(config.identity(),replace(config,pressure_weight=20.).identity())
        for kwargs in (dict(pressure_enabled=1),dict(pressure_weight=-1),dict(pressure_weight=float('nan'))):
            with self.assertRaises(ValueError):
                SearchConfig(**kwargs)
        with patch('intransitive.heuristics.pressure.pressure_totals',side_effect=AssertionError('over budget')):
            with self.assertRaises(BudgetExpired):
                Evaluator(IntransitiveGame(),config)._pressure(np.zeros((9,9),dtype=np.int8),Budget(0,30),0)

    def test_terminal_winning_run_beats_even_huge_pressure(self):
        state = position({'H8':2,'D4':-1,'E4':-1,'F4':-1})
        config = SearchConfig(pressure_enabled=True,pressure_weight=1e9,max_depth=2,
                              node_limit=1000000,time_limit=30.)
        result = AlphaBetaPlayer(config=config).analyze(state)
        self.assertGreater(result.score,MATE_THRESHOLD)
        child,_ = IntransitiveGame().getNextState(state,0,result.action)
        self.assertEqual(IntransitiveGame().getGameEnded(child,1)[0],1)
        quiet = position({'D4':1,'F5':-2})
        self.assertEqual(Evaluator(IntransitiveGame(),replace(config,proof_nodes=0)).score(
            quiet,0,unlimited()),HEURISTIC_LIMIT)

    def test_pressure_adds_same_term_with_every_other_feature_combination(self):
        state = position({'D4':1,'F5':-2,'G5':-3,'C6':3})
        game = IntransitiveGame()
        expected = 10.*reference_pressure(state[:,:,0])
        for attack,defence,overloaded in product((False,True),repeat=3):
            base = SearchConfig(proof_nodes=0,attack_enabled=attack,defence_enabled=defence,
                                overload_enabled=overloaded)
            config = replace(base,pressure_enabled=True,pressure_weight=10.)
            old = Evaluator(game,base).score(state,0,unlimited())
            evaluator = Evaluator(game,config)
            current = evaluator.score(state,0,unlimited())
            self.assertAlmostEqual(current-old,expected)
            self.assertAlmostEqual(current,evaluator.explain(state,0,unlimited(),diagnostics=False)['score'])

    def test_search_matches_exhaustive_and_public_reference(self):
        state = position({'E5':1,'F5':-3})
        config = SearchConfig(pressure_enabled=True,pressure_weight=10.,proof_nodes=0,
                              max_depth=2,node_limit=1000000,time_limit=30.)
        reference = exhaustive_minimax(IntransitiveGame(),state,2,config,unlimited())
        for compact in (False,True):
            result = AlphaBetaPlayer(config=config,use_compact=compact).analyze(state)
            self.assertEqual(result.completed_depth,2)
            self.assertEqual(result.score,reference[0])

    def test_opening_blunder_is_no_longer_tied_with_safe_alternative_at_five_plies(self):
        from intransitive.benchmarks.pressure.reproduce import opening_state
        from intransitive.IntransitiveDisplay import parse_move
        state = opening_state()
        game = IntransitiveGame()
        scores = {}
        for weight in (0.,10.):
            config = SearchConfig(pressure_enabled=bool(weight),pressure_weight=weight,
                                  max_depth=4,node_limit=1_000_000_000,time_limit=60.)
            scores[weight] = []
            for move in ('F6 E5','F6 F5'):
                child,_ = game.getNextState(state,1,parse_move(move))
                result = AlphaBetaPlayer(config=config).analyze(child)
                self.assertEqual(result.completed_depth,4)
                scores[weight].append(-result.score)
        self.assertAlmostEqual(*scores[0.])
        self.assertGreater(scores[10.][1]-scores[10.][0],20.)


if __name__=='__main__':
    unittest.main()
