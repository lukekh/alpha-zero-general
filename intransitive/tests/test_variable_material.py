"""Variable material formula, effective mode identity, and native/search parity."""
from dataclasses import replace
import math
import tempfile
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

import numpy as np

from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.heuristics import AlphaBetaPlayer, SearchConfig
from intransitive.heuristics.evaluation import Evaluator, HEURISTIC_LIMIT, MATE
from intransitive.heuristics.material import (BASE, REG, count_pieces,
    variable_piece_values, variable_material_total, warm_material_kernels)
from intransitive.rust_teacher import BINARY, RustTeacher
from intransitive.tests.test_heuristics import position, unlimited
from intransitive.tests.test_tournament import FakeEngine
from intransitive.tournament.spec import candidate, manifest, position as opening, protocol, effective_config
from intransitive.tournament.runner import play_match, validate_manifest
from intransitive.benchmarks.evolution.adjudicate import adjudicate, configuration_hash

UNKNOWN = {'status': 'unknown'}


class VariableMaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warm_material_kernels()
        cls.game = IntransitiveGame()
        cls.config = SearchConfig(variable_material_enabled=True, advantage_weight=0,
            attack_enabled=False, defence_enabled=False, proof_depth=0, proof_nodes=0,
            max_depth=1, node_limit=10**7, time_limit=10)

    def test_formula_and_cyclic_piece_types(self):
        own, enemy = (1, 4, 4), (5, 8, 2)  # ROCK, SCISSORS, PAPER
        expected = [100 * 8.25 / 2.25 * math.sqrt(3.25 / 1.25),
                    100 * 2.25 / 5.25 * math.sqrt(3.25 / 4.25),
                    100 * 5.25 / 8.25 * math.sqrt(3.25 / 4.25)]
        np.testing.assert_allclose(variable_piece_values(own, enemy), expected, rtol=1e-15)
        np.testing.assert_allclose(variable_piece_values(own[1:]+own[:1], enemy[1:]+enemy[:1]), expected[1:]+expected[:1])
        self.assertEqual((BASE, REG), (100., .25))

    def test_balance_extinction_and_absent_types(self):
        np.testing.assert_array_equal(variable_piece_values((2,2,2), (3,3,3)), [100.,100.,100.])
        rare = variable_piece_values((1,4,4), (3,3,3))
        self.assertGreater(rare[0], rare[1])
        self.assertEqual(rare[1], rare[2])
        self.assertTrue(np.isfinite(variable_piece_values((0,0,0), (0,0,0))).all())
        self.assertEqual(variable_material_total((0,0,0,9,0,0), 0), 0.)
        self.assertEqual(variable_material_total((1,0,0,0,0,0), 0),
            100 * math.sqrt((1/3+.25)/1.25))
        self.assertGreater(variable_piece_values((1,1,1), (0,3,0))[0],
                           variable_piece_values((1,1,1), (0,0,3))[0])

    def test_cached_reference_and_player_perspective(self):
        rng = np.random.default_rng(66)
        state = self.game.getInitBoard()
        for _ in range(35):
            for config in (self.config, replace(self.config, advantage_weight=23., attack_enabled=True, defence_enabled=True)):
                evaluator = Evaluator(self.game, config)
                values=[]
                for side in (0,1):
                    explained=evaluator.explain(state, side, unlimited(), proof=UNKNOWN, diagnostics=False)
                    actual=evaluator.score(state, side, unlimited(), proof=UNKNOWN, counts=count_pieces(state))
                    self.assertEqual(actual, explained['score'])
                    values.append(actual)
                self.assertEqual(values[0], -values[1])
            side=int(state[:,:,82:84].flat[1]);legal=np.flatnonzero(self.game.getValidMoves(state,side))
            state,side=self.game.getNextState(state,side,int(rng.choice(legal)))
            if self.game.getGameEnded(state,side).any():break

    def test_explanation_replacement_and_search(self):
        state=position({'D4':1,'E4':1,'H7':-2,'G6':-3})
        evaluator=Evaluator(self.game,self.config)
        result=evaluator.explain(state,0,unlimited(),proof=UNKNOWN,diagnostics=False)
        counts=count_pieces(state)
        expected=variable_material_total(counts,0)-variable_material_total(counts,1)
        self.assertAlmostEqual(result['raw_score'],expected)
        self.assertIn('rock', result['piece_values']['own'])
        self.assertEqual(Evaluator(self.game,replace(self.config,variable_material_enabled=False)).score(state,0,unlimited(),proof=UNKNOWN),0.)
        search=AlphaBetaPlayer(config=self.config).analyze(state)
        self.assertFalse(search.stopped)
        self.assertEqual(search.completed_depth,1)
        self.assertTrue(self.game.getValidMoves(state,0)[search.action])
        self.assertLessEqual(abs(result['score']),HEURISTIC_LIMIT)

    def test_capture_updates_all_values_and_preserves_parent(self):
        from intransitive.IntransitiveConstants import action_destination
        state=position({'D4':1,'E5':-2,'H7':-3,'F6':2})
        evaluator=Evaluator(self.game,self.config)
        before=evaluator.score(state,0,unlimited(),proof=UNKNOWN)
        action=next(int(a) for a in np.flatnonzero(self.game.getValidMoves(state,0))
                    if action_destination(int(a)) == (4,4))
        child,_=self.game.getNextState(state,0,action)
        counts=count_pieces(child)
        self.assertEqual(sum(counts),3)
        expected=variable_material_total(counts,0)-variable_material_total(counts,1)
        self.assertAlmostEqual(evaluator.score(child,0,unlimited(),proof=UNKNOWN,counts=counts),expected)
        self.assertEqual(evaluator.score(state,0,unlimited(),proof=UNKNOWN),before)

    def test_mcts_uses_variable_leaf_values(self):
        from intransitive.heuristics.mcts import HeuristicMCTSPlayer, HeuristicValue
        state=position({'D4':1,'E4':1,'H7':-2,'G6':-3})
        evaluator=Evaluator(self.game,self.config)
        expected=evaluator.score(state,0,unlimited(),proof=UNKNOWN)
        _,values=HeuristicValue(evaluator,unlimited(),400.).predict(state,self.game.getValidMoves(state,0))
        self.assertAlmostEqual(values[0],math.tanh(expected/400.))
        self.assertEqual(values[0],-values[1])
        result=HeuristicMCTSPlayer(config=self.config,settings=protocol('mcts',simulations=8)['mcts']).analyze(state)
        self.assertFalse(result.stopped)
        self.assertEqual(result.completed_simulations,8)
        self.assertTrue(self.game.getValidMoves(state,0)[result.action])

    def test_clipping_and_decisive_scores(self):
        state=position({'D4':1,'E4':1,'H8':-2})
        config=replace(self.config,count_weight=10**9)
        result=Evaluator(self.game,config).explain(state,0,unlimited(),proof=UNKNOWN,diagnostics=False)
        self.assertEqual(result['score'],HEURISTIC_LIMIT)
        self.assertGreater(result['raw_score'],HEURISTIC_LIMIT)
        self.assertEqual(Evaluator(self.game,config).score(state,0,unlimited(),
            proof={'status':'proven','score':MATE-1}),MATE-1)
        terminal=position({'I9':1,'D4':-2})
        self.assertEqual(Evaluator(self.game,config).score(terminal,0,unlimited(),proof=UNKNOWN),MATE)

    def test_mode_identity_and_consensus_use_effective_configuration(self):
        flat=candidate('flat');variable=candidate('variable',variable_material_enabled=True)
        self.assertEqual(flat['genome'],variable['genome'])
        self.assertNotEqual(flat['sha256'],variable['sha256'])
        self.assertNotEqual(configuration_hash(flat),configuration_hash(variable))
        spec=manifest([flat,variable],[opening([],seed=54,stage='official',pool='search')],
                      [protocol('depth',depth=1,max_plies=1,seconds=10)])
        validate_manifest(spec)
        self.assertTrue(effective_config(variable,spec['protocols'][0]).variable_material_enabled)
        FakeEngine.action=FakeEngine.cancel=FakeEngine.fail=None;FakeEngine.stopped=False
        with tempfile.TemporaryDirectory() as folder:
            row=play_match(spec,spec['tasks'][0],Path(folder)/'game.json',threading.Event(),engine_factory=FakeEngine)
        modes=[]
        def score(evaluator,*args,**kwargs):
            modes.append(evaluator.config.variable_material_enabled)
            return 1.
        with patch.object(Evaluator,'score',score):result=adjudicate(spec,row)
        self.assertEqual(sorted(modes),[False,True])
        self.assertEqual(result['outcome'],'consensus')
        with self.assertRaises(ValueError):SearchConfig(variable_material_enabled=1)
        with self.assertRaises(ValueError):RustTeacher.material_mode(1)
        from intransitive.heuristics.tuning import Genome
        with self.assertRaises(ValueError):Genome.from_genes().to_config(SearchConfig(variable_material_enabled=True))

    @unittest.skipUnless(BINARY.exists(), 'Build the native teacher first')
    def test_native_parity_and_mode_switch_invalidates_reused_search(self):
        with RustTeacher() as native:
            for turn in (0,1):
                state=position({'D4':1,'E4':1,'F6':2,'H7':-2,'G6':-3},turn=turn)
                for enabled in (False,True,False):
                    config=replace(self.config,variable_material_enabled=enabled,advantage_weight=23.,attack_enabled=True,defence_enabled=True)
                    expected=Evaluator(self.game,config).score(state,turn,unlimited(),proof=UNKNOWN)
                    args=dict(material=config.count_weight,advantage=config.advantage_weight,
                        attack=config.attack_weight,defence=config.defence_weight,variable_material_enabled=enabled)
                    result=native.inspect(state,**args)
                    self.assertAlmostEqual(result['score'],expected,places=10)
                    reused=native.analyze(state,depth=1,seconds=10,proof_depth=0,proof_nodes=0,reuse=True,**args)
                    fresh=native.analyze(state,depth=1,seconds=10,proof_depth=0,proof_nodes=0,reuse=False,**args)
                    self.assertTrue(reused['complete'])
                    self.assertEqual((reused['action'],reused['score']),(fresh['action'],fresh['score']))


if __name__=='__main__':unittest.main()


class LinearScarcityTests(unittest.TestCase):
    """The variant that drops the square root from the own-scarcity factor.

    It is a different valuation, not a cheaper approximation of the same one,
    so these tests pin the formula and the guard rather than any closeness.
    """

    ARMIES = (((3, 3, 4), (2, 4, 4)), ((1, 0, 5), (0, 3, 3)),
              ((2, 2, 2), (2, 2, 2)), ((0, 0, 1), (1, 0, 0)), ((4, 1, 1), (2, 2, 2)))

    def test_both_forms_match_the_written_formula(self):
        from intransitive.heuristics.material import BASE, REG, variable_piece_values
        for own, enemy in self.ARMIES:
            target = sum(own) / 3.
            for linear in (False, True):
                values = variable_piece_values(own, enemy, linear)
                for kind in range(3):
                    prey, predator = enemy[(kind + 1) % 3], enemy[(kind + 2) % 3]
                    scarcity = (target + REG) / (own[kind] + REG)
                    expected = (BASE * (prey + REG) / (predator + REG)
                                * (scarcity if linear else math.sqrt(scarcity)))
                    with self.subTest(own=own, enemy=enemy, linear=linear, kind=kind):
                        self.assertAlmostEqual(values[kind], expected, places=12)

    def test_an_even_army_is_unchanged_by_the_variant(self):
        from intransitive.heuristics.material import variable_piece_values
        # Scarcity is exactly one there, and one is its own square root.
        np.testing.assert_allclose(variable_piece_values((2, 2, 2), (2, 2, 2), False),
                                   variable_piece_values((2, 2, 2), (2, 2, 2), True))

    def test_the_variant_prices_concentration_more_aggressively(self):
        from intransitive.heuristics.material import variable_piece_values
        own, enemy = (1, 0, 5), (0, 3, 3)
        sqrt = variable_piece_values(own, enemy, False)
        linear = variable_piece_values(own, enemy, True)
        # Scarce types gain, abundant ones lose, relative to the square root.
        self.assertGreater(linear[0], sqrt[0])
        self.assertLess(linear[2], sqrt[2])

    def test_the_flag_requires_variable_material(self):
        from intransitive.heuristics.config import SearchConfig
        with self.assertRaises(ValueError):
            SearchConfig(variable_material_linear=True)
        self.assertTrue(SearchConfig(variable_material_enabled=True,
                                     variable_material_linear=True).variable_material_linear)

    def test_the_variant_is_off_by_default(self):
        from intransitive.heuristics.config import SearchConfig
        self.assertFalse(SearchConfig().variable_material_linear)
        self.assertFalse(SearchConfig(variable_material_enabled=True).variable_material_linear)

    def test_the_two_modes_score_positions_differently(self):
        from intransitive.heuristics.budget import Budget
        from intransitive.heuristics.config import SearchConfig
        from intransitive.heuristics.evaluation import Evaluator
        from intransitive.IntransitiveGame import IntransitiveGame
        from intransitive.tests.test_attribution import random_positions
        game = IntransitiveGame()
        common = dict(variable_material_enabled=True, count_weight=5., advantage_weight=0.,
                      attack_enabled=False, defence_enabled=False, proof_depth=0)
        sqrt = Evaluator(game, SearchConfig(**common))
        linear = Evaluator(game, SearchConfig(variable_material_linear=True, **common))
        states = random_positions(games=4, plies=40, seed=83)
        differing = sum(sqrt.score(s, 0, Budget(10**12, 3600.))
                        != linear.score(s, 0, Budget(10**12, 3600.)) for s in states)
        self.assertGreater(differing, 0, 'the variant never changed a score')
