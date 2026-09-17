"""Value-aware capture priorities, complete move sets and exact unpruned scores."""
from dataclasses import replace
from itertools import product
import unittest
import numpy as np
from intransitive.heuristics import AlphaBetaPlayer,SearchConfig,exhaustive_minimax
from intransitive.heuristics.ordering import ordered_actions,material_order_values,warm_ordering
from intransitive.heuristics.material import count_pieces
from intransitive.heuristics.position import SearchPosition
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveDisplay import parse_move
from intransitive.rust_teacher import BINARY,RustTeacher
from intransitive.tests.test_heuristics import position,unlimited


def fixture():
    return position({'D4':1,'F4':2,'C6':3,'B2':3,'E4':-2,'G4':-3,'D6':-1})


class MvvLvaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warm_ordering()
        AlphaBetaPlayer(config=SearchConfig())._prepare()

    def test_formula_uses_each_armys_own_values_and_restores_after_capture(self):
        counts=(1,4,4,5,8,2)
        values=material_order_values(counts,0,True)
        self.assertAlmostEqual(values[0,0],100*8.25/2.25*(3.25/1.25)**.5)
        self.assertAlmostEqual(values[1,0],100.)
        np.testing.assert_array_equal(material_order_values(counts,1,True),values[::-1])
        np.testing.assert_array_equal(material_order_values(counts,0,False),np.full((2,3),100.))
        p=SearchPosition(fixture());before=material_order_values(p.counts,0,True)
        p.push(parse_move('D4 E4'))
        self.assertFalse(np.array_equal(before,material_order_values(p.counts,0,True)))
        p.pop();np.testing.assert_array_equal(before,material_order_values(p.counts,0,True))

    def test_most_valuable_victim_then_least_valuable_attacker(self):
        board=fixture()[:,:,0]
        actions=np.array([parse_move(s) for s in ('D4 E4','F4 G4','C6 D6')],dtype=np.int64)
        args=(board,actions,0,80,-1,np.full(648,-np.inf),np.full(2,-1,dtype=np.int64),np.zeros(648,dtype=np.int64),False)
        values=np.array([[10.,20.,30.],[100.,200.,200.]])
        for kernel in (ordered_actions,ordered_actions.py_func):
            np.testing.assert_array_equal(kernel(*args,values),actions)
            high_victim=values.copy();high_victim[1,0]=300.
            np.testing.assert_array_equal(kernel(*args,high_victim),actions[[2,0,1]])
            preferred=list(args);preferred[4]=actions[1]
            self.assertEqual(kernel(*preferred,high_victim)[0],actions[1])

    def test_reference_compiled_compact_and_enhanced_keep_every_move(self):
        state=fixture();game=IntransitiveGame();legal=set(np.flatnonzero(game.getValidMoves(state,0)))
        for variable,enhanced in product((False,True),repeat=2):
            outputs=[]
            for compiled,compact in product((False,True),repeat=2):
                cfg=SearchConfig(mvv_lva_enabled=True,variable_material_enabled=variable,
                    ordering_enabled=enhanced,compiled_ordering_enabled=compiled)
                player=AlphaBetaPlayer(config=cfg);player._prepare()
                board=SearchPosition(state) if compact else state
                order=[a for a,_ in player._ordered(board,0,None,unlimited())]
                self.assertEqual(set(order),legal);self.assertEqual(len(order),len(legal))
                outputs.append(order)
            self.assertTrue(all(o==outputs[0] for o in outputs))
            if not variable:
                baseline=AlphaBetaPlayer(config=SearchConfig(ordering_enabled=enhanced));baseline._prepare()
                self.assertEqual(outputs[0],[a for a,_ in baseline._ordered(state,0,None,unlimited())])
        winning=position({'I8':1,'D4':1,'E4':-2,'F6':-3})
        player=AlphaBetaPlayer(config=SearchConfig(mvv_lva_enabled=True,variable_material_enabled=True));player._prepare()
        self.assertEqual(next(player._ordered(winning,0,parse_move('D4 E4'),unlimited()))[0],parse_move('I8 I9'))

    def test_exact_scores_configuration_identity_and_budget_restoration(self):
        state=fixture();before=state.tobytes();game=IntransitiveGame()
        for variable in (False,True):
            cfg=SearchConfig(variable_material_enabled=variable,max_depth=2,proof_nodes=0,time_limit=60,node_limit=10**9,pvs_enabled=True)
            expected,_=exhaustive_minimax(game,state,2,cfg,unlimited())
            player=AlphaBetaPlayer(config=cfg)
            for enabled in (False,True,False):
                player.config=replace(cfg,mvv_lva_enabled=enabled)
                result=player.analyze(state)
                self.assertAlmostEqual(result.score,expected)
                self.assertEqual(result.ordering['mvv_lva_enabled'],enabled)
                self.assertEqual(result.completed_depth,2)
                if enabled:self.assertGreater(result.ordering['mvv_lva_captures'],0)
                self.assertEqual(state.tobytes(),before)
            self.assertNotEqual(cfg.identity(),replace(cfg,mvv_lva_enabled=True).identity())
            for cap in (0,10,1000):
                result=AlphaBetaPlayer(config=replace(cfg,mvv_lva_enabled=True,node_limit=cap)).analyze(state)
                self.assertTrue(game.getValidMoves(state,0)[result.action]);self.assertEqual(state.tobytes(),before)
        with self.assertRaises(ValueError):SearchConfig(mvv_lva_enabled=1)

    def test_material_weight_five_is_exactly_base_divided_by_twenty(self):
        from intransitive.heuristics.evaluation import Evaluator
        cfg=SearchConfig(variable_material_enabled=True)
        game=IntransitiveGame();state=fixture();proof={'status':'unknown'}
        ordinary=Evaluator(game,cfg).explain(state,0,unlimited(),proof=proof,diagnostics=False)
        scaled=Evaluator(game,replace(cfg,count_weight=5.)).explain(state,0,unlimited(),proof=proof,diagnostics=False)
        self.assertAlmostEqual(scaled['terms']['piece_count'],ordinary['terms']['piece_count']/20.)
        for key in ordinary['terms']:
            if key!='piece_count':self.assertEqual(ordinary['terms'][key],scaled['terms'][key])
        self.assertAlmostEqual(scaled['raw_score'],sum(scaled['terms'].values()))

    @unittest.skipUnless(BINARY.exists(),'build Rust first')
    def test_native_protocol_parity_and_reuse_isolation(self):
        state=fixture()
        with RustTeacher() as rust:
            for variable in (False,True):
                for enabled in (False,True,False):
                    kwargs=dict(depth=2,seconds=60.,proof_nodes=0,variable_material_enabled=variable,mvv_lva_enabled=enabled)
                    native=rust.analyze(state,reuse=True,**kwargs)
                    fresh=rust.analyze(state,reuse=False,**kwargs)
                    cfg=SearchConfig(max_depth=2,time_limit=60,node_limit=10**9,proof_nodes=0,pvs_enabled=True,
                        ordering_enabled=True,compiled_ordering_enabled=True,variable_material_enabled=variable,mvv_lva_enabled=enabled)
                    py=AlphaBetaPlayer(config=cfg).analyze(state)
                    self.assertTrue(native['complete']);self.assertAlmostEqual(native['score'],py.score,places=8)
                    self.assertEqual((native['action'],native['score']),(fresh['action'],fresh['score']))
                    self.assertEqual(native['ordering']['mvv_lva_enabled'],enabled)
                    if enabled:self.assertGreater(native['ordering']['mvv_lva_captures'],0)
            combined=rust.analyze(state,depth=2,seconds=60.,proof_nodes=0,mvv_lva_enabled=True,
                variable_material_enabled=True,nmp_enabled=True,futility_enabled=True,selective_evaluator_enabled=True)
            self.assertTrue(combined['ordering']['mvv_lva_enabled'])
            self.assertTrue(combined['selective']['effective'])
            with self.assertRaises(ValueError):rust.analyze(state,mvv_lva_enabled=1)

if __name__=='__main__':unittest.main()
