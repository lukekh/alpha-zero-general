"""Signed scale semantics, native route parity and heuristic PUCT integration."""
from dataclasses import asdict, replace
import random
import unittest
import numpy as np
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.heuristics.config import SearchConfig
from intransitive.heuristics.tuning import Genome, DEFAULTS, BOUNDS
from intransitive.heuristics.evaluation import Evaluator
from intransitive.heuristics.mcts import HeuristicMCTSPlayer, HeuristicValue
from intransitive.evolution.strategy import mutate, Settings
from intransitive.rust_teacher import BINARY, RustTeacher
from intransitive.tests.test_heuristics import position, unlimited
from intransitive.tournament.spec import protocol, candidate, manifest, generate_positions


class SignedTests(unittest.TestCase):
    def test_sign_reversal_material_and_bound(self):
        state = position({'D4': 1, 'E4': 1, 'H8': -2})
        game = IntransitiveGame(); values = {}
        for sign in (-1, 1):
            g = Genome.from_genes({k:sign*v for k,v in DEFAULTS.items()})
            c = replace(g.to_config(), proof_nodes=0)
            values[sign] = Evaluator(game,c).explain(state,0,unlimited(),proof={'status':'unknown'},diagnostics=False)
        self.assertAlmostEqual(values[-1]['raw_score'], -values[1]['raw_score'])
        for term, value in values[1]['terms'].items():
            self.assertAlmostEqual(values[-1]['terms'][term], -value)
        g = Genome.from_genes({k:-100 for k in BOUNDS})
        m = g.manifest(implementation_revision='signed-test')
        self.assertGreater(m['conservative_absolute_bound'], 0)
        self.assertTrue(m['saturation_possible'])
        self.assertEqual(g.to_config().count_weight, -100)

    def test_mutation_crosses_sign_and_preserves_zero_atom(self):
        rng=random.Random(83);settings=Settings(mutation_rate=1,toggle_rate=0,mutation_sigma=.5)
        parent=Genome.from_genes({k:1 for k in BOUNDS})
        children=[mutate(parent,rng,settings) for _ in range(50)]
        for i in range(len(BOUNDS)):
            self.assertTrue(any(g.values[i]<0 for g in children))
            self.assertTrue(any(g.values[i]>0 for g in children))
        self.assertTrue(all(-100<=v<=100 for g in children for v in g.values))

    def test_mcts_budget_values_and_determinism(self):
        c=SearchConfig(time_limit=20,node_limit=10**7,proof_nodes=0,proof_depth=0)
        limits=protocol('mcts',simulations=8)
        state=position({'D4':1,'E5':-2,'H8':-3})
        model=HeuristicValue(Evaluator(IntransitiveGame(),c),unlimited(),400.)
        legal=IntransitiveGame().getValidMoves(state,0)
        priors,values=model.predict(state,legal)
        self.assertAlmostEqual(priors.sum(),1.)
        self.assertTrue(np.all(priors[legal==0]==0))
        self.assertEqual(values[0],-values[1]);self.assertLessEqual(abs(values[0]),1.)
        a=HeuristicMCTSPlayer(config=c,settings=limits['mcts']).analyze(state)
        b=HeuristicMCTSPlayer(config=c,settings=limits['mcts']).analyze(state)
        self.assertEqual((a.action,a.score,a.work),(b.action,b.score,b.work))
        self.assertEqual(a.completed_depth,0);self.assertEqual(a.completed_simulations,8)
        self.assertGreater(a.max_tree_depth,0);self.assertFalse(a.stopped)
        self.assertTrue(legal[a.action])
        stopped=HeuristicMCTSPlayer(config=replace(c,time_limit=0),settings=limits['mcts']).analyze(state)
        self.assertTrue(stopped.stopped);self.assertEqual(stopped.completed_simulations,0)
        self.assertEqual(stopped.stop_reason,'time')

    def test_harness_rejects_incomplete_simulations(self):
        import tempfile, threading
        from pathlib import Path
        from intransitive.tests.test_tournament import FakeEngine
        from intransitive.tournament.runner import play_match
        class Partial(FakeEngine):
            action = cancel = fail = None
            stopped = False
            def receive(self, deadline, cancelled):
                row = super().receive(deadline, cancelled)
                if row['kind'] == 'move':
                    row['result'].update(search_kind='mcts', completed_depth=0,
                        completed_simulations=3, requested_simulations=4, stopped=True,
                        selection_source='partial_simulations')
                return row
        spec=manifest([candidate('a'),candidate('b',{'count_weight':-100})],
            generate_positions(54,9),[protocol('mcts',simulations=4,max_plies=2)],position_limit=1)
        with tempfile.TemporaryDirectory() as folder:
            row=play_match(spec,spec['tasks'][0],Path(folder)/'match.json',threading.Event(),engine_factory=Partial)
        self.assertEqual(row['status'],'simulation_incomplete')
        self.assertIsNone(row['winner']);self.assertEqual(row['moves'],[])

    def test_mcts_protocol_identity_includes_search_settings(self):
        starts=generate_positions(54,9)
        players=[candidate('a'),candidate('b',{'count_weight':-100})]
        ids=[]
        for settings in ({},{'simulations':64},{'value_scale':100},{'cpuct':2}):
            ids.append(manifest(players,starts,[protocol('mcts',**settings)])['sha256'])
        self.assertEqual(len(set(ids)),4)


@unittest.skipUnless(BINARY.exists(),'Build Rust teacher first')
class SignedNativeTests(unittest.TestCase):
    def test_routes_signed_material_and_candidate_switches(self):
        from intransitive.tournament.spec import generate_positions,unpack
        fixtures=[position({'B2':2,'D4':-3,'H8':1}),position({'G7':3,'A9':-1}),
                  position({'D4':1,'E5':-2,'H8':-3},a1=1)]
        fixtures += [unpack(p['state']) for p in generate_positions(739,6) if p['stage'] in ('endgame','midgame')]
        game=IntransitiveGame()
        with RustTeacher() as rust:
            for overrides in ({},{'material':-80,'advantage':-20,'attack':-35,'defence':-40,'pressure':-10},
                              {'material':0,'advantage':0,'attack':0,'defence':80,'pressure':0}):
                genome=Genome.from_genes(overrides,backend='rust');config=genome.to_config()
                kwargs=genome.native_arguments(replace(SearchConfig(),proof_nodes=0,max_depth=1,time_limit=15,node_limit=10**8))
                for state in fixtures:
                    side=int(state[:,:,82:84].flat[1])
                    if game.getGameEnded(state,side).any():continue
                    native=rust.inspect(state,**{k:kwargs[k] for k in ('radius','weight','material','advantage','attack','defence')})
                    expected=Evaluator(game,config).score(state,side,unlimited(),proof={'status':'unknown'})
                    self.assertAlmostEqual(native['score'],expected,places=9)
                state=fixtures[0]
                a=rust.analyze(state,**dict(kwargs,reuse=True))
                b=rust.analyze(state,**kwargs)
                self.assertTrue(a['complete']);self.assertEqual((a['score'],a['action']),(b['score'],b['action']))
