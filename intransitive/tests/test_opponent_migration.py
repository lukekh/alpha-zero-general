"""Opponent-only RL continuation with fresh replay and unchanged selection."""
import copy
import pickle
from pathlib import Path
import tempfile
import time
import unittest
import zlib
from unittest.mock import patch

import torch
from intransitive.greedy_process import SETTINGS, make_net, game_tasks, episode
from intransitive.greedy_training import (initial_bundle, iteration, migrate_training_opponent,
    publish, load_bundle, validate)
from intransitive.rust_teacher import BINARY
from intransitive.tests.test_rust_label_backend import backend
from intransitive.tests.test_greedy_process import StubGames


@unittest.skipUnless(BINARY.exists(),'Build the Rust teacher first')
class OpponentMigrationTests(unittest.TestCase):
    def test_native_training_episode_emits_mcts_policy_and_actual_outcome_targets(self):
        from intransitive.heuristics import AlphaBetaPlayer
        cfg=dict(SETTINGS,opponent='AlphaBetaPlayer',opponent_backend=backend(),
                 opponent_search=dict(max_depth=1,time_limit=30.),numMCTSSims=2)
        net=make_net(cfg)
        with patch.object(AlphaBetaPlayer,'play',side_effect=AssertionError('Python fallback')):
            examples,row=episode(net,318,0,True,cfg)
        self.assertTrue(examples)
        self.assertEqual(len(examples)%12,0)
        self.assertEqual(row['opponent'],'AlphaBetaPlayer')
        self.assertTrue(row['opponent_searches'])
        expected={'win':1.,'loss':-1.,'model_draw':1e-4}[row['outcome']]
        for blob in examples:
            _,policy,value,legal,q=pickle.loads(zlib.decompress(blob))
            self.assertAlmostEqual(float(value[0]),expected,places=5)
            self.assertAlmostEqual(float(policy.sum()),1.,places=5)
            self.assertFalse(policy[~legal].any())
            self.assertFalse(q.any())

    def test_preserves_current_best_optimizer_scores_and_deadline_then_rebuilds_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            old=dict(SETTINGS,opponent='FlybrainPlayer',batch_size=8,opponent_backend=backend(),
                     evaluation_opponents=[dict(games=20,opponent='FlybrainPlayer')])
            net=make_net(old)
            bundle=initial_bundle(net,old,dict(score=.5),time.time()-100,time.time()+600,folder)
            pool=StubGames(old)
            bundle,_=iteration(bundle,pool,folder)
            bundle,_=iteration(bundle,pool,folder)
            before=copy.deepcopy(bundle)
            cfg=dict(old,opponent='AlphaBetaPlayer',opponent_search=dict(max_depth=6,time_limit=300.),
                     value_targets='actual model-vs-minimax rollout outcome; no search-Q mixing')
            moved=migrate_training_opponent(bundle,cfg)
            for key in ('iteration','optimizer_updates','best_iteration','best_score','selection',
                        'selection_iteration','deadline_epoch','started_epoch'):
                self.assertEqual(moved[key],before[key])
            self.assertEqual(moved['replay'],[])
            self.assertEqual(moved['replay_started_at'],2)
            self.assertEqual(bundle['replay_sha256'],before['replay_sha256'])
            for key in ('current','best'):
                for name,value in before[key]['state_dict'].items():
                    self.assertTrue(torch.equal(value,moved[key]['state_dict'][name]))
            for index,state in before['current']['optim_state']['state'].items():
                for name,value in state.items():
                    self.assertTrue(torch.equal(value,moved['current']['optim_state']['state'][index][name]))
            path=Path(folder)/'migrated.pt'
            publish(path,moved)
            resumed=load_bundle(path)
            first,_=iteration(resumed,pool,folder)
            validate(first)
            self.assertEqual(len(first['replay']),1)
            self.assertGreater(first['optimizer_updates'],bundle['optimizer_updates'])
            second,_=iteration(first,pool,folder)
            validate(second)
            self.assertEqual(len(second['replay']),2)
            for setting,value in (('learn_rate',.1),('evaluation_games',40),('evaluation_gate','new')):
                with self.assertRaises(ValueError):
                    migrate_training_opponent(bundle,dict(cfg,**{setting:value}))
            bad=dict(moved,replay_started_at=3)
            with self.assertRaises(ValueError):
                validate(bad)

    def test_rollouts_use_minimax_but_fly_stage_never_inherits_its_search_config(self):
        cfg=dict(SETTINGS,opponent='AlphaBetaPlayer',opponent_search=dict(max_depth=6),
            evaluation_games=40,evaluation_gate='no_flybrain_losses',
            evaluation_opponents=[dict(games=30,opponent='FlybrainPlayer'),
                dict(games=10,opponent='AlphaBetaPlayer',opponent_search=dict(max_depth=6))])
        training=game_tasks('unused',range(8),True,cfg)
        self.assertTrue(all(t[2]['opponent']=='AlphaBetaPlayer' and t[2]['opponent_search']['max_depth']==6 for t in training))
        self.assertEqual([t[4] for t in training],[0,1]*4)
        fly=game_tasks('unused',range(800000,800030),False,cfg,evaluation_stage=0)
        self.assertTrue(all(t[2]['opponent']=='FlybrainPlayer' and 'opponent_search' not in t[2] for t in fly))
        minimax=game_tasks('unused',range(800030,800040),False,cfg,evaluation_stage=1)
        self.assertTrue(all(t[2]['opponent']=='AlphaBetaPlayer' for t in minimax))


if __name__=='__main__':
    unittest.main()
