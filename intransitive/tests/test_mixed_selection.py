"""Mixed checkpoint selection and transactional baseline replacement."""
import copy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from intransitive.greedy_process import SETTINGS, game_tasks, selection, make_net, episode, GameProcesses
from intransitive.greedy_training import initial_bundle, rebase_selection, validate, evaluate_checkpoint
from intransitive.rust_teacher import BINARY


def settings():
    return dict(SETTINGS,opponent='FlybrainPlayer',evaluation_games=40,
        evaluation_opponents=[dict(games=30,opponent='FlybrainPlayer'),
            dict(games=10,opponent='AlphaBetaPlayer',opponent_search=dict(max_depth=6))])


class MixedSelectionTests(unittest.TestCase):
    def test_minimax_evaluation_simulations_do_not_leak_to_training_or_fly(self):
        cfg=settings()
        cfg.update(opponent='AlphaBetaPlayer',opponent_search=dict(max_depth=6))
        cfg['evaluation_opponents'][1]['numMCTSSims']=512
        tasks=game_tasks('unused',range(40),False,cfg)
        self.assertEqual([t[2]['numMCTSSims'] for t in tasks],[32]*30+[512]*10)
        training=game_tasks('unused',range(8),True,cfg)
        self.assertTrue(all(t[2]['numMCTSSims']==32 for t in training))
        self.assertTrue(all(t[2]['opponent_search']==dict(max_depth=6) for t in training))
        for bad in (0,1,-1,True,512.,'512'):
            cfg['evaluation_opponents'][1]['numMCTSSims']=bad
            with self.assertRaises(ValueError):
                game_tasks('unused',range(40),False,cfg)

    def test_minimax_training_selection_rebase_preserves_replay_and_optimizer(self):
        with tempfile.TemporaryDirectory() as folder:
            old=settings()
            old.update(opponent='AlphaBetaPlayer',opponent_search=dict(max_depth=6),
                       opponent_backend=dict(implementation='test-backend'))
            net=make_net(old)
            bundle=initial_bundle(net,old,dict(score=.75),time.time()-100,time.time()+600,folder)
            new=copy.deepcopy(old)
            new['evaluation_opponents'][1]['numMCTSSims']=512
            class Pool:
                def collect(self,snapshot,seeds,training,deadline):
                    tasks=game_tasks(snapshot,seeds,training,new)
                    self_test.assertEqual([t[2]['numMCTSSims'] for t in tasks],[32]*30+[512]*10)
                    return [([],dict(opponent=t[2]['opponent'],outcome='win' if i<35 else 'loss'))
                            for i,t in enumerate(tasks)],{}
            self_test=self
            migrated,report=rebase_selection(bundle,new,Pool(),folder)
            validate(migrated)
            self.assertEqual(migrated['best_score'],35/40)
            self.assertEqual(bundle['best_score'],.75)
            self.assertIs(migrated['current'],bundle['current'])
            self.assertIs(migrated['best'],bundle['best'])
            for key in ('replay','replay_sha256','optimizer_updates','started_epoch','deadline_epoch'):
                self.assertEqual(migrated[key],bundle[key])
            for changed in (dict(numMCTSSims=512),dict(opponent_search=dict(max_depth=3)),
                            dict(opponent_backend=dict(implementation='other'))):
                with self.assertRaisesRegex(ValueError,'training settings'):
                    rebase_selection(bundle,dict(new,**changed),Pool(),folder)

    def test_gated_pool_rejects_accidental_full_suite_dispatch(self):
        pool=object.__new__(GameProcesses)
        pool.pool=True
        pool.settings=dict(settings(),evaluation_gate='no_flybrain_losses')
        with self.assertRaisesRegex(ValueError,'separately'):
            pool.collect('unused',range(800000,800040),False)

    def test_gate_waits_for_all_fly_games_and_never_dispatches_minimax_after_loss(self):
        cfg=dict(settings(),evaluation_gate='no_flybrain_losses')
        for outcome, expected in (('loss',[0]),('model_draw',[0,1]),('win',[0,1])):
            with self.subTest(outcome=outcome):
                calls=[]
                class Pool:
                    def collect(self,snapshot,seeds,training,deadline,evaluation_stage):
                        calls.append(evaluation_stage)
                        self_test.assertEqual(calls,expected[:len(calls)])
                        tasks=game_tasks(snapshot,seeds,training,cfg,evaluation_stage=evaluation_stage)
                        return [([],dict(index=i,seed=t[3],model_side=t[4],opponent=t[2]['opponent'],
                            outcome=(outcome if i==29 else 'win') if evaluation_stage==0 else 'loss'))
                            for i,t in enumerate(tasks)],dict(seconds=.1)
                self_test=self
                rows,timing,score=evaluate_checkpoint(Pool(),'unused',cfg,deadline=time.time()+30)
                self.assertEqual(calls,expected)
                self.assertEqual(len(rows),30 if outcome=='loss' else 40)
                self.assertEqual(score['minimax_qualified'],outcome!='loss')
                self.assertEqual(score['score'],(29 if outcome=='loss' else 29.5 if outcome=='model_draw' else 30)/40)
                self.assertEqual(score['losses'],1 if outcome=='loss' else 10)
                self.assertEqual([r['index'] for _,r in rows],list(range(len(rows))))
                self.assertEqual([r['seed'] for _,r in rows],list(range(800000,800000+len(rows))))
                if outcome=='loss':
                    self.assertIsNone(score['by_opponent']['AlphaBetaPlayer'])
                    self.assertEqual(score['minimax_skipped'],10)

    def test_gate_scores_remain_comparable_and_do_not_create_false_perfect_score(self):
        # Losing to all Minimax games after 30 fly wins outranks a skipped
        # Minimax stage after 29 fly wins and one loss (75% versus 72.5%).
        cfg=dict(settings(),evaluation_gate='no_flybrain_losses')
        class Pool:
            def collect(self,snapshot,seeds,training,deadline,evaluation_stage):
                return [([],dict(outcome='win' if evaluation_stage==0 else 'loss',
                    opponent='FlybrainPlayer' if evaluation_stage==0 else 'AlphaBetaPlayer'))
                    for _ in seeds],{}
        _,_,score=evaluate_checkpoint(Pool(),'unused',cfg,deadline=time.time()+30)
        self.assertEqual(score['score'],.75)
        self.assertGreater(score['score'],29/40)
        self.assertLess(score['score'],1.)

    def test_schedule_is_balanced_ordered_and_evaluation_only(self):
        cfg=settings()
        tasks=game_tasks('unused',range(800000,800040),False,cfg)
        self.assertEqual([t[2]['opponent'] for t in tasks],['FlybrainPlayer']*30+['AlphaBetaPlayer']*10)
        self.assertEqual([t[4] for t in tasks[:30]].count(0),15)
        self.assertEqual([t[4] for t in tasks[30:]].count(0),5)
        self.assertEqual([t[3] for t in tasks],list(range(800000,800040)))
        training=game_tasks('unused',range(8),True,cfg)
        self.assertTrue(all(t[2]['opponent']=='FlybrainPlayer' for t in training))
        self.assertTrue(all('opponent_search' not in t[2] for t in training))

    def test_invalid_suite_rejected(self):
        for blocks in ([dict(games=39,opponent='FlybrainPlayer')],
                       [dict(games=40,opponent='bad')],
                       [dict(games=40,opponent='FlybrainPlayer',numMCTSSims=1)]):
            with self.assertRaises(ValueError):
                game_tasks('unused',range(40),False,dict(settings(),evaluation_opponents=blocks))
        with self.assertRaises(ValueError):
            game_tasks('unused',range(20),False,settings())

    def test_combined_score_and_per_opponent_breakdown(self):
        rows=[dict(opponent='FlybrainPlayer',outcome='win') for _ in range(30)]
        rows += [dict(opponent='AlphaBetaPlayer',outcome='loss') for _ in range(9)]
        rows.append(dict(opponent='AlphaBetaPlayer',outcome='model_draw'))
        score=selection(rows)
        self.assertEqual(score['score'],30.5/40)
        self.assertEqual(score['by_opponent']['FlybrainPlayer']['games'],30)
        self.assertEqual(score['by_opponent']['AlphaBetaPlayer']['score'],.05)

    def test_rebase_replaces_incomparable_score_preserves_training_and_can_promote(self):
        with tempfile.TemporaryDirectory() as folder:
            old=dict(SETTINGS,opponent='FlybrainPlayer')
            net=make_net(old)
            bundle=initial_bundle(net,old,dict(score=1.),time.time()-100,time.time()+600,folder)
            # Simulate a complete unscheduled iteration with a previously evaluated incumbent.
            bundle.update(iteration=4,replay_iteration=4,replay=[[],[]],selection_iteration=1)
            from intransitive.greedy_training import replay_digest
            bundle['replay_sha256']=replay_digest(bundle['replay'])
            class Pool:
                calls=0
                def collect(self,snapshot,seeds,training,deadline):
                    self.calls+=1
                    rows=[]
                    for i,t in enumerate(game_tasks(snapshot,seeds,training,settings())):
                        rows.append(([],dict(opponent=t[2]['opponent'],outcome=
                            'win' if i < (20 if self.calls==1 else 25) else 'loss')))
                    return rows,{}
            before=copy.deepcopy(bundle)
            migrated,report=rebase_selection(bundle,settings(),Pool(),folder)
            validate(migrated)
            self.assertEqual(report['scores']['best']['score'],.5)
            self.assertEqual(migrated['best_score'],.625)
            self.assertEqual(migrated['best_iteration'],4)
            self.assertEqual(migrated['selection_iteration'],4)
            self.assertEqual(bundle['best_score'],1.)
            for key in ('deadline_epoch','started_epoch','optimizer_updates','iteration','replay_sha256','replay'):
                self.assertEqual(migrated[key],before[key])
            for key,value in before['current']['state_dict'].items():
                self.assertTrue(torch.equal(value,migrated['current']['state_dict'][key]))
            with self.assertRaises(ValueError):
                rebase_selection(bundle,dict(settings(),learn_rate=.9),Pool(),folder)
            with self.assertRaises(KeyboardInterrupt):
                rebase_selection(bundle,settings(),Pool(),folder,
                                 lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
            self.assertEqual(bundle['settings'],old)

    @unittest.skipUnless(BINARY.exists(),'Build native binary first')
    def test_native_opponent_episode_closes_process_and_reports_depth(self):
        from intransitive.tests.test_rust_label_backend import backend
        from intransitive.rust_teacher.adapter import RustAlphaBetaPlayer
        cfg=dict(SETTINGS,numMCTSSims=2,opponent='AlphaBetaPlayer',
            opponent_backend=backend(),opponent_search=dict(max_depth=1,time_limit=30.))
        net=make_net(cfg)
        with patch.object(RustAlphaBetaPlayer,'close',autospec=True,side_effect=RustAlphaBetaPlayer.close) as close:
            _,row=episode(net,314,0,False,cfg)
        self.assertTrue(close.called)
        self.assertEqual(row['opponent'],'AlphaBetaPlayer')
        self.assertTrue(row['opponent_searches'])
        self.assertTrue(all(r['completed_depth']>=1 or r['stop_reason']=='proven_result'
                            for r in row['opponent_searches']))
        self.assertIsNotNone(close.call_args.args[0].client.process.poll())


if __name__=='__main__':
    unittest.main()
