"""Spawn lifecycle, semantic parity and transactional continuation regressions."""
import copy
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import signal
import tempfile
import time
import unittest
from unittest.mock import patch
import zlib

import numpy as np
import torch
from intransitive.greedy_process import (SETTINGS, GameProcesses, _job, episode,
                                        make_net, seed_all)
from intransitive.greedy_training import (checkpoint, initial_bundle, iteration,
                                         load_bundle, publish, replay_digest, capture_legacy)


def delayed_job(task):
    if task[0] == 0:
        time.sleep(.5)
    return _job(task)


class ProcessGames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.settings = dict(SETTINGS, numMCTSSims=2)
        seed_all(48)
        cls.net = make_net(cls.settings)
        cls.snapshot = checkpoint(cls.net, cls.folder.name)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_spawn_matches_inline_actions_targets_and_colour_quota(self):
        expected = [episode(self.net, s, i % 2, True, self.settings)
                    for i, s in enumerate((480,481,482,483))]
        with GameProcesses(2, snapshot=self.snapshot, settings=self.settings, _task=delayed_job) as pool:
            actual, metrics = pool.collect(self.snapshot, (480,481,482,483), True)
            self.assertEqual([r['index'] for _,r in actual], list(range(4)))
            self.assertEqual([r['model_side'] for _,r in actual], [0,1,0,1])
            self.assertEqual(metrics['games'], 4)
            for (eb, er), (ab, ar) in zip(expected, actual):
                self.assertEqual(eb, ab)
                for key in er:
                    self.assertEqual(er[key], ar[key])
                for blob in ab:
                    state, pi, v, mask, q = pickle.loads(zlib.decompress(blob))
                    self.assertFalse(np.any(pi[~mask]))
                    self.assertAlmostEqual(float(pi.sum()),1,places=5)
                    reward={'win':1.,'loss':-1.,'model_draw':1e-4}[ar['outcome']]
                    self.assertAlmostEqual(float(v[0]), reward, places=5)
                    np.testing.assert_array_equal(q, [0,0])
                self.assertGreater(len(ab), 0)
        self.assertTrue(all(not p.is_alive() for p in pool.initial_workers))

    def test_exception_deadline_keyboard_interrupt_and_hard_exit_reap_workers(self):
        for mode in ('exception','deadline','interrupt','hard_exit'):
            with self.subTest(mode=mode):
                pool=GameProcesses(1,snapshot=self.snapshot,settings=self.settings)
                with self.assertRaises((RuntimeError, ValueError, FileNotFoundError, TimeoutError, KeyboardInterrupt)):
                    if mode=='exception':
                        pool.collect(Path(self.folder.name)/'missing.pt', [1,2], True)
                    elif mode=='deadline':
                        pool.collect(self.snapshot,[1,2],True,deadline=time.time()-1)
                    elif mode=='hard_exit':
                        os.kill(pool.pids[0], signal.SIGKILL)
                        pool.collect(self.snapshot,[1,2],True)
                    else:
                        with patch('intransitive.greedy_process.time.sleep', side_effect=KeyboardInterrupt):
                            pool.collect(self.snapshot,[1,2],True)
                self.assertIsNone(pool.pool)
                self.assertTrue(all(not p.is_alive() for p in pool.initial_workers))

    def test_flybrain_spawn_targets_and_value_head_training(self):
        settings = dict(self.settings, opponent='FlybrainPlayer')
        expected = [episode(self.net, s, i % 2, True, settings)
                    for i, s in enumerate((580, 581))]
        with GameProcesses(2, snapshot=self.snapshot, settings=settings) as pool:
            actual, _ = pool.collect(self.snapshot, (580, 581), True)
        for (eb, er), (ab, ar) in zip(expected, actual):
            self.assertEqual(eb, ab)
            self.assertEqual(er['actions'], ar['actions'])
            self.assertEqual(ar['opponent'], 'FlybrainPlayer')
            self.assertTrue(ar['modelling_only'])
            reward = {'win': 1., 'loss': -1., 'model_draw': 1e-4}[ar['outcome']]
            for blob in ab:
                state, pi, value, legal, q = pickle.loads(zlib.decompress(blob))
                self.assertFalse(np.any(pi[~legal]))
                self.assertAlmostEqual(float(value[0]), reward, places=5)
                np.testing.assert_array_equal(q, [0, 0])
        net = make_net(settings)
        net.load_checkpoint(str(self.snapshot.parent), self.snapshot.name)
        self.assertTrue(all(p.requires_grad for p in net.nnet.value_head.parameters()))
        before = [p.detach().clone() for p in net.nnet.value_head.parameters()]
        net.args['batches_per_epoch'] = 1
        net.train([blob for batch, _ in actual for blob in batch])
        self.assertEqual(net.optimizer_updates, 1)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, net.nnet.value_head.parameters())))

    def test_bad_initializer_and_quota(self):
        before={p.pid for p in mp.active_children()}
        with self.assertRaises(TimeoutError):
            GameProcesses(2,snapshot=self.snapshot,settings=self.settings,timeout=.001)
        self.assertEqual({p.pid for p in mp.active_children()},before)
        with self.assertRaises(RuntimeError):
            GameProcesses(2,snapshot=Path(self.folder.name)/'missing.pt',settings=self.settings)
        self.assertEqual({p.pid for p in mp.active_children()},before)
        with GameProcesses(1,snapshot=self.snapshot,settings=self.settings) as pool:
            for seeds in ([],[1],[1,2,3]):
                with self.assertRaises(ValueError):
                    pool.collect(self.snapshot,seeds,False)


class StubGames:
    """Fixed legal replay removes game runtime from interruption boundary tests."""
    def __init__(self, settings):
        game=make_net(settings).game
        board=game.getInitBoard()
        mask=game.getValidMoves(board,0)
        pi=mask.astype(np.float32)/mask.sum()
        self.examples=[zlib.compress(pickle.dumps((s,p,np.array([1.,-1.],np.float32),m,
                       np.zeros(2,np.float32))),level=1) for s,p,m in game.getSymmetries(board,pi,mask)]

    def collect(self,snapshot,seeds,training,deadline=None):
        rows=[(self.examples if training else [],dict(seed=s,model_side=i%2,outcome='loss'))
              for i,s in enumerate(seeds)]
        return rows, dict(games=len(rows),seconds=0,worker_cpu_seconds=0)


class Continuation(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path=Path(self.folder.name)/'state.pt'
        self.settings=dict(SETTINGS, batch_size=8)
        seed_all(48)
        net=make_net(self.settings)
        self.bundle=initial_bundle(net,self.settings,dict(score=.5),100.,time.time()+3600,self.folder.name)
        publish(self.path,self.bundle)
        self.pool=StubGames(self.settings)

    def test_restarts_preserve_adamw_replay_numbering_best_and_deadline(self):
        first,_=iteration(self.bundle,self.pool,self.folder.name)
        publish(self.path,first)
        resumed=load_bundle(self.path)
        uninterrupted,_=iteration(first,self.pool,self.folder.name)
        continued,_=iteration(resumed,self.pool,self.folder.name)
        self.assertEqual(continued['iteration'],2)
        self.assertEqual(continued['replay_sha256'],uninterrupted['replay_sha256'])
        self.assertEqual(continued['deadline_epoch'],self.bundle['deadline_epoch'])
        self.assertEqual(continued['best_iteration'],0)
        self.assertEqual(continued['best_score'],.5)
        self.assertGreater(continued['optimizer_updates'],first['optimizer_updates'])
        for key in uninterrupted['current']['state_dict']:
            self.assertTrue(torch.equal(continued['current']['state_dict'][key],
                                       uninterrupted['current']['state_dict'][key]),key)
        for index,state in uninterrupted['current']['optim_state']['state'].items():
            for key,value in state.items():
                self.assertTrue(torch.equal(continued['current']['optim_state']['state'][index][key],value))

    def test_interruption_boundaries_never_publish_mixed_generations(self):
        old_digest=self.path.read_bytes()
        def fail(stage):
            raise KeyboardInterrupt(stage)
        for stage in ('rollouts','optimization','selection'):
            with self.assertRaises(KeyboardInterrupt):
                iteration(self.bundle,self.pool,self.folder.name,
                          lambda actual: fail(actual) if actual==stage else None)
            self.assertEqual(self.path.read_bytes(),old_digest)
            self.assertEqual(load_bundle(self.path)['iteration'],0)
        new,_=iteration(self.bundle,self.pool,self.folder.name)
        with self.assertRaises(KeyboardInterrupt):
            publish(self.path,new,lambda stage: fail(stage) if stage=='before_replace' else None)
        self.assertEqual(self.path.read_bytes(),old_digest)
        with self.assertRaises(KeyboardInterrupt):
            publish(self.path,new,lambda stage: fail(stage) if stage=='after_replace' else None)
        self.assertEqual(load_bundle(self.path)['iteration'],1)
        self.assertFalse(list(Path(self.folder.name).glob('*.pending')))

    def test_selection_promotes_only_strict_improvement(self):
        self.bundle['settings']=dict(self.settings,evaluation_interval=1)
        collect=self.pool.collect
        def wins(*args,**kwargs):
            rows,metrics=collect(*args,**kwargs)
            for _,row in rows:
                row['outcome']='win'
            return rows,metrics
        with patch.object(self.pool,'collect',side_effect=wins):
            first,_=iteration(self.bundle,self.pool,self.folder.name)
            second,_=iteration(first,self.pool,self.folder.name)
        self.assertEqual(first['best_score'],1.)
        self.assertEqual(first['best_iteration'],1)
        self.assertEqual(second['best_iteration'],1)
        self.assertEqual(second['selection_iteration'],2)
        for key,value in first['best']['state_dict'].items():
            self.assertTrue(torch.equal(second['best']['state_dict'][key],value))

    def test_mid_update_cancellation_cannot_mutate_committed_adamw_or_best(self):
        first,_=iteration(self.bundle,self.pool,self.folder.name)
        # A promoted best can share the current checkpoint in a bundle.
        first['best']=first['current']
        first['best_iteration']=1
        original=copy.deepcopy(first['current']['optim_state'])
        real_step=torch.optim.AdamW.step
        def interrupted(optimizer,*args,**kwargs):
            real_step(optimizer,*args,**kwargs)
            raise KeyboardInterrupt
        with patch.object(torch.optim.AdamW,'step',interrupted):
            with self.assertRaises(KeyboardInterrupt):
                iteration(first,self.pool,self.folder.name)
        for index,state in original['state'].items():
            for key,value in state.items():
                self.assertTrue(torch.equal(first['current']['optim_state']['state'][index][key],value))
                self.assertTrue(torch.equal(first['best']['optim_state']['state'][index][key],value))

    def test_legacy_capture_accepts_only_consistent_rollout_boundaries(self):
        import json
        first,_=iteration(self.bundle,self.pool,self.folder.name)
        source=Path(self.folder.name)/'legacy'
        source.mkdir()
        torch.save(dict(first['current'],run_iteration=1),source/'latest.pt')
        torch.save(dict(first['best'],run_iteration=0,selection=dict(score=.5)),source/'best.pt')
        (source/'replay.pkl').write_bytes(pickle.dumps(first['replay']))
        (source/'settings.json').write_text(json.dumps(self.settings))
        status=dict(phase='opponent_rollouts',iteration=2,completed_iterations=1,
                    started_epoch=first['started_epoch'],deadline_epoch=first['deadline_epoch'])
        (source/'status.json').write_text(json.dumps(status))
        (source/'progress.json').write_text(json.dumps(dict(iteration=1,
            optimizer_updates=first['optimizer_updates'],examples=sum(map(len,first['replay'])))))
        (source/'training-games.jsonl').write_text('\n'.join(json.dumps(dict(iteration=1,
            seed=self.settings['seed']+100+i,model_side=i%2,examples=len(self.pool.examples))) for i in range(8)))
        (source/'evaluations.jsonl').write_text(json.dumps(dict(iteration=1,score=0.,promoted=False)))
        destination=Path(self.folder.name)/'captured.pt'
        captured=capture_legacy(source,destination)
        self.assertEqual(captured['replay_sha256'],first['replay_sha256'])
        self.assertEqual(captured['deadline_epoch'],self.bundle['deadline_epoch'])
        self.assertEqual(load_bundle(destination)['optimizer_updates'],first['optimizer_updates'])
        for phase in ('optimization','selection_evaluation'):
            (source/'status.json').write_text(json.dumps(dict(status,phase=phase)))
            with self.assertRaises(ValueError):
                capture_legacy(source,source/'refused.pt')
        # Even equal-sized replay generations cannot prove consistency once
        # the last rollout status has been written and replay may be replaced.
        (source/'status.json').write_text(json.dumps(dict(status,episodes_completed=8)))
        with self.assertRaisesRegex(ValueError,'last rollout'):
            capture_legacy(source,source/'refused.pt')
        (source/'status.json').write_text(json.dumps(dict(status,episodes_completed=7)))
        read_text=Path.read_text
        observations=[]
        def crossed(path,*args,**kwargs):
            if path==source/'status.json':
                observations.append(path)
                return json.dumps(dict(status,episodes_completed=7 if len(observations)==1 else 8))
            return read_text(path,*args,**kwargs)
        with patch.object(Path,'read_text',crossed):
            with self.assertRaisesRegex(ValueError,'last rollout'):
                capture_legacy(source,source/'refused.pt')
        (source/'status.json').write_text(json.dumps(status))
        (source/'replay.pkl').write_bytes(pickle.dumps([]))
        with self.assertRaises(ValueError):
            capture_legacy(source,source/'refused.pt')

    def test_corruption_and_expired_deadline_are_rejected(self):
        for change in (dict(replay_iteration=1),dict(replay_sha256='bad'),dict(optimizer_updates=20),
                       dict(selection_iteration=1)):
            bad=dict(self.bundle,**change)
            with self.assertRaises(ValueError):
                publish(self.path,bad)
        expired=dict(self.bundle,deadline_epoch=101.)
        with self.assertRaises(TimeoutError):
            iteration(expired,self.pool,self.folder.name)
        self.assertEqual(load_bundle(self.path)['iteration'],0)


if __name__=='__main__':
    unittest.main()
