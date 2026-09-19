"""Completion-order delivery, durable worker results and pending-seed recovery."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from intransitive import supervised_minimax as sm
from intransitive import expand_minimax_dataset as extra
from intransitive.greedy_process import GameProcesses
from intransitive.tests.test_expand_minimax_dataset import record


class Connection:
    def __init__(self, held=False):
        self.held, self.sent = held, []
        self.task = None

    def send(self, task):
        self.sent.append(task)
        self.task = task

    def poll(self):
        return self.task is not None and not self.held

    def recv(self):
        task, self.task = self.task, None
        return 'result', ([], dict(index=task[0], seed=task[3]))


class StreamedLabelTests(unittest.TestCase):
    def pool(self):
        pool = object.__new__(GameProcesses)
        pool.pool, pool.settings = True, dict(sm.SETTINGS)
        pool.connections = [Connection(held=True), Connection()]
        pool._check_workers = lambda: None
        pool.close = lambda: setattr(pool,'pool',None)
        return pool

    def test_fast_worker_saved_and_replenished_before_slow_worker_finishes(self):
        pool = self.pool()
        stream = pool.iter_results('unused', [10,11,12], True)
        (rows, first), _ = next(stream)
        self.assertEqual(first['seed'],11)
        self.assertEqual([task[3] for task in pool.connections[1].sent],[11,12])
        self.assertEqual(next(stream)[0][1]['seed'],12)
        pool.connections[0].held = False
        self.assertEqual(next(stream)[0][1]['seed'],10)
        self.assertIsNone(next(stream,None))

    def test_cancel_stream_closes_workers_with_pending_work(self):
        pool = self.pool()
        stream = pool.iter_results('unused',[10,11],True)
        next(stream)
        stream.close()
        self.assertIsNone(pool.pool)

    def test_worker_result_is_durable_and_recovered_without_research(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = dict(sm.SETTINGS, output=temporary, dataset_seed=123,
                            teacher=dict(sm.TEACHER,max_depth=6), persist_label_results=True)
            # Rejections are durable too, so an interrupted coordinator does not
            # repeat an expensive search which already hit its work limit.
            with patch.object(sm,'generate_position',side_effect=RuntimeError('fixture rejection')):
                original = sm.label_job((0,'unused',settings,123,0,True))
            path = sm.label_result_path(settings,123)
            self.assertTrue(path.exists())
            with patch.object(sm,'generate_position',side_effect=AssertionError('must not research')):
                rows, recovered = sm.label_job((7,'unused',settings,123,0,True))
            self.assertEqual(rows,original[0])
            self.assertEqual(recovered['index'],7)
            self.assertTrue(recovered['recovered_from_outbox'])
            changed = dict(settings,teacher=dict(settings['teacher'],max_depth=8))
            self.assertNotEqual(path,sm.label_result_path(changed,123))
            sm.acknowledge_label_result(settings,123)
            self.assertFalse(path.exists())

    def test_explicit_downgrade_preserves_identity_and_evaluator(self):
        old = dict(primary='/primary',target_positions=1000000,
                   settings=dict(dataset_seed=123,teacher=dict(sm.TEACHER,max_depth=8)))
        new = deepcopy(old)
        new['settings'].update(persist_label_results=True)
        new['settings']['teacher'].update(max_depth=6,time_limit=300.)
        with self.assertRaises(ValueError):
            extra.upgrade_config(old,new)
        history = extra.upgrade_config(old,new,allow_depth_change=True)
        self.assertFalse(history['existing_records_relabelled'])
        changed = deepcopy(new)
        changed['settings']['teacher']['count_weight'] = 999
        with self.assertRaises(ValueError):
            extra.upgrade_config(old,changed,allow_depth_change=True)
        with self.assertRaises(ValueError):
            extra.upgrade_config(old,dict(new,target_positions=10),allow_depth_change=True)

    def test_extra_commits_one_worker_and_resumes_only_unfinished_seeds(self):
        calls = []
        worker_counts = []
        class Pool:
            pids = []
            def __init__(self,*args,**kwargs):
                worker_counts.append(args[0])
            def iter_results(self,snapshot,seeds,training,**kwargs):
                calls.append(list(seeds))
                if len(calls)==1:
                    yield ([record('new',depth=6)],dict(seed=seeds[1])), dict(seconds=.01)
                raise KeyboardInterrupt
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as temporary:
            output, primary = Path(temporary)/'extra', Path(temporary)/'primary'
            primary.mkdir()
            with patch.object(extra,'GameProcesses',Pool), patch.object(extra,'primary_active',return_value=True):
                extra.run(output,primary,300,teacher_config=dict(sm.TEACHER,max_depth=6),
                          generation_workers=6)
                with extra.sqlite3.connect(output/'corpus.sqlite3') as db:
                    self.assertEqual(db.execute('SELECT count(*) FROM positions').fetchone()[0],1)
                    metadata = {k:json.loads(v) for k,v in db.execute('SELECT key,value FROM metadata')}
                self.assertEqual(len(metadata['pending_seeds']),23)
                self.assertNotIn(calls[0][1],metadata['pending_seeds'])
                self.assertEqual(metadata['next_seed'],calls[0][-1]+1)
                extra.run(output,primary,300,teacher_config=dict(sm.TEACHER,max_depth=6),
                          generation_workers=8)
                self.assertEqual(calls[1],metadata['pending_seeds'])
                self.assertEqual(worker_counts,[6,8])

    def test_trainer_commits_one_worker_and_recovers_remaining_queue(self):
        calls = []
        class Pool:
            pids = []
            def __init__(self,*args,**kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self,*args):
                pass
            def iter_results(self,snapshot,seeds,training,**kwargs):
                calls.append(list(seeds))
                if len(calls)==1:
                    yield ([record('new',depth=6)],dict(seed=seeds[1],rejected=None)),dict(seconds=.01)
                raise KeyboardInterrupt
        net = MagicMock()
        net.load_checkpoint.return_value = {}
        net.optimizer_updates = 0
        net.args = {}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output/'dataset').mkdir()
            with patch.object(sm,'GameProcesses',Pool), patch.object(sm,'make_net',return_value=net), \
                 patch.object(sm,'save_model'):
                sm.run(output,output/'unused.pt',sm.time.time()+300,300,teacher_config=dict(sm.TEACHER,max_depth=6))
                manifest = json.loads((output/'dataset-manifest.json').read_text())
                self.assertEqual(manifest['unique_positions'],1)
                self.assertEqual(len(manifest['pending_seeds']),15)
                self.assertNotIn(calls[0][1],manifest['pending_seeds'])
                self.assertEqual(manifest['next_seed'],calls[0][-1]+1)
                sm.run(output,output/'unused.pt',sm.time.time()+300,300,resume=True,
                       teacher_config=dict(sm.TEACHER,max_depth=6))
                self.assertEqual(calls[1],manifest['pending_seeds'])


if __name__=='__main__':
    unittest.main()
