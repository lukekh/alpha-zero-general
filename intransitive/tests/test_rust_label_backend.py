"""Production-backend migration, durable label paths and native worker cleanup."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile
import time
import unittest
from unittest.mock import patch

from intransitive.rust_teacher import BINARY, RustTeacher
from intransitive.rust_teacher.adapter import RustAlphaBetaPlayer, validate_backend
from intransitive import supervised_minimax as sm
from intransitive.expand_minimax_dataset import upgrade_backend
from intransitive.greedy_process import GameProcesses, make_net, SETTINGS


def backend(directory=None):
    row=dict(implementation='rust-v1',binary=str(BINARY),sha256=hashlib.sha256(BINARY.read_bytes()).hexdigest(),
             node_limit_unit='node_visits')
    if directory:
        row['worker_status_dir']=str(directory)
    return row


def long_native_job(task):
    _,_,settings,_,_,_=task
    game=sm.IntransitiveGame()
    player=RustAlphaBetaPlayer(game,sm.SearchConfig(max_depth=32,time_limit=300.,node_limit=10**12),settings['teacher_backend'])
    try:
        return player.analyze(game.getInitBoard())
    finally:
        player.close()


@unittest.skipUnless(BINARY.exists(),'Build native binary first')
class NativeBackendTests(unittest.TestCase):
    def test_migration_only_changes_backend(self):
        old=dict(primary='/primary',target_positions=1000000,
            settings=dict(teacher=dict(sm.TEACHER,max_depth=6),dataset_seed=1))
        new=deepcopy(old);new['settings']['teacher_backend']=backend()
        row=upgrade_backend(old,new)
        self.assertTrue(row['backend_only'])
        for key,value in (('max_depth',5),('pressure_enabled',True),('time_limit',60.)):
            bad=deepcopy(new);bad['settings']['teacher'][key]=value
            with self.assertRaises(ValueError):
                upgrade_backend(old,bad)
        bad=deepcopy(new);bad['target_positions']=100
        with self.assertRaises(ValueError):
            upgrade_backend(old,bad)
        with self.assertRaises(ValueError):
            validate_backend(sm.SearchConfig(count_weight=101),backend())

    def test_outboxes_have_separate_backend_namespaces(self):
        settings=dict(output='/tmp/unused-native-test',teacher=sm.TEACHER)
        self.assertNotEqual(sm.label_result_path(settings,1),
            sm.label_result_path(dict(settings,teacher_backend=backend()),1))

    def test_reuse_keeps_scores_and_separates_configs(self):
        state,_=sm.generate_position(123,'opening')
        with RustTeacher() as native:
            first=native.analyze(state,depth=5,weight=0.,reuse=True)
            again=native.analyze(state,depth=5,weight=0.,reuse=True)
            self.assertEqual(first['score'],again['score'])
            self.assertEqual(first['action'],again['action'])
            self.assertGreater(again['tt_hits'],0)
            self.assertLess(again['nodes'],first['nodes'])
            game=sm.IntransitiveGame()
            child,_=game.getNextState(state,0,first['action'])
            reused=native.analyze(child,depth=5,weight=0.,reuse=True)
            fresh=native.analyze(child,depth=5,weight=0.)
            self.assertAlmostEqual(reused['score'],fresh['score'],places=8)
            self.assertGreater(reused['tt_hits'],0)
            pressured=native.analyze(child,depth=5,weight=10.,reuse=True)
            fresh_pressure=native.analyze(child,depth=5,weight=10.)
            self.assertAlmostEqual(pressured['score'],fresh_pressure['score'],places=8)

    def test_entire_label_family_uses_native_and_is_recoverable(self):
        with tempfile.TemporaryDirectory() as folder:
            settings=dict(sm.SETTINGS,output=folder,dataset_seed=123,
                teacher=dict(sm.TEACHER,time_limit=60.),teacher_backend=backend(folder),
                persist_label_results=True,promoted_roots=2,ablations=True)
            # Any accidental Python fallback in root, descendants or ablations fails.
            with patch.object(sm.AlphaBetaPlayer,'analyze',side_effect=AssertionError('Python fallback')):
                rows,row=sm.label_job((0,'unused',settings,123,0,True))
                recovered,recovery=sm.label_job((0,'unused',settings,123,0,True))
            self.assertTrue(rows)
            self.assertIsNone(row['rejected'])
            self.assertGreater(row['ablation_searches'],0)
            self.assertTrue(any(r['provenance'].get('source')=='promoted_tree_root' for r in rows))
            for r in rows:
                if r['provenance'].get('source')=='promoted_tree_root':
                    self.assertGreater(r['tt_entries_available'],0)
                    self.assertGreater(r['tt_hits'],0)
            self.assertEqual([r['orbit'] for r in rows],[r['orbit'] for r in recovered])
            self.assertTrue(recovery['recovered_from_outbox'])
            for r in rows:
                self.assertTrue(sm.valid_teacher_record(r))
                self.assertEqual(r['teacher_backend'],settings['teacher_backend'])
                self.assertEqual(r['teacher_work_unit'],'node_visits')
            registry=json.loads(next(Path(folder).glob('rust-worker-*.json')).read_text())
            self.assertFalse(registry['active'])

    def test_pool_reaps_native_even_after_python_worker_is_killed(self):
        with tempfile.TemporaryDirectory() as folder:
            settings=dict(SETTINGS,teacher_backend=backend(folder))
            net=make_net(settings);net.save_checkpoint(folder,'seed.pt')
            pool=GameProcesses(1,snapshot=Path(folder)/'seed.pt',settings=settings,_task=long_native_job)
            worker=pool.initial_workers[0]
            native_pid=None
            try:
                pool.connections[0].send((0,str(Path(folder)/'seed.pt'),settings,1,0,True))
                registry=Path(folder)/f'rust-worker-{worker.pid}.json'
                deadline=time.monotonic()+30
                while not registry.exists():
                    self.assertLess(time.monotonic(),deadline)
                    time.sleep(.05)
                native_pid=json.loads(registry.read_text())['native_pid']
                os.kill(worker.pid,signal.SIGKILL)
                worker.join(timeout=5)
            finally:
                pool.close()
            self.assertFalse(worker.is_alive())
            if native_pid:
                import subprocess
                state=subprocess.run(['ps','-p',str(native_pid),'-o','state='],capture_output=True,text=True).stdout.strip()
                self.assertTrue(not state or state.startswith('Z'))


if __name__=='__main__':
    unittest.main()
