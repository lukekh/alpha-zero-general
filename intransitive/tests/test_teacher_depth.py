from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from intransitive import supervised_minimax as sm
from intransitive.expand_minimax_dataset import Catalog, upgrade_config


class TeacherDepthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state, cls.provenance = sm.generate_position(123, 'opening')
        cls.legal = sm.IntransitiveGame().getValidMoves(cls.state, 0)
        cls.action = int(np.flatnonzero(cls.legal)[0])

    def result(self, depth, reason='maximum_depth'):
        return SimpleNamespace(completed_depth=depth, stop_reason=reason,
            action=self.action, elapsed=.01, work=1, score=0.)

    def test_requested_eight_rejects_partial_depth_five_through_seven(self):
        for depth in (5, 6, 7):
            self.assertFalse(sm.completed_teacher(self.result(depth), 8))
        self.assertTrue(sm.completed_teacher(self.result(8), 8))
        self.assertTrue(sm.completed_teacher(self.result(1, 'proven_result'), 8))

    def test_label_worker_uses_requested_depth_and_tags_accepted_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings=dict(sm.SETTINGS,output=temporary,dataset_seed=123,
                          teacher=dict(sm.TEACHER,max_depth=8),ablations=False)
            with patch.object(sm,'generate_position',return_value=(self.state,self.provenance)), \
                 patch.object(sm,'promote_tree_roots',return_value=[]):
                for depth,reason,accepted in ((5,'maximum_depth',False),(7,'time_limit',False),
                                              (8,'maximum_depth',True),(1,'proven_result',True)):
                    with patch.object(sm.AlphaBetaPlayer,'analyze',return_value=self.result(depth,reason)):
                        rows,_=sm.label_job((0,'unused',settings,123,0,True))
                    self.assertEqual(bool(rows),accepted)
                    if accepted:
                        self.assertEqual(rows[0]['teacher_target_depth'],8)

    def test_catalog_accepts_legacy_five_but_rejects_incomplete_new_eight(self):
        row=dict(orbit='a',family='a',split='train',provenance=dict(stage='opening'),
                 teacher_depth=5,teacher_reason='maximum_depth',legal=[True],action=0)
        with tempfile.TemporaryDirectory() as temporary:
            catalog=Catalog(Path(temporary)/'catalog.sqlite3',300)
            with catalog.db:
                self.assertEqual(catalog.insert([row],'extra'),1)
                with self.assertRaises(ValueError):
                    catalog.insert([dict(row,orbit='b',family='b',teacher_target_depth=8)],'extra')
                self.assertEqual(catalog.insert([dict(row,orbit='c',family='c',teacher_depth=8,
                                                     teacher_target_depth=8)],'extra'),1)
            catalog.close()

    def test_promoted_descendants_must_also_reach_eight(self):
        game=sm.IntransitiveGame()
        child,side=game.getNextState(self.state,0,self.action)
        action=int(np.flatnonzero(game.getValidMoves(child,side))[0])
        parent=dict(provenance=self.provenance,split='train',family='a',orbit='a')
        teacher=SimpleNamespace(game=game,config=sm.SearchConfig(max_depth=8),table={})
        for depth in (7,8):
            result=self.result(depth)
            result.action=action
            result.tt_hits=0
            teacher.analyze=lambda state: result
            rows=sm.promote_tree_roots(teacher,self.state,self.result(8),parent,limit=1)
            self.assertEqual(len(rows),int(depth==8))
            if rows:
                self.assertEqual(rows[0]['teacher_target_depth'],8)

    def test_verified_ablations_must_also_reach_eight(self):
        parent=dict(state=self.state,legal=self.legal,action=self.action,orbit='a',
                    family='a',split='train',provenance=self.provenance)
        with patch.object(sm,'perturb_position',return_value=(self.state,dict(operation='fixture'))), \
             patch.object(sm.Evaluator,'score',return_value=0.):
            for depth in (7,8):
                with patch.object(sm.AlphaBetaPlayer,'analyze',return_value=self.result(depth)):
                    rows,_=sm.verified_ablations(parent,self.result(8),sm.SearchConfig(max_depth=8),max_searches=1)
                self.assertEqual(len(rows),int(depth==8))
                if rows:
                    self.assertEqual(rows[0]['teacher_target_depth'],8)

    def test_upgrade_does_not_allow_dataset_identity_or_evaluator_changes(self):
        old=dict(primary='/primary',target_positions=1000000,
                 settings=dict(dataset_seed=123,teacher=dict(sm.TEACHER)))
        new=deepcopy(old)
        new['settings']['teacher'].update(max_depth=8,time_limit=1800.)
        self.assertFalse(upgrade_config(old,new)['existing_records_relabelled'])
        for changed in (dict(new,target_positions=10), old):
            with self.assertRaises(ValueError):
                upgrade_config(old,changed)
        changed=deepcopy(new)
        changed['settings']['teacher']['count_weight']=999
        with self.assertRaises(ValueError):
            upgrade_config(old,changed)

    def test_search_only_migration_preserves_depth_limits_and_evaluation(self):
        old=dict(primary='/primary',target_positions=1000000,
                 settings=dict(dataset_seed=123,teacher=dict(sm.TEACHER,max_depth=6,time_limit=300.)))
        new=deepcopy(old)
        new['settings']['teacher'].update(pvs_enabled=True,aspiration_enabled=True,
            ordering_enabled=True,compiled_ordering_enabled=True,depth_replacement_enabled=True,table_entries=50000)
        history=upgrade_config(old,new,allow_search_optimization=True)
        self.assertTrue(history['search_optimization_only'])
        self.assertFalse(history['existing_records_relabelled'])
        with self.assertRaises(ValueError):
            upgrade_config(old,new)
        for key,value in (('max_depth',5),('time_limit',100.),('node_limit',12),
                          ('pressure_enabled',True),('count_weight',999),('proof_depth',3)):
            changed=deepcopy(new)
            changed['settings']['teacher'][key]=value
            with self.assertRaises(ValueError,msg=key):
                upgrade_config(old,changed,allow_search_optimization=True)


if __name__=='__main__':
    unittest.main()
