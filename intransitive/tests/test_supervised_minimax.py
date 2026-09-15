import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

from intransitive import supervised_minimax as sm
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveLogicNumba import validate_state


class SupervisedMinimaxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.positions={stage:sm.generate_position(2026091500+i,stage) for i,stage in enumerate(sm.STAGES)}

    def record(self,stage='midgame'):
        state,provenance=self.positions[stage]
        legal=IntransitiveGame().getValidMoves(state,0)
        action=int(np.flatnonzero(legal)[0])
        key=sm.orbit_key(state,action,legal)
        return dict(state=state,legal=legal,action=action,provenance=provenance,
                    orbit=key,split=sm.split_for(key))

    def test_stages_are_reachable_by_recorded_legal_actions(self):
        for stage,(state,provenance) in self.positions.items():
            live=IntransitiveGame(modelling_draws=False)
            board,side=live.getInitBoard(),0
            for action in provenance['actions']:
                self.assertTrue(live.getValidMoves(board,side)[action])
                board,side=live.getNextState(board,side,action)
            observed=IntransitiveGame().getSearchObservation(live.getCanonicalForm(board,side))
            np.testing.assert_array_equal(state,observed)
            self.assertFalse(IntransitiveGame().getGameEnded(state,0).any())
            count=provenance['pieces']
            self.assertTrue(count>=18 if stage=='opening' else 9<=count<=17 if stage=='midgame' else 2<=count<=8)

    def test_all_twelve_symmetries_preserve_legal_targets_and_mask_unknown_values(self):
        record=self.record()
        lazy=sm.SymmetryExamples([record])
        self.assertEqual(len(lazy),12)
        game=IntransitiveGame()
        for i,(state,policy,legal) in enumerate(sm.symmetries(record['state'],record['action'],record['legal'])):
            sample=lazy[i]
            np.testing.assert_array_equal(sample[0],state)
            np.testing.assert_array_equal(sample[1],policy)
            np.testing.assert_array_equal(legal,game.getValidMoves(state,0))
            self.assertEqual(float(policy.sum()),1.)
            self.assertFalse(policy[~legal].any())
            self.assertFalse(sample[5].any() or sample[6].any())
            self.assertEqual(sm.orbit_key(state,int(np.argmax(policy)),legal),record['orbit'])

    def test_piece_perturbations_are_valid_and_do_not_mutate_parent(self):
        record=self.record()
        before=record['state'].copy()
        operations=set()
        for seed in range(100):
            candidate=sm.perturb_position(record['state'],record['action'],np.random.default_rng(seed))
            if candidate is not None:
                state,edit=candidate
                validate_state(state)
                self.assertTrue(IntransitiveGame().getValidMoves(state,0)[record['action']])
                self.assertEqual(int(state[:,:,82:84].flat[4]),1)
                operations.add(edit['operation'])
        np.testing.assert_array_equal(before,record['state'])
        self.assertEqual(operations,{'add_pair','remove_pair','relocate'})

    def test_ablations_require_same_static_score_completed_teacher_move_and_score(self):
        record=self.record()
        candidate=next(sm.perturb_position(record['state'],record['action'],np.random.default_rng(seed))
            for seed in range(100) if sm.perturb_position(record['state'],record['action'],np.random.default_rng(seed)) is not None)
        parent=SimpleNamespace(score=42.)
        base=dict(action=record['action'],score=42.,completed_depth=5,stop_reason='maximum_depth',elapsed=.1,work=10)
        with patch.object(sm,'perturb_position',return_value=candidate), patch.object(sm.Evaluator,'score',return_value=0.):
            for changed in (dict(action=(record['action']+1)%648),dict(score=41.),
                            dict(completed_depth=4,stop_reason='time')):
                with patch.object(sm.AlphaBetaPlayer,'analyze',return_value=SimpleNamespace(**dict(base,**changed))):
                    variants,_=sm.verified_ablations(record,parent,sm.SearchConfig(**sm.TEACHER))
                    self.assertFalse(variants)
            with patch.object(sm.AlphaBetaPlayer,'analyze',return_value=SimpleNamespace(**base)):
                variants,_=sm.verified_ablations(record,parent,sm.SearchConfig(**sm.TEACHER))
                self.assertEqual(len(variants),1)
                self.assertEqual(variants[0]['split'],record['split'])
                self.assertEqual(variants[0]['ablation']['parent_orbit'],record['orbit'])

    def test_policy_only_update_changes_policy_not_frozen_value_head(self):
        settings=dict(sm.SETTINGS,no_compression=True,batch_size=8,batches_per_epoch=1)
        net=sm.make_net(settings)
        net.args['batches_per_epoch']=1
        for p in net.nnet.value_head.parameters():
            p.requires_grad_(False)
        frozen={k:v.detach().clone() for k,v in net.nnet.value_head.state_dict().items()}
        old=net.nnet.policy_head.weight.detach().clone()
        net.train(sm.SymmetryExamples([self.record()]))
        self.assertFalse(torch.equal(old,net.nnet.policy_head.weight))
        for k,v in frozen.items():
            self.assertTrue(torch.equal(v,net.nnet.value_head.state_dict()[k]))
        self.assertEqual(net.optimizer_updates,1)


if __name__=='__main__':
    unittest.main()
