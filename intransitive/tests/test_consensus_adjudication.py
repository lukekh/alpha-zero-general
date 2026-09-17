"""Consensus cannot reverse official outcomes or promote broken searches."""
from copy import deepcopy
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from intransitive.benchmarks.evolution.adjudicate import adjudicate, consensus
from intransitive.heuristics.budget import BudgetExpired
from intransitive.tests.test_tournament import FakeEngine
from intransitive.tournament.runner import play_match
from intransitive.tournament.spec import candidate, manifest, position, protocol


class ConsensusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = manifest([candidate('A'), candidate('B', {'advantage_weight': 50.})],
            [position([], seed=54, stage='official', pool='search')],
            [protocol('depth', depth=1, max_plies=1, seconds=10.)])
        FakeEngine.action = FakeEngine.cancel = FakeEngine.fail = None
        FakeEngine.stopped = False
        with tempfile.TemporaryDirectory() as folder:
            cls.row = play_match(cls.spec, cls.spec['tasks'][0], Path(folder)/'game.json',
                                 threading.Event(), engine_factory=FakeEngine)
        assert cls.row['status'] == 'unfinished' and len(cls.row['moves']) == 1

    def test_agreement_disagreement_ties_nonfinite(self):
        self.assertEqual(consensus([2., 100.]), 0)
        self.assertEqual(consensus([-2., -100.]), 1)
        for values in ([2., -100.], [0., 2.], [0., 0.], [float('nan'), 2.], [None, -1.]):
            self.assertIsNone(consensus(values))

    def test_common_perspective_and_immutable_journal(self):
        before = deepcopy(self.row)
        with patch('intransitive.benchmarks.evolution.adjudicate.Evaluator.score', side_effect=[-5., -9.]) as score:
            result = adjudicate(self.spec, self.row)
        self.assertEqual(result['adjudicated_winner'], 1)
        self.assertIsNone(result['official_winner'])
        self.assertEqual(self.row, before)
        for call in score.call_args_list:
            self.assertEqual(call.args[1], 0)
            self.assertEqual(call.kwargs['proof'], {'status': 'unknown'})

    def test_evaluation_budget_and_disagreement_abstain(self):
        for scores in ([1., -1.], [BudgetExpired('work'), 1.]):
            with patch('intransitive.benchmarks.evolution.adjudicate.Evaluator.score', side_effect=scores):
                result = adjudicate(self.spec, self.row)
            self.assertEqual(result['outcome'], 'inconclusive')
            self.assertIsNone(result['effective_winner'])

    def test_incomplete_search_excluded(self):
        row = deepcopy(self.row);row['status'] = 'depth_incomplete'
        with patch('intransitive.benchmarks.evolution.adjudicate.Evaluator.score') as score:
            result = adjudicate(self.spec, row)
        score.assert_not_called()
        self.assertEqual(result['outcome'], 'excluded')


if __name__ == '__main__':
    unittest.main()
