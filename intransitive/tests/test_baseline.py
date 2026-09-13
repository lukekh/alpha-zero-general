"""Guard evaluation colour accounting, uncertainty, and rejected-model selection."""
import unittest

from intransitive.baseline import model_outcome, select_checkpoint, summarize, wilson


class BaselineAccounting(unittest.TestCase):
    def test_colour_swap_and_nonzero_draw_reward(self):
        self.assertEqual(model_outcome(1., 'Blue'), 'wins')
        self.assertEqual(model_outcome(1., 'Red'), 'losses')
        self.assertEqual(model_outcome(-1., 'Blue'), 'losses')
        self.assertEqual(model_outcome(-1., 'Red'), 'wins')
        for colour in ('Blue', 'Red'):
            self.assertEqual(model_outcome(0.0001, colour), 'draws')

    def test_selection_keeps_rejection_visible(self):
        rows = [{'accepted': False}] * 4
        self.assertEqual(select_checkpoint(rows)[0], 'candidate_4.pt')
        self.assertIn('rejected', select_checkpoint(rows)[1])
        rows[1] = {'accepted': True}
        self.assertEqual(select_checkpoint(rows)[0], 'candidate_2.pt')

    def test_draws_remain_in_denominator_and_intervals_are_nonzero(self):
        rows = [dict(outcome=x, reason='no-capture limit', plies=30,
                     trajectory_sha256=str(i)) for i, x in enumerate(['wins']*2 + ['draws']*12 + ['losses']*2)]
        result = summarize(rows)
        self.assertEqual((result['games'], result['wins'], result['draws'], result['losses']), (16, 2, 12, 2))
        self.assertEqual(result['score'], .5)
        self.assertEqual(result['win_probability_95_wilson'], wilson(2, 16))
        self.assertAlmostEqual(wilson(0, 16)[1], .1936076805)
        self.assertLess(result['score_95_hoeffding'][0], .5)
        self.assertGreater(result['score_95_hoeffding'][1], .5)
        self.assertIsNone(summarize([])['score'])


if __name__ == '__main__':
    unittest.main()
