import unittest
import numpy as np

from intransitive.defence_examples import TEMPLATES, candidate, refutation, attack_proof, label, planned_refutation
from intransitive.IntransitiveDisplay import parse_move
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveSymmetries import transform_action
from intransitive.supervised_minimax import symmetries


class DefenceExamplesTests(unittest.TestCase):
    def test_counterexamples_cover_all_legal_replies_not_a_cooperative_line(self):
        game = IntransitiveGame(modelling_draws=False)
        for name in TEMPLATES:
            if name == 'strictly_shorter_race':
                continue
            state, d = candidate(13, name, 3)
            proof = refutation(state, d['bad_action'], d['attacker_capture'])
            self.assertIsNotNone(proof, name)
            after, side = game.getNextState(state, 0, proof['prefix'][0])
            after, side = game.getNextState(after, side, proof['prefix'][1])
            if proof['loss_within_plies'] == 2:
                self.assertEqual(game.getGameEnded(after, side)[1], 1)
            else:
                legal = set(map(int, np.flatnonzero(game.getValidMoves(after, side))))
                self.assertEqual({r['defender_action'] for r in proof['replies']}, legal)
                for row in proof['replies']:
                    child, turn = game.getNextState(after, side, row['defender_action'])
                    self.assertTrue(game.getValidMoves(child, turn)[row['winning_response']])
                    terminal, turn = game.getNextState(child, turn, row['winning_response'])
                    self.assertEqual(game.getGameEnded(terminal, turn)[1], 1)

    def test_race_is_a_forced_win_and_not_just_a_distance_estimate(self):
        state, d = candidate(13, 'race_instead_of_defending', 3)
        proof = attack_proof(state, d['good_action'])
        self.assertIsNotNone(proof)
        game = IntransitiveGame(modelling_draws=False)
        child, side = game.getNextState(state, 0, d['good_action'])
        legal = set(map(int, np.flatnonzero(game.getValidMoves(child, side))))
        self.assertEqual({r['opponent_action'] for r in proof['replies']}, legal)
        for row in proof['replies']:
            after, turn = game.getNextState(child, side, row['opponent_action'])
            terminal, turn = game.getNextState(after, turn, row['winning_response'])
            self.assertEqual(game.getGameEnded(terminal, turn)[0], 1)

    def test_illegal_or_effective_defence_is_not_refuted(self):
        state, d = candidate(13, 'capturable_route_block', 3)
        self.assertIsNone(refutation(state, d['good_action'], d['attacker_capture']))

    def test_twelve_variants_preserve_good_and_bad_moves(self):
        game = IntransitiveGame(modelling_draws=False)
        for name in TEMPLATES:
            if name == 'strictly_shorter_race':
                continue
            state, d = candidate(13, name, 3)
            legal = game.getValidMoves(state, 0)
            for s, (transformed, pi, mask) in enumerate(symmetries(state, d['good_action'], legal)):
                self.assertTrue(mask[transform_action(d['bad_action'], s)])
                self.assertTrue(mask[int(np.argmax(pi))])
                np.testing.assert_array_equal(mask, game.getValidMoves(transformed, 0))
                self.assertIsNotNone(refutation(transformed, transform_action(d['bad_action'], s),
                                               transform_action(d['attacker_capture'], s)))
                if name == 'race_instead_of_defending':
                    self.assertIsNotNone(attack_proof(transformed, int(np.argmax(pi))))

    def test_actual_teacher_prefers_attack_and_keeps_bad_moves_legal(self):
        state, d = candidate(13, 'race_instead_of_defending', 3)
        record = label(state, d)
        self.assertIsNotNone(record)
        self.assertEqual(record['action'], d['good_action'])
        self.assertTrue(record['legal'][d['bad_action']])
        self.assertEqual(len(record['defence_lesson']['rejected_defences']), 3)
        self.assertTrue(record['teacher_depth'] >= 5 or record['teacher_reason'] == 'proven_result')

    def test_strictly_shorter_race_is_won_and_all_three_defences_are_lost(self):
        state, d = candidate(13, 'strictly_shorter_race', 3)
        record = label(state, d)
        self.assertIsNotNone(record)
        lesson = record['defence_lesson']
        self.assertEqual(lesson['race_distances'], dict(own=1, opponent=2))
        self.assertEqual(lesson['winning_attack']['win_within_plies'], 1)
        self.assertEqual(len(lesson['rejected_defences']), 3)
        for bad in lesson['rejected_defences']:
            self.assertEqual(bad['proof']['loss_within_plies'], 6)
            self.assertGreater(bad['proof']['terminal_leaves'], 0)
        line = [parse_move(m) for m in ('G7 H8', 'C3 B2', 'B2 A1')]
        game = IntransitiveGame(modelling_draws=False)
        for s, (transformed, pi, _) in enumerate(symmetries(state, d['good_action'], game.getValidMoves(state, 0))):
            self.assertEqual(attack_proof(transformed, int(np.argmax(pi)))['win_within_plies'], 1)
            self.assertIsNotNone(planned_refutation(transformed, transform_action(d['bad_action'], s),
                                                   [transform_action(a, s) for a in line]))


if __name__ == '__main__':
    unittest.main()
