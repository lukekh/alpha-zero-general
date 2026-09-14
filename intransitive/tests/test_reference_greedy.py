"""Reference feature semantics, canonical perspective and player integration."""

import unittest

import numpy as np

from intransitive.IntransitiveDisplay import parse_move
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveLogicNumba import Board
from intransitive.IntransitivePlayers import ReferenceGreedyPlayer
from intransitive.reference_greedy import FEATURES, features, reference_value
from intransitive.tests.test_rules import load_position


class ReferenceGreedyTests(unittest.TestCase):
    def setUp(self):
        self.game = IntransitiveGame()

    def position(self, entries):
        pieces = np.zeros((9, 9), dtype=np.int8)
        for square, piece in entries.items():
            pieces[int(square[1])-1, ord(square[0])-ord('A')] = piece
        return load_position(Board(), pieces)

    def test_feature_fixtures_for_both_corners_and_all_piece_types(self):
        cases = [
            # goal, capture, progress, danger, base_threat
            ({'E5': 1, 'A8': -1}, 'E5 F6', (0, 0, 1, 0, 0), 2),
            ({'E5': 1, 'A8': -1}, 'E5 F5', (0, 0, 0, 0, 0), 0),
            ({'E5': 1, 'F6': -2, 'G6': -3}, 'E5 F6', (0, 1, 1, 1, 0), 4),
            # The captured piece is removed before computing danger.
            ({'E5': 1, 'F6': -2, 'A8': -1}, 'E5 F6', (0, 1, 1, 0, 0), 12),
            ({'E5': 1, 'B2': -1}, 'E5 F6', (0, 0, 1, 0, 1), -498),
            # An occupied home base is threatened only by a legal capture.
            ({'E5': 1, 'A1': 1, 'B2': -2}, 'E5 F6', (0, 0, 1, 0, 0), 2),
            ({'E5': 1, 'A1': 1, 'B2': -3}, 'E5 F6', (0, 0, 1, 0, 1), -498),
            ({'B2': 1, 'A2': -2}, 'B2 A2', (0, 1, 0, 0, 0), 10),
            # Winning suppresses both danger and a simultaneous base threat.
            ({'H8': 1, 'I9': -2, 'H9': -3, 'B2': -1}, 'H8 I9',
             (1, 1, 1, 0, 0), 1012),
        ]
        from intransitive.IntransitiveSymmetries import transform_action, transform_state
        for entries, move, expected, score in cases:
            state = self.position(entries)
            action = parse_move(move)
            for symmetry in range(12):
                mapped = transform_state(state, symmetry)
                player = int(mapped[:, :, 32].flat[1])
                canonical = self.game.getCanonicalForm(mapped, player)
                mapped_action = int(transform_action(action, symmetry))
                with self.subTest(move=move, symmetry=symmetry):
                    self.assertTrue(self.game.getValidMoves(canonical, 0)[mapped_action])
                    saved = canonical.copy()
                    actual = features(canonical, mapped_action)
                    self.assertEqual(tuple(actual[key] for key in FEATURES), expected)
                    self.assertEqual(reference_value(actual), score)
                    np.testing.assert_array_equal(canonical, saved)

    def test_seeded_ties_select_only_maxima_without_mutation(self):
        state = self.game.getInitBoard()
        saved = state.copy()
        legal = np.flatnonzero(self.game.getValidMoves(state, 0))
        scores = {int(a): reference_value(features(state, int(a))) for a in legal}
        best = {a for a, score in scores.items() if score == max(scores.values())}
        first = ReferenceGreedyPlayer(self.game, seed=8)
        second = ReferenceGreedyPlayer(self.game, seed=8)
        choices = [first.play(state) for _ in range(60)]
        self.assertEqual(choices, [second.play(state) for _ in range(60)])
        self.assertLessEqual(set(choices), best)
        self.assertGreater(len(set(choices)), 1)
        np.testing.assert_array_equal(state, saved)

    def test_win_and_defence_priorities(self):
        for entries, move in (
            ({'H8': 1, 'I9': -2, 'H9': -3, 'B2': -1}, 'H8 I9'),
            ({'B2': 1, 'A2': -2, 'C3': -2}, 'B2 A2'),
        ):
            self.assertEqual(ReferenceGreedyPlayer(self.game, seed=8).play(self.position(entries)),
                             parse_move(move))

    def test_browser_opponent_plays_either_colour(self):
        from intransitive.play import OpponentFactory
        factory = OpponentFactory()
        self.assertIn('reference-greedy', factory.choices)
        opponent = factory.create('reference-greedy')
        state = self.game.getInitBoard()
        for player in (0, 1):
            action = opponent.choose(state, player)
            self.assertTrue(self.game.getValidMoves(state, player)[action])
            state, _ = self.game.getNextState(state, player, action)


if __name__ == '__main__':
    unittest.main()
