"""The run certificate must never claim a win that is not forced.

A proof result replaces the whole weighted sum with a mate score and prunes the
lines that would refute it, so a false positive is not a small error. Every
claim these tests collect is re-searched full width to the claimed depth;
alpha-beta returns the true minimax value, so a mate-range score confirms it
and anything else refutes it.
"""
import unittest

import numpy as np

from intransitive.IntransitiveConstants import NO_CAPTURE_LIMIT
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.heuristics import AlphaBetaPlayer
from intransitive.heuristics.budget import Budget
from intransitive.heuristics.clear_run import NO_RUN, certify
from intransitive.heuristics.config import SearchConfig
from intransitive.heuristics.evaluation import MATE, MATE_THRESHOLD
from intransitive.heuristics.search import prove
from intransitive.tests.test_attribution import random_positions
from intransitive.tests.test_heuristics import position, unlimited


def clock_left(state):
    meta = state[:, :, 82:84].ravel()
    return max(0, NO_CAPTURE_LIMIT - max(int(meta[3]), max(0, int(meta[4]) - 1)))


def claim(state, side=None):
    meta = state[:, :, 82:84].ravel()
    turn, a1 = int(meta[1]), int(meta[2])
    return certify(state[:, :, 0], turn if side is None else side, turn, a1,
                   clock_left(state), 20)


class ClearRunCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()
        cls.states = random_positions(games=14, plies=120, seed=73)

    def test_every_claim_survives_a_full_width_search(self):
        checked = 0
        for state in self.states:
            if self.game.getGameEnded(state, int(state[:, :, 82:84].ravel()[1])).any():
                continue
            plies = claim(state)
            if plies == NO_RUN or not 1 <= plies <= 5:
                continue
            checked += 1
            config = SearchConfig(attack_enabled=False, defence_enabled=False,
                                  proof_depth=0, certificate_enabled=False,
                                  max_depth=int(plies), node_limit=20_000_000,
                                  time_limit=60.)
            result = AlphaBetaPlayer(self.game, config).analyze(state)
            if result.stopped or result.score is None:
                continue
            with self.subTest(plies=plies):
                self.assertGreater(result.score, MATE_THRESHOLD,
                                   'certificate claimed an unforced win')
                self.assertLessEqual(MATE - result.score, plies,
                                     'the real win is slower than claimed')
        self.assertGreater(checked, 0, 'the corpus never exercised the certificate')

    def test_a_runner_one_step_out_is_not_claimed_when_it_can_be_taken(self):
        # Blue paper on H8 is one step from I9, but Red moves and Red scissors
        # on G7 takes paper. The earlier version claimed a win here.
        state = position({'H8': 3, 'G7': -2, 'A1': 1, 'B2': -1}, turn=1)
        self.assertEqual(claim(state, side=0), NO_RUN)

    def test_a_runner_one_step_out_is_claimed_when_it_moves_first(self):
        state = position({'H8': 3, 'G7': -2, 'A1': 1, 'B2': -1}, turn=0)
        self.assertEqual(claim(state, side=0), 1)

    def test_a_blocked_corner_is_not_claimed(self):
        # Every square adjacent to I9 is held, so no run reaches it.
        state = position({'G7': 3, 'H8': 1, 'H9': 1, 'I8': 1, 'A1': -1, 'B1': -2})
        self.assertEqual(claim(state, side=0), NO_RUN)

    def test_a_faster_opponent_race_is_not_claimed(self):
        # Blue needs four moves to I9; Red sits one step from A1 and moves next.
        state = position({'E5': 3, 'B2': -3, 'A9': 1}, turn=1)
        self.assertEqual(claim(state, side=0), NO_RUN)

    def test_an_expiring_draw_clock_is_not_claimed(self):
        # Blue paper two steps from I9, Red far from its own corner.
        state = position({'G7': 3, 'A5': -1})
        self.assertEqual(certify(state[:, :, 0], 0, 0, 0, 80, 20), 3)
        # With fewer plies left than the run needs, the same run is refused.
        self.assertEqual(certify(state[:, :, 0], 0, 0, 0, 3, 20), NO_RUN)

    def test_a_piece_already_on_the_goal_is_not_claimed(self):
        # That position is terminal and belongs to the rules, not the proof.
        state = position({'I9': 3, 'A5': -1})
        self.assertEqual(certify(state[:, :, 0], 0, 0, 0, 80, 20), NO_RUN)

    def test_the_certificate_is_off_unless_enabled(self):
        # Three plies out, so the two-ply terminal proof cannot see it.
        state = position({'G7': 3, 'A5': -1})
        off = SearchConfig(proof_depth=2, proof_nodes=64, certificate_enabled=False)
        on = SearchConfig(proof_depth=2, proof_nodes=64, certificate_enabled=True)
        self.assertNotEqual(prove(self.game, state, on, unlimited())['status'], 'unknown')
        self.assertEqual(prove(self.game, state, off, unlimited())['status'], 'unknown')

    def test_a_claim_is_reported_from_the_mover_perspective(self):
        # Blue runs; with Red to move the score must read as a loss for Red.
        state = position({'G7': 3, 'A5': -1}, turn=1)
        config = SearchConfig(proof_depth=2, proof_nodes=64, certificate_enabled=True)
        result = prove(self.game, state, config, unlimited())
        if result['status'] == 'proven':
            self.assertLess(result['score'], 0)


if __name__ == '__main__':
    unittest.main()
