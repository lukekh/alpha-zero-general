"""The proof gate must never skip a position the proof could have decided.

`no_terminal_win_in_horizon` is a sufficient condition for "no terminal result
within the horizon". A false positive would silently discard a forced win, so
these tests run an ungated, full-strength proof on every position the gate
skips and assert it finds nothing.
"""
import unittest

import numpy as np

from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.heuristics.budget import Budget
from intransitive.heuristics.config import SearchConfig
from intransitive.heuristics.kernels import (
    crowded_side_has_moves, no_terminal_win_in_horizon, open_side_has_moves,
)
from intransitive.heuristics.search import prove_reference
from intransitive.tests.test_attribution import random_positions
from intransitive.tests.test_heuristics import position, unlimited


def board_of(state):
    return state[:, :, 0]


def meta(state):
    return int(state[:, :, 82:84].flat[1]), int(state[:, :, 82:84].flat[2])


class ProofGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()
        cls.states = random_positions(games=10, plies=70, seed=61)

    def test_the_gate_never_skips_a_position_the_proof_can_decide(self):
        # A generous node budget so the reference verdict is never truncated.
        config = SearchConfig(proof_depth=2, proof_nodes=100000)
        skipped = 0
        for state in self.states:
            turn, a1 = meta(state)
            if not no_terminal_win_in_horizon(board_of(state), turn, a1, config.proof_depth):
                continue
            skipped += 1
            result = prove_reference(self.game, state, config, unlimited())
            with self.subTest(skipped=skipped):
                self.assertNotEqual(result['status'], 'proven')
        self.assertGreater(skipped, 0, 'the corpus never exercised the gate')

    def test_adding_the_open_rule_only_ever_widens_the_gate(self):
        # Anything the pair rule accepted must still be accepted.
        for state in self.states:
            pieces = board_of(state)
            turn, a1 = meta(state)
            for depth in (1, 2, 3):
                corner_clear = no_terminal_win_in_horizon(pieces, turn, a1, depth)
                pair_only = all(crowded_side_has_moves(pieces, side, depth) for side in (0, 1))
                if pair_only and self._corners_clear(pieces, turn, a1, depth):
                    with self.subTest(depth=depth):
                        self.assertTrue(corner_clear)

    @staticmethod
    def _corners_clear(pieces, turn, a1, depth):
        for y in range(9):
            for x in range(9):
                piece = int(pieces[y, x])
                if not piece:
                    continue
                side = int(piece < 0)
                corner = 8 if side == a1 else 0
                if max(abs(x - corner), abs(y - corner)) <= (depth + int(side == turn)) // 2:
                    return False
        return True

    def test_the_open_rule_fires_where_the_pair_rule_cannot(self):
        # Three pieces a side, well spaced: too few for 2*depth+1 = 5 pairs,
        # but each has far more than two empty neighbours.
        state = position({'C3': 1, 'E3': 2, 'C5': 3, 'G7': -1, 'E7': -2, 'G5': -3})
        pieces = board_of(state)
        for side in (0, 1):
            self.assertFalse(crowded_side_has_moves(pieces, side, 2))
            self.assertTrue(open_side_has_moves(pieces, side, 2))
        turn, a1 = meta(state)
        self.assertTrue(no_terminal_win_in_horizon(pieces, turn, a1, 2))

    def test_the_pair_rule_still_carries_the_opening(self):
        state = self.game.getInitBoard()
        pieces = board_of(state)
        turn, a1 = meta(state)
        for side in (0, 1):
            self.assertTrue(crowded_side_has_moves(pieces, side, 2))
        self.assertTrue(no_terminal_win_in_horizon(pieces, turn, a1, 2))

    def test_a_piece_within_reach_of_its_corner_still_blocks_the_gate(self):
        # Blue paper on H8 is one step from I9, so a corner win is not excluded
        # however roomy the rest of the board is.
        state = position({'H8': 3, 'C3': 1, 'E3': 2, 'G7': -1, 'E7': -2, 'G5': -3})
        turn, a1 = meta(state)
        self.assertFalse(no_terminal_win_in_horizon(board_of(state), turn, a1, 2))

    def test_a_side_that_could_run_out_of_pieces_is_not_cleared(self):
        # Two pieces a side cannot satisfy either argument at depth two: both
        # could genuinely be captured inside the horizon.
        state = position({'C3': 1, 'E3': 2, 'G7': -1, 'E7': -2})
        pieces = board_of(state)
        for side in (0, 1):
            self.assertFalse(crowded_side_has_moves(pieces, side, 2))
            self.assertFalse(open_side_has_moves(pieces, side, 2))
        turn, a1 = meta(state)
        self.assertFalse(no_terminal_win_in_horizon(pieces, turn, a1, 2))

    def test_a_walled_in_side_is_never_cleared(self):
        # Blue rock sealed into the A1 corner by friendly pieces it cannot take.
        entries = {'A1': 1, 'A2': 1, 'B1': 1, 'B2': 1,
                   'G7': -1, 'E7': -2, 'G5': -3, 'E5': -1}
        state = position(entries)
        pieces = board_of(state)
        self.assertFalse(open_side_has_moves(pieces, 0, 2))


if __name__ == '__main__':
    unittest.main()
