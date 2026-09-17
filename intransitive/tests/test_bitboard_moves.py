"""Differential legality including lane boundaries, captures and restoration."""
import unittest
import numpy as np
from intransitive.IntransitiveLogicNumba import raw_movement_mask
from intransitive.heuristics.moves import masks_from_board, legal_actions, has_move, move_board
from intransitive.heuristics.position import SearchPosition
from intransitive.tests.reference_rules import Position


class BitboardMovesTests(unittest.TestCase):
    def check(self, board, masks):
        np.testing.assert_array_equal(masks, masks_from_board(board))
        for side in (0, 1):
            expected = np.flatnonzero(raw_movement_mask(board, side))
            np.testing.assert_array_equal(legal_actions(masks, side), expected)
            self.assertEqual(has_move(masks, side), bool(len(expected)))

    def test_every_source_direction_and_capture_pair(self):
        from intransitive.tests.reference_rules import destination
        for source in range(81):
            for mover in (-3, -2, -1, 1, 2, 3):
                for direction in range(8):
                    target = destination(8 * source + direction)
                    for occupant in range(-3, 4):
                        board = np.zeros((9, 9), dtype=np.int8)
                        board.flat[source] = mover
                        if target is not None:
                            board.flat[target] = occupant
                        masks = masks_from_board(board)
                        self.check(board, masks)
                        if target is not None and 8 * source + direction in legal_actions(masks, int(mover < 0)):
                            saved = board.copy()
                            move_board(board, masks, source, target, occupant)
                            self.check(board, masks)
                            move_board(board, masks, source, target, occupant, True)
                            np.testing.assert_array_equal(board, saved)
                            self.check(board, masks)

    def test_empty_and_fully_blocked_boards(self):
        for code in range(-3, 4):
            board = np.full((9, 9), code, dtype=np.int8)
            self.check(board, masks_from_board(board))
            for side in (0, 1):
                self.assertFalse(has_move(masks_from_board(board), side))

    def test_dense_and_reachable_nested_undo(self):
        rng = np.random.default_rng(62)
        for _ in range(300):
            board = rng.integers(-3, 4, (9, 9), dtype=np.int8)
            self.check(board, masks_from_board(board))
        for _ in range(10):
            node = SearchPosition(Position.initial().storage())
            saved = node.export().tobytes()
            for ply in range(100):
                self.check(node.pieces, node.masks)
                actions = node.legal()
                if not len(actions):
                    break
                node.push(int(rng.choice(actions)))
            while node.stack:
                node.pop()
                self.check(node.pieces, node.masks)
            self.assertEqual(saved, node.export().tobytes())
