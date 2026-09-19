"""Storage isolation and trusted internal shortcuts in the search hot paths."""
import unittest
from unittest.mock import patch
import numpy as np

from intransitive.heuristics import AlphaBetaPlayer, SearchConfig
from intransitive.heuristics.budget import Budget
from intransitive.heuristics.position import SearchPosition
from intransitive.heuristics.proof import compact_proof, native_compact_proof
from intransitive.IntransitiveConstants import MAX_TOTAL_PLY
from intransitive.tests.reference_rules import Position, position


class SearchEfficiencyTests(unittest.TestCase):
    def test_default_compiles_existing_order_without_enhanced_heuristics(self):
        self.assertTrue(SearchConfig().compiled_ordering_enabled)
        self.assertFalse(SearchConfig().ordering_enabled)

    def test_only_checked_search_skips_public_terminal_query(self):
        node = SearchPosition(Position.initial().storage())
        expected = node.legal()
        player = AlphaBetaPlayer(config=SearchConfig(proof_nodes=0, time_limit=60.))
        player._prepare()
        with patch.object(node, 'terminal', side_effect=AssertionError('duplicate terminal check')):
            np.testing.assert_array_equal(node.raw_legal(), expected)
            actions = [a for a, _ in player._ordered(node, node.side, None,
                       Budget(10**9, 60.), terminal_checked=True)]
            self.assertEqual(sorted(actions), list(expected))
            with self.assertRaisesRegex(AssertionError, 'duplicate'):
                node.legal()
            with self.assertRaisesRegex(AssertionError, 'duplicate'):
                list(player._ordered(node, node.side, None, Budget(10**9, 60.)))
        terminal = SearchPosition(Position.fixture(position({'I9': 1, 'D4': -2})).storage())
        self.assertTrue(len(terminal.raw_legal()))
        self.assertEqual(len(terminal.legal()), 0)

    def test_reused_proof_buffers_reset_after_budget_and_exception(self):
        node = SearchPosition(Position.fixture(position({'H8': 3, 'D4': -1})).storage())
        saved = node.export().tobytes()
        score, line, counts = compact_proof(node, 2, 64, 129)
        retained_line, retained_counts = line.copy(), counts.copy()
        scratch = node._proof_scratch
        arrays = tuple(id(getattr(scratch, name)) for name in
                       ('pieces', 'masks', 'history', 'keys', 'lines', 'counts'))
        for work in (0, 1, 2, 17, 129):
            result = compact_proof(node, 2, 64, work)
            length = len(node.history)
            history = np.empty((length + 2, 82), dtype=np.int8)
            history[:length] = np.frombuffer(b''.join(node.history), dtype=np.int8).reshape(length, 82)
            keys = np.zeros(length + 2, dtype=np.int64)
            fresh = native_compact_proof(node.pieces.copy(), history, keys, length, node.side,
                    node.a1, node.total, node.modelling_draws, 2, 64, work, node.masks.copy())
            self.assertEqual(result[0], fresh[0])
            np.testing.assert_array_equal(result[1], fresh[1])
            np.testing.assert_array_equal(result[2], fresh[2])
            self.assertIs(node._proof_scratch, scratch)
        total = node.total
        node.total = MAX_TOTAL_PLY
        with self.assertRaisesRegex(ValueError, 'overflow'):
            compact_proof(node, 2, 64, 129)
        node.total = total
        # Simulate an exception leaving arbitrary scratch content; the next call
        # must fully overwrite every readable input/output region.
        for name in ('pieces', 'masks', 'history', 'keys', 'lines', 'counts'):
            getattr(scratch, name).fill(7)
        result = compact_proof(node, 2, 64, 129)
        self.assertEqual(score, result[0])
        np.testing.assert_array_equal(result[1], retained_line)
        np.testing.assert_array_equal(result[2], retained_counts)
        np.testing.assert_array_equal(line, retained_line)
        np.testing.assert_array_equal(counts, retained_counts)
        self.assertEqual(node.export().tobytes(), saved)
        self.assertEqual(arrays, tuple(id(getattr(scratch, name)) for name in
                         ('pieces', 'masks', 'history', 'keys', 'lines', 'counts')))
        other = SearchPosition(node.export())
        compact_proof(other, 2, 64, 129)
        self.assertIsNot(other._proof_scratch, scratch)
