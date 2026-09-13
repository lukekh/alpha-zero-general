"""Check benchmark accounting against explicit byte sizes and known distributions."""
import pickle
import unittest
import zlib

import numpy as np

from intransitive.benchmark_training import distribution, replay_metrics
from intransitive.IntransitiveGame import IntransitiveGame


class BenchmarkAccounting(unittest.TestCase):
    def test_distribution_includes_empty_and_tail(self):
        self.assertEqual(distribution([]), {'count': 0})
        result = distribution([10, 20, 30, 40, 100])
        self.assertEqual(result['median'], 30)
        self.assertEqual(result['max'], 100)
        self.assertAlmostEqual(result['p95'], 88)

    def test_replay_accounting_includes_all_fields_and_compression(self):
        game = IntransitiveGame()
        state = game.getInitBoard()
        mask = game.getValidMoves(state, 0)
        example = (state, mask.astype(np.float32) / mask.sum(),
                   np.array([1., -1.], dtype=np.float32), mask, [np.float32(0.), np.float32(0.)])
        raw = pickle.dumps(example)
        blob = zlib.compress(raw, level=1)
        metrics = replay_metrics([blob, blob])
        self.assertEqual(metrics['examples'], 2)
        self.assertEqual(metrics['compressed_payload_bytes'], 2 * len(blob))
        self.assertEqual(metrics['pickle_payload_bytes'], 2 * len(raw))
        self.assertGreater(metrics['compressed_resident_bytes'], 2 * len(blob))
        self.assertGreater(metrics['decoded_resident_upper_estimate_bytes'], 2 * len(raw))
        self.assertEqual(metrics['history_length']['max'], 1)
        self.assertEqual(metrics['array_payload_bytes']['mean'],
                         sum(x.nbytes for x in example if isinstance(x, np.ndarray)))


if __name__ == '__main__':
    unittest.main()
