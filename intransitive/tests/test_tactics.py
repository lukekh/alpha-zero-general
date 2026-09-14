"""100 explicit move regressions, each independently certified before search.

Run: uv run --locked python -m unittest intransitive.tests.test_tactics -v
"""

from collections import Counter
import json
from pathlib import Path
import unittest

import numpy as np

from intransitive.heuristics import AlphaBetaPlayer, SearchConfig
from intransitive.tests.tactical_oracle import certify, load_case, move_name


FIXTURES = Path(__file__).parents[1] / 'heuristics' / 'tactics.json'
CASES = json.loads(FIXTURES.read_text())['positions']

# Keep depth and all heuristic weights fixed. The higher work/time ceilings
# prevent machine speed from turning a move-quality test into a timeout test.
CONFIG = SearchConfig(max_depth=3, node_limit=2_000_000, time_limit=30)


class TacticalRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert len(CASES) == 100
        assert len({c['id'] for c in CASES}) == 100
        assert Counter(c['category'] for c in CASES) == {
            'immediate_goal': 20, 'clear_run': 20, 'goal_defence': 20,
            'save_piece': 20, 'safe_capture': 20,
        }
        assert Counter(c['player'] for c in CASES) == {0: 50, 1: 50}
        assert len({load_case(c)[0].pieces for c in CASES}) == 100


def make_test(case):
    def test(self):
        certify(case)
        reference, expected = load_case(case)
        state = reference.storage()
        before = state.copy()
        # Fresh player: no transposition-table hints from another fixture.
        player = AlphaBetaPlayer(config=CONFIG)
        result = player.analyze(state)
        detail = (f"{case['id']}: {case['reason']}\n"
                  f"Expected {case['expected']}, got {move_name(result.action)}; "
                  f"score={result.score}, depth={result.completed_depth}, "
                  f"work={result.work}, pv={list(map(move_name, result.pv))}\n"
                  f"pieces={case['pieces']}, player={case['player']}")
        self.assertFalse(result.stopped, f'Search budget exhausted: {detail}')
        self.assertGreater(result.completed_depth, 0, detail)
        self.assertEqual(result.action, expected, detail)
        np.testing.assert_array_equal(state, before)
    test.__doc__ = case['reason']
    return test


for case in CASES:
    setattr(TacticalRegressionTests, f"test_{case['id']}", make_test(case))


if __name__ == '__main__':
    unittest.main()
