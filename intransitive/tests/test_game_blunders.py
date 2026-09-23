"""Hard regressions from the 79-ply game, including currently failing defences.

Run: uv run --locked python -m unittest intransitive.tests.test_game_blunders -v
"""

import json
from pathlib import Path
import unittest

import numpy as np

from dataclasses import replace

from intransitive.heuristics import AlphaBetaPlayer, SearchConfig
from intransitive.record import load_record, parse_record_move, legacy_state_hash as state_hash
from intransitive.tests.reference_rules import Position, map_action, position
from intransitive.tests.tactical_oracle import material, move_name, tactical_proof


DATA = json.loads((Path(__file__).parents[1] / 'heuristics' /
                   'game_blunders.json').read_text())
# Each case records the horizon its tactic needs; searching shallower than that
# asks the engine to find something it cannot see. A flat depth 3 silently
# mis-specified red_14 (needs 4) and red_31 (needs 5), which both then chose the
# fixture's own recorded bad move.
#
# The work ceiling is the deterministic bound and the clock is only a runaway
# guard, so that this file keeps its promise of no machine-dependent timing
# assertions. Measured worst case is red_18 at depth 5: 409,882,387 work in
# 21.2s, so the cap sits at ~2.4x that and the clock far above it. Before
# 9424a35 adopted the route-based attack and defence terms these searches cost
# ~57k work, which is why 2,000,000 once fit and now does not.
CONFIG = SearchConfig(max_depth=3, node_limit=1_000_000_000, time_limit=300)


def config_for(case):
    """The shared limits, deepened to the horizon this case's tactic needs."""
    return replace(CONFIG, max_depth=max(CONFIG.max_depth, case['horizon_after_move']))


def meets_objective(board, action, case):
    """Judge the selected move by the tactic, never by the bot's evaluation."""
    side = board.player
    child = board.move(action)[0]
    objective = case['objective']
    remaining = case['horizon_after_move']
    if objective in ('avoid_material_loss', 'avoid_fork'):
        first = case.get('opponent_first')
        return not tactical_proof(
            child, 1 - side, remaining,
            material_target=material(board, 1 - side) + 1,
            first=parse_record_move(first) if first else None)
    if objective == 'avoid_goal_run':
        return not tactical_proof(child, 1 - side, remaining)
    if objective == 'win_material':
        return tactical_proof(child, side, remaining,
                              material_target=material(board, side) + 1)
    if objective == 'win_goal_run':
        return tactical_proof(child, side, remaining)
    if objective == 'immediate_goal':
        reason, rewards = child.terminal()
        return reason == 'corner' and rewards[side] == 1
    raise AssertionError(f'Unknown objective: {objective}')


def choose(test, board, case):
    state = board.storage()
    before = state.copy()
    result = AlphaBetaPlayer(config=config_for(case)).analyze(state)
    detail = (f"{case['id']}: {case['reason']}\n"
              f"Selected {move_name(result.action)}; score={result.score}, "
              f"depth={result.completed_depth}, work={result.work}, "
              f"pv={list(map(move_name, result.pv))}")
    test.assertFalse(result.stopped, f'Search exhausted its budget: {detail}')
    test.assertGreater(result.completed_depth, 0, detail)
    np.testing.assert_array_equal(state, before)
    return result.action, detail


class GameBlunderRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert len(DATA['original_positions']) == 6
        assert len(DATA['simplified_positions']) == 3
        cases = DATA['original_positions'] + DATA['simplified_positions']
        assert len({c['id'] for c in cases}) == len(cases)
        record = load_record(DATA['record_pgn'])
        assert len(record.actions) == 79
        assert state_hash(record.states[-1]) == (
            'ccebb0cda1373506e4d180bd12d547ec0441b80fa63b9abe8678e8a6f6e16a85')
        # Replay independently, preserving actual repetition/capture history.
        board = Position.initial()
        cls.states = [board]
        np.testing.assert_array_equal(board.storage(), record.states[0])
        for action, stored in zip(record.actions, record.states[1:]):
            board, _ = board.move(action)
            np.testing.assert_array_equal(board.storage(), stored)
            cls.states.append(board)


def original_test(case):
    def test(self):
        board = self.states[case['ply']]
        self.assertEqual(state_hash(board.storage()), case['state_sha256'])
        # Verify known good and bad controls first. A broken fixture/proof must
        # not look like an evaluator regression. Good moves are examples, not
        # an exhaustive whitelist: another valid defence must also pass.
        for move in case['good_moves']:
            self.assertTrue(meets_objective(board, parse_record_move(move), case),
                            f"Invalid positive control: {case['id']} {move}")
        self.assertFalse(meets_objective(board, parse_record_move(case['bad_move']), case),
                         f"Invalid negative control: {case['id']}")
        action, detail = choose(self, board, case)
        self.assertTrue(meets_objective(board, action, case), detail)
    test.__doc__ = case['reason']
    return test


def simplified_test(case, symmetry):
    def test(self):
        board = Position.fixture(position(case['pieces']), player=case['player'])
        board = board.transform(symmetry)
        expected = map_action(parse_record_move(case['expected']), symmetry)
        good = {a for a in board.legal() if meets_objective(board, a, case)}
        self.assertEqual(good, {expected},
                         f"Fixture must have a unique answer: {case['id']}; "
                         f"certified moves={list(map(move_name, sorted(good)))}")
        action, detail = choose(self, board, case)
        self.assertEqual(action, expected,
                         f'Expected {move_name(expected)}. {detail}')
    test.__doc__ = case['reason']
    return test


for case in DATA['original_positions']:
    setattr(GameBlunderRegressionTests, f"test_game_{case['id']}", original_test(case))
for case in DATA['simplified_positions']:
    for symmetry, suffix in ((0, 'original_colours'), (6, 'exchanged_colours')):
        setattr(GameBlunderRegressionTests, f"test_simplified_{case['id']}_{suffix}",
                simplified_test(case, symmetry))


if __name__ == '__main__':
    unittest.main()
