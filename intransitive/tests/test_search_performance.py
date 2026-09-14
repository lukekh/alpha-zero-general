"""Correctness guards for optimizations; no machine-dependent timing assertions."""
from itertools import product
from unittest.mock import patch
import unittest
import numpy as np

from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveLogicNumba import Board
from intransitive.record import parse_record_move
from intransitive.heuristics import AlphaBetaPlayer, SearchConfig
from intransitive.heuristics.budget import Budget
from intransitive.heuristics.evaluation import Evaluator, terminal_value, MATE
from intransitive.heuristics.kernels import no_terminal_win_in_horizon, winning_actions, warm_search_kernels
from intransitive.tests.test_heuristics import position, unlimited


LINES = (
    'D4-E5 G5-F4 C3-D4 F4-F5 D4-E4 F5-E6 C5-D6 H5-G4 E3-F3 G4-G5 F3-G4 E6-D7 E2-F3',
    'C4-D5 F6-E5 B4-C4 E5-E4 E3-F3 G5-F4 D2-E3 E4-D3 C3-D3',
)


def replay(line):
    board = Board()
    for move in line.split():
        board.make_move(parse_record_move(move), board.get_next_player())
    return board.get_state()


class SearchPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warm_search_kernels()
        cls.game = IntransitiveGame()
        cls.states = [cls.game.getInitBoard()] + [replay(line) for line in LINES]

    def test_scalar_and_detailed_evaluations_agree_for_every_module_combination(self):
        states = self.states + [position({'G7': 3, 'I1': -1}),
            position({'H8': 3, 'B2': -1}), position({'A5': 2, 'C5': -3, 'D1': -3, 'E5': -1})]
        for flags in product((False, True), repeat=3):
            config = SearchConfig(attack_enabled=flags[0], defence_enabled=flags[1], overload_enabled=flags[2])
            evaluator = Evaluator(self.game, config)
            for state in states:
                before = state.tobytes()
                for side in (0, 1):
                    full = evaluator.explain(state, side, unlimited())
                    compact = evaluator.explain(state, side, unlimited(), diagnostics=False)
                    self.assertEqual(evaluator.score(state, side, unlimited()), full['score'])
                    self.assertEqual(compact['terms'], full['terms'])
                    self.assertEqual(compact['features'], full['features'])
                self.assertEqual(state.tobytes(), before)

    def test_core_search_never_builds_routes_or_route_explanations(self):
        with (patch('intransitive.heuristics.geometry.distance_map', side_effect=AssertionError('unused route')),
              patch('intransitive.heuristics.evaluation.race_candidates', side_effect=AssertionError('unused diagnostic'))):
            result = AlphaBetaPlayer(config=SearchConfig(max_depth=2, time_limit=60)).analyze(self.states[1])
        self.assertEqual(result.completed_depth, 2)
        self.assertEqual(result.module_calls.get('routes', 0), 0)
        self.assertEqual(result.module_calls.get('clear_run', 0), 0)

    def test_saved_games_keep_original_fixed_depth_scores_and_moves(self):
        config = SearchConfig(max_depth=3, time_limit=60, node_limit=10**8)
        for state, action, score in zip(self.states[1:], (406, 469), (0., 1.979166666666654)):
            result = AlphaBetaPlayer(config=config).analyze(state)
            self.assertEqual(result.completed_depth, 3)
            self.assertEqual(result.action, action)
            self.assertEqual(result.score, score)

    def test_winning_move_flags_match_full_engine_including_stalemate(self):
        rng = np.random.default_rng(17)
        states = self.states + [position({'D4': 1, 'E5': -2}),
                                position({'H8': 3, 'I9': -1, 'D4': -2})]
        # Include reached positions, captures, changing turn and history.
        state = self.game.getInitBoard()
        for _ in range(35):
            side = int(state[:, :, 82:84].flat[1])
            actions = np.flatnonzero(self.game.getValidMoves(state, side))
            if not len(actions):
                break
            state, _ = self.game.getNextState(state, side, int(rng.choice(actions)))
            states.append(state)
        for state in states:
            side = int(state[:, :, 82:84].flat[1])
            actions = np.flatnonzero(self.game.getValidMoves(state, side))
            goal = 80 if side == int(state[:, :, 82:84].flat[2]) else 0
            before = state.tobytes()
            wins = winning_actions(state[:, :, 0], actions, side, goal)
            for action, win in zip(actions, wins):
                child, _ = self.game.getNextState(state, side, int(action))
                self.assertEqual(bool(win), terminal_value(self.game, child, side) == MATE)
            self.assertEqual(state.tobytes(), before)

    def test_no_win_bound_against_complete_two_ply_trees(self):
        def check_tree(state, depth):
            side = int(state[:, :, 82:84].flat[1])
            value = terminal_value(self.game, state, side)
            self.assertIn(value, (None, 0.))
            if not depth or value is not None:
                return
            for action in np.flatnonzero(self.game.getValidMoves(state, side)):
                child, _ = self.game.getNextState(state, side, int(action))
                check_tree(child, depth - 1)

        canonical = [self.game.getCanonicalForm(s, int(s[:, :, 82:84].flat[1])) for s in self.states]
        for state in self.states + canonical:
            side = int(state[:, :, 82:84].flat[1])
            a1 = int(state[:, :, 82:84].flat[2])
            self.assertTrue(no_terminal_win_in_horizon(state[:, :, 0], side, a1, 2))
            check_tree(state, 2)
        for state in (position({'H8': 3, 'B2': -1}), position({'D4': 1, 'E5': -2})):
            self.assertFalse(no_terminal_win_in_horizon(state[:, :, 0], 0, 0, 2))

    def test_move_ordering_creates_history_states_only_when_visited(self):
        state = self.states[0]
        player = AlphaBetaPlayer(config=SearchConfig())
        player._prepare()
        with patch.object(player.game, 'getNextState', wraps=player.game.getNextState) as transition:
            ordered = player._ordered(state, 0, None, Budget(10**8, 60))
            next(ordered)
            self.assertEqual(transition.call_count, 1)


if __name__ == '__main__':
    unittest.main()
