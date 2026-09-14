"""Correctness, tactical counterexamples and independent modular ablations."""
from dataclasses import replace
from itertools import product
from math import inf
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

from intransitive.heuristics import AlphaBetaPlayer, SearchConfig, exhaustive_minimax
from intransitive.heuristics.config import TIME_FIRST_LIMITS
from intransitive.heuristics.search import position_key
from intransitive.heuristics.budget import Budget, BudgetExpired
from intransitive.heuristics.evaluation import (
    Evaluator, MATE, HEURISTIC_LIMIT, MODULES, piece_advantage, race_candidates,
    attacking_position, defensive_position, overload, terminal_value, coverage,
)
from intransitive.heuristics.geometry import Geometry, arrival, captures
from intransitive.heuristics.search import prove, to_table, from_table
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveLogicNumba import Board
from intransitive.IntransitiveDisplay import parse_move
from intransitive.IntransitiveSymmetries import transform_state
from intransitive.tests.test_rules import load_position
from intransitive.tests.test_draws import load_history


def position(entries, turn=0, a1=0):
    pieces = np.zeros((9, 9), dtype=np.int8)
    for square, piece in entries.items():
        pieces[int(square[1]) - 1, ord(square[0]) - ord('A')] = piece
    return load_position(Board(), pieces, turn, a1)


def unlimited():
    return Budget(10**12, 3600)


class HeuristicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()
        cls.config = SearchConfig(time_limit=60, node_limit=10**9, proof_nodes=0)
        # Warm compiled rule and route kernels before wall-time assertions.
        s = cls.game.getInitBoard()
        Geometry(s, unlimited())
        cls.game.getNextState(s, 0, parse_move('B5 B6'))

    def geometry(self, entries, turn=0):
        return Geometry(position(entries, turn), unlimited())

    def explain(self, state, config=None, side=0):
        return Evaluator(self.game, config or self.config).explain(state, side, unlimited())

    def test_capture_increases_material_and_preserves_state(self):
        state = position({'D4': 1, 'E5': -2, 'H8': -3})
        before = state.copy()
        child, _ = self.game.getNextState(state, 0, parse_move('D4 E5'))
        self.assertEqual(self.explain(state)['terms']['piece_count'], -100)
        self.assertEqual(self.explain(child)['terms']['piece_count'], 0)
        np.testing.assert_array_equal(state, before)

    def test_cycle_zero_predator_bonus_and_gradual_scarcity(self):
        for kind in (1, 2, 3):
            predator = (kind + 1) % 3 + 1
            prey = kind % 3 + 1
            for gradual in (0., .5):
                config = replace(self.config, predator_scarcity_bonus=gradual)
                values = []
                for count in (2, 1, 0):
                    entries = {'D4': kind, 'E4': kind, 'H8': -prey}
                    entries.update({s: -predator for s in ('G7', 'F7')[:count]})
                    values.append(piece_advantage(self.geometry(entries), 0, config))
                self.assertGreater(values[2], values[1])
                self.assertGreaterEqual(values[1], values[0])
                if gradual:
                    self.assertGreater(values[1], values[0])
                self.assertEqual(piece_advantage(self.geometry({'H8': -prey}), 0, config), 0.)
            self.assertTrue(captures(kind, -prey))
            self.assertFalse(captures(kind, -kind))
        first = self.explain(position({'D4': 1, 'E4': 1, 'H7': -2, 'H8': -2}))
        second = self.explain(position({'D4': 1, 'E4': 1, 'H7': -3, 'H8': -3}))
        self.assertEqual(first['terms']['piece_count'], second['terms']['piece_count'])
        self.assertGreater(first['terms']['piece_advantage'], second['terms']['piece_advantage'])

    def test_goal_blockade_capturable_corner_and_interceptor(self):
        blocked = self.geometry({'G7': 3, 'I9': -3})
        self.assertEqual(race_candidates(blocked, 0)[0]['route'], [])
        capturable = self.geometry({'G7': 3, 'I9': -1})
        self.assertEqual(race_candidates(capturable, 0)[0]['arrival'], 3)
        intercepted = self.geometry({'E5': 3, 'G6': -2, 'G8': -1})
        race = race_candidates(intercepted, 0)[0]
        self.assertTrue(race['interceptors'])
        self.assertEqual(race['status'], 'unknown')
        self.assertEqual(race['estimate'], 0.)

    def test_clear_run_has_no_partial_positional_score(self):
        # Even an attractive route gets no credit when no win has been proved.
        for entries in ({'G7': 3, 'I1': -1}, {'E5': 3, 'G6': -2},
                        {'G7': 3, 'I9': -3}):
            state = position(entries)
            for weight in (0., 40., 10**9):
                explanation = self.explain(state, replace(self.config, race_weight=weight))
                self.assertEqual(explanation['terms']['clear_run'], 0.)
                for side in ('own', 'opponent'):
                    self.assertEqual(explanation['features'][side]['clear_run'], 0.)
                    self.assertTrue(all(c['estimate'] == 0 for c in explanation['races'][side]))

    def test_proven_run_overrides_material_and_is_json_safe(self):
        # Blue can finish in two moves; even huge material weights cannot oppose it.
        state = position({'G7': 3, 'I1': -1, 'H1': -2, 'G1': -3})
        before = state.tobytes()
        config = replace(self.config, proof_depth=3, proof_nodes=10000,
                         count_weight=10**9, race_weight=0.)
        own = self.explain(state, config)
        enemy = self.explain(state, config, side=1)
        self.assertEqual(own['proof']['status'], 'proven')
        self.assertEqual(own['score'], MATE - 3)
        self.assertEqual(own['terms']['clear_run'], MATE - 3)
        self.assertEqual(own['features']['own']['clear_run'], 1.)
        self.assertEqual(enemy['features']['opponent']['clear_run'], 1.)
        self.assertEqual(enemy['score'], -own['score'])
        json.dumps(own, allow_nan=False)
        self.assertEqual(state.tobytes(), before)
        shallow = self.explain(state, replace(config, proof_depth=2))
        self.assertEqual(shallow['proof']['status'], 'unknown')
        self.assertEqual(shallow['terms']['clear_run'], 0.)

    def test_faster_enemy_run_is_a_decisive_loss(self):
        state = position({'G7': 3, 'B2': -1})
        result = self.explain(state, replace(self.config, proof_nodes=500, proof_depth=2))
        self.assertEqual(result['score'], -MATE + 2)
        self.assertEqual(result['features']['own']['clear_run'], 0.)
        self.assertEqual(result['features']['opponent']['clear_run'], 1.)

    def test_old_race_weight_cannot_change_binary_scoring(self):
        config = SearchConfig(evaluator_version='intransitive-heuristics-v1', race_weight=40.)
        self.assertEqual(config.evaluator_version, 'intransitive-heuristics-v2')
        state = position({'H8': 3, 'B2': -1})
        for weight in (0., 40., 10**9):
            result = self.explain(state, replace(config, race_weight=weight))
            self.assertEqual(result['score'], MATE - 1)

    def test_same_type_and_friendly_detours(self):
        g = self.geometry({'G7': 3, 'H8': -3, 'H7': 1, 'G8': 2})
        runner = g.by_square[60]
        self.assertGreater(runner.distance, 2)
        self.assertNotIn(70, g.route(runner))

    def test_tempo_competing_type_and_proven_clear_run(self):
        state = position({'H8': 3, 'B2': -1})
        for turn in (0, 1):
            s = position({'H8': 3, 'B2': -1}, turn)
            proof = prove(self.game, s, replace(self.config, proof_nodes=500, proof_depth=3), unlimited())
            self.assertEqual(proof['status'], 'proven')
            self.assertEqual(proof['score'], MATE - 1)
            g = Geometry(s, unlimited())
            own = race_candidates(g, 0)[0]
            self.assertEqual(own['arrival'], 1 if turn == 0 else 2)
            self.assertEqual(own['opponent_arrival'], 2 if turn == 0 else 1)
        self.assertEqual(arrival(3, 0, 0), 5)
        self.assertEqual(arrival(3, 0, 1), 6)
        s = position({'G7': 3, 'B2': -1})
        result = prove(self.game, s, replace(self.config, proof_nodes=500, proof_depth=2), unlimited())
        self.assertEqual(result['score'], -MATE + 2)

    def test_immediate_stalemate_is_ordered_and_selected_as_a_win(self):
        state = position({'D4': 1, 'E5': -2})
        player = AlphaBetaPlayer(config=replace(self.config, max_depth=1))
        result = player.analyze(state)
        self.assertEqual(result.action, parse_move('D4 E5'))
        self.assertEqual(result.score, MATE - 1)
        child, side = self.game.getNextState(state, 0, result.action)
        self.assertEqual(terminal_value(self.game, child, 0), MATE)
        self.assertEqual(Board().get_terminal_reason(), 'ongoing')
        board = Board()
        board.copy_state(child, False)
        self.assertEqual(board.get_terminal_reason(), 'stalemate')

    def test_proof_exhaustion_is_unknown(self):
        s = position({'E5': 3, 'G6': -2})
        proof = prove(self.game, s, replace(self.config, proof_nodes=1), unlimited())
        self.assertEqual(proof['status'], 'unknown')
        self.assertEqual(proof['reason'], 'proof budget')
        self.assertNotIn('score', proof)
        result = self.explain(s, replace(self.config, proof_nodes=1))
        self.assertEqual(result['proof']['status'], 'unknown')
        self.assertEqual(result['terms']['clear_run'], 0.)
        b = Budget(1, 60)
        with self.assertRaises(BudgetExpired):
            prove(self.game, s, replace(self.config, proof_nodes=100), b)
        self.assertLessEqual(b.work, 1)

    def test_draws_and_official_win_precedence(self):
        state = position({'H8': 3, 'D4': -1})
        pieces = state[:, :, 0]
        from intransitive.tests.test_draws import clock_history
        state = load_history(Board(), clock_history(pieces, 79), first_player=1)
        self.assertIsNone(terminal_value(self.game, state, 0))
        child, _ = self.game.getNextState(state, 0, parse_move('H8 I9'))
        self.assertEqual(terminal_value(self.game, child, 0), MATE)
        child, _ = self.game.getNextState(state, 0, parse_move('H8 G8'))
        self.assertEqual(terminal_value(self.game, child, 0), 0)
        repeated = load_history(Board(), [pieces] * 5)
        self.assertEqual(self.explain(repeated)['score'], 0)
        proof = prove(self.game, repeated, replace(self.config, proof_nodes=100), unlimited())
        self.assertNotEqual(proof['status'], 'proven')

    def test_attack_safe_progress_and_unreachable_capture(self):
        safe = self.geometry({'G7': 3, 'A9': -1})
        unsafe = self.geometry({'G7': 3, 'H8': -2, 'A9': -1})
        self.assertGreater(attacking_position(safe, 0, self.config),
                           attacking_position(unsafe, 0, self.config))
        blocked = self.geometry({'H8': 3, 'I9': -3, 'A9': -1})
        self.assertEqual(attacking_position(blocked, 0, self.config), 0)

    def test_defence_timing_safety_and_capture_vs_block(self):
        timely = self.geometry({'B2': 2, 'D4': -3, 'H8': 1})
        late = self.geometry({'I1': 2, 'D4': -3, 'H8': 1})
        self.assertGreater(defensive_position(timely, 0, self.config),
                           defensive_position(late, 0, self.config))
        exposed = self.geometry({'B2': 2, 'D4': -3, 'B3': -1, 'H8': 1})
        self.assertLess(defensive_position(exposed, 0, self.config),
                        defensive_position(timely, 0, self.config))
        for kind, expected in ((3, True), (1, False)):
            g = self.geometry({'B2': kind, 'D4': -3, 'H8': 1})
            replies = g.intercepts(g.by_square[10], g.by_square[30])
            self.assertEqual(bool(replies), expected)
            if replies:
                self.assertTrue(all(r['kind'] == 'block' for r in replies))

    def test_overload_and_real_redundancy(self):
        entries = {'A5': 2, 'C5': -3, 'D1': -3, 'E5': -1}
        base = overload(self.geometry(entries), 0, self.config, self.game)
        self.assertEqual(base, 1.)
        for support, expected in (({'A1': 3}, 0.), ({'A1': 2}, 0.),
                                  ({'A1': 1}, base), ({'I1': 2}, base),
                                  ({'H8': 3}, 0.)):
            with self.subTest(support=support):
                self.assertEqual(overload(self.geometry(dict(entries, **support)), 0,
                                          self.config, self.game), expected)
        # Two nearby threats alone do not imply overload: the scissors can hold
        # the goal or capture one while retaining coverage of the other.
        alternate = self.geometry({'B2': 2, 'C2': -3, 'B3': -3})
        self.assertEqual(overload(alternate, 0, self.config, self.game), 0.)

    def test_all_module_combinations_skip_disabled_computation(self):
        s = position({'D4': 1, 'G7': -2})
        for attack, defence, overloaded in product((False, True), repeat=3):
            config = replace(self.config, attack_enabled=attack, defence_enabled=defence, overload_enabled=overloaded)
            b = unlimited()
            explanation = Evaluator(self.game, config).explain(s, 0, b)
            json.dumps(explanation)
            self.assertLessEqual(abs(explanation['score']), HEURISTIC_LIMIT)
            for name, active in zip(MODULES[3:], (attack, defence, overloaded)):
                self.assertEqual(b.module_calls[name], int(active))
                if not active:
                    self.assertEqual(explanation['terms'][name], 0)
            other = self.explain(s, config, 1)
            self.assertAlmostEqual(explanation['score'], -other['score'])
        with patch('intransitive.heuristics.evaluation.attacking_position', side_effect=AssertionError), \
             patch('intransitive.heuristics.evaluation.defensive_position', side_effect=AssertionError), \
             patch('intransitive.heuristics.evaluation.overload', side_effect=AssertionError):
            self.explain(s)

    def test_config_validation_clamp_and_roundtrip(self):
        config = replace(self.config, count_weight=10**9)
        s = position({'D4': 1, 'E4': 1, 'H8': -2})
        self.assertEqual(self.explain(s, config)['score'], HEURISTIC_LIMIT)
        self.assertEqual(SearchConfig(**json.loads(config.identity())), config)
        for kw in ({'attack_enabled': 1}, {'time_limit': float('nan')}, {'overload_weight': -1},
                   {'max_depth': 1000}, {'node_limit': True}, {'proof_depth': 9}):
            with self.assertRaises(ValueError):
                SearchConfig(**kw)
        preset = SearchConfig.from_file(
            Path(__file__).parents[1] / 'heuristics' / 'configs' / 'time-first.json')
        self.assertEqual({name: getattr(preset, name) for name in TIME_FIRST_LIMITS},
                         TIME_FIRST_LIMITS)

    def test_search_matches_exhaustive_with_and_without_cache(self):
        for entries, depth in (({'G7': 3, 'C3': -1}, 2),
                               ({'D4': 1, 'E5': -2, 'G7': -3}, 2),
                               ({'H8': 3, 'B2': -1}, 2)):
            s = position(entries)
            reference, _ = exhaustive_minimax(self.game, s, depth, self.config)
            for cached in (False, True):
                player = AlphaBetaPlayer(config=replace(self.config, max_depth=depth), use_table=cached)
                value, pv = self.search(player, s, depth)
                self.assertAlmostEqual(value, reference)
                self.assertTrue(self.game.getValidMoves(s, 0)[pv[0]])
                value2, _ = self.search(player, s, depth)
                self.assertAlmostEqual(value2, reference)

    def search(self, player, state, depth):
        player._prepare()
        return player._search(state, depth, -inf, inf, 0, unlimited())

    def test_random_positions_and_leaf_proofs_match_reference(self):
        rng = np.random.default_rng(3434)
        for i in range(6):
            squares = rng.choice([y * 9 + x for y in range(1, 8) for x in range(1, 8)], 4, replace=False)
            entries = {chr(65 + int(s) % 9) + str(int(s) // 9 + 1): int(kind)
                       for s, kind in zip(squares, (1, 3, -2, -3))}
            s = position(entries, i % 2)
            config = replace(self.config, proof_nodes=20 if i % 2 else 0, proof_depth=2)
            value, _ = self.search(AlphaBetaPlayer(config=config), s, 2)
            reference, _ = exhaustive_minimax(self.game, s, 2, config)
            self.assertAlmostEqual(value, reference)

    def test_bound_entries_are_not_exact_and_research_agrees(self):
        s = position({'G7': 3, 'C3': -1})
        p = AlphaBetaPlayer(config=self.config)
        p._prepare()
        p._search(s, 2, -1., 1., 0, unlimited())
        self.assertIn(p.table[(position_key(s), 2)].bound, ('lower', 'upper'))
        score, _ = self.search(p, s, 2)
        reference, _ = exhaustive_minimax(self.game, s, 2, self.config)
        self.assertAlmostEqual(score, reference)
        self.assertEqual(p.table[(position_key(s), 2)].bound, 'exact')

    def test_history_separation_and_config_invalidation(self):
        s = position({'D4': 1, 'F6': -2})
        alternative = s[:, :, 0].copy()
        alternative[0, 4] = 3
        history = load_history(Board(), [s[:, :, 0], alternative, s[:, :, 0]])
        p = AlphaBetaPlayer(config=self.config)
        for state in (s, history):
            value, _ = self.search(p, state, 1)
            reference, _ = exhaustive_minimax(self.game, state, 1, self.config)
            self.assertAlmostEqual(value, reference)
        self.assertIn((position_key(s), 1), p.table)
        self.assertIn((position_key(history), 1), p.table)
        p.config = replace(self.config, race_weight=123)
        p._prepare()
        self.assertFalse(p.table)

    def test_mate_normalization_and_parent_to_child_reuse(self):
        for score in (MATE - 5, -MATE + 5, 25.):
            self.assertEqual(from_table(to_table(score, 3), 3), score)
        s = position({'G7': 3, 'I1': -1})
        p = AlphaBetaPlayer(config=self.config)
        self.search(p, s, 3)
        child, _ = self.game.getNextState(s, 0, parse_move('G7 H8'))
        cached, _ = self.search(p, child, 2)
        fresh, _ = self.search(AlphaBetaPlayer(config=self.config), child, 2)
        self.assertEqual(cached, fresh)

    def test_symmetries_canonical_colour_and_state_preservation(self):
        s = position({'H8': 3, 'C4': -2, 'D3': 1})
        before = s.tobytes()
        config = replace(self.config, attack_enabled=True, defence_enabled=True, overload_enabled=True)
        value = self.explain(s, config)['score']
        for symmetry in range(12):
            transformed = transform_state(s, symmetry)
            side = int(transformed[:, :, 82:84].flat[1])
            self.assertAlmostEqual(self.explain(transformed, config, side)['score'], value)
            p = AlphaBetaPlayer(config=replace(config, max_depth=1))
            result = p.analyze(transformed)
            child, _ = self.game.getNextState(transformed, side, result.action)
            self.assertEqual(terminal_value(self.game, child, side), MATE)
            self.assertEqual(result.score, MATE - 1)  # equivalent immediate wins may tie
            canonical = self.game.getCanonicalForm(transformed, side)
            self.assertEqual(p.play(canonical), result.action)
        self.assertEqual(s.tobytes(), before)

    def test_zero_budget_and_last_completed_iteration(self):
        s = position({'D4': 1, 'F6': -2})
        for config in (replace(self.config, node_limit=0), replace(self.config, time_limit=0)):
            r = AlphaBetaPlayer(config=config).analyze(s)
            self.assertEqual(r.completed_depth, 0)
            self.assertIsNone(r.score)
            self.assertTrue(self.game.getValidMoves(s, 0)[r.action])
            self.assertTrue(r.stopped)
        self.assertEqual(AlphaBetaPlayer(config=replace(self.config, node_limit=0)).analyze(s).stop_reason,
                         'work')
        self.assertEqual(AlphaBetaPlayer(config=replace(self.config, time_limit=0)).analyze(s).stop_reason,
                         'time')
        first = AlphaBetaPlayer(config=replace(self.config, max_depth=1)).analyze(s)
        limited = AlphaBetaPlayer(config=replace(self.config, node_limit=first.work + 1, max_depth=3)).analyze(s)
        self.assertEqual(limited.completed_depth, 1)
        self.assertEqual(limited.action, first.action)
        self.assertEqual(limited.score, first.score)
        self.assertLessEqual(limited.work, first.work + 1)


if __name__ == '__main__':
    unittest.main()
