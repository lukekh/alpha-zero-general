"""Window correctness and deterministic interruptions, without timing assertions."""
from dataclasses import replace
from itertools import product
from math import inf, nextafter
import unittest

from intransitive.heuristics import AlphaBetaPlayer, SearchConfig, exhaustive_minimax
from intransitive.heuristics.budget import Budget
from intransitive.heuristics.position import SearchPosition
from intransitive.heuristics.search import RootProgress, from_table, position_key, to_table
from intransitive.heuristics.evaluation import MATE
from intransitive.tests.reference_rules import Position, position
from intransitive.tests.test_anytime_search import Clock


def unlimited():
    return Budget(10**9, 600)


class WindowChildren(AlphaBetaPlayer):
    """Real root/TT/ordering with deterministic fail-soft child returns.

    A child value outside its supplied window is explicitly just a bound.
    Interruption events occur at the actual probe/research call boundaries.
    """
    def __init__(self, state, config, values, interrupt=None, reason='time'):
        super().__init__(config=config, use_table=False)
        self.clock = Clock()
        self._prepare()
        side = int(state[:, :, 82:84].flat[1])
        ordered = list(self._ordered(state, side, None, unlimited()))
        self.actions = [a for a, _ in ordered]
        self.indices = {s.tobytes(): i for i, (_, s) in enumerate(ordered)}
        self.values = values
        self.interrupt = interrupt
        self.reason = reason
        self.calls = []

    def _search(self, state, depth, alpha, beta, ply, budget):
        if ply != 1:
            return super()._search(state, depth, alpha, beta, ply, budget)
        snapshot = state.export() if isinstance(state, SearchPosition) else state
        index = self.indices[snapshot.tobytes()]
        event = (depth, index, alpha, beta)
        self.calls.append(event)
        if self.interrupt and self.interrupt(self, event):
            if self.reason == 'time':
                self.clock.now = 2
                budget.check()
            else:
                budget.charge(budget.limit - budget.work + 1)
        budget.visit()
        row = self.values[min(depth, len(self.values) - 1)]
        return -row[min(index, len(row) - 1)], []


class SearchWindowTests(unittest.TestCase):
    def setUp(self):
        self.state = Position.fixture(position({'D4': 1, 'F6': -2})).storage()
        self.config = SearchConfig(max_depth=2, proof_nodes=0, time_limit=600,
                                   node_limit=10**9, aspiration_window=.125)

    def test_config_roundtrip_and_validation(self):
        import json
        config = replace(self.config, pvs_enabled=True, aspiration_enabled=True)
        self.assertEqual(SearchConfig(**json.loads(config.identity())), config)
        for field, values in [('pvs_enabled', [1, 'yes']), ('aspiration_enabled', [0, None]),
                              ('aspiration_window', [0, -1, inf, float('nan'), True, '1'])]:
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(config, **{field: value})

    def test_adjacent_fractional_scores_signed_zero_and_infinite_endpoints(self):
        for first in (-100.125, -.125, -0., 0., .125, MATE - 8., -MATE + 8.):
            challenger = nextafter(first, inf)
            config = replace(self.config, max_depth=1, pvs_enabled=True)
            player = WindowChildren(self.state, config, [[first, challenger, first]])
            result = player.analyze(self.state)
            self.assertEqual(result.score, challenger)
            self.assertEqual(result.action, player.actions[1])
            self.assertEqual(result.score_bound, 'exact')
            self.assertGreater(result.pvs_researches, 0)
            self.assertEqual(player.calls[0][2:], (-inf, inf))
            self.assertEqual(player.calls[1][2:], (-challenger, -first))
        for score in (-0., .125, MATE - 8., -MATE + 8., inf, -inf):
            self.assertEqual(from_table(to_table(score, 3), 3), score)

    def assert_pv(self, game, state, result, config):
        child = state
        for ply, action in enumerate(result.pv, 1):
            side = int(child[:, :, 82:84].flat[1])
            self.assertTrue(game.getValidMoves(child, side)[action])
            child, _ = game.getNextState(child, side, action)
            if ply <= result.selected_depth:
                value, _ = exhaustive_minimax(game, child, result.selected_depth-ply,
                                               config, unlimited())
                root_value = from_table(value if ply % 2 == 0 else -value, ply)
                self.assertEqual(root_value, result.score)
        first, _ = game.getNextState(state, int(state[:, :, 82:84].flat[1]), result.action)
        value, _ = exhaustive_minimax(game, first, result.selected_depth - 1, config, unlimited())
        self.assertEqual(result.score, from_table(-value, 1))

    def test_real_variants_match_exhaustive_both_colours_with_proofs_and_draws(self):
        boards = [Position.fixture(position(p)) for p in (
            {'D4': 1, 'E5': -2, 'G7': -3}, {'G7': 3, 'C3': -1},
            {'H8': 3, 'B2': -1}, {'I1': 2, 'B2': -3})]
        # A playable second occurrence and a position one quiet move from draw.
        a = position({'D4': 1, 'F6': -2})
        b = position({'D5': 1, 'F6': -2})
        boards.append(Position.fixture(a, history=(a, b, a)))
        from intransitive.tests.test_draws import clock_history, load_history
        from intransitive.IntransitiveLogicNumba import Board
        boards.append(Position.from_storage(load_history(Board(), clock_history(
            Position.fixture(a).storage()[:, :, 0], 79), first_player=1)))
        for board, colour, proof in product(boards, (0, 6), (0, 20)):
            state = board.transform(colour).storage()
            config = replace(self.config, proof_nodes=proof, count_weight=1.125,
                             advantage_weight=.137)
            game = AlphaBetaPlayer().game
            expected, _ = exhaustive_minimax(game, state, 2, config, unlimited())
            for pvs, aspiration, cached in product((False, True), repeat=3):
                with self.subTest(pieces=board.pieces, colour=colour, proof=proof,
                                  pvs=pvs, aspiration=aspiration, cached=cached):
                    cfg = replace(config, pvs_enabled=pvs, aspiration_enabled=aspiration)
                    player = AlphaBetaPlayer(config=cfg, use_table=cached)
                    before = state.tobytes()
                    result = player.analyze(state)
                    self.assertEqual(result.score, expected)
                    self.assertEqual(result.score_bound, 'exact')
                    self.assert_pv(game, state, result, cfg)
                    self.assertEqual(state.tobytes(), before)

    def test_tt_bounds_recovery_depth_identity_and_mate_distance(self):
        for pvs in (False, True):
            player = AlphaBetaPlayer(config=replace(self.config, pvs_enabled=pvs))
            player._prepare()
            expected, _ = exhaustive_minimax(player.game, self.state, 2, self.config)
            for low, high, bound in ((expected - 2, expected - 1, 'lower'),
                                     (expected + 1, expected + 2, 'upper')):
                player.reload()
                player._search(self.state, 2, low, high, 0, unlimited())
                self.assertEqual(player.table[(position_key(self.state), 2)].bound, bound)
                value, _ = player._search(self.state, 2, -inf, inf, 0, unlimited())
                self.assertEqual(value, expected)
                self.assertEqual(player.table[(position_key(self.state), 2)].bound, 'exact')
            shallow, _ = player._search(self.state, 1, -inf, inf, 0, unlimited())
            self.assertEqual(shallow, exhaustive_minimax(player.game, self.state, 1, self.config)[0])
            state = Position.fixture(position({'G7': 3, 'I1': -1})).storage()
            value, line = player._search(state, 3, -inf, inf, 0, unlimited())
            child, _ = player.game.getNextState(state, 0, line[0])
            cached, _ = player._search(child, 2, -inf, inf, 0, unlimited())
            self.assertEqual(cached, to_table(-value, 1))

    def run_script(self, values, interrupt, *, pvs=True, aspiration=False, reason='time'):
        config = replace(self.config, pvs_enabled=pvs, aspiration_enabled=aspiration)
        player = WindowChildren(self.state, config, values, interrupt, reason)
        result = player.analyze(self.state, Budget(10**7, 1, clock=player.clock))
        self.assertTrue(result.stopped)
        self.assertEqual(result.stop_reason, reason)
        self.assertEqual(result.completed_depth, 1)
        self.assertLessEqual(result.work, 10**7)
        return player, result

    def test_interrupted_probes_and_full_research_keep_verified_incumbent(self):
        for reason, research in product(('time', 'work'), (False, True)):
            def stop(player, event):
                return (event[:2] == (1, 1) and
                        (not research or player.calls.count(event) == 1 and
                         event[2] == -inf))
            player, result = self.run_script([[10, 8, 0], [1, 20, 0]], stop, reason=reason)
            self.assertEqual(result.action, player.actions[0])
            self.assertEqual(result.score, 1)
            self.assertEqual(result.score_bound, 'exact')
            self.assertEqual(result.root_moves_completed, 1)
            self.assertEqual(result.pvs_researches, int(research))

    def test_completed_pvs_challenger_survives_later_interruption(self):
        player, result = self.run_script([[10, 8, 0], [1, 20, 0]],
                                        lambda p, e: e[:2] == (1, 2))
        self.assertEqual(result.action, player.actions[1])
        self.assertEqual(result.score, 20)
        self.assertEqual(result.selected_depth, 2)
        self.assertEqual(result.score_bound, 'exact')

    def test_aspiration_recovery_both_directions_and_subnormal_width(self):
        for pvs, later in product((False, True), ([-100, -120, -130], [1, 20, 0])):
            for width in (.125, nextafter(0., inf)):
                player = WindowChildren(self.state, replace(self.config, pvs_enabled=pvs,
                    aspiration_enabled=True, aspiration_window=width), [[10, 8, 0], later])
                result = player.analyze(self.state)
                self.assertEqual(result.score, max(later))
                self.assertEqual(result.completed_depth, 2)
                self.assertEqual(result.score_bound, 'exact')
                self.assertGreater(result.aspiration_researches, 0)
                self.assertLessEqual(result.aspiration_researches, 8)
                if max(later) > 10:
                    self.assertGreater(result.aspiration_fail_highs, 0)
                else:
                    self.assertGreater(result.aspiration_fail_lows, 0)

    def test_interrupted_aspiration_bounds_do_not_replace_previous_verified_result(self):
        for reason, later in product(('time', 'work'), ([100, 0, 0], [-100, -120, -130])):
            player, result = self.run_script([[10, 8, 0], later],
                lambda p, e: e[:2] == (1, 0) and sum(c[:2] == (1, 0) for c in p.calls) == 2,
                aspiration=True, reason=reason)
            self.assertEqual(result.action, player.actions[0])
            self.assertEqual(result.score, 10)
            self.assertEqual(result.score_bound, 'exact')
            self.assertEqual(result.selected_depth, 1)
            self.assertEqual(result.partial_depth, 2)

    def test_proved_losing_incumbent_avoids_interrupted_challenger(self):
        for aspiration, reason in product((False, True), ('time', 'work')):
            player, result = self.run_script([[10, 8, 0], [-MATE + 4, 20, 0]],
                lambda p, e: e[:2] == (1, 1), aspiration=aspiration, reason=reason)
            self.assertEqual(result.action, player.actions[1])
            self.assertEqual(result.selection_source, 'unrefuted_fallback')
            self.assertNotEqual(result.explanation['proof']['status'], 'proven')

    def test_progress_retains_exact_sibling_and_labels_failed_windows(self):
        progress = RootProgress(2, 3, 0)
        progress.record(0, 10, [], 0, 20)
        progress.record(0, 11, [], 11, 20)
        progress.record(1, 20, [], 0, 20)
        progress.record(2, -10, [], 0, 20)
        self.assertEqual((progress.moves[0]['score'], progress.moves[0]['bound']), (10, 'exact'))
        self.assertEqual(progress.moves[1]['bound'], 'lower')
        self.assertEqual(progress.moves[2]['bound'], 'upper')

    def test_aspiration_retry_preserves_completed_siblings_during_pvs_research(self):
        for reason, after_challenger in product(('time', 'work'), (False, True)):
            def stop(player, event):
                retried = sum(c[:2] == (1, 0) for c in player.calls) == 2
                return retried and (event[:2] == (1, 2) if after_challenger else
                                     event[:2] == (1, 1) and event[2] < -20)
            player, result = self.run_script([[10, 8, 0], [10, 20, 5]], stop,
                                            aspiration=True, reason=reason)
            self.assertEqual(result.action, player.actions[int(after_challenger)])
            self.assertEqual(result.score, 20 if after_challenger else 10)
            self.assertEqual(result.score_bound, 'exact')
            self.assertEqual(result.selected_depth, 2)
            self.assertEqual(result.aspiration_fail_highs, 1)

    def test_deadline_at_end_of_failed_window_is_not_completed_depth(self):
        from unittest.mock import patch
        for later, completed in (([-100, -120, -130], False), ([10, 8, 0], True)):
            player = WindowChildren(self.state, replace(self.config, aspiration_enabled=True,
                                    pvs_enabled=True), [[10, 8, 0], later])
            ordered = player._ordered
            def expire_after_children(*args, **kwargs):
                yield from ordered(*args, **kwargs)
                if player._root_progress.depth == 2:
                    player.clock.now = 2
            with patch.object(player, '_ordered', expire_after_children):
                result = player.analyze(self.state, Budget(10**7, 1, clock=player.clock))
            self.assertTrue(result.stopped)
            self.assertEqual(result.completed_depth, 2 if completed else 1)
            self.assertEqual(result.partial_depth, 0 if completed else 2)
            self.assertEqual(result.score, 10)
            self.assertEqual(result.score_bound, 'exact')

    def test_failed_low_visited_upper_bound_alternative_remains_unrefuted(self):
        # Every sibling has returned, but the narrow window has proved only
        # the incumbent losing. Its old optimistic score must not survive.
        player, result = self.run_script([[10, 8, 0], [-MATE + 4, -100, -200]],
            lambda p, e: e[:2] == (1, 0) and sum(c[:2] == (1, 0) for c in p.calls) == 2,
            aspiration=True)
        self.assertEqual(result.action, player.actions[1])
        self.assertEqual(result.selection_source, 'unrefuted_fallback')
        self.assertEqual(result.score, -100)
        self.assertEqual(result.score_bound, 'upper')
        self.assertNotEqual(result.explanation['proof']['status'], 'proven')

    def test_generated_depth_three_positions_match_exhaustive(self):
        import numpy as np
        rng = np.random.default_rng(410041)
        for sample in range(6):
            board = Position.fixture(position({'D4': 1, 'E5': -2, 'G7': -3}))
            for _ in range(sample + 1):
                board, _ = board.move(int(rng.choice(sorted(board.legal()))))
                if board.terminal()[0] != 'ongoing':
                    break
            if board.terminal()[0] != 'ongoing':
                continue
            state = board.storage()
            cfg = replace(self.config, max_depth=3, count_weight=.125,
                          advantage_weight=.3333333333333333)
            expected = exhaustive_minimax(AlphaBetaPlayer().game, state, 3, cfg)[0]
            for pvs, aspiration in product((False, True), repeat=2):
                player = AlphaBetaPlayer(config=replace(cfg, pvs_enabled=pvs,
                                                       aspiration_enabled=aspiration))
                result = player.analyze(state)
                self.assertEqual(result.score, expected)
                self.assert_pv(player.game, state, result, cfg)

    def test_actual_work_caps_respect_selected_move_bounds(self):
        state = Position.fixture(position({'D4': 1, 'E5': -2, 'G7': -3})).storage()
        for pvs, aspiration, cap in product((False, True), (False, True), (100, 500, 1000, 3000, 8000)):
            cfg = replace(self.config, max_depth=3, pvs_enabled=pvs,
                          aspiration_enabled=aspiration, node_limit=cap)
            player = AlphaBetaPlayer(config=cfg)
            result = player.analyze(state)
            self.assertLessEqual(result.work, cap)
            if result.selected_depth:
                child, _ = player.game.getNextState(state, 0, result.action)
                expected = from_table(-exhaustive_minimax(player.game, child,
                    result.selected_depth - 1, cfg, unlimited())[0], 1)
                if result.score_bound == 'exact':
                    self.assertEqual(result.score, expected)
                elif result.score_bound == 'upper':
                    self.assertGreaterEqual(result.score, expected)
                else:
                    self.fail(result.score_bound)


if __name__ == '__main__':
    unittest.main()
