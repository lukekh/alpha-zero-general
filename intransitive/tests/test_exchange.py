"""Cyclic static exchange evaluation, quiescence filters and delta pruning.

The series is specified in `heuristics/EXCHANGE.md`. These tests pin the two
properties that specification rests on — the recapturing kind is forced by the
cycle, and the series terminates on the neighbourhood rather than on the cycle —
and then the three application sites, each of which must be exactly inert when
switched off and must never be mistaken for a certificate.
"""
from dataclasses import replace
import unittest

import numpy as np

from intransitive.IntransitiveDisplay import parse_move
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.heuristics import AlphaBetaPlayer, SearchConfig, exhaustive_minimax
from intransitive.heuristics import exchange
from intransitive.heuristics.evaluation import MATE_THRESHOLD
from intransitive.heuristics.material import count_pieces
from intransitive.heuristics.ordering import ordered_actions, warm_ordering
from intransitive.heuristics.position import SearchPosition
from intransitive.tests.test_attribution import random_positions
from intransitive.tests.test_heuristics import position, unlimited


BASE = dict(max_depth=3, time_limit=240., node_limit=50_000_000, ordering_enabled=True)
ROCK, SCISSORS, PAPER = 1, 2, 3
# A full ring around E5: every one of the eight neighbours joins the exchange.
RING = {'E5': -SCISSORS, 'D5': ROCK, 'E6': -PAPER, 'D6': SCISSORS, 'F5': -ROCK,
        'E4': PAPER, 'F6': -SCISSORS, 'D4': ROCK, 'F4': -PAPER}
# The same chain one attacker short, so the mover has the last word.
CHAIN = {k: v for k, v in RING.items() if k != 'F4'}


def square(name):
    return (int(name[1]) - 1) * 9 + (ord(name[0]) - ord('A'))


def board(entries, turn=0):
    state = position(entries, turn=turn)
    return state[:, :, 0].copy(), np.asarray(count_pieces(state), dtype=np.int64)


def swing(entries, move, *, side=0, variable=False, linear=False, weight=100.):
    """Compiled swing/gain, asserted equal to the Python reference every time."""
    pieces, counts = board(entries, turn=side)
    source, target = square(move[:2]), square(move[3:])
    args = (pieces, counts, side, source, target, weight, variable, linear)
    compiled = exchange.exchange_swing(*args)
    reference = exchange.exchange_swing_reference(*args)
    assert compiled == reference, (move, compiled, reference)
    return compiled


class SeriesTests(unittest.TestCase):
    """The exchange series itself, before any valuation or application."""

    def test_an_undefended_capture_wins_the_victim_outright(self):
        self.assertEqual(swing({'D4': ROCK, 'E4': -SCISSORS}, 'D4 E4'), (100., 100.))

    def test_a_defended_capture_is_an_even_trade_under_flat_material(self):
        # Red paper answers the rock that took the scissors: one for one.
        self.assertEqual(swing({'D4': ROCK, 'E4': -SCISSORS, 'F4': -PAPER}, 'D4 E4'),
                         (0., 100.))

    def test_only_the_kind_that_beats_the_occupant_may_recapture(self):
        # A second red scissors cannot take a rock, so nothing answers.
        self.assertEqual(swing({'D4': ROCK, 'E4': -SCISSORS, 'F4': -SCISSORS}, 'D4 E4'),
                         (100., 100.))

    def test_the_series_is_the_forced_cyclic_sequence(self):
        pieces, _ = board(CHAIN)
        series = exchange.exchange_series(pieces, 0, square('D5'), square('E5'))
        self.assertEqual(series, [(1, 1), (0, 0), (1, 2), (0, 1), (1, 0), (0, 2), (1, 1)])
        for step, (owner, kind) in enumerate(series):
            # Sides alternate; the kind removed walks the cycle backwards.
            self.assertEqual(owner, 1 - step % 2)
            self.assertEqual(kind, (1 + 2 * step) % 3)

    def test_a_chain_that_would_cycle_forever_stops_at_the_neighbourhood(self):
        pieces, counts = board(RING)
        series = exchange.exchange_series(pieces, 0, square('D5'), square('E5'))
        # Eight neighbours, eight steps: the kinds would cycle indefinitely.
        self.assertEqual(len(series), 8)
        self.assertEqual(len(series), len(exchange.neighbours(square('E5'))))
        self.assertEqual(exchange.exchange_swing(pieces, counts, 0, square('D5'),
                                                 square('E5'), 100., False, False), (0., 100.))
        # One attacker fewer and the mover, not the defender, has the last word.
        pieces, counts = board(CHAIN)
        self.assertEqual(len(exchange.exchange_series(pieces, 0, square('D5'), square('E5'))), 7)
        self.assertEqual(exchange.exchange_swing(pieces, counts, 0, square('D5'),
                                                 square('E5'), 100., False, False), (100., 100.))

    def test_reinforcements_outside_the_neighbourhood_are_ignored(self):
        """A stated inaccuracy, pinned so it cannot change silently."""
        near = exchange.exchange_series(board(CHAIN)[0], 0, square('D5'), square('E5'))
        far = exchange.exchange_series(board(dict(CHAIN, G5=-PAPER))[0], 0,
                                       square('D5'), square('E5'))
        self.assertEqual(near, far)

    def test_a_quiet_move_and_an_impossible_capture_score_zero(self):
        self.assertEqual(swing({'D4': ROCK, 'E4': -SCISSORS}, 'D4 D5'), (0., 0.))
        self.assertEqual(swing({'D4': ROCK, 'E4': -PAPER}, 'D4 E4'), (0., 0.))
        # Not the side to move's piece.
        self.assertEqual(swing({'D4': ROCK, 'E4': -SCISSORS}, 'D4 E4', side=1), (0., 0.))

    def test_both_sides_may_break_off(self):
        """Under variable material the defender can prefer not to recapture."""
        result, gain = swing({'D4': ROCK, 'E4': -SCISSORS, 'F4': -PAPER}, 'D4 E4', variable=True)
        # Taking the last rock would strip the paper of its prey, so it declines
        # and the swing is exactly the first step.
        self.assertAlmostEqual(result, gain)
        self.assertGreater(result, 0.)


class ValuationTests(unittest.TestCase):
    def test_flat_material_makes_every_capture_win_nothing_or_one_piece(self):
        """Then SEE pruning at threshold zero cannot fire; see EXCHANGE.md."""
        seen = set()
        for state in random_positions(games=4, plies=30, seed=67):
            pieces = state[:, :, 0].copy()
            counts = np.asarray(count_pieces(state), dtype=np.int64)
            side = int(state[:, :, 82:84].flat[1])
            actions = np.flatnonzero(IntransitiveGame().getValidMoves(state, side))
            swings, _, captures = exchange.exchange_swings(pieces, counts, side, actions,
                                                           100., False, False)
            if captures:
                seen.update(np.unique(swings[swings != 0]).tolist())
                self.assertGreaterEqual(swings.min(), 0.)
        self.assertEqual(seen, {100.})

    def test_the_contextual_valuation_changes_the_verdict(self):
        entries, move = {'D4': ROCK, 'E4': -SCISSORS}, 'D4 E4'
        self.assertEqual(swing(entries, move), (100., 100.))
        variable, gain = swing(entries, move, variable=True)
        # The rock's only prey is the piece it is taking: flat says +1 piece,
        # the contextual valuation says the capture costs the rock its value.
        self.assertLess(variable, 0.)
        self.assertLess(gain, 0.)
        self.assertEqual(variable, gain)
        self.assertNotEqual(np.sign(variable), np.sign(swing(entries, move)[0]))

    def test_the_valuation_follows_the_configured_material_mode(self):
        pieces, counts = board(CHAIN)
        args = (pieces, counts, 0, square('D5'), square('E5'), 100.)
        distinct = {exchange.exchange_swing(*args, variable, linear)[0]
                    for variable, linear in ((False, False), (True, False), (True, True))}
        self.assertEqual(len(distinct), 3)

    def test_the_material_edge_is_the_evaluator_s_own_term(self):
        from intransitive.heuristics.evaluation import Evaluator
        game = IntransitiveGame()
        state = position(CHAIN)
        counts = np.asarray(count_pieces(state), dtype=np.int64)
        for variable in (False, True):
            config = SearchConfig(variable_material_enabled=variable, attack_enabled=False,
                                  defence_enabled=False, advantage_weight=0.)
            terms = Evaluator(game, config).explain(state, 0, unlimited(),
                                                    proof={'status': 'unknown'}, diagnostics=False)
            self.assertAlmostEqual(exchange.material_edge(counts, 0, 100., variable, False),
                                   terms['terms']['piece_count'])


class ParityTests(unittest.TestCase):
    def test_compiled_and_reference_agree_exactly_on_a_corpus(self):
        game = IntransitiveGame()
        compared = 0
        for state in random_positions(games=5, plies=35, seed=670):
            pieces = state[:, :, 0].copy()
            counts = np.asarray(count_pieces(state), dtype=np.int64)
            side = int(state[:, :, 82:84].flat[1])
            actions = np.flatnonzero(game.getValidMoves(state, side)).astype(np.int64)
            for variable, linear in ((False, False), (True, False), (True, True)):
                args = (pieces, counts, side, actions, 100., variable, linear)
                fast, reference = exchange.exchange_swings(*args), exchange.exchange_swings_reference(*args)
                np.testing.assert_array_equal(fast[0], reference[0])
                np.testing.assert_array_equal(fast[1], reference[1])
                self.assertEqual(fast[2], reference[2])
                compared += fast[2]
        self.assertGreater(compared, 200)

    def test_the_configuration_switch_selects_the_reference(self):
        state = position(CHAIN)
        actions = np.flatnonzero(IntransitiveGame().getValidMoves(state, 0)).astype(np.int64)
        counts = count_pieces(state)
        outputs = []
        for compiled in (True, False):
            config = SearchConfig(compiled_see_enabled=compiled, variable_material_enabled=True)
            outputs.append(exchange.survey(state[:, :, 0], counts, 0, actions, config))
        np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
        np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
        self.assertEqual(outputs[0][2], outputs[1][2])


class SettingsTests(unittest.TestCase):
    def test_every_switch_is_off_by_default(self):
        config = SearchConfig()
        for name in ('see_ordering_enabled', 'see_quiescence_ordering_enabled',
                     'see_quiescence_pruning_enabled', 'delta_pruning_enabled'):
            self.assertFalse(getattr(config, name), name)
        self.assertTrue(config.compiled_see_enabled)
        self.assertEqual((config.see_threshold, config.delta_margin), (0., 1.))

    def test_bounds_and_types_are_validated(self):
        for bad in (dict(delta_margin=-1.), dict(delta_margin=17.), dict(delta_margin=float('nan')),
                    dict(see_threshold=float('inf')), dict(see_ordering_enabled=1),
                    dict(compiled_see_enabled='yes'), dict(delta_pruning_enabled=None)):
            with self.subTest(**bad):
                with self.assertRaises(ValueError):
                    SearchConfig(**bad)
        SearchConfig(delta_margin=0, see_threshold=-250.5)

    def test_each_switch_enters_the_search_identity(self):
        config = SearchConfig()
        for name in ('see_ordering_enabled', 'see_quiescence_ordering_enabled',
                     'see_quiescence_pruning_enabled', 'delta_pruning_enabled',
                     'compiled_see_enabled'):
            other = replace(config, **{name: not getattr(config, name)})
            self.assertNotEqual(config.identity(), other.identity(), name)
        self.assertNotEqual(config.identity(), replace(config, see_threshold=-1.).identity())
        self.assertNotEqual(config.identity(), replace(config, delta_margin=2.).identity())

    def test_stored_configurations_load_with_everything_off(self):
        from pathlib import Path
        for path in sorted(Path(__file__).parents[1].joinpath('heuristics/configs').glob('*.json')):
            if path.name.startswith('tuning-'):
                continue  # module-scale genomes, read by heuristics.tuning
            if path.name.endswith('.record.json'):
                continue  # experiment records: provenance and decision, not a config
            with self.subTest(config=path.name):
                loaded = SearchConfig.from_file(path)
                self.assertFalse(loaded.see_ordering_enabled or loaded.delta_pruning_enabled
                                 or loaded.see_quiescence_pruning_enabled
                                 or loaded.see_quiescence_ordering_enabled)


class InertWhenOffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()
        cls.states = random_positions(games=3, plies=40, seed=671)[:10]

    def test_values_set_but_not_enabled_never_reach_the_search(self):
        for state in self.states:
            plain = AlphaBetaPlayer(self.game, SearchConfig(**BASE)).analyze(state)
            dressed = AlphaBetaPlayer(self.game, SearchConfig(
                see_threshold=-500., delta_margin=0., compiled_see_enabled=False, **BASE)).analyze(state)
            with self.subTest(state=id(state)):
                self.assertEqual((plain.action, plain.score, plain.nodes, plain.work),
                                 (dressed.action, dressed.score, dressed.nodes, dressed.work))

    def test_quiescence_filters_are_inert_without_quiescence(self):
        config = SearchConfig(see_quiescence_pruning_enabled=True, delta_pruning_enabled=True,
                              see_quiescence_ordering_enabled=True, variable_material_enabled=True, **BASE)
        for state in self.states[:4]:
            plain = AlphaBetaPlayer(self.game, SearchConfig(variable_material_enabled=True, **BASE)).analyze(state)
            filtered = AlphaBetaPlayer(self.game, config).analyze(state)
            self.assertEqual((plain.action, plain.score, plain.nodes), (filtered.action, filtered.score, filtered.nodes))
            self.assertFalse(any(filtered.ordering['see_effective'].values()))


class OrderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warm_ordering()
        cls.game = IntransitiveGame()

    def test_the_exchange_key_prefers_the_capture_that_keeps_material(self):
        # Blue rock may take either red scissors; only F5 is answered by a paper.
        state = position({'D4': ROCK, 'E4': -SCISSORS, 'F4': ROCK, 'F5': -SCISSORS, 'G6': -PAPER})
        actions = np.array([parse_move('F4 F5'), parse_move('D4 E4')], dtype=np.int64)
        pieces = state[:, :, 0].copy()
        counts = np.asarray(count_pieces(state), dtype=np.int64)
        swings = exchange.exchange_swings(pieces, counts, 0, actions, 100., False, False)[0]
        self.assertEqual(list(swings), [0., 100.])
        args = (pieces, actions, 0, 80, -1, np.full(648, -np.inf),
                np.full(2, -1, dtype=np.int64), np.zeros(648, dtype=np.int64), False)
        for kernel in (ordered_actions, ordered_actions.py_func):
            np.testing.assert_array_equal(kernel(*args, None, swings), actions[::-1])
            # An explicitly preferred move still outranks the exchange key.
            preferred = list(args)
            preferred[4] = int(actions[0])
            self.assertEqual(kernel(*preferred, None, swings)[0], actions[0])

    def test_every_legal_move_survives_on_every_ordering_path(self):
        state = position(dict(RING, A1=ROCK, I9=-PAPER))
        legal = set(np.flatnonzero(self.game.getValidMoves(state, 0)).tolist())
        outputs = []
        for compiled in (False, True):
            for enhanced in (False, True):
                for mvv in (False, True):
                    for compact in (False, True):
                        config = SearchConfig(see_ordering_enabled=True, mvv_lva_enabled=mvv,
                                              variable_material_enabled=True, ordering_enabled=enhanced,
                                              compiled_ordering_enabled=compiled)
                        player = AlphaBetaPlayer(self.game, config)
                        player._prepare()
                        subject = SearchPosition(state) if compact else state
                        order = [a for a, _ in player._ordered(subject, 0, None, unlimited())]
                        self.assertEqual(set(order), legal)
                        self.assertEqual(len(order), len(legal))
                        outputs.append((compiled, enhanced, mvv, compact, order))
        for compiled, enhanced, mvv, compact, order in outputs:
            expected = next(o for c, e, m, k, o in outputs if (e, m) == (enhanced, mvv) and (c, k) == (True, True))
            self.assertEqual(order, expected, (compiled, enhanced, mvv, compact))

    def test_an_immediate_win_still_outranks_every_exchange(self):
        state = position({'H8': PAPER, 'D4': ROCK, 'E4': -SCISSORS, 'F6': -PAPER, 'A2': -ROCK})
        config = SearchConfig(see_ordering_enabled=True, variable_material_enabled=True)
        player = AlphaBetaPlayer(self.game, config)
        player._prepare()
        self.assertEqual(next(player._ordered(state, 0, None, unlimited()))[0], parse_move('H8 I9'))

    def test_ordering_alone_never_changes_a_completed_score(self):
        state = position({'D4': ROCK, 'F4': SCISSORS, 'C6': PAPER, 'B2': PAPER,
                          'E4': -SCISSORS, 'G4': -PAPER, 'D6': -ROCK})
        before = state.tobytes()
        for variable in (False, True):
            config = SearchConfig(variable_material_enabled=variable, max_depth=2, proof_nodes=0,
                                  time_limit=120., node_limit=10**9, pvs_enabled=True)
            expected, _ = exhaustive_minimax(self.game, state, 2, config, unlimited())
            for enabled in (False, True):
                result = AlphaBetaPlayer(self.game, replace(config, see_ordering_enabled=enabled)).analyze(state)
                self.assertAlmostEqual(result.score, expected)
                self.assertEqual(result.completed_depth, 2)
                if enabled:
                    self.assertGreater(result.ordering['see_captures'], 0)
        self.assertEqual(state.tobytes(), before)

    def test_first_move_cutoff_rate_is_reported(self):
        result = AlphaBetaPlayer(self.game, SearchConfig(**BASE)).analyze(self.game.getInitBoard())
        self.assertGreater(result.ordering['cutoffs'], 0)
        self.assertLessEqual(result.ordering['first_cutoffs'], result.ordering['cutoffs'])


class QuiescenceFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()
        cls.states = random_positions(games=3, plies=55, seed=672)[:10]

    def settings(self, **extra):
        return SearchConfig(quiescence_enabled=True, variable_material_enabled=True, **BASE, **extra)

    def test_losing_captures_are_skipped_and_counted(self):
        plain = self.settings()
        pruned = self.settings(see_quiescence_pruning_enabled=True)
        skips = captures = baseline = 0
        for state in self.states:
            before = AlphaBetaPlayer(self.game, plain).analyze(state)
            after = AlphaBetaPlayer(self.game, pruned).analyze(state)
            baseline += before.selective['quiescence_captures']
            captures += after.selective['quiescence_captures']
            skips += after.selective['quiescence_see_skips']
            self.assertTrue(after.ordering['see_effective']['quiescence_pruning'])
        self.assertGreater(skips, 0)
        self.assertLess(captures, baseline)

    def test_delta_pruning_fires_and_is_counted(self):
        skips = 0
        for state in self.states:
            result = AlphaBetaPlayer(self.game, self.settings(
                delta_pruning_enabled=True, delta_margin=0.)).analyze(state)
            skips += result.selective['quiescence_delta_skips']
            self.assertEqual(result.ordering['delta_allowance'], 0.)
        self.assertGreater(skips, 0)

    def test_the_margin_is_stated_in_evaluator_units(self):
        config = self.settings(delta_pruning_enabled=True)
        allowance = exchange.delta_allowance(config)
        advantage = abs(config.advantage_weight) * (max(config.predator_zero_bonus,
                                                        config.predator_scarcity_bonus) + config.prey_bonus)
        self.assertAlmostEqual(allowance, advantage + 3*abs(config.attack_weight)
                               + 4*abs(config.defence_weight))
        self.assertAlmostEqual(exchange.delta_allowance(replace(config, delta_margin=2.)), 2*allowance)
        # Disabled modules contribute nothing; a wider margin prunes no more.
        quiet = replace(config, attack_enabled=False, defence_enabled=False)
        self.assertAlmostEqual(exchange.delta_allowance(quiet), advantage)

    def test_ordering_the_captures_leaves_the_capture_set_alone(self):
        for state in self.states[:5]:
            plain = AlphaBetaPlayer(self.game, self.settings()).analyze(state)
            sorted_ = AlphaBetaPlayer(self.game, self.settings(see_quiescence_ordering_enabled=True)).analyze(state)
            self.assertTrue(sorted_.ordering['see_effective']['quiescence_ordering'])
            self.assertGreater(sorted_.ordering['see_nodes'], 0)
            # Ordering may cut off sooner, never search more captures.
            self.assertLessEqual(sorted_.selective['quiescence_captures'],
                                 plain.selective['quiescence_captures'])

    def test_a_capture_onto_the_goal_corner_is_never_skipped(self):
        """Every other capture is skipped; the one that wins on the spot is not."""
        entries = {'H8': PAPER, 'I9': -ROCK, 'D4': ROCK, 'E4': -SCISSORS, 'F4': -PAPER,
                   'A5': SCISSORS, 'B7': PAPER, 'G2': -SCISSORS, 'C8': -PAPER}
        config = replace(self.settings(see_quiescence_pruning_enabled=True,
                                       delta_pruning_enabled=True, delta_margin=0.),
                         see_threshold=10.**6, proof_nodes=0)
        player = AlphaBetaPlayer(self.game, config)
        player._prepare()
        subject = SearchPosition(position(entries, turn=0))
        player._material_counts = subject.counts
        value, line = player._quiesce(subject, 0, -np.inf, np.inf, 0, unlimited(), 8)
        self.assertGreater(value, MATE_THRESHOLD)
        self.assertEqual(line[0], parse_move('H8 I9'))
        self.assertEqual(player._selective_stats['quiescence_captures'], 1)
        self.assertGreater(player._selective_stats['quiescence_see_skips'], 0)

    def test_pruning_is_refused_when_a_reply_could_stalemate(self):
        pieces = position({'A1': ROCK, 'B1': -SCISSORS})[:, :, 0]
        self.assertTrue(exchange.stalemate_safe(np.ascontiguousarray(
            position(dict(RING))[:, :, 0]), 0))
        self.assertFalse(exchange.stalemate_safe(np.ascontiguousarray(pieces), 0))

    def test_results_stay_selective_and_the_board_is_restored(self):
        state = position(CHAIN)
        before = state.tobytes()
        config = self.settings(see_quiescence_pruning_enabled=True, delta_pruning_enabled=True,
                               see_quiescence_ordering_enabled=True)
        for cap in (0, 50, 10**9):
            result = AlphaBetaPlayer(self.game, replace(config, node_limit=cap)).analyze(state)
            self.assertTrue(self.game.getValidMoves(state, 0)[result.action])
            self.assertEqual(state.tobytes(), before)
            if result.score_bound is not None:
                self.assertTrue(result.score_bound.startswith('selective_'))
            self.assertEqual(result.explanation['proof']['status'], 'unknown')


if __name__ == '__main__':
    unittest.main()
