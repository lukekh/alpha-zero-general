"""Issue #68: razoring, reverse futility, move-count and mate-distance pruning.

The first three are heuristic cutoffs and must declare themselves as such. The
fourth only narrows the window to bounds the true value already satisfies, so
it must leave every result — including a certificate — exactly as it found it.

Measured evidence for the margins and for each technique's firing rate lives in
`intransitive/benchmarks/shallow_pruning/`.
"""
from dataclasses import replace
from math import inf, isfinite, nextafter
import unittest
from unittest.mock import patch

import numpy as np

from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.heuristics import AlphaBetaPlayer, SearchConfig, exhaustive_minimax
from intransitive.heuristics import selective
from intransitive.heuristics.budget import Budget, BudgetExpired
from intransitive.heuristics.evaluation import MATE, MATE_THRESHOLD
from intransitive.heuristics.position import SearchPosition
from intransitive.heuristics.search import prove, selective_mode_early
from intransitive.IntransitiveConstants import action_destination
from intransitive.IntransitiveDisplay import parse_move
from intransitive.tests.test_attribution import random_positions
from intransitive.tests.test_heuristics import position

# The scale `selective.supported()` accepts without an experimental opt-in, and
# PVS, without which issue #66 showed the shared non-PV guard never opens.
BASE = dict(count_weight=100., advantage_weight=25., attack_enabled=False,
            defence_enabled=False, node_limit=10**9, time_limit=120.,
            proof_depth=0, proof_nodes=0, pvs_enabled=True, ordering_enabled=True)
MODES = dict(razoring=dict(quiescence_enabled=True, razoring_enabled=True),
             reverse_futility=dict(reverse_futility_enabled=True),
             move_count=dict(move_count_pruning_enabled=True))


def quiet_state():
    return position({'A2': 1, 'B2': 2, 'A3': 3, 'B3': 1, 'D4': 2,
                     'H7': -1, 'I7': -2, 'H8': -3, 'I8': -1})


def budget():
    return Budget(10**9, 120.)


def config(**kwargs):
    return SearchConfig(**dict(BASE, **kwargs))


class SettingsTests(unittest.TestCase):
    def test_every_technique_is_off_by_default(self):
        defaults = SearchConfig()
        for name in ('razoring_enabled', 'reverse_futility_enabled',
                     'move_count_pruning_enabled', 'mate_distance_pruning_enabled'):
            self.assertFalse(getattr(defaults, name), name)
        self.assertFalse(defaults.selective_pruning())

    def test_ranges_and_types_are_validated(self):
        for bad in (dict(razoring_max_depth=0), dict(razoring_max_depth=5),
                    dict(reverse_futility_max_depth=0), dict(reverse_futility_max_depth=7),
                    dict(move_count_max_depth=0), dict(move_count_max_depth=9),
                    dict(move_count_base=0), dict(move_count_base=65),
                    dict(move_count_base=8.), dict(razoring_margin=0.),
                    dict(razoring_margin=65.), dict(razoring_margin=float('nan')),
                    dict(razoring_margin=True), dict(reverse_futility_margin=0.),
                    dict(reverse_futility_margin=float('inf')),
                    dict(razoring_enabled=1), dict(mate_distance_pruning_enabled='yes')):
            with self.subTest(**bad), self.assertRaises(ValueError):
                SearchConfig(**bad)

    def test_margins_admit_the_fractions_the_adopted_genome_needs(self):
        # `allowance()` sums weight ceilings, so the measured safe multiplier on
        # the route genome is below one. A floor of one would strand it.
        self.assertEqual(SearchConfig(razoring_margin=.45).razoring_margin, .45)
        self.assertEqual(SearchConfig(reverse_futility_margin=.2).reverse_futility_margin, .2)

    def test_razoring_refuses_to_run_without_quiescence(self):
        with self.assertRaisesRegex(ValueError, 'quiescence'):
            SearchConfig(razoring_enabled=True)
        SearchConfig(razoring_enabled=True, quiescence_enabled=True)

    def test_defaults_cover_the_measured_swing_on_the_supported_scale(self):
        # benchmarks/shallow_pruning/evidence/calibration.json: 1.476 and 0.729
        # allowances per ply, over 480 positions. Larger is the safe direction.
        self.assertGreaterEqual(SearchConfig().razoring_margin, 1.4763888888888892)
        self.assertGreaterEqual(SearchConfig().reverse_futility_margin, .7291666666666666)

    def test_heuristic_members_declare_themselves_selective(self):
        for name, settings in MODES.items():
            with self.subTest(name):
                self.assertTrue(config(**settings).selective_pruning())
                self.assertTrue(selective_mode_early(config(**settings)))
        # Mate-distance pruning is value preserving, so it is neither.
        mate = config(mate_distance_pruning_enabled=True)
        self.assertFalse(mate.selective_pruning())
        self.assertFalse(selective_mode_early(mate))

    def test_they_require_the_compact_backend(self):
        game = IntransitiveGame()
        for name, settings in MODES.items():
            with self.subTest(name), self.assertRaises(ValueError):
                AlphaBetaPlayer(game, config(**settings), use_compact=False).analyze(
                    game.getInitBoard())

    def test_the_native_backend_rejects_them_instead_of_ignoring_them(self):
        from intransitive.heuristics.tuning import Genome
        genome = Genome.from_genes({}, backend='rust')
        self.assertTrue(genome.native_arguments(SearchConfig()))
        for settings in list(MODES.values()) + [dict(mate_distance_pruning_enabled=True)]:
            with self.subTest(**settings), self.assertRaisesRegex(ValueError, 'Rust'):
                genome.native_arguments(SearchConfig(**settings))

    def test_exhaustive_label_pipelines_reject_them(self):
        from intransitive.teacher_learning import teacher_config
        for settings in MODES.values():
            search = dict(SearchConfig(**settings, max_depth=6).to_dict())
            with self.subTest(**settings), self.assertRaisesRegex(ValueError, 'selective'):
                teacher_config({'teacher': {'search': search}})


class InertWhenOffTests(unittest.TestCase):
    """Values set but not enabled must not reach the search."""

    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()
        cls.states = random_positions(games=3, plies=40, seed=680922)[:8]

    def test_disabled_settings_change_nothing(self):
        dressed = dict(razoring_max_depth=4, razoring_margin=1.,
                       reverse_futility_max_depth=6, reverse_futility_margin=1.,
                       move_count_max_depth=8, move_count_base=1)
        for state in self.states:
            plain = AlphaBetaPlayer(self.game, config(max_depth=3)).analyze(state)
            other = AlphaBetaPlayer(self.game, config(max_depth=3, **dressed)).analyze(state)
            with self.subTest(state=id(state)):
                self.assertEqual(plain.action, other.action)
                self.assertEqual(plain.score, other.score)
                self.assertEqual(plain.nodes, other.nodes)
                self.assertEqual(plain.score_bound, 'exact')


class FiringTests(unittest.TestCase):
    """Each technique must be shown firing, per issue #66's standard."""

    def search(self, alpha, depth=1, **settings):
        player = AlphaBetaPlayer(config=config(**settings))
        player._prepare()
        state = quiet_state()
        p = SearchPosition(state)
        score, line = player._search(p, depth, alpha, nextafter(alpha, inf), 1, budget())
        # Every push must have been undone, on every path out of the search.
        np.testing.assert_array_equal(p.export(), state)
        self.assertEqual(p.stack, [])
        return score, player._selective_stats

    def test_razoring_drops_to_quiescence_and_reports_the_fail_low(self):
        score, stats = self.search(9000., **MODES['razoring'])
        self.assertGreater(stats['razoring_eligible'], 0)
        self.assertGreater(stats['razoring_applied'], 0)
        self.assertLessEqual(score, 9000.)
        self.assertTrue(isfinite(score))

    def test_reverse_futility_returns_the_bound_without_a_probe(self):
        score, stats = self.search(-9000., **MODES['reverse_futility'])
        self.assertGreater(stats['reverse_futility_eligible'], 0)
        self.assertGreater(stats['reverse_futility_pruned'], 0)
        # No hypothetical pass is made, so no null subtree is ever entered.
        self.assertEqual(stats['null_nodes'], 0)
        self.assertEqual(stats['nmp_attempts'], 0)
        self.assertGreaterEqual(score, -9000.)

    def test_move_count_skips_late_quiet_children_outright(self):
        score, stats = self.search(500., move_count_pruning_enabled=True, move_count_base=1)
        self.assertGreater(stats['move_count_eligible'], 0)
        self.assertGreater(stats['move_count_pruned'], 0)
        # Skipped, not reduced: no reduction and no re-search was recorded.
        self.assertEqual(stats['lmr_reduced'], 0)
        self.assertTrue(isfinite(score))

    def test_mate_distance_cuts_on_a_corpus_of_certified_tactics(self):
        from intransitive.tests.test_tactics import CASES
        from intransitive.tests.tactical_oracle import load_case
        # BASE leaves the bounded proof off deliberately. With it on, a
        # certified tactic is settled by the leaf oracle at depth one and the
        # narrowing is never reached; the mates this can act on are the ones
        # the tree has to find for itself.
        cfg = config(max_depth=4, mate_distance_pruning_enabled=True)
        pruned = sum(AlphaBetaPlayer(config=cfg).analyze(load_case(case)[0].storage())
                     .selective['mate_distance_pruned'] for case in CASES[:25])
        self.assertGreater(pruned, 0)


class SafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()

    def all_on(self, **kwargs):
        return config(quiescence_enabled=True, razoring_enabled=True,
                      reverse_futility_enabled=True, move_count_pruning_enabled=True,
                      mate_distance_pruning_enabled=True, **kwargs)

    def test_a_node_always_searches_at_least_one_move(self):
        """Stalemate loses, so an emptied node is not a draw — it is a bug.

        Both guards are forced wide open: every board is eligible and every
        move counts as quiet, at the lowest legal move-count base. A returned
        principal variation is the evidence, since one is recorded only when a
        child search has actually come back.
        """
        player = AlphaBetaPlayer(config=config(
            max_depth=2, move_count_pruning_enabled=True, move_count_base=1,
            futility_enabled=True))
        player._prepare()
        state = quiet_state()
        p = SearchPosition(state)
        with patch.object(selective, 'quiet', return_value=True), \
             patch.object(selective, 'guarded', return_value=True):
            score, line = player._search(p, 2, 0., nextafter(0., inf), 1, budget())
        stats = player._selective_stats
        self.assertGreater(stats['move_count_pruned'] + stats['futility_pruned'], 0)
        self.assertTrue(line, 'every child was pruned away')
        self.assertTrue(SearchPosition(state).legal().size)
        self.assertIn(line[0], [int(a) for a in SearchPosition(state).legal()])
        self.assertTrue(isfinite(score))
        self.assertNotEqual(score, 0.)
        np.testing.assert_array_equal(p.export(), state)

    def test_a_stalemated_node_is_a_loss_not_a_draw(self):
        for entries, turn in (({'I8': -1, 'H8': -2, 'I7': -3}, 0), ({'B2': 1, 'A2': 2}, 1)):
            state = position(entries, turn=turn)
            self.assertEqual(SearchPosition(state).terminal()[1], 'stalemate')
            player = AlphaBetaPlayer(config=self.all_on())
            player._prepare()
            with self.subTest(entries=tuple(entries)):
                for ply in (0, 3):
                    score, _ = player._search(SearchPosition(state), 3, -inf, inf, ply, budget())
                    self.assertEqual(score, -MATE + ply)

    def test_the_root_and_full_window_nodes_are_never_pruned(self):
        """Only the last iteration's root has the full requested remaining depth.

        Its window is also the widest one in the search, so a call to the board
        guard at that depth would mean a root node had become eligible.
        """
        player = AlphaBetaPlayer(config=self.all_on(max_depth=3))
        with patch.object(selective, 'guarded', wraps=selective.guarded) as guard:
            player.analyze(quiet_state())
        self.assertTrue(guard.call_args_list, 'the guard was never consulted at all')
        self.assertFalse([call for call in guard.call_args_list if call.args[1] == 3],
                         'a root node reached the board guard')

    def test_state_is_restored_when_a_pruned_search_is_interrupted(self):
        player = AlphaBetaPlayer(config=self.all_on(max_depth=3))
        player._prepare()
        state = quiet_state()
        p = SearchPosition(state)
        real = player._quiesce

        def fail(*args, **kwargs):
            raise BudgetExpired('work')

        with patch.object(player, '_quiesce', side_effect=fail), \
             self.assertRaises(BudgetExpired):
            player._search(p, 2, 9000., nextafter(9000., inf), 1, budget())
        np.testing.assert_array_equal(p.export(), state)
        self.assertEqual(p.stack, [])
        self.assertFalse(player._selective_disabled)
        self.assertFalse(player._null_context)
        self.assertTrue(player.use_table)

    def test_selective_results_are_bound_qualified_and_namespaced(self):
        from intransitive.heuristics.search import position_key
        alphas = dict(razoring=9000., reverse_futility=-9000., move_count=500.)
        for name, settings in MODES.items():
            player = AlphaBetaPlayer(config=config(max_depth=3, **settings))
            player._prepare()
            p = SearchPosition(quiet_state())
            key = position_key(p)
            player._search(p, 1, alphas[name], nextafter(alphas[name], inf), 1, budget())
            with self.subTest(name):
                self.assertTrue(player.table)
                self.assertTrue(all(stored.startswith(b'selective-v1\0')
                                    for stored, _ in player.table))
                entry = player.table.get((b'selective-v1\0' + key, 1))
                if name == 'razoring':
                    self.assertEqual(entry.bound, 'upper')
                elif name == 'reverse_futility':
                    self.assertEqual(entry.bound, 'lower')
        # An unpruned search of the same node writes under the plain key, so no
        # selective bound can be read back as an ordinary exact value.
        plain = AlphaBetaPlayer(config=config(max_depth=3))
        plain._prepare()
        p = SearchPosition(quiet_state())
        plain._search(p, 1, 500., nextafter(500., inf), 1, budget())
        self.assertTrue(plain.table)
        self.assertFalse([stored for stored, _ in plain.table
                          if stored.startswith(b'selective-v1\0')])

    def test_the_bounded_proof_and_certificate_are_untouched(self):
        state = position({'H8': 1, 'C3': -2})
        expected = prove(self.game, SearchPosition(state),
                         config(proof_depth=2, proof_nodes=64), budget())
        self.assertEqual(expected['status'], 'proven')
        configurations = {name: config(proof_depth=2, proof_nodes=64, **settings)
                          for name, settings in MODES.items()}
        configurations['all'] = self.all_on(proof_depth=2, proof_nodes=64)
        for name, cfg in configurations.items():
            with self.subTest(name):
                self.assertEqual(prove(self.game, SearchPosition(state), cfg, budget()),
                                 expected)

    def test_heuristic_members_withdraw_the_certificate_and_mate_distance_does_not(self):
        state = position({'H8': 1, 'C3': -2})
        for name, settings in MODES.items():
            result = AlphaBetaPlayer(config=config(max_depth=3, proof_depth=2,
                                                   proof_nodes=64, **settings)).analyze(state)
            with self.subTest(name):
                self.assertTrue(result.score_bound.startswith('selective_'))
                self.assertEqual(result.explanation['proof']['status'], 'unknown')
        mate = AlphaBetaPlayer(config=config(max_depth=3, proof_depth=2, proof_nodes=64,
                                             mate_distance_pruning_enabled=True)).analyze(state)
        self.assertEqual(mate.score_bound, 'exact')
        self.assertEqual(mate.explanation['proof']['status'], 'proven')


class MateDistanceTests(unittest.TestCase):
    """It may only change how quickly a mate is found, never which."""

    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()

    def test_it_agrees_with_the_bounded_proof_oracle_on_certified_tactics(self):
        from intransitive.tests.test_tactics import CASES
        from intransitive.tests.tactical_oracle import load_case
        for case in CASES[:25]:
            reference, expected = load_case(case)
            state = reference.storage()
            results = {}
            for enabled in (False, True):
                cfg = config(max_depth=4, proof_depth=2, proof_nodes=64,
                             mate_distance_pruning_enabled=enabled)
                results[enabled] = AlphaBetaPlayer(config=cfg).analyze(state)
                self.assertEqual(prove(self.game, SearchPosition(state), cfg, budget()),
                                 prove(self.game, SearchPosition(state),
                                       replace(cfg, mate_distance_pruning_enabled=False),
                                       budget()))
            with self.subTest(case=case['id']):
                self.assertEqual(results[True].action, int(expected))
                self.assertEqual(results[True].action, results[False].action)
                self.assertEqual(results[True].score, results[False].score)
                self.assertEqual(results[True].stop_reason, results[False].stop_reason)

    def test_it_matches_an_exhaustive_reference_on_ordinary_positions(self):
        cfg = config(max_depth=2, mate_distance_pruning_enabled=True)
        for state in random_positions(games=2, plies=30, seed=680923)[:6]:
            expected, _ = exhaustive_minimax(self.game, state, 2, cfg, budget())
            with self.subTest(state=id(state)):
                self.assertAlmostEqual(AlphaBetaPlayer(config=cfg).analyze(state).score,
                                       expected)

    def test_it_is_inert_under_every_pvs_and_aspiration_combination(self):
        # The narrowing touches the same alpha/beta that PVS reads to build a
        # null window and that aspiration re-searches on, so the interaction
        # is checked rather than argued.
        states = random_positions(games=1, plies=30, seed=680925)[:4]
        for pvs in (False, True):
            for aspiration in (False, True):
                cfg = config(max_depth=3, proof_depth=2, proof_nodes=64,
                             aspiration_enabled=aspiration)
                cfg = replace(cfg, pvs_enabled=pvs)
                for state in states:
                    off = AlphaBetaPlayer(config=cfg).analyze(state)
                    on = AlphaBetaPlayer(config=replace(
                        cfg, mate_distance_pruning_enabled=True)).analyze(state)
                    with self.subTest(pvs=pvs, aspiration=aspiration, state=id(state)):
                        self.assertEqual(off.action, on.action)
                        self.assertEqual(off.score, on.score)
                        self.assertEqual(off.stop_reason, on.stop_reason)
                        self.assertEqual(off.score_bound, on.score_bound)

    def test_a_forced_win_is_still_found_when_the_window_is_narrowed(self):
        state = position({'H8': 3, 'A1': 1, 'E5': -1, 'D4': -2}, turn=0)
        result = AlphaBetaPlayer(config=config(
            max_depth=5, mate_distance_pruning_enabled=True)).analyze(state)
        self.assertGreater(result.score, MATE_THRESHOLD)


class MarginTests(unittest.TestCase):
    def test_every_margin_is_a_multiple_of_the_shared_allowance(self):
        cfg = config()
        unit = selective.allowance(cfg)
        self.assertEqual(unit, cfg.count_weight / 2 + cfg.advantage_weight)
        for depth in (1, 2, 3):
            self.assertAlmostEqual(selective.margin(cfg, depth),
                                   depth * cfg.futility_margin * unit)
            self.assertAlmostEqual(selective.razor_margin(cfg, depth),
                                   depth * cfg.razoring_margin * unit)
            self.assertAlmostEqual(selective.reverse_margin(cfg, depth),
                                   depth * cfg.reverse_futility_margin * unit)

    def test_signed_and_route_weights_keep_the_allowance_positive(self):
        evolved = config(selective_evaluator_enabled=True, count_weight=-100.,
                         advantage_weight=-25., attack_enabled=True, attack_weight=-20.,
                         defence_enabled=True, defence_weight=-30.)
        self.assertGreater(selective.allowance(evolved), 0)
        self.assertGreater(selective.razor_margin(evolved, 1), 0)
        self.assertGreater(selective.reverse_margin(evolved, 1), 0)

    def test_variable_material_needs_the_position_it_is_priced_from(self):
        variable = config(selective_evaluator_enabled=True, variable_material_enabled=True)
        with self.assertRaises(ValueError):
            selective.razor_margin(variable, 1)
        p = SearchPosition(quiet_state())
        self.assertGreater(selective.reverse_margin(variable, 1, p), 0)


class QuietReviewTests(unittest.TestCase):
    """Issue #68 required `selective.quiet()` to be reviewed before reuse."""

    def test_a_square_on_a_shortest_enemy_route_is_not_quiet(self):
        p = SearchPosition(quiet_state())
        # Blue runs to I9, so A1 is the corner Red is running at.
        self.assertTrue(selective.blocks(0, 80, 0))
        self.assertTrue(selective.blocks(40, 80, 0))
        self.assertFalse(selective.blocks(8, 80, 0))
        # Vacating a blocking square and occupying one are both defence.
        self.assertFalse(selective.quiet(p, parse_move('B2 C3')))
        self.assertFalse(selective.quiet(p, parse_move('A2 A1')))
        self.assertTrue(selective.quiet(p, parse_move('A3 A4')))

    def test_captures_threat_creation_and_the_fastest_runner_stay_excluded(self):
        p = SearchPosition(quiet_state())
        self.assertFalse(selective.quiet(p, parse_move('D4 E5')))  # fastest runner
        for action in map(int, p.legal()):
            if selective.quiet(p, action):
                x, y = action_destination(action)
                self.assertFalse(p.pieces[y, x], 'a capture was called quiet')
                self.assertGreater(selective.distance(y * 9 + x, 80), 3)

    def test_the_review_only_ever_refuses_more(self):
        """The reviewed predicate must be a subset of the issue #60 one."""
        def original(position, action):
            pieces = position.pieces.ravel()
            source = action // 8
            x, y = action_destination(action)
            target = y * 9 + x
            if pieces[target]:
                return False
            goal = 80 if position.side == position.a1 else 0
            own = [int(s) for s in np.flatnonzero(pieces)
                   if int(pieces[s] < 0) == position.side]
            if selective.distance(source, goal) <= min(selective.distance(s, goal)
                                                       for s in own):
                return False
            for square in np.flatnonzero(pieces):
                if (int(pieces[square] < 0) != position.side
                        and min(selective.distance(int(square), source),
                                selective.distance(int(square), target)) <= 2):
                    return False
            return selective.distance(target, goal) > 3

        refused = 0
        for state in random_positions(games=2, plies=40, seed=680924)[:10]:
            p = SearchPosition(state)
            if p.terminal()[1] != 'ongoing':
                continue
            for action in map(int, p.legal()):
                new, old = selective.quiet(p, action), original(p, action)
                self.assertFalse(new and not old, 'the review permitted a new move')
                refused += old and not new
        self.assertGreater(refused, 0, 'the review changed nothing at all')


if __name__ == '__main__':
    unittest.main()
