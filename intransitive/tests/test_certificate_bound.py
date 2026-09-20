"""The certificate promoted to a search bound, and the race/horizon tests.

A cutoff is a proof entering the tree, so the questions here are narrow and
hard: does the gate ever hide a certificate the full argument would have found,
is the distance the search reports the distance that was certified, is the
entry it leaves in the table a bound rather than an exact score, and is the
race test admissible against an exhaustive check rather than against intuition.
"""
from dataclasses import replace
from math import inf
import unittest

import numpy as np

from intransitive.IntransitiveConstants import NO_CAPTURE_LIMIT
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.heuristics import AlphaBetaPlayer
from intransitive.heuristics.budget import Budget, BudgetExpired
from intransitive.heuristics.clear_run import NO_RUN, NO_SIDE, certify, race_gate
from intransitive.heuristics.config import SearchConfig
from intransitive.heuristics.evaluation import MATE, MATE_THRESHOLD
from intransitive.heuristics.kernels import no_capture_in_horizon, no_terminal_win_in_horizon
from intransitive.heuristics.position import SearchPosition
from intransitive.heuristics.search import certificate, from_table, position_key, to_table
from intransitive.tests.test_attribution import random_positions
from intransitive.tests.test_heuristics import position, unlimited

# Material-only weights, no route modules and no bounded proof: the certificate
# and the race tests are the only things left that can decide a node.
ISOLATED = dict(attack_enabled=False, defence_enabled=False, advantage_weight=25.,
                proof_depth=0, proof_nodes=0, node_limit=10**9, time_limit=600.)

# Blue paper on F6 runs to I9; red cannot reach A1 first and cannot intercept.
BLUE_TO_LOSE = position({'F6': 3, 'C3': -2, 'A3': -1, 'I1': 1, 'B7': 2})
# The mirror: the side to move holds the run.
BLUE_TO_WIN = position({'D4': -3, 'G7': 1, 'H2': 2, 'I5': 3, 'B8': -1})
# A certified seven-ply run that the existing board guard would happily prune
# through: every piece is more than three steps from its own corner, no capture
# is available, and both sides have plenty of moves.
DISTANT_RUN = position({'E5': 1, 'B2': 1, 'B8': 1, 'H2': 1,
                        'A5': -1, 'E1': -1, 'A7': -1, 'G1': -1})
# Six same-type pieces a side, so no capture exists at all, none within four
# steps of its corner, and enough spacious pieces for the stalemate argument.
QUIET_RACE = position({'B2': 1, 'D2': 1, 'B4': 1, 'D4': 1, 'B6': 1, 'D6': 1,
                       'F4': -1, 'H4': -1, 'F6': -1, 'H6': -1, 'F8': -1, 'H8': -1})


def clock_left(state):
    meta = state[:, :, 82:84].ravel()
    return max(0, NO_CAPTURE_LIMIT - max(int(meta[3]), max(0, int(meta[4]) - 1)))


def sparse_positions(count, seed, game=None):
    """Legal sparse endgames, where a forced run is something that happens.

    Random legal play reaches a decided corner race only rarely, and a cutoff
    corpus assembled from it is mostly empty. These boards are not a sample of
    play; they are a sample of the positions the certificate is about.
    """
    game = game or IntransitiveGame()
    generator, states = np.random.default_rng(seed), []
    while len(states) < count:
        squares = generator.choice(81, size=int(generator.integers(3, 7)), replace=False)
        entries = {f'{chr(ord("A") + int(square) % 9)}{int(square) // 9 + 1}':
                   int(generator.integers(1, 4)) * (1 if index % 2 == 0 else -1)
                   for index, square in enumerate(squares)}
        if not (any(code > 0 for code in entries.values())
                and any(code < 0 for code in entries.values())):
            continue
        try:
            state = position(entries, turn=int(generator.integers(0, 2)))
        except ValueError:
            continue  # both winning corners held at once is not a legal board
        if game.getGameEnded(state, int(state[:, :, 82:84].ravel()[1])).any():
            continue
        states.append(state)
    return states


def sweep(state, limit=20):
    """What `certify` says for both sides, ignoring the gate."""
    meta = state[:, :, 82:84].ravel()
    turn, a1 = int(meta[1]), int(meta[2])
    return [certify(state[:, :, 0], side, turn, a1, clock_left(state), limit)
            for side in (0, 1)]


class GateTests(unittest.TestCase):
    """The gate exists to make an interior probe affordable.

    What has to hold is one-sided: it may refuse a position `certify` would
    also refuse, but never one that really holds a run.
    """

    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()
        cls.states = random_positions(games=14, plies=120, seed=73)

    def test_the_gate_never_hides_a_certificate_on_random_boards(self):
        generator = np.random.default_rng(4111)
        passes = certificates = 0
        for _ in range(20000):
            pieces = np.zeros((9, 9), dtype=np.int8)
            for _ in range(int(generator.integers(2, 10))):
                square = int(generator.integers(0, 81))
                kind = int(generator.integers(1, 4))
                pieces[square // 9, square % 9] = kind if generator.integers(0, 2) else -kind
            turn, a1 = int(generator.integers(0, 2)), int(generator.integers(0, 2))
            allowance, limit = int(generator.integers(0, 81)), int(generator.integers(1, 21))
            gate = int(race_gate(pieces, turn, a1, allowance, limit))
            passes += gate != NO_SIDE
            for side in (0, 1):
                if certify(pieces, side, turn, a1, allowance, limit) != NO_RUN:
                    certificates += 1
                    self.assertEqual(gate, side, 'the gate refused a real certificate')
        self.assertGreater(certificates, 0, 'the corpus never produced a certificate')
        self.assertGreater(passes, 0, 'the gate refused every position')

    def test_the_gate_never_hides_a_certificate_on_reachable_positions(self):
        passes = certificates = 0
        for state in self.states:
            meta = state[:, :, 82:84].ravel()
            gate = int(race_gate(state[:, :, 0], int(meta[1]), int(meta[2]),
                                 clock_left(state), 20))
            passes += gate != NO_SIDE
            for side, plies in enumerate(sweep(state)):
                if plies != NO_RUN:
                    certificates += 1
                    self.assertEqual(gate, side)
        self.assertGreater(certificates, 0, 'the corpus never exercised the certificate')
        # The whole cost argument is that reachable play rarely has a runner in
        # range. A gate that let most nodes through would not be worth having.
        self.assertLess(passes, len(self.states) // 10)

    def test_the_gated_certificate_agrees_with_the_full_sweep(self):
        config = SearchConfig(certificate_enabled=True, **ISOLATED)
        for state in self.states:
            if self.game.getGameEnded(state, int(state[:, :, 82:84].ravel()[1])).any():
                continue
            turn = int(state[:, :, 82:84].ravel()[1])
            plies = sweep(state)
            expected = None
            for side in (turn, 1 - turn):
                if plies[side] != NO_RUN:
                    score = MATE - plies[side]
                    expected = score if side == turn else -score
                    break
            self.assertEqual(certificate(state, config, unlimited()), expected)

    def test_a_held_corner_and_a_covered_corner_are_refused_without_certifying(self):
        # Blue paper two steps from I9 certifies, until the corner is occupied,
        # and again until a red piece can stand on it first.
        state = position({'G7': 3, 'A5': -1})
        self.assertEqual(int(race_gate(state[:, :, 0], 0, 0, 80, 20)), 0)
        self.assertNotEqual(certify(state[:, :, 0], 0, 0, 0, 80, 20), NO_RUN)
        held = position({'G7': 3, 'A5': -1, 'I9': -1})
        self.assertEqual(int(race_gate(held[:, :, 0], 0, 0, 80, 20)), NO_SIDE)
        covered = position({'G7': 3, 'A5': -1, 'H8': -1})
        self.assertEqual(int(race_gate(covered[:, :, 0], 0, 0, 80, 20)), NO_SIDE)

    def test_the_gate_respects_the_move_limit_and_the_draw_clock(self):
        state = position({'G7': 3, 'A5': -1})
        self.assertEqual(int(race_gate(state[:, :, 0], 0, 0, 80, 1)), NO_SIDE)
        self.assertEqual(int(race_gate(state[:, :, 0], 0, 0, 3, 20)), NO_SIDE)
        self.assertEqual(int(race_gate(state[:, :, 0], 0, 0, 4, 20)), 0)

    def test_every_probe_is_charged(self):
        config = SearchConfig(certificate_enabled=True, **ISOLATED)
        # A gate refusal costs one board pass; the full argument costs a pass
        # per piece on top of it.
        refused = position({'E5': 1, 'E6': -1, 'I9': -1})
        budget = unlimited()
        self.assertIsNone(certificate(refused, config, budget))
        self.assertEqual(budget.work, 81)
        state = position({'G7': 3, 'A5': -1})
        budget = unlimited()
        self.assertIsNotNone(certificate(state, config, budget))
        self.assertEqual(budget.work, 81 + 81 * 2)
        # A charge that cannot be paid stops the search rather than running free.
        with self.assertRaises(BudgetExpired):
            certificate(state, config, Budget(40, 3600))


class CutoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()
        cls.config = SearchConfig(max_depth=3, certificate_cutoff_enabled=True,
                                  certificate_cutoff_min_depth=1, **ISOLATED)
        AlphaBetaPlayer(cls.game, cls.config)._prepare()

    def node(self, state, config, depth, ply):
        """`_search` on one interior node, with the root machinery out of the way."""
        player = AlphaBetaPlayer(self.game, config)
        player._prepare()
        compact = SearchPosition(state, modelling_draws=self.game.board.modelling_draws)
        value, line = player._search(compact, depth, -inf, inf, ply, unlimited())
        return player, value, line

    def test_the_leaf_certificate_no_longer_crashes_the_search(self):
        # `prove` published no `pv` for a certificate, so the first leaf that
        # certified raised KeyError and the option could never be used at all.
        state = position({'G7': 3, 'A5': -1})
        config = SearchConfig(max_depth=2, certificate_enabled=True, proof_depth=2,
                              proof_nodes=64, attack_enabled=False, defence_enabled=False,
                              node_limit=10**9, time_limit=600.)
        result = AlphaBetaPlayer(self.game, config).analyze(state)
        self.assertGreater(result.score, MATE_THRESHOLD)
        self.assertIn(result.action, list(map(int, np.flatnonzero(
            self.game.getValidMoves(state, 0)))))

    def test_a_cutoff_reports_the_certified_distance_from_that_node(self):
        for state in (BLUE_TO_WIN, BLUE_TO_LOSE):
            run = certificate(state, self.config, unlimited())
            self.assertIsNotNone(run)
            for ply in (1, 2, 5):
                with self.subTest(ply=ply, run=run):
                    _, value, line = self.node(state, self.config, 3, ply)
                    # Neither shorter nor longer than the run that was proved.
                    self.assertEqual(value, from_table(run, ply))
                    self.assertEqual(int(MATE - abs(value)) - ply,
                                     int(MATE - abs(run)))
                    self.assertEqual(line, [])

    def test_a_cutoff_is_stored_as_a_bound_and_never_as_an_exact_score(self):
        for state, bound in ((BLUE_TO_WIN, 'lower'), (BLUE_TO_LOSE, 'upper')):
            with self.subTest(bound=bound):
                player, value, _ = self.node(state, self.config, 3, 2)
                entry = player.table[(position_key(SearchPosition(
                    state, modelling_draws=self.game.board.modelling_draws)), 3)]
                self.assertEqual(entry.bound, bound)
                self.assertEqual(entry.score, to_table(value, 2))
                self.assertEqual(entry.pv, ())

    def test_the_root_keeps_searching_and_returns_a_legal_move(self):
        # A score without a move is useless at the root, so the root never cuts.
        result = AlphaBetaPlayer(self.game, self.config).analyze(BLUE_TO_WIN)
        legal = list(map(int, np.flatnonzero(self.game.getValidMoves(BLUE_TO_WIN, 0))))
        self.assertIn(result.action, legal)
        self.assertGreater(result.score, MATE_THRESHOLD)

    def test_a_hypothetical_subtree_never_consults_the_certificate(self):
        # The side-to-move flip in a null-move probe never happened, and the
        # certificate reads the turn, so it must not be asked there.
        player = AlphaBetaPlayer(self.game, self.config)
        player._prepare()
        player._null_context = True
        compact = SearchPosition(BLUE_TO_LOSE, modelling_draws=self.game.board.modelling_draws)
        value, _ = player._search(compact, 3, -inf, inf, 1, unlimited())
        self.assertEqual(player._certificate_stats['probes'], 0)
        self.assertLess(abs(value), MATE_THRESHOLD)

    def test_a_cutoff_result_survives_a_full_width_search(self):
        """Alpha-beta returns the true minimax value at a fixed depth.

        So a full-width re-search to the claimed distance either confirms a
        cutoff-derived mate score or refutes it; there is no third answer.
        """
        checked = 0
        for state in sparse_positions(220, 2027, self.game):
            result = AlphaBetaPlayer(self.game, self.config).analyze(state)
            if (not result.certificate['cutoffs'] or result.score is None
                    or abs(result.score) <= MATE_THRESHOLD):
                continue
            plies = int(MATE - abs(result.score))
            if plies > 4:
                continue  # full width past four plies is not worth the minutes
            reference = AlphaBetaPlayer(self.game, replace(
                self.config, certificate_cutoff_enabled=False, max_depth=plies)).analyze(state)
            if reference.stopped or reference.score is None:
                continue
            checked += 1
            with self.subTest(plies=plies, score=result.score):
                signed = reference.score * (1. if result.score > 0 else -1.)
                self.assertGreater(signed, MATE_THRESHOLD,
                                   'a certificate cutoff claimed an unforced result')
                self.assertLessEqual(MATE - abs(reference.score), plies,
                                     'the real result is slower than the cutoff claimed')
        self.assertGreater(checked, 0, 'the corpus never exercised a cutoff')

    def test_a_cutoff_never_contradicts_the_same_search_without_it(self):
        """The cutoff may resolve more positions; it may not resolve them differently."""
        without = replace(self.config, certificate_cutoff_enabled=False,
                          certificate_enabled=True)
        resolved = 0
        for state in sparse_positions(150, 6161, self.game):
            plain = AlphaBetaPlayer(self.game, without).analyze(state)
            cutoff = AlphaBetaPlayer(self.game, replace(
                self.config, certificate_enabled=True)).analyze(state)
            if plain.score is None or abs(plain.score) <= MATE_THRESHOLD:
                continue
            resolved += 1
            with self.subTest(score=plain.score):
                self.assertGreater(cutoff.score * (1. if plain.score > 0 else -1.),
                                   MATE_THRESHOLD)
        self.assertGreater(resolved, 0, 'the corpus never resolved a position')


class GuardTests(unittest.TestCase):
    """The guard matters exactly where the search cannot see the decision.

    `DISTANT_RUN` holds a certified seven-ply run with every piece more than
    three steps from its corner, so `selective.guarded` permits pruning through
    it and — with the leaf certificate off — the children score as ordinary
    quiet positions. That is the case the certificate is needed for.
    """

    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()
        cls.base = dict(nmp_enabled=True, futility_enabled=True, lmr_enabled=True,
                        ordering_enabled=True, selective_evaluator_enabled=True, **ISOLATED)

    def pair(self, **overrides):
        settings = dict(self.base, **overrides)
        return (AlphaBetaPlayer(self.game, SearchConfig(
                    certificate_guard_enabled=True, **settings)).analyze(DISTANT_RUN),
                AlphaBetaPlayer(self.game, SearchConfig(**settings)).analyze(DISTANT_RUN))

    def test_a_certified_node_is_refused_null_move_and_futility_eligibility(self):
        # Narrow windows are what make a node a pruning candidate at all.
        guarded, plain = self.pair(max_depth=4, pvs_enabled=True)
        self.assertGreater(guarded.certificate['certified'], 0,
                           'the fixture never certified anything')
        self.assertGreater(guarded.certificate['guards'], 0)
        self.assertLess(guarded.selective['static_evaluations'],
                        plain.selective['static_evaluations'])
        self.assertLessEqual(guarded.selective['futility_pruned'],
                             plain.selective['futility_pruned'])
        self.assertEqual(guarded.score, plain.score)

    def test_no_child_of_a_certified_node_gets_a_shallower_look(self):
        # Reductions only reach a node's children when PVS is not probing them.
        guarded, plain = self.pair(max_depth=5, pvs_enabled=False)
        self.assertGreater(guarded.certificate['unreduced'], 0)
        # Every prevented reduction is one the unguarded search performed.
        self.assertEqual(plain.selective['lmr_reduced'] - guarded.selective['lmr_reduced'],
                         guarded.certificate['unreduced'])
        self.assertEqual(guarded.score, plain.score)

    def test_the_guard_alone_never_produces_a_cutoff(self):
        result = AlphaBetaPlayer(self.game, SearchConfig(
            certificate_guard_enabled=True, max_depth=4, pvs_enabled=True,
            **self.base)).analyze(DISTANT_RUN)
        self.assertGreater(result.certificate['certified'], 0)
        self.assertEqual(result.certificate['cutoffs'], 0)


class RaceHorizonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()

    def test_no_capture_in_horizon_is_admissible(self):
        """Exhaustive check: when it says no capture, no line finds one."""
        generator = np.random.default_rng(5150)
        claimed = refused = 0
        for _ in range(400):
            pieces = np.zeros((9, 9), dtype=np.int8)
            for _ in range(int(generator.integers(3, 8))):
                square = int(generator.integers(0, 81))
                kind = int(generator.integers(1, 4))
                pieces[square // 9, square % 9] = kind if generator.integers(0, 2) else -kind
            if not pieces.any() or not (pieces > 0).any() or not (pieces < 0).any():
                continue
            state = position({f'{chr(ord("A") + x)}{y + 1}': int(pieces[y, x])
                              for y in range(9) for x in range(9) if pieces[y, x]})
            compact = SearchPosition(state, modelling_draws=False)
            if compact.terminal()[1] != 'ongoing':
                continue
            depth = int(generator.integers(1, 4))
            quiet, _ = no_capture_in_horizon(compact.pieces, depth)
            if not quiet:
                refused += 1
                continue
            claimed += 1
            self.assertFalse(self.capture_within(compact, depth),
                             'a capture was available inside a horizon called quiet')
        self.assertGreater(claimed, 0, 'the corpus never called a horizon quiet')
        self.assertGreater(refused, 0, 'the corpus never refused a horizon')

    def capture_within(self, compact, depth):
        if depth == 0:
            return False
        for action in map(int, compact.legal()):
            captured = compact.push(action)
            try:
                if captured or self.capture_within(compact, depth - 1):
                    return True
            finally:
                compact.pop()
        return False

    def test_a_near_goal_race_is_never_called_quiet(self):
        # A runner three steps from its corner can finish inside a four-ply
        # horizon, and the terminal test says so.
        racing = position({'F6': 3, 'D6': 1, 'B2': -1, 'D2': -2, 'B6': 1, 'F2': -1})
        self.assertFalse(no_terminal_win_in_horizon(racing[:, :, 0], 0, 0, 5))
        self.assertTrue(no_terminal_win_in_horizon(QUIET_RACE[:, :, 0], 0, 0, 3))

    def test_a_blocked_runner_is_still_not_quiet(self):
        # Blocking is a fact about this board, not about the horizon: the
        # blocker can step aside, so the free-board bound still applies and the
        # horizon test still refuses.
        blocked = position({'H8': 3, 'H9': 1, 'I8': 1, 'A1': -1, 'B3': -2,
                            'D5': 1, 'F5': -1})
        self.assertFalse(no_terminal_win_in_horizon(blocked[:, :, 0], 0, 0, 3))

    def test_a_runner_captured_en_route_is_not_a_quiet_branch(self):
        # Red scissors two steps from the blue paper runner: the capture pair
        # is inside the horizon, so the branch is not quiet whatever the
        # runner's route looks like.
        hunted = position({'E5': 3, 'G7': -2, 'A1': 1, 'C1': 1, 'I9': -1, 'G1': -1})
        self.assertFalse(no_capture_in_horizon(hunted[:, :, 0], 3)[0])
        self.assertTrue(no_capture_in_horizon(hunted[:, :, 0], 1)[0])

    def test_the_race_reduction_only_relaxes_the_index_guess(self):
        base = dict(max_depth=5, lmr_enabled=True, ordering_enabled=True, **ISOLATED)
        plain = AlphaBetaPlayer(self.game, SearchConfig(**base)).analyze(QUIET_RACE)
        raced = AlphaBetaPlayer(self.game, SearchConfig(
            race_reduction_enabled=True, **base)).analyze(QUIET_RACE)
        self.assertGreater(raced.certificate['race_quiet'], 0)
        self.assertGreater(raced.certificate['race_reductions'], 0)
        self.assertGreater(raced.selective['lmr_reduced'], plain.selective['lmr_reduced'])
        self.assertLess(raced.nodes, plain.nodes)
        self.assertEqual(raced.score, plain.score)
        self.assertEqual(plain.certificate['race_probes'], 0)

    def test_a_race_reduction_is_a_selective_result_and_not_a_proof(self):
        config = SearchConfig(max_depth=3, lmr_enabled=True, race_reduction_enabled=True,
                              ordering_enabled=True, certificate_enabled=True, **ISOLATED)
        result = AlphaBetaPlayer(self.game, config).analyze(BLUE_TO_WIN)
        self.assertEqual(result.explanation['proof']['status'], 'unknown')
        self.assertEqual(result.explanation['proof']['reason'],
                         'selective search is not a certificate')
        self.assertTrue(result.score_bound.startswith('selective_'))

    def test_a_certificate_cutoff_is_not_a_selective_result(self):
        config = SearchConfig(max_depth=3, certificate_cutoff_enabled=True,
                              certificate_guard_enabled=True, certificate_enabled=True,
                              **ISOLATED)
        result = AlphaBetaPlayer(self.game, config).analyze(BLUE_TO_WIN)
        self.assertGreater(result.score, MATE_THRESHOLD)
        self.assertEqual(result.explanation['proof']['status'], 'proven')
        self.assertEqual(result.score_bound, 'exact')


class ConfigurationTests(unittest.TestCase):
    def test_the_new_options_default_to_off(self):
        config = SearchConfig()
        self.assertFalse(config.certificate_cutoff_enabled)
        self.assertFalse(config.certificate_guard_enabled)
        self.assertFalse(config.race_reduction_enabled)
        self.assertEqual(config.certificate_cutoff_min_depth, 2)
        for name in ('certificate_cutoff_enabled', 'certificate_cutoff_min_depth',
                     'certificate_guard_enabled', 'race_reduction_enabled'):
            self.assertIn(name, config.to_dict())
            self.assertIn(name, config.identity())

    def test_an_inert_race_flag_is_rejected(self):
        with self.assertRaises(ValueError):
            SearchConfig(race_reduction_enabled=True)
        SearchConfig(race_reduction_enabled=True, lmr_enabled=True)

    def test_the_probe_depth_threshold_is_bounded(self):
        for value in (0, 33, 2.0, True):
            with self.assertRaises(ValueError):
                SearchConfig(certificate_cutoff_min_depth=value)


if __name__ == '__main__':
    unittest.main()
