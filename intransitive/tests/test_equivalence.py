"""Independent rules and bounded, reproducible compiled metamorphic gate (#7)."""

from contextlib import contextmanager
import random
import unittest

import numpy as np
from numba import config, njit

from MCTS import MCTS, get_next_best_action_and_canonical_state
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveLogicNumba import Board
from intransitive.IntransitiveSymmetries import (
    transform_action, transform_action_vector, transform_player_vector, transform_state,
)
from intransitive.tests.reference_rules import (
    Position, action_between, map_action, position,
)
from intransitive.tests.test_game import UniformNetwork, search_args


RANDOM_SEEDS = (4, 7, 11)
MCTS_SEEDS = (17, 170)
SAMPLE_PLIES = (0, 7, 19)
# At most 19 captures, each preceded by at most 79 quiet plies, then 80
# quiet plies. Official wins and repetition can only shorten this bound.
MAX_GAME_PLIES = 1600


@contextmanager
def game_diagnostics(seed, reference, state):
    # Unlike subTest inside a while-loop, a failure must stop this rollout;
    # otherwise a failure before the transition would retry the same ply forever.
    try:
        yield
    except Exception as error:
        raise AssertionError(
            f'seed={seed}, ply={reference.ply}, state_hex={state.tobytes().hex()}\n{error}'
        ) from error


@njit
def inspect_compiled(board, state):
    board.copy_state(state, False)
    return (board.valid_moves(0), board.valid_moves(1),
            board.check_end_game(board.get_next_player()),
            board.get_terminal_reason(), board.get_repetition_count(),
            board.get_next_player(), board.get_a1_defender(),
            board.get_no_capture_count(), board.get_history_length(),
            board.get_total_ply())


@njit
def move_compiled(board, state, action, copy):
    board.copy_state(state, copy)
    next_player = board.make_move(action, board.get_next_player())
    return board.get_state(), next_player


def clock_fixture(pieces, clock, player=0, a1_defender=0, ply=None):
    """Distinct synthetic earlier boards isolate clock/precedence thresholds."""
    earlier = []
    for i in range(clock):
        history_piece = [0] * 81
        history_piece[(9 + i) % 81] = 3
        earlier.append(tuple(history_piece))
    return Position.fixture(pieces, player, a1_defender,
                            history=earlier + [pieces], ply=ply)


class IndependentEquivalence(unittest.TestCase):
    def setUp(self):
        self.assertFalse(config.DISABLE_JIT, 'This milestone gate requires compiled Numba')
        self.board = Board()
        self.game = IntransitiveGame()

    def assert_position(self, state, reference):
        # Full bytes cover every history plane/turn, both counters, goals, next
        # player, version, and unused padding, rather than only the piece plane.
        np.testing.assert_array_equal(state, reference.storage())
        self.assertEqual(Position.from_storage(state), reference)
        blue, red, reward, reason, repetitions, player, goal, clock, length, ply = (
            inspect_compiled(self.board, state))
        for p, mask in enumerate((blue, red)):
            self.assertEqual(set(np.flatnonzero(mask)), reference.legal(p))
        expected_reason, expected_reward = reference.terminal()
        np.testing.assert_array_equal(reward, np.asarray(expected_reward, dtype=np.float32))
        self.assertEqual((reason, repetitions, player, goal, clock, length, ply),
                         (expected_reason, reference.repetitions(), reference.player,
                          reference.a1_defender, reference.clock,
                          len(reference.history), reference.ply))
        np.testing.assert_array_equal(state, reference.storage())

    def assert_equivariant(self, reference, actions=None):
        state = reference.storage()
        saved = state.copy()
        self.assert_position(state, reference)
        legal = self.board.valid_moves(reference.player)
        actions = sorted(reference.legal()) if actions is None else actions
        children = []
        for action in actions:
            child_ref, captured = reference.move(action)
            child, next_player = move_compiled(self.board, state, action, False)
            self.assertEqual(next_player, child_ref.player)
            self.assert_position(child, child_ref)
            self.assertEqual(np.count_nonzero(state[:, :, 0]) -
                             np.count_nonzero(child[:, :, 0]), int(captured))
            children.append((action, child, child.copy(), child_ref, captured))
        for symmetry in range(12):
            with self.subTest(symmetry=symmetry):
                transformed = transform_state(state, symmetry)
                transformed_saved = transformed.copy()
                transformed_ref = reference.transform(symmetry)
                self.assert_position(transformed, transformed_ref)
                np.testing.assert_array_equal(
                    transform_action_vector(legal, symmetry),
                    self.board.valid_moves(transformed_ref.player))
                expected_reward = np.asarray(reference.terminal()[1], dtype=np.float32)
                np.testing.assert_array_equal(
                    transform_player_vector(expected_reward, symmetry),
                    np.asarray(transformed_ref.terminal()[1], dtype=np.float32))
                # Prove target coordinates track the player labels as well.
                goal = 80 if reference.player == reference.a1_defender else 0
                transformed_goal = (80 if transformed_ref.player ==
                                    transformed_ref.a1_defender else 0)
                self.assertEqual(transformed_goal, 80 - goal if symmetry >= 6 else goal)
                for action, child, child_saved, child_ref, captured in children:
                    with self.subTest(action=action):
                        mapped = map_action(action, symmetry)
                        self.assertEqual(transform_action(action, symmetry), mapped)
                        expected, mapped_capture = transformed_ref.move(mapped)
                        self.assertEqual(mapped_capture, captured)
                        self.assertEqual(child_ref.transform(symmetry), expected)
                        # Both copy and borrow paths participate in sibling search.
                        actual, next_player = move_compiled(
                            self.board, transformed, mapped, bool(symmetry % 2))
                        self.assertEqual(next_player, expected.player)
                        self.assert_position(actual, expected)
                        self.assertEqual(np.count_nonzero(transformed[:, :, 0]) -
                                         np.count_nonzero(actual[:, :, 0]), int(captured))
                        np.testing.assert_array_equal(
                            transform_state(child, symmetry), actual)
                        np.testing.assert_array_equal(child, child_saved)
                np.testing.assert_array_equal(transformed, transformed_saved)
        np.testing.assert_array_equal(state, saved)

    def test_seeded_reachable_transitions_all_actions_all_symmetries(self):
        sampled, branches, captures = 0, 0, 0
        for seed in RANDOM_SEEDS:
            rng = random.Random(seed)
            reference = Position.initial()
            physical = self.game.getInitBoard()
            retained = []
            sampled_capture = False
            while reference.terminal()[0] == 'ongoing':
                with game_diagnostics(seed, reference, physical):
                    self.assertLess(reference.ply, MAX_GAME_PLIES)
                    self.assert_position(physical, reference)
                    action = rng.choice(sorted(reference.legal()))
                    child_ref, captured = reference.move(action)
                    # Fixed checkpoints plus the first capture parent guarantee
                    # generated coverage of both quiet and capturing branches.
                    if reference.ply in SAMPLE_PLIES or (captured and not sampled_capture):
                        with self.subTest(seed=seed, ply=reference.ply,
                                          state_hex=physical.tobytes().hex()):
                            self.assert_equivariant(reference)
                        sampled += 1
                        branches += len(reference.legal()) * 12
                        sampled_capture |= captured
                    retained.append((physical, physical.copy()))
                    physical, player = self.game.getNextState(physical, reference.player, action)
                    self.assertEqual(player, child_ref.player)
                    self.assert_position(physical, child_ref)
                    captures += int(captured)
                    reference = child_ref
            self.assert_position(physical, reference)
            for state, saved in retained:
                np.testing.assert_array_equal(state, saved)
            print(f'Random seed={seed}: plies={reference.ply}, reason={reference.terminal()[0]}')
        self.assertGreaterEqual(sampled, 9)
        self.assertGreater(captures, 0)
        self.assertTrue(inspect_compiled.nopython_signatures)
        self.assertTrue(move_compiled.nopython_signatures)
        print(f'Generated gate: states={sampled}, transformed branches={branches}, captures={captures}')

    def test_rare_transition_thresholds_and_canonical_i9_defender(self):
        fixtures = []
        for clock in (78, 79):
            # Capture at plies 79/80, preserving total-ply carry at 127 -> 128.
            ref = clock_fixture(position({'B2': 1, 'C2': -2, 'H8': -1}),
                                clock, ply=127)
            fixtures.append((f'capture-clock-{clock}', ref, 'B2', 'C2', 'ongoing', 0))
            ref = clock_fixture(position({'B2': 1, 'H8': -1}), clock)
            fixtures.append((f'quiet-clock-{clock}', ref, 'B2', 'C2',
                             'ongoing' if clock == 78 else 'no-capture limit', clock + 1))
        # Every type can win; occupying the own corner is allowed, with both
        # corner assignments and both labels (including canonical 0 defending I9).
        for piece in (1, 2, 3):
            for defender in (0, 1):
                source, target, own = ('I8', 'I9', 'A1') if defender == 0 else ('A2', 'A1', 'I9')
                ref = clock_fixture(position({source: piece, 'E5': -1, own: 3}),
                                    79, a1_defender=defender)
                fixtures.append((f'corner-{piece}-{defender}', ref, source, target, 'corner', 80))
                # Occupied goals require a legal RPS capture.
                ps = dict(zip((source, target, 'E5'), (piece, -(piece % 3 + 1), -1)))
                ref = clock_fixture(position(ps), 79, a1_defender=defender)
                fixtures.append((f'occupied-goal-{piece}-{defender}', ref, source, target, 'corner', 0))
        blocked = position({'D4': -1, 'E4': -1, 'F4': -1, 'C5': -1,
                            'E5': 1, 'F5': -1, 'D6': -1, 'E6': -1, 'F6': -1})
        fixtures.append(('stalemate-clock', clock_fixture(blocked, 79, player=1),
                         'C5', 'D5', 'stalemate', 80))
        fixtures.append(('last-piece', Position.fixture(position({'B2': 1, 'C2': -2})),
                         'B2', 'C2', 'stalemate', 0))
        for label, reference, source, target, reason, clock in fixtures:
            for relabel in (False, True):
                ref = reference.relabel() if relabel else reference
                action = action_between(source, target)
                with self.subTest(fixture=label, relabel=relabel):
                    child, _ = ref.move(action)
                    self.assertEqual(child.terminal()[0], reason)
                    self.assertEqual(child.clock, clock)
                    self.assert_equivariant(ref, [action])

    def test_repetition_orbits_and_simultaneous_precedence(self):
        base = Position.fixture(position({'B2': 1, 'E3': 3, 'H8': -1}))
        # Each other orbit member appears twice with the CURRENT side to move.
        # A faulty orbit-normalized history would see three occurrences.
        for symmetry in range(1, 12):
            different = base.transform(symmetry).pieces
            self.assertNotEqual(different, base.pieces)
            ref = Position.fixture(base.pieces, history=(different, base.pieces,
                                                        different, base.pieces, base.pieces))
            with self.subTest(orbit=symmetry):
                self.assertEqual(ref.repetitions(), 1)
                self.assertEqual(ref.terminal()[0], 'ongoing')
                self.assert_equivariant(ref, [action_between('B2', 'C2')])

        # A legal cycle: initial occurrence counts, second continues, third ends.
        ref = Position.fixture(position({'B2': 1, 'H8': -1}))
        cycle = [action_between(a, b) for a, b in
                 [('B2', 'C2'), ('H8', 'G8'), ('C2', 'B2'), ('G8', 'H8')]]
        for index, action in enumerate(cycle * 2):
            self.assert_equivariant(ref, [action])
            ref, _ = ref.move(action)
            if index == 3:
                self.assertEqual(ref.repetitions(), 2)
                self.assertEqual(ref.terminal()[0], 'ongoing')
        self.assertEqual(ref.terminal()[0], 'repetition')

        for reason, pieces, player in (
            ('repetition', base.pieces, 0),
            ('corner', position({'I9': 1, 'B2': 1, 'H8': -1}), 1),
            ('stalemate', position({'B2': 1}), 1),
        ):
            # All three thresholds hold; official wins override both draws.
            ref = Position.fixture(pieces, player, history=(pieces,) * 81)
            with self.subTest(precedence=reason):
                self.assertEqual(ref.terminal()[0], reason)
                self.assertGreaterEqual(ref.repetitions(), 3)
                self.assertEqual(ref.clock, 80)
                self.assert_equivariant(ref)

    def test_complete_seeded_mcts_games_match_reference(self):
        for seed in MCTS_SEEDS:
            rng = np.random.default_rng(seed)
            search = MCTS(self.game, UniformNetwork(), search_args())
            search.rng = np.random.default_rng(seed)
            reference = Position.initial()
            physical = self.game.getInitBoard()
            retained = []
            while reference.terminal()[0] == 'ongoing':
                with game_diagnostics(seed, reference, physical):
                    self.assertLess(reference.ply, MAX_GAME_PLIES)
                    self.assert_position(physical, reference)
                    canonical = self.game.getCanonicalForm(physical, reference.player)
                    canonical_ref = reference.relabel() if reference.player else reference
                    self.assert_position(canonical, canonical_ref)
                    saved = canonical.copy()
                    probs, values, full = search.getActionProb(canonical, force_full_search=True)
                    probs = np.asarray(probs)
                    self.assertTrue(full)
                    self.assertTrue(np.isfinite(probs).all())
                    self.assertTrue(np.isfinite(values).all())
                    self.assertTrue((probs >= 0).all())
                    self.assertAlmostEqual(float(probs.sum()), 1)
                    support = set(np.flatnonzero(probs))
                    self.assertTrue(support)
                    self.assertLessEqual(support, reference.legal())
                    action = int(rng.choice(648, p=probs))
                    np.testing.assert_array_equal(canonical, saved)
                    retained.extend(((physical, physical.copy()), (canonical, saved)))
                    child_ref, _ = reference.move(action)
                    physical, player = self.game.getNextState(physical, reference.player, action)
                    self.assertEqual(player, child_ref.player)
                    self.assert_position(physical, child_ref)
                    reference = child_ref
            self.assert_position(physical, reference)
            self.assertTrue(self.game.getGameEnded(physical, reference.player).any())
            for state, saved in retained:
                np.testing.assert_array_equal(state, saved)
            self.assertGreater(len(search.nodes_data), 1)
            print(f'MCTS seed={seed}: plies={reference.ply}, reason={reference.terminal()[0]}')
        self.assertTrue(get_next_best_action_and_canonical_state.nopython_signatures)


if __name__ == '__main__':
    unittest.main()
