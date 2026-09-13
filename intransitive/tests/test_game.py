"""Canonical action equivalence and the actual shared compiled MCTS path."""

import contextlib
import io
from types import SimpleNamespace
import unittest

import numpy as np

from MCTS import MCTS, get_next_best_action_and_canonical_state
from intransitive.IntransitiveConstants import E, N, S, W, encode_action
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveLogicNumba import Board, validate_state
from intransitive.tests.test_draws import (
    load_history, noncapture_actions, play, sparse_position,
)


def relabel(state):
    """Independent fixed-square reference, including only populated history."""
    out = state.copy()
    length = int(state[:, :, 32].flat[4])
    out[:, :, :length + 1] *= -1
    meta = out[:, :, 32]
    for offset in [1, 2] + list(range(10, 10 + length)):
        meta.flat[offset] = 1 - meta.flat[offset]
    return out


def with_ply(state, ply):
    out = state.copy()
    for i in range(5):
        out[:, :, 32].flat[5 + i] = ply % 128
        ply //= 128
    return out


class UniformNetwork:
    def predict(self, board, valid):
        policy = valid.astype(np.float32)
        return policy / policy.sum(), np.zeros(2, dtype=np.float32)


def search_args():
    return SimpleNamespace(
        numMCTSSims=8, prob_fullMCTS=1.0, ratio_fullMCTS=1,
        forced_playouts=False, universes=1, no_mem_optim=False,
        cpuct=1.0, fpu=0.0,
    )


class CanonicalGameIntegration(unittest.TestCase):
    def setUp(self):
        self.game = IntransitiveGame()

    def compiled_child(self, canonical, action):
        valid = self.game.getValidMoves(canonical, 0)
        policy = np.zeros(648, dtype=np.float32)
        policy[action] = 1
        chosen, child, player = get_next_best_action_and_canonical_state(
            np.zeros(2, dtype=np.float32), valid, policy, 1,
            np.full(648, -42.0), np.zeros(648, dtype=np.int64),
            0.0, 1.0, self.game.board, canonical, False, 0, 0.0, 17,
        )
        self.assertEqual(chosen, action)
        self.assertEqual(player, 1)
        validate_state(child)
        return child

    def test_adapter_contract_and_owned_identity_hook(self):
        state = self.game.getInitBoard()
        saved = state.copy()
        self.assertEqual(self.game.getBoardSize(), (9, 9, 33))
        self.assertEqual(self.game.getActionSize(), 648)
        self.assertEqual(self.game.getNumberOfPlayers(), 2)
        self.assertEqual(self.game.num_players, 2)
        self.assertEqual(self.game.getRound(state), 0)
        self.assertEqual([self.game.getScore(state, p) for p in (0, 1)], [10, 10])
        np.testing.assert_array_equal(self.game.getGameEnded(state, 0), [0, 0])
        valid = self.game.getValidMoves(state, 0)
        policy = valid.astype(np.float32) / valid.sum()
        triples = self.game.getSymmetries(state, policy.tolist(), valid)
        self.assertEqual(len(triples), 1)
        b, p, v = triples[0]
        np.testing.assert_array_equal(b, state)
        np.testing.assert_allclose(p, policy)
        np.testing.assert_array_equal(v, valid)
        self.assertEqual(p.dtype, np.float32)
        b[:] = 0
        p[:] = 0
        v[:] = False
        np.testing.assert_array_equal(state, saved)
        self.assertTrue(valid.any())
        self.assertAlmostEqual(float(policy.sum()), 1)
        canonical = self.game.getCanonicalForm(state, 0)
        canonical[:] = 0
        np.testing.assert_array_equal(state, saved)
        with self.assertRaises(ValueError):
            self.game.getSymmetries(state, [1], valid)

    def test_swap_complete_history_involution_and_borrowed_ownership(self):
        board = Board()
        actions = noncapture_actions()
        for length in (1, 5, 31):
            load_history(board, [sparse_position()])
            play(board, actions[:length - 1])
            for defender in (0, 1):
                original = with_ply(board.get_state(), 129)
                original[:, :, 32].flat[2] = defender
                for copy in (False, True):
                    board.copy_state(original, copy)
                    before = original.copy()
                    board.swap_players(1)
                    swapped = board.get_state()
                    np.testing.assert_array_equal(swapped, relabel(original))
                    validate_state(swapped)
                    board.swap_players(1)
                    np.testing.assert_array_equal(board.get_state(), original)
                    np.testing.assert_array_equal(original, before)
                    np.testing.assert_array_equal(swapped, relabel(original))
                    for invalid in (-1, 2, 0.5):
                        with self.assertRaises(ValueError):
                            board.swap_players(invalid)
                    np.testing.assert_array_equal(board.get_state(), original)

    def test_all_legal_action_round_trips_both_players_and_goals(self):
        board = Board()
        cycle = [encode_action(1, 4, N), encode_action(4, 7, W),
                 encode_action(1, 5, S), encode_action(3, 7, E)]
        play(board, cycle)
        for player in (0, 1):
            if player:
                board.make_move(cycle[0], 0)
            for defender in (0, 1):
                original = board.get_state()
                original[:, :, 32].flat[2] = defender
                saved = original.copy()
                canonical = self.game.getCanonicalForm(original, player)
                np.testing.assert_array_equal(canonical, relabel(original) if player else original)
                np.testing.assert_array_equal(
                    self.game.getValidMoves(canonical, 0),
                    self.game.getValidMoves(original, player))
                for action in np.flatnonzero(self.game.getValidMoves(canonical, 0)):
                    physical_child, next_player = self.game.getNextState(original, player, int(action))
                    expected = relabel(physical_child) if next_player else physical_child
                    np.testing.assert_array_equal(self.compiled_child(canonical, int(action)), expected)
                np.testing.assert_array_equal(original, saved)
        self.assertTrue(get_next_best_action_and_canonical_state.nopython_signatures)

    def test_capture_crosses_127_without_resetting_round(self):
        board = Board()
        pieces = sparse_position()
        pieces[1, 2] = -2
        previous = pieces.copy()
        previous[1, 1], previous[2, 1] = 0, 1
        state = load_history(board, [previous, pieces], first_player=1)
        state = with_ply(state, 127)
        action = encode_action(1, 1, E)
        for player in (0, 1):
            original = relabel(state) if player else state
            canonical = self.game.getCanonicalForm(original, player)
            physical, next_player = self.game.getNextState(original, player, action)
            self.assertEqual(self.game.getRound(original), 127)
            self.assertEqual(self.game.getRound(physical), 128)
            self.assertEqual(physical[:, :, 32].flat[3], 0)
            self.assertEqual(physical[:, :, 32].flat[4], 1)
            expected = relabel(physical) if next_player else physical
            np.testing.assert_array_equal(self.compiled_child(canonical, action), expected)
            self.assertEqual(self.game.getRound(expected), 128)

    def test_search_keys_cover_history_counters_and_goals(self):
        board = Board()
        pieces = sparse_position()
        different = pieces.copy()
        different[1, 1] = 2
        base = load_history(board, [different, different, pieces])
        history = base.copy()
        history[1, 1, 1] = 3
        goal = base.copy()
        goal[:, :, 32].flat[2] = 1
        turn = base.copy()
        for offset in (1, 10, 11, 12):
            turn[:, :, 32].flat[offset] = 1 - turn[:, :, 32].flat[offset]
        shorter = with_ply(load_history(board, [pieces]), 2)
        states = [base, history, goal, turn, shorter, with_ply(base, 130)]
        keys = [self.game.stringRepresentation(s) for s in states]
        self.assertEqual(len(set(keys)), len(states))
        for state, key in zip(states, keys):
            self.assertEqual(key, state.tobytes(order="C"))
            self.assertEqual(len(key), 2673)
            board.copy_state(state, False)
            self.assertEqual(board.get_repetition_count(), 1)
        self.assertEqual(self.game.stringRepresentation(np.asfortranarray(base)), keys[0])

    def test_player_zero_defending_i9_can_win_at_a1(self):
        board = Board()
        pieces = sparse_position()
        pieces[1, 1], pieces[1, 0] = 0, 1
        original = load_history(board, [pieces], a1_defender=1)
        child = self.compiled_child(original, encode_action(0, 1, S))
        np.testing.assert_array_equal(self.game.getGameEnded(child, 0), [-1, 1])
        self.assertFalse(self.game.getValidMoves(child, 0).any())
        search = MCTS(self.game, UniformNetwork(), search_args())
        # The mover wins; MCTS rolls the child's loss back to the root player.
        search.search(original)
        key = self.game.stringRepresentation(original)
        search.nodes_data[key][2][:] = 0
        search.nodes_data[key][2][encode_action(0, 1, S)] = 1
        np.testing.assert_array_equal(search.search(original), [1, -1])

    def test_compiled_draw_branches_keep_history_and_reward_perspective(self):
        board = Board()
        cycle = [encode_action(1, 1, E), encode_action(7, 7, W),
                 encode_action(2, 1, W), encode_action(6, 7, E)]
        for kind in ("repetition", "no-capture limit"):
            load_history(board, [sparse_position()])
            actions = cycle * 2 if kind == "repetition" else noncapture_actions()
            play(board, actions[:-1])
            original = board.get_state()
            canonical = self.game.getCanonicalForm(original, 1)
            saved = canonical.copy()
            self.assertFalse(self.game.getGameEnded(canonical, 0).any())
            drawn = self.compiled_child(canonical, actions[-1])
            drawn_saved = drawn.copy()
            result = self.game.getGameEnded(drawn, 0)
            np.testing.assert_array_equal(result, np.full(2, 1e-4, dtype=np.float32))
            self.assertTrue(result.any())
            self.assertFalse(self.game.getValidMoves(drawn, 0).any())
            self.assertEqual(self.game.board.get_terminal_reason(), kind)
            if kind == "repetition":
                sibling = self.compiled_child(canonical, encode_action(6, 7, W))
                self.assertFalse(self.game.getGameEnded(sibling, 0).any())
            np.testing.assert_array_equal(canonical, saved)
            np.testing.assert_array_equal(drawn, drawn_saved)
            search = MCTS(self.game, UniformNetwork(), search_args())
            np.testing.assert_array_equal(search.search(drawn), result)

    def test_mcts_selects_playable_physical_actions_and_cleans_by_total_ply(self):
        original = self.game.getInitBoard()
        original, player = self.game.getNextState(original, 0, encode_action(1, 4, N))
        original = with_ply(original, 129)
        canonical = self.game.getCanonicalForm(original, player)
        saved = canonical.copy()
        search = MCTS(self.game, UniformNetwork(), search_args())
        # An actual earlier root must be removed even though the capture clock is 1.
        old = self.game.getInitBoard()
        search.search(old)
        old_key = self.game.stringRepresentation(old)
        probs, q, full = search.getActionProb(canonical, force_full_search=True)
        self.assertTrue(full)
        self.assertTrue(np.isfinite(probs).all())
        self.assertAlmostEqual(sum(probs), 1)
        self.assertEqual(len(q), 2)
        self.assertEqual(search.last_cleaning, 129)
        self.assertNotIn(old_key, search.nodes_data)
        self.assertGreater(len(search.nodes_data), 1)
        physical_mask = self.game.getValidMoves(original, player)
        self.assertFalse(np.asarray(probs)[~physical_mask].any())
        action = int(np.argmax(probs))
        child, next_player = self.game.getNextState(original, player, action)
        self.assertEqual(next_player, 0)
        self.assertEqual(self.game.getRound(child), 130)
        np.testing.assert_array_equal(canonical, saved)

    def test_display_and_invalid_public_inputs(self):
        state = self.game.getInitBoard()
        saved = state.copy()
        self.assertEqual(self.game.moveToString(encode_action(1, 4, E), 0), "B5->C5")
        self.assertEqual(self.game.moveToString(encode_action(0, 0, S), 1), "A1->off-board")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.game.printBoard(state)
        self.assertIn("A  B  C  D  E  F  G  H  I", output.getvalue())
        self.assertIn("A1 defended by Blue", output.getvalue())
        for player in (1, -1, 0.5):
            with self.assertRaises(ValueError):
                self.game.getCanonicalForm(state, player)
        with self.assertRaises(ValueError):
            self.game.getNextState(state, 0, -1)
        with self.assertRaises(ValueError):
            self.game.getGameEnded(state, 1)
        np.testing.assert_array_equal(state, saved)


if __name__ == "__main__":
    unittest.main()
