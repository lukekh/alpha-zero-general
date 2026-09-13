"""Physical display, human retries, tactical priorities, and real Arena games."""

import base64
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

import numpy as np

from Arena import Arena
from intransitive.IntransitiveConstants import action_stays_on_board
from intransitive.IntransitiveDisplay import (
    format_board, move_to_str, parse_move, player_colour,
)
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveLogicNumba import Board
from intransitive.IntransitivePlayers import GreedyPlayer, HumanPlayer, RandomPlayer
from intransitive.tests.test_draws import load_history, noncapture_actions, play, sparse_position
from intransitive.tests.test_rules import load_position


class DisplayAndPlayers(unittest.TestCase):
    def setUp(self):
        self.game = IntransitiveGame()

    def position(self, entries, next_player=0):
        pieces = np.zeros((9, 9), dtype=np.int8)
        for square, piece in entries.items():
            pieces[int(square[1]) - 1, ord(square[0]) - ord('A')] = piece
        return load_position(Board(), pieces, next_player)

    def test_display_opening_and_canonical_red_keep_physical_colours(self):
        initial = self.game.getInitBoard()
        text = format_board(initial)
        rows = text.splitlines()[1:10]
        self.assertEqual([int(row.split()[0]) for row in rows], list(range(9, 0, -1)))
        cells = {f'{column}{9-i}': row.split()[j+1]
                 for i, row in enumerate(rows) for j, column in enumerate('ABCDEFGHI')}
        for token, squares in {
            'BP': 'B5 C4 D3 E2', 'BR': 'B4 C3 D2', 'BS': 'C5 D4 E3',
            'RP': 'E8 F7 G6 H5', 'RR': 'F8 G7 H6', 'RS': 'E7 F6 G5',
        }.items():
            self.assertEqual({s for s, cell in cells.items() if cell == token}, set(squares.split()))
        self.assertEqual(cells['A1'], '..')
        self.assertEqual(cells['I9'], '..')
        physical, player = self.game.getNextState(initial, 0, parse_move('B5 B6'))
        saved = physical.copy()
        canonical = self.game.getCanonicalForm(physical, player)
        self.assertEqual(format_board(physical).splitlines()[:13], format_board(canonical).splitlines()[:13])
        self.assertIn('Red to move (player 0)', format_board(canonical))
        self.assertIn('Player 0: Red; player 1: Blue', format_board(canonical))
        self.assertIn("I9 defended by Red (Blue's target)", text)
        np.testing.assert_array_equal(physical, saved)

    def test_occupied_goal_keeps_piece_and_goal_labels(self):
        text = format_board(self.position({'I9': 2, 'B2': -1}))
        self.assertEqual(text.splitlines()[1].split()[9], 'BS')
        self.assertIn("I9 defended by Red (Blue's target)", text)

    def test_all_on_board_actions_round_trip_and_bad_syntax_rejected(self):
        for action in range(648):
            if action_stays_on_board(action):
                self.assertEqual(parse_move(move_to_str(action).replace('->', ' ').lower()), action)
        for text in ('', 'B5', 'B5 C5 D5', 'J1 I1', 'A0 A1', 'A1 A1', 'A1 A3', '12 13'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_move(text)

    def test_human_retries_without_mutation_for_both_colours(self):
        initial = self.game.getInitBoard()
        red, _ = self.game.getNextState(initial, 0, parse_move('B5 B6'))
        for physical, player, legal in ((initial, 0, 'B5 B6'), (red, 1, 'E8 D8')):
            state = self.game.getCanonicalForm(physical, player)
            before = state.copy()
            attempts = iter(['nonsense', 'A1 A9', 'A1 B1', 'B5 C5', legal.lower()])
            prompts = []

            def answer(prompt):
                np.testing.assert_array_equal(state, before)
                prompts.append(prompt)
                return next(attempts)

            output = io.StringIO()
            with patch('builtins.input', side_effect=answer), contextlib.redirect_stdout(output):
                action = HumanPlayer(self.game).play(state, 1)
            self.assertEqual(action, parse_move(legal))
            self.assertEqual(output.getvalue().count('Invalid move:'), 4)
            self.assertTrue(all(('Blue' if player == 0 else 'Red') in p for p in prompts))
            child, next_player = self.game.getNextState(physical, player, action)
            self.assertEqual(next_player, 1-player)
            self.assertEqual(np.count_nonzero(child[:, :, 0]), 20)
            np.testing.assert_array_equal(state, before)

    def test_human_eof_and_interrupt_propagate(self):
        state = self.game.getInitBoard()
        for error in (EOFError, KeyboardInterrupt):
            with patch('builtins.input', side_effect=error), self.assertRaises(error):
                HumanPlayer(self.game).play(state)

    def test_seeded_random_only_legal_and_canonical_action_equivalence(self):
        physical = self.game.getInitBoard()
        for player in (0, 1):
            if player:
                physical, _ = self.game.getNextState(physical, 0, parse_move('B5 B6'))
            state = self.game.getCanonicalForm(physical, player)
            saved = state.copy()
            first, second = RandomPlayer(self.game, seed=8), RandomPlayer(self.game, seed=8)
            actions = [first.play(state, i) for i in range(100)]
            self.assertEqual(actions, [second.play(state, i) for i in range(100)])
            self.assertGreater(len(set(actions)), 1)
            self.assertTrue(all(self.game.getValidMoves(physical, player)[a] for a in actions))
            np.testing.assert_array_equal(state, saved)

    def test_greedy_immediate_corner_win_for_both_physical_colours(self):
        for entries, player, move in (
            ({'H8': 1, 'I9': -2, 'G8': -2}, 0, 'H8 I9'),
            ({'B2': -1, 'A1': 2, 'C2': 2}, 1, 'B2 A1'),
        ):
            physical = self.position(entries, player)
            state = self.game.getCanonicalForm(physical, player)
            saved = state.copy()
            greedy = GreedyPlayer(self.game)
            self.assertEqual([greedy.play(state, n) for n in range(3)], [parse_move(move)] * 3)
            child, next_player = self.game.getNextState(physical, player, parse_move(move))
            self.assertEqual(self.game.getGameEnded(child, next_player)[player], 1)
            np.testing.assert_array_equal(state, saved)

    def test_greedy_stalemate_win_beats_target_distance(self):
        state = self.position({'E5': 1, 'D4': -2})
        action = GreedyPlayer(self.game).play(state)
        self.assertEqual(action, parse_move('E5 D4'))
        child, player = self.game.getNextState(state, 0, action)
        np.testing.assert_array_equal(self.game.getGameEnded(child, player), [1, -1])

    def test_greedy_prevents_corner_loss_before_more_advanced_capture(self):
        state = self.position({'B2': 1, 'A2': -2, 'C3': -2})
        self.assertEqual(GreedyPlayer(self.game).play(state), parse_move('B2 A2'))

    def test_greedy_avoids_reply_capturing_its_last_piece(self):
        state = self.position({'D4': 2, 'E4': -1, 'I8': -3})
        action = GreedyPlayer(self.game).play(state)
        child, player = self.game.getNextState(state, 0, action)
        for reply in np.flatnonzero(self.game.getValidMoves(child, player)):
            reply_state, next_player = self.game.getNextState(child, player, int(reply))
            self.assertNotEqual(self.game.getGameEnded(reply_state, next_player)[1], 1)

    def test_greedy_capture_then_distance_then_lowest_action(self):
        greedy = GreedyPlayer(self.game)
        state = self.position({'E5': 1, 'D4': -2, 'A8': -1})
        self.assertEqual(greedy.play(state), parse_move('E5 D4'))
        state = self.position({'E5': 1, 'A8': -1})
        self.assertEqual(greedy.play(state), parse_move('E5 F6'))
        state = self.position({'E6': 1, 'A8': -1})
        self.assertEqual(greedy.play(state), parse_move('E6 F7'))
        state = self.position({'F5': 1, 'A8': -1})
        # F5->F6 and F5->G6 tie at distance 3; N has the lower action ID.
        self.assertEqual(greedy.play(state), parse_move('F5 F6'))

    def test_pit_player_factory_callback_contract(self):
        import pit
        from intransitive import IntransitivePlayers

        # Only game discovery is pending #11; exercise pit's actual factory.
        with patch.object(pit, 'game', None), patch.object(pit, 'players', None, create=True), \
                patch.object(pit, 'NNet', None, create=True), \
                patch.object(pit, 'import_game', return_value=(
                    IntransitiveGame, None, IntransitivePlayers, 2)):
            for name in ('random', 'greedy', 'human'):
                callback = pit.create_player(name, SimpleNamespace(game='intransitive'))
                state = pit.game.getInitBoard()
                with patch('builtins.input', return_value='B5 B6'):
                    action = callback(state, 1)
                self.assertTrue(pit.game.getValidMoves(state, 0)[action])

    def terminal_states(self):
        yield self.position({'I9': 1, 'B2': -1})
        yield self.position({'E5': -1})
        board = Board()
        load_history(board, [sparse_position()])
        play(board, noncapture_actions())
        yield board.get_state()
        load_history(board, [sparse_position()])
        play(board, [parse_move(m) for m in ('B2 C2', 'H8 G8', 'C2 B2', 'G8 H8')] * 2)
        yield board.get_state()

    def test_terminal_players_fail_without_input_and_arena_skips_callbacks(self):
        for state in self.terminal_states():
            player = int(state[:, :, 32].flat[1])
            canonical = self.game.getCanonicalForm(state, player)
            with patch('builtins.input', side_effect=AssertionError('Must not prompt')):
                for cls in (RandomPlayer, HumanPlayer, GreedyPlayer):
                    with self.assertRaisesRegex(ValueError, 'terminal'):
                        cls(self.game).play(canonical, 1)
            payload = state.tobytes() + bytes([player]) + (30).to_bytes(2, 'big')
            restored = base64.b64encode(zlib.compress(payload, wbits=-15)).decode()
            def forbidden(*args):
                self.fail('Arena called a player on a terminal state')
            with contextlib.redirect_stdout(io.StringIO()):
                result = Arena(forbidden, forbidden, self.game).playGame(initial_state=restored)
            self.assertEqual(result, self.game.getGameEnded(state, player)[0])

    def test_seeded_arena_games_both_assignments_preserve_opening_and_states(self):
        official = self.game.getInitBoard()
        outcomes = []
        for seed in (8, 80):
            for swapped in (False, True):
                colours = []
                def checked(agent):
                    def callback(state, turn):
                        self.assertLessEqual(turn, 600)
                        self.assertFalse(self.game.getGameEnded(state, 0).any())
                        colours.append(player_colour(state, 0))
                        if turn == 1:
                            np.testing.assert_array_equal(state, official)
                        saved = state.copy()
                        action = agent.play(state, turn)
                        np.testing.assert_array_equal(state, saved)
                        self.assertTrue(self.game.getValidMoves(state, 0)[action])
                        return action
                    return callback
                arena = Arena(checked(GreedyPlayer(self.game)),
                              checked(RandomPlayer(self.game, seed)), self.game)
                outcomes.append(float(arena.playGame(other_way=swapped)))
                self.assertEqual(colours, ['Blue' if i % 2 == 0 else 'Red' for i in range(len(colours))])
        self.assertTrue(all(result in (1., -1.) or 0 < result < .001 for result in outcomes))
        np.testing.assert_array_equal(self.game.getInitBoard(), official)


if __name__ == '__main__':
    unittest.main()
