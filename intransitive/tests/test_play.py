"""Browser-session checks against actual engine transitions and history."""

import unittest

from intransitive.IntransitiveConstants import parse_coordinate
from intransitive.play import GameSession
from intransitive.IntransitiveLogicNumba import Board


class FirstLegalOpponent:
    label = 'Test model'

    def __init__(self):
        self.reloads = 0
        self.fail = False

    def reload(self):
        if self.fail:
            raise RuntimeError('Unavailable checkpoint')
        self.reloads += 1

    def choose(self, state, player):
        if self.fail:
            raise RuntimeError('Inference failed')
        board = Board()
        board.copy_state(state, True)
        return next(i for i, valid in enumerate(board.valid_moves(player)) if valid)


class PlaySessionTests(unittest.TestCase):
    def setUp(self):
        self.game = GameSession()

    def move(self, source, target):
        source, target = list(parse_coordinate(source)), list(parse_coordinate(target))
        move = next(m for m in self.game.snapshot()["legal"]
                    if m["source"] == source and m["target"] == target)
        return self.game.update("move", dict(action=move["action"], revision=self.game.revision))

    def test_capture_and_undo_restore_exact_history(self):
        for source, target in [("C5", "D5"), ("H5", "G4"), ("D5", "E5"), ("G4", "F5")]:
            self.move(source, target)
        before = self.game.board.get_state().tobytes()
        state = self.move("E5", "F5")
        self.assertEqual(state["board"][4][5], 2)
        self.assertEqual(state["noncapture"], 0)
        self.assertEqual(state["moves"][-1], "E5 × F5")
        self.game.update("undo", dict(revision=self.game.revision))
        self.assertEqual(self.game.board.get_state().tobytes(), before)

    def test_repetition_terminal_and_undo(self):
        for _ in range(2):
            for source, target in [("B5", "B6"), ("H5", "H4"), ("B6", "B5"), ("H4", "H5")]:
                state = self.move(source, target)
        self.assertEqual(state["reason"], "repetition")
        self.assertIsNone(state["winner"])
        self.assertEqual(state["legal"], [])
        state = self.game.update("undo", dict(revision=self.game.revision))
        self.assertEqual(state["reason"], "ongoing")
        self.assertTrue(state["legal"])

    def test_invalid_and_stale_moves_are_atomic(self):
        self.move("B5", "B6")
        before = self.game.snapshot()
        for data in [dict(action=-1, revision=1), dict(action=True, revision=1),
                     dict(action=before["legal"][0]["action"], revision=0)]:
            with self.assertRaises(ValueError):
                self.game.update("move", data)
            self.assertEqual(self.game.snapshot(), before)

    def test_restart_clears_history_and_counters(self):
        initial = self.game.board.get_state().tobytes()
        self.move("B5", "B6")
        state = self.game.update("restart", dict(revision=1))
        self.assertEqual(self.game.board.get_state().tobytes(), initial)
        self.assertEqual(state["moves"], [])
        self.assertEqual(state["ply"], 0)
        with self.assertRaises(ValueError):
            self.game.update("undo", dict(revision=2))


class AIPlaySessionTests(unittest.TestCase):
    def setUp(self):
        self.opponent = FirstLegalOpponent()
        self.game = GameSession(self.opponent)

    def update(self, command, **data):
        return self.game.update(command, dict(revision=self.game.revision, **data))

    def human_move(self):
        return self.update('move', action=self.game.snapshot()['legal'][0]['action'])

    def test_reply_and_undo_restore_complete_human_turn(self):
        initial = self.game.board.get_state().tobytes()
        self.assertTrue(self.human_move()['ai_turn'])
        reply = self.update('ai')
        self.assertFalse(reply['ai_turn'])
        self.assertEqual(reply['ply'], 2)
        self.assertEqual(reply['player'], 0)
        self.update('undo')
        self.assertEqual(self.game.board.get_state().tobytes(), initial)

    def test_red_opening_and_undo_keep_ai_opening(self):
        state = self.update('restart', human_player=1)
        self.assertTrue(state['ai_turn'])
        self.assertEqual(self.opponent.reloads, 1)
        opening = self.update('ai')
        self.assertFalse(opening['can_undo'])
        before = self.game.board.get_state().tobytes()
        with self.assertRaises(ValueError):
            self.update('undo')
        self.human_move()
        self.update('ai')
        self.update('undo')
        self.assertEqual(self.game.board.get_state().tobytes(), before)

    def test_turn_ownership_and_stale_ai_requests_are_atomic(self):
        with self.assertRaises(ValueError):
            self.update('ai')
        self.human_move()
        before = self.game.snapshot()
        with self.assertRaises(ValueError):
            self.human_move()
        with self.assertRaises(ValueError):
            self.game.update('ai', {'revision': 0})
        self.assertEqual(self.game.snapshot(), before)
        self.update('ai')
        with self.assertRaises(ValueError):
            self.update('ai')

    def test_inference_and_reload_failure_preserve_game(self):
        self.human_move()
        before = self.game.snapshot()
        self.opponent.fail = True
        for command in ('ai', 'restart'):
            with self.assertRaises(RuntimeError):
                self.update(command)
            self.assertEqual(self.game.snapshot(), before)
        self.opponent.fail = False
        self.update('ai')

    def test_human_can_undo_before_ai_reply(self):
        self.human_move()
        state = self.update('undo')
        self.assertEqual(state['ply'], 0)
        self.assertFalse(state['ai_turn'])

    def test_terminal_human_move_does_not_request_ai(self):
        # Repetition ends the game without scheduling another AI move.
        for _ in range(2):
            for source, target in [('B5', 'B6'), ('H5', 'H4'), ('B6', 'B5'), ('H4', 'H5')]:
                self.game.human_player = self.game.board.get_next_player()
                source, target = list(parse_coordinate(source)), list(parse_coordinate(target))
                action = next(m['action'] for m in self.game.snapshot()['legal']
                              if m['source'] == source and m['target'] == target)
                state = self.update('move', action=action)
        self.assertEqual(state['reason'], 'repetition')
        self.assertFalse(state['ai_turn'])
        with self.assertRaises(ValueError):
            self.update('ai')


if __name__ == "__main__":
    unittest.main()
