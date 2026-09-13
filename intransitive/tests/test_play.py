"""Browser-session checks against actual engine transitions and history."""

import unittest

from intransitive.IntransitiveConstants import parse_coordinate
from intransitive.play import GameSession


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


if __name__ == "__main__":
    unittest.main()
