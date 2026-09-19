"""Separate browser engines, native HTTP-thread execution and diagnostics."""
from concurrent.futures import ThreadPoolExecutor
import signal
import unittest
from unittest.mock import patch

from intransitive.heuristics import AlphaBetaPlayer, SearchConfig
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.play import GameSession, OpponentFactory
from intransitive.rust_teacher import BINARY, RustTeacher
from intransitive.rust_teacher.browser import RustMinimaxOpponent


@unittest.skipUnless(BINARY.exists(), 'Build the native Rust minimax binary first')
class RustBrowserTests(unittest.TestCase):
    def setUp(self):
        self.config = SearchConfig(max_depth=2, time_limit=10., node_limit=10000000)
        self.factory = OpponentFactory(config=self.config)

    def test_independent_engines_and_native_thread_execution(self):
        python = self.factory.create('alphabeta')
        rust = self.factory.create('rust-minimax')
        self.assertIsInstance(python, AlphaBetaPlayer)
        self.assertIsInstance(rust, RustMinimaxOpponent)
        self.assertNotEqual(python.label, rust.label)
        state = IntransitiveGame().getInitBoard()
        handler = signal.getsignal(signal.SIGTERM)
        processes = []
        def client(binary):
            teacher = RustTeacher(binary)
            processes.append(teacher.process)
            return teacher
        with patch('intransitive.rust_teacher.browser.RustTeacher', side_effect=client), \
             patch.object(AlphaBetaPlayer, 'analyze', side_effect=AssertionError('Python search used')):
            with ThreadPoolExecutor(1) as executor:
                action = executor.submit(rust.choose, state, 0).result(timeout=20)
        self.assertTrue(IntransitiveGame().getValidMoves(state, 0)[action])
        self.assertEqual(rust.last_result.completed_depth, 2)
        self.assertEqual(signal.getsignal(signal.SIGTERM), handler)
        self.assertTrue(all(p.poll() is not None for p in processes))
        self.assertAlmostEqual(rust.last_result.score, python.analyze(state).score)

    def test_watch_both_engines_and_preserve_diagnostics_in_history(self):
        session = GameSession(opponent_factory=self.factory)
        start = session.update('restart', dict(revision=0, mode='watch',
            blue_opponent='alphabeta', red_opponent='rust-minimax'))
        self.assertEqual(start['watch_opponents'], ['alphabeta', 'rust-minimax'])
        self.assertEqual(start['bot_labels'], ['Minimax · Python', 'Minimax · Rust'])
        rows = []
        for expected in ('Python', 'Rust', 'Python', 'Rust'):
            row = session.update('ai', dict(revision=session.revision))
            self.assertEqual(row['analysis']['backend'], expected)
            self.assertGreater(row['analysis']['nodes'], 0)
            self.assertGreaterEqual(row['analysis']['elapsed_seconds'], 0)
            rows.append(row)
        history = session.snapshots()
        self.assertIsNone(history[0]['analysis'])
        for row, past in zip(rows, history[1:]):
            self.assertEqual(past['analysis'], row['analysis'])
        self.assertEqual(rows[1]['analysis']['work_unit'], 'node_visits')
        from intransitive.record import load_record
        self.assertEqual(load_record(rows[-1]['pgn']).tags['Red'], 'Minimax · Rust')

    def test_native_zero_budget_is_explicit_legal_fallback(self):
        rust = self.factory.create('rust-minimax', dict(time_limit=0.))
        game = IntransitiveGame()
        state = game.getInitBoard()
        action = rust.choose(state, 0)
        self.assertTrue(game.getValidMoves(state, 0)[action])
        self.assertEqual(rust.last_result.selection_source, 'legal_fallback')
        self.assertEqual(rust.last_result.completed_depth, 0)
        self.assertIsNone(rust.last_result.score)
        self.assertIsNone(rust.last_result.score_bound)

    def test_unsupported_config_rejects_restart_without_changing_session(self):
        session = GameSession(opponent_factory=self.factory)
        before = session.snapshot()
        for options in (dict(attack_enabled=True), dict(defence_enabled=True),
                        dict(overload_enabled=True), dict(max_depth=0), dict(max_depth=33)):
            with self.assertRaises(ValueError):
                session.update('restart', dict(revision=0, opponent='rust-minimax', ab_options=options))
            self.assertEqual(session.snapshot(), before)

    def test_native_errors_propagate_without_python_fallback(self):
        rust = self.factory.create('rust-minimax')
        with patch.object(RustTeacher, 'analyze', side_effect=RuntimeError('native failed')), \
             patch.object(AlphaBetaPlayer, 'analyze', side_effect=AssertionError('Python search used')):
            with self.assertRaisesRegex(RuntimeError, 'native failed'):
                rust.choose(IntransitiveGame().getInitBoard(), 0)
        self.assertIsNone(rust.last_result)


if __name__ == '__main__':
    unittest.main()
