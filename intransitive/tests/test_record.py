"""Copied games must reproduce full state and explain the actual AI decision."""

from dataclasses import replace
import json
import subprocess
import sys
import unittest

from intransitive.heuristics import SearchConfig
from intransitive.heuristics.analyze import analyze_record, evaluate_position
from intransitive.play import GameSession, OpponentFactory
from intransitive.record import load_record, parse_record_move, state_hash


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.config = SearchConfig(max_depth=1, time_limit=10, node_limit=500000,
                                   proof_depth=0, count_weight=123, defence_enabled=True)
        self.session = GameSession(opponent_factory=OpponentFactory(config=self.config))

    def update(self, command, **kwargs):
        return self.session.update(command, dict(revision=self.session.revision, **kwargs))

    def move(self, move):
        return self.update('move', action=parse_record_move(move))

    def test_capture_round_trip_and_every_prior_ply(self):
        for move in ('C5-D5', 'H5-G4', 'D5-E5', 'G4-F5', 'E5xF5'):
            self.move(move)
        pgn = self.session.snapshot()['pgn']
        self.assertIn('3. E5xF5 *', pgn)
        record = load_record(pgn)
        self.assertEqual(record.config, self.config)
        for actual, expected in zip(record.states, self.session.history + [self.session.board.get_state()]):
            self.assertEqual(actual.tobytes(), expected.tobytes())
        self.update('undo')
        self.assertEqual(load_record(self.session.snapshot()['pgn']).states[-1].tobytes(),
                         self.session.board.get_state().tobytes())
        self.update('restart')
        self.assertEqual(load_record(self.session.snapshot()['pgn']).actions, [])

    def test_repetition_continues_and_analysis_gets_fresh_window(self):
        for move in ('B5-B6', 'H5-H4', 'B6-B5', 'H4-H5') * 2:
            self.move(move)
        pgn = self.session.snapshot()['pgn']
        record = load_record(pgn)
        self.assertEqual(record.tags['Result'], '*')
        self.assertEqual(record.states[-1].tobytes(), self.session.board.get_state().tobytes())
        report = analyze_record(pgn)
        self.assertFalse(report['evaluation']['terminal'])
        self.assertIsNotNone(report['search'])
        self.assertNotEqual(report['state_sha256'], report['search_state_sha256'])

    def test_long_official_record_and_legacy_draw_record(self):
        from intransitive.IntransitiveLogicNumba import Board
        from intransitive.record import export_record, RULES
        cycle = ('B5-B6', 'H5-H4', 'B6-B5', 'H4-H5')
        for move in cycle * 10:
            self.move(move)
        record = load_record(self.session.snapshot()['pgn'])
        self.assertEqual(len(record.actions), 40)
        for actual, expected in zip(record.states, self.session.history + [self.session.board.get_state()]):
            self.assertEqual(actual.tobytes(), expected.tobytes())
        legacy = Board()
        for move in cycle * 2:
            legacy.make_move(parse_record_move(move), legacy.get_next_player())
        pgn = export_record(cycle * 2, legacy, self.config)
        self.assertEqual(load_record(pgn).tags['Rules'], RULES)
        self.assertEqual(load_record(pgn).tags['Result'], '1/2-1/2')
        self.assertIsNone(analyze_record(pgn)['search'])

    def test_corrupt_records_and_illegal_moves_rejected(self):
        pgn = self.move('B5-B6')['pgn']
        for corrupt in (pgn.replace('1. B5-B6', '1. A1-A2'),
                        pgn.replace('1. B5-B6', '1. B5-B4'),
                        pgn.replace('[PlyCount "1"]', '[PlyCount "2"]'),
                        pgn.replace('[Result "*"]', '[Result "1-0"]'),
                        pgn.replace('Intransitive-PGN-2', 'Chess'),
                        pgn.replace('1. B5-B6', '2. B5-B6'),
                        '[SetUp "1"]\n' + pgn):
            with self.subTest(corrupt=corrupt[-80:]), self.assertRaises(ValueError):
                load_record(corrupt)

    def test_version_one_official_record_replays_to_expanded_history(self):
        from intransitive.record import legacy_state_hash
        for move in ('B5-B6', 'H5-H4', 'B6-B5', 'H4-H5') * 10:
            self.move(move)
        state = self.session.board.get_state()
        pgn = self.session.snapshot()['pgn'].replace('Intransitive-PGN-2', 'Intransitive-PGN-1')
        pgn = pgn.replace(state_hash(state), legacy_state_hash(state))
        loaded = load_record(pgn)
        self.assertEqual(loaded.states[-1].tobytes(), state.tobytes())
        self.assertEqual(loaded.states[-1][:, :, 82:84].flat[4], 41)

    def test_version_one_quiet_limit_is_preserved_during_record_validation(self):
        import numpy as np
        from intransitive.IntransitiveConstants import action_destination
        from intransitive.IntransitiveLogicNumba import Board
        from intransitive.IntransitiveDisplay import move_to_str
        from intransitive.record import export_record, legacy_state_hash
        board = Board()
        rng = np.random.default_rng(80)
        moves = []
        for _ in range(30):
            before = board.get_state()
            player = board.get_next_player()
            for action in rng.permutation(np.flatnonzero(board.valid_moves(player))):
                x, y = action_destination(int(action))
                if before[y, x, 0] != 0:
                    continue
                board.copy_state(before, True)
                board.make_move(int(action), player)
                if board.get_terminal_reason() == 'ongoing':
                    moves.append(move_to_str(int(action)))
                    break
            else:
                self.fail('Could not construct the quiet legacy record')
        state = board.get_state()
        pgn = export_record(moves, board, self.config)
        self.assertEqual(load_record(pgn).tags['Result'], '*')
        pgn = pgn.replace('Intransitive-PGN-2', 'Intransitive-PGN-1')
        pgn = pgn.replace('80-plies-without-capture', '30-plies-without-capture')
        pgn = pgn.replace(state_hash(state), legacy_state_hash(state))
        pgn = pgn.replace('[Result "*"]', '[Result "1/2-1/2"]')
        pgn = pgn.rstrip()[:-1] + '1/2-1/2\n'
        loaded = load_record(pgn)
        self.assertEqual(loaded.tags['Result'], '1/2-1/2')
        self.assertEqual(loaded.states[-1].tobytes(), state.tobytes())

    def test_original_ai_search_and_active_configuration(self):
        self.update('restart', opponent='alphabeta', human_player=1,
                    ab_options=dict(max_depth=0, defence_enabled=False))
        before = self.session.board.get_state().tobytes()
        state = self.update('ai')
        record = load_record(state['pgn'])
        self.assertEqual(record.last_ai['ply'], 0)
        self.assertEqual(record.states[0].tobytes(), before)
        self.assertEqual(record.last_ai['search']['action'], record.actions[0])
        self.assertEqual(record.last_ai['search']['stop_reason'], 'maximum_depth')
        self.assertEqual(record.last_ai['search']['diagnostics_status'], 'completed')
        self.assertEqual(record.last_ai['search']['effective_limits']['max_depth'], 0)
        self.assertFalse(record.config.defence_enabled)
        self.assertEqual(record.config.count_weight, 123)
        report = analyze_record(state['pgn'], last_ai=True)
        self.assertEqual(report['recorded_ai'], record.last_ai)
        self.assertEqual(report['ply'], 0)
        self.move('H5-H4')
        self.update('ai')
        self.update('undo')
        self.assertEqual(load_record(self.session.snapshot()['pgn']).last_ai, record.last_ai)
        self.update('restart')
        self.assertIsNone(load_record(self.session.snapshot()['pgn']).last_ai)

    def test_candidate_scores_use_original_player_and_do_not_mutate(self):
        pgn = self.move('B5-B6')['pgn']
        report = analyze_record(pgn, ply=0, moves=['C5-D5'])
        self.assertEqual(report['perspective'], 'Blue')
        self.assertEqual(report['search']['completed_depth'], 1)
        record = load_record(pgn)
        before = record.states[1].tobytes()
        blue = evaluate_position(record.states[1], self.config, perspective=0)
        red = evaluate_position(record.states[1], self.config, perspective=1)
        self.assertEqual(blue['score'], -red['score'])
        self.assertEqual(record.states[1].tobytes(), before)
        played = next(c for c in report['candidates'] if c['played'])
        self.assertEqual(played['evaluation']['terms'], blue['terms'])
        self.assertEqual(report['state_sha256'], state_hash(record.states[0]))
        for candidate in report['candidates']:
            self.assertEqual(candidate['evaluation']['perspective'], 'Blue')

    def test_invalid_selection_and_exhausted_evaluation(self):
        pgn = self.session.snapshot()['pgn']
        for kwargs in (dict(ply=-1), dict(ply=1), dict(last_ai=True), dict(moves=['A1-A2'])):
            with self.assertRaises(ValueError):
                analyze_record(pgn, **kwargs)
        result = evaluate_position(self.session.board.get_state(), replace(self.config, node_limit=0))
        self.assertIsNone(result['score'])
        self.assertIn('budget exhausted', result['status'])

    def test_stdin_cli(self):
        process = subprocess.run([sys.executable, '-m', 'intransitive.heuristics.analyze', '-',
                                  '--depth', '0', '--move', 'C5-D5'],
                                 input=self.session.snapshot()['pgn'], text=True,
                                 capture_output=True, check=True)
        report = json.loads(process.stdout)
        self.assertEqual(report['config']['max_depth'], 0)
        self.assertTrue(any(c['move'] == 'C5->D5' for c in report['candidates']))


if __name__ == '__main__':
    unittest.main()
