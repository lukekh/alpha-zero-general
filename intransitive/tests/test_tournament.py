"""Official outcomes, immutable scheduling, resumability and process cleanup."""
from copy import deepcopy
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from intransitive.IntransitiveConstants import E, W, encode_action
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveLogicNumba import Board
from intransitive.heuristics.config import SearchConfig
from intransitive.heuristics.search import SearchResult
from intransitive.record import state_hash
from intransitive.tests.test_draws import load_history, sparse_position
from intransitive.tournament.report import report
from intransitive.tournament.runner import (EngineProcess, MatchFailure,
                                           play_match, replay, run)
from intransitive.tournament.spec import (candidate, effective_config, manifest,
                                         generate_positions, orbit_hash, pack, position, protocol)


def crash_worker(connection, *args):
    connection.close()


def hang_worker(connection, *args):
    time.sleep(30)


class FakeEngine:
    instances = []
    action = None
    stopped = False
    cancel = None
    fail = None

    def __init__(self, item, limits, seed):
        self.item, self.limits, self.seed = item, limits, seed
        self.connection = self
        self.state = None
        self.closed = False
        self.instances.append(self)

    def send(self, state):
        self.state = state
        assert int(state[:, :, 82:84].flat[1]) == 0

    def receive(self, deadline, cancelled):
        if self.fail:
            raise MatchFailure(self.fail, 'injected')
        if self.state is None:
            return dict(kind='ready', candidate=self.item['sha256'],
                        config=effective_config(self.item, self.limits).to_dict(),
                        startup_seconds=0., cpu_seconds=0., peak_rss_bytes=10)
        legal = np.flatnonzero(IntransitiveGame().getValidMoves(self.state, 0))
        action = int(legal[0]) if self.action is None else type(self).action(self.state)
        result = SearchResult(action, 0., 1, [action], 1, 1, 0, .001, self.stopped,
                              selected_depth=1, stop_reason='time' if self.stopped else 'maximum_depth')
        if self.cancel:
            self.cancel.set()
        return dict(kind='move', result=asdict(result), latency_seconds=.001,
                    cpu_seconds=.001, peak_rss_bytes=10)

    def close(self):
        self.closed = True


class TournamentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Compile rules before short child deadlines and fake-engine tests.
        game = IntransitiveGame()
        game.getValidMoves(game.getInitBoard(), 0)

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name)
        FakeEngine.instances = []
        FakeEngine.action = FakeEngine.cancel = FakeEngine.fail = None
        FakeEngine.stopped = False

    def tearDown(self):
        self.folder.cleanup()

    def spec(self, *, max_plies=4, mode='depth'):
        return manifest([candidate('A'), candidate('B', {'advantage_weight': 50.})],
                        [position([], seed=54, stage='official', pool='search')],
                        [protocol(mode, depth=1, max_plies=max_plies, seconds=10.)])

    def play(self, spec=None, *, cancelled=None, name='game.json'):
        spec = spec or self.spec()
        return play_match(spec, spec['tasks'][0], self.path / name,
                          cancelled or threading.Event(), engine_factory=FakeEngine)

    def test_candidate_validation_and_contract_roundtrip(self):
        item = candidate('A')
        self.assertEqual(candidate('A', genome=item['genome']), item)
        self.assertEqual(effective_config(item, protocol('depth')).to_dict(),
                         SearchConfig(max_depth=2, time_limit=.05, node_limit=10**9).to_dict())
        for weights in ({'race_weight': 1}, {'count_weight': 101}, {'attack_enabled': True},
                        {'pressure_weight': float('nan')}, {'attack_weight': float('inf')},
                        {'advantage_weight': -101}, {'pressure_weight': 101}, {'attack_weight': True}):
            with self.assertRaises(ValueError):
                candidate('bad', weights)
        with self.assertRaises(ValueError):
            candidate('native', backend='rust')
        self.assertTrue(candidate('pressure', {'pressure_weight': 2.})['evaluation']['pressure_enabled'])
        self.assertFalse(candidate('zero', {'attack_weight': 0.})['evaluation']['attack_enabled'])
        self.assertNotEqual(item['sha256'], candidate('other', {'advantage_weight': 0.})['sha256'])

    def test_schedule_swaps_colours_and_rejects_duplicate_evidence(self):
        spec = self.spec()
        first, second = spec['tasks']
        self.assertEqual(first['pair_id'], second['pair_id'])
        self.assertEqual(first['seed'], second['seed'])
        self.assertEqual(first['colours'], second['colours'][::-1])
        self.assertEqual(spec, self.spec())
        with self.assertRaises(ValueError):
            manifest([candidate('A'), candidate('same')], spec['positions'], spec['protocols'])
        duplicate = deepcopy(spec['positions'][0])
        duplicate['pool'] = 'heldout'
        with self.assertRaises(ValueError):
            manifest(spec['candidates'], spec['positions'] + [duplicate], spec['protocols'])
        with self.assertRaises(ValueError):
            manifest(spec['candidates'], spec['positions'], spec['protocols'], position_limit=2)

    def test_shared_genome_contract_reaches_effective_engine_config(self):
        from intransitive.heuristics.tuning import Genome
        limits = protocol('depth', depth=2, seconds=10.)
        for genes in ({}, {'attack': 12., 'defence': 10., 'overload': 5., 'pressure': 10.},
                      {'advantage': 0., 'attack': 0., 'pressure': 20.}):
            genome = Genome.from_genes(genes)
            frozen = candidate('shared-contract', genome=genome.to_dict())
            expected = genome.to_config(SearchConfig(**limits['search']))
            self.assertEqual(effective_config(frozen, limits), expected)
            self.assertEqual(frozen['genome'], genome.to_dict())

    def test_exact_history_and_symmetry_identity(self):
        from intransitive.IntransitiveSymmetries import transform_state
        board = Board()
        first = load_history(board, [sparse_position()])
        second = load_history(board, [sparse_position()] * 3)
        self.assertNotEqual(orbit_hash(first), orbit_hash(second))
        self.assertEqual(orbit_hash(first), orbit_hash(transform_state(first, 7)))
        other = load_history(board, [sparse_position()], a1_defender=1)
        self.assertNotEqual(orbit_hash(first), orbit_hash(other))

    def test_generated_corpus_has_legal_diverse_positions_and_frozen_splits(self):
        corpus = generate_positions(54, 9)
        self.assertEqual(corpus, generate_positions(54, 9))
        self.assertEqual({p['stage'] for p in corpus}, {'official', 'opening', 'midgame', 'endgame'})
        self.assertEqual({p['pool'] for p in corpus}, {'search', 'validation', 'heldout'})
        self.assertEqual(len({p['orbit_sha256'] for p in corpus}), len(corpus))
        for seed in {p['seed'] for p in corpus}:
            self.assertEqual(len({p['pool'] for p in corpus if p['seed'] == seed}), 1)
        spec = self.spec()
        for pool in ('search', 'validation', 'heldout'):
            selected = manifest(spec['candidates'], corpus, spec['protocols'], pool=pool, position_limit=1)
            self.assertEqual(len(selected['tasks']), 2)

    def test_population_incumbent_and_archive_exact_pair_matrix(self):
        spec = self.spec()
        candidates = spec['candidates'] + [candidate('incumbent', {'pressure_weight': 1}, role='incumbent'),
                                         candidate('archive', {'pressure_weight': 2}, role='archive')]
        matrix = manifest(candidates, spec['positions'], spec['protocols'])
        self.assertEqual(len(matrix['tasks']), 10)  # A/B, A/I, B/I, A/H, B/H; two colours.
        references = {c['sha256'] for c in candidates if c['role'] != 'population'}
        self.assertTrue(all(set(t['colours']) != references for t in matrix['tasks']))

    def test_replay_legal_canonical_and_candidate_handshake(self):
        spec = self.spec()
        row = self.play(spec)
        self.assertEqual(row['status'], 'unfinished')
        self.assertEqual(len(row['moves']), 4)
        self.assertEqual([m['side'] for m in row['moves']], [0, 1, 0, 1])
        self.assertTrue(all(e.closed for e in FakeEngine.instances))
        self.assertEqual([e.item['sha256'] for e in FakeEngine.instances], row['colours'])
        self.assertEqual(replay(spec['positions'][0], row).tobytes(),
                         replay(spec['positions'][0], json.loads(json.dumps(row))).tobytes())
        altered = deepcopy(row)
        altered['moves'][0]['after'] = 'wrong'
        with self.assertRaises(ValueError):
            replay(spec['positions'][0], altered)

    def test_cancellation_resumes_exactly_and_final_is_not_repeated(self):
        spec = self.spec()
        cancelled = threading.Event()
        FakeEngine.cancel = cancelled
        interrupted = self.play(spec, cancelled=cancelled)
        self.assertEqual(interrupted['status'], 'cancelled')
        self.assertEqual(len(interrupted['moves']), 1)
        self.assertTrue(all(e.closed for e in FakeEngine.instances))
        FakeEngine.cancel = None
        resumed = self.play(spec)
        fresh = self.play(spec, name='fresh.json')
        self.assertEqual(resumed['trajectory_sha256'], fresh['trajectory_sha256'])
        self.assertEqual(resumed['final_state_sha256'], fresh['final_state_sha256'])
        count = len(FakeEngine.instances)
        self.assertEqual(self.play(spec), resumed)
        self.assertEqual(len(FakeEngine.instances), count)

    def test_terminal_win_without_launching_children(self):
        spec = self.spec()
        board = Board(modelling_draws=False)
        pieces = sparse_position()
        pieces[8, 8] = 1
        state = load_history(board, [pieces])
        start = spec['positions'][0]
        start.update(state=pack(state), sha256=state_hash(state))
        spec['tasks'][0]['position'] = start['sha256']
        row = self.play(spec)
        self.assertEqual((row['status'], row['winner'], row['reason']), ('win', 0, 'corner'))
        self.assertFalse(FakeEngine.instances)
        replay(start, row)

    def test_winning_move_is_recorded_and_replayed_as_official_win(self):
        spec = self.spec(max_plies=1)
        board = Board(modelling_draws=False)
        pieces = sparse_position()
        pieces[1, 1] = 0
        pieces[8, 7] = 1
        state = load_history(board, [pieces])
        start = spec['positions'][0]
        start.update(state=pack(state), sha256=state_hash(state))
        spec['tasks'][0]['position'] = start['sha256']
        FakeEngine.action = lambda state: encode_action(7, 8, E)
        row = self.play(spec)
        self.assertEqual((row['status'], row['winner'], len(row['moves'])), ('win', 0, 1))
        replay(start, row)
        forged = deepcopy(row)
        forged['status'] = 'unfinished'
        forged['winner'] = None
        with self.assertRaises(ValueError):
            replay(start, forged)

    def test_cycle_past_modelling_draw_is_unfinished_not_draw(self):
        spec = self.spec(max_plies=12)
        board = Board(modelling_draws=False)
        state = load_history(board, [sparse_position()])
        start = spec['positions'][0]
        start.update(state=pack(state), sha256=state_hash(state))
        spec['tasks'][0]['position'] = start['sha256']

        def cycle(state):
            y, x = np.argwhere(state[:, :, 0] > 0)[0]
            return encode_action(int(x), int(y), E if x in (1, 7) else W)

        FakeEngine.action = cycle
        row = self.play(spec)
        self.assertEqual(row['status'], 'unfinished')
        self.assertEqual(len(row['moves']), 12)
        final = replay(start, row)
        self.assertTrue(IntransitiveGame().getGameEnded(final, 0).any())
        self.assertFalse(IntransitiveGame(modelling_draws=False).getGameEnded(final, 0).any())
        summary = report(spec, [row])['leaderboards']['depth'][0]
        self.assertEqual(summary['win_points_lower'], 0.)
        self.assertFalse(summary['eligible'])

    def test_red_canonical_goal_and_absolute_winner(self):
        spec = self.spec(max_plies=1)
        pieces = sparse_position()
        pieces[7, 7] = 0
        pieces[0, 1] = -1
        state = load_history(Board(modelling_draws=False), [pieces], first_player=1)
        start = spec['positions'][0]
        start.update(state=pack(state), sha256=state_hash(state))
        spec['tasks'][0]['position'] = start['sha256']

        def red_win(canonical):
            self.assertEqual(int(canonical[:, :, 82:84].flat[2]), 1)
            self.assertEqual(int(canonical[0, 1, 0]), 1)
            return encode_action(1, 0, W)

        FakeEngine.action = red_win
        row = self.play(spec)
        self.assertEqual((row['status'], row['winner'], row['moves'][0]['side']), ('win', 1, 1))
        replay(start, row)

    def test_explicit_failures_and_stopped_depth(self):
        for status in ('crash', 'infrastructure_timeout', 'cancelled'):
            FakeEngine.fail = status
            row = self.play(name=status + '.json')
            self.assertEqual(row['status'], status)
            self.assertIsNone(row['winner'])
        FakeEngine.fail = None
        FakeEngine.action = lambda state: -1
        self.assertEqual(self.play(name='illegal.json')['status'], 'illegal_move')
        FakeEngine.action = None
        FakeEngine.stopped = True
        stopped = self.play(name='stopped.json')
        self.assertEqual(stopped['status'], 'depth_incomplete')
        self.assertFalse(stopped['moves'])
        wall = self.play(self.spec(mode='wall'), name='wall.json')
        self.assertEqual(wall['status'], 'unfinished')
        self.assertTrue(all(e.closed for e in FakeEngine.instances))

    def test_real_crashed_hung_and_cancelled_children_are_reaped(self):
        baseline = {p.pid for p in mp.active_children()}
        for target, status in ((crash_worker, 'crash'), (hang_worker, 'infrastructure_timeout')):
            process = EngineProcess({}, {}, 1, target=target)
            try:
                with self.assertRaises(MatchFailure) as raised:
                    process.receive(time.perf_counter() + (10 if target == crash_worker else .1), threading.Event())
                self.assertEqual(raised.exception.status, status)
            finally:
                process.close()
        process = EngineProcess({}, {}, 1, target=hang_worker)
        cancelled = threading.Event()
        cancelled.set()
        try:
            with self.assertRaises(MatchFailure) as raised:
                process.receive(time.perf_counter() + 10, cancelled)
            self.assertEqual(raised.exception.status, 'cancelled')
        finally:
            process.close()
        self.assertEqual({p.pid for p in mp.active_children()}, baseline)

    def test_game_time_limit_is_unfinished_while_child_timeout_is_failure(self):
        class TimedOutMove(FakeEngine):
            def receive(self, deadline, cancelled):
                if self.state is not None:
                    raise MatchFailure('infrastructure_timeout', 'injected move timeout')
                return super().receive(deadline, cancelled)

        for seconds, expected in ((.01, 'unfinished'), (100., 'infrastructure_timeout')):
            spec = self.spec()
            spec['protocols'][0]['game_seconds'] = seconds
            row = play_match(spec, spec['tasks'][0], self.path / (expected + '.json'),
                             threading.Event(), engine_factory=TimedOutMove)
            self.assertEqual(row['status'], expected)
            self.assertIsNone(row['winner'])
            self.assertFalse(row['moves'])
        self.assertTrue(all(e.closed for e in FakeEngine.instances))

    def test_worker_completion_order_and_resume_do_not_change_quota(self):
        spec = self.spec(max_plies=2)
        original = play_match

        def delayed(spec, task, path, cancelled):
            if task['index'] == 0:
                time.sleep(.05)
            return original(spec, task, path, cancelled, engine_factory=FakeEngine)

        with patch('intransitive.tournament.runner.play_match', side_effect=delayed):
            parallel = run(spec, self.path / 'parallel', workers=2)
            serial = run(spec, self.path / 'serial', workers=1)
            resumed = run(spec, self.path / 'parallel', workers=1)
        self.assertEqual(parallel['scheduled_matches'], 2)
        self.assertEqual(resumed['final_matches'], 2)
        for task in spec['tasks']:
            name = task['id'] + '.json'
            a = json.loads((self.path / 'parallel' / 'matches' / name).read_text())
            b = json.loads((self.path / 'serial' / 'matches' / name).read_text())
            self.assertEqual(a['trajectory_sha256'], b['trajectory_sha256'])
        self.assertEqual(parallel['distinct_trajectories'], serial['distinct_trajectories'])

    def test_unfinished_cannot_improve_lower_score_or_eligibility(self):
        spec = self.spec()
        rows = []
        for task in spec['tasks']:
            row = play_match(spec, task, self.path / (task['id'] + '.json'), threading.Event(), engine_factory=FakeEngine)
            # Report-only synthetic outcomes isolate the ranking rule.
            row.update(status='win', winner=0)
            rows.append(row)
        before = report(spec, rows)['leaderboards']['depth']
        rows[0].update(status='unfinished', winner=None)
        after = report(spec, rows)['leaderboards']['depth']
        for current in after:
            previous = next(i for i in before if i['candidate'] == current['candidate'])
            self.assertLessEqual(current['win_points_lower'], previous['win_points_lower'])
            self.assertLessEqual(current['completion_rate'], previous['completion_rate'])
        self.assertEqual(after[0]['seed_clusters'], 1)
        self.assertEqual(after[0]['pairs'], 1)


if __name__ == '__main__':
    unittest.main()


class ChildReapingTests(unittest.TestCase):
    """Reaping a child must not fail a match it has already completed."""

    def test_close_survives_a_handle_that_reports_still_running(self):
        from intransitive.tournament.runner import EngineProcess
        engine = EngineProcess.__new__(EngineProcess)
        engine.connection = Mock()
        engine.process = Mock()
        engine.process.is_alive.return_value = False
        # multiprocessing raises this when its bookkeeping still believes the
        # child runs; under concurrent matches that happens for a dead child.
        engine.process.close.side_effect = [
            ValueError('Cannot close a process while it is still running.'), None]
        engine.close()
        self.assertEqual(engine.process.close.call_count, 2)
        engine.process.join.assert_called()

    def test_close_gives_up_quietly_if_the_handle_never_releases(self):
        from intransitive.tournament.runner import EngineProcess
        engine = EngineProcess.__new__(EngineProcess)
        engine.connection = Mock()
        engine.process = Mock()
        engine.process.is_alive.return_value = False
        engine.process.close.side_effect = ValueError('still running')
        engine.close()  # must not raise: the child is signalled and joined
        self.assertEqual(engine.process.close.call_count, 2)
