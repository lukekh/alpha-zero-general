"""Deterministic evolution, exact resume/cache identity, and conservative fitness."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from intransitive.evolution.runner import Search, batch, genome, jobs, prepare, run, validate
from intransitive.evolution.strategy import Settings, initialize, mutate, next_population
from intransitive.evolution.engines import EnginePool, ResettableConnection
from intransitive.heuristics.tuning import Genome
from intransitive.tests.test_tournament import FakeEngine
from intransitive.tournament.runner import play_match
from intransitive.tournament.spec import generate_positions, protocol


def safe_preflight(g, states, *, base):
    return dict(flagged=False, config_hash=g.config_hash)


def fake_match(spec, task, path, cancelled):
    return play_match(spec, task, path, cancelled, engine_factory=FakeEngine)


class EvolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.positions = generate_positions(54, 9)

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name)
        FakeEngine.instances = []
        FakeEngine.action = FakeEngine.cancel = FakeEngine.fail = None
        FakeEngine.stopped = False

    def tearDown(self):
        self.folder.cleanup()

    def spec(self, **overrides):
        settings = replace(Settings(population=2, generations=2, search_positions=1,
                                    max_games=100, max_nodes=100_000_000), **overrides)
        return prepare(settings, self.positions,
                       protocol('depth', depth=1, seconds=10., node_limit=10000,
                                proof_depth=0, proof_nodes=0, max_plies=2), revision='test-revision')

    def run_search(self, spec, name='run', **kwargs):
        return run(spec, self.path / name, preflight=safe_preflight,
                   match_player=kwargs.pop('match_player', fake_match), **kwargs)

    def state(self, name='run'):
        return json.loads((self.path / name / 'checkpoint.json').read_text())

    def test_seeded_initialization_selection_mutation_and_zero_activation(self):
        settings = Settings(population=6)
        a, b = random.Random(55), random.Random(55)
        pa, pb = initialize(a, settings, set()), initialize(b, settings, set())
        self.assertEqual(pa, pb)
        ranking = [g.config_hash for g in pa]
        na = next_population(pa, ranking, a, settings, set())
        nb = next_population(pb, ranking, b, settings, set())
        self.assertEqual(na, nb)
        self.assertEqual(na[0], pa[0])
        toggles = replace(settings, mutation_rate=1., toggle_rate=1.)
        activated = mutate(Genome.from_genes(), random.Random(1), toggles)
        self.assertEqual(activated.values[0], 0.)
        original = Genome.from_genes().values
        for before, after in zip(original, activated.values):
            self.assertEqual(after == 0., before != 0.)
        for _ in range(100):
            activated = mutate(activated, a, replace(settings, mutation_sigma=1e300))
            self.assertEqual(Genome.from_json(activated.to_json()), activated)
            self.assertTrue(all(-100 <= v <= 100 for v in activated.values))

    def test_invalid_settings_and_genomes_fail_before_matches(self):
        for values in ({'mutation_rate': float('nan')}, {'mutation_sigma': float('inf')},
                       {'max_seconds': True}, {'max_nodes': -1}, {'population': 1},
                       {'elites': 4}, {'tournament_size': 5}, {'random_off_rate': -1}):
            with self.assertRaises(ValueError):
                Settings(**values)
        for genes in ({'pressure': 101}, {'attack': float('nan')}, {'race': 1}):
            with self.assertRaises(ValueError):
                Genome.from_genes(genes)
        spec = self.spec()
        validate(spec)
        for field in ('revision', 'implementation', 'backend'):
            changed = deepcopy(spec)
            changed[field] = 'changed'
            with self.assertRaises(ValueError):
                validate(changed)
        with self.assertRaises(ValueError):
            self.spec(generations=1000)

    def test_common_search_fresh_validation_and_no_heldout(self):
        spec = self.spec()
        search = Search(spec, self.path, threading.Event(), preflight=safe_preflight)
        pop = [genome(g) for g in search.state['population']]
        first = batch(spec, pop, [], 0, 'search')
        second = batch(spec, pop, [], 1, 'search')
        self.assertEqual(first, second)
        val_a = batch(spec, pop, [], 0, 'validation')
        val_b = batch(spec, pop, [], 1, 'validation')
        self.assertFalse(set(val_a['selected_positions']) & set(val_b['selected_positions']))
        self.assertFalse({p['seed'] for p in val_a['positions']} & {p['seed'] for p in val_b['positions']})
        self.assertTrue(all(p['pool'] != 'heldout' for s in (first, val_a, val_b) for p in s['positions']))
        self.assertEqual(len(first['tasks']), 10)  # contemporaries + incumbent + archive, paired

    def test_full_match_cache_identity(self):
        spec = self.spec()
        pop = [Genome.from_genes({'advantage': 30}), Genome.from_genes({'advantage': 40})]
        schedule = batch(spec, pop, [], 0, 'search')
        original = {key for _, key, _, _ in jobs(schedule)}
        # Same pair survives surrounding population changes and task-index changes.
        bigger = batch(spec, pop + [Genome.from_genes({'advantage': 60})], [], 0, 'search')
        self.assertTrue(original < {key for _, key, _, _ in jobs(bigger)})
        for change in ('node_limit', 'time_limit'):
            altered = deepcopy(schedule)
            altered['protocols'][0]['search'][change] *= 2
            self.assertFalse(original & {key for _, key, _, _ in jobs(altered)})
        altered = deepcopy(schedule)
        altered['positions'][0]['seed'] += 1
        for task in altered['tasks']:
            task['seed'] += 1
        self.assertFalse(original & {key for _, key, _, _ in jobs(altered)})
        val = batch(spec, pop, [], 0, 'validation')
        self.assertFalse(original & {key for _, key, _, _ in jobs(val)})

    def test_cancel_resume_exact_population_rng_and_no_duplicate_evidence(self):
        spec = self.spec()
        fresh = self.run_search(spec, 'fresh')
        event, count = threading.Event(), 0

        def interrupt(spec, task, path, cancelled):
            nonlocal count
            count += 1
            if count == 4:
                FakeEngine.cancel = event
            row = fake_match(spec, task, path, cancelled)
            FakeEngine.cancel = None
            return row

        partial = self.run_search(spec, cancelled=event, match_player=interrupt)
        self.assertEqual(partial['status'], 'cancelled')
        self.assertTrue(all(e.closed for e in FakeEngine.instances))
        resumed = self.run_search(spec)
        a, b = self.state('fresh'), self.state()
        for key in ('rng', 'population', 'hall', 'generation', 'phase', 'seen'):
            self.assertEqual(a[key], b[key], key)
        self.assertEqual(set(a['ledger']), set(b['ledger']))
        self.assertEqual({k: v['trajectory'] for k, v in a['ledger'].items()},
                         {k: v['trajectory'] for k, v in b['ledger'].items()})
        self.assertEqual(fresh['unique_matches'], resumed['unique_matches'])
        self.assertGreater(resumed['games_reserved'], fresh['games_reserved'])
        before = len(FakeEngine.instances)
        self.run_search(spec)
        self.assertEqual(len(FakeEngine.instances), before)
        self.assertEqual(len(a['ledger']), len(set(a['ledger'])))
        self.assertEqual(resumed['eligible_exports'], 0)
        self.assertTrue(all(e['acceptance'].startswith('UNACCEPTED') for e in b['exports']))

    def test_crash_between_final_journal_and_checkpoint_is_reconciled(self):
        spec = self.spec(generations=1)
        calls = 0

        def crash(spec, task, path, cancelled):
            nonlocal calls
            calls += 1
            fake_match(spec, task, path, cancelled)
            raise RuntimeError('power loss after journal commit')

        with self.assertRaisesRegex(RuntimeError, 'power loss'):
            self.run_search(spec, match_player=crash)
        self.assertEqual(self.state()['games_reserved'], 1)
        self.assertFalse(self.state()['ledger'])
        result = self.run_search(spec)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['games_reserved'], result['unique_matches'])

    def test_game_node_wall_budgets_and_changed_resume(self):
        result = self.run_search(self.spec(max_games=1), 'games')
        self.assertEqual(result['status'], 'game_budget')
        self.assertEqual(result['games_reserved'], 1)
        result = self.run_search(self.spec(max_nodes=1), 'nodes')
        self.assertEqual(result['status'], 'node_budget')
        self.assertEqual(result['games_reserved'], 0)
        result = self.run_search(self.spec(max_seconds=.000001), 'wall')
        self.assertEqual(result['status'], 'wall_budget')
        self.assertEqual(result['games_reserved'], 0)
        with self.assertRaises(ValueError):
            self.run_search(self.spec(max_games=2), 'games')

    def test_saturation_and_low_depth_never_become_eligible(self):
        spec = self.spec(generations=1)
        FakeEngine.stopped = True
        result = run(spec, self.path / 'depth', match_player=fake_match,
                     preflight=lambda *a, **kw: dict(flagged=True))
        self.assertGreater(result['failures'], 0)
        self.assertEqual(result['eligible_exports'], 0)
        self.assertFalse(self.state('depth')['hall'])
        for generation in result['convergence']:
            for phase in ('search', 'validation'):
                self.assertTrue(all(not r['eligible'] for r in generation[phase]['leaderboards']['depth']))

    def test_hall_requires_search_and_fresh_validation_eligibility(self):
        from intransitive.tournament.report import report as harness_report

        def scores(spec, rows):
            # Synthetic report-only outcomes isolate archive admission. The
            # underlying journals still contain replayable unfinished games.
            result = harness_report(spec, rows)
            for entry in result['leaderboards']['depth']:
                entry['eligible'] = spec['pool'] == 'search'
            return result

        with patch('intransitive.evolution.runner.report', side_effect=scores):
            self.run_search(self.spec(generations=1), 'rejected')
        self.assertFalse(self.state('rejected')['hall'])

        def eligible_scores(spec, rows):
            result = harness_report(spec, rows)
            for entry in result['leaderboards']['depth']:
                entry['eligible'] = True
            return result

        with patch('intransitive.evolution.runner.report', side_effect=eligible_scores):
            self.run_search(self.spec(hall_size=1), 'admitted')
        state = self.state('admitted')
        self.assertEqual(len(state['hall']), 1)
        self.assertEqual(len(state['history'][1]['hall_before']), 1)
        self.assertEqual(state['hall'][0], state['exports'][-1]['genome'])

    def test_incomplete_wall_depth_and_saturation_override_successful_fitness(self):
        from intransitive.tournament.report import report as harness_report
        base = self.spec(generations=1, min_completed_depth=2)
        limits = protocol('wall', seconds=.05, node_limit=10000, max_plies=2)
        spec = prepare(Settings(**base['settings']), self.positions, limits, revision='test-revision')

        def successful(spec, rows):
            result = harness_report(spec, rows)
            for entry in result['leaderboards']['wall']:
                entry['eligible'] = True
            return result

        with patch('intransitive.evolution.runner.report', side_effect=successful):
            self.run_search(spec, 'low')
            run(spec, self.path / 'saturated', match_player=fake_match,
                preflight=lambda *a, **kw: dict(flagged=True))
        for name in ('low', 'saturated'):
            state = self.state(name)
            self.assertFalse(state['hall'])
            self.assertTrue(all(not e['eligible'] for e in state['exports']))
            self.assertTrue(all(e['validation']['depth_violations'] > 0 for e in state['exports']))

    def test_parallel_matches_reproduce_the_serial_board(self):
        """Workers change throughput, never the result.

        Each match is self-contained and carries its own seed, so completion
        order must not reach the board. This runs the same search serially and
        on four workers and compares what the run actually concluded.
        """
        live, peak, guard = [0], [0], threading.Lock()

        def watched(spec, task, path, cancelled):
            with guard:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            try:
                return fake_match(spec, task, path, cancelled)
            finally:
                with guard:
                    live[0] -= 1

        self.run_search(self.spec(match_workers=1), name='serial')
        serial = self.state('serial')
        FakeEngine.instances = []
        self.run_search(self.spec(match_workers=4, resident_engines=12),
                        name='parallel', match_player=watched)
        parallel = self.state('parallel')

        self.assertGreater(peak[0], 1, 'matches never actually overlapped')
        def outcomes(state):
            return {key: (row['status'], row['trajectory'])
                    for key, row in state['ledger'].items()}
        self.assertEqual(outcomes(serial), outcomes(parallel))
        # Clocks/memory differ by nature; run_manifest differs because
        # match_workers IS a recorded setting, so the two runs are honestly
        # different manifests. Nothing the search decides may differ.
        volatile = ('seconds', 'bytes', 'elapsed', 'latency', 'reserved', 'lease',
                    'run_manifest')

        def stable(value):
            if isinstance(value, dict):
                return {k: stable(v) for k, v in value.items()
                        if not any(mark in k for mark in volatile)}
            return [stable(v) for v in value] if isinstance(value, list) else value

        self.assertEqual(stable(serial['history']), stable(parallel['history']))
        self.assertEqual(serial['population'], parallel['population'])
        self.assertEqual(stable(serial['exports']), stable(parallel['exports']))

    def test_long_real_runs_require_smoke_before_launch(self):
        with self.assertRaisesRegex(ValueError, 'bounded smoke first'):
            run(self.spec(), self.path / 'long')
        self.assertFalse(FakeEngine.instances)

    def test_process_pool_isolates_candidates_resets_seeds_and_reaps(self):
        processes = []

        def factory(*args, **kwargs):
            process = Mock()
            processes.append(process)
            return process

        pool = EnginePool(2, factory)
        a = pool.acquire({'sha256': 'A'}, {'depth': 1}, 55)
        b = pool.acquire({'sha256': 'B'}, {'depth': 1}, 55)
        a.close()
        reused = pool.acquire({'sha256': 'A'}, {'depth': 1}, 56)
        self.assertIs(reused.process, a.process)
        a.process.connection.send.assert_called_once_with({'reset_seed': 56})
        reused.close()
        c = pool.acquire({'sha256': 'C'}, {'depth': 1}, 56)
        a.process.close.assert_called_once()
        b.process.close.assert_not_called()  # An active lease cannot be evicted.
        c.process.receive.side_effect = RuntimeError('crashed')
        with self.assertRaises(RuntimeError):
            c.receive(0., threading.Event())
        c.close()
        c.process.close.assert_called_once()
        b.close()
        changed_limits = pool.acquire({'sha256': 'B'}, {'depth': 2}, 56)
        self.assertIsNot(changed_limits.process, b.process)
        changed_limits.close()
        pool.close()
        self.assertTrue(all(p.close.call_count == 1 for p in processes))

    def test_reused_worker_seed_handshake(self):
        import numpy as np
        connection = Mock()
        connection.recv.side_effect = [{'reset_seed': 56}, 'next-observation']
        adapter = ResettableConnection(connection)
        adapter.send(dict(kind='ready', candidate='A', config={'depth': 1},
                          startup_seconds=20., cpu_seconds=18.))
        self.assertEqual(adapter.recv(), 'next-observation')
        ready = connection.send.call_args.args[0]
        self.assertEqual((ready['candidate'], ready['startup_seconds'], ready['cpu_seconds']), ('A', 0., 0.))
        self.assertTrue(ready['reused_process'])
        self.assertEqual(random.random(), random.Random(56).random())
        self.assertEqual(np.random.random(), np.random.RandomState(56).random())

    def test_saved_evidence_audit_detects_modified_journal(self):
        from intransitive.benchmarks.evolution.verify import verify
        result = self.run_search(self.spec(generations=1))
        audited = verify(self.path / 'run')
        self.assertEqual(audited['verified_matches'], result['unique_matches'])
        entry = next(iter(self.state()['ledger'].values()))
        path = self.path / 'run' / entry['record']
        row = json.loads(path.read_text())
        row['reason'] = 'modified'
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            verify(self.path / 'run')

    def test_random_control_same_initialization_and_budget(self):
        a, b = self.spec(), self.spec(algorithm='random')
        sa = Search(a, self.path / 'a', threading.Event())
        sb = Search(b, self.path / 'b', threading.Event())
        self.assertEqual(sa.state['population'], sb.state['population'])
        for key in ('max_games', 'max_nodes', 'max_seconds'):
            self.assertEqual(a['settings'][key], b['settings'][key])
        evolved, random_control = self.run_search(a, 'a'), self.run_search(b, 'b')
        self.assertEqual(evolved['status'], 'complete')
        self.assertEqual(random_control['status'], 'complete')
        self.assertNotEqual(self.state('a')['population'], self.state('b')['population'])


if __name__ == '__main__':
    unittest.main()


class ConcurrentEngineTests(unittest.TestCase):
    """The pool must serve concurrent matches without sharing an engine."""

    @staticmethod
    def factory(store):
        def build(*args, **kwargs):
            process = Mock()
            store.append(process)
            return process
        return build

    def test_one_candidate_can_play_two_games_at_once(self):
        processes = []
        pool = EnginePool(5, self.factory(processes), workers=2)
        first = pool.acquire({'sha256': 'A'}, {'depth': 1}, 55)
        second = pool.acquire({'sha256': 'A'}, {'depth': 1}, 56)
        # Same candidate, concurrent games: two processes, never one shared.
        self.assertIsNot(first.process, second.process)
        self.assertEqual(pool.live, 2)
        first.close()
        second.close()
        # Both return to the same bucket and are reused before anything new.
        third = pool.acquire({'sha256': 'A'}, {'depth': 1}, 57)
        self.assertIn(third.process, (first.process, second.process))
        self.assertEqual(pool.live, 2)
        third.close()

    def test_capacity_covers_two_engines_for_every_worker(self):
        with self.assertRaisesRegex(ValueError, 'cover two per concurrent match'):
            EnginePool(2, self.factory([]), workers=2)
        with self.assertRaisesRegex(ValueError, 'cover two per concurrent match'):
            Settings(resident_engines=2, match_workers=2)
        # Exactly two per worker leaves no slot to cache a warm engine, so every
        # match would pay process startup again: that is rejected too.
        with self.assertRaisesRegex(ValueError, 'exceed two per concurrent match'):
            EnginePool(4, self.factory([]), workers=2)
        with self.assertRaisesRegex(ValueError, 'exceed two per concurrent match'):
            Settings(resident_engines=4, match_workers=2)
        self.assertEqual(Settings(resident_engines=5, match_workers=2).match_workers, 2)
        self.assertEqual(EnginePool(16, self.factory([]), workers=4).capacity, 16)

    def test_concurrent_matches_never_share_an_engine(self):
        processes = []
        pool = EnginePool(9, self.factory(processes), workers=4)
        seen, clashes = set(), []
        barrier = threading.Barrier(4)

        def match(index):
            # Two leases per match, held together, exactly as play_match does.
            held = [pool.acquire({'sha256': f'C{index % 2}'}, {'depth': 1}, index),
                    pool.acquire({'sha256': f'C{(index + 1) % 2}'}, {'depth': 1}, index)]
            barrier.wait(timeout=10)  # force all four matches to overlap
            for lease in held:
                if lease.process in seen:
                    clashes.append(lease.process)
                seen.add(lease.process)
            barrier.wait(timeout=10)
            for lease in held:
                lease.close()

        threads = [threading.Thread(target=match, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        self.assertFalse(any(t.is_alive() for t in threads), 'acquire deadlocked')
        self.assertEqual(clashes, [], 'an engine was shared across concurrent games')
        self.assertEqual(pool.live, 8)
        self.assertEqual(pool.active, 0)
        pool.close()

    def test_broken_lease_is_reaped_and_not_reused(self):
        processes = []
        pool = EnginePool(5, self.factory(processes), workers=2)
        lease = pool.acquire({'sha256': 'A'}, {'depth': 1}, 55)
        lease.process.receive.side_effect = RuntimeError('crashed')
        with self.assertRaises(RuntimeError):
            lease.receive(0., threading.Event())
        lease.close()
        lease.process.close.assert_called_once()
        self.assertEqual(pool.live, 0)
        replacement = pool.acquire({'sha256': 'A'}, {'depth': 1}, 56)
        self.assertIsNot(replacement.process, lease.process)
