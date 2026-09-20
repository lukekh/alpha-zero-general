"""Branch-parallel native search: interlocks, equivalence and reproducibility.

Determinism is claimed per thread count, for searches that finish inside their
limits. These tests assert exactly that and nothing wider: a fixed thread count
repeats itself, every thread count agrees on the answer, and a request that
cannot be run is refused before the search starts rather than during it.
"""
import unittest

import numpy as np

from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveLogicNumba import search_observation
from intransitive.rust_teacher import BINARY, RustTeacher
from intransitive.tests.test_heuristics import position

THREADS = (1, 2, 3, 4)
LIMITS = dict(seconds=300., proof_nodes=0, weight=0., table_entries=20000, node_limit=10**9)


def fixtures():
    game = IntransitiveGame(modelling_draws=False)
    state, side = game.getInitBoard(), 0
    rng = np.random.default_rng(650920)
    for _ in range(24):
        action = int(rng.choice(np.flatnonzero(game.getValidMoves(state, side))))
        state, side = game.getNextState(state, side, action)
    return [('opening', game.getInitBoard()), ('midgame', state),
            ('endgame', position({'C3': 1, 'D3': 2, 'C4': 3, 'G7': -1, 'F7': -2, 'G6': -3}))]


class ParallelInterlockTests(unittest.TestCase):
    """Refusals need no binary: they are stated by the client before any search."""

    def settings(self, **kwargs):
        return dict(dict(threads=1, split_min_depth=4, split_min_siblings=2,
                         table_entries=50000, futility_max_depth=2), **kwargs)

    def test_thread_and_memory_interlocks_state_their_reason(self):
        for kwargs, expected in [
                (dict(threads=0), 'threads must be an integer in 1..64'),
                (dict(threads=65), 'threads must be an integer in 1..64'),
                (dict(threads=True), 'threads must be an integer in 1..64'),
                (dict(threads=2.0), 'threads must be an integer in 1..64'),
                (dict(split_min_depth=1), 'split_min_depth must be an integer in 2..32'),
                (dict(split_min_depth=2), 'split_min_depth must exceed futility_max_depth'),
                (dict(split_min_siblings=1), 'split_min_siblings must be an integer in 2..64'),
                (dict(threads=4, table_entries=500_000),
                 'threads * table_entries must be <= 1000000')]:
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError) as caught:
                    RustTeacher.parallel_settings(**self.settings(**kwargs))
                self.assertTrue(str(caught.exception).startswith(expected), caught.exception)

    def test_defaults_are_recognised_as_the_unextended_wire_form(self):
        # False here means the request keeps the wire shape older binaries
        # understand, which is what makes threads = 1 a compatibility promise.
        self.assertFalse(RustTeacher.parallel_settings(**self.settings()))
        self.assertTrue(RustTeacher.parallel_settings(**self.settings(threads=2)))
        self.assertTrue(RustTeacher.parallel_settings(**self.settings(split_min_depth=5)))
        self.assertTrue(RustTeacher.parallel_settings(**self.settings(split_min_siblings=3)))
        self.assertTrue(RustTeacher.parallel_settings(**self.settings(threads=8, table_entries=125_000)))
        self.assertTrue(RustTeacher.parallel_settings(**self.settings(split_min_depth=2,
                                                                     futility_max_depth=1)))


@unittest.skipUnless(BINARY.exists(), 'build Rust first')
class ParallelSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.teacher = RustTeacher()
        cls.states = [(stage, search_observation(state)) for stage, state in fixtures()]

    @classmethod
    def tearDownClass(cls):
        cls.teacher.close()

    def analyze(self, state, threads, depth=5, **kwargs):
        return self.teacher.analyze(state, depth=depth, threads=threads, **dict(LIMITS, **kwargs))

    def test_one_thread_never_splits_and_keeps_the_sequential_stop_reasons(self):
        for stage, state in self.states:
            with self.subTest(stage=stage):
                result = self.analyze(state, 1)
                self.assertEqual(result['parallel']['split_nodes'], 0)
                self.assertEqual(result['parallel']['helper_nodes'], 0)
                self.assertEqual(result['score_bound'], 'exact')
                self.assertIn(result['stop_reason'], ('maximum_depth', 'proven_result'))
                self.assertEqual(result['search_identity']['threads'], 1)

    def test_every_thread_count_agrees_on_the_answer(self):
        for stage, state in self.states:
            reference = self.analyze(state, 1)
            for threads in THREADS[1:]:
                with self.subTest(stage=stage, threads=threads):
                    result = self.analyze(state, threads)
                    self.assertEqual(result['score'], reference['score'])
                    self.assertEqual(result['action'], reference['action'])
                    self.assertEqual(result['completed_depth'], reference['completed_depth'])
                    self.assertEqual(result['search_identity']['threads'], threads)

    def test_a_fixed_thread_count_reproduces_itself(self):
        for stage, state in self.states:
            for threads in THREADS:
                with self.subTest(stage=stage, threads=threads):
                    runs = [self.analyze(state, threads) for _ in range(3)]
                    keys = {(r['score'], r['action'], tuple(r['pv']), r['nodes'],
                             r['completed_depth'], r['stop_reason']) for r in runs}
                    self.assertEqual(len(keys), 1, keys)

    def test_splitting_actually_happens_and_is_charged_to_the_search(self):
        splits = 0
        for stage, state in self.states:
            result = self.analyze(state, 4, depth=6)
            report = result['parallel']
            self.assertEqual(report['threads'], 4)
            if report['split_nodes']:
                splits += 1
                self.assertGreater(report['parallel_siblings'], 0)
                # Helper work is part of the reported total, never free.
                self.assertGreater(report['helper_nodes'], 0)
                self.assertGreater(result['nodes'], report['helper_nodes'])
        self.assertGreater(splits, 0, 'no fixture ever reached a split point')

    def test_the_node_budget_bounds_every_thread_together(self):
        for cap in (500, 5000, 50000):
            for threads in THREADS:
                with self.subTest(cap=cap, threads=threads):
                    result = self.analyze(self.states[1][1], threads, depth=8, node_limit=cap)
                    self.assertLessEqual(result['nodes'], cap)
                    self.assertLessEqual(result['work'], cap)

    def test_a_parallel_result_is_never_offered_as_a_certificate(self):
        # A blue runner three plies from I9: a genuine proven win, not a
        # position that was already terminal before the search started. The
        # teacher-label paths accept 'proven_result' and 'exact', so a search
        # with helpers running must not be able to present itself as either.
        state = search_observation(position({'F6': 3, 'A5': -1, 'B4': -1}))
        for threads in THREADS:
            with self.subTest(threads=threads):
                result = self.analyze(state, threads, depth=5, proof_nodes=64)
                self.assertGreater(result['score'], 90000., 'fixture stopped proving a win')
                self.assertEqual(result['score_bound'], 'exact' if threads == 1 else 'parallel_exact')
                self.assertEqual(result['stop_reason'],
                                 'proven_result' if threads == 1 else 'parallel_result')

    def test_the_search_identity_records_the_parallel_settings(self):
        identity = self.analyze(self.states[0][1], 3, split_min_depth=5)['search_identity']
        self.assertEqual(identity['threads'], 3)
        self.assertEqual(identity['split_min_depth'], 5)
        self.assertEqual(identity['split_min_siblings'], 2)
        self.assertTrue(identity['deterministic_for_thread_count'])

    def test_a_native_genome_can_only_ask_for_a_single_threaded_search(self):
        from intransitive.heuristics.tuning import Genome
        arguments = Genome.from_genes({}, backend='rust').native_arguments()
        self.assertEqual(arguments['threads'], 1)


if __name__ == '__main__':
    unittest.main()
