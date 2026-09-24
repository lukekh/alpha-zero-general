"""Unit tests for the off-policy replay pool.

These cover the pool, the replay network and the sweep helpers with a stub
game, so they need neither a checkpoint nor the compiled rules engine. The
end-to-end guarantee -- that replaying the recording configuration reproduces
the recorded decisions exactly -- is checked by `python -m replay check`.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from replay.bench import wall_clock_per_move
from replay.record import load_positions, sample_action, save_positions, split_of
from replay.simulator import ReplayMiss, ReplayNet
from replay.store import EvalStore, PoolError, state_key
from replay.sweep import expand_grid, make_args

ACTIONS = 8
PLAYERS = 2


class StubGame:
    num_players = PLAYERS

    def getActionSize(self):
        return ACTIONS

    def stringRepresentation(self, board):
        return np.asarray(board).tobytes()


def mask_of(*indices):
    mask = np.zeros(ACTIONS, dtype=bool)
    mask[list(indices)] = True
    return mask


class StateKeyTests(unittest.TestCase):
    def test_key_is_stable_and_distinguishing(self):
        game = StubGame()
        one = np.arange(4, dtype=np.int8)
        again = np.arange(4, dtype=np.int8)
        other = np.arange(1, 5, dtype=np.int8)
        self.assertEqual(state_key(game, one), state_key(game, again))
        self.assertNotEqual(state_key(game, one), state_key(game, other))
        self.assertEqual(len(state_key(game, one)), 16)

    def test_string_representation_may_be_text(self):
        class TextGame(StubGame):
            def stringRepresentation(self, board):
                return 'abc'

        self.assertEqual(len(state_key(TextGame(), None)), 16)


class EvalStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = EvalStore(ACTIONS, PLAYERS)
        self.mask = mask_of(1, 3, 5)
        self.pi = np.zeros(ACTIONS, dtype=np.float32)
        self.pi[self.mask] = [0.2, 0.3, 0.5]
        self.value = np.array([0.25, -0.25], dtype=np.float32)

    def test_masked_policy_round_trips_exactly(self):
        self.assertTrue(self.store.add(b'k', self.pi, self.value, self.mask))
        self.assertEqual(len(self.store.sparse), 1)
        self.assertEqual(self.store.mask_violations, 0)
        policy, value = self.store.get(b'k', self.mask)
        np.testing.assert_array_equal(policy, self.pi)
        np.testing.assert_array_equal(value, self.value)

    def test_returned_arrays_are_fresh_copies(self):
        # MCTS renormalises the policy in place and keeps it in its own tree.
        self.store.add(b'k', self.pi, self.value, self.mask)
        first, _ = self.store.get(b'k', self.mask)
        first *= 0.0
        second, _ = self.store.get(b'k', self.mask)
        np.testing.assert_array_equal(second, self.pi)

    def test_mass_outside_the_mask_is_kept_densely_and_counted(self):
        leaky = self.pi.copy()
        leaky[0] = 0.01
        self.store.add(b'k', leaky, self.value, self.mask)
        self.assertEqual(self.store.mask_violations, 1)
        self.assertEqual(len(self.store.sparse), 0)
        policy, _ = self.store.get(b'k', self.mask)
        np.testing.assert_array_equal(policy, leaky)

    def test_duplicate_add_is_ignored(self):
        self.assertTrue(self.store.add(b'k', self.pi, self.value, self.mask))
        self.assertFalse(self.store.add(b'k', self.pi * 0, self.value, self.mask))

    def test_miss_returns_none(self):
        self.assertIsNone(self.store.get(b'absent', self.mask))

    def test_disagreeing_legal_count_is_refused(self):
        self.store.add(b'k', self.pi, self.value, self.mask)
        with self.assertRaises(PoolError):
            self.store.get(b'k', mask_of(1, 3))

    def test_wrong_action_size_is_refused(self):
        with self.assertRaises(PoolError):
            self.store.add(b'k', np.zeros(3, dtype=np.float32), self.value, self.mask)

    def test_save_load_round_trip_verifies_checksums(self):
        self.store.add(b'sparse-key-0000', self.pi, self.value, self.mask)
        leaky = self.pi.copy()
        leaky[0] = 0.5
        self.store.add(b'dense-key-00000', leaky, self.value, self.mask)
        self.store.metadata['note'] = 'round trip'
        with tempfile.TemporaryDirectory() as folder:
            manifest = self.store.save(folder, shard_entries=1)
            self.assertEqual(manifest['unique_states'], 2)
            self.assertGreaterEqual(len(manifest['shards']), 2)
            loaded, reloaded = EvalStore.load(folder)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded.metadata['note'], 'round trip')
            self.assertEqual(reloaded['mask_violations'], 1)
            np.testing.assert_array_equal(loaded.get(b'sparse-key-0000', self.mask)[0], self.pi)
            np.testing.assert_array_equal(loaded.get(b'dense-key-00000', self.mask)[0], leaky)

            corrupt = Path(folder) / manifest['shards'][0]['file']
            corrupt.write_bytes(corrupt.read_bytes() + b'x')
            with self.assertRaises(PoolError):
                EvalStore.load(folder)

    def test_measure_reports_counts_and_sizes(self):
        self.store.add(b'k', self.pi, self.value, self.mask)
        measurement = self.store.measure()
        self.assertEqual(measurement['unique_states'], 1)
        self.assertEqual(measurement['mean_legal_moves'], 3.0)
        self.assertGreater(measurement['bytes_per_state'], 0)


class ReplayNetTests(unittest.TestCase):
    def setUp(self):
        self.game = StubGame()
        self.board = np.arange(4, dtype=np.int8)
        self.mask = mask_of(0, 2)
        self.store = EvalStore(ACTIONS, PLAYERS)
        self.pi = np.zeros(ACTIONS, dtype=np.float32)
        self.pi[self.mask] = [0.4, 0.6]
        self.store.add(state_key(self.game, self.board), self.pi,
                       np.array([1.0, -1.0], dtype=np.float32), self.mask)

    def test_hit_is_counted_and_exact(self):
        net = ReplayNet(self.store, self.game)
        policy, value = net.predict(self.board, self.mask)
        np.testing.assert_array_equal(policy, self.pi)
        self.assertEqual((net.hits, net.misses), (1, 0))
        self.assertEqual(net.miss_rate, 0.0)

    def test_uniform_fallback_counts_the_miss(self):
        net = ReplayNet(self.store, self.game)
        policy, value = net.predict(np.array([9, 9, 9, 9], dtype=np.int8), self.mask)
        self.assertEqual((net.hits, net.misses, net.distinct_misses), (0, 1, 1))
        np.testing.assert_allclose(policy[self.mask], [0.5, 0.5])
        np.testing.assert_array_equal(value, np.zeros(PLAYERS, dtype=np.float32))
        self.assertEqual(net.miss_rate, 1.0)

    def test_repeat_miss_reuses_the_side_cache(self):
        net = ReplayNet(self.store, self.game)
        absent = np.array([9, 9, 9, 9], dtype=np.int8)
        net.predict(absent, self.mask)
        net.predict(absent, self.mask)
        self.assertEqual((net.misses, net.distinct_misses), (2, 1))

    def test_strict_mode_raises_on_a_miss(self):
        net = ReplayNet(self.store, self.game, on_miss='strict')
        with self.assertRaises(ReplayMiss):
            net.predict(np.array([9, 9, 9, 9], dtype=np.int8), self.mask)

    def test_delegate_mode_needs_a_fallback(self):
        with self.assertRaises(ValueError):
            ReplayNet(self.store, self.game, on_miss='delegate')

    def test_delegate_mode_uses_the_real_network(self):
        class Fallback:
            def predict(self, board, valid_actions):
                return np.full(ACTIONS, 0.125, dtype=np.float32), np.zeros(PLAYERS, np.float32)

        net = ReplayNet(self.store, self.game, on_miss='delegate', fallback=Fallback())
        policy, _ = net.predict(np.array([9, 9, 9, 9], dtype=np.int8), self.mask)
        np.testing.assert_allclose(policy, 0.125)
        self.assertEqual(net.misses, 1)

    def test_unknown_miss_policy_is_refused(self):
        with self.assertRaises(ValueError):
            ReplayNet(self.store, self.game, on_miss='guess')


class SplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_roughly_proportional(self):
        first = [split_of(i, 7, 0.5) for i in range(500)]
        self.assertEqual(first, [split_of(i, 7, 0.5) for i in range(500)])
        self.assertAlmostEqual(first.count('select') / 500, 0.5, delta=0.08)

    def test_fraction_is_honoured(self):
        share = [split_of(i, 0, 0.25) for i in range(500)].count('select') / 500
        self.assertAlmostEqual(share, 0.25, delta=0.06)

    def test_seed_changes_the_assignment(self):
        self.assertNotEqual([split_of(i, 0, 0.5) for i in range(200)],
                            [split_of(i, 1, 0.5) for i in range(200)])


class SampleActionTests(unittest.TestCase):
    def test_zero_temperature_is_greedy(self):
        rng = np.random.default_rng(0)
        self.assertEqual(sample_action([0.1, 0.7, 0.2], 0.0, rng), 1)

    def test_sampling_is_reproducible_for_a_seed(self):
        distribution = [0.25, 0.5, 0.25]
        first = [sample_action(distribution, 1.0, np.random.default_rng(3)) for _ in range(5)]
        again = [sample_action(distribution, 1.0, np.random.default_rng(3)) for _ in range(5)]
        self.assertEqual(first, again)


class SweepHelperTests(unittest.TestCase):
    def test_expand_grid_is_a_sorted_cartesian_product(self):
        grid = expand_grid({'numMCTSSims': [1, 2], 'cpuct': [0.5]})
        self.assertEqual(grid, [{'cpuct': 0.5, 'numMCTSSims': 1},
                                {'cpuct': 0.5, 'numMCTSSims': 2}])

    def test_empty_grid_yields_one_empty_config(self):
        self.assertEqual(expand_grid({}), [{}])

    def test_make_args_layers_base_then_overrides(self):
        args = make_args({'cpuct': 2.0}, {'numMCTSSims': 64})
        self.assertEqual(args.cpuct, 2.0)
        self.assertEqual(args.numMCTSSims, 64)
        self.assertEqual(args.prob_fullMCTS, 1.0)

    def test_unknown_setting_is_refused(self):
        with self.assertRaises(ValueError):
            make_args({}, {'cpcut': 1.0})


class WallClockModelTests(unittest.TestCase):
    """The measured curve for this repo is flat, so collecting leaves per call
    cannot save time; these pin that the model says so rather than assuming it."""

    FLAT = [dict(batch=1, median_seconds=0.001157),
            dict(batch=8, median_seconds=0.009039),
            dict(batch=32, median_seconds=0.036463)]

    def test_collection_one_is_one_call_per_evaluation(self):
        self.assertAlmostEqual(wall_clock_per_move(18, 1, self.FLAT), 18 * 0.001157)

    def test_partial_final_batch_still_costs_a_whole_call(self):
        # ceil(18 / 8) = 3
        self.assertAlmostEqual(wall_clock_per_move(18, 8, self.FLAT), 3 * 0.009039)

    def test_no_work_costs_nothing(self):
        self.assertEqual(wall_clock_per_move(0, 8, self.FLAT), 0)

    def test_a_flat_curve_makes_collection_a_penalty(self):
        sequential = wall_clock_per_move(18, 1, self.FLAT)
        for collection in (8, 32):
            self.assertGreater(wall_clock_per_move(18, collection, self.FLAT), sequential)

    def test_unmeasured_batch_size_is_refused(self):
        with self.assertRaises(ValueError):
            wall_clock_per_move(18, 5, self.FLAT)


class PositionStorageTests(unittest.TestCase):
    def test_positions_round_trip_with_a_checksum(self):
        positions = [dict(game_index=0, split='select', ply=0,
                          state=np.arange(4, dtype=np.int8), key=b'k' * 16,
                          reference_action=3, reference_pi=np.array([0.5, 0.5]),
                          legal_moves=2, reference_q=np.array([0.0, 0.0], dtype=np.float32),
                          reference_evals=7, current_player=0)]
        summaries = [dict(game_index=0, split='select', plies=1, terminated=True,
                          result=[1.0, -1.0])]
        with tempfile.TemporaryDirectory() as folder:
            manifest = save_positions(folder, positions, summaries, metadata=dict(game='stub'))
            self.assertEqual(manifest['positions_by_split'], {'select': 1})
            self.assertEqual(manifest['games_by_split'], {'select': 1})
            loaded, reloaded = load_positions(folder)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(reloaded['metadata']['game'], 'stub')
            np.testing.assert_array_equal(loaded[0]['state'], positions[0]['state'])

            path = Path(folder) / 'positions.pkl.z'
            path.write_bytes(path.read_bytes() + b'x')
            with self.assertRaises(ValueError):
                load_positions(folder)


if __name__ == '__main__':
    unittest.main()
