"""Registration and actual shared self-play, replay, Arena and worker paths."""

import base64
from collections import deque
from contextlib import redirect_stdout
import io
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

import numpy as np
import torch

from Arena import Arena
from Coach import Coach
from GameSwitcher import import_game, import_logicnumba
from MCTS import MCTS
from intransitive.IntransitiveConstants import E, N, S, W, encode_action
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveLogicNumba import Board, validate_state
from intransitive.tests.test_augmentation import RecordingGame, ScriptedSearch
from intransitive.tests.test_draws import load_history, noncapture_actions, play, sparse_position
from intransitive.tests.test_network import args as network_args

ROOT = Path(__file__).resolve().parents[2]
CYCLE = [encode_action(1, 4, N), encode_action(4, 7, W),
         encode_action(1, 5, S), encode_action(3, 7, E)]
DRAW = np.full(2, 1e-4, dtype=np.float32)


def coach_args(**changes):
    values = dict(numMCTSSims=2, prob_fullMCTS=1., ratio_fullMCTS=1,
                  forced_playouts=False, universes=0, cpuct=1., fpu=0.,
                  no_mem_optim=False, dirichletAlpha=0., parallel_inferences=1,
                  temperature=[1., 1., 1.], tempThreshold=10, no_compression=True,
                  numEps=2, maxlenOfQueue=100000, numIters=1, numItersHistory=2,
                  profile=False, arenaCompare=10, updateThreshold=0.6,
                  stop_after_N_fail=2, useray=False)
    values.update(changes)
    return SimpleNamespace(**values)


def make_coach(game=None, **changes):
    Game, Net, _, _ = import_game('intransitive')
    game = Game() if game is None else game
    args = coach_args(**changes)
    net_args = network_args()
    net_args['no_compression'] = args.no_compression
    return Coach(game, Net(game, net_args), args)


def decode(example):
    return pickle.loads(zlib.decompress(example)) if isinstance(example, bytes) else example


class SharedPipeline(unittest.TestCase):
    def tearDown(self):
        MCTS.reset_all_search_trees()

    def assert_examples_equal(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            for field, original in zip(decode(a), decode(b)):
                np.testing.assert_array_equal(field, original)

    def test_registry_official_start_and_unrelated_game_smoke(self):
        Game, Net, players, count = import_game('intransitive')
        self.assertIs(Game, IntransitiveGame)
        self.assertIs(import_logicnumba('intransitive'), Board)
        self.assertEqual(count, 2)
        game = Game()
        state = game.getInitBoard()
        np.testing.assert_array_equal(state, Board().get_state())
        self.assertEqual(state[:, :, 32].flat[1], 0)
        self.assertEqual(np.count_nonzero(state[:, :, 0]), 20)
        self.assertEqual(Net(game, network_args()).num_players, 2)
        for name in ('RandomPlayer', 'GreedyPlayer', 'HumanPlayer'):
            self.assertTrue(callable(getattr(players, name)(game).play))

        Game, _, players, count = import_game('santorini')
        game = Game()
        initial = game.getInitBoard()
        canonical = game.getCanonicalForm(initial, 0)
        action = players.RandomPlayer(game).play(canonical, 1)
        self.assertTrue(game.getValidMoves(canonical, 0)[action])
        child, player = game.getNextState(initial, 0, action)
        self.assertEqual(game.getGameEnded(child, player).shape, (count,))
        arena = Arena(None, None, game)
        restored, p, turn = arena.restore_state(arena.serialize_state(child, player, 1))
        np.testing.assert_array_equal(restored, child)
        self.assertEqual((p, turn), (player, 1))

    def test_main_cli_selects_real_game_network_coach_and_starts_blue(self):
        import main
        episodes = []

        def bounded_learning(coach):
            self.assertIsInstance(coach.game, IntransitiveGame)
            initial = coach.game.getInitBoard()
            np.testing.assert_array_equal(initial, Board().get_state())
            self.assertEqual(coach.game.board.get_next_player(), 0)
            coach.nnet.device['inference'] = 'cpu'
            episodes.append(coach.executeEpisode())

        with tempfile.TemporaryDirectory() as folder:
            argv = ['main.py', 'intransitive', '-C', folder, '-m', '2', '-P', '1',
                    '--prob-fullMCTS', '1', '-d', '0', '-n', '1', '-e', '1']
            with patch.object(sys, 'argv', argv), patch.object(Coach, 'learn', bounded_learning):
                with redirect_stdout(io.StringIO()):
                    main.main()
            self.assertEqual(len(episodes), 1)
            self.assertTrue(episodes[0])
            for example in episodes[0]:
                validate_state(decode(example)[0])
            self.assertTrue(list(Path(folder).glob('source_backups/*/intransitive/NNet.py')))

    def test_coach_draw_labels_replay_roundtrips_and_real_training(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed), tempfile.TemporaryDirectory() as folder:
                game = RecordingGame()
                coach = make_coach(game, no_compression=not compressed, checkpoint=folder)
                coach.mcts = ScriptedSearch(game, CYCLE * 2)
                examples = coach.executeEpisode()
                self.assertEqual(len(game.physical), 9)
                for example in examples:
                    state, policy, outcome, valid, q = decode(example)
                    validate_state(state)
                    np.testing.assert_array_equal(outcome, DRAW)
                    self.assertEqual(len(q), 2)
                    self.assertAlmostEqual(sum(policy), 1)
                    self.assertFalse(np.any(np.asarray(policy)[~valid]))
                # Persist using Coach, load both as-is and with compression conversion.
                coach.trainExamplesHistory = [deque(examples)]
                coach.saveTrainExamples()
                for no_compression in (False, True):
                    loaded = make_coach(no_compression=no_compression,
                                        load_folder_file=folder + '/best.pt')
                    loaded.loadTrainExamples()
                    replay = loaded.trainExamplesHistory[0]
                    self.assert_examples_equal(replay, examples)
                    self.assertIsInstance(replay[0], tuple if no_compression else bytes)
                    picked = loaded.nnet.pick_examples(replay, [0, len(replay) - 1])
                    for example, original in zip(zip(*picked), (examples[0], examples[-1])):
                        self.assert_examples_equal([example], [original])
                before = next(coach.nnet.nnet.parameters()).detach().clone()
                coach.nnet.train(examples[:4])
                self.assertFalse(torch.equal(before, next(coach.nnet.nnet.parameters())))
                self.assertTrue(all(torch.isfinite(p).all() for p in coach.nnet.nnet.parameters()))

    def test_arena_restore_preserves_subsequent_draws_and_terminal_mcts(self):
        for reason in ('repetition', 'no-capture limit'):
            with self.subTest(reason=reason):
                board = Board()
                actions = CYCLE * 2
                if reason == 'no-capture limit':
                    load_history(board, [sparse_position()])
                    actions = noncapture_actions()
                play(board, actions[:-1])
                state = board.get_state()
                player, turn = board.get_next_player(), board.get_total_ply()
                game = RecordingGame()
                calls = []

                def last_move(canonical, n):
                    calls.append((canonical.copy(), n))
                    return actions[-1]

                arena = Arena(last_move, last_move, game)
                encoded = arena.serialize_state(state, player, turn)
                # Byte-for-byte compatibility with the previously inline format.
                legacy = base64.b64encode(zlib.compress(
                    state.tobytes() + bytes([player]) + turn.to_bytes(2, 'big'),
                    level=9, wbits=-15)).decode('ascii')
                self.assertEqual(encoded, legacy)
                restored, p, t = arena.restore_state(encoded)
                np.testing.assert_array_equal(restored, state)
                self.assertTrue(restored.flags.owndata and restored.flags.writeable)
                self.assertEqual((p, t), (player, turn))
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(arena.playGame(initial_state=encoded), DRAW[0])
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][1], turn + 1)
                terminal, next_player = game.physical[-1]
                game.getGameEnded(terminal, next_player)
                self.assertEqual(game.board.get_terminal_reason(), reason)
                coach = make_coach(game)
                canonical = game.getCanonicalForm(terminal, next_player)
                with patch.object(coach.nnet, 'predict', side_effect=AssertionError('terminal inference')):
                    np.testing.assert_array_equal(coach.mcts.search(canonical), DRAW)
                    np.testing.assert_array_equal(coach.mcts.search(canonical), DRAW)
                # History-free reconstruction must still be ongoing at the same placement.
                fresh = load_history(Board(), [state[:, :, 0]], first_player=player)
                child, p = game.getNextState(fresh, player, actions[-1])
                self.assertFalse(game.getGameEnded(child, p).any())

    def test_arena_assigns_wins_losses_draws_when_agents_swap(self):
        for value, expected in ((1., (2, 2, 0)), (-1., (2, 2, 0)), (DRAW[0], (0, 0, 4))):
            arena = Arena(None, None, IntransitiveGame())
            with patch.object(arena, 'playGame', return_value=value) as play_game:
                self.assertEqual(arena.playGames(4), expected)
                self.assertEqual([c.kwargs['other_way'] for c in play_game.call_args_list],
                                 [False, True, True, False])
        # Actual Arena games from an official Blue-first board, with agent order swapped.
        for swapped in (False, True):
            game = RecordingGame()
            seen = []
            def move(agent):
                def choose(state, turn):
                    seen.append(agent)
                    return CYCLE[(turn - 1) % 4]
                return choose
            arena = Arena(move(0), move(1), game)
            self.assertEqual(arena.playGame(other_way=swapped), DRAW[0])
            np.testing.assert_array_equal(game.physical[0][0], Board().get_state())
            self.assertEqual(seen, [int(swapped), int(not swapped)] * 4)

        # Real terminal vectors flow through playGame and playGames for either winner.
        for winner in (0, 1):
            pieces = sparse_position()
            pieces[8, 8] = 1 if winner == 0 else 0
            pieces[0, 0] = -1 if winner == 1 else 0
            initial = load_history(Board(), [pieces])
            game = RecordingGame(initial)
            arena = Arena(None, None, game)
            self.assertEqual(arena.playGame(), 1 if winner == 0 else -1)
            self.assertEqual(arena.playGames(4), (2, 2, 0))

    def test_arena_restore_rejects_wrong_size_or_player(self):
        arena = Arena(None, None, IntransitiveGame())
        state = arena.game.getInitBoard()
        for board, player in ((state[:, :, 0], 0), (state, 2)):
            encoded = arena.serialize_state(board, player, 0)
            with self.assertRaises(ValueError):
                arena.restore_state(encoded)
        with self.assertRaisesRegex(ValueError, 'int8'):
            arena.serialize_state(state.astype(np.float32), 0, 0)

    def test_candidate_acceptance_excludes_draws_and_rejects_all_draws(self):
        for results, accepted in (((0, 0, 10), False), ((3, 2, 5), True), ((2, 2, 6), False)):
            with self.subTest(results=results), tempfile.TemporaryDirectory() as folder:
                coach = make_coach(checkpoint=folder)
                coach.skipFirstSelfPlay = True
                coach.trainExamplesHistory = [[('already tested replay',)]]
                with patch.object(coach, 'saveTrainExamples'), \
                     patch.object(coach.nnet, 'train'), \
                     patch.object(coach.nnet, 'save_checkpoint') as save, \
                     patch.object(coach.nnet, 'load_checkpoint') as restore, \
                     patch.object(coach.pnet, 'load_checkpoint'), \
                     patch.object(Arena, 'playGames', return_value=results):
                    coach.learn()
                self.assertEqual('best.pt' in [c.kwargs['filename'] for c in save.call_args_list], accepted)
                self.assertEqual(restore.called, not accepted)

    def test_sequential_episodes_receive_fresh_search_trees(self):
        coach = make_coach()
        searches = []
        def episode():
            searches.append(coach.mcts)
            return []
        with patch.object(coach, 'executeEpisode', side_effect=episode):
            coach.executeEpisodes()
        self.assertEqual(len(searches), 2)
        self.assertIsNot(searches[0], searches[1])
        self.assertIsNot(searches[1], coach.mcts)

    def test_parallel_real_onnx_workers_have_independent_complete_states(self):
        # A subprocess bounds failures in the existing lock-based worker protocol.
        result = subprocess.run([sys.executable, '-m', 'intransitive.tests.test_pipeline', '--workers'],
                                cwd=ROOT, text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('independent workers verified', result.stdout)

    def test_pit_cli_random_opponents(self):
        result = subprocess.run([sys.executable, 'pit.py', 'intransitive', 'random', 'random', '-n', '2'],
                                cwd=ROOT, text=True, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('random vs random', result.stdout)


def check_workers():
    class WorkerGame(RecordingGame):
        instances = []
        def __init__(self):
            super().__init__()
            self.instances.append(self)

    coach = make_coach(WorkerGame(), parallel_inferences=2, numEps=2)
    untouched = coach.game.getInitBoard().copy()
    server = coach.nnet.predict_server
    def delayed_server(*args):
        # Guarantee the coordinator polls before the first batch is available.
        time.sleep(1.2)
        server(*args)
    coach.nnet.predict_server = delayed_server
    examples = coach.executeEpisodes()
    workers = WorkerGame.instances[1:]
    assert len(workers) >= 2
    assert examples
    assert coach.examplesQueue.empty()
    assert len(examples) == sum(len(triples) for w in workers for _, triples in w.batches)
    for example in examples:
        state, policy, outcome, mask, q = decode(example)
        validate_state(state)
        assert np.isfinite(policy).all() and np.isfinite(q).all()
        assert np.isclose(sum(policy), 1) and not np.any(np.asarray(policy)[~mask])
        assert np.array_equal(outcome, DRAW) or sorted(outcome) == [-1, 1]
    for worker in workers:
        np.testing.assert_array_equal(worker.physical[0][0], untouched)
        # Worker setup initializes once, then executeEpisode starts the game again.
        np.testing.assert_array_equal(worker.physical[1][0], untouched)
        assert worker is not coach.game and worker.board is not coach.game.board
        for (parent, player), (child, next_player) in zip(worker.physical[1:], worker.physical[2:]):
            validate_state(child)
            assert next_player == 1 - player
            # Match each actual recorded transition against independent legal replay.
            probe = IntransitiveGame()
            valid = probe.getValidMoves(parent, player)
            assert any(np.array_equal(probe.getNextState(parent, player, int(a))[0], child)
                       for a in np.flatnonzero(valid))
        state, player = worker.physical[-1]
        assert worker.getGameEnded(state, player).any()
    for i, worker in enumerate(workers):
        for other in workers[i + 1:]:
            assert not np.shares_memory(worker.board.state, other.board.state)
    snapshots = [w.board.get_state() for w in workers]
    workers[0].getInitBoard()
    for worker, saved in zip(workers[1:], snapshots[1:]):
        np.testing.assert_array_equal(worker.board.get_state(), saved)
    np.testing.assert_array_equal(coach.game.board.get_state(), untouched)
    print(f'{len(workers)} independent workers verified; {len(examples)} replay examples')


if __name__ == '__main__':
    if '--workers' in sys.argv:
        check_workers()
    else:
        unittest.main()
