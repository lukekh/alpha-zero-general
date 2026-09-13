"""Bounded, audited Coach run; see smoke/README.md for reproduction commands."""

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import logging
from pathlib import Path
import pickle
import platform
import random
import subprocess
import sys
import time
import zlib

import numpy as np
import onnxruntime as ort
import torch

from Arena import Arena
from Coach import Coach
from MCTS import MCTS
from source_backup import backup_run_sources
from .IntransitiveConstants import METADATA_PLANE, META_HISTORY_LENGTH
from .IntransitiveGame import IntransitiveGame
from .IntransitiveLogicNumba import validate_state
from .IntransitivePlayers import RandomPlayer
from .NNet import NNetWrapper


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(example):
    return pickle.loads(zlib.decompress(example)) if isinstance(example, bytes) else example


class AuditedGame(IntransitiveGame):
    """Observe real moves through the adapter; MCTS uses the compiled Board."""

    def __init__(self):
        super().__init__()
        self.phase = 'self_play'
        self.games = []
        self.positions = 0
        self.augmented = 0
        self.symmetry_counts = Counter()
        self.max_history = 0

    def getNextState(self, board, player, action, random_seed=0):
        require(bool(self.getValidMoves(board, player)[action]), 'Illegal played move')
        state, next_player = super().getNextState(board, player, action, random_seed)
        validate_state(state)
        result = self.getGameEnded(state, next_player)
        if result.any():
            self.games.append(dict(phase=self.phase, plies=self.getRound(state),
                                   reason=self.board.get_terminal_reason(),
                                   reward=result.tolist()))
        return state, next_player

    def getSymmetries(self, board, pi, valid_actions):
        examples = super().getSymmetries(board, pi, valid_actions)
        require(1 <= len(examples) <= 12, 'Invalid augmentation count')
        for state, policy, mask in examples:
            validate_state(state)
            np.testing.assert_array_equal(mask, self.getValidMoves(state, 0))
            require(np.isfinite(policy).all() and (policy >= 0).all(), 'Invalid policy')
            np.testing.assert_allclose(policy.sum(), 1., atol=1e-6)
            require(not policy[~mask].any(), 'Policy targets illegal moves')
            self.max_history = max(self.max_history,
                                   int(state[:, :, METADATA_PLANE].flat[META_HISTORY_LENGTH]))
        self.positions += 1
        self.augmented += len(examples)
        self.symmetry_counts[len(examples)] += 1
        return examples


class AuditedNet(NNetWrapper):
    def __init__(self, game, nn_args):
        super().__init__(game, nn_args)
        self.device['inference'] = nn_args['inference_backend']
        self.losses = {'policy': [], 'value': []}
        self.training = []

    def loss_pi(self, targets, outputs):
        loss = super().loss_pi(targets, outputs)
        require(bool(torch.isfinite(loss)), 'Nonfinite policy loss')
        self.losses['policy'].append(loss.item())
        return loss

    def loss_v(self, targets_v, targets_q, outputs):
        loss = super().loss_v(targets_v, targets_q, outputs)
        require(bool(torch.isfinite(loss)), 'Nonfinite value loss')
        self.losses['value'].append(loss.item())
        return loss

    def train(self, examples, *args, **kwargs):
        before = {k: v.detach().clone() for k, v in self.nnet.named_parameters()}
        start = time.monotonic()
        super().train(examples, *args, **kwargs)
        changes = [float((value.detach() - before[key]).abs().max())
                   for key, value in self.nnet.named_parameters()]
        require(max(changes) > 0, 'Optimizer did not change parameters')
        require(all(bool(torch.isfinite(v).all()) for v in self.nnet.parameters()),
                'Nonfinite trained parameters')
        self.training.append(dict(examples=len(examples),
                                  updates=len(self.losses['policy']),
                                  max_parameter_delta=max(changes),
                                  seconds=time.monotonic() - start))


def settings(cli):
    if cli.settings:
        values = json.loads(Path(cli.settings).read_text())
    else:
        # At most 19 captures, then 30 noncaptures: <=600 plies/game.
        # 2 games * 600 plies * 12 symmetries fits without truncation.
        values = dict(game='intransitive', numMCTSSims=4, prob_fullMCTS=1.,
                      ratio_fullMCTS=1, forced_playouts=False, universes=0,
                      cpuct=1.25, fpu=0., no_mem_optim=False, dirichletAlpha=0.,
                      parallel_inferences=1, temperature=[1., 1., 1.],
                      tempThreshold=10, no_compression=False, numEps=2,
                      maxlenOfQueue=14400, numIters=1, numItersHistory=2,
                      profile=False, arenaCompare=2, updateThreshold=0.6,
                      stop_after_N_fail=2, useray=False, forget_examples=False,
                      learn_rate=0.0003, dropout=0., epochs=1, batch_size=64,
                      nn_version=1, q_weight=0.5)
    values.update(checkpoint=str(Path(cli.checkpoint).resolve()), seed=cli.seed,
                  inference_backend=cli.backend, load_model=bool(cli.resume),
                  load_folder_file=str(Path(cli.resume).resolve()) if cli.resume else None)
    require(values['parallel_inferences'] == 1 and values['numIters'] == 1,
            'This smoke auditor requires one worker and one iteration')
    require(values['numEps'] == 2 and values['arenaCompare'] == 2,
            'This smoke gate requires two self-play and two arena games')
    require(values['prob_fullMCTS'] == 1. and values['dirichletAlpha'] == 0.,
            'Reproducibility requires full search without unseeded Dirichlet noise')
    require(not values['no_compression'], 'Replay auditor requires compressed examples')
    return argparse.Namespace(**values)


def checkpoint_checks(folder, args, game, report):
    candidate = NNetWrapper(game, dict(report['nn_args'], nn_version=-1))
    saved = candidate.load_checkpoint(str(folder), 'candidate_1.pt')
    require(saved['candidate_iteration'] == 1, 'Missing candidate metadata')
    report['checkpoint_metadata'] = {key: saved[key] for key in
                                    ('intransitive_checkpoint', 'intransitive_config', 'nn_args')}
    # Inspect a replay state with actual history through both loaded inference paths.
    with (folder / 'checkpoint.examples').open('rb') as stream:
        histories = pickle.load(stream)
    examples = [decode(x) for history in histories for x in history]
    state, _, _, mask, _ = max(examples, key=lambda x:
                             int(x[0][:, :, METADATA_PLANE].flat[META_HISTORY_LENGTH]))
    candidate.device['inference'] = 'cpu'
    cpu = candidate.predict(state, mask)
    candidate.device['inference'] = 'onnx'
    onnx = candidate.predict(state, mask)
    for left, right in zip(cpu, onnx):
        np.testing.assert_allclose(left, right, atol=2e-6, rtol=1e-5)
    report['reload_parity_max_abs'] = [float(np.max(np.abs(a - b))) for a, b in zip(cpu, onnx)]
    report['checkpoint_games'] = []
    for backend in ('cpu', 'onnx'):
        random.seed(args.seed + 10)
        np.random.seed(args.seed + 10)
        candidate.device['inference'] = backend
        game.phase = 'checkpoint_' + backend
        search = MCTS(game, candidate, args)
        result = Arena(lambda b, n: int(np.argmax(search.getActionProb(
            b, temp=0, force_full_search=True)[0])),
            RandomPlayer(game, seed=args.seed + 10).play, game).playGames(2)
        report['checkpoint_games'].append(dict(backend=backend, wins=result[0],
                                               losses=result[1], draws=result[2]))
    return histories


def run(cli):
    # Avoid the ORT 1.30 macOS telemetry shutdown race observed during this gate.
    ort.disable_telemetry_events()
    args = settings(cli)
    folder = Path(args.checkpoint)
    require(not folder.exists(), 'Use a new checkpoint directory for each run')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.monotonic()
    nn_args = {key: getattr(args, key) for key in
               ('learn_rate', 'dropout', 'epochs', 'batch_size', 'nn_version',
                'no_compression', 'q_weight', 'inference_backend')}
    game = AuditedGame()
    net = AuditedNet(game, nn_args)
    if args.load_model:
        net.load_checkpoint(str(Path(args.load_folder_file).parent),
                            Path(args.load_folder_file).name)
    coach = Coach(game, net, args)
    loaded_histories = 0
    if args.load_model:
        coach.loadTrainExamples()
        loaded_histories = len(coach.trainExamplesHistory)
        require(loaded_histories > 0, 'Resume did not load replay')
    snapshot = backup_run_sources(args)
    settings_json = json.dumps(vars(args), indent=2) + '\n'
    (folder / 'settings.json').write_text(settings_json)
    (snapshot / 'settings.json').write_text(settings_json)
    dependencies = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (folder / 'requirements.txt').write_text(dependencies)
    (snapshot / 'requirements.txt').write_text(dependencies)
    source_manifest = {str(p.relative_to(snapshot)): digest(p)
                       for p in sorted(snapshot.rglob('*.py'))}
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    report = dict(settings=vars(args), nn_args=nn_args, command=sys.argv,
                  git_revision=revision.stdout.strip() if revision.returncode == 0 else None,
                  python=sys.version, platform=platform.platform(),
                  onnx_telemetry=False,
                  machine=platform.machine(), torch_threads=torch.get_num_threads(),
                  dependencies={name: version(name) for name in
                                ('numpy', 'numba', 'llvmlite', 'torch', 'onnx', 'onnxruntime', 'tqdm')},
                  source_backup=str(snapshot), source_sha256=source_manifest,
                  resumed_from_sha256=digest(args.load_folder_file) if args.load_model else None,
                  loaded_replay_histories=loaded_histories)
    # The real Coach performs all self-play, replay persistence, training and arena.
    coach.learn()
    require(len(net.training) == 1 and net.training[0]['updates'] > 0, 'No training')
    # Adapter games occur in order: two self-play, then two candidate arena games.
    require(len(game.games) == 4, 'Expected two self-play and two arena games')
    for item in game.games[2:]:
        item['phase'] = 'candidate_arena'
    require(game.positions == sum(x['plies'] for x in game.games[:2]),
            'Not all played positions were augmented')
    require(game.augmented == len(coach.trainExamplesHistory[-1]), 'Replay truncated')
    outcomes = [x['reward'][0] * (1 if i == 0 else -1)
                for i, x in enumerate(game.games[2:])]
    wins, losses = outcomes.count(1.), outcomes.count(-1.)
    draws = 2 - wins - losses
    accepted = wins + losses > 0 and wins / (wins + losses) >= args.updateThreshold
    expected = folder / ('best.pt' if accepted else 'temp.pt')
    checkpoint = torch.load(expected, map_location='cpu', weights_only=False)
    for key, tensor in net.nnet.state_dict().items():
        torch.testing.assert_close(tensor, checkpoint['state_dict'][key], atol=0, rtol=0)
    require((folder / 'best.pt').exists() == accepted, 'Incorrect candidate handling')
    net.save_checkpoint(str(folder), 'retained.pt', additional_keys=vars(args))
    report.update(training=net.training, losses=net.losses,
                  self_play_positions=game.positions, augmented_examples=game.augmented,
                  symmetry_counts=dict(game.symmetry_counts), max_history=game.max_history,
                  arena=dict(candidate_wins=wins, candidate_losses=losses, draws=draws,
                             accepted=accepted, retained=expected.name))
    histories = checkpoint_checks(folder, args, game, report)
    require(len(histories) == min(loaded_histories + 1, args.numItersHistory),
            'Unexpected resumed replay history')
    for a_history, b_history in zip(histories, coach.trainExamplesHistory):
        require(list(a_history) == list(b_history), 'Compressed replay changed on reload')
    report.update(games=game.games, replay_histories=len(histories),
                  elapsed_seconds=time.monotonic() - start,
                  artifacts={p.name: dict(bytes=p.stat().st_size, sha256=digest(p))
                             for p in sorted(folder.iterdir()) if p.is_file()}, passed=True)
    (folder / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('passed', 'self_play_positions',
          'augmented_examples', 'training', 'arena', 'checkpoint_games', 'elapsed_seconds')}, indent=2))
    print('Report:', folder / 'report.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--backend', choices=('cpu', 'onnx'), default='onnx')
    parser.add_argument('--resume', help='Checkpoint weights; adjacent replay is also required')
    parser.add_argument('--settings', help='Restore settings.json from a source backup')
    cli = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(cli)


if __name__ == '__main__':
    main()
