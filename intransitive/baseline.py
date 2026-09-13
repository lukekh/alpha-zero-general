"""Reproduce the fixed issue #15 training budget and seeded opponent evaluation."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import pickle
import random
import resource
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

# The API toggle runs after ORT's native environment has already initialized.
# Prevent its telemetry uploader from starting at all (ORT 1.30 on macOS).
os.environ['ORT_DISABLE_TELEMETRY'] = '1'
import numpy as np
import onnxruntime as ort

# Disable before importing Torch and compiling any game code (macOS ORT 1.30).
ort.disable_telemetry_events()
import torch

from Arena import Arena
from MCTS import MCTS
from source_backup import backup_run_sources
from .benchmark_training import (MeasuredCoach, MeasuredGame, MeasuredNet, Metrics,
                                 REASONS, distribution, hardware, replay_metrics)
from .IntransitivePlayers import GreedyPlayer, RandomPlayer
from .NNet import NNetWrapper
from .smoke import decode, digest, require

BUDGET = Path(__file__).parent / 'benchmarks/training/baseline.json'


class BaselineGame(MeasuredGame):
    def getNextState(self, *args, **kwargs):
        before = len(self.metrics.games)
        state, player = super().getNextState(*args, **kwargs)
        if len(self.metrics.games) > before:
            self.metrics.games[-1]['captures'] = 20 - int(np.count_nonzero(state[:, :, 0]))
        return state, player


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + '\n')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    ort.disable_telemetry_events()


def select_checkpoint(arena):
    """Predeclared selection, independent of post-training opponent results."""
    accepted = [i for i, row in enumerate(arena, 1) if row['accepted']]
    if accepted:
        return f'candidate_{accepted[-1]}.pt', 'latest accepted candidate'
    return f'candidate_{len(arena)}.pt', 'final rejected candidate; no accepted trained incumbent'


def model_outcome(blue_reward, colour):
    if blue_reward not in (-1., 1.):
        return 'draws'
    return 'wins' if blue_reward == (1. if colour == 'Blue' else -1.) else 'losses'


def wilson(successes, n):
    """Marginal 95% binomial Wilson interval; draws remain in the denominator."""
    if not n:
        return None
    z = 1.959963984540054
    p = successes / n
    scale = 1 + z*z/n
    centre = (p + z*z/(2*n)) / scale
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / scale
    return [max(0., centre-half), min(1., centre+half)]


def summarize(rows):
    n = len(rows)
    counts = Counter(row['outcome'] for row in rows)
    reasons = Counter(row['reason'] for row in rows)
    score = (counts['wins'] + .5*counts['draws']) / n if n else None
    # Hoeffding applies to independent bounded scores, without treating half-draws
    # as binomial wins. It is conservative at this deliberately small budget.
    half = math.sqrt(math.log(40)/(2*n)) if n else None
    return dict(games=n, **{k: counts[k] for k in ('wins', 'draws', 'losses')},
                score=score, score_95_hoeffding=([max(0., score-half), min(1., score+half)] if n else None),
                win_probability_95_wilson=wilson(counts['wins'], n),
                draw_probability_95_wilson=wilson(counts['draws'], n),
                unique_trajectories=len({r['trajectory_sha256'] for r in rows}),
                reasons={r: reasons[r] for r in REASONS},
                length_plies=distribution([r['plies'] for r in rows]))


def instrument_search(metrics):
    original = MCTS.getActionProb

    def measured(searcher, state, *args, **kwargs):
        before = state.tobytes()
        tick = time.perf_counter()
        result = original(searcher, state, *args, **kwargs)
        bucket = metrics.bucket()
        bucket['search_worker_seconds'] += time.perf_counter() - tick
        bucket['search_calls'] += 1
        bucket['simulations'] += searcher.step + 1
        require(state.tobytes() == before, 'MCTS mutated root history')
        return result
    return patch.object(MCTS, 'getActionProb', measured)


def rates(metrics):
    for bucket in metrics.phases.values():
        if bucket['seconds']:
            bucket['simulations_per_wall_second'] = bucket['simulations'] / bucket['seconds']
            bucket['inference_states_per_wall_second'] = bucket['inference_states'] / bucket['seconds']
    return metrics.phases


def train(folder):
    start = time.perf_counter()
    budget = json.loads(BUDGET.read_text())
    args = argparse.Namespace(**dict(budget['settings'], checkpoint=str(folder)))
    seed_all(args.seed)
    metrics = Metrics()
    MeasuredGame.metrics = MeasuredNet.metrics = metrics
    game = BaselineGame()
    nn_args = {k: getattr(args, k) for k in ('learn_rate', 'dropout', 'epochs',
        'batch_size', 'nn_version', 'no_compression', 'q_weight')}
    net = MeasuredNet(game, nn_args)
    coach = MeasuredCoach(game, net, args)
    snapshot = backup_run_sources(args)
    for target in (folder, snapshot):
        write_json(target / 'settings.json', vars(args))
        write_json(target / 'budget.json', budget)
    packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (folder / 'requirements.txt').write_text(packages)
    (snapshot / 'requirements.txt').write_text(packages)
    net.save_checkpoint(str(folder), 'initial.pt', additional_keys=vars(args))
    arenas = []
    original_arena = Arena.playGames

    def measured_arena(arena, *positional, **keywords):
        with metrics.timed('candidate_arena'):
            wins, losses, draws = original_arena(arena, *positional, **keywords)
        accepted = wins + losses > 0 and wins/(wins+losses) >= args.updateThreshold
        arenas.append(dict(iteration=len(arenas)+1, wins=wins, draws=draws,
            losses=losses, accepted=accepted, all_draw_rejection=wins+losses == 0))
        # Retain progress even if the supervisor terminates a later phase.
        write_json(folder / 'training-progress.json', dict(arena=arenas, games=metrics.games,
                   training=net.training, losses=net.losses))
        return wins, losses, draws

    tick = time.perf_counter()
    with instrument_search(metrics), patch.object(Arena, 'playGames', measured_arena):
        coach.learn()
    learn_seconds = time.perf_counter() - tick
    require(len(arenas) == 4 and len(metrics.games) == 64, 'Incomplete training budget')
    require(set(metrics.symmetries) == {12}, 'Missing augmentation coverage')
    require(sum(g['plies'] for g in metrics.games if g['phase'] == 'self_play') == len(metrics.legal),
            'Position count mismatch')
    for phase in ('self_play', 'candidate_arena'):
        require(sum(g['phase'] == phase for g in metrics.games) == 32, 'Wrong phase game count')
    with (folder / 'checkpoint.examples').open('rb') as stream:
        histories = pickle.load(stream)
    require(histories == coach.trainExamplesHistory and len(histories) == 2, 'Replay changed')
    replay = [replay_metrics(h) for h in histories]
    selected, selection = select_checkpoint(arenas)
    shutil.copyfile(folder / selected, folder / 'baseline.pt')
    # Fixed earlier trained opponent, even if its candidate arena rejected it.
    shutil.copyfile(folder / 'candidate_1.pt', folder / 'earlier.pt')
    offset = 0
    for row in net.training:
        end = offset + row['updates']
        row['policy_loss_mean'] = float(np.mean(net.losses['policy'][offset:end]))
        row['value_loss_mean'] = float(np.mean(net.losses['value'][offset:end]))
        offset = end
    net.switch_target('inference')
    report = dict(passed=True, command=sys.argv, hardware=hardware(), budget=budget,
        settings=vars(args), nn_args=nn_args, requirements=packages.splitlines(),
        git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_backup=str(snapshot),
        source_sha256={str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob('*.py'))},
        format=net.checkpoint_format(), feature_config=net.nnet.feature_config,
        torch_threads=torch.get_num_threads(), onnx_providers=net.ort_session.get_providers(),
        original_positions=len(metrics.legal), augmented_examples=len(metrics.legal)*12,
        symmetry_counts=dict(metrics.symmetries), legal_actions=distribution(metrics.legal),
        training=net.training, losses=net.losses, arena=arenas, games=metrics.games,
        phases=rates(metrics), learn_seconds=learn_seconds, replay=replay,
        selected_checkpoint=selected, selection_reason=selection,
        earlier_checkpoint='candidate_1.pt',
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
        artifacts={p.name: dict(bytes=p.stat().st_size, sha256=digest(p))
                   for p in sorted(folder.iterdir()) if p.is_file()},
        elapsed_seconds=time.perf_counter()-start)
    write_json(folder / 'training.json', report)


def load_net(game, path, nn_args):
    net = MeasuredNet(game, dict(nn_args, nn_version=-1))
    metadata = net.load_checkpoint(str(path.parent), path.name)
    require(metadata is not None, 'Checkpoint did not load')
    return net


def evaluate(folder, model_folder, representative=False):
    folder.mkdir(parents=True)
    start = time.perf_counter()
    training = json.loads((model_folder / 'training.json').read_text())
    budget = training['budget']
    seed_all(150)
    metrics = Metrics()
    MeasuredGame.metrics = MeasuredNet.metrics = metrics
    game = BaselineGame()
    args = argparse.Namespace(**training['settings'])
    model = load_net(game, model_folder / 'baseline.pt', training['nn_args'])
    earlier = load_net(game, model_folder / 'earlier.pt', training['nn_args'])
    for name in ('baseline.pt', 'earlier.pt'):
        require(digest(model_folder / name) == training['artifacts'][name]['sha256'],
                'Checkpoint hash mismatch')
    with (model_folder / 'checkpoint.examples').open('rb') as stream:
        state, _, _, mask, _ = decode(pickle.load(stream)[-1][-1])
    # Loaded model parity on a state containing real game history.
    model.device['inference'] = 'cpu'
    cpu = model.predict(state, mask)
    model.device['inference'] = 'onnx'
    onnx = model.predict(state, mask)
    for a, b in zip(cpu, onnx):
        np.testing.assert_allclose(a, b, atol=2e-6, rtol=1e-5)
    parity = [float(np.max(np.abs(a-b))) for a, b in zip(cpu, onnx)]
    rows = []
    names = ('random',) if representative else ('random', 'greedy', 'earlier_checkpoint')
    with instrument_search(metrics):
        for name in names:
            for index in range(2 if representative else 16):
                colour = 'Blue' if index % 4 in (0, 3) else 'Red'
                base_seed = budget['seeds']['evaluation_by_opponent'][name]
                streams = np.random.SeedSequence([base_seed, index]).generate_state(4)
                seed_all(int(streams[0]))
                search = MCTS(game, model, args)
                search.rng = np.random.default_rng(int(streams[1]))
                def model_play(board, turn):
                    return int(np.argmax(search.getActionProb(board, temp=0, force_full_search=True)[0]))
                if name == 'random':
                    opponent = RandomPlayer(game, seed=int(streams[2])).play
                elif name == 'greedy':
                    opponent = GreedyPlayer(game).play
                else:
                    previous = MCTS(game, earlier, args)
                    previous.rng = np.random.default_rng(int(streams[3]))
                    def opponent(board, turn):
                        return int(np.argmax(previous.getActionProb(board, temp=0, force_full_search=True)[0]))
                with metrics.timed(name):
                    reward = Arena(model_play, opponent, game).playGame(other_way=colour == 'Red')
                row = dict(metrics.games[-1], opponent=name, index=index, model_colour=colour,
                           physical_first_player='Blue', seeds=[int(s) for s in streams],
                           outcome=model_outcome(reward, colour))
                rows.append(row)
                write_json(folder / 'evaluation-progress.json', rows)
    comparisons = {name: dict(overall=summarize([r for r in rows if r['opponent'] == name]),
        by_colour={c: summarize([r for r in rows if r['opponent'] == name and r['model_colour'] == c])
                   for c in ('Blue', 'Red')}) for name in names}
    if representative:
        expected = json.loads((model_folder / 'evaluation/evaluation.json').read_text())['games'][:2]
        require(rows == expected, 'Representative game trajectories/results did not reproduce')
    report = dict(passed=True, representative=representative, command=sys.argv,
        checkpoint_sha256={n: digest(model_folder / n) for n in ('baseline.pt', 'earlier.pt')},
        reload_parity_max_abs=parity, games=rows, comparisons=comparisons,
        phases=rates(metrics), elapsed_seconds=time.perf_counter()-start,
        seed_policy='SeedSequence([opponent_seed, game_index]) -> NumPy/Python/Torch, model MCTS, random opponent, earlier MCTS',
        action_policy='32 full simulations; maximum visits; seeded random tie breaking; fresh trees each game',
        uncertainty='Marginal Wilson win/draw intervals; distribution-free Hoeffding score interval. Independent seeded games conditional on these fixed agents/opening only; no training-seed or opening generalization.')
    write_json(folder / 'evaluation.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'evaluate', 'verify'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-folder')
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    cli = parser.parse_args()
    folder = Path(cli.output).resolve()
    require(not folder.exists(), 'Use a fresh output directory')
    require(cli.phase == 'train' or cli.model_folder, 'Evaluation requires --model-folder')
    if cli.child:
        if cli.phase == 'train':
            train(folder)
        else:
            evaluate(folder, Path(cli.model_folder).resolve(), cli.phase == 'verify')
        return
    budget = json.loads(BUDGET.read_text())['budgets']
    timeout = budget['training_wall_seconds' if cli.phase == 'train' else 'evaluation_wall_seconds']
    command = [sys.executable, '-m', 'intransitive.baseline', *sys.argv[1:], '--child']
    started = time.perf_counter()
    try:
        result = subprocess.run(command, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise SystemExit(f'{cli.phase} exceeded {timeout}s; incomplete, not a draw; inspect progress JSON')
    require(result.returncode == 0, f'{cli.phase} failed with exit code {result.returncode}')
    report = folder / ('training.json' if cli.phase == 'train' else 'evaluation.json')
    data = json.loads(report.read_text())
    data.update(process_exit_code=0, child_process_seconds=time.perf_counter()-started,
                hard_timeout_seconds=timeout)
    write_json(report, data)
    print('Completed:', report)


if __name__ == '__main__':
    main()
