"""Bounded CPU/ONNX Coach measurements. See benchmarks/training/README.md."""

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import random
import resource
import subprocess
import sys
import time
from unittest.mock import patch
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
from .NNet import NNetWrapper
from .smoke import decode, digest, require, settings as smoke_settings

REASONS = ('corner', 'stalemate', 'repetition', 'no-capture limit')


def distribution(values):
    data = np.asarray(values, dtype=float)
    if not len(data):
        return dict(count=0)
    return dict(count=len(data), mean=float(data.mean()), min=float(data.min()),
                median=float(np.median(data)), p95=float(np.percentile(data, 95)),
                max=float(data.max()))


def deep_size(value, seen=None):
    """Owned Python/NumPy bytes, excluding allocator overhead; deduplicate references."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        if value.base is not None:
            size += deep_size(value.base, seen)
    elif isinstance(value, (tuple, list)):
        size += sum(deep_size(item, seen) for item in value)
    return size


def replay_metrics(examples):
    """Streaming inspection avoids holding a second, decompressed replay in memory."""
    compressed, serialized, decoded, arrays, histories = [], [], [], [], []
    decode_seconds = encode_seconds = 0.
    for blob in examples:
        start = time.perf_counter()
        raw = zlib.decompress(blob)
        example = pickle.loads(raw)
        decode_seconds += time.perf_counter() - start
        validate_state(example[0])
        start = time.perf_counter()
        encoded = zlib.compress(pickle.dumps(example), level=1)
        encode_seconds += time.perf_counter() - start
        # Pickle memoization/dtype aliases can change bytes after decoding.
        # The wire bytes stored by Coach are measured above; require value parity.
        restored = pickle.loads(zlib.decompress(encoded))
        for original, actual in zip(example, restored):
            require(np.array_equal(original, actual), 'Replay values did not round trip')
        compressed.append(len(blob))
        serialized.append(len(raw))
        decoded.append(deep_size(example))
        arrays.append(sum(x.nbytes for x in example if isinstance(x, np.ndarray)))
        histories.append(int(example[0][:, :, METADATA_PLANE].flat[META_HISTORY_LENGTH]))
    return dict(examples=len(examples), compressed_sample_bytes=distribution(compressed),
                pickle_sample_bytes=distribution(serialized),
                decoded_sample_owned_bytes=distribution(decoded),
                array_payload_bytes=distribution(arrays), history_length=distribution(histories),
                compressed_payload_bytes=sum(compressed),
                compressed_resident_bytes=sys.getsizeof(examples) + sum(sys.getsizeof(x) for x in examples),
                pickle_payload_bytes=sum(serialized),
                decoded_resident_upper_estimate_bytes=sys.getsizeof(examples) + sum(decoded),
                encode_seconds=encode_seconds, decode_seconds=decode_seconds)


class Metrics:
    def __init__(self):
        self.phase = 'setup'
        self.phases = {}
        self.games = []
        self.legal = []
        self.symmetries = Counter()
        self.original_replay = []
        self.iteration = 0

    def bucket(self):
        return self.phases.setdefault(self.phase, dict(seconds=0., search_calls=0,
            simulations=0, search_worker_seconds=0., inference_calls=0,
            inference_states=0, inference_seconds=0., inference_batches=Counter()))

    @contextmanager
    def timed(self, phase):
        previous, self.phase = self.phase, phase
        start = time.perf_counter()
        try:
            yield
        finally:
            self.bucket()['seconds'] += time.perf_counter() - start
            self.phase = previous


class MeasuredGame(IntransitiveGame):
    metrics = None

    def getInitBoard(self):
        self.trace = hashlib.sha256()
        state = super().getInitBoard()
        self.trace.update(state.tobytes())
        return state

    def getNextState(self, board, player, action, random_seed=0):
        require(bool(self.getValidMoves(board, player)[action]), 'Illegal played action')
        before = board.tobytes()
        state, next_player = super().getNextState(board, player, action, random_seed)
        require(board.tobytes() == before, 'Transition mutated parent history')
        self.trace.update(int(action).to_bytes(2, 'little'))
        self.trace.update(state.tobytes())
        result = self.getGameEnded(state, next_player)
        if result.any():
            reason = self.board.get_terminal_reason()
            require(reason in REASONS, 'Unrecognized terminal reason')
            self.metrics.games.append(dict(phase=self.metrics.phase,
                iteration=self.metrics.iteration, plies=self.getRound(state), reason=reason,
                reward=result.tolist(), trajectory_sha256=self.trace.hexdigest()))
        return state, next_player

    def getSymmetries(self, board, pi, valid_actions):
        triples = super().getSymmetries(board, pi, valid_actions)
        self.metrics.legal.append(int(np.count_nonzero(valid_actions)))
        self.metrics.symmetries[len(triples)] += 1
        return triples


class TimedSession:
    def __init__(self, session, metrics):
        self.session, self.metrics = session, metrics

    def __getattr__(self, name):
        return getattr(self.session, name)

    def run(self, outputs, inputs):
        start = time.perf_counter()
        result = self.session.run(outputs, inputs)
        elapsed = time.perf_counter() - start
        bucket = self.metrics.bucket()
        size = len(inputs['board'])
        bucket['inference_seconds'] += elapsed
        bucket['inference_calls'] += 1
        bucket['inference_states'] += size
        bucket['inference_batches'][size] += 1
        return result


class MeasuredNet(NNetWrapper):
    metrics = None

    def __init__(self, *args):
        super().__init__(*args)
        self.training = []
        self.losses = {'policy': [], 'value': []}

    def export_and_load_onnx(self):
        super().export_and_load_onnx()
        self.ort_session = TimedSession(self.ort_session, self.metrics)

    def loss_pi(self, targets, outputs):
        loss = super().loss_pi(targets, outputs)
        require(bool(torch.isfinite(loss)), 'Nonfinite policy loss')
        self.losses['policy'].append(loss.item())
        return loss

    def loss_v(self, targets, qs, outputs):
        loss = super().loss_v(targets, qs, outputs)
        require(bool(torch.isfinite(loss)), 'Nonfinite value loss')
        self.losses['value'].append(loss.item())
        return loss

    def train(self, examples, *args, **kwargs):
        count = len(self.losses['policy'])
        with self.metrics.timed('training'):
            start = time.perf_counter()
            super().train(examples, *args, **kwargs)
            elapsed = time.perf_counter() - start
        updates = len(self.losses['policy']) - count
        require(updates > 0, 'No optimizer updates')
        require(all(bool(torch.isfinite(v).all()) for v in self.nnet.parameters()),
                'Nonfinite parameters')
        self.training.append(dict(examples=len(examples), updates=updates,
            sampled_examples=updates * self.args['batch_size'], seconds=elapsed,
            updates_per_second=updates / elapsed))


class MeasuredCoach(Coach):
    def executeEpisodes(self):
        self.game.metrics.iteration += 1
        self.args.selfplay_seed = self.args.seed + self.game.metrics.iteration - 1
        before = len(self.game.metrics.legal)
        with self.game.metrics.timed('self_play'):
            examples = super().executeEpisodes()
        positions = len(self.game.metrics.legal) - before
        multiplicity = getattr(self.args, 'symmetry_count', 12)
        require(len(examples) == positions * multiplicity,
                'Augmentation lost or replay truncated')
        fingerprint = hashlib.sha256()
        for index, example in enumerate(examples):
            if index % multiplicity == 0:
                for value in decode(example):
                    value = np.asarray(value)
                    fingerprint.update(str((value.shape, value.dtype)).encode())
                    fingerprint.update(value.tobytes())
        self.game.metrics.original_replay.append(dict(positions=positions,
            identity_examples_sha256=fingerprint.hexdigest()))
        return examples


def inference_microbenchmark(net, example, repeats):
    state, _, _, mask, _ = decode(example)
    net.switch_target('inference')
    # Measure the actual ORT session, excluding conversions and export, after warmup.
    session = net.ort_session.session
    rows = []
    for size in (1, 2, 32):
        inputs = dict(board=np.repeat(state[None].astype(np.float32), size, axis=0),
                      valid_actions=np.repeat(mask[None], size, axis=0))
        for _ in range(3):
            session.run(None, inputs)
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            session.run(None, inputs)
            times.append(time.perf_counter() - start)
        rows.append(dict(batch=size, repeats=repeats, seconds=distribution(times),
                         states_per_second=size * repeats / sum(times)))
    return rows


def hardware():
    result = dict(platform=platform.platform(), machine=platform.machine(),
                  logical_cpus=os.cpu_count(), python=sys.version)
    if sys.platform == 'darwin':
        for key, name in (('cpu', 'machdep.cpu.brand_string'), ('ram_bytes', 'hw.memsize')):
            result[key] = subprocess.check_output(['sysctl', '-n', name], text=True).strip()
    else:
        result['cpu'] = platform.processor()
        result['ram_bytes'] = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    return result


def run_child(cli):
    ort.disable_telemetry_events()
    start = time.perf_counter()
    args = smoke_settings(argparse.Namespace(settings=None, checkpoint=cli.output,
        seed=cli.seed, backend='onnx', resume=None))
    args.parallel_inferences, args.numEps = cli.workers, cli.games
    args.numMCTSSims, args.arenaCompare = cli.simulations, cli.arena_games
    # 600 plies is a rule-derived upper bound, preserving every symmetry even in tails.
    args.maxlenOfQueue = cli.games * 600 * 12
    args.numItersHistory = 1
    args.batch_size = cli.batch_size
    args.selfplay_seed = cli.seed
    folder = Path(args.checkpoint)
    require(not folder.exists(), 'Use a new output directory')
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    metrics = Metrics()
    MeasuredGame.metrics = MeasuredNet.metrics = metrics
    game = MeasuredGame()
    nn_args = {key: getattr(args, key) for key in
        ('learn_rate', 'dropout', 'epochs', 'batch_size', 'nn_version', 'no_compression', 'q_weight')}
    net = MeasuredNet(game, nn_args)
    coach = MeasuredCoach(game, net, args)
    snapshot = backup_run_sources(args)
    settings = json.dumps(vars(args), indent=2) + '\n'
    (folder / 'settings.json').write_text(settings)
    (snapshot / 'settings.json').write_text(settings)
    dependencies = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (folder / 'requirements.txt').write_text(dependencies)
    (snapshot / 'requirements.txt').write_text(dependencies)
    # Warm compile search/transitions/symmetries and export ONNX before measured self-play.
    state = game.getInitBoard()
    mask = game.getValidMoves(state, 0)
    net.predict(state, mask)
    policy, _, _ = coach.mcts.getActionProb(state)
    game.getSymmetries(state, policy, mask)
    coach.mcts = MCTS(game, net, args)
    game.getInitBoard()
    metrics.legal.clear()
    metrics.symmetries.clear()
    setup_seconds = time.perf_counter() - start
    search = MCTS.getActionProb
    arena_play = Arena.playGames
    arena_results = []

    def measured_search(searcher, state, *positional, **keywords):
        before = state.tobytes()
        tick = time.perf_counter()
        result = search(searcher, state, *positional, **keywords)
        elapsed = time.perf_counter() - tick
        require(state.tobytes() == before, 'MCTS mutated root history')
        bucket = metrics.bucket()
        bucket['search_calls'] += 1
        bucket['simulations'] += searcher.step + 1
        bucket['search_worker_seconds'] += elapsed
        return result

    def measured_arena(arena, *positional, **keywords):
        with metrics.timed('evaluation'):
            result = arena_play(arena, *positional, **keywords)
        wins, losses, draws = result
        arena_results.append(dict(wins=wins, losses=losses, draws=draws,
            accepted=wins + losses > 0 and wins / (wins + losses) >= args.updateThreshold))
        return result

    tick = time.perf_counter()
    with patch.object(MCTS, 'getActionProb', measured_search), patch.object(Arena, 'playGames', measured_arena):
        coach.learn()
    learn_seconds = time.perf_counter() - tick
    require(len(metrics.games) == cli.games + cli.arena_games, 'Wrong game budget')
    require(sum(g['plies'] for g in metrics.games if g['phase'] == 'self_play') == len(metrics.legal),
            'Position count differs from actual moves')
    require(set(metrics.symmetries) == {12}, 'Missing 12-way symmetry coverage')
    examples = coach.trainExamplesHistory[-1]
    with (folder / 'checkpoint.examples').open('rb') as stream:
        require(list(pickle.load(stream)[-1]) == list(examples), 'Persisted replay changed')
    replay = replay_metrics(examples)
    replay['file_bytes'] = (folder / 'checkpoint.examples').stat().st_size
    micro = inference_microbenchmark(net, examples[-1], cli.inference_repeats)
    summaries = {}
    for phase in ('self_play', 'evaluation'):
        games = [g for g in metrics.games if g['phase'] == phase]
        counts = Counter(g['reason'] for g in games)
        summaries[phase] = dict(length_plies=distribution([g['plies'] for g in games]),
                               reasons={reason: counts[reason] for reason in REASONS})
    for bucket in metrics.phases.values():
        if bucket['seconds']:
            bucket['simulations_per_wall_second'] = bucket['simulations'] / bucket['seconds']
            bucket['inference_states_per_wall_second'] = bucket['inference_states'] / bucket['seconds']
        if bucket['inference_seconds']:
            bucket['inference_states_per_session_second'] = bucket['inference_states'] / bucket['inference_seconds']
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = dict(passed=True, command=sys.argv, hardware=hardware(), settings=vars(args),
        nn_args=nn_args, git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob('*.py'))},
        source_backup=str(snapshot), requirements=dependencies.splitlines(),
        format=net.checkpoint_format(), feature_config=net.nnet.feature_config,
        torch_threads=torch.get_num_threads(), onnx_providers=net.ort_session.get_providers(),
        setup_seconds=setup_seconds, learn_seconds=learn_seconds,
        learn_other_seconds=learn_seconds - sum(metrics.phases[p]['seconds'] for p in ('self_play', 'training', 'evaluation')),
        phases={p: metrics.phases[p] for p in ('self_play', 'training', 'evaluation')},
        original_positions=len(metrics.legal), augmented_examples=len(examples),
        symmetry_counts=dict(metrics.symmetries), legal_actions=distribution(metrics.legal),
        training=net.training, losses=net.losses, arena=arena_results, games=metrics.games,
        game_summary=summaries, replay=replay, inference_microbenchmark=micro,
        process_peak_rss_bytes=peak if sys.platform == 'darwin' else peak * 1024,
        artifacts={p.name: dict(bytes=p.stat().st_size, sha256=digest(p))
                   for p in sorted(folder.iterdir()) if p.is_file()},
        elapsed_seconds=time.perf_counter() - start)
    (folder / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('passed', 'original_positions', 'augmented_examples',
                     'training', 'game_summary', 'learn_seconds', 'process_peak_rss_bytes')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=1)
    parser.add_argument('--games', type=int, default=4)
    parser.add_argument('--arena-games', type=int, default=4)
    parser.add_argument('--simulations', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=140)
    parser.add_argument('--inference-repeats', type=int, default=30)
    parser.add_argument('--timeout', type=int, default=600, help='Hard subprocess deadline in seconds')
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    cli = parser.parse_args()
    require(1 <= cli.games <= 32 and 2 <= cli.arena_games <= 32 and cli.arena_games % 2 == 0,
            'Use 1..32 self-play and 2..32 even arena games')
    require(2 <= cli.simulations <= 128 and 2 <= cli.batch_size <= 256,
            'Use 2..128 simulations and batch 2..256')
    require(1 <= cli.timeout <= 3600 and 1 <= cli.inference_repeats <= 1000,
            'Timeout/repetitions exceed benchmark bounds')
    if cli.child:
        run_child(cli)
    else:
        # Threads in the shared inference protocol cannot be forcibly cancelled safely.
        # A process deadline bounds hangs as well as unexpectedly expensive games.
        command = [sys.executable, '-m', 'intransitive.benchmark_training', *sys.argv[1:], '--child']
        started = time.perf_counter()
        try:
            result = subprocess.run(command, timeout=cli.timeout)
        except subprocess.TimeoutExpired:
            raise SystemExit(f'Benchmark exceeded {cli.timeout}s; no successful result')
        if result.returncode:
            raise SystemExit(result.returncode)
        report = Path(cli.output).resolve() / 'report.json'
        require(report.exists(), 'Child exited without a complete report')
        data = json.loads(report.read_text())
        data.update(process_exit_code=0, child_process_seconds=time.perf_counter() - started,
                    hard_timeout_seconds=cli.timeout)
        report.write_text(json.dumps(data, indent=2) + '\n')
        print('Completed:', report)


if __name__ == '__main__':
    main()
