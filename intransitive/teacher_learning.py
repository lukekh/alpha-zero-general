"""Reproducible heuristic-teacher and bounded continuation experiment.

The experiment deliberately keeps teacher policy, terminal outcome and MCTS-Q
semantics separate.  See benchmarks/teacher/README.md for the frozen protocol.
"""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import resource
import shutil
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

os.environ['ORT_DISABLE_TELEMETRY'] = '1'
import numpy as np
import onnxruntime as ort

ort.disable_telemetry_events()
import torch

from Coach import Coach
from MCTS import MCTS
from source_backup import backup_run_sources
from .heuristics import AlphaBetaPlayer, SearchConfig
from .IntransitiveConstants import action_destination
from .IntransitiveGame import IntransitiveGame
from .IntransitiveLogicNumba import Board, validate_state
from .IntransitivePlayers import GreedyPlayer, RandomPlayer
from .IntransitiveSymmetries import NUM_SYMMETRIES, transform_action_vector, transform_state
from .NNet import NNetWrapper
from .baseline import distribution, hardware, seed_all, summarize
from .smoke import digest, require
from .tests.tactical_oracle import load_case


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = Path(__file__).parent / 'benchmarks/teacher/protocol.json'
TACTICS = Path(__file__).parent / 'heuristics/tactics.json'
FROZEN_ARCHIVE = Path(__file__).parent / 'benchmarks/symmetry/symmetry-artifacts.tar.gz'
DATA_SCHEMA = 'intransitive-teacher-dataset-1'
REPORT_SCHEMA = 'intransitive-teacher-experiment-1'
ARMS = ('scratch_selfplay', 'pretrained_only', 'pretrained_selfplay', 'annealed_mix')


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def stable_pickle(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
            pickle.dump(data, stream, protocol=5)
    temporary.replace(path)


def load_dataset(path):
    with gzip.open(path, 'rb') as stream:
        data = pickle.load(stream)
    validate_dataset(data)
    return data


def policy_entropy(policy):
    positive = np.asarray(policy)[np.asarray(policy) > 0]
    return float(-np.sum(positive * np.log(positive)))


def state_hash(state):
    return hashlib.sha256(state.tobytes()).hexdigest()


def parameter_hash(net):
    value = hashlib.sha256()
    for name, parameter in sorted(net.nnet.state_dict().items()):
        value.update(name.encode())
        value.update(parameter.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def optimizer_hash(net):
    state = (net.optimizer.state_dict() if net.optimizer is not None
             else net._pending_optimizer_state)
    return hashlib.sha256(pickle.dumps(state, protocol=5)).hexdigest() if state else None


def canonical_transform(game, state, policy, legal, symmetry):
    transformed = transform_state(state, symmetry)
    mover = int(transformed[:, :, 82:84].flat[1])
    transformed = game.getCanonicalForm(transformed, mover)
    transformed_policy = transform_action_vector(policy, symmetry).astype(np.float32)
    transformed_legal = transform_action_vector(legal, symmetry).astype(np.bool_)
    return transformed, transformed_policy, transformed_legal


def orbit_hash(game, canonical):
    empty = np.zeros(game.getActionSize(), dtype=np.float32)
    legal = game.getValidMoves(canonical, 0)
    states = [canonical_transform(game, canonical, empty, legal, symmetry)[0].tobytes()
              for symmetry in range(NUM_SYMMETRIES)]
    return hashlib.sha256(min(states)).hexdigest()


def immediate_wins(game, canonical):
    wins = []
    for action in map(int, np.flatnonzero(game.getValidMoves(canonical, 0))):
        child, side = game.getNextState(canonical, 0, action)
        if game.getGameEnded(child, side)[0] == 1:
            wins.append(action)
    return wins


def safe_actions(game, canonical):
    safe = []
    for action in map(int, np.flatnonzero(game.getValidMoves(canonical, 0))):
        child, side = game.getNextState(canonical, 0, action)
        if game.getGameEnded(child, side).any():
            safe.append(action)
            continue
        reply = game.getCanonicalForm(child, side)
        if not immediate_wins(game, reply):
            safe.append(action)
    return safe


def decision_categories(game, canonical, action):
    result = []
    legal = list(map(int, np.flatnonzero(game.getValidMoves(canonical, 0))))
    wins = immediate_wins(game, canonical)
    if action in wins:
        result.append('immediate_win')
    x, y = action_destination(action)
    if canonical[y, x, 0] != 0:
        result.append('capture')
    safe = safe_actions(game, canonical)
    if action in safe and len(safe) < len(legal):
        result.append('avoids_immediate_loss')
    goal = 80 if int(canonical[:, :, 82:84].flat[2]) == 0 else 0
    if max(abs(x - goal % 9), abs(y - goal // 9)) <= 1:
        result.append('goal_threat')
    return result or ['quiet']


def teacher_config(protocol):
    config = SearchConfig(**protocol['teacher']['search'])
    if config.nmp_enabled or config.futility_enabled:
        raise ValueError('Teacher protocol requires selective pruning disabled')
    return config


def split_trajectories(records, split_seed, fractions):
    trajectory_ids = sorted({record['trajectory_id'] for record in records})
    require(len(trajectory_ids) >= 5, 'Need at least five trajectories for isolated splits')
    rng = np.random.default_rng(split_seed)
    rng.shuffle(trajectory_ids)
    validation_count = max(1, round(len(trajectory_ids) * fractions['validation']))
    test_count = max(1, round(len(trajectory_ids) * fractions['test']))
    require(validation_count + test_count < len(trajectory_ids), 'No training trajectories')
    assignments = {}
    for index, trajectory_id in enumerate(trajectory_ids):
        assignments[trajectory_id] = ('test' if index < test_count else
                                      'validation' if index < test_count + validation_count else
                                      'train')

    # The official opening and shared prefixes occur in several trajectories.
    # Give each complete symmetry orbit one split owner, then drop cross-split
    # duplicates.  This is stricter than merely splitting before augmentation.
    owners, kept, dropped = {}, [], Counter()
    priority = {'test': 0, 'validation': 1, 'train': 2}
    for record in sorted(records, key=lambda row: (
            priority[assignments[row['trajectory_id']]], row['trajectory_id'], row['ply'])):
        split = assignments[record['trajectory_id']]
        owner = owners.setdefault(record['orbit_sha256'], split)
        if owner == split:
            record['split'] = split
            kept.append(record)
        else:
            dropped[f'{split}_overlaps_{owner}'] += 1
    require({r['split'] for r in kept} == {'train', 'validation', 'test'}, 'Empty split')
    return kept, assignments, dict(dropped)


def augment_records(game, records):
    result = {'train': [], 'validation': [], 'test': []}
    for record in records:
        for symmetry in range(NUM_SYMMETRIES):
            state, policy, legal = canonical_transform(
                game, record['state'], record['policy'], record['legal'], symmetry)
            result[record['split']].append(dict(
                state=state, policy=policy, outcome=record['outcome'].copy(),
                legal=legal, q=np.zeros(2, dtype=np.float32),
                value_mask=np.ones(2, dtype=np.bool_),
                q_mask=np.zeros(2, dtype=np.bool_),
                trajectory_id=record['trajectory_id'], ply=record['ply'],
                source=record['source'], symmetry=symmetry,
                original_state_sha256=record['state_sha256'],
                orbit_sha256=record['orbit_sha256']))
    return result


def training_tuple(record):
    return tuple(record[key] for key in (
        'state', 'policy', 'outcome', 'legal', 'q', 'value_mask', 'q_mask'))


def validate_dataset(data):
    require(data.get('schema') == DATA_SCHEMA, 'Unknown teacher dataset schema')
    game = IntransitiveGame()
    trajectory_sets, orbit_sets = {}, {}
    for split in ('train', 'validation', 'test'):
        examples = data['examples'][split]
        require(examples, f'Empty {split} split')
        trajectory_sets[split] = {row['trajectory_id'] for row in examples}
        orbit_sets[split] = {row['orbit_sha256'] for row in examples}
        groups = defaultdict(set)
        for row in examples:
            validate_state(row['state'])
            require(int(row['state'][:, :, 82:84].flat[1]) == 0, 'Example is not canonical')
            require(not game.getGameEnded(row['state'], 0).any(), 'Terminal training state')
            require(row['policy'].shape == row['legal'].shape == (game.getActionSize(),),
                    'Invalid action target shape')
            require(np.count_nonzero(row['policy']) == 1 and row['policy'].sum() == 1,
                    'Teacher target must be one-hot')
            require(not row['policy'][~row['legal']].any(), 'Teacher selected an illegal move')
            require(row['value_mask'].all() and not row['q_mask'].any(),
                    'Teacher target masks changed')
            groups[(row['trajectory_id'], row['ply'])].add(row['symmetry'])
        require(all(ids == set(range(NUM_SYMMETRIES)) for ids in groups.values()),
                f'Missing symmetry in {split}')
    for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        require(trajectory_sets[left].isdisjoint(trajectory_sets[right]), 'Trajectory leakage')
        require(orbit_sets[left].isdisjoint(orbit_sets[right]), 'Symmetry-orbit leakage')
    return True


def generate_dataset(output, protocol):
    output = Path(output)
    game = IntransitiveGame()
    config = teacher_config(protocol)
    # Warm every compiled rule/search query before accounting teacher work.
    AlphaBetaPlayer(game, config).analyze(game.getInitBoard())
    GreedyPlayer(game).play(game.getInitBoard())
    records, trajectories = [], []
    started = time.perf_counter()
    for spec in protocol['dataset']['trajectories']:
        trajectory_id, actors = spec['id'], spec['players']
        rng = np.random.default_rng(np.random.SeedSequence(
            [protocol['dataset']['seed'], spec['seed']]))
        state, side, actions = game.getInitBoard(), 0, []
        positions = []
        teacher = AlphaBetaPlayer(game, config)
        for ply in range(1, 602):
            outcome = game.getGameEnded(state, side)
            if outcome.any():
                break
            canonical = game.getCanonicalForm(state, side)
            before = canonical.copy()
            analysis = teacher.analyze(canonical)
            require(not analysis.stopped and analysis.completed_depth == config.max_depth,
                    'Teacher failed to complete its fixed-depth target')
            action = int(analysis.action)
            legal = game.getValidMoves(canonical, 0)
            policy = np.zeros(game.getActionSize(), dtype=np.float32)
            policy[action] = 1
            actor = actors[side]
            if actor == 'teacher':
                played = action
            elif actor == 'greedy':
                played = GreedyPlayer(game).play(canonical)
            elif actor == 'random':
                played = int(rng.choice(np.flatnonzero(legal)))
            else:
                raise ValueError(f'Unknown dataset actor {actor}')
            require(legal[played], 'Dataset actor selected an illegal move')
            categories = decision_categories(game, canonical, action)
            positions.append(dict(
                state=canonical, policy=policy, legal=legal.astype(np.bool_),
                physical_player=side, teacher_action=action, played_action=played,
                teacher_depth=analysis.completed_depth, teacher_nodes=analysis.nodes,
                teacher_work=analysis.work, teacher_proof_nodes=analysis.proof_nodes,
                teacher_seconds=analysis.elapsed, categories=categories,
                trajectory_id=trajectory_id, ply=ply,
                source=f'{actors[0]}-blue_vs_{actors[1]}-red',
                state_sha256=state_hash(canonical), orbit_sha256=orbit_hash(game, canonical)))
            np.testing.assert_array_equal(canonical, before)
            actions.append(played)
            state, side = game.getNextState(state, side, played)
        else:
            raise AssertionError('Dataset game exceeded the rules-derived bound')
        outcome = game.getGameEnded(state, side).astype(np.float32)
        require(outcome.any(), 'Dataset trajectory did not terminate under exact rules')
        board = Board()
        board.copy_state(state, False)
        trajectory_sha = hashlib.sha256(np.asarray(actions, dtype=np.int16).tobytes()).hexdigest()
        for position in positions:
            position['outcome'] = np.roll(outcome, -position['physical_player']).copy()
            records.append(position)
        trajectories.append(dict(
            id=trajectory_id, players=actors, seed=spec['seed'], plies=len(actions),
            actions=actions, outcome=outcome.tolist(), reason=board.get_terminal_reason(),
            captures=20-int(np.count_nonzero(state[:, :, 0])),
            teacher_disagreements=sum(p['teacher_action'] != p['played_action'] for p in positions),
            trajectory_sha256=trajectory_sha, final_state_sha256=state_hash(state)))
    records, assignments, dropped = split_trajectories(
        records, protocol['dataset']['split_seed'], protocol['dataset']['split_fractions'])
    examples = augment_records(game, records)
    # Search latency belongs in the measured manifest, not in the target corpus:
    # excluding it makes the seeded compressed dataset byte-reproducible.
    payload_records = [{key: value for key, value in record.items()
                        if key != 'teacher_seconds'} for record in records]
    data = dict(schema=DATA_SCHEMA, teacher_config=config.to_dict(), records=payload_records,
                examples=examples, trajectories=trajectories, split_assignments=assignments,
                dropped_cross_split_orbits=dropped)
    validate_dataset(data)
    stable_pickle(output, data)
    categories = Counter(category for row in records for category in row['categories'])
    outcomes = Counter('draw' if abs(t['outcome'][0]) < 1 else
                       'blue_win' if t['outcome'][0] == 1 else 'red_win'
                       for t in trajectories)
    manifest = dict(
        schema=DATA_SCHEMA, file=output.name, sha256=digest(output), bytes=output.stat().st_size,
        teacher_config=config.to_dict(), teacher_config_sha256=hashlib.sha256(
            config.identity().encode()).hexdigest(), elapsed_seconds=time.perf_counter()-started,
        original_positions=len(records), augmented_examples=sum(map(len, examples.values())),
        examples_by_split={name: len(rows) for name, rows in examples.items()},
        original_positions_by_split={name: sum(r['split'] == name for r in records)
                                     for name in examples},
        trajectories_by_split={name: sorted(t for t, split in assignments.items() if split == name)
                               for name in examples},
        trajectory_sets_disjoint=True, symmetry_orbit_sets_disjoint=True,
        split_before_augmentation=True, symmetries_per_original=NUM_SYMMETRIES,
        categories=dict(categories), outcomes=dict(outcomes),
        physical_players=dict(Counter(str(r['physical_player']) for r in records)),
        teacher_analyses=len(records), teacher_nodes=sum(r['teacher_nodes'] for r in records),
        teacher_work=sum(r['teacher_work'] for r in records),
        teacher_proof_nodes=sum(r['teacher_proof_nodes'] for r in records),
        teacher_search_seconds=sum(r['teacher_seconds'] for r in records),
        learner_mistake_positions=sum(r['teacher_action'] != r['played_action'] for r in records),
        terminal_states_in_training=0, policy_target='one-hot completed-depth alpha-beta action',
        value_target='exact eventual terminal outcome in canonical current-player order',
        q_target='absent; zero storage placeholder with q_mask false',
        dropped_cross_split_orbits=dropped, trajectories=trajectories)
    write_json(output.with_name('dataset-manifest.json'), manifest)
    return data, manifest


class TrackingGame(IntransitiveGame):
    def __init__(self):
        super().__init__()
        self.trajectories, self.current, self.policies = [], [], []

    def getInitBoard(self):
        if self.current:
            self.trajectories.append(self.current)
        self.current = []
        return super().getInitBoard()

    def getNextState(self, board, player, action, random_seed=0):
        child, side = super().getNextState(board, player, action, random_seed)
        self.current.append(int(action))
        if self.getGameEnded(child, side).any():
            self.trajectories.append(self.current)
            self.current = []
        return child, side

    def getSymmetries(self, board, policy, valid):
        self.policies.append(dict(entropy=policy_entropy(policy),
                                  legal=int(np.count_nonzero(valid)),
                                  state_sha256=state_hash(board)))
        return super().getSymmetries(board, policy, valid)


def nn_args(protocol, *, persistent=True):
    values = dict(protocol['training']['network'])
    values['persist_optimizer'] = persistent
    return values


def coach_args(protocol, checkpoint, seed, noise):
    values = dict(protocol['self_play']['settings'])
    values.update(checkpoint=str(checkpoint), no_compression=True,
                  numEps=protocol['self_play']['games_per_phase'],
                  selfplay_seed=seed, dirichletAlpha=noise)
    return SimpleNamespace(**values)


def generate_selfplay(net, protocol, checkpoint, seed, noise=None):
    noise = protocol['self_play']['root_noise'] if noise is None else noise
    game = TrackingGame()
    # The caller's weights are copied into a wrapper attached to the tracking
    # game, avoiding shared mutable Board instances between network and Coach.
    actor = NNetWrapper(game, dict(net.args))
    actor.nnet.load_state_dict(net.nnet.state_dict())
    actor.device['inference'] = 'cpu'
    args = coach_args(protocol, checkpoint, seed, noise)
    coach = Coach(game, actor, args)
    coach.mcts.rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4D435453]))
    started, cpu = time.perf_counter(), time.process_time()
    examples = list(coach.executeEpisodes())
    trajectories = list(game.trajectories)
    require(len(trajectories) == args.numEps, 'Incomplete self-play phase')
    summaries = []
    replay = IntransitiveGame()
    for index, actions in enumerate(trajectories):
        state, side = replay.getInitBoard(), 0
        captures = 0
        for action in actions:
            x, y = action_destination(action)
            captures += int(state[y, x, 0] != 0)
            state, side = replay.getNextState(state, side, action)
        outcome = replay.getGameEnded(state, side)
        require(outcome.any(), 'Self-play record is incomplete')
        board = Board()
        board.copy_state(state, False)
        summaries.append(dict(index=index, actions=actions, plies=len(actions), captures=captures,
                              outcome=outcome.tolist(), reason=board.get_terminal_reason(),
                              trajectory_sha256=hashlib.sha256(
                                  np.asarray(actions, dtype=np.int16).tobytes()).hexdigest()))
    return examples, dict(
        games=len(trajectories), original_positions=len(game.policies),
        augmented_examples=len(examples), root_noise=noise,
        policy_entropy=distribution([row['entropy'] for row in game.policies]),
        legal_actions=distribution([row['legal'] for row in game.policies]),
        unique_position_hashes=len({row['state_sha256'] for row in game.policies}),
        unique_trajectories=len({row['trajectory_sha256'] for row in summaries}),
        decisive_games=sum(abs(row['outcome'][0]) == 1 for row in summaries),
        captures=sum(row['captures'] for row in summaries),
        reasons=dict(Counter(row['reason'] for row in summaries)), games_detail=summaries,
        wall_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu)


def exploration_experiment(protocol, output):
    rows = []
    for seed in protocol['training']['seeds']:
        seed_all(seed)
        game = IntransitiveGame()
        initial = NNetWrapper(game, nn_args(protocol))
        for label, noise in (('baseline_no_noise', 0.),
                             ('root_noise', protocol['self_play']['root_noise'])):
            phase_seed = int(np.random.SeedSequence([seed, 33, 0]).generate_state(1)[0])
            _, metrics = generate_selfplay(initial, protocol, Path(output) / label,
                                           phase_seed, noise=noise)
            rows.append(dict(seed=seed, arm=label, phase_seed=phase_seed, **metrics))
    summary = {}
    for arm in ('baseline_no_noise', 'root_noise'):
        selected = [row for row in rows if row['arm'] == arm]
        summary[arm] = dict(
            seeds=len(selected), games=sum(row['games'] for row in selected),
            decisive_games=sum(row['decisive_games'] for row in selected),
            captures=sum(row['captures'] for row in selected),
            unique_trajectories=sum(row['unique_trajectories'] for row in selected),
            unique_positions=sum(row['unique_position_hashes'] for row in selected),
            mean_policy_entropy=float(np.mean([row['policy_entropy']['mean'] for row in selected])),
            wall_seconds=sum(row['wall_seconds'] for row in selected),
            cpu_seconds=sum(row['cpu_seconds'] for row in selected))
    result = dict(settings_held_equal=sorted(protocol['self_play']['settings']),
                  changed_field='dirichletAlpha', rows=rows, summary=summary,
                  limits='Bounded paired trajectory pilot; diversity and outcomes are descriptive, not a strength estimate.')
    write_json(Path(output) / 'exploration.json', result)
    return result


def sample_mix(teacher, selfplay, teacher_fraction, seed):
    require(0 <= teacher_fraction < 1, 'Teacher fraction must be in [0, 1)')
    if not teacher_fraction:
        return list(selfplay), dict(teacher_examples=0, selfplay_examples=len(selfplay),
                                    effective_teacher_fraction=0.)
    teacher_count = max(1, round(len(selfplay) * teacher_fraction / (1-teacher_fraction)))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(teacher), size=teacher_count, replace=teacher_count > len(teacher))
    mixed = list(selfplay) + [teacher[int(index)] for index in chosen]
    rng.shuffle(mixed)
    return mixed, dict(teacher_examples=teacher_count, selfplay_examples=len(selfplay),
                       effective_teacher_fraction=teacher_count/len(mixed))


def evaluate_examples(net, examples):
    net.device['inference'] = 'cpu'
    correct, value_error, entropy = 0, [], []
    for row in examples:
        policy, value = net.predict(row['state'], row['legal'])
        correct += int(np.argmax(policy) == np.argmax(row['policy']))
        value_error.extend((value-row['outcome']) ** 2)
        entropy.append(policy_entropy(policy))
    return dict(examples=len(examples), policy_top1=correct/len(examples),
                value_mse=float(np.mean(value_error)),
                policy_entropy=distribution(entropy))


def train_phase(net, examples, phase, output, seed, metadata):
    seed_all(seed)
    net.args['batches_per_epoch'] = metadata['updates']
    before = parameter_hash(net)
    optimizer_before = optimizer_hash(net)
    started, cpu = time.perf_counter(), time.process_time()
    net.train(examples)
    checkpoint = Path(output) / f'{phase}.pt'
    net.save_checkpoint(str(checkpoint.parent), checkpoint.name,
                        additional_keys=dict(teacher_learning=metadata, phase=phase, seed=seed))
    return dict(phase=phase, seed=seed, examples=len(examples), updates=metadata['updates'],
                cumulative_optimizer_updates=net.optimizer_updates,
                parameter_sha256_before=before, parameter_sha256_after=parameter_hash(net),
                optimizer_sha256_before=optimizer_before, optimizer_sha256_after=optimizer_hash(net),
                checkpoint=checkpoint.name, checkpoint_sha256=digest(checkpoint),
                wall_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu,
                target_semantics=metadata['target_semantics'], mixture=metadata.get('mixture'))


def reload_net(path, protocol):
    game = IntransitiveGame()
    net = NNetWrapper(game, nn_args(protocol))
    metadata = net.load_checkpoint(str(Path(path).parent), Path(path).name)
    require(net.optimizer_updates == metadata['optimizer_updates'], 'Optimizer update count changed')
    require(net._pending_optimizer_state is not None, 'Optimizer state was not restored')
    return net


def train_arm(arm, seed, dataset, protocol, output):
    output = Path(output)
    output.mkdir(parents=True)
    seed_all(seed)
    net = NNetWrapper(IntransitiveGame(), nn_args(protocol))
    initial = parameter_hash(net)
    teacher = [training_tuple(row) for row in dataset['examples']['train']]
    phases, selfplay_metrics = [], []
    updates = protocol['training']['updates_per_phase']
    semantics = dict(policy='teacher one-hot or neural MCTS visits',
                     value='exact terminal outcome',
                     q='masked for teacher; MCTS estimate for self-play')

    if arm != 'scratch_selfplay':
        phase_seed = int(np.random.SeedSequence([seed, 35, 0]).generate_state(1)[0])
        phases.append(train_phase(net, teacher, 'pretrain', output, phase_seed,
            dict(updates=updates, target_semantics=semantics,
                 mixture=dict(teacher_examples=len(teacher), selfplay_examples=0,
                              effective_teacher_fraction=1.))))
        if arm == 'pretrained_only':
            shutil.copyfile(output / 'pretrain.pt', output / 'final.pt')
            return net, dict(arm=arm, seed=seed, initial_parameter_sha256=initial,
                             phases=phases, selfplay=[], final_checkpoint='final.pt',
                             final_checkpoint_sha256=digest(output / 'final.pt'),
                             optimizer_updates=net.optimizer_updates)
        net = reload_net(output / 'pretrain.pt', protocol)

    fractions = (protocol['training']['anneal_teacher_fractions'] if arm == 'annealed_mix'
                 else [0.] * len(protocol['training']['anneal_teacher_fractions']))
    for index, fraction in enumerate(fractions, 1):
        play_seed = int(np.random.SeedSequence([seed, 35, index]).generate_state(1)[0])
        selfplay, metrics = generate_selfplay(net, protocol, output, play_seed)
        selfplay_metrics.append(dict(phase=index, seed=play_seed, **metrics))
        examples, mixture = sample_mix(teacher, selfplay, fraction,
            int(np.random.SeedSequence([seed, 35, index, 1]).generate_state(1)[0]))
        phase_seed = int(np.random.SeedSequence([seed, 35, index, 2]).generate_state(1)[0])
        name = f'continue_{index}'
        phases.append(train_phase(net, examples, name, output, phase_seed,
            dict(updates=updates, target_semantics=semantics, mixture=mixture)))
        # Exercise disk continuation, not merely persistence in one Python object.
        net = reload_net(output / f'{name}.pt', protocol)
    shutil.copyfile(output / f'continue_{len(fractions)}.pt', output / 'final.pt')
    return net, dict(arm=arm, seed=seed, initial_parameter_sha256=initial,
                     phases=phases, selfplay=selfplay_metrics,
                     final_checkpoint='final.pt', final_checkpoint_sha256=digest(output / 'final.pt'),
                     optimizer_updates=net.optimizer_updates)


def tactical_evaluation(net):
    game = IntransitiveGame()
    cases = json.loads(TACTICS.read_text())['positions']
    correct, by_category = 0, defaultdict(lambda: [0, 0])
    teacher_correct = 0
    config = SearchConfig(max_depth=1, node_limit=500000, time_limit=60,
                          proof_depth=2, proof_nodes=64)
    teacher = AlphaBetaPlayer(game, config)
    net.device['inference'] = 'cpu'
    for case in cases:
        reference, expected = load_case(case)
        physical = reference.storage()
        canonical = game.getCanonicalForm(physical, case['player'])
        legal = game.getValidMoves(canonical, 0)
        policy, _ = net.predict(canonical, legal)
        hit = int(np.argmax(policy) == expected)
        correct += hit
        by_category[case['category']][0] += hit
        by_category[case['category']][1] += 1
        teacher_correct += int(teacher.analyze(canonical).action == expected)
    return dict(cases=len(cases), policy_top1=correct/len(cases), correct=correct,
                by_category={name: dict(correct=value[0], cases=value[1],
                                        accuracy=value[0]/value[1])
                             for name, value in sorted(by_category.items())},
                dataset_teacher_depth_one_accuracy=teacher_correct/len(cases),
                teacher_bias_note=('The dataset teacher is itself imperfect on this independent suite; '
                                   'student agreement cannot exceed or repair that bias automatically.'))


class ModelPolicyPlayer:
    kind = 'model_only'

    def __init__(self, game, net):
        self.game, self.net = game, net

    def play(self, canonical, nb_moves=0):
        legal = self.game.getValidMoves(canonical, 0)
        policy, _ = self.net.predict(canonical, legal)
        return int(np.argmax(policy))


class TacticalHybridPlayer(ModelPolicyPlayer):
    """Network policy with an exact two-ply win/loss safety filter."""

    kind = 'hybrid_tactical_verifier'

    def play(self, canonical, nb_moves=0):
        legal = self.game.getValidMoves(canonical, 0)
        policy, _ = self.net.predict(canonical, legal)
        wins = immediate_wins(self.game, canonical)
        candidates = wins or safe_actions(self.game, canonical) or list(map(int, np.flatnonzero(legal)))
        return max(candidates, key=lambda action: (policy[action], -action))


class MCTSPlayer:
    kind = 'neural_mcts'

    def __init__(self, game, net, protocol):
        args = SimpleNamespace(**protocol['self_play']['settings'])
        self.search = MCTS(game, net, args)

    def play(self, canonical, nb_moves=0):
        policy, _, _ = self.search.getActionProb(canonical, temp=0, force_full_search=True)
        return int(np.argmax(policy))


def play_game(candidate, opponent, candidate_colour, seed):
    seed_all(seed)
    game = IntransitiveGame()
    state, side, actions, latencies = game.getInitBoard(), 0, [], []
    for _ in range(601):
        outcome = game.getGameEnded(state, side)
        if outcome.any():
            break
        canonical = game.getCanonicalForm(state, side)
        actor = candidate if side == candidate_colour else opponent
        started = time.perf_counter()
        action = int(actor.play(canonical, len(actions) + 1))
        elapsed = time.perf_counter() - started
        require(game.getValidMoves(canonical, 0)[action], 'Evaluation player selected illegal move')
        if side == candidate_colour:
            latencies.append(elapsed)
        actions.append(action)
        state, side = game.getNextState(state, side, action)
    else:
        raise AssertionError('Evaluation game exceeded the rules-derived bound')
    board = Board()
    board.copy_state(state, False)
    reward = outcome[candidate_colour]
    label = 'wins' if reward == 1 else 'losses' if reward == -1 else 'draws'
    return dict(outcome=label, reason=board.get_terminal_reason(), plies=len(actions),
                captures=20-int(np.count_nonzero(state[:, :, 0])),
                candidate_colour='Blue' if candidate_colour == 0 else 'Red',
                physical_first_player='Blue', trajectory_sha256=hashlib.sha256(
                    np.asarray(actions, dtype=np.int16).tobytes()).hexdigest(),
                latency_seconds=distribution(latencies))


def frozen_net(protocol, output):
    target = Path(output) / 'frozen-issue16.pt'
    if not target.exists():
        with tarfile.open(FROZEN_ARCHIVE) as archive:
            target.write_bytes(archive.extractfile('on/baseline.pt').read())
    game = IntransitiveGame()
    net = NNetWrapper(game, dict(nn_args(protocol), nn_version=-1,
                                 persist_optimizer=False))
    net.load_checkpoint(str(target.parent), target.name)
    net.device['inference'] = 'cpu'
    return net, target


def summarize_games(rows):
    summary = summarize(rows)
    summary['by_colour'] = {
        colour: summarize([row for row in rows if row['candidate_colour'] == colour])
        for colour in ('Blue', 'Red')}
    summary['latency_seconds'] = distribution([
        row['latency_seconds']['mean'] for row in rows])
    summary['captures'] = sum(row['captures'] for row in rows)
    return summary


def strength_evaluation(net, protocol, frozen, seed, agents=('model_only',), opponents=None):
    started, cpu = time.perf_counter(), time.process_time()
    rows = []
    for agent_name in agents:
        for opponent_name in (opponents or protocol['evaluation']['opponents']):
            for colour in (0, 1):
                stream = int(np.random.SeedSequence(
                    [seed, 3500, len(rows), colour]).generate_state(1)[0])
                game = IntransitiveGame()
                if agent_name == 'model_only':
                    candidate = ModelPolicyPlayer(game, net)
                elif agent_name == 'neural_mcts':
                    candidate = MCTSPlayer(game, net, protocol)
                elif agent_name == 'hybrid':
                    candidate = TacticalHybridPlayer(game, net)
                elif agent_name == 'heuristic_only':
                    candidate = AlphaBetaPlayer(game, teacher_config(protocol))
                else:
                    raise ValueError(agent_name)
                if opponent_name == 'random':
                    opponent = RandomPlayer(game, stream)
                elif opponent_name == 'greedy':
                    opponent = GreedyPlayer(game)
                elif opponent_name == 'frozen_neural':
                    opponent = ModelPolicyPlayer(game, frozen)
                elif opponent_name == 'alpha_beta':
                    opponent = AlphaBetaPlayer(game, teacher_config(protocol))
                else:
                    raise ValueError(opponent_name)
                row = play_game(candidate, opponent, colour, stream)
                rows.append(dict(agent=agent_name, opponent=opponent_name,
                                 seed=seed, stream=stream, **row))
    grouped = {}
    for key in sorted({(row['agent'], row['opponent']) for row in rows}):
        selected = [row for row in rows if (row['agent'], row['opponent']) == key]
        grouped['/'.join(key)] = summarize_games(selected)
    return dict(rows=rows, summaries=grouped,
                wall_seconds=time.perf_counter()-started,
                cpu_seconds=time.process_time()-cpu)


def run_experiment(root, dataset, protocol):
    root = Path(root)
    frozen, frozen_path = frozen_net(protocol, root)
    runs, untrained_runs, evaluations, assisted_evaluations = [], [], [], []
    validation = dataset['examples']['validation']
    test = dataset['examples']['test']
    for seed in protocol['training']['seeds']:
        seed_all(seed)
        untrained = NNetWrapper(IntransitiveGame(), nn_args(protocol))
        initial_strength = strength_evaluation(untrained, protocol, frozen, seed)
        initial_assisted = strength_evaluation(
            untrained, protocol, frozen, seed, agents=('neural_mcts',),
            opponents=protocol['evaluation']['search_assisted_opponents'])
        untrained_runs.append(dict(
            seed=seed, parameter_sha256=parameter_hash(untrained),
            validation=evaluate_examples(untrained, validation),
            test=evaluate_examples(untrained, test), tactical=tactical_evaluation(untrained),
            model_only_strength=initial_strength['summaries'],
            neural_mcts_strength=initial_assisted['summaries']))
        evaluations.extend(dict(arm='untrained', **row) for row in initial_strength['rows'])
        assisted_evaluations.extend(dict(arm='untrained', **row)
                                    for row in initial_assisted['rows'])
        expected_initial = parameter_hash(untrained)
        for arm in ARMS:
            folder = root / 'runs' / str(seed) / arm
            net, run = train_arm(arm, seed, dataset, protocol, folder)
            require(run['initial_parameter_sha256'] == expected_initial,
                    'Matched initialization changed between arms')
            run['validation'] = evaluate_examples(net, validation)
            run['test'] = evaluate_examples(net, test)
            run['tactical'] = tactical_evaluation(net)
            strength = strength_evaluation(net, protocol, frozen, seed)
            assisted = strength_evaluation(
                net, protocol, frozen, seed, agents=('neural_mcts',),
                opponents=protocol['evaluation']['search_assisted_opponents'])
            run['model_only_strength'] = strength['summaries']
            run['neural_mcts_strength'] = assisted['summaries']
            run['model_only_evaluation_wall_seconds'] = strength['wall_seconds']
            run['model_only_evaluation_cpu_seconds'] = strength['cpu_seconds']
            run['neural_mcts_evaluation_wall_seconds'] = assisted['wall_seconds']
            run['neural_mcts_evaluation_cpu_seconds'] = assisted['cpu_seconds']
            evaluations.extend(dict(arm=arm, **row) for row in strength['rows'])
            assisted_evaluations.extend(dict(arm=arm, **row) for row in assisted['rows'])
            write_json(folder / 'run.json', run)
            runs.append(run)
    # Use validation policy accuracy only; the test set and game results cannot
    # select the published reloadable checkpoint.
    candidates = [run for run in runs if run['arm'] == 'annealed_mix']
    selected = max(candidates, key=lambda run: (run['validation']['policy_top1'], -run['seed']))
    selected_folder = root / 'runs' / str(selected['seed']) / selected['arm']
    shutil.copyfile(selected_folder / 'final.pt', root / 'selected.pt')
    selected_net = reload_net(root / 'selected.pt', protocol)
    deployment = strength_evaluation(selected_net, protocol, frozen, selected['seed'],
        agents=('model_only', 'neural_mcts', 'hybrid', 'heuristic_only'))
    # Prove existing inference entry points need no teacher service.
    selected_net.device['inference'] = 'cpu'
    state = IntransitiveGame().getInitBoard()
    cpu = selected_net.predict(state, IntransitiveGame().getValidMoves(state, 0))
    selected_net.device['inference'] = 'onnx'
    onnx = selected_net.predict(state, IntransitiveGame().getValidMoves(state, 0))
    parity = [float(np.max(np.abs(a-b))) for a, b in zip(cpu, onnx)]
    from .play import ModelOpponent
    browser_model = ModelOpponent(root / 'selected.pt', protocol['evaluation']['mcts_simulations'])
    browser_action = browser_model.choose(state, 0)
    require(IntransitiveGame().getValidMoves(state, 0)[browser_action],
            'Browser/terminal model loader selected an illegal action')
    return dict(runs=runs, untrained_runs=untrained_runs,
                model_only_games=evaluations,
                neural_mcts_games=assisted_evaluations,
                selected=dict(arm=selected['arm'], seed=selected['seed'],
                              criterion='highest validation policy top-1 within annealed_mix; seed tie-break',
                              checkpoint='selected.pt', sha256=digest(root / 'selected.pt')),
                deployment=deployment, frozen_checkpoint_sha256=digest(frozen_path),
                reload=dict(cpu_onnx_max_abs=parity, browser_model_action=browser_action,
                            teacher_required_at_inference=False))


def aggregate_runs(runs):
    result = {}
    for arm in ARMS:
        selected = [run for run in runs if run['arm'] == arm]
        result[arm] = dict(
            seeds=len(selected), optimizer_updates=[run['optimizer_updates'] for run in selected],
            validation_policy_top1=distribution([run['validation']['policy_top1'] for run in selected]),
            test_policy_top1=distribution([run['test']['policy_top1'] for run in selected]),
            test_value_mse=distribution([run['test']['value_mse'] for run in selected]),
            tactical_accuracy=distribution([run['tactical']['policy_top1'] for run in selected]),
            wall_seconds=sum(sum(p['wall_seconds'] for p in run['phases']) +
                             sum(p['wall_seconds'] for p in run['selfplay']) for run in selected),
            cpu_seconds=sum(sum(p['cpu_seconds'] for p in run['phases']) +
                            sum(p['cpu_seconds'] for p in run['selfplay']) for run in selected),
            evaluation_wall_seconds=sum(run['model_only_evaluation_wall_seconds'] for run in selected),
            evaluation_cpu_seconds=sum(run['model_only_evaluation_cpu_seconds'] for run in selected),
            neural_mcts_evaluation_wall_seconds=sum(
                run['neural_mcts_evaluation_wall_seconds'] for run in selected),
            neural_mcts_evaluation_cpu_seconds=sum(
                run['neural_mcts_evaluation_cpu_seconds'] for run in selected),
            original_selfplay_positions=sum(sum(p['original_positions'] for p in run['selfplay'])
                                            for run in selected),
            materialized_selfplay_examples=sum(sum(p['augmented_examples'] for p in run['selfplay'])
                                               for run in selected))
    return result


def aggregate_strength(rows):
    result = {}
    scores = {'wins': 1., 'draws': .5, 'losses': 0.}
    for arm, opponent in sorted({(row['arm'], row['opponent']) for row in rows}):
        selected = [row for row in rows if row['arm'] == arm and row['opponent'] == opponent]
        summary = summarize_games(selected)
        units = [float(np.mean([scores[row['outcome']] for row in selected
                                if row['seed'] == seed]))
                 for seed in sorted({row['seed'] for row in selected})]
        half = math.sqrt(math.log(40)/(2*len(units)))
        mean = float(np.mean(units))
        summary['training_seed_units'] = len(units)
        summary['seed_pair_scores'] = units
        summary['seed_level_score_95_hoeffding'] = [max(0., mean-half), min(1., mean+half)]
        result[f'{arm}/{opponent}'] = summary
    return result


def recommendations(training, exploration, runs):
    by_arm_seed = {(run['arm'], run['seed']): run for run in runs}
    seeds = sorted({run['seed'] for run in runs})
    annealed_test_wins = sum(
        by_arm_seed[('annealed_mix', seed)]['test']['policy_top1'] >
        by_arm_seed[('scratch_selfplay', seed)]['test']['policy_top1'] for seed in seeds)
    annealed_tactical_wins = sum(
        by_arm_seed[('annealed_mix', seed)]['tactical']['policy_top1'] >
        by_arm_seed[('scratch_selfplay', seed)]['tactical']['policy_top1'] for seed in seeds)
    noise = exploration['summary']
    return {
        'root_noise': dict(decision='defer', evidence=(
            f"captured {noise['root_noise']['captures']} vs {noise['baseline_no_noise']['captures']} "
            f"pieces and visited {noise['root_noise']['unique_positions']} vs "
            f"{noise['baseline_no_noise']['unique_positions']} unique positions, but both had zero "
            'decisive games in the paired three-game pilot')),
        'teacher_pretraining': dict(decision='reject as a baseline replacement', evidence=(
            'pretrained-only held-out imitation and model-only strength did not improve consistently '
            'over scratch at this bounded budget')),
        'selfplay_after_pretraining': dict(decision='defer', evidence=(
            'held-out imitation improved over scratch in only two of three seeds and did not transfer '
            'to a fixed-opponent strength gain')),
        'annealed_teacher_mix': dict(decision='defer as promising follow-up', evidence=(
            f'held-out imitation exceeded scratch in {annealed_test_wins}/{len(seeds)} seeds, while '
            f'tactical accuracy improved in only {annealed_tactical_wins}/{len(seeds)} and fixed-opponent '
            'game counts remain too small for a strength claim')),
        'optimizer_persistence': dict(decision='adopt for labelled continuous-learner experiments',
            evidence='all phase checkpoints resumed AdamW moments and exact cumulative update counts; legacy recreate mode remains default'),
        'tactical_hybrid': dict(decision='defer for play-time use', evidence=(
            'the exact two-ply verifier is separately measured and prevents immediate blunders, but its '
            'small deployment sample does not establish aggregate strength')),
        'search_efficiency': dict(decision='defer', evidence='scoped and regression-guarded by issues #37-#41')}


def build_report(root, protocol, dataset_manifest, exploration, experiment, started, snapshot):
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    training = aggregate_runs(experiment['runs'])
    report = dict(
        schema=REPORT_SCHEMA, passed=True, protocol=protocol,
        source=dict(git_revision=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True, cwd=ROOT).strip(),
            snapshot=str(snapshot.relative_to(root)),
            sha256={str(path.relative_to(snapshot)): digest(path)
                    for path in sorted(snapshot.rglob('*.py'))}),
        hardware=hardware(), torch=torch.__version__, numpy=np.__version__,
        dataset=dataset_manifest, exploration=exploration,
        training=training, runs=experiment['runs'], untrained=experiment['untrained_runs'],
        selected=experiment['selected'], reload=experiment['reload'],
        fixed_opponent_sha256=experiment['frozen_checkpoint_sha256'],
        model_only_games=experiment['model_only_games'],
        model_only_strength=aggregate_strength(experiment['model_only_games']),
        neural_mcts_games=experiment['neural_mcts_games'],
        neural_mcts_strength=aggregate_strength(experiment['neural_mcts_games']),
        deployment=experiment['deployment'],
        recommendations=recommendations(training, exploration, experiment['runs']),
        limitations=('Three fixed training seeds and a deliberately short CPU pilot. Game intervals are '
                     'conditional on one opening and fixed opponents; repeated deterministic agents can '
                     'repeat trajectories. No positive result is promoted without consistent seed-level '
                     'held-out and model-only evidence.'),
        official_rules=('Exact corner/stalemate wins; exact threefold and 80-noncapture modelling draws. '
                        'No relabelled draws, adjudication, truncation or reward shaping.'),
        efficiency=dict(teacher_generation_one_time_seconds=dataset_manifest['elapsed_seconds'],
                        teacher_search_seconds=dataset_manifest['teacher_search_seconds'],
                        deployment_evaluation_wall_seconds=experiment['deployment']['wall_seconds'],
                        deployment_evaluation_cpu_seconds=experiment['deployment']['cpu_seconds'],
                        note='One-time dataset cost is not hidden inside per-arm training; both and total run cost are reported.'),
        total_wall_seconds=time.perf_counter()-started,
        process_cpu_seconds=time.process_time(),
        process_peak_rss_bytes=int(peak if sys.platform == 'darwin' else peak*1024))
    report['budget_compliance'] = dict(
        wall_seconds_limit=protocol['budgets']['wall_seconds'],
        wall_seconds_passed=report['total_wall_seconds'] <= protocol['budgets']['wall_seconds'],
        peak_memory_bytes_limit=protocol['budgets']['peak_memory_bytes'],
        peak_memory_passed=report['process_peak_rss_bytes'] <= protocol['budgets']['peak_memory_bytes'])
    require(all((report['budget_compliance']['wall_seconds_passed'],
                 report['budget_compliance']['peak_memory_passed'])), 'Experiment exceeded a declared budget')
    write_json(root / 'report.json', report)
    return report


def verify_run(root):
    root = Path(root)
    protocol = json.loads((root / 'protocol.json').read_text())
    data = load_dataset(root / 'teacher-dataset.pkl.gz')
    manifest = json.loads((root / 'dataset-manifest.json').read_text())
    require(digest(root / 'teacher-dataset.pkl.gz') == manifest['sha256'], 'Dataset hash mismatch')
    report = json.loads((root / 'report.json').read_text())
    require(report['schema'] == REPORT_SCHEMA and report['passed'], 'Incomplete report')
    require(report['protocol'] == protocol, 'Protocol changed')
    require(len(report['runs']) == len(ARMS) * len(protocol['training']['seeds']),
            'Incomplete arm/seed matrix')
    require(len(report['untrained']) == len(protocol['training']['seeds']),
            'Incomplete untrained seed matrix')
    for run in report['runs']:
        require(run['seed'] in protocol['training']['seeds'] and run['arm'] in ARMS,
                'Unexpected run')
        if run['arm'] != 'pretrained_only':
            require(run['optimizer_updates'] == protocol['training']['updates_per_phase'] *
                    len(run['phases']), 'Optimizer continuation count changed')
    checkpoint = root / report['selected']['checkpoint']
    require(digest(checkpoint) == report['selected']['sha256'], 'Selected checkpoint hash mismatch')
    net = reload_net(checkpoint, protocol)
    game = IntransitiveGame()
    state = game.getInitBoard()
    policy, value = net.predict(state, game.getValidMoves(state, 0))
    require(np.isfinite(policy).all() and np.isfinite(value).all(), 'Checkpoint prediction is not finite')
    require(not report['reload']['teacher_required_at_inference'], 'Teacher inference dependency changed')
    return dict(passed=True, runs=len(report['runs']), dataset_examples=sum(
        len(rows) for rows in data['examples'].values()), selected_sha256=digest(checkpoint),
        optimizer_updates=net.optimizer_updates)


def package_run(root, destination):
    root, destination = Path(root), Path(destination)
    include = ['protocol.json', 'teacher-dataset.pkl.gz', 'dataset-manifest.json',
               'exploration.json', 'report.json', 'verification.json',
               'selected.pt', 'requirements.txt',
               'source_backups']
    with tarfile.open(destination, 'w:gz') as archive:
        for name in include:
            archive.add(root / name, arcname=name)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('run', 'verify', 'package'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if args.phase == 'verify':
        print(json.dumps(verify_run(root), indent=2))
        return
    if args.phase == 'package':
        require(args.archive, 'package requires --archive')
        package_run(root, args.archive)
        return
    require(not root.exists(), 'Use a fresh output directory')
    protocol = json.loads(args.protocol.read_text())
    root.mkdir(parents=True)
    write_json(root / 'protocol.json', protocol)
    started = time.perf_counter()
    settings = SimpleNamespace(game='intransitive', checkpoint=str(root), protocol=protocol)
    snapshot = backup_run_sources(settings)
    requirements = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (root / 'requirements.txt').write_text(requirements)
    dataset, manifest = generate_dataset(root / 'teacher-dataset.pkl.gz', protocol)
    exploration = exploration_experiment(protocol, root)
    experiment = run_experiment(root, dataset, protocol)
    build_report(root, protocol, manifest, exploration, experiment, started, snapshot)
    write_json(root / 'verification.json', verify_run(root))


if __name__ == '__main__':
    main()
