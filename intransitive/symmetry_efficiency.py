"""Controlled, update-matched augmentation experiment for issue #16.

Run from the repository root; see benchmarks/symmetry/README.md.
"""

# Import baseline first: it disables native ORT telemetry before initialization.
from . import baseline

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .benchmark_training import MeasuredNet
from .smoke import digest, require


# Fixed before measurements, from #15's four observed optimizer-update counts.
UPDATES = (60, 105, 101, 107)
EARLIER_SHA256 = '40b8fc6eecc606e4cfaaa45b0af23cf7f5d1fc89d2ad21cfb27aa80dd0b50e14'
ARMS = ('on', 'off')


class IdentityGame(baseline.BaselineGame):
    """A benchmark-only identity hook; avoid computing discarded symmetries."""

    def getSymmetries(self, board, pi, valid_actions):
        state = self.getCanonicalForm(board, 0).copy()
        policy = np.array(pi, dtype=np.float32, copy=True)
        mask = np.array(valid_actions, dtype=np.bool_, copy=True)
        require(policy.shape == mask.shape == (self.getActionSize(),), 'Invalid example')
        self.metrics.legal.append(int(np.count_nonzero(mask)))
        self.metrics.symmetries[1] += 1
        return [(state, policy, mask)]


def parameter_digest(net):
    result = hashlib.sha256()
    for name, value in sorted(net.nnet.named_parameters()):
        result.update(name.encode())
        result.update(value.detach().cpu().numpy().tobytes())
    return result.hexdigest()


class MatchedNet(MeasuredNet):
    def train(self, examples, *args, **kwargs):
        iteration = len(self.training) + 1
        self.args['batches_per_epoch'] = UPDATES[iteration - 1]
        # Independent of replay size, Python's shuffle consumption and eval calls.
        seed = int(np.random.SeedSequence([150, iteration, 16]).generate_state(1)[0])
        baseline.seed_all(seed)
        initial = parameter_digest(self)
        super().train(examples, *args, **kwargs)
        final = parameter_digest(self)
        require(initial != final, 'Training did not change parameters')
        require(self.training[-1]['updates'] == UPDATES[iteration - 1], 'Unmatched compute')
        self.training[-1].update(iteration=iteration, optimizer_seed=seed,
            parameter_sha256_before=initial, parameter_sha256_after=final)


def budget_for(arm):
    require(arm in ARMS, 'Unknown augmentation arm')
    budget = json.loads(baseline.BUDGET.read_text())
    budget['purpose'] = 'Issue #16 paired augmentation experiment; equal updates per iteration'
    budget['settings']['symmetry_count'] = 12 if arm == 'on' else 1
    # Equal capacity in original positions, preserving both complete histories.
    budget['settings']['maxlenOfQueue'] = 8 * 1600 * budget['settings']['symmetry_count']
    budget['comparison'] = dict(augmentation=arm, updates_per_iteration=UPDATES,
        optimizer_seed='SeedSequence([150, iteration_1_based, 16])',
        fixed_earlier_sha256=EARLIER_SHA256,
        selection='Evaluate every candidate, including rejected ones; no selection by opponent scores',
        evaluation_games_per_arm=192, training_seed_replicates=1)
    budget['budgets'].update(post_training_games=192, evaluation_wall_seconds=1200,
        total_wall_seconds=1800, examples_per_history=budget['settings']['maxlenOfQueue'],
        total_retained_examples=2 * budget['settings']['maxlenOfQueue'])
    # #14's byte estimate was for augmented replay; actual memory is measured here.
    budget['replay_derivation']['configured_examples_per_history'] = budget['settings']['maxlenOfQueue']
    return budget


def score_difference(on, off):
    """Pair by opponent/seed/colour, retaining draws as scores of one half."""
    require(len(on) == len(off) and len(on) > 0, 'Missing paired games')
    scores = dict(wins=1., draws=.5, losses=0.)
    differences = []
    for a, b in zip(on, off):
        for key in ('opponent', 'index', 'model_colour', 'seeds', 'physical_first_player'):
            require(a[key] == b[key], 'Evaluation protocols differ')
        differences.append(scores[a['outcome']] - scores[b['outcome']])
    mean = float(np.mean(differences))
    # Independent game pairs, each difference bounded in [-1, 1].
    half = float(np.sqrt(2 * np.log(40) / len(differences)))
    return dict(games=len(differences), on_minus_off=mean,
        difference_95_hoeffding=[max(-1., mean-half), min(1., mean+half)])


def build_report(root):
    training = {arm: json.loads((root / arm / 'training.json').read_text()) for arm in ARMS}
    settings = {a: copy.deepcopy(t['settings']) for a, t in training.items()}
    for values in settings.values():
        for key in ('checkpoint', 'symmetry_count', 'maxlenOfQueue'):
            values.pop(key)
    require(settings['on'] == settings['off'], 'Uncontrolled settings difference')
    require(training['on']['format'] == training['off']['format'], 'Model formats differ')
    require(training['on']['source_sha256'] == training['off']['source_sha256'], 'Sources differ')
    require(training['on']['requirements'] == training['off']['requirements'], 'Environments differ')
    for arm, report in training.items():
        require(report['passed'] and report['process_exit_code'] == 0, 'Incomplete training')
        require([r['updates'] for r in report['training']] == list(UPDATES), 'Update mismatch')
        require(report['artifacts']['earlier.pt']['sha256'] == EARLIER_SHA256, 'Opponent changed')
        require(report['symmetry_counts'] == {str(12 if arm == 'on' else 1): report['original_positions']},
                'Incomplete per-position symmetry coverage')
    require(training['on']['training'][0]['parameter_sha256_before'] ==
            training['off']['training'][0]['parameter_sha256_before'], 'Initial weights differ')
    traces = {a: [g['trajectory_sha256'] for g in t['games'] if g['phase'] == 'self_play']
              for a, t in training.items()}
    # Record whether closed-loop self-play diverged; never assume equal data from equal seeds.
    same_positions = traces['on'] == traces['off']
    same_examples = training['on']['original_replay'] == training['off']['original_replay']
    points = []
    for iteration in range(1, 5):
        evaluations = {a: json.loads((root / a / f'evaluation-{iteration}' / 'evaluation.json').read_text())
                       for a in ARMS}
        point = dict(iteration=iteration, arms={}, paired_differences={})
        for arm, evaluation in evaluations.items():
            t = training[arm]
            require(evaluation['passed'] and evaluation['process_exit_code'] == 0, 'Incomplete evaluation')
            require(evaluation['checkpoint_sha256']['earlier.pt'] == EARLIER_SHA256, 'Opponent changed')
            require(evaluation['checkpoint_sha256'][f'candidate_{iteration}.pt'] ==
                    t['artifacts'][f'candidate_{iteration}.pt']['sha256'], 'Candidate changed')
            require(len(evaluation['games']) == 48, 'Incomplete evaluation game budget')
            for comparison in evaluation['comparisons'].values():
                require(all(comparison['by_colour'][c]['games'] == 8 for c in ('Blue', 'Red')),
                        'Unbalanced colours')
            point['arms'][arm] = dict(t['arena'][iteration - 1],
                optimization=t['training'][iteration - 1], strength=evaluation['comparisons'],
                evaluation_process_seconds=evaluation['child_process_seconds'])
        for name in ('random', 'greedy', 'earlier_checkpoint'):
            point['paired_differences'][name] = score_difference(
                *[[g for g in evaluations[a]['games'] if g['opponent'] == name] for a in ARMS])
        points.append(point)
    result = dict(passed=True, protocol=training['on']['budget']['comparison'],
        identical_self_play_trajectories=same_positions,
        identical_original_training_examples=same_examples, points=points,
        totals={a: dict(original_positions=t['original_positions'],
            materialized_examples=t['augmented_examples'], optimizer_updates=sum(UPDATES),
            sampled_examples=sum(UPDATES)*64, training_process_seconds=t['child_process_seconds'],
            process_cpu_seconds=t['process_cpu_seconds'], peak_rss_bytes=t['process_peak_rss_bytes'],
            phases=t['phases'], replay=t['replay'], selection=t['selection_reason'],
            evaluation_process_seconds=sum(p['arms'][a]['evaluation_process_seconds'] for p in points))
                for a, t in training.items()},
        limits='One paired training seed; one opening; fixed earlier opponent. Marginal conditional game intervals, not training-seed uncertainty. Repeated checkpoint evaluations share seeds: do not pool them as independent replicates. Equal optimizer updates do not imply equal wall time; self-play/arena simulations and CPU time are reported separately.')
    baseline.write_json(root / 'comparison.json', result)
    return result


def supervised(arguments, report, timeout):
    started = time.perf_counter()
    command = [sys.executable, '-m', 'intransitive.symmetry_efficiency', *arguments, '--child']
    try:
        result = subprocess.run(command, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f'Exceeded {timeout}s; incomplete, not a modelling draw; inspect progress JSON')
    require(result.returncode == 0, f'Child failed with exit code {result.returncode}')
    data = json.loads(report.read_text())
    data.update(child_process_seconds=time.perf_counter()-started, process_exit_code=0,
                hard_timeout_seconds=timeout)
    baseline.write_json(report, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('run', 'train', 'evaluate', 'verify', 'report'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--baseline-folder', help='Extracted issue #15 artifact directory')
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--model-folder')
    parser.add_argument('--iteration', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    cli = parser.parse_args()
    root = Path(cli.output).resolve()
    if cli.phase == 'report':
        build_report(root)
        return
    require(not root.exists(), 'Use a fresh output directory')
    if cli.phase in ('run', 'train'):
        require(cli.baseline_folder, '--baseline-folder is required')
        earlier = Path(cli.baseline_folder).resolve() / 'earlier.pt'
        require(digest(earlier) == EARLIER_SHA256, 'Use the retained issue #15 earlier opponent')
    if cli.phase == 'run':
        root.mkdir(parents=True)
        baseline.write_json(root / 'protocol.json', {a: budget_for(a) for a in ARMS})
        for arm in ARMS:
            supervised(['train', '--output', str(root / arm), '--arm', arm,
                '--baseline-folder', cli.baseline_folder], root / arm / 'training.json', 600)
        # Alternate evaluation order, with isolated serial processes to avoid contention.
        for iteration in range(1, 5):
            for arm in (ARMS[::-1] if iteration % 2 else ARMS):
                folder = root / arm / f'evaluation-{iteration}'
                supervised(['evaluate', '--output', str(folder), '--model-folder', str(root / arm),
                    '--iteration', str(iteration)], folder / 'evaluation.json', 300)
        for arm in ARMS:
            folder = root / arm / 'verification'
            supervised(['verify', '--output', str(folder), '--model-folder', str(root / arm)],
                       folder / 'evaluation.json', 300)
        build_report(root)
    elif not cli.child:
        supervised(sys.argv[1:], root / ('training.json' if cli.phase == 'train' else 'evaluation.json'),
                   600 if cli.phase == 'train' else 300)
    elif cli.phase == 'train':
        require(cli.arm, '--arm is required')
        baseline.train(root, budget_for(cli.arm),
            game_class=baseline.BaselineGame if cli.arm == 'on' else IdentityGame,
            net_class=MatchedNet, earlier_checkpoint=earlier)
    else:
        require(cli.model_folder, '--model-folder is required')
        model_folder = Path(cli.model_folder).resolve()
        baseline.evaluate(root, model_folder, representative=cli.phase == 'verify',
            checkpoint=f'candidate_{cli.iteration}.pt',
            expected_report=model_folder / f'evaluation-{cli.iteration}' / 'evaluation.json')


if __name__ == '__main__':
    main()
