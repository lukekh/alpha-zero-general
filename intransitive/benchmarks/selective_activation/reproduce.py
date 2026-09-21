"""Issue #66 activation measurement; run as a module from the repository root.

Every arm searches the same positions with the same genome and differs only in
which selective technique is switched on, so a node count can be attributed to
one technique rather than to the configuration as a whole.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np

from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveSymmetries import transform_state
from intransitive.heuristics import AlphaBetaPlayer, SearchConfig
from intransitive.heuristics.config import TECHNIQUE_COUNTERS, activation
from intransitive.tests.test_heuristics import position

HERE = Path(__file__).resolve().parent
SCOUT = dict(pvs_enabled=True)
VARIABLE = dict(pvs_enabled=True, variable_material_enabled=True)
# Each arm names the arm it is attributed against. PVS is the reference for the
# selective techniques rather than plain alpha-beta, because NMP and futility
# only ever see a null window underneath a scout search: their cost has to be
# compared with the scout searches already being paid for. The two variable
# arms exist only so MVV-LVA has a reference where it can reorder anything at
# all; they are a control for that one question, not a second genome under test.
ARMS = {
    'baseline': ({}, 'baseline'),
    'pvs': (SCOUT, 'baseline'),
    'nmp': (dict(SCOUT, nmp_enabled=True), 'pvs'),
    'futility': (dict(SCOUT, futility_enabled=True), 'pvs'),
    'futility_tuned': (dict(SCOUT, futility_enabled=True, futility_margin=1/16), 'pvs'),
    'lmr': (dict(SCOUT, lmr_enabled=True), 'pvs'),
    'mvv_lva': (dict(SCOUT, mvv_lva_enabled=True), 'pvs'),
    'quiescence': (dict(SCOUT, quiescence_enabled=True), 'pvs'),
    'all': (dict(SCOUT, nmp_enabled=True, futility_enabled=True, futility_margin=1/16,
                 lmr_enabled=True, mvv_lva_enabled=True), 'pvs'),
    'variable_material': (VARIABLE, 'variable_material'),
    'variable_mvv_lva': (dict(VARIABLE, mvv_lva_enabled=True), 'variable_material'),
    # Adaptive reductions are a knob, not a default: they are measured here so
    # the divisors govern code whose effect is on record.
    'lmr_adaptive': (dict(SCOUT, lmr_enabled=True, lmr_depth_divisor=3,
                          lmr_index_divisor=4), 'lmr'),
}
OPTIONS = {name: options for name, (options, _) in ARMS.items()}
REFERENCE = {name: reference for name, (_, reference) in ARMS.items()}


def base_config(plan):
    """The adopted genome, with proofs off so only the heuristic tree is measured."""
    return SearchConfig(max_depth=plan['depths'][0], time_limit=plan['fixed_depth_seconds_cap'],
                        node_limit=10**9, proof_nodes=0, proof_depth=0,
                        selective_evaluator_enabled=True, compiled_ordering_enabled=True)


def fixtures(plan):
    game = IntransitiveGame(modelling_draws=False)
    state, side = game.getInitBoard(), 0
    rng = np.random.default_rng(plan['held_out_seed'])
    for _ in range(24):
        action = int(rng.choice(np.flatnonzero(game.getValidMoves(state, side))))
        state, side = game.getNextState(state, side, action)
    return [('opening', game.getInitBoard()), ('midgame', state),
            ('endgame', position({'D4': 1, 'F6': -2}))]


def configure(base, **options):
    """An arm's config, or None where this revision cannot express it.

    The improvement arms use settings that the pre-#66 revision rejects. Return
    None there instead of failing, so the same script runs at both revisions and
    the before/after comparison is reproducible rather than hand-assembled.
    """
    try:
        return replace(base, **options)
    except (TypeError, ValueError):
        return None


def measure(config, state):
    """One fresh engine per measurement, exactly as the harness plays a move."""
    player = AlphaBetaPlayer(config=config)
    start = perf_counter()
    result = player.analyze(state)
    row = {key: getattr(result, key) for key in
           ('action', 'score', 'completed_depth', 'nodes', 'work', 'tt_hits',
            'stopped', 'stop_reason', 'pvs_probes', 'pvs_researches')}
    return dict(row, seconds=perf_counter() - start, pv=list(result.pv),
                selective={k: v for k, v in result.selective.items() if k != 'identity'},
                ordering=dict(result.ordering), identity=config.identity())


def wilson(wins, losses):
    n = wins + losses
    if not n:
        return [0., 1.]
    p, z = wins / n, 1.96
    mid = (p + z*z/(2*n))/(1 + z*z/n)
    half = z*((p*(1-p)/n + z*z/(4*n*n))**.5)/(1 + z*z/n)
    return [mid - half, mid + half]


def paired_games(config, plan, boards, report):
    """Fixed-depth paired play: identical depth for both sides, cost differs.

    Equal depth, not equal time, is the comparison the issue asks for: the
    question is whether pruning changes the move played, with its cost reported
    separately. Both colours are played from every start.
    """
    game = IntransitiveGame(modelling_draws=False)
    candidate = replace(config, **OPTIONS['all'])
    reference = replace(config, **OPTIONS['baseline'])
    for stage, start in boards:
        for candidate_side in (0, 1):
            state, moves, winner = start.copy(), [], None
            for _ in range(plan['paired_plies_cap']):
                side = int(state[:, :, 82:84].flat[1])
                terminal = game.getGameEnded(state, side)
                if terminal.any():
                    winner = int(np.argmax(terminal))
                    break
                row = measure(candidate if side == candidate_side else reference, state)
                moves.append(dict(side=side, action=row['action'], nodes=row['nodes'],
                                  seconds=row['seconds'], depth=row['completed_depth']))
                state, _ = game.getNextState(state, side, row['action'])
            terminal = game.getGameEnded(state, int(state[:, :, 82:84].flat[1]))
            if terminal.any():
                winner = int(np.argmax(terminal))
            report['paired'].append(dict(stage=stage, candidate_side=candidate_side,
                outcome='unfinished' if winner is None else
                        'win' if winner == candidate_side else 'loss', moves=moves))
            yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-paired', action='store_true',
                        help='Fixed-depth attribution only, without the paired games')
    args = parser.parse_args()
    plan = json.loads((HERE / 'plan.json').read_text())
    if list(OPTIONS) != plan['arms']:
        raise ValueError('The predeclared plan and the implemented arms disagree')
    config = base_config(plan)
    report = dict(plan=plan, platform=platform.platform(),
                  revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_diff_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'])).hexdigest(),
                  base_config=config.to_dict(),
                  arms={name: dict(options=dict(options), reference=REFERENCE[name])
                        for name, options in OPTIONS.items()},
                  preconditions={}, fixed=[], margin_sweep=[], paired=[])

    def save():
        args.output.write_text(json.dumps(report, indent=2, default=float) + '\n')

    for name, options in OPTIONS.items():
        arm = configure(config, **options)
        report['preconditions'][name] = ({technique: row['blockers'] for technique, row
                                          in activation(arm).items() if row['enabled']}
                                         if arm else None)
    # Compile every kernel any arm will use before the first timed measurement,
    # including the variable-material ones the control arms need.
    start = perf_counter()
    for name in ('all', 'quiescence', 'variable_mvv_lva'):
        AlphaBetaPlayer(config=replace(config, **OPTIONS[name]))._prepare()
    report['python_warmup_seconds'] = perf_counter() - start
    save()

    boards = [(stage, transform_state(state, symmetry)) for stage, state in fixtures(plan)
              for symmetry in plan['symmetries']]
    for depth in plan['depths']:
        for index, (stage, board) in enumerate(boards):
            if depth != plan['depths'][0] and (stage not in plan['deep_stages']
                                               or index % len(plan['symmetries'])):
                continue
            for name, options in OPTIONS.items():
                arm = configure(config, max_depth=depth, **options)
                entry = dict(arm=name, stage=stage, depth=depth,
                             state_sha256=hashlib.sha256(board.tobytes()).hexdigest())
                report['fixed'].append(dict(entry, result=measure(arm, board)) if arm
                                       else dict(entry, unavailable='not expressible at this revision'))
                save()
    for margin in plan['margin_sweep']:
        for stage, board in boards:
            arm = configure(config, pvs_enabled=True, futility_enabled=True, futility_margin=margin)
            if arm is None:
                continue
            report['margin_sweep'].append(dict(margin=margin, stage=stage,
                state_sha256=hashlib.sha256(board.tobytes()).hexdigest(),
                result=measure(arm, board)))
            save()
    if not args.skip_paired:
        for _ in paired_games(replace(config, max_depth=plan['paired_depth']), plan,
                              fixtures(plan), report):
            save()
    summarize(report)
    save()


def summarize(report):
    """Attribute node counts per technique and record every firing counter."""
    results = {}
    for row in report['fixed']:
        if 'result' in row:
            results[row['arm'], row['stage'], row['depth'], row['state_sha256']] = row['result']
    attribution = {}
    for row in report['fixed']:
        if 'result' not in row:
            continue
        # Every arm is compared with the one arm it differs from by a single
        # switch, so a node count belongs to that switch and not to the regime.
        reference = report['arms'][row['arm']]['reference']
        against = results[reference, row['stage'], row['depth'], row['state_sha256']]
        entry = attribution.setdefault((row['arm'], row['depth']),
                                       dict(arm=row['arm'], depth=row['depth'], reference=reference,
                                            positions=0, nodes=0, reference_nodes=0, seconds=0.,
                                            reference_seconds=0., agreed_action=0, agreed_score=0,
                                            counters=dict.fromkeys(sorted(set(TECHNIQUE_COUNTERS.values())
                                                | {'nmp_cutoffs', 'futility_pruned', 'lmr_researches',
                                                   'static_evaluations', 'null_nodes',
                                                   'verification_nodes'}), 0)))
        entry['positions'] += 1
        entry['nodes'] += row['result']['nodes']
        entry['seconds'] += row['result']['seconds']
        entry['reference_nodes'] += against['nodes']
        entry['reference_seconds'] += against['seconds']
        entry['agreed_action'] += row['result']['action'] == against['action']
        entry['agreed_score'] += row['result']['score'] == against['score']
        for name in entry['counters']:
            source = row['result']['ordering'] if name.startswith('mvv_lva') else row['result']['selective']
            entry['counters'][name] += int(source.get(name, 0))
    for entry in attribution.values():
        entry['node_reduction'] = 1 - entry['nodes'] / entry['reference_nodes'] if entry['reference_nodes'] else None
        entry['seconds_ratio'] = entry['reference_seconds'] / entry['seconds'] if entry['seconds'] else None
        entry['fired'] = sorted(name for name, counter in TECHNIQUE_COUNTERS.items()
                                if entry['counters'][counter])
    order = list(report['arms'])
    report['attribution'] = sorted(attribution.values(), key=lambda e: (e['depth'], order.index(e['arm'])))
    sweep = {}
    for row in report['margin_sweep']:
        entry = sweep.setdefault(row['margin'], dict(margin=row['margin'], positions=0, nodes=0,
                                                     futility_eligible=0, futility_pruned=0))
        entry['positions'] += 1
        entry['nodes'] += row['result']['nodes']
        for name in ('futility_eligible', 'futility_pruned'):
            entry[name] += row['result']['selective'][name]
    report['margin_summary'] = sorted(sweep.values(), key=lambda e: -e['margin'])
    counts = {outcome: sum(r['outcome'] == outcome for r in report['paired'])
              for outcome in ('win', 'loss', 'unfinished')}
    report['paired_summary'] = dict(counts, games=len(report['paired']),
                                    decisive_wilson_95=wilson(counts['win'], counts['loss']))
    report['firing'] = {technique: sum(e['counters'][counter] for e in report['attribution'])
                        for technique, counter in TECHNIQUE_COUNTERS.items()}


if __name__ == '__main__':
    main()
