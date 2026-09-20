"""Issue #68 bounded measurements; run as a module from the repository root.

Five stages, each writing one JSON evidence file:

    --stage calibrate    margins, from the evaluator's own score distribution
    --stage fixed        per-technique firing and node cost at a fixed depth
    --stage sensitivity  firing rate against agreement, per parameter
    --stage mate         mate-distance pruning against the bounded-proof oracle
    --stage paired       equal-time paired play, colour balanced, Wilson interval

Nothing here tunes a weight. The calibration corpus and the fixed-depth corpus
use disjoint seeds, and the paired stage plays only held-out positions.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np

from ...IntransitiveGame import IntransitiveGame
from ...heuristics import AlphaBetaPlayer, SearchConfig
from ...heuristics import selective
from ...heuristics.budget import Budget
from ...heuristics.evaluation import Evaluator, MATE_THRESHOLD
from ...heuristics.position import SearchPosition
from ...heuristics.search import prove
from ...tests.test_attribution import random_positions

HERE = Path(__file__).resolve().parent

# The scale `selective.supported()` accepts without the experimental opt-in.
SUPPORTED = dict(count_weight=100., advantage_weight=25.,
                 attack_enabled=False, defence_enabled=False, overload_enabled=False,
                 pressure_enabled=False)
# The adopted route genome, which needs the experimental opt-in to prune.
ADOPTED = dict(selective_evaluator_enabled=True)

# Razoring's reference is quiescence, not the all-off baseline: it can only be
# enabled together with quiescence, so the difference between them is its cost.
MODES = {
    'baseline': {},
    'quiescence': dict(quiescence_enabled=True),
    'razoring': dict(quiescence_enabled=True, razoring_enabled=True),
    'reverse_futility': dict(reverse_futility_enabled=True),
    'move_count': dict(move_count_pruning_enabled=True),
    'mate_distance': dict(mate_distance_pruning_enabled=True),
    'all': dict(quiescence_enabled=True, razoring_enabled=True,
                reverse_futility_enabled=True, move_count_pruning_enabled=True,
                mate_distance_pruning_enabled=True),
}
REFERENCE = {name: 'quiescence' if name == 'razoring' else 'baseline' for name in MODES}
COUNTERS = ('razoring_eligible', 'razoring_applied', 'razoring_nodes',
            'reverse_futility_eligible', 'reverse_futility_pruned',
            'move_count_eligible', 'move_count_pruned',
            'mate_distance_eligible', 'mate_distance_pruned',
            'static_evaluations', 'quiescence_captures')


def base_config(plan, scale=None, **overrides):
    """Null windows need PVS; shallow techniques need a depth that reaches them.

    Issue #66 established that without PVS the shared non-PV guard is never
    satisfied, so a protocol without it cannot measure any of this.
    """
    return SearchConfig(max_depth=plan['fixed_depth']['depth'],
                        time_limit=plan['fixed_depth']['seconds_cap'],
                        node_limit=10**9, proof_depth=0, proof_nodes=0,
                        pvs_enabled=True, ordering_enabled=True,
                        compiled_ordering_enabled=True, table_entries=10000,
                        **(SUPPORTED if scale is None else scale), **overrides)


def corpus(spec):
    states = random_positions(games=spec['games'], plies=spec['plies'], seed=spec['seed'])
    return [s for s in states if SearchPosition(s).terminal()[1] == 'ongoing']


def header(plan, stage):
    """Identity is the files that decide behaviour, not the whole-tree diff.

    Stages run at different times while documentation is still being written,
    so a working-tree digest would differ between them for reasons that cannot
    affect a measurement. These hashes are the comparable identity: if they
    match across the evidence files, the stages measured the same search.
    """
    package = HERE.parents[1]
    sources = sorted(package.glob('heuristics/*.py')) + sorted(HERE.glob('*.py')) + [
        HERE / 'plan.json', package / 'tests' / 'test_shallow_pruning.py']
    return dict(issue=68, stage=stage, plan=plan, platform=platform.platform(),
                revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                python=sys.version.split()[0],
                source_sha256={str(path.relative_to(package)):
                               hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sources})


def percentiles(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), min=float(a.min()), max=float(a.max()),
                **{f'p{q}': float(np.percentile(a, q)) for q in (1, 5, 50, 95, 99)})


def calibrate(plan):
    """Express the evaluator's own one-to-three ply swing in allowance units.

    A margin is only meaningful as a multiple of `selective.allowance()`, so
    the measured quantity is the ratio, not a raw evaluator score. Constants
    borrowed from a chess engine would be meaningless against these weights.
    """
    spec = plan['calibration']
    game, states = IntransitiveGame(), corpus(spec['corpus'])
    report = dict(header(plan, 'calibrate'), positions=len(states), scales={})
    for name, scale in (('supported', SUPPORTED), ('adopted', ADOPTED)):
        template = base_config(plan, scale=scale)
        unit = selective.allowance(template)
        swings, branching = {d: [] for d in spec['depths']}, []
        for state in states:
            position = SearchPosition(state)
            branching.append(int(len(position.legal())))
            budget = Budget(10**9, 600.)
            static = Evaluator(game, template).score(
                SearchPosition(state), position.side, budget, proof={'status': 'unknown'})
            if abs(static) > MATE_THRESHOLD:
                continue
            for depth in spec['depths']:
                result = AlphaBetaPlayer(config=replace(template, max_depth=depth)).analyze(state)
                if result.score is None or abs(result.score) > MATE_THRESHOLD:
                    continue
                swings[depth].append((result.score - static) / unit)
        if not all(swings.values()):
            raise ValueError('Calibration corpus produced no non-mate swings')
        report['scales'][name] = dict(
            allowance=unit, branching=percentiles(branching),
            swing={str(d): percentiles(v) for d, v in swings.items()},
            # Razoring claims the value cannot rise above alpha, so it needs the
            # largest rise; reverse futility claims it cannot fall below beta,
            # so it needs the largest fall. Both are per remaining ply.
            required_razoring_margin=max(max(v) / d for d, v in swings.items()),
            required_reverse_margin=max(-min(v) / d for d, v in swings.items()))
    template = base_config(plan)
    report['summary'] = {name: dict(
        allowance=scale['allowance'],
        required_razoring_margin=scale['required_razoring_margin'],
        configured_razoring_margin=template.razoring_margin,
        required_reverse_margin=scale['required_reverse_margin'],
        configured_reverse_futility_margin=template.reverse_futility_margin,
        median_branching=scale['branching']['p50'],
        configured_move_count_base=template.move_count_base)
        for name, scale in report['scales'].items()}
    return report


def fixed_depth(plan):
    spec = plan['fixed_depth']
    states = corpus(spec['corpus'])[:spec['sample']]
    report = dict(header(plan, 'fixed'), positions=len(states), rows=[], summary={})
    for mode, settings in MODES.items():
        config = base_config(plan, **settings)
        for index, state in enumerate(states):
            start = perf_counter()
            result = AlphaBetaPlayer(config=config).analyze(state)
            report['rows'].append(dict(
                mode=mode, index=index,
                state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                action=result.action, score=result.score, nodes=result.nodes,
                work=result.work, completed_depth=result.completed_depth,
                stopped=result.stopped, seconds=perf_counter() - start,
                selective_enabled=result.selective['enabled'],
                effective=result.selective['effective'],
                **{name: result.selective[name] for name in COUNTERS}))
    for mode in MODES:
        rows = [r for r in report['rows'] if r['mode'] == mode]
        reference = [r for r in report['rows'] if r['mode'] == REFERENCE[mode]]
        assert len(rows) == len(reference)
        nodes, base_nodes = sum(r['nodes'] for r in rows), sum(r['nodes'] for r in reference)
        report['summary'][mode] = dict(
            reference=REFERENCE[mode], nodes=nodes, reference_nodes=base_nodes,
            node_ratio=nodes / base_nodes, seconds=sum(r['seconds'] for r in rows),
            reference_seconds=sum(r['seconds'] for r in reference),
            changed_action=sum(a['action'] != b['action'] for a, b in zip(rows, reference)),
            changed_score=sum(a['score'] != b['score'] for a, b in zip(rows, reference)),
            capped=sum(r['stopped'] for r in rows),
            **{name: sum(r[name] for r in rows) for name in COUNTERS})
    return report


SWEEPS = (
    ('razoring', 'razoring_margin', (0.125, 0.25, 0.5, 1.0, 1.5)),
    ('reverse_futility', 'reverse_futility_margin', (0.25, 0.5, 1.0, 2.0)),
    ('move_count', 'move_count_base', (2, 4, 8, 12, 24)),
)


def sensitivity(plan):
    """How each parameter trades firing rate against agreement with its reference.

    A margin large enough to be safe may be large enough never to fire. That is
    a property of this evaluator and this window, not a tuning accident, so it
    is measured rather than asserted.
    """
    spec = plan['sensitivity']
    states = corpus(plan['fixed_depth']['corpus'])[:spec['sample']]
    report = dict(header(plan, 'sensitivity'), positions=len(states), rows=[], summary={})
    references = {}
    for mode in {name for name, _, _ in SWEEPS} | {'baseline', 'quiescence'}:
        if mode in ('baseline', 'quiescence'):
            config = base_config(plan, **MODES[mode])
            references[mode] = [AlphaBetaPlayer(config=config).analyze(s) for s in states]
    for mode, parameter, values in SWEEPS:
        reference = references[REFERENCE[mode]]
        for value in values:
            config = base_config(plan, **MODES[mode], **{parameter: value})
            rows = [AlphaBetaPlayer(config=config).analyze(s) for s in states]
            fired = sum(r.selective[f'{mode}_applied' if mode == 'razoring'
                                    else f'{mode}_pruned'] for r in rows)
            eligible = sum(r.selective[f'{mode}_eligible'] for r in rows)
            report['rows'].append(dict(
                mode=mode, parameter=parameter, value=value,
                nodes=sum(r.nodes for r in rows),
                reference_nodes=sum(r.nodes for r in reference),
                node_ratio=sum(r.nodes for r in rows) / sum(r.nodes for r in reference),
                eligible=eligible, fired=fired,
                changed_action=sum(a.action != b.action for a, b in zip(rows, reference)),
                changed_score=sum(a.score != b.score for a, b in zip(rows, reference))))
    report['summary'] = {f"{r['mode']}@{r['value']}": dict(
        nodes=r['node_ratio'], fired=r['fired'], eligible=r['eligible'],
        changed_action=r['changed_action']) for r in report['rows']}
    return report


def mate_oracle(plan):
    """Mate-distance pruning must change only the cost of finding a mate.

    Two configurations, because they exercise different code. With the bounded
    proof enabled a certified tactic is settled by the leaf oracle at depth
    one, so the narrowing is never reached and the technique is inert. With the
    proof disabled the same mates have to be found by the tree, which is where
    a narrowed window can actually cut. Both must agree with the oracle.
    """
    from ...tests.test_tactics import CASES
    from ...tests.tactical_oracle import load_case
    report = dict(header(plan, 'mate'), rows=[], disagreements=[], summary={})
    configurations = dict(
        proof=dict(proof_depth=2, proof_nodes=64),
        tree=dict(proof_depth=0, proof_nodes=0))
    for kind, proof_settings in configurations.items():
        for case in CASES:
            reference, expected = load_case(case)
            state = reference.storage()
            row = dict(case=case['id'], configuration=kind, expected=int(expected))
            for mode in ('off', 'on'):
                config = replace(base_config(plan), max_depth=4, **proof_settings,
                                 mate_distance_pruning_enabled=mode == 'on')
                result = AlphaBetaPlayer(config=config).analyze(state)
                proof = prove(IntransitiveGame(), SearchPosition(state), config,
                              Budget(10**9, 600.))
                row[mode] = dict(action=result.action, score=result.score,
                                 nodes=result.nodes, completed_depth=result.completed_depth,
                                 stop_reason=result.stop_reason,
                                 proof_status=proof['status'], proof_score=proof.get('score'),
                                 pruned=result.selective['mate_distance_pruned'],
                                 eligible=result.selective['mate_distance_eligible'])
            row['matches_expected'] = row['on']['action'] == int(expected)
            if (row['on']['action'] != row['off']['action']
                    or row['on']['score'] != row['off']['score']
                    or row['on']['proof_status'] != row['off']['proof_status']
                    or row['on']['proof_score'] != row['off']['proof_score']
                    or not row['matches_expected']):
                report['disagreements'].append(row)
            report['rows'].append(row)
    for kind in configurations:
        rows = [r for r in report['rows'] if r['configuration'] == kind]
        report['summary'][kind] = dict(
            cases=len(rows),
            disagreements=sum(r in report['disagreements'] for r in rows),
            expected_action=sum(r['matches_expected'] for r in rows),
            pruned=sum(r['on']['pruned'] for r in rows),
            eligible=sum(r['on']['eligible'] for r in rows),
            nodes_on=sum(r['on']['nodes'] for r in rows),
            nodes_off=sum(r['off']['nodes'] for r in rows),
            node_ratio=sum(r['on']['nodes'] for r in rows) / sum(r['off']['nodes'] for r in rows))
    return report


def wilson(wins, losses):
    total = wins + losses
    if not total:
        return [0., 1.]
    p, z = wins / total, 1.96
    mid = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** .5) / (1 + z * z / total)
    return [mid - half, mid + half]


def paired(plan, output=None):
    """Equal wall clock per move, colours swapped on the same start.

    The #54 harness varies the genome between the two engines and shares one
    search protocol, so it cannot itself pit two search settings against each
    other. Its rules are reproduced here instead: official wins only, a capped
    game is unfinished and never a draw, and both colours are played.
    """
    from ...tournament.spec import generate_positions, unpack
    spec = plan['paired']
    positions = [p for p in generate_positions(seed=6854) if p['pool'] == 'heldout']
    game = IntransitiveGame(modelling_draws=False)
    template = replace(base_config(plan), max_depth=64, time_limit=spec['seconds_per_move'],
                       proof_depth=2, proof_nodes=64)
    report = dict(header(plan, 'paired'), positions=len(positions), games=[], summary={})
    for mode in [m for m in MODES if m not in ('baseline', 'quiescence')]:
        settings = MODES[mode]
        for item in positions:
            for candidate_side in (0, 1):
                state, winner, moves, fallbacks = unpack(item['state']), None, [], 0
                for _ in range(spec['max_plies']):
                    side = int(state[:, :, 82:84].flat[1])
                    if game.getGameEnded(state, side).any():
                        winner = int(np.argmax(game.getGameEnded(state, side)))
                        break
                    live = dict(settings) if side == candidate_side else dict(
                        MODES[REFERENCE[mode]])
                    started = perf_counter()
                    if SearchPosition(state).terminal()[1] == 'ongoing':
                        # Modelling draws live inside the search; official play
                        # continues past them, so a search that declines the
                        # root still owes the match a legal move.
                        result = AlphaBetaPlayer(config=replace(template, **live)).analyze(state)
                        action, depth, nodes = int(result.action), result.completed_depth, result.nodes
                    else:
                        action, depth, nodes = int(SearchPosition(state).raw_legal()[0]), 0, 0
                        fallbacks += 1
                    assert game.getValidMoves(state, side)[action]
                    moves.append(dict(side=side, action=action, depth=depth, nodes=nodes,
                                      seconds=perf_counter() - started))
                    state, _ = game.getNextState(state, side, action)
                report['games'].append(dict(
                    mode=mode, reference=REFERENCE[mode], position=item['sha256'],
                    stage=item['stage'], candidate_side=candidate_side, plies=len(moves),
                    modelling_draw_fallbacks=fallbacks,
                    outcome='unfinished' if winner is None else
                            'win' if winner == candidate_side else 'loss',
                    candidate_depths=[m['depth'] for m in moves if m['side'] == candidate_side],
                    reference_depths=[m['depth'] for m in moves if m['side'] != candidate_side],
                    candidate_seconds=sum(m['seconds'] for m in moves if m['side'] == candidate_side),
                    reference_seconds=sum(m['seconds'] for m in moves if m['side'] != candidate_side)))
                if output is not None:  # an hour of games should survive a crash
                    write(output, report)
    for mode in {row['mode'] for row in report['games']}:
        rows = [r for r in report['games'] if r['mode'] == mode]
        counts = {o: sum(r['outcome'] == o for r in rows) for o in ('win', 'loss', 'unfinished')}
        report['summary'][mode] = dict(
            counts, games=len(rows), reference=rows[0]['reference'],
            decisive_wilson_95=wilson(counts['win'], counts['loss']),
            modelling_draw_fallbacks=sum(r['modelling_draw_fallbacks'] for r in rows),
            candidate_mean_depth=float(np.mean([d for r in rows for d in r['candidate_depths']])),
            reference_mean_depth=float(np.mean([d for r in rows for d in r['reference_depths']])),
            candidate_seconds=sum(r['candidate_seconds'] for r in rows),
            reference_seconds=sum(r['reference_seconds'] for r in rows))
    return report


STAGES = dict(calibrate=calibrate, fixed=fixed_depth, sensitivity=sensitivity,
              mate=mate_oracle, paired=paired)
INCREMENTAL = {'paired'}


def write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False, sort_keys=True) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=sorted(STAGES), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    plan = json.loads((HERE / 'plan.json').read_text())
    start = perf_counter()
    AlphaBetaPlayer(config=base_config(plan))._prepare()
    warmup = perf_counter() - start
    report = (STAGES[args.stage](plan, args.output) if args.stage in INCREMENTAL
              else STAGES[args.stage](plan))
    report['warmup_seconds'] = warmup
    write(args.output, report)
    print(json.dumps(report.get('summary', {}), indent=2)[:4000])


if __name__ == '__main__':
    main()
