"""Bounded measurements for issue #70; run as a module from the repository root.

Five sections, each writable on its own: the probe's own cost, fixed-depth node
and work totals, the trigger sweep over `certificate_cutoff_min_depth`, an
oracle check of every cutoff-derived result, and paired equal-time games.

    python -m intransitive.benchmarks.certificate.reproduce --output report.json
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

from intransitive.IntransitiveConstants import NO_CAPTURE_LIMIT
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveSymmetries import transform_state
from intransitive.heuristics import AlphaBetaPlayer, SearchConfig
from intransitive.heuristics.clear_run import NO_RUN, NO_SIDE, certify, race_gate
from intransitive.heuristics.evaluation import MATE, MATE_THRESHOLD
from intransitive.tests.test_attribution import random_positions
from intransitive.tests.test_certificate_bound import sparse_positions
from intransitive.tests.test_heuristics import position

HERE = Path(__file__).resolve().parent
SECTIONS = ('probe', 'fixed', 'trigger', 'oracle', 'paired')

# One engine for every mode, so the only difference between two rows is the
# option named by the mode. Ordering, PVS and aspiration are on because that is
# what a competitive search runs with, and the proof depth is the default.
ENGINE = dict(max_depth=6, time_limit=8., node_limit=10**9, proof_depth=2, proof_nodes=64,
              ordering_enabled=True, compiled_ordering_enabled=True,
              pvs_enabled=True, aspiration_enabled=True, table_entries=50000)

# Every option is measured against the same search with that option removed and
# nothing else changed. `guard` needs narrow windows for null-move and futility
# to be reachable at all, so it keeps PVS; `race` turns PVS off, because LMR is
# only consulted where PVS is not already probing a child.
GUARD = dict(certificate_enabled=True, certificate_guard_enabled=True,
             nmp_enabled=True, futility_enabled=True, lmr_enabled=True,
             selective_evaluator_enabled=True)
RACE = dict(lmr_enabled=True, race_reduction_enabled=True, pvs_enabled=False)
PAIRS = {
    'leaf': (dict(certificate_enabled=True), {}),
    'cutoff': (dict(certificate_enabled=True, certificate_cutoff_enabled=True),
               dict(certificate_enabled=True)),
    'guard': (GUARD, {k: v for k, v in GUARD.items() if k != 'certificate_guard_enabled'}),
    'race': (RACE, {k: v for k, v in RACE.items() if k != 'race_reduction_enabled'}),
}
CUTOFF = PAIRS['cutoff'][0]


def wilson(wins, losses):
    total = wins + losses
    if not total:
        return [0., 1.]
    proportion, z = wins / total, 1.96
    middle = (proportion + z * z / (2 * total)) / (1 + z * z / total)
    half = z * ((proportion * (1 - proportion) / total
                 + z * z / (4 * total * total)) ** .5) / (1 + z * z / total)
    return [middle - half, middle + half]


def clock_left(state):
    meta = state[:, :, 82:84].ravel()
    return max(0, NO_CAPTURE_LIMIT - max(int(meta[3]), max(0, int(meta[4]) - 1)))


def fixtures(game):
    """Four stages: the opening, seeded play, a sparse endgame and a certified race."""
    state, side = game.getInitBoard(), 0
    generator = np.random.default_rng(600917)
    for _ in range(24):
        action = int(generator.choice(np.flatnonzero(game.getValidMoves(state, side))))
        state, side = game.getNextState(state, side, action)
    return [('opening', game.getInitBoard()), ('midgame', state),
            ('sparse', position({'D4': 1, 'F6': -2, 'B7': 3, 'H3': -1})),
            # Blue holds a certified seven-ply run that no board guard sees.
            ('certified', position({'E5': 1, 'B2': 1, 'B8': 1, 'H2': 1,
                                    'A5': -1, 'E1': -1, 'A7': -1, 'G1': -1}))]


def paired_starts(game, count, seed):
    """Distinct reachable starts, so no two games are the same trajectory."""
    generator, starts, seen = np.random.default_rng(seed), [], set()
    while len(starts) < count:
        state, side = game.getInitBoard(), 0
        for _ in range(int(generator.integers(6, 26))):
            legal = np.flatnonzero(game.getValidMoves(state, side))
            if not legal.size or game.getGameEnded(state, side).any():
                break
            state, side = game.getNextState(state, side, int(generator.choice(legal)))
        if game.getGameEnded(state, side).any():
            continue
        digest = hashlib.sha256(state[:, :, 0].tobytes() + bytes((side,))).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        starts.append((digest, state))
    return starts


def search(game, state, settings, *, depth=None, seconds=None):
    config = SearchConfig(**dict(ENGINE, **settings))
    if depth is not None:
        config = replace(config, max_depth=depth)
    if seconds is not None:
        config = replace(config, time_limit=seconds)
    result = AlphaBetaPlayer(game, config).analyze(state)
    return dict(action=result.action, score=result.score, nodes=result.nodes,
                work=result.work, completed_depth=result.completed_depth,
                stopped=result.stopped, stop_reason=result.stop_reason,
                elapsed=result.elapsed, tt_hits=result.tt_hits,
                certificate=dict(result.certificate),
                lmr_reduced=result.selective.get('lmr_reduced', 0),
                futility_pruned=result.selective.get('futility_pruned', 0),
                static_evaluations=result.selective.get('static_evaluations', 0))


def probe_section(game, plan):
    """What one probe costs, and how often the gate lets one through."""
    corpus = plan['probe_corpus']
    reachable = random_positions(games=corpus['reachable_games'],
                                 plies=corpus['reachable_plies'], seed=corpus['reachable_seed'])
    sparse = sparse_positions(corpus['sparse_positions'], corpus['sparse_seed'], game)
    rows = {}
    for name, states in (('reachable', reachable), ('sparse', sparse)):
        passes = certificates = hidden = 0
        for state in states:
            meta = state[:, :, 82:84].ravel()
            turn, a1 = int(meta[1]), int(meta[2])
            gate = int(race_gate(state[:, :, 0], turn, a1, clock_left(state), 20))
            passes += gate != NO_SIDE
            for side in (0, 1):
                if certify(state[:, :, 0], side, turn, a1, clock_left(state), 20) != NO_RUN:
                    certificates += 1
                    hidden += gate != side
        rows[name] = dict(positions=len(states), gate_passes=passes,
                          gate_pass_rate=passes / len(states),
                          certificates=certificates, hidden_certificates=hidden)
    # Per-call cost on a representative board, after warming both kernels.
    board = reachable[len(reachable) // 2][:, :, 0].copy()
    pieces = int(np.count_nonzero(board))
    race_gate(board, 0, 0, 80, 20)
    certify(board, 0, 0, 0, 80, 20)
    timings = {}
    for label, call in (('race_gate', lambda: race_gate(board, 0, 0, 80, 20)),
                        ('certify', lambda: certify(board, 0, 0, 0, 80, 20))):
        start = perf_counter()
        for _ in range(20000):
            call()
        timings[label + '_microseconds'] = (perf_counter() - start) / 20000 * 1e6
    gated = timings['race_gate_microseconds']
    full = timings['certify_microseconds']
    rate = rows['reachable']['gate_pass_rate']
    return dict(corpora=rows, pieces=pieces, **timings,
                ungated_microseconds=2 * full, gated_microseconds=gated + rate * full,
                speedup=(2 * full) / (gated + rate * full))


def fixed_section(game, plan):
    rows = []
    for stage, start in fixtures(game):
        for symmetry in (0, plan['held_out_symmetry']):
            state = transform_state(start, symmetry)
            for depth in plan['depths']:
                for option, (candidate, reference) in PAIRS.items():
                    for name, settings in (('on', candidate), ('off', reference)):
                        row = search(game, state, settings, depth=depth,
                                     seconds=plan['fixed_depth_seconds_cap'])
                        rows.append(dict(stage=stage, symmetry=symmetry, depth=depth,
                                         option=option, arm=name, **row))
    return rows


def trigger_section(game, plan):
    """Does paying for probes lower down pay for itself?"""
    rows = []
    for stage, start in fixtures(game):
        for depth in plan['depths']:
            reference = search(game, start, PAIRS['cutoff'][1], depth=depth,
                               seconds=plan['fixed_depth_seconds_cap'])
            for least in plan['cutoff_min_depths']:
                row = search(game, start, dict(CUTOFF,
                                               certificate_cutoff_min_depth=least),
                             depth=depth, seconds=plan['fixed_depth_seconds_cap'])
                rows.append(dict(stage=stage, depth=depth, cutoff_min_depth=least,
                                 reference_nodes=reference['nodes'],
                                 reference_work=reference['work'],
                                 reference_score=reference['score'],
                                 reference_elapsed=reference['elapsed'], **row))
    return rows


def oracle_section(game, plan):
    """Every cutoff-derived mate score, re-searched full width to its distance."""
    settings = dict(CUTOFF, certificate_cutoff_min_depth=1,
                    attack_enabled=False, defence_enabled=False,
                    proof_depth=0, proof_nodes=0)
    checked, confirmed, refuted, rows = 0, 0, 0, []
    for state in sparse_positions(plan['probe_corpus']['sparse_positions'],
                                  plan['probe_corpus']['sparse_seed'], game):
        result = search(game, state, settings, depth=3)
        if (not result['certificate']['cutoffs'] or result['score'] is None
                or abs(result['score']) <= MATE_THRESHOLD):
            continue
        plies = int(MATE - abs(result['score']))
        if plies > plan['oracle']['maximum_verified_plies']:
            continue
        reference = search(game, state, dict(settings, certificate_cutoff_enabled=False,
                                             certificate_enabled=False), depth=plies)
        if reference['stopped'] or reference['score'] is None:
            continue
        checked += 1
        signed = reference['score'] * (1. if result['score'] > 0 else -1.)
        agrees = signed > MATE_THRESHOLD and MATE - abs(reference['score']) <= plies
        confirmed += agrees
        refuted += not agrees
        if not agrees:
            rows.append(dict(claimed=result['score'], reference=reference['score'],
                             plies=plies,
                             state_sha256=hashlib.sha256(state.tobytes()).hexdigest()))
    return dict(checked=checked, confirmed=confirmed, refuted=refuted, refutations=rows)


def play(game, start, candidate, baseline, candidate_side, plan):
    state, winner, moves = start.copy(), None, []
    for _ in range(plan['paired_plies_cap']):
        side = int(state[:, :, 82:84].flat[1])
        terminal = game.getGameEnded(state, side)
        if terminal.any():
            winner = int(np.argmax(terminal))
            break
        settings = candidate if side == candidate_side else baseline
        row = search(game, state, dict(settings, max_depth=20),
                     seconds=plan['paired_seconds_per_move'])
        action = row['action']
        if action is None or not game.getValidMoves(state, side)[action]:
            raise ValueError('The engine returned a move the rules reject')
        moves.append(dict(side=side, action=int(action), depth=row['completed_depth'],
                          nodes=row['nodes'], work=row['work'],
                          cutoffs=row['certificate']['cutoffs']))
        state, _ = game.getNextState(state, side, action)
    terminal = game.getGameEnded(state, int(state[:, :, 82:84].flat[1]))
    if terminal.any():
        winner = int(np.argmax(terminal))
    return winner, moves


def paired_section(game, plan):
    starts = paired_starts(game, plan['paired_starts'], plan['paired_start_seed'])
    rows = []
    for mode, (candidate, baseline) in PAIRS.items():
        for digest, start in starts:
            for candidate_side in (0, 1):
                winner, moves = play(game, start, candidate, baseline, candidate_side, plan)
                rows.append(dict(
                    mode=mode, start_sha256=digest, candidate_side=candidate_side,
                    outcome=('unfinished' if winner is None else
                             'win' if winner == candidate_side else
                             'loss' if winner == 1 - candidate_side else 'draw'),
                    plies=len(moves),
                    candidate_nodes=sum(m['nodes'] for m in moves if m['side'] == candidate_side),
                    baseline_nodes=sum(m['nodes'] for m in moves if m['side'] != candidate_side),
                    candidate_depth=[m['depth'] for m in moves if m['side'] == candidate_side],
                    baseline_depth=[m['depth'] for m in moves if m['side'] != candidate_side],
                    cutoffs=sum(m['cutoffs'] for m in moves if m['side'] == candidate_side)))
    summary = {}
    for mode in PAIRS:
        subset = [row for row in rows if row['mode'] == mode]
        counts = {outcome: sum(row['outcome'] == outcome for row in subset)
                  for outcome in ('win', 'loss', 'draw', 'unfinished')}
        summary[mode] = dict(
            counts, games=len(subset),
            decisive_wilson_95=wilson(counts['win'], counts['loss']),
            median_candidate_depth=float(np.median(
                [d for row in subset for d in row['candidate_depth']] or [0])),
            median_baseline_depth=float(np.median(
                [d for row in subset for d in row['baseline_depth']] or [0])))
    return dict(starts=[digest for digest, _ in starts], games=rows, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sections', nargs='+', choices=SECTIONS, default=list(SECTIONS))
    parser.add_argument('--extended', action='store_true',
                        help="Apply the plan's `extended` overrides, which add power to the "
                             'oracle and paired sections without changing what they compare')
    parser.add_argument('--archived-source', help='Exact git archive revision, when outside a checkout')
    args = parser.parse_args()
    plan = json.loads((HERE / 'plan.json').read_text())
    if args.extended:
        extended = plan['extended']
        plan['oracle'] = dict(plan['oracle'], **extended['oracle'])
        plan['probe_corpus'] = dict(plan['probe_corpus'],
                                    sparse_positions=extended['oracle']['sparse_positions'])
        for name in ('paired_seconds_per_move', 'paired_plies_cap', 'paired_starts'):
            plan[name] = extended[name]
    game = IntransitiveGame()
    report = dict(
        plan=plan, backend='python', platform=platform.platform(),
        revision=args.archived_source or subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_diff_sha256=None if args.archived_source else hashlib.sha256(
            subprocess.check_output(['git', 'diff', 'HEAD'])).hexdigest(),
        engine=ENGINE, pairs={k: dict(on=v[0], off=v[1]) for k, v in PAIRS.items()},
        sections=args.sections, extended=args.extended, effective_plan=plan)
    start = perf_counter()
    warm = dict(ENGINE)
    for candidate, _ in PAIRS.values():
        warm.update(candidate)
    AlphaBetaPlayer(config=SearchConfig(**warm))._prepare()
    report['python_warmup_seconds'] = perf_counter() - start

    def publish():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    publish()
    for name, section in (('probe', probe_section), ('fixed', fixed_section),
                          ('trigger', trigger_section), ('oracle', oracle_section),
                          ('paired', paired_section)):
        if name not in args.sections:
            continue
        started = perf_counter()
        report[name] = section(game, plan)
        report.setdefault('section_seconds', {})[name] = perf_counter() - started
        publish()


if __name__ == '__main__':
    main()
