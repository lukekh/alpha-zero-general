"""Issue #65 YBWC measurements; run as a module from the repository root.

Four stages, each writing one JSON record. `baseline` is run against the
pre-change binary and frozen; `equivalence` replays it with the current binary
at one thread and diffs; `overhead` measures what parallel search costs in
nodes and buys in depth; `paired` plays the equal-time matches the decision
rests on.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np

from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveLogicNumba import search_observation
from intransitive.rust_teacher import RustTeacher, BINARY
from intransitive.tournament.spec import generate_positions, unpack

HERE = Path(__file__).resolve().parent
STAGES = ('opening', 'midgame', 'endgame')


def wilson(wins, losses):
    """Same interval #54's report uses, on decisive games only."""
    n = wins + losses
    if not n:
        return [0., 1.]
    p, z = wins / n, 1.96
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** .5) / (1 + z * z / n)
    return [mid - half, mid + half]


CORPUS = {}


def starts(per_stage):
    """#54's generated openings, taken in state-hash order for stability."""
    if per_stage in CORPUS:
        return CORPUS[per_stage]
    corpus = generate_positions(seed=65, lines=9)
    chosen = [p for p in corpus if p['stage'] == 'official']
    for stage in STAGES:
        rows = sorted((p for p in corpus if p['stage'] == stage), key=lambda p: p['sha256'])
        chosen += rows[:per_stage]
    CORPUS[per_stage] = [dict(stage=p['stage'], sha256=p['sha256'],
                              opening_actions=p['opening_actions'], state=unpack(p['state']))
                         for p in chosen]
    return CORPUS[per_stage]


def identity(args, binary):
    return dict(plan=json.loads((HERE / 'plan.json').read_text()), stage=args.stage,
                platform=platform.platform(), machine=platform.machine(),
                binary=str(binary), binary_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
                revision=args.revision or subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                source_diff_sha256=None if args.revision else
                hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'])).hexdigest(),
                rows=[])


def save(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')


FIELDS = ('action', 'score', 'pv', 'nodes', 'completed_depth', 'complete', 'stop_reason',
          'proof_nodes', 'tt_hits', 'work', 'score_bound')


def record(result):
    return {name: result[name] for name in FIELDS}


def stage_baseline(args, teacher, report):
    """The frozen single-thread record threads = 1 has to reproduce."""
    for start in starts(2):
        for proof_nodes in (0, 64):
            for depth in range(1, args.max_depth + 1):
                result = teacher.analyze(search_observation(start['state']), depth=depth,
                                         seconds=600., proof_nodes=proof_nodes, weight=0.,
                                         table_entries=50000)
                report['rows'].append(dict(sha256=start['sha256'], stage=start['stage'],
                                           depth=depth, proof_node_limit=proof_nodes,
                                           **record(result)))
                save(report, args.output)


def stage_equivalence(args, teacher, report):
    """Replay the frozen rows at one thread, then repeat each thread count."""
    frozen = json.loads(Path(args.baseline).read_text())
    report['baseline'] = {k: frozen[k] for k in ('revision', 'binary_sha256', 'source_diff_sha256')}
    index = {(r['sha256'], r['depth'], r['proof_node_limit']): r for r in frozen['rows']}
    mismatches = []
    for key, expected in index.items():
        sha, depth, proof_nodes = key
        start = next(s for s in starts(2) if s['sha256'] == sha)
        result = teacher.analyze(search_observation(start['state']), depth=depth, seconds=600.,
                                 proof_nodes=proof_nodes, weight=0., table_entries=50000, threads=1)
        got = record(result)
        differing = {n: [expected[n], got[n]] for n in FIELDS if expected[n] != got[n]}
        report['rows'].append(dict(sha256=sha, depth=depth, proof_node_limit=proof_nodes,
                                   matches=not differing, differing=differing))
        if differing:
            mismatches.append(dict(key=list(key), differing=differing))
        save(report, args.output)
    report['baseline_equivalence'] = dict(rows=len(index), mismatches=mismatches,
                                          exact=not mismatches)
    # Determinism is a per-thread-count claim, so it is tested per thread count.
    repeats = []
    for start in starts(2):
        for threads in args.threads:
            observed = [teacher.analyze(search_observation(start['state']), depth=args.max_depth,
                                        seconds=600., proof_nodes=0, weight=0.,
                                        table_entries=50000, threads=threads)
                        for _ in range(args.repeats)]
            keys = {(r['action'], r['score'], tuple(r['pv']), r['nodes'], r['completed_depth'])
                    for r in observed}
            repeats.append(dict(sha256=start['sha256'], stage=start['stage'], threads=threads,
                                runs=args.repeats, identical=len(keys) == 1,
                                nodes=[r['nodes'] for r in observed],
                                seconds=[round(r['seconds'], 4) for r in observed],
                                parallel=observed[0]['parallel']))
            save(report, args.output)
    report['determinism'] = dict(rows=repeats, reproducible=all(r['identical'] for r in repeats))
    # How much bigger a tree each thread count searched for the same answer.
    report['node_ratios'] = [
        dict(sha256=row['sha256'], threads=row['threads'],
             node_ratio=row['nodes'][0] / next(r['nodes'][0] for r in repeats
                                               if r['sha256'] == row['sha256'] and r['threads'] == 1))
        for row in repeats]


def stage_overhead(args, teacher, report):
    """Nodes for the same depth, and depth reached inside the same wall clock."""
    plan = report['plan']
    for start in starts(2):
        for depth in plan['fixed_depth']['depths']:
            for threads in args.threads:
                result = teacher.analyze(search_observation(start['state']), depth=depth,
                                         seconds=float(plan['fixed_depth']['seconds_cap']),
                                         proof_nodes=plan['fixed_depth']['proof_nodes'], weight=0.,
                                         table_entries=50000, threads=threads)
                report['rows'].append(dict(kind='fixed_depth', sha256=start['sha256'],
                                           stage=start['stage'], depth=depth, threads=threads,
                                           seconds=result['seconds'], parallel=result['parallel'],
                                           **record(result)))
                save(report, args.output)
        for seconds in plan['fixed_time']['seconds']:
            for threads in args.threads:
                result = teacher.analyze(search_observation(start['state']),
                                         depth=plan['fixed_time']['depth'], seconds=float(seconds),
                                         proof_nodes=0, weight=0., table_entries=50000,
                                         threads=threads)
                report['rows'].append(dict(kind='fixed_time', sha256=start['sha256'],
                                           stage=start['stage'], budget_seconds=seconds,
                                           threads=threads, seconds=result['seconds'],
                                           parallel=result['parallel'], **record(result)))
                save(report, args.output)
    fixed = [r for r in report['rows'] if r['kind'] == 'fixed_depth']
    summary = {}
    for depth in plan['fixed_depth']['depths']:
        base = {r['sha256']: r for r in fixed if r['depth'] == depth and r['threads'] == 1}
        for threads in args.threads:
            rows = [r for r in fixed if r['depth'] == depth and r['threads'] == threads
                    and r['completed_depth'] == depth and base[r['sha256']]['completed_depth'] == depth]
            if not rows:
                continue
            overhead = [r['nodes'] / base[r['sha256']]['nodes'] for r in rows]
            speed = [base[r['sha256']]['seconds'] / r['seconds'] for r in rows if r['seconds'] > 0]
            summary[f'depth{depth}_threads{threads}'] = dict(
                positions=len(rows), median_node_overhead=float(np.median(overhead)),
                median_time_speedup=float(np.median(speed)),
                total_nodes=sum(r['nodes'] for r in rows),
                identical_action=all(r['action'] == base[r['sha256']]['action'] for r in rows),
                identical_score=all(r['score'] == base[r['sha256']]['score'] for r in rows),
                identical_pv=all(r['pv'] == base[r['sha256']]['pv'] for r in rows))
    # Effective branching factor from consecutive completed depths, per thread
    # count: a larger tree for the same depth shows up directly here.
    branching = {}
    for threads in args.threads:
        ratios = []
        for start in starts(2):
            rows = {r['depth']: r for r in fixed if r['threads'] == threads
                    and r['sha256'] == start['sha256'] and r['completed_depth'] == r['depth']}
            for depth in plan['fixed_depth']['depths']:
                if depth - 1 in rows and depth in rows and rows[depth - 1]['nodes']:
                    ratios.append(rows[depth]['nodes'] / rows[depth - 1]['nodes'])
        branching[threads] = float(np.median(ratios)) if ratios else None
    timed = [r for r in report['rows'] if r['kind'] == 'fixed_time']
    depths = {}
    for seconds in plan['fixed_time']['seconds']:
        for threads in args.threads:
            rows = [r for r in timed if r['budget_seconds'] == seconds and r['threads'] == threads]
            depths[f'{seconds}s_threads{threads}'] = dict(
                mean_completed_depth=float(np.mean([r['completed_depth'] for r in rows])),
                total_nodes=sum(r['nodes'] for r in rows))
    report['summary'] = dict(fixed_depth=summary, effective_branching_factor=branching,
                             fixed_time=depths)


def play(teacher, state, engines, plan, moves):
    """One equal-time game. Both engines get the same clock on every move."""
    game = IntransitiveGame(modelling_draws=False)
    winner = None
    for _ in range(plan['paired_protocol']['max_plies']):
        side = int(state[:, :, 82:84].flat[1])
        terminal = game.getGameEnded(state, side)
        if terminal.any():
            winner = int(np.argmax(terminal))
            break
        result = teacher.analyze(search_observation(state), depth=20,
                                 seconds=float(plan['paired_protocol']['seconds_per_move']),
                                 proof_nodes=64, weight=0., table_entries=50000,
                                 threads=engines[side])
        action = result['action']
        if action is None:
            action = int(np.flatnonzero(game.getValidMoves(state, side))[0])
        assert game.getValidMoves(state, side)[action]
        moves.append(dict(side=side, threads=engines[side], action=action,
                          depth=result['completed_depth'], nodes=result['nodes'],
                          seconds=round(result['seconds'], 4)))
        state, _ = game.getNextState(state, side, action)
    else:
        terminal = game.getGameEnded(state, int(state[:, :, 82:84].flat[1]))
        if terminal.any():
            winner = int(np.argmax(terminal))
    return winner


def stage_paired(args, teacher, report):
    plan = report['plan']
    # Overrides exist so a second time control can be run without editing the
    # predeclared plan. Leaving them unset reproduces the plan exactly, so the
    # primary run's code path is the one the plan describes.
    protocol = dict(plan['paired_protocol'])
    if args.seconds_per_move is not None:
        protocol['seconds_per_move'] = args.seconds_per_move
    if args.paired_threads:
        protocol['thread_counts'] = args.paired_threads
    # Record the start count actually used; the plan's own string describes the
    # default and stops being true as soon as --per-stage is passed.
    protocol['per_stage'] = args.per_stage
    protocol['predeclared'] = (args.seconds_per_move is None and not args.paired_threads
                               and args.per_stage == 8)
    plan = dict(plan, paired_protocol=protocol)
    report['plan'] = plan
    report['paired_protocol'] = protocol
    for threads in protocol['thread_counts']:
        for start in starts(args.per_stage):
            for candidate_side in (0, 1):
                engines = {candidate_side: threads, 1 - candidate_side: 1}
                moves = []
                begin = perf_counter()
                winner = play(teacher, start['state'].copy(), engines, plan, moves)
                report['rows'].append(dict(
                    threads=threads, sha256=start['sha256'], stage=start['stage'],
                    candidate_side=candidate_side, plies=len(moves),
                    wall_seconds=round(perf_counter() - begin, 3),
                    outcome=('unresolved' if winner is None else
                             'win' if winner == candidate_side else 'loss'),
                    candidate_mean_depth=float(np.mean([m['depth'] for m in moves
                                                        if m['side'] == candidate_side] or [0])),
                    baseline_mean_depth=float(np.mean([m['depth'] for m in moves
                                                       if m['side'] != candidate_side] or [0])),
                    moves=moves))
                save(report, args.output)
    summary = {}
    for threads in plan['paired_protocol']['thread_counts']:
        rows = [r for r in report['rows'] if r['threads'] == threads]
        counts = Counter(r['outcome'] for r in rows)
        summary[threads] = dict(
            games=len(rows), wins=counts['win'], losses=counts['loss'],
            unresolved=counts['unresolved'],
            score=(counts['win'] + .5 * counts['unresolved']) / len(rows) if rows else None,
            decisive_wilson_95=wilson(counts['win'], counts['loss']),
            by_stage={stage: dict(Counter(r['outcome'] for r in rows if r['stage'] == stage))
                      for stage in ('official',) + STAGES},
            mean_depth=dict(candidate=float(np.mean([r['candidate_mean_depth'] for r in rows])),
                            baseline=float(np.mean([r['baseline_mean_depth'] for r in rows]))))
    report['summary'] = summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True,
                        choices=('baseline', 'equivalence', 'overhead', 'paired'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=BINARY)
    parser.add_argument('--baseline', type=Path, help='Frozen baseline record, for --stage equivalence')
    parser.add_argument('--revision', help='Exact revision, when measuring an archived source tree')
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--per-stage', type=int, default=8)
    parser.add_argument('--seconds-per-move', type=float,
                        help='Override the predeclared paired time control; marks the run exploratory')
    parser.add_argument('--paired-threads', type=int, nargs='+',
                        help='Override the predeclared paired thread counts; marks the run exploratory')
    args = parser.parse_args()
    if args.stage == 'equivalence' and not args.baseline:
        parser.error('--stage equivalence needs the frozen --baseline record')
    report = identity(args, args.binary)
    start = perf_counter()
    with RustTeacher(args.binary) as teacher:
        report['native_startup_seconds'] = perf_counter() - start
        {'baseline': stage_baseline, 'equivalence': stage_equivalence,
         'overhead': stage_overhead, 'paired': stage_paired}[args.stage](args, teacher, report)
    report['total_seconds'] = perf_counter() - start
    save(report, args.output)
    print(json.dumps(report.get('summary') or report.get('baseline_equivalence')
                     or dict(rows=len(report['rows'])), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
