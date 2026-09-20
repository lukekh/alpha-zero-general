"""Bounded measurements for issue #67; run as a module from the repository root.

Three questions, in the order the issue asks them: does the exchange evaluation
agree with its reference, does it make quiescence cheaper, and does the engine
play at least as well at equal time. Every ratio compares rows measured in the
same run on the same position in the same material mode.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np

from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.heuristics import AlphaBetaPlayer, SearchConfig
from intransitive.heuristics import exchange
from intransitive.heuristics.budget import Budget, BudgetExpired
from intransitive.heuristics.evaluation import Evaluator
from intransitive.heuristics.material import count_pieces
from intransitive.tournament import spec

HERE = Path(__file__).resolve().parent

WORK_MODES = {
    'baseline': {},
    'quiescence': dict(quiescence_enabled=True),
    'quiescence+see': dict(quiescence_enabled=True, see_quiescence_pruning_enabled=True,
                           see_quiescence_ordering_enabled=True),
    'quiescence+delta0': dict(quiescence_enabled=True, delta_pruning_enabled=True, delta_margin=0.),
    'quiescence+delta1': dict(quiescence_enabled=True, delta_pruning_enabled=True, delta_margin=1.),
    'quiescence+see+delta0': dict(quiescence_enabled=True, see_quiescence_pruning_enabled=True,
                                  see_quiescence_ordering_enabled=True,
                                  delta_pruning_enabled=True, delta_margin=0.),
}
ORDERING_MODES = {
    'killers': {},
    'mvv': dict(mvv_lva_enabled=True),
    'see': dict(see_ordering_enabled=True),
    'see+mvv': dict(see_ordering_enabled=True, mvv_lva_enabled=True),
}
PAIRED_ARMS = {
    'see-ordering': dict(material='flat', shared={}, candidate=dict(see_ordering_enabled=True)),
    'quiescence-see': dict(material='variable', shared=dict(quiescence_enabled=True),
                           candidate=dict(see_quiescence_pruning_enabled=True,
                                          see_quiescence_ordering_enabled=True)),
    'quiescence-delta': dict(material='flat', shared=dict(quiescence_enabled=True),
                             candidate=dict(delta_pruning_enabled=True, delta_margin=0.)),
}


def wilson(wins, losses):
    n = wins + losses
    if not n:
        return [0., 1.]
    p, z = wins / n, 1.96
    mid = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * ((p*(1-p)/n + z*z/(4*n*n)) ** .5) / (1 + z*z/n)
    return [mid - half, mid + half]


def base_config(material, plan, depth, seconds):
    return SearchConfig(max_depth=depth, time_limit=seconds, node_limit=10**9,
                        ordering_enabled=True, compiled_ordering_enabled=True,
                        pvs_enabled=True, aspiration_enabled=True,
                        variable_material_enabled=material == 'variable')


def row(result):
    keys = ('action', 'score', 'completed_depth', 'nodes', 'work', 'tt_hits', 'stopped', 'elapsed')
    out = {k: getattr(result, k) for k in keys}
    out.update({k: result.selective[k] for k in
                ('quiescence_captures', 'quiescence_see_skips', 'quiescence_delta_skips')})
    out.update({k: result.ordering[k] for k in
                ('cutoffs', 'first_cutoffs', 'see_nodes', 'see_captures')})
    return out


def corpus(plan, limit=None):
    """Engine self-play positions where captures are actually on the board.

    Issue #66's multiplier came from real weight-evolution games. Random legal
    play does not reproduce that regime: `spec.generate_positions` spends its
    capture bias on the sampled ply, so all eight positions it selects have no
    legal capture at all and quiescence resolves nothing. Here each line plays a
    short seeded capture-biased prefix and then lets the baseline engine play
    itself, sampling the first ply at or after `min_ply` with at least
    `min_captures` legal captures, one position per stage per line. The position
    record, opening-action replay, orbit hash and pool split are the #54
    generator's, unchanged.
    """
    from intransitive.IntransitiveConstants import action_destination
    settings = plan['positions']
    game = IntransitiveGame(modelling_draws=False)
    engine = AlphaBetaPlayer(game, base_config('flat', plan, plan['work']['depth'],
                                               settings['engine_seconds']))
    selected, seen = [], set()
    for line in range(settings['lines']):
        stream = settings['seed'] + line + 1
        rng = np.random.default_rng(stream)
        state, side, actions, stages = game.getInitBoard(), 0, [], set()
        for ply in range(settings['max_plies']):
            legal = list(map(int, np.flatnonzero(game.getValidMoves(state, side))))
            if not legal:
                break
            captures = [a for a in legal
                        if state[action_destination(a)[1], action_destination(a)[0], 0]]
            pieces = int(np.count_nonzero(state[:, :, 0]))
            stage = 'opening' if ply < 12 else 'endgame' if pieces <= 12 else 'midgame'
            if (len(captures) >= settings['min_captures'] and ply >= settings['min_ply']
                    and stage not in stages):
                item = spec.position(list(actions), seed=stream, stage=stage)
                stages.add(stage)
                if item['orbit_sha256'] not in seen and item['pool'] in settings['pools']:
                    seen.add(item['orbit_sha256'])
                    item['captures_available'] = len(captures)
                    selected.append(item)
            if ply < settings['prefix_plies']:
                action = int(rng.choice(captures if captures and rng.random() < .9 else legal))
            else:
                action = int(engine.analyze(state).action)
            state, side = game.getNextState(state, side, action)
            actions.append(action)
            if game.getGameEnded(state, side).any():
                break
    selected.sort(key=lambda item: item['sha256'])
    return selected[:limit or settings['limit']]


def parity(items, report, write):
    """Compiled kernel against the Python reference on every legal capture."""
    game = IntransitiveGame()
    compared = mismatches = 0
    for item in items:
        state = spec.unpack(item['state'])
        side = int(state[:, :, 82:84].flat[1])
        pieces = np.ascontiguousarray(state[:, :, 0])
        counts = np.asarray(count_pieces(state), dtype=np.int64)
        actions = np.flatnonzero(game.getValidMoves(state, side)).astype(np.int64)
        for variable, linear in ((False, False), (True, False), (True, True)):
            args = (pieces, counts, side, actions, 100., variable, linear)
            fast, slow = exchange.exchange_swings(*args), exchange.exchange_swings_reference(*args)
            compared += fast[2]
            mismatches += int(not (np.array_equal(fast[0], slow[0])
                                   and np.array_equal(fast[1], slow[1]) and fast[2] == slow[2]))
    report['parity'] = dict(captures_compared=compared, mismatches=mismatches)
    write()


def fixed(items, plan, report, write, section, modes, key):
    settings = plan[section]
    for item in items:
        state = spec.unpack(item['state'])
        for material in settings['material_modes']:
            config = base_config(material, plan, settings['depth'], settings['seconds_cap'])
            for mode, extra in modes.items():
                player = AlphaBetaPlayer(IntransitiveGame(), replace(config, **extra))
                record = row(player.analyze(state))
                record.update(stage=item['stage'], position=item['sha256'],
                              material=material, mode=mode)
                if mode == 'quiescence+delta1':
                    record['delta_allowance'] = player.last_result.ordering['delta_allowance']
                report[key].append(record)
                write()


def play(items, plan, report, write):
    game = IntransitiveGame(modelling_draws=False)
    limits = plan['paired']
    for name in limits['arms']:
        arm = PAIRED_ARMS[name]
        config = replace(base_config(arm['material'], plan, limits['max_depth'],
                                     limits['seconds_per_move']), **arm['shared'])
        candidate_config = replace(config, **arm['candidate'])
        for item in items:
            for candidate_side in range(limits['colours']):
                state, moves, winner = spec.unpack(item['state']), [], None
                players = {side: AlphaBetaPlayer(game, candidate_config if side == candidate_side else config)
                           for side in (0, 1)}
                for _ in range(limits['plies_cap']):
                    side = int(state[:, :, 82:84].flat[1])
                    terminal = game.getGameEnded(state, side)
                    if terminal.any():
                        winner = -1 if terminal[side] == terminal[1-side] else int(np.argmax(terminal))
                        break
                    result = players[side].analyze(state)
                    action = result.action
                    assert game.getValidMoves(state, side)[action]
                    moves.append(dict(action=int(action), side=side, depth=result.completed_depth,
                                      nodes=result.nodes, work=result.work))
                    state, _ = game.getNextState(state, side, action)
                outcome = ('unfinished' if winner is None else 'draw' if winner < 0
                           else 'win' if winner == candidate_side else 'loss')
                verdict = None
                if outcome == 'unfinished':
                    verdict = adjudicate(game, state, config, candidate_side)
                report['paired'].append(dict(arm=name, material=arm['material'], stage=item['stage'],
                                             position=item['sha256'], candidate_side=candidate_side,
                                             outcome=outcome, adjudicated=verdict, moves=moves))
                write()
    report['paired_summary'] = {}
    for name in limits['arms']:
        rows = [r for r in report['paired'] if r['arm'] == name]
        counts = {o: sum(r['outcome'] == o for r in rows) for o in ('win', 'loss', 'draw', 'unfinished')}
        judged = {o: sum(r['adjudicated'] == o for r in rows) for o in ('win', 'loss', 'unresolved')}
        report['paired_summary'][name] = dict(
            counts, adjudicated=judged,
            decisive_wilson_95=wilson(counts['win'], counts['loss']),
            with_adjudicated_wilson_95=wilson(counts['win'] + judged['win'],
                                              counts['loss'] + judged['loss']),
            # Equal time only buys equal search if both sides really got it.
            # On a shared host they may not have; these say what each side saw.
            **nodes_per_move(rows))
    write()


def nodes_per_move(rows):
    def median(values):
        values = sorted(values)
        return values[len(values)//2] if values else None
    def side_of(row, own):
        return [m for m in row['moves'] if (m['side'] == row['candidate_side']) == own]
    candidate = [m for r in rows for m in side_of(r, True)]
    baseline = [m for r in rows for m in side_of(r, False)]
    return dict(candidate_median_nodes=median([m['nodes'] for m in candidate]),
                baseline_median_nodes=median([m['nodes'] for m in baseline]),
                candidate_median_depth=median([m['depth'] for m in candidate]),
                baseline_median_depth=median([m['depth'] for m in baseline]),
                candidate_moves=len(candidate), baseline_moves=len(baseline))


def adjudicate(game, state, config, candidate_side):
    """The #54 capped-game rule, with the caveat that one evaluator is shared."""
    evaluator = Evaluator(game, replace(config, proof_depth=0, proof_nodes=0))
    budget = Budget(5_000_000, 2.)
    try:
        value = float(evaluator.score(state, candidate_side, budget, proof={'status': 'unknown'}))
        budget.check()
    except BudgetExpired:
        return 'unresolved'
    return 'win' if value > 0 else 'loss' if value < 0 else 'unresolved'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--positions', type=int, default=None)
    parser.add_argument('--sections', default='parity,work,ordering,paired')
    args = parser.parse_args()
    plan = json.loads((HERE / 'plan.json').read_text())
    sections = args.sections.split(',')
    report = dict(plan=plan, platform=platform.platform(),
                  revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_diff_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'])).hexdigest(),
                  runtime=spec.runtime_versions(), backend_version=spec.backend_version(),
                  work=[], ordering=[], paired=[])

    def write():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')

    start = perf_counter()
    items = corpus(plan, args.positions)
    report['positions'] = [{k: item[k] for k in ('sha256', 'orbit_sha256', 'stage', 'pool',
                                                  'seed', 'captures_available', 'opening_actions')}
                           for item in items]
    report['corpus_seconds'] = perf_counter() - start
    start = perf_counter()
    AlphaBetaPlayer(config=SearchConfig(quiescence_enabled=True, see_ordering_enabled=True,
                                        see_quiescence_pruning_enabled=True,
                                        delta_pruning_enabled=True))._prepare()
    report['python_warmup_seconds'] = perf_counter() - start
    write()
    began = perf_counter()
    if 'parity' in sections:
        parity(items, report, write)
    if 'work' in sections:
        fixed(items, plan, report, write, 'work', WORK_MODES, 'work')
        report['work_multipliers'] = multipliers(report['work'])
    if 'ordering' in sections:
        fixed(items, plan, report, write, 'ordering', ORDERING_MODES, 'ordering')
        report['ordering_summary'] = ordering_summary(report['ordering'])
    if 'paired' in sections:
        play(items, plan, report, write)
    report['measurement_seconds'] = perf_counter() - began
    write()


def multipliers(rows):
    """Per-position quiescence cost against the same position without it."""
    index = {(r['position'], r['material'], r['mode']): r for r in rows}
    summary = {}
    for material in sorted({r['material'] for r in rows}):
        for mode in sorted({r['mode'] for r in rows} - {'baseline'}):
            ratios = []
            for position in sorted({r['position'] for r in rows}):
                off = index.get((position, material, 'baseline'))
                on = index.get((position, material, mode))
                if off and on and off['work']:
                    ratios.append(dict(position=position, ratio=on['work'] / off['work'],
                                       baseline_work=off['work'], work=on['work'],
                                       captures=on['quiescence_captures'],
                                       see_skips=on['quiescence_see_skips'],
                                       delta_skips=on['quiescence_delta_skips'],
                                       same_action=off['action'] == on['action']))
            if ratios:
                values = sorted(r['ratio'] for r in ratios)
                summary[f'{material}/{mode}'] = dict(
                    rows=ratios, minimum=values[0], maximum=values[-1],
                    median=values[len(values)//2],
                    total_ratio=sum(r['work'] for r in ratios) / sum(r['baseline_work'] for r in ratios))
    return summary


def ordering_summary(rows):
    summary = {}
    for material in sorted({r['material'] for r in rows}):
        for mode in sorted({r['mode'] for r in rows}):
            selected = [r for r in rows if r['material'] == material and r['mode'] == mode]
            cutoffs = sum(r['cutoffs'] for r in selected)
            summary[f'{material}/{mode}'] = dict(
                positions=len(selected), nodes=sum(r['nodes'] for r in selected),
                work=sum(r['work'] for r in selected), cutoffs=cutoffs,
                first_cutoffs=sum(r['first_cutoffs'] for r in selected),
                first_move_cutoff_rate=(sum(r['first_cutoffs'] for r in selected) / cutoffs) if cutoffs else None,
                completed=sum(not r['stopped'] for r in selected))
    return summary


if __name__ == '__main__':
    main()
