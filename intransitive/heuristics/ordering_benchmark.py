"""Move-ordering ablation: python -m intransitive.heuristics.ordering_benchmark.

Two bounded experiments, both on one fixed corpus and one fixed genome:

``fixed``      one search per position per configuration at a fixed depth, so
               nodes, work and the ordering statistics are directly comparable.
``equal-time`` paired games at an equal per-move deadline, so an ordering change
               is judged by the games it wins rather than by the nodes it saves.

Node counts alone are not acceptance. The ordering statistics (first-move cutoff
rate, mean cutoff index, cutoffs by depth) are what predict whether a reduction
schedule is safe; the equal-time games are what establish strength.
"""
import argparse
import json
import math
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np

from . import AlphaBetaPlayer, SearchConfig
from ..IntransitiveGame import IntransitiveGame
from ..record import state_hash
from .search import ORDERING_COUNTERS

SCHEMA = 'intransitive-ordering-v1'

# The killers/history ordering the issue names as the baseline to beat, and the
# mechanisms added on top of it. Every row shares one evaluator: only ordering
# and search settings differ.
LADDER = {
    'killers-history': {},
    'counter-move': dict(counter_move_enabled=True),
    'continuation-1': dict(continuation_enabled=True, continuation_plies=1),
    'continuation-2': dict(continuation_enabled=True, continuation_plies=2),
    'history-aging': dict(history_aging_enabled=True),
    # `iir_min_depth` must be reachable below the root: at a depth-d search the
    # deepest non-root node has d-1 remaining, so the default 4 cannot fire in a
    # depth-4 ablation at all. This is the #66 failure mode, and `firing` below
    # reports it rather than leaving it to be discovered by hand.
    'iir-reduce': dict(iir_enabled=True, iir_mode='reduce', iir_min_depth=3),
    'iir-deepen': dict(iir_enabled=True, iir_mode='deepen', iir_min_depth=3),
    'counter+continuation': dict(counter_move_enabled=True, continuation_enabled=True,
                                 continuation_plies=2),
    'all-ordering': dict(counter_move_enabled=True, continuation_enabled=True,
                         continuation_plies=2, history_aging_enabled=True,
                         iir_enabled=True, iir_mode='deepen', iir_min_depth=3),
}

# The counters each mechanism must move if it ran at all.
FIRING = {
    'counter_move_enabled': ('counter_move_updates',),
    'continuation_enabled': ('continuation_updates',),
    'history_aging_enabled': ('history_decays',),
    'iir_enabled': ('iir_nodes',),
}

# Two reduction schedules. `lmr` is the configuration whose re-search rate the
# issue quotes as the baseline to beat; `lmr-pvs` adds the null windows that
# #66 found NMP and futility need, and is measured separately because the scout
# searches change both the tree and the cutoff distribution.
SCHEDULES = {
    'plain': {},
    'lmr': dict(lmr_enabled=True),
    'lmr-pvs': dict(lmr_enabled=True, pvs_enabled=True),
}

# The adopted default evaluator, and the documented core preset, which is an
# order of magnitude cheaper per node because it builds no route maps.
GENOMES = {
    'adopted': {},
    'core': dict(count_weight=100., advantage_weight=25., attack_enabled=False,
                 defence_enabled=False, overload_enabled=False),
}

# One source of truth for the counter names the search publishes.
STATISTICS = ORDERING_COUNTERS


def corpus(game, *, games=8, plies=40, seed=911, size=24):
    """Reachable positions from seeded random legal play, including captures.

    Sampling every position of a playout would weight the corpus towards the
    openings all playouts share, so take an even spread of each game instead.
    """
    generator, states = np.random.default_rng(seed), []
    for _ in range(games):
        state, side, line = game.getInitBoard(), 0, []
        for _ in range(plies):
            if game.getGameEnded(state, side).any():
                break
            legal = np.flatnonzero(game.getValidMoves(state, side))
            if not legal.size:
                break
            state, side = game.getNextState(state, side, int(generator.choice(legal)))
            if not game.getGameEnded(state, side).any():
                line.append(state.copy())
        step = max(1, len(line) // max(1, size // games))
        states.extend(line[step - 1::step][:max(1, size // games)])
    return states[:size]


def base_config(genome='adopted', **overrides):
    """One genome for every row in a report; only ordering settings differ."""
    return SearchConfig(ordering_enabled=True, compiled_ordering_enabled=True,
                        **GENOMES[genome], **overrides)


def firing(config, totals):
    """Mechanisms enabled in this configuration that never executed."""
    return sorted(name for name, counters in FIRING.items()
                  if getattr(config, name) and not any(totals[c] for c in counters))


def measure(game, states, config):
    """One fresh engine per position, exactly as the match harness plays."""
    totals = dict.fromkeys(STATISTICS, 0)
    nodes = work = 0
    reduced = researches = 0
    depths = {}
    elapsed = 0.
    per_position = []
    for state in states:
        start = perf_counter()
        result = AlphaBetaPlayer(game, config).analyze(state)
        seconds = perf_counter() - start
        elapsed += seconds
        nodes += result.nodes
        work += result.work
        reduced += result.selective['lmr_reduced']
        researches += result.selective['lmr_researches']
        for name in STATISTICS:
            totals[name] += result.ordering[name]
        for level, counts in result.ordering['cutoffs_by_depth'].items():
            row = depths.setdefault(int(level), dict(cutoffs=0, first_move=0, index_sum=0))
            for field, value in counts.items():
                row[field] += value
        per_position.append(dict(state_sha256=state_hash(state), action=int(result.action),
                                 score=result.score, nodes=result.nodes, work=result.work,
                                 completed_depth=result.completed_depth,
                                 stopped=result.stopped, seconds=seconds))
    cutoffs = totals['beta_cutoffs']
    return dict(nodes=nodes, work=work, seconds=elapsed, positions=len(states),
                lmr_reduced=reduced, lmr_researches=researches,
                lmr_research_rate=researches / reduced if reduced else None,
                first_move_cutoff_rate=totals['first_move_cutoffs'] / cutoffs if cutoffs else None,
                mean_cutoff_index=totals['cutoff_index_sum'] / cutoffs if cutoffs else None,
                cutoffs_by_depth={str(level): depths[level] for level in sorted(depths)},
                **totals, positions_detail=per_position)


def fixed(args):
    game = IntransitiveGame()
    states = corpus(game, games=args.games, plies=args.plies, seed=args.seed, size=args.positions)
    limits = dict(max_depth=args.depth, time_limit=args.seconds, node_limit=args.work,
                  table_entries=args.table_entries)
    rows, inert = {}, {}
    ladder = {name: flags for name, flags in LADDER.items()
              if not args.rows or name in args.rows or name == 'killers-history'}
    for schedule in args.schedules:
        for name, flags in ladder.items():
            config = base_config(args.genome, **limits, **SCHEDULES[schedule], **flags)
            label = name if schedule == 'plain' else f'{name}+{schedule}'
            print(f'{label} ...', file=sys.stderr, flush=True)
            measured = measure(game, states, config)
            rows[label] = dict(config=config.to_dict(), schedule=schedule,
                               mechanisms=sorted(flags), **measured)
            # #66's fifth proposal: a technique that is enabled and then never
            # executes belongs in the report, not in a later investigation.
            rows[label]['inert_mechanisms'] = firing(config, measured)
            if rows[label]['inert_mechanisms']:
                inert[label] = rows[label]['inert_mechanisms']
    for label, row in rows.items():
        suffix = '' if row['schedule'] == 'plain' else '+' + row['schedule']
        reference = rows['killers-history' + suffix]
        row['baseline'] = 'killers-history' + suffix
        row['nodes_vs_baseline'] = row['nodes'] / reference['nodes'] if reference['nodes'] else None
        row['work_vs_baseline'] = row['work'] / reference['work'] if reference['work'] else None
    config = base_config(args.genome, **limits)
    return dict(schema=SCHEMA, experiment='fixed-depth', environment=environment(),
                corpus=dict(positions=[state_hash(s) for s in states], games=args.games,
                            plies=args.plies, seed=args.seed),
                limits=limits, baseline='killers-history', genome_name=args.genome,
                table_entries=args.table_entries,
                schedules=list(args.schedules), inert_mechanisms=inert,
                genome={k: v for k, v in config.to_dict().items()
                        if k.endswith(('_weight', '_bonus', '_enabled'))
                        and not k.startswith(('lmr', 'nmp', 'futility', 'counter', 'continuation',
                                              'history', 'iir', 'pvs', 'ordering', 'compiled',
                                              'aspiration', 'mvv', 'quiescence', 'selective',
                                              'certificate', 'depth_replacement'))},
                rows=rows)


def play(game, state, side, engines, *, max_plies):
    """One official game from a shared opening; engines are keyed by physical side.

    `getGameEnded` reports absolute players, so the returned outcome is indexed
    by physical colour and needs no reorientation. A game that reaches the ply
    cap is reported as unresolved rather than adjudicated.
    """
    actions, plies = [], 0
    # Per side: completed depth and the counters that say whether the mechanisms
    # under test could fire at all at this deadline. An equal-time result from
    # searches too shallow to reach them would say nothing about them.
    firing = {0: dict(moves=0, depth=0, fallbacks=0, nodes=0, iir_nodes=0,
                      counter_move_updates=0, continuation_updates=0),
              1: dict(moves=0, depth=0, fallbacks=0, nodes=0, iir_nodes=0,
                      counter_move_updates=0, continuation_updates=0)}
    outcome = game.getGameEnded(state, side)
    while not outcome.any() and plies < max_plies:
        result = AlphaBetaPlayer(game, engines[side]).analyze(state)
        row = firing[side]
        row['moves'] += 1
        row['depth'] += result.completed_depth
        row['fallbacks'] += result.completed_depth == 0
        row['nodes'] += result.nodes
        for name in ('iir_nodes', 'counter_move_updates', 'continuation_updates'):
            row[name] += result.ordering[name]
        action = int(result.action)
        actions.append(action)
        state, side = game.getNextState(state, side, action)
        outcome, plies = game.getGameEnded(state, side), plies + 1
    for row in firing.values():
        row['mean_depth'] = row['depth'] / row['moves'] if row['moves'] else 0.
    if not outcome.any():
        return actions, None, 'unresolved', firing
    board = IntransitiveGame().board
    board.copy_state(state, False)
    return actions, [float(value) for value in outcome], board.get_terminal_reason(), firing


def openings(game, *, count, plies, seed):
    generator, found = np.random.default_rng(seed), []
    while len(found) < count:
        state, side = game.getInitBoard(), 0
        for _ in range(plies):
            if game.getGameEnded(state, side).any():
                break
            legal = np.flatnonzero(game.getValidMoves(state, side))
            state, side = game.getNextState(state, side, int(generator.choice(legal)))
        if not game.getGameEnded(state, side).any():
            found.append((state, side))
    return found


def equal_time(args):
    """Paired games at an equal per-move deadline, both colours per opening.

    Both engines are deterministic, so the opening pool is the only source of
    variation, and one opening played from both sides is one paired unit.
    Unresolved games score half and are also reported separately: they are a
    limitation of the ply cap, not evidence of equal strength.
    """
    game = IntransitiveGame()
    limits = dict(max_depth=args.depth, time_limit=args.seconds, node_limit=args.work,
                  table_entries=args.table_entries)
    challenger = base_config(args.genome, **limits, **SCHEDULES[args.schedule],
                             **LADDER[args.challenger])
    incumbent = base_config(args.genome, **limits, **SCHEDULES[args.schedule],
                            **LADDER[args.incumbent])
    pool = openings(game, count=args.openings, plies=args.opening_plies, seed=args.seed)
    games, units = [], []
    for index, (state, side) in enumerate(pool):
        unit = []
        for first in (0, 1):
            engines = {first: challenger, 1 - first: incumbent}
            start = perf_counter()
            actions, outcome, reason, firing = play(game, state.copy(), side, engines,
                                                    max_plies=args.max_plies)
            score = .5 if outcome is None else (
                1. if outcome[first] == 1. else 0. if outcome[1 - first] == 1. else .5)
            unit.append(score)
            games.append(dict(opening=index, challenger_side=first, reason=reason,
                              outcome=outcome, challenger_score=score, plies=len(actions),
                              challenger_search=firing[first], incumbent_search=firing[1 - first],
                              actions=actions, seconds=perf_counter() - start))
            print(f'opening {index} side {first}: {reason} {score}', file=sys.stderr, flush=True)
        units.append(sum(unit) / len(unit))
    score = float(np.mean(units)) if units else 0.
    # The same conservative Hoeffding radius the ablation benchmark uses: one
    # paired opening is one independent unit and the count is deliberately small.
    radius = math.sqrt(math.log(40) / (2 * len(units))) if units else 1.
    return dict(schema=SCHEMA, experiment='equal-time', environment=environment(),
                challenger=args.challenger, incumbent=args.incumbent,
                challenger_config=challenger.to_dict(), incumbent_config=incumbent.to_dict(),
                limits=limits, genome_name=args.genome, schedule=args.schedule,
                openings=len(pool), opening_plies=args.opening_plies,
                seed=args.seed, max_plies=args.max_plies, games=len(games),
                unresolved=sum(row['reason'] == 'unresolved' for row in games),
                wins=sum(row['challenger_score'] == 1. for row in games),
                draws=sum(row['challenger_score'] == .5 for row in games),
                losses=sum(row['challenger_score'] == 0. for row in games),
                score=score, score_interval=[max(0., score - radius), min(1., score + radius)],
                interval='conservative 95% Hoeffding over paired openings',
                challenger_search=aggregate(games, 'challenger_search'),
                incumbent_search=aggregate(games, 'incumbent_search'),
                games_detail=games)


def aggregate(games, field):
    """Totals across one side's searches, plus the mean completed depth."""
    rows = [row[field] for row in games]
    totals = {name: sum(row[name] for row in rows)
              for name in ('moves', 'depth', 'fallbacks', 'nodes', 'iir_nodes',
                           'counter_move_updates', 'continuation_updates')}
    totals['mean_depth'] = totals['depth'] / totals['moves'] if totals['moves'] else 0.
    return totals


def table(args):
    """Render a saved report as the markdown the benchmark README publishes."""
    report = json.loads(args.report.read_text())
    if report['experiment'] == 'equal-time':
        lines = [f"| challenger | incumbent | games | W/D/L | unresolved | score | 95% interval |",
                 '| --- | --- | ---: | ---: | ---: | ---: | --- |',
                 '| `{challenger}` | `{incumbent}` | {games} | {wins}/{draws}/{losses} | '
                 '{unresolved} | {score:.3f} | [{low:.3f}, {high:.3f}] |'.format(
                     low=report['score_interval'][0], high=report['score_interval'][1], **report)]
        lines += ['', '| side | moves | mean completed depth | fallbacks | nodes | iir nodes '
                  '| counter updates | continuation updates |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for name in ('challenger', 'incumbent'):
            row = report[name + '_search']
            lines.append('| {label} | {moves:,} | {mean_depth:.3f} | {fallbacks:,} | {nodes:,} '
                         '| {iir_nodes:,} | {counter_move_updates:,} | {continuation_updates:,} |'.format(
                             label=f'`{report[name]}`', **row))
        return print('\n'.join(lines))
    header = ('| row | mechanisms | nodes | vs base | fmc | idx | cutoffs | '
              'from TT/killer/counter/other | LMR reduced | re-searched | rate | inert |')
    lines = [header, '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for label, row in report['rows'].items():
        sources = '/'.join(str(row['cutoff_from_' + name])
                           for name in ('preferred', 'killer', 'counter', 'other'))
        lines.append('| `{label}` | {mechanisms} | {nodes:,} | {ratio:.4f} | {fmc:.4f} | {idx:.4f} '
                     '| {cutoffs:,} | {sources} | {reduced:,} | {researches:,} | {rate} | {inert} |'.format(
                         label=label,
                         mechanisms=', '.join(m.replace('_enabled', '') for m in row['mechanisms']) or '—',
                         nodes=row['nodes'], ratio=row['nodes_vs_baseline'],
                         fmc=row['first_move_cutoff_rate'], idx=row['mean_cutoff_index'],
                         cutoffs=row['beta_cutoffs'], sources=sources,
                         reduced=row['lmr_reduced'], researches=row['lmr_researches'],
                         rate=('%.4f' % row['lmr_research_rate']) if row['lmr_research_rate'] is not None else '—',
                         inert=', '.join(row['inert_mechanisms']) or 'none'))
    print('\n'.join(lines))


def environment():
    import numba
    return dict(python=sys.version.split()[0], numpy=np.__version__, numba=numba.__version__,
                platform=platform.platform(), machine=platform.machine())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Write the JSON report here')
    sub = parser.add_subparsers(dest='command', required=True)

    ablation = sub.add_parser('fixed', help='Fixed-depth ordering ablation')
    ablation.add_argument('--depth', type=int, default=4)
    ablation.add_argument('--positions', type=int, default=24)
    ablation.add_argument('--games', type=int, default=8)
    ablation.add_argument('--plies', type=int, default=40)
    ablation.add_argument('--seed', type=int, default=911)
    ablation.add_argument('--seconds', type=float, default=3600.)
    ablation.add_argument('--work', type=int, default=10**12)
    ablation.add_argument('--genome', default='adopted', choices=sorted(GENOMES))
    ablation.add_argument('--table-entries', type=int, default=SearchConfig().table_entries,
                          help='Transposition-table capacity; a small table is the only '
                               'regime in which a node reaches IIR without a stored move')
    ablation.add_argument('--rows', nargs='*', default=[], choices=sorted(LADDER),
                          help='Restrict the ladder; the baseline row is always included')
    ablation.add_argument('--schedules', nargs='+', default=['plain', 'lmr', 'lmr-pvs'],
                          choices=sorted(SCHEDULES), help='Reduction schedules to run the ladder under')
    ablation.set_defaults(run=fixed)

    match = sub.add_parser('equal-time', help='Paired equal-time games')
    match.add_argument('--challenger', default='all-ordering', choices=sorted(LADDER))
    match.add_argument('--incumbent', default='killers-history', choices=sorted(LADDER))
    match.add_argument('--depth', type=int, default=64)
    match.add_argument('--seconds', type=float, default=1.,
                       help='Equal per-move deadline; below this the default genome '
                            'mostly returns its legal fallback')
    match.add_argument('--work', type=int, default=10**9)
    match.add_argument('--openings', type=int, default=8)
    match.add_argument('--opening-plies', type=int, default=6)
    match.add_argument('--max-plies', type=int, default=160)
    match.add_argument('--seed', type=int, default=2026)
    match.add_argument('--genome', default='adopted', choices=sorted(GENOMES))
    match.add_argument('--schedule', default='plain', choices=sorted(SCHEDULES))
    match.add_argument('--table-entries', type=int, default=SearchConfig().table_entries)
    match.set_defaults(run=equal_time)

    rendered = sub.add_parser('table', help='Render a saved report as markdown')
    rendered.add_argument('report', type=Path)
    rendered.set_defaults(run=table)

    args = parser.parse_args(argv)
    report = args.run(args)
    if report is None:
        return None
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n')
    else:
        print(text)
    return report


if __name__ == '__main__':
    main()
