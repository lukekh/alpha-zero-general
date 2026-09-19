"""Module scale and influence over a corpus.

Run: uv run --locked python -m intransitive.heuristics.audit

Per-position attribution answers "what is this square worth". This answers the
questions that only a corpus can:

* **Scale** — how large is each module's term, by game phase? A module whose
  term is two orders of magnitude below material never changes a decision.
* **Reach** — how often is it exactly zero (dead), or pinned at a cap (blind to
  further change)? Both look like a working module in a single position.
* **Influence** — how often does removing it change which reply the *static*
  evaluation prefers? This is the only measure that says a module matters: a
  module with a large term that never flips a choice is a constant offset.

Influence is measured one ply deep on purpose. It isolates what the evaluation
contributes before search covers for it; the gap between a module's influence
here and its effect on played strength is the work depth is doing instead.
"""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from ..IntransitiveConstants import action_destination
from ..IntransitiveGame import IntransitiveGame
from .attribution import DEFAULT_MODULES, attribute, module_active
from .budget import Budget, BudgetExpired
from .config import SearchConfig
from .evaluation import HEURISTIC_LIMIT, MODULES, Evaluator

# Phase by material remaining, not ply: a position with four pieces is an
# endgame whether it arrived there on move 12 or move 60.
PHASES = (('opening', 16), ('middlegame', 8), ('endgame', 0))


def warm_kernels(config):
    """Compile everything the live search compiles before its first move.

    Without this the evaluator silently falls back to `prove_reference`, whose
    pure-Python traversal costs an order of magnitude more than the native
    proof the search actually runs — which would be measured as the heuristic's
    cost rather than as a cold start.
    """
    from .proof import warm_compact_proof_kernel, warm_proof_kernel
    from .search import AlphaBetaPlayer
    AlphaBetaPlayer(IntransitiveGame(), config)._prepare(warm_proof=True)
    warm_proof_kernel()
    warm_compact_proof_kernel()


def phase_of(pieces):
    return next(name for name, floor in PHASES if pieces >= floor)


def sample_positions(game, games, plies, seed, capture_bias=.4):
    """Reachable positions from legal play, biased toward captures.

    Unbiased random play almost never trades, so a plain random walk stays in
    the opening bucket and the middlegame and endgame rows come back empty.
    `capture_bias` is the chance of taking an available capture; it shapes the
    corpus only, and the positions it produces are ordinary legal positions.
    """
    generator = np.random.default_rng(seed)
    for _ in range(games):
        state, player = game.getInitBoard(), 0
        for _ in range(plies):
            if game.getGameEnded(state, player).any():
                break
            legal = np.flatnonzero(game.getValidMoves(state, player))
            if not legal.size:
                break
            captures = [action for action in legal
                        if state[action_destination(int(action))[1],
                                 action_destination(int(action))[0], 0]]
            choices = captures if captures and generator.random() < capture_bias else legal
            state, player = game.getNextState(state, player, int(generator.choice(choices)))
            yield state.copy(), player


def child_terms(game, evaluator, state, side, config, seconds):
    """Every legal reply's per-module terms, all from the mover's perspective."""
    rows = []
    for action in np.flatnonzero(game.getValidMoves(state, side)):
        child, _ = game.getNextState(state, side, int(action))
        budget = Budget(config.node_limit, seconds)
        try:
            explanation = evaluator.explain(child, side, budget, diagnostics=False)
        except BudgetExpired:
            return []
        if explanation.get('terminal') or explanation['proof'].get('status') == 'proven':
            # Terminal and proven children replace the weighted sum, so they
            # cannot be decomposed into per-module votes. One is enough to make
            # this position's choice not a heuristic choice at all.
            return []
        rows.append(explanation['terms'])
    return rows


def ranking(rows, names):
    """Reply indices ordered by the static score of the named modules."""
    totals = [sum(row[name] for name in names) for row in rows]
    return sorted(range(len(rows)), key=lambda i: -totals[i]), totals


def influence(rows, names):
    """Which modules change the preferred reply when they are removed."""
    if len(rows) < 2:
        return {}
    totals = [sum(row[name] for name in names) for row in rows]
    best = max(range(len(rows)), key=lambda i: totals[i])
    flipped = {}
    for name in names:
        without = [total - row[name] for total, row in zip(totals, rows)]
        alternative = max(range(len(rows)), key=lambda i: without[i])
        # A tie the module did not break is not a flip.
        flipped[name] = alternative != best and without[alternative] != without[best]
    return flipped


# Enabling any of attack/defence/overload switches the evaluator off its cheap
# material-only path and onto the full piece index with king-route maps, so a
# module's cost is not separable from that shared setup. Measure each one on
# its own against core, then measure what it adds on top of the others.
ABLATIONS = (
    ('core (material + advantage)', dict(attack_enabled=False, defence_enabled=False,
                                         overload_enabled=False, pressure_enabled=False)),
    ('+ pressure only', dict(attack_enabled=False, defence_enabled=False,
                             overload_enabled=False, pressure_enabled=True)),
    ('+ attack only', dict(attack_enabled=True, defence_enabled=False,
                           overload_enabled=False, pressure_enabled=False)),
    ('+ defence only', dict(attack_enabled=False, defence_enabled=True,
                            overload_enabled=False, pressure_enabled=False)),
    ('+ overload only', dict(attack_enabled=False, defence_enabled=False,
                             overload_enabled=True, pressure_enabled=False)),
    ('attack + defence', dict(attack_enabled=True, defence_enabled=True,
                              overload_enabled=False, pressure_enabled=False)),
    ('all modules', dict(attack_enabled=True, defence_enabled=True,
                         overload_enabled=True, pressure_enabled=True)),
)


def timing_report(config, buckets, warmup=3):
    """Instrumented per-module timers plus ablation deltas, by game phase.

    The instrumented timers come from the evaluator's own `module_seconds`, so
    they show where wall time goes in one configuration. They are not marginal
    costs: `Geometry` memoizes interceptions and safety answers, and whichever
    module asks first pays for the shared work. The ablation columns are the
    honest "what does enabling this cost me" number.
    """
    game = IntransitiveGame()
    warm_kernels(config)
    report = {}
    for phase, positions in buckets.items():
        if not positions:
            continue
        evaluator = Evaluator(game, config)
        for state, side in positions[:warmup]:
            evaluator.score(state, side, Budget(10**12, 3600.))
        seconds, calls, total = defaultdict(float), defaultdict(int), 0.
        for state, side in positions:
            budget = Budget(10**12, 3600.)
            start = perf_counter()
            evaluator.score(state, side, budget)
            total += perf_counter() - start
            for name, value in budget.module_seconds.items():
                seconds[name] += value
            for name, value in budget.module_calls.items():
                calls[name] += value
        count = len(positions)
        ablation, baseline = {}, None
        for label, overrides in ABLATIONS:
            variant = Evaluator(game, replace(config, **overrides))
            for state, side in positions[:warmup]:
                variant.score(state, side, Budget(10**12, 3600.))
            samples = []
            for state, side in positions:
                budget = Budget(10**12, 3600.)
                start = perf_counter()
                variant.score(state, side, budget)
                samples.append(perf_counter() - start)
            micro = 1e6 * median(samples)
            baseline = micro if baseline is None else baseline
            ablation[label] = dict(microseconds=micro, over_core=micro - baseline)
        report[phase] = dict(
            positions=count, microseconds=1e6 * total / count,
            timers={name: dict(microseconds=1e6 * value / count, share=value / total,
                               calls=calls[name] / count)
                    for name, value in sorted(seconds.items(), key=lambda kv: -kv[1])},
            untimed=dict(microseconds=1e6 * (total - sum(seconds.values())) / count,
                         share=(total - sum(seconds.values())) / total),
            ablation=ablation)
    return report


def render_timing(report):
    lines = []
    for phase, data in report.items():
        lines.append('')
        lines.append(f"{phase}: {data['positions']} positions, "
                     f"{data['microseconds']:.0f} us per full evaluation")
        lines.append(f"  {'timer':22s}{'us/eval':>10s}{'share':>8s}{'calls':>8s}")
        for name, row in data['timers'].items():
            lines.append(f"  {name:22s}{row['microseconds']:10.1f}{row['share']:8.0%}"
                         f"{row['calls']:8.1f}")
        lines.append(f"  {'(untimed remainder)':22s}{data['untimed']['microseconds']:10.1f}"
                     f"{data['untimed']['share']:8.0%}")
        lines.append(f"  {'ablation':30s}{'us/eval':>10s}{'over core':>11s}")
        for label, row in data['ablation'].items():
            lines.append(f"  {label:30s}{row['microseconds']:10.1f}{row['over_core']:+11.1f}")
    lines.append('')
    lines.append('Instrumented timers show where wall time goes in one configuration; the shared '
                 'route/interception cache means whichever module asks first pays for it. The '
                 'ablation rows are the marginal cost of enabling a module.')
    return '\n'.join(lines)


def summarize(values):
    if not values:
        return dict(count=0)
    magnitudes = sorted(abs(value) for value in values)
    return dict(count=len(values), median=median(magnitudes),
                p90=magnitudes[min(len(magnitudes) - 1, int(.9 * len(magnitudes)))],
                max=magnitudes[-1])


def audit(config, *, games=12, plies=40, seed=7, seconds=30., decisions=True, modules=None,
          search_depth=None, search_seconds=2., max_positions=None, capture_bias=.4):
    names = tuple(modules or [name for name in DEFAULT_MODULES if name != 'clear_run'])
    active = module_active(config)
    scoring = tuple(name for name in names if active[name])
    game = IntransitiveGame()
    warm_kernels(config)
    evaluator = Evaluator(game, config)
    terms = defaultdict(lambda: defaultdict(list))
    caps = defaultdict(lambda: defaultdict(int))
    zeros = defaultdict(lambda: defaultdict(int))
    flips = defaultdict(lambda: defaultdict(int))
    scores = defaultdict(list)
    counts = defaultdict(int)
    decided = defaultdict(int)
    skipped = defaultdict(int)
    searched = defaultdict(int)
    agreed = defaultdict(int)
    static_ranks = defaultdict(list)
    without_agreed = defaultdict(lambda: defaultdict(int))
    player = None
    if search_depth:
        from .search import AlphaBetaPlayer
        player = AlphaBetaPlayer(game, replace(config, max_depth=search_depth,
                                               time_limit=search_seconds))
    for state, side in sample_positions(game, games, plies, seed, capture_bias):
        budget = Budget(config.node_limit, seconds)
        try:
            explanation = evaluator.explain(state, side, budget, diagnostics=False)
        except BudgetExpired:
            continue
        if explanation.get('terminal') or explanation['proof'].get('status') == 'proven':
            skipped[phase_of(int(np.count_nonzero(state[:, :, 0])))] += 1
            continue
        phase = phase_of(int(np.count_nonzero(state[:, :, 0])))
        counts[phase] += 1
        scores[phase].append(explanation['raw_score'])
        # Binding caps come from the attribution report rather than a guess at
        # each module's ceiling, so the two views can never disagree.
        try:
            report = attribute(game, state, side, budget, config,
                               modules=scoring, explanation=explanation)
        except BudgetExpired:
            report = None
        for name in scoring:
            value = explanation['terms'][name]
            terms[phase][name].append(value)
            zeros[phase][name] += value == 0.
            module = (report or {}).get('modules', {}).get(name)
            if module is not None:
                caps[phase][name] += any(cap['binding'] for cap in module['caps'])
        if decisions or player is not None:
            rows = child_terms(game, evaluator, state, side, config, seconds)
            flipped = influence(rows, scoring)
            if flipped:
                decided[phase] += 1
                for name, changed in flipped.items():
                    flips[phase][name] += changed
            if player is not None and len(rows) > 1:
                actions = [int(a) for a in np.flatnonzero(game.getValidMoves(state, side))]
                chosen = actions.index(player.analyze(state).action)
                order, _ = ranking(rows, scoring)
                searched[phase] += 1
                agreed[phase] += order[0] == chosen
                static_ranks[phase].append(order.index(chosen) + 1)
                # Leave-one-out: a module whose removal costs agreement is
                # pointing the same way as depth, whatever its term size.
                for name in scoring:
                    remaining = [other for other in scoring if other != name]
                    if remaining:
                        without_agreed[phase][name] += ranking(rows, remaining)[0][0] == chosen
        if max_positions and sum(counts.values()) >= max_positions:
            break
    report = dict(config=config.to_dict(), scoring_modules=list(scoring),
                  corpus=dict(games=games, plies=plies, seed=seed, capture_bias=capture_bias,
                              positions=sum(counts.values()),
                              skipped_decisive=sum(skipped.values())),
                  phases={})
    for phase, _ in PHASES:
        if not counts[phase]:
            continue
        total = counts[phase]
        magnitudes = {name: sum(abs(v) for v in terms[phase][name]) for name in scoring}
        overall = sum(magnitudes.values()) or 1.
        depth_row = None
        if searched[phase]:
            ranks = sorted(static_ranks[phase])
            depth_row = dict(depth=search_depth, positions=searched[phase],
                             agreement=agreed[phase] / searched[phase],
                             static_rank_median=median(ranks),
                             static_rank_p90=ranks[min(len(ranks) - 1, int(.9 * len(ranks)))],
                             without={name: without_agreed[phase][name] / searched[phase]
                                      for name in scoring})
        report['phases'][phase] = dict(
            positions=total, decided=decided[phase], skipped_decisive=skipped[phase],
            depth_comparison=depth_row,
            score=dict(summarize(scores[phase]),
                       saturated=sum(abs(s) >= HEURISTIC_LIMIT for s in scores[phase]) / total),
            modules={name: dict(summarize(terms[phase][name]),
                                share=magnitudes[name] / overall,
                                zero=zeros[phase][name] / total,
                                at_cap=caps[phase][name] / total,
                                flips=(flips[phase][name] / decided[phase]) if decided[phase] else None,
                                agreement_cost=(
                                    (agreed[phase] - without_agreed[phase][name]) / searched[phase]
                                    if searched[phase] else None))
                     for name in scoring})
    return report


def render(report):
    lines = [f"positions {report['corpus']['positions']} "
             f"(skipped {report['corpus']['skipped_decisive']} terminal/proven)"]
    for phase, data in report['phases'].items():
        lines.append('')
        lines.append(f"{phase}: {data['positions']} positions, {data['decided']} with a "
                     f"decomposable reply set; |score| median {data['score']['median']:.1f}, "
                     f"p90 {data['score']['p90']:.1f}, max {data['score']['max']:.1f}, "
                     f"saturated {data['score']['saturated']:.0%}")
        lines.append(f"  {'module':22s}{'|term| med':>11s}{'p90':>9s}{'max':>10s}"
                     f"{'share':>8s}{'zero':>7s}{'at cap':>8s}{'flips':>8s}"
                     f"{'agree+' if data['depth_comparison'] else '':>9s}")
        for name, row in sorted(data['modules'].items(), key=lambda item: -item[1]['share']):
            flips = '—' if row['flips'] is None else f"{row['flips']:.0%}"
            cost = '' if row['agreement_cost'] is None else f"{row['agreement_cost']:+8.1%}"
            lines.append(f"  {name:22s}{row['median']:11.2f}{row['p90']:9.2f}{row['max']:10.2f}"
                         f"{row['share']:8.0%}{row['zero']:7.0%}{row['at_cap']:8.0%}{flips:>8s}{cost}")
        depth_row = data['depth_comparison']
        if depth_row:
            lines.append(f"  depth {depth_row['depth']} agrees with the static pick in "
                         f"{depth_row['agreement']:.0%} of {depth_row['positions']} positions; "
                         f"its choice sits at static rank {depth_row['static_rank_median']:.0f} "
                         f"(p90 {depth_row['static_rank_p90']})")
    lines.append('')
    lines.append("share = this module's fraction of total absolute term mass; zero = positions "
                 'where it contributed nothing; at cap = positions where a side hit its ceiling; '
                 'flips = positions where removing it changes the preferred reply at one ply; '
                 'agree+ = agreement with the deeper search that this module adds (negative means '
                 'the static pick matches depth more often without it).')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, help='SearchConfig preset; default is the shipped configuration')
    parser.add_argument('--games', type=int, default=12, help='Random games sampled')
    parser.add_argument('--plies', type=int, default=40, help='Maximum plies taken from each game')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--seconds', type=float, default=30., help='Budget per evaluation')
    parser.add_argument('--modules', nargs='+', choices=[n for n in MODULES if n != 'clear_run'])
    parser.add_argument('--no-decisions', action='store_true',
                        help='Skip the one-ply influence pass, which evaluates every legal reply')
    parser.add_argument('--search-depth', type=int,
                        help='Also search each position to this depth and report how often the '
                             'static evaluation already prefers the reply the search chooses')
    parser.add_argument('--search-seconds', type=float, default=2., help='Budget per comparison search')
    parser.add_argument('--max-positions', type=int, help='Stop after this many scored positions')
    parser.add_argument('--capture-bias', type=float, default=.4,
                        help='Chance of taking an available capture while sampling, so the corpus '
                             'reaches the middlegame and endgame buckets')
    parser.add_argument('--attack', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--defence', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--overload', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--pressure', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--timing', action='store_true',
                        help='Measure per-module evaluation cost instead of scale and influence')
    parser.add_argument('--timing-positions', type=int, default=150,
                        help='Positions timed per phase')
    parser.add_argument('--output', type=Path, help='Write the full report as JSON')
    args = parser.parse_args()
    config = SearchConfig.from_file(args.config) if args.config else SearchConfig()
    overrides = {field: value for field, value in
                 (('attack_enabled', args.attack), ('defence_enabled', args.defence),
                  ('overload_enabled', args.overload), ('pressure_enabled', args.pressure))
                 if value is not None}
    config = replace(config, **overrides)
    if args.timing:
        buckets = defaultdict(list)
        for state, side in sample_positions(IntransitiveGame(), args.games, args.plies,
                                            args.seed, args.capture_bias):
            phase = phase_of(int(np.count_nonzero(state[:, :, 0])))
            if len(buckets[phase]) < args.timing_positions:
                buckets[phase].append((state, side))
            if all(len(buckets[name]) >= args.timing_positions for name, _ in PHASES):
                break
        buckets = {name: buckets[name] for name, _ in PHASES if buckets[name]}
        report = dict(config=config.to_dict(),
                      corpus=dict(games=args.games, plies=args.plies, seed=args.seed,
                                  capture_bias=args.capture_bias,
                                  positions={k: len(v) for k, v in buckets.items()}),
                      timing=timing_report(config, buckets))
        print(render_timing(report['timing']))
    else:
        report = audit(config, games=args.games, plies=args.plies, seed=args.seed,
                       seconds=args.seconds, decisions=not args.no_decisions, modules=args.modules,
                       search_depth=args.search_depth, search_seconds=args.search_seconds,
                       max_positions=args.max_positions, capture_bias=args.capture_bias)
        print(render(report))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
        print(f'\nwrote {args.output}')


if __name__ == '__main__':
    main()
