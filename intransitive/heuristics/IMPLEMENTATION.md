# Standalone alpha–beta player

`AlphaBetaPlayer` is an Arena-compatible opponent with the version-2
rules, state and 648 action slots. It supports physical states through `choose`
and canonical current-player-zero states through `play`. No network is required.
The full design remains in [README.md](README.md).

## Play and inspect

Use the environment described in [the game setup instructions](../README.md).
From the repository root:

```sh
python pit.py intransitive human alphabeta --ab-depth 3 --ab-time 1
python pit.py intransitive alphabeta greedy -n 2 --ab-config intransitive/heuristics/configs/core.json
python pit.py intransitive human alphabeta --ab-config intransitive/heuristics/configs/time-first.json
python pit.py intransitive human alphabeta --ab-attack --ab-defence --no-ab-overload
python -m intransitive.play --opponent alphabeta
python -m intransitive.play --checkpoint /path/to/frozen.pt --opponent alphabeta
```

In the browser, select Alpha–beta, set Attack/Defence/Overload independently,
and choose maximum depth (0–64 plies), time per move in seconds, and work limit
per move. These fields start with the server's configuration, including any
`--ab-config` file. Search stops at the first limit reached; work includes
heuristic calculations as well as search visits. A zero limit uses a legal
fallback. **Time first · 5 seconds** sets depth 20, five seconds and a
1,000,000,000-work safety cap; it does not disable the cap or alter evaluation
toggles. Press **New game** to apply changes; the current game's settings remain
fixed until then. **Core only** clears the three evaluation toggles and leaves
search limits unchanged. Either preset only changes the pending form. Choose either colour;
Blue always starts. Local, random, greedy and (when a checkpoint is supplied)
model opponents remain available. Expand **Last AI analysis** for the previous
AI root's features, weighted contributions, configuration, proof status and PV.
The analysis describes the position before the AI moved.

```python
from dataclasses import asdict
from intransitive.heuristics import AlphaBetaPlayer, SearchConfig

player = AlphaBetaPlayer(config=SearchConfig(time_limit=1, max_depth=3))
result = player.analyze(player.game.getInitBoard())
print(asdict(result))
```

The JSON examples in [configs](configs/) contain every weight, switch, evaluator
version and budget. [`time-first.json`](configs/time-first.json) is the same
five-second preset for the browser server and record analyser:

```sh
uv run intransitive --opponent alphabeta --ab-config intransitive/heuristics/configs/time-first.json
uv run intransitive-analyze position.pgn --ply 35 --config intransitive/heuristics/configs/time-first.json
```

CLI overrides apply after loading a config. Assigning a new immutable config
clears the transposition table on the next search. Existing defaults and presets
remain unchanged; an explicit lower work cap still wins when reached first.

## Search and budget contract

The independent exhaustive max/min reference and full-width negamax alpha–beta
use identical leaf evaluation, terminal checks and proof settings. No moves are
removed by heuristic pruning. Ordering prefers exact wins (including stalemate),
the incumbent/cached move, previous root results, goal defences, captures and goal
progress. All legal quiet moves remain available. Interrupted iterations retain
fully searched root children and can return a better move from that partial
iteration. Incomplete children are discarded, and a proved losing incumbent can
yield to an unrefuted alternative. With no completed child, a legal fallback has
`score: null`. `completed_depth` and `selected_depth` distinguish full iteration
coverage from the chosen branch's depth. Terminal roots reject move selection.
See [ANYTIME_SEARCH.md](ANYTIME_SEARCH.md) for result fields and draw-safe folding.

Enhanced ordering additionally keeps killers and history, and can keep counter
moves and continuation history; a moveless deep node can buy an ordering move
with a shallow search. All of that is opt-in and documented in
[MOVE_ORDERING.md](MOVE_ORDERING.md), together with the first-move cutoff rate,
mean cutoff index and per-depth cutoff distribution reported in
`SearchResult.ordering`.

Each result reports `stop_reason` (`time`, `work`, `maximum_depth`, or
`proven_result`), `effective_limits`, and `diagnostics_status`. Time is checked
before work on each charged operation and therefore wins a simultaneous
observation. Diagnostics skipped after search are reported separately and do
not make a maximum-depth or proven-result search look budget-stopped. These
fields are retained in the exported `LastAI` record beside completed, selected
and partial depth.

`node_limit` caps **work units**, not just tree nodes. Each tree/proof visit,
transition, ordering item, immediate-win probe, bounded no-win check and interceptor check costs one unit; each occupied-board
BFS costs 81 units (its maximum number of expanded squares). `nodes`,
`proof_nodes` and `work` are reported separately. Route, feature and proof work
share the same deadline. Disabled optional modules perform no analysis and have
zero calls and terms. Core-only search counts pieces without constructing route
maps. Interception, safety and shortest-route results are cached within each
immutable position when enabled modules need them. A cache hit does not repeat
or charge the underlying traversal.

Search calls the scalar `Evaluator.score`; it never builds diagnostic race
explanations at leaves. Root features and weighted terms are computed only when
budget remains after move search; otherwise diagnostics are explicitly deferred.
`Evaluator.explain` and the PGN analyser still provide detailed routes on demand.
This separation preserves the binary clear-run score and all optional-module
formulas. See [PERFORMANCE.md](PERFORMANCE.md) for measured before/after results.

Time checks are cooperative between bounded operations. Rule/search kernels and
enabled route kernels are warmed before the timed search. Initial preparation
still has a startup cost, and OS scheduling can delay a cooperative deadline.
No extra unbudgeted evaluation is performed after a timeout. Memory inspection
and result assembly have small additional overhead included in reported elapsed
time. Table size is capped by `table_entries` (zero disables storage).

TT keys contain every byte of the state, including goal ownership, turn, no-capture
clock and the full repetition history. Entries store depth, score, exact/lower/upper
bound, best move and a legal PV. Only the **same remaining depth** is used for
score reuse; deeper heuristic values are not equivalent to shallower minimax.
Mate scores are converted between root-relative and node-relative distance on
store/probe. Aborted nodes do not publish entries; completed child entries remain
valid. All evaluation/config changes invalidate the table.

Move ordering checks immediate corner and stalemate wins on a temporary board
plane, preserving the previous order exactly. Full history states are then built
only for children actually searched; alpha-beta cutoffs avoid building the rest.
All applied moves still use the validated rules engine.

Before expanding a bounded proof tree, a sufficient no-win check can rule out
terminal victories within that proof horizon. Neither side can reach its target
within its available moves, and each has more than `2 * proof_depth` disjoint
piece/empty-neighbour pairs. Since a move changes at most two squares, at least
one pair per side must survive untouched, guaranteeing a legal move and excluding
stalemate. If either condition fails, the original full proof search runs. This
shortcut awards no win credit, does not declare the position safe beyond the
horizon, and does not prune the main minimax tree.

The organizational reference is [Stockfish search.cpp](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp):
iterative deepening, completed-result retention, TT bounds and mate conversion.
This implementation is written independently; chess-specific pruning is absent.

## Feature definitions and limits

A type `t` captures `t % 3 + 1`; its predator is `(t + 1) % 3 + 1`, with the engine's
encoding rock=1, scissors=2, paper=3. There is no absolute type ranking.

| Module | Raw own-side feature | Weight |
| --- | --- | ---: |
| Piece count | Number of surviving pieces | 100 |
| Clear run | Zero unless a forced win/loss is proved; then decisive score, below | Not weighted |
| Piece advantage | Sum over surviving types of `n * (1[predators=0] + 0.5/(1+predators) + 0.25*prey/(1+prey))` | 25 |
| Attack | Safe next-step goal progress, capped at 2, plus at most one safe immediate capture | 12 |
| Defence | Timely interception duties, deduplicated per runner; safe same-type goal-hold credit; total capped at 4 | 10 |
| Overload | Distinct overloaded defenders, capped at 2 | −5 |

Ordinary weighted contributions are `weight * (own - opponent)`, clamped
to ±10,000. Clear-run scoring is binary: an unproven run contributes zero;
a proven win/loss overrides the sum. A proven win is `100000 - plies`; a proven
loss is `-100000 + plies`. These finite decisive scores serve the role of positive
and negative infinity while preserving mate-distance ordering, transposition
arithmetic and valid JSON. A proved result reports a winning-side clear-run
feature of 1 and a losing-side feature of 0; its decisive term is not clamped.
Other modules are skipped once that proof settles the evaluation.
A real engine draw has utility zero. These values never alter game rewards or
neural training labels. The overload penalty is at most 10 points per side,
compared with 100 per piece. Feature totals and weighted terms are logged separately.

The scarcity formula includes both an explicit last-predator threshold and a
smooth bonus. A threshold-only ablation sets `predator_scarcity_bonus` to zero.
No relevant surviving pieces means no bonus. An uncapturable piece may still be
blocked or lose the competing race; its matchup bonus is never a win certificate.

Version 2 removes the former fractional race bonus. `race_weight` is accepted
only for compatibility and has no scoring effect. Old v1 configuration files load
with their other settings preserved and their evaluator version updated to v2.
The committed v1 benchmark results describe the previous evaluator, not v2.

### Route diagnostics versus binary proof

Each piece gets occupied-board king-move distance maps from its source and target.
Friendly and uncapturable enemies block squares; capturable occupants can be
traversed. Estimates hold other pieces stationary and allow sequential captures,
so they describe plausible routes, not a prediction of future occupancy. A blocked
target is unreachable. The actual target follows the state's goal ownership,
including canonical colour swaps. Scoring considers all squares in the shortest
route DAG; the explanation displays one illustrative shortest route. Its tie
selection can differ under reflection without changing the scalar feature.

An uninterrupted `k`-move arrival is ply `2*k-1` for the current mover, `2*k` for
the other side. Capture interceptors must reach a route square by the reply to
runner arrival, or before arrival at the winning corner. A runner can also be
captured before its first move. Same-type blockers receive coverage only at the
unavoidable goal. Predators of a defender can invalidate its apparent safety.
This is conservative exposure screening with static occupancy, not joint planning.

Routes, arrival times and interceptors are diagnostic information only. Candidate
`estimate` fields are always zero; no partial credit is awarded for proximity,
a favourable race margin or having fewer interceptors. Positional progress belongs
to the optional attack module. Static routes do not settle repetition, possible
captures, detours or competing wins, so candidate routes report `status: unknown`.

A separate bounded adversarial proof search uses only actual full-state legal
transitions and terminal rules. Horizon leaves are unknown, never exact draws.
A mate score requires a proven terminal outcome against every relevant defence,
including opponent corner wins and modelling draws. Proof node exhaustion returns
unknown; exhaustion of the shared budget interrupts the whole iteration. Proof
work has independent depth/node caps and is charged to the main budget. Search
leaves use a completed proof before ordinary features, and the standalone evaluator
uses the same proof logic. Horizon or proof-budget exhaustion yields zero clear-run
credit, not a claim that no winning run exists. Exact terminal handling
always precedes both and respects official-win precedence over draws.

### Defence and overload limitations

Capture coverage means a safe timed intersection exists on a plausible shortest
route; it does not mean every alternate route is covered. Blocking coverage
requires a same-type goal hold, where alternative routes cannot avoid the blocker.
The defender must arrive before the attacker and survive the threat analysis.
Nearby pieces that cannot arrive on time receive no coverage credit.

Overload is analysed only for the side actually on move; it never simulates a
pass. It requires two runners within four moves of the goal whose only potential
responder is the same defender. Every legal first response is simulated, then
routes and coverage are recomputed. A win/draw, a safely removed threat, distinct
timely responders or a safe goal hold removes the conflict. This includes quiet
alternative defences and immediate counter-wins. An enemy paper can capture a
rock blocker, so that rock cannot supply paper-blocking redundancy.

Surviving conflicts are a **small heuristic opportunity cost**, not a forced-loss
proof. The horizon, static threat coverage and on-turn-only convention deliberately
limit the first version. Enabling defence does not enable overload or vice versa.
The fixed overload fixtures test mitigation by a paper/scissors goal defender,
non-mitigation by rock or distant support, and alternate defences that remove it.

## Reproduce validation and comparisons

```sh
python -m unittest intransitive.tests.test_heuristics intransitive.tests.test_play -v
python -m unittest discover -s intransitive/tests -v
python -m unittest discover -s tests -v
python -m intransitive.heuristics.benchmark --output checkpoints/issue34-benchmark \
  --seeds 0 1 --modes nodes wall --depth 2 --nodes 200000 --seconds .1 --simulations 8
python -m intransitive.heuristics.benchmark \
  --verify-report intransitive/heuristics/evidence/comparison.json.gz
python -m intransitive.heuristics.ordering_benchmark --output /tmp/ordering.json \
  fixed --depth 4 --positions 16 --schedules plain lmr lmr-pvs
python -m intransitive.heuristics.ordering_benchmark table /tmp/ordering.json
```

`ordering_benchmark` is the separate ablation for the opt-in ordering
mechanisms: a fixed-depth ladder that reports nodes, first-move cutoff rate,
mean cutoff index and the per-depth cutoff distribution, plus paired equal-time
games. Its corpus comes from its seed, and every row records any enabled
mechanism that then executed zero times. See
[the ordering results](../benchmarks/ordering/README.md).

The benchmark extracts only `on/baseline.pt` from the committed #16 archive into
its output directory, or accepts an explicit frozen `--checkpoint`. It records
the checkpoint SHA-256, every full config, seeds, complete game action traces,
terminal reasons and state hashes. It does not read or change a live training run.
All eight optional-module combinations face random, greedy, frozen neural-MCTS
and core-only alpha–beta as both colours. The candidate and core opponent share
the current comparison budget. Random, greedy and neural settings remain fixed
across configurations; the neural opponent uses eight full simulations, not a
matched wall-time budget. The wall comparison matches **candidate variants** and
the core opponent at 100 ms per move. The diagnostic comparison uses depth two
and 200,000 work units with a 60-second safety deadline.

Tuning fixtures and held-out diagnostic positions are separate in
[positions.json](positions.json). The threshold/scarcity comparison is recorded
only on the tuning fixtures; weights are frozen before the final comparisons.
Games use the official opening, exact draw limits and no extra adjudication.
Both colours use the same seed stream for each opponent/seed pair. Timing can
change search depth and actions even with fixed seeds, so wall-time runs are
statistical reproductions, not byte-identical trajectories.

Reports include W/D/L, score and conservative 95% colour-paired-seed score
intervals, depth, tree/proof nodes, total work, module time, p50/p95 latency and
memory. `max_table_bytes` is a shallow Python table/key/entry/PV footprint estimate;
process peak RSS also includes Python, NumPy, Numba, Torch and ONNX and is not
attributable to a single module. Tiny samples and repeated deterministic opponents
cannot establish general playing strength. See [MEASUREMENTS.md](MEASUREMENTS.md)
for measured recommendations and interactions.
