# Standalone alpha–beta player

`AlphaBetaPlayer` is an Arena-compatible opponent with the unchanged version-1
rules, state and 648 action slots. It supports physical states through `choose`
and canonical current-player-zero states through `play`. No network is required.
The full design remains in [README.md](README.md).

## Play and inspect

Use the environment described in [the game setup instructions](../README.md).
From the repository root:

```sh
python pit.py intransitive human alphabeta --ab-depth 3 --ab-time 1
python pit.py intransitive alphabeta greedy -n 2 --ab-config intransitive/heuristics/configs/core.json
python pit.py intransitive human alphabeta --ab-attack --ab-defence --no-ab-overload
python -m intransitive.play --opponent alphabeta
python -m intransitive.play --checkpoint /path/to/frozen.pt --opponent alphabeta
```

In the browser, select Alpha–beta, set Attack/Defence/Overload independently,
and press **New game**. **Core only** clears all three. Choose either colour;
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
version and budget. CLI overrides apply after loading a config. Assigning a new
immutable config clears the transposition table on the next search.

## Search and budget contract

The independent exhaustive max/min reference and full-width negamax alpha–beta
use identical leaf evaluation, terminal checks and proof settings. No moves are
removed by heuristic pruning. Ordering prefers exact wins (including stalemate),
plausible goal defences, a previous-depth/TT move, captures and goal progress.
All legal quiet moves remain available. Iterative deepening commits only complete
iterations; an interrupted iteration returns the last complete move/score/PV.
If none completed, the first legal action is returned with `score: null` and
`completed_depth: 0`. Terminal roots reject move selection.

`node_limit` caps **work units**, not just tree nodes. Each tree/proof visit,
transition, ordering item and interceptor check costs one unit; each occupied-board
BFS costs 81 units (its maximum number of expanded squares). `nodes`,
`proof_nodes` and `work` are reported separately. Route, feature and proof work
share the same deadline. Disabled optional modules perform no analysis and have
zero calls and terms. Shared routes are required by the core race module.

Time checks are cooperative between bounded operations. JIT compilation and OS
scheduling can exceed a very small first-call deadline; warm the rule, route and
search kernels before steady-state latency measurements. The benchmark does so.
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

The organizational reference is [Stockfish search.cpp](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp):
iterative deepening, completed-result retention, TT bounds and mate conversion.
This implementation is written independently; chess-specific pruning is absent.

## Feature definitions and limits

A type `t` captures `t % 3 + 1`; its predator is `(t + 1) % 3 + 1`, with the engine's
encoding rock=1, scissors=2, paper=3. There is no absolute type ranking.

| Module | Raw own-side feature | Weight |
| --- | --- | ---: |
| Piece count | Number of surviving pieces | 100 |
| Clear run | Best bounded timed-route estimate, below | 40 |
| Piece advantage | Sum over surviving types of `n * (1[predators=0] + 0.5/(1+predators) + 0.25*prey/(1+prey))` | 25 |
| Attack | Safe next-step goal progress, capped at 2, plus at most one safe immediate capture | 12 |
| Defence | Timely interception duties, deduplicated per runner; safe same-type goal-hold credit; total capped at 4 | 10 |
| Overload | Distinct overloaded defenders, capped at 2 | −5 |

Each weighted contribution is `weight * (own - opponent)`. The total is clamped
to ±10,000. A proven win is `100000 - plies`; a proven loss is `-100000 + plies`.
A real engine draw has utility zero. These values never alter game rewards or
neural training labels. The overload penalty is at most 10 points per side,
compared with 100 per piece. Feature totals and weighted terms are logged separately.

The scarcity formula includes both an explicit last-predator threshold and a
smooth bonus. A threshold-only ablation sets `predator_scarcity_bonus` to zero.
No relevant surviving pieces means no bonus. An uncapturable piece may still be
blocked or lose the competing race; its matchup bonus is never a win certificate.

### Route estimate versus proof

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

The race estimate is `clock_factor * race_factor / ((1+arrival)*(1+interceptors))`.
The clock factor is 1 if arrival is within the remaining no-capture clock,
otherwise 0.25; the competing-race factor is 1/0.5/0.25 for earlier/equal/later
arrival than the opponent's best route **of any type**. Missing routes score zero.
Repetition and possible captures/detours remain unresolved by this estimate.
Every candidate explicitly reports `status: unknown`.

A separate bounded adversarial proof search uses only actual full-state legal
transitions and terminal rules. Horizon leaves are unknown, never exact draws.
A mate score requires a proven terminal outcome against every relevant defence,
including opponent corner wins and modelling draws. Proof node exhaustion returns
unknown; exhaustion of the shared budget interrupts the whole iteration. Proof
work has independent depth/node caps and is charged to the main budget. Search
leaves use a completed proof before ordinary features. Exact terminal handling
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
```

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
