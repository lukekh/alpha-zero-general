# Intransitive minimax and heuristic evaluation

Design specification for a minimax player with alpha–beta pruning. The first
implementation should provide a useful opponent without a trained network and
make it straightforward to measure whether individual heuristics help or hurt.
The standalone player is implemented. See [implementation, formulas and runnable commands](IMPLEMENTATION.md)
and [measured ablation results](MEASUREMENTS.md). The default bounded proof uses
[compiled traversal and last-ply specialisation](../benchmarks/proof/README.md),
with unchanged accounting and a reference fallback for larger proof budgets.
Core scalar leaves use [incremental counts and exact material reuse](../benchmarks/material/README.md),
preserving board-order floating-point sums and the public detailed evaluator.
Independent [PVS and aspiration experiments](../benchmarks/windows/README.md)
preserve exact-depth values and safe anytime selection. Both remain opt-in;
the report includes the archived baseline and measured retain/reject decision.
An opt-in [square-ring pressure experiment](../benchmarks/pressure/README.md)
adds blocker-blind RPS threats discounted by nearby defenders. It is disabled
by default and is not enabled in the running supervised generators.
An optional [variable-material mode](VARIABLE_MATERIAL.md) values each piece
using the opposing prey/predator ratio and its scarcity in the own army. It is
available in Python and Rust, with BASE 100 and REG 0.25.
The compact Python search and its compiled proofs use incremental legality
bitboards. See the [Python/Rust node-throughput measurements](../benchmarks/legal_moves/README.md).
Compiled legacy move ordering is now the default; enhanced ordering remains opt-in.
[Further search efficiencies and measurements](../benchmarks/search_efficiencies/README.md)
cover reusable proof storage and removal of redundant search work.
The forced corner-run certificate can also be used as an interior search bound,
with a guard that exempts certified branches from pruning and admissible
race/horizon reductions. See the [contract](CERTIFICATE_SEARCH.md) and its
[bounded validation](../benchmarks/certificate/README.md); every flag is opt-in
and defaults to false. That work also fixed a missing `pv` key which made
`certificate_enabled` raise as soon as a leaf certified.
The original design follows below.

Tracked in [implementation issue #34](https://github.com/lukekh/alpha-zero-general/issues/34),
a focused follow-up to [training-efficiency issue #33](https://github.com/lukekh/alpha-zero-general/issues/33).
The [game rules and modelling-only draw limits](../README.md#board-and-players)
remain authoritative. Reuse the existing exact rules engine and action encoding.

The opt-in [versioned module-scale contract](TUNING.md) provides validated genomes,
backend capability manifests and saturation reporting for future tuning experiments.
It does not change these defaults or start a tuning run.

## Evaluation modules

The first three features form the core evaluation. The remaining features are optional,
independent modules, initially disabled to establish a core-only baseline.
Each module has a configurable nonnegative weight and reports its contribution.

| # | Module | Purpose | Baseline |
| --- | --- | --- | --- |
| 1 | `piece_count` | Reward a numerical material advantage. | Enabled |
| 2 | `clear_run` | Binary: a proved winning run is decisive; otherwise contributes zero. | Enabled |
| 3 | `piece_advantage` | Reward favourable rock/paper/scissors matchups, especially pieces with no remaining predator. | Enabled |
| 4 | `attacking_position` | Reward useful progress toward the goal and accessible capture opportunities. | Optional, disabled |
| 5 | `defensive_position` | Reward defenders that can intercept threats while remaining safe themselves. | Optional, disabled |
| 6 | `overload` | Apply a small penalty when one defender cannot cover multiple threats. | Optional, disabled |
| 7 | `local_pressure` | Square-ring RPS attacks/threats discounted by the victim's defenders. | Experimental, disabled |

### 1. Piece count

Start with one unit for every surviving piece, regardless of type. The material
term is `own_piece_count - opponent_piece_count`. A capture therefore improves
material, but a terminal win or loss always takes precedence over material.

Keep raw count separate from piece-type advantage and positional bonuses so that
we can explain why a move scores well. Do not introduce a fixed chess-like ranking
of rock, paper and scissors: each type captures one type and loses to another.

### 2. Early win calculation from a clear run

Identify pieces with a viable route to the opponent's defended corner. This is a
binary win calculation, not a graded positional bonus: a proved winning run is
decisive, and an unproven run contributes zero. The intended example is a paper runner closer to the
opponent's corner than opposing paper or scissors, provided no opposing piece of
any type can win its own race first.

Distance alone is a useful screening feature, not a sufficient proof. Evaluate:

1. **Route:** legal king moves to the actual target corner, accounting for friendly
   pieces, uncapturable enemy pieces and the corner's occupant. Chebyshev distance
   `max(abs(dx), abs(dy))` is a lower bound on moves in an empty board, not the
   guaranteed travel time through an occupied board.
2. **Interception:** whether an opponent can capture the runner at any point on its
   route, rather than only whether that opponent is close to the corner. An enemy
   scissors threatens a paper runner; an enemy paper can block it. Enemy rocks
   are capturable by paper, but still occupy squares and can pursue their own win.
3. **Tempo:** whose turn it is, and the alternating move schedule. In an
   uninterrupted race requiring `k` moves, a piece moving on the current turn
   finishes on ply `2*k - 1`; a piece whose side moves next finishes on ply `2*k`.
   Recompute timing if a capture, detour or defensive move is required. Players
   cannot pass, and reaching the goal wins immediately, before any later reply.
4. **Competing race:** the opponent's earliest feasible win with any piece type,
   aimed at our corner. Being ahead of enemy paper/scissors at their corner does
   not exclude an enemy rock reaching our corner first.
5. **Other termination:** exact repetition/no-capture history and possible
   stalemate. A race must finish before a modelling draw, except that an official
   win takes precedence when the events coincide under the existing rules.

Keep diagnostics separate from scoring:

- **Route diagnostics**, including the candidate runner, route, arrival estimate
  and identified interceptors. These contribute zero to the score.
- A **proven forced result**, only when a conservative sufficient-condition check
  or bounded adversarial search establishes success against every legal defence.
  A single plausible route is not enough. Proof search must use the complete
  state and account for opponent wins, not only capture/block responses. If its
  budget expires, return unknown and contribute zero for clear-run scoring.

Only a proven forced result may produce a mate-like score or an exact search
cutoff. Turning off optional positional modules must not disable exact terminal
checks or make unproven distance estimates count as wins.

The implementation uses the finite win value (`100000`, adjusted for proven
distance to termination) as the equivalent of infinity. It outranks every ordinary
heuristic and keeps alpha–beta arithmetic and JSON output well-defined. A zero
clear-run score means no win was established within the proof budget, not that
every possible longer winning route has been disproved.

### 3. Piece advantage

Value the distribution of types in addition to total material. Use the actual
capture cycle:

| Our type | Captures | Captured by | Same-type interaction |
| --- | --- | --- | --- |
| Rock | Scissors | Paper | Cannot capture; can block |
| Paper | Rock | Scissors | Cannot capture; can block |
| Scissors | Paper | Rock | Cannot capture; can block |

For each type, consider the number of our surviving pieces, the number of opposing
predators and the number of opposing pieces they can capture. Give an explicit
bonus when the opponent has no surviving predator for that type. Test a gradual
predator-scarcity bonus as well as this zero-predator threshold; document the
chosen formula and weights rather than hiding them in search code.

For example, after eliminating all enemy paper, our rocks can no longer be
captured. Keeping all our rocks then provides a substantial advantage, especially
against enemy scissors. Equal total piece counts can still have very different
matchup values. The bonus should scale with our surviving relevant pieces and
not assign an advantage to a type we no longer have.

An uncapturable piece is not automatically unstoppable: same-type pieces can
block it, friendly pieces can obstruct it, and an opponent can win at the other
corner. This is a material-matchup bonus, not a forced-win declaration. Keep it
separate from actual interception and travel-time features.

### 4. Optional attacking positional advantage

Reward pieces for useful proximity to the opponent's goal, access to viable
approach routes and capture opportunities against types they can capture. Account
for whether an attack is legal/reachable and for exposure to an enemy predator.

Examples to compare include safe goal progress, threats that require a defensive
reply and reachable captures. A piece should not score highly merely because it
is geometrically close to a target behind an uncapturable blocker. Avoid repeatedly
rewarding the same opportunity for every nearby piece without accounting for
mutually exclusive moves.

Keep this term smooth and bounded. Module 2 is a binary win calculation;
this optional module evaluates broader attacking opportunities. Log contributions
separately to detect double counting.

### 5. Optional defensive positional advantage

Reward a piece that can respond to several enemy pieces it can capture, especially
when it lies between those threats and our defended corner. Interpret “between”
using reachable attack routes and interception times, rather than only a straight
line or geometric midpoint.

A valuable defender should be able to reach an interception or blocking square
before the attacker passes it, protect a relevant approach to the corner, and
avoid being captured by an enemy predator. Greater separation from enemy predators
is useful when it improves defensive safety; raw distance alone is not protection.

Distinguish **capture defence** from **blocking defence**. A piece can sometimes
hold a square against a type it cannot capture, but only if that attacker cannot
capture it either and cannot win via another route. Consider the legal timing and
occupancy of the defended square, not just the existence of a nearby friendly piece.

Multiple capture opportunities increase a defender's potential value. Module 6
then corrects the case where those duties cannot all be fulfilled in time.

### 6. Optional overloading disadvantage

Apply a slight penalty when a defender is responsible for multiple credible threats
and responding to one lets another attacker make a clear run to our corner. This
is an opportunity-cost penalty for conflicting defensive duties, not a penalty
for every defender near two enemy pieces.

A first implementation can examine pairs of threats:

1. Identify attackers with plausible timed routes to our goal and defenders that
   could intercept or block each route.
2. Check whether the same defender is essential to both responses. Simulate a
   short legal chase or necessary defence against one threat and recompute
   whether the second route remains covered. “Forced” requires considering valid
   alternative defences, counterattacks and our own immediate wins.
3. Penalize uncovered alternatives only when their timing makes the conflict
   credible. Cap and deduplicate the penalty so many overlapping threat pairs do
   not swamp material or a proven win.
4. Reduce or remove the penalty if another defender can actually take over before
   the second attacker arrives. A nearby same-type defender or a safe goal blocker
   can provide redundancy, but proximity without timely coverage is insufficient.

Example: one friendly scissors faces two enemy paper runners. Chasing one paper
may leave the other free to reach our corner. A second friendly scissors can
cover the other runner; a friendly paper may also block its route or the corner,
because paper cannot capture paper, if it can arrive in time and survive other
threats.

A friendly rock is **not** a safe blocker of an enemy paper: paper captures rock.
Rock support must be evaluated according to its actual role. For example, a rock
may block an enemy rock that threatens our defending scissors. Do not award a
generic support bonus just because a rock is nearby. Similarly, being uncapturable
by one attacker does not make a blocker safe from that attacker's other pieces.

## Score composition and configuration

For a selected player `p`, define each module's feature from that player's
perspective and combine own and opponent values consistently:

```text
evaluation(state, p) =
    w_count     * (count(p)     - count(opponent))
  + w_advantage * (advantage(p) - advantage(opponent))
  + attack_enabled  * w_attack  * (attack(p)  - attack(opponent))
  + defence_enabled * w_defence * (defence(p) - defence(opponent))
  - overload_enabled * w_overload * (overload(p) - overload(opponent))
```

Clear-run proof overrides this sum with a decisive win/loss score; otherwise it
contributes zero. `race_weight` is a deprecated, ignored compatibility setting.
Use terminal/proven-result handling before this weighted sum. Clamp ordinary
heuristic totals below the forced-result range; prefer shorter proven wins and
longer forced losses. Do not invent exact win distances from a heuristic estimate.
Actual draws use the existing draw semantics and map to neutral search utility.
Keep heuristic evaluation separate from game rewards and training outcome labels.

Expose a serializable configuration with an evaluator version, all weights,
independent booleans for modules 4–6, and search/proof budgets. The core preset has
all optional booleans false. Weights require measurement; a large feature range
must not accidentally make the intended slight overload penalty dominant.

The opt-in [`time-first.json`](configs/time-first.json) preset keeps core-only
evaluation, searches to at most depth 20 for five seconds, and retains a
1,000,000,000-work safety cap. This high cap avoids the measured premature
200,000-work stop without redefining zero, changing the existing presets, or
preventing callers from selecting a lower explicit cap. The comparison and
replay fixture are in [the search-budget report](../benchmarks/search_budget/README.md).

Provide an explanation containing raw own/opponent features, weighted terms,
race proof/unknown status and selected principal variation. Disabling a module
must skip its computation as well as its contribution. Record configuration in
results and invalidate evaluation-dependent caches when it changes.

The scalar terms above say how much each module contributed, not where it came
from or whether it still responds to the board. [Attribution and
audit](ATTRIBUTION.md) adds both: `attribution.py` decomposes every module into
signed per-square contributions that must sum back to its own term (the browser
and the tests both check that residual), and `audit.py` measures each module's
magnitude, dead fraction, binding caps and one-ply decision influence by game
phase. See also the observed [module scales](MODULE_SCALE.md).

### Shared route maps

Route work is indexed by piece **code**, not by piece. The BFS traversal rule —
enter a square if it is empty or holds a capturable enemy — depends only on the
moving code, so every piece of one code shares a frontier.

`Geometry.threat_map(code)` runs one multi-source search per predator type and
returns the earliest ply any predator of that code reaches each square. Because
`arrival` is monotone in distance, the minimum over predators of their arrival
is the arrival of the nearest one, so `safe` is an array comparison rather than
a scan of every enemy for every piece, square and deadline.
`Geometry.profile(runner)` hoists a runner's interception squares, deadlines and
arrival plies out of the per-defender loop, since all three depend only on the
runner. Both are built on demand: a sparse endgame never pays for maps or
profiles nothing asks about, which an eager version made measurably slower.

Neither charges the budget. Together they cost at most `81 * min(6, pieces)`
against the `162 * pieces` the piece index already charges, and they replace
per-enemy scans that were never charged, so a fixed work budget still buys the
same search. Equivalence with the per-piece scans is asserted in
[`test_geometry.py`](../tests/test_geometry.py), which keeps the replaced code
as the reference implementation.

Route maps themselves are a bitboard flood fill ([`flood.py`](flood.py)) over
the same 81-square, two-lane `uint64` layout [`moves.py`](moves.py) already
maintains. A search ring is one dilation — eight masked shifts of the frontier —
intersected with the squares that code may enter, so passability is decided once
per position instead of re-derived for every neighbour visited. The same fill
serves the single-source per-piece maps and the multi-source threat maps.

The two positional features are compiled ([`features.py`](features.py)) and read
`Geometry`'s stacked `(pieces, 81)` route blocks with no repacking. Beyond
compilation, each drops work the reference could not:

- **Attack** only ever matches at distance one, so it walks the eight
  neighbours rather than the shortest-path DAG and the enemy list. Its capture
  half is a boolean, not a count: `opportunities` holds one per distinct square
  under `min(1., sum)`, so it is one exactly when some piece has a safe adjacent
  capture.
- **Defence** builds each runner's interception squares and deadlines once and
  shares them across every defender, as one matrix pass instead of a call per
  pair. `coverage` reads that matrix; `Geometry.intercepts` remains the source
  for the per-square detail the race diagnostics print.

`evaluation` keeps both readable definitions as `attacking_position_reference`
and `coverage_reference`, and [`test_features.py`](../tests/test_features.py)
asserts the compiled versions match them in value *and* in charged work.

## Minimax with alpha–beta pruning

Implement a depth-limited exhaustive minimax reference, followed by an equivalent
negamax/alpha–beta search over the same evaluator and legal game transitions.
Use iterative deepening with a maximum depth and explicit node/time limits.
Retain completed root branches from interrupted iterations, comparing them at
the same depth after rechecking the incumbent. Fall back to the last completed
iteration or a legal move when no newer branch completes. Count heuristic
analysis and race-proof work in the budget; run diagnostics after move selection.

Order moves using exact immediate wins, promising defensive replies, captures,
the previous principal variation and cached best moves. Ordering must retain all
legal moves. In particular, captures are optional and a quiet move to a corner can
win. Do not add speculative selective pruning to the first correctness baseline.

A transposition table must distinguish future-play states, including side to move,
goals, repetition occurrence counts and the no-capture clock. Equivalent history
orders and different total move numbers can share entries. Store depth, score, bound type
(exact/lower/upper) and best move. Include evaluator/config identity or clear the
table when configuration changes. Normalize root-relative mate distances when
reusing entries, and do not treat a bound or incomplete proof as an exact result.

Reuse `IntransitiveGame`, `Board` and the current fixed action slots. Search and
evaluation must not mutate the caller's state. Maintain perspective consistently
through canonical colour swaps; do not assume canonicalization always puts the
goal at the same physical corner. Evaluation and selected moves should behave
consistently under valid board/colour symmetries, allowing equivalent tied moves.

Use [Stockfish's search implementation](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp)
as a primary implementation reference for search organization, not as a source of
game-specific pruning assumptions. The proposed Intransitive feature definitions
above are hypotheses to validate in this game.

### The clear-run certificate

`prove` searches two plies. A runner that cannot be stopped often wins further
out than that, so [`clear_run.py`](clear_run.py) certifies those directly:
a layered search in which a square is usable only when every enemy's earliest
possible arrival is later than the runner's stay there. Opt in with
`certificate_enabled`; it is off by default.

Every bound in it errs toward silence:

- Enemy arrival uses **free-board Chebyshev distance**. Occupancy can only slow
  an enemy, so a square this calls contested may really be safe, never the
  reverse.
- **Blocking and capturing count alike.** Any enemy able to stand on a square in
  time disqualifies it, not only one that could capture there.
- The runner **walks only empty squares**, so its route never depends on an
  exchange going as hoped.
- It must **arrive strictly first**, ahead of any enemy reaching their own
  corner on the same lower bound.
- The **draw clock must not expire**: the run makes no captures, so the
  no-capture counter runs its whole length. Repetition cannot occur, because the
  runner's distance to the goal strictly decreases every move.
- The runner's **own starting square** is checked too. When the opponent moves
  first they get a ply to capture it where it stands — the hole that produced
  every false positive in the first version.

A negative answer means only that this argument did not apply, never that the
position is not winnable.

Measured on a corpus of random reachable positions: it proves a win on about
17% of endgames and 1% of middlegames, a median of four plies out — distances
the two-ply terminal search structurally cannot reach. Of 1,244 claims
re-searched full width to the claimed depth, **0 were refuted and 0 were slower
than claimed**. Python and Rust certify the same 492 of 492 positions.

The counterexample harness is the point, not the passing run: the first version
claimed a win on 6 of 1,283 positions (0.47%), all of them "mate in two", which
is only reachable when the opponent moves first — the missing starting-square
check.

### Unproven runner pressure

`runner_pressure` is the same idea priced as an ordinary weighted module rather
than a proof: for the piece of each type nearest the goal, count the enemies
that could answer it inside the box it spans with the goal, and score on
distance when none can, on distance plus one when exactly one can, and zero
otherwise. It is off by default (`runner_enabled`).

It is deliberately approximate, and wrong in the ways the certificate is careful
about — a piece outside the box can step in, any piece can block rather than
only a capturing one, and the opponent may simply win the race. That is exactly
why it is a weighted term and not a proof: a mis-scored heuristic is corrected
by deeper search, whereas a false proof emits a mate score and prunes the line
that would refute it. It is not part of the evolved genome; run the audit's
`flips` and `agree+` columns first to see whether it earns a coefficient.

### Gating the proof

`prove` is a bounded terminal-only search, so most positions cannot contain a
result for it to find. `no_terminal_win_in_horizon` decides that up front, and
returning true there costs one node instead of the configured allowance.

It is a *sufficient* condition for "no terminal result within the horizon" —
never a claim that one exists — built from a corner clause and a stalemate
clause. Chebyshev distance rules out reaching either winning corner. Stalemate
is ruled out by whichever of two independent counting arguments applies:

- **The pair rule** (`crowded_side_has_moves`): more than `2*depth` disjoint
  (own piece, empty adjacent square) pairs. A move changes at most two squares,
  so `depth` plies damage at most `2*depth` pairs and one survives.
- **The spacious rule** (`open_side_has_moves`): more than `depth` pieces with
  more than `depth` empty neighbours. A ply adds at most one occupied square,
  because captures only remove pieces and each ply vacates only the square it
  moves from; and a piece leaves its square only by moving or being captured,
  which happens to at most `depth` of them across the horizon.

The pair rule needs `2*depth+1` pieces a side, which a sparse endgame simply
does not have — it was measured failing on 100% of endgame positions for that
reason alone, never the corner clause. The spacious rule covers exactly that
case. A side is cleared if either holds, so the gate only ever widens.

Neither argument can clear a side down to two pieces at depth two, and that is
correct rather than weak: both could genuinely be captured inside the horizon.

Measured effect, with every module enabled and the proof kernels warm:

| Phase | gate fires before | after | proof cost before | after |
| --- | ---: | ---: | ---: | ---: |
| opening | 100% | 100% | 5.3 us | 5.3 us |
| middlegame | 66% | **100%** | 22.2 us | **5.1 us** |
| endgame | 0% | **39%** | 162.6 us | **64.4 us** |

Endgame evaluation falls from 265 to 165 microseconds. Scores are unchanged
everywhere; what changes is the proof diagnostic, where 211 of 800 corpus
positions move from `reason: proof budget` (the search gave up at its node
cap) to `reason: horizon` (nothing can happen). Work counts drop, which is the
point of a gate — unlike the route and feature work, this one does let a
work-limited search go further.

In Rust the same gate halves proof nodes but barely moves wall time, because a
native proof node already costs around 0.15 microseconds; the win there is node
budget rather than seconds.

[`test_proof_gate.py`](../tests/test_proof_gate.py) runs an ungated,
full-strength proof on every position the gate skips and asserts it finds
nothing; `cargo test --release` does the same against an exhaustive two-ply
expansion.

## Validation and comparisons

[Search under a deadline](ANYTIME_SEARCH.md) explains retained partial-iteration
work, move prioritisation, and transposition reuse that preserves repetition rules.

Per-module attribution is checked by summation, not by inspection: every
module's per-square contributions must add up to the evaluator's own term
across a corpus of random reachable positions, for the core, all-modules,
variable-material and signed-coefficient configurations, from both
perspectives. Run it with:

```sh
uv run --locked python -m unittest intransitive.tests.test_attribution -v
```

The implemented [100-position tactical regression suite](TACTICS.md) checks
exact bot moves for immediate wins, clear runs, mandatory goal defences, forced
escapes, and safe captures. Each expected move is independently certified using
the Python reference rules. Another [12 game-blunder regressions](GAME_REGRESSIONS.md)
cover deeper forks, exchanges, and abandoning the last goal defender. Run both with:

```sh
uv run --locked python -m unittest intransitive.tests.test_tactics intransitive.tests.test_game_blunders -v
```

Add targeted positions and assertions for:

- Material changes after a capture, and piece-type advantage at equal total counts.
- Removing the last enemy paper improving our rocks' matchup value; removing all
  our rocks eliminating their bonus; analogous cases for the other types.
- A genuine clear-run win; an apparently clear route intercepted on the way;
  same-type blockade; a capturable blocker; a blocked corner; a faster opponent
  runner of another type; equal geometric distances with different turns to move.
- No-capture/repetition draws before a race finishes, and official-win precedence.
- Useful attacking progress versus an unsafe advance or unreachable capture.
- A timely defender between attacker and goal versus a nearby but late defender;
  safety from predators and distinctions between blocking and capturing.
- One scissors overloaded by two papers; mitigation by a timely second scissors
  or paper blocker; no mitigation from an exposed rock in the paper's path;
  support too far away to matter; an alternate defence that removes the overload.
- All eight enable/disable combinations of modules 4–6; disabled terms contribute
  zero and perform no analysis; consistent signs, bounded scores and explain output.
- Alpha–beta matching exhaustive minimax scores at shallow depth; cached/uncached
  agreement with different histories; state preservation; colour/symmetry handling;
  timeout fallback and completed-depth reporting; unknown proof on budget exhaustion.

Measure the core preset, each optional module individually and all combinations.
Use fixed test positions and balanced games as both colours against existing
random, greedy, frozen neural-MCTS and core-only alpha–beta opponents. Hold rules,
opponents and seed policy fixed. Keep tuning positions separate from final tests.

Compare at both equal depth/node settings for diagnosis and **equal wall time per
move** for practical strength: a more expensive heuristic can improve evaluation
and still weaken play by reducing search depth. Report wins/draws/losses, score,
uncertainty, depth, nodes, per-module evaluation time, p50/p95 move latency and
memory. Include interactions between optional modules, especially defensive value
and overload correction. A module that fails to help can remain disabled.

## Delivery

- Implement the search player and six separately inspectable evaluation modules.
- Expose it in the terminal player interface and browser opponent selector, with
  independent toggles for attack, defence and overload, plus a core-only preset.
- Retain the existing greedy and neural opponents for controlled comparisons.
- Publish runnable configuration examples, test fixtures, bounded benchmark
  commands and measured recommendations for optional modules and weights.
- Keep use as a training teacher or neural-search hybrid as follow-up work under
  #33; this issue first establishes the standalone heuristic player.

Done means a correct playable search implementation with reproducible comparisons
and documented module effects, including negative results. This design does not
change the active training run, official rules or modelling draw limits.

Experimental [guarded null move and forward futility pruning](SELECTIVE_SEARCH.md)
are independently configurable in Python/native search and default off. Their
completed depths are selective; they are not exhaustive labels or certificates.
Enabling one with evaluator scales its margins are not calibrated for is a
configuration error rather than a silent no-op, and each search reports which
declared technique could and did execute. The
[activation measurement](../benchmarks/selective_activation/README.md) records
the configuration in which they first run and attributes a node count to each.
Issue #68 adds razoring, reverse futility and move-count pruning on the same
guard (Python only, also off by default), plus value-preserving mate-distance
pruning, which keeps ordinary bounds and certificates. The
[bounded validation](../benchmarks/shallow_pruning/README.md) recommends
adopting none of the three heuristic members as defaults.

See [variable-value MVV-LVA ordering](MVV_LVA.md) and the [module-scale example](MODULE_SCALE.md).

The cyclic capture rule has no least-valuable-attacker order, so the
[static exchange evaluation](EXCHANGE.md) is a different algorithm rather than a
port: the recapturing kind is forced by the cycle and the series terminates on the
target's neighbourhood. It supplies an optional capture-ordering key and two
optional quiescence filters, alongside quiescence delta pruning whose margin is
stated in evaluator units. All of them default off; the
[bounded measurements](../benchmarks/exchange/README.md) carry the adoption decision.
