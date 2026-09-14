# Intransitive minimax and heuristic evaluation

Design specification for a minimax player with alpha–beta pruning. The first
implementation should provide a useful opponent without a trained network and
make it straightforward to measure whether individual heuristics help or hurt.
The standalone player is implemented. See [implementation, formulas and runnable commands](IMPLEMENTATION.md)
and [measured ablation results](MEASUREMENTS.md). The default bounded proof uses
[compiled traversal and last-ply specialisation](../benchmarks/proof/README.md),
with unchanged accounting and a reference fallback for larger proof budgets.
The original design follows below.

Tracked in [implementation issue #34](https://github.com/lukekh/alpha-zero-general/issues/34),
a focused follow-up to [training-efficiency issue #33](https://github.com/lukekh/alpha-zero-general/issues/33).
The [game rules and modelling-only draw limits](../README.md#board-and-players)
remain authoritative. Reuse the existing exact rules engine and action encoding.

## Evaluation modules

The first three features form the core evaluation. The last three are optional,
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

## Validation and comparisons

[Search under a deadline](ANYTIME_SEARCH.md) explains retained partial-iteration
work, move prioritisation, and transposition reuse that preserves repetition rules.

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
