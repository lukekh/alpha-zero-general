# Static exchange evaluation for cyclic captures

Issue #67. This document specifies the exchange series **before** the
implementation, as the issue requires, and states where the result is exact and
where it is wrong. Every switch described here defaults to false.

Intransitive's capture rule is cyclic — rock takes scissors, scissors takes
paper, paper takes rock — so the standard chess SEE recipe ("resolve the square
by repeatedly taking with the least valuable attacker") has no meaning: there is
no total order on piece values that decides who wins an exchange. What follows
is not a port of that algorithm. It is a different one that happens to end with
the same backward induction.

## The exchange series

Write kinds zero-based, `rock = 0, scissors = 1, paper = 2`, so kind `k`
captures kind `(k + 1) % 3` and the only kind that captures `k` is
`(k + 2) % 3`. A piece may move one king step onto an empty square or onto an
enemy piece it beats, so **adjacency and kind are exactly the capture
condition**: no blocking, no pins, no line pieces.

Fix a capture `m` by side `A`: the piece of kind `a` on square `s` takes the
side-`B` piece of kind `v = (a + 1) % 3` on square `t`.

- **Neighbourhood.** `N(t)` is the at most eight board squares a king step from
  `t`. Only pieces standing there can ever capture on `t`, and no piece outside
  `N(t)` enters it during the series, because every step of the series moves a
  piece *onto* `t`.
- **Pool.** `pool[p][k]` counts side `p`'s kind-`k` pieces standing on `N(t)`
  when the series starts, less one for the mover itself.
- **Steps.** Step `i` moves a piece of side `m_i` and kind `k_i` onto `t`,
  capturing whatever step `i - 1` left there. The occupant after step `0` is the
  original victim, of kind `v` and side `B`. Then

  ```
  m_1 = A,  k_1 = a
  m_i = the side that did not play step i-1
  k_i = (k_{i-1} + 2) % 3          # the only kind that beats the occupant
  ```

  so `k_i = (a + 2(i-1)) % 3` and the kinds cycle with period three while the
  sides alternate with period two: the whole series repeats with period six.
- **Availability.** Step 1 is available because `m` is legal. Step `i > 1` is
  available exactly when `pool[m_i][k_i] > 0`; taking it decrements that entry.
  `n` is the largest `i` for which every step up to `i` is available.

**There is no attacker-selection freedom.** In chess SEE the recapturing side
picks its least valuable attacker; here the kind is forced by the cycle, and two
pieces of the same side and kind are interchangeable because the valuation below
reads only per-side, per-kind counts. The series is therefore a single sequence,
not a search over orders.

### Termination

The chain of *kinds* never terminates on its own — it cycles with period three,
which is exactly the property that breaks the chess argument (there, the
exchange ends because values are exhausted in a strict order). Termination here
comes from the pool instead:

> Every step consumes one distinct piece from `pool` and permanently vacates its
> square. Nothing is ever added to `N(t)` during the series, because the only
> squares a series move writes to are `t` (occupied) and the vacated source.
> Hence `n ≤ |N(t)| ≤ 8`, so `n ≤ 8` steps and at most nine board configurations.

Two secondary consequences, both needed for the series to be well defined:

- Each step removes a piece, so the board strictly loses material and **no
  position in the series can repeat**; the threefold rule cannot fire inside it.
- Each step is a capture, so the no-capture clock resets every step and **the
  80-ply rule cannot fire inside it** either.

### Valuation and the swing

Let `c_0` be the current six piece counts and `c_i` the counts after `i` steps —
fully determined, since the side and kind removed at each step are known. Let

```
E_i = count_weight * (material(c_i, A) - material(c_i, B))
```

be exactly the evaluator's material term from `A`'s point of view, where
`material` is the flat piece count or, in variable mode,
`variable_material_total(...) / BASE` — the same function
`intransitive/heuristics/material.py` already gives the leaf evaluator. The
series is priced in evaluator units, not in an invented piece table.

Because the series' count trajectory is known exactly, the contextual variable
valuation costs nothing extra to follow: `n + 1 ≤ 9` evaluations of a three-term
formula per capture, which is a constant, not a search.

The two sides then choose when to stop. Only step 1 is compulsory — it is the
move under evaluation. After `i` steps it is `A`'s turn when `i` is even:

```
V_n = E_n
V_i = max(E_i, V_{i+1})   if i is even   (A chooses)
V_i = min(E_i, V_{i+1})   if i is odd    (B chooses)
swing(m) = V_1 - E_0
```

`swing(m)` is the material the capture is worth to the mover, in evaluator
units, allowing either side to break off.

### Exactness

`swing(m)` is the **exact** material-only minimax value of the sub-game in which

1. the only moves are "continue the exchange on `t`" and "stop";
2. the evaluation is the material term alone;
3. no terminal event (corner, stalemate, draw) occurs during the series.

Assumption 1 is the standard SEE idealisation and is where every inaccuracy
below lives. Assumptions 2 and 3 are why an SEE verdict is never a certificate.

### Known inaccuracies

1. **Better replies exist.** A side that "should" recapture may instead run for
   the corner, answer a bigger threat, or retreat. SEE plays the exchange in
   isolation and will misjudge every position where the exchange is not the main
   event.
2. **Only material is priced.** `piece_advantage`, `attacking_position`,
   `defensive_position`, `overload`, `local_pressure` and `runner_pressure` are
   ignored. A capture that loses material and wins a route is scored negative.
   The delta margin below exists to cover this gap, and is an allowance, not a
   bound.
3. **Terminal events are invisible.** A capture may win on the spot by reaching
   a corner or stalemating the opponent, and a recapture may do the same. SEE
   never sees it, so every pruning site carries its own terminal guard.
4. **Abandoned duties.** A recapture credited by the series may be the piece
   that was holding a corner or covering a runner. The series moves it anyway.
5. **Reinforcements are ignored.** A piece two king steps from `t` can join one
   ply later. The series only ever knows `N(t)`.
6. **Non-linear values.** In variable mode a type's worth moves with both
   armies' counts, so an exchange is not the sum of its parts. The trajectory is
   followed exactly, but the fact that `E` bends means SEE is more sensitive to
   the army composition than a chess SEE is to its piece table.
7. **One square only.** Simultaneous exchanges on two squares, and captures that
   are only possible because another exchange vacated a square, are out of scope.

## Flat material makes SEE two-valued

Under flat material every piece contributes the same `count_weight`, so
`E_i = E_0 + count_weight` for odd `i` and `E_0` for even `i`, and therefore

```
swing(m) ∈ {0, count_weight}
```

for every legal capture. **No capture is ever a losing capture under flat
material**, because every exchange is one piece for one piece. Two things follow
and are measured rather than assumed:

- Skipping "losing" captures can never fire with flat material at threshold
  zero. Flat-material SEE pruning is inert by construction.
- SEE still orders usefully with flat material: it separates a free capture from
  a defended one, which MVV-LVA cannot do at all when all victims are worth the
  same.

This is the answer to the issue's valuation question. **Variable material is not
one of two equally reasonable choices; it is the only one under which the
verdict "this capture loses material" exists.** The implementation therefore
prices the series with whatever material mode the evaluator is configured for,
and the measurements report flat and variable separately.

### What variable material makes SEE say

The contextual valuation prices a type by its opponent's prey and predator
counts, so the series inherits two consequences that look wrong until you read
them as the valuation's own opinion rather than the exchange's:

- **Taking your last prey scores negative.** A lone blue rock taking the only
  red scissors scores `-259.6` at `count_weight = 100`, because afterwards the
  rock has no prey and its own value collapses from 341.6 to 68.3. Flat material
  scores the same capture `+100`. This is the fixture the issue asks for, where
  the contextual valuation changes the verdict.
- **A defender may decline a free recapture** for the same reason, which is why
  the backward induction cannot be replaced by "always recapture".

Neither is a bug in the series; both are the variable valuation showing through
it, and both are reasons the pruning site needs a playing-strength measurement
rather than a node count.

## Where it is applied

Three independently switchable sites, none of them on by default.

| Switch | Site | Effect |
|---|---|---|
| `see_ordering_enabled` | main search, `_ordered` | a capture rank key ordered by descending swing, placed above the MVV-LVA victim/attacker keys, so it augments MVV-LVA when both are on and replaces it when MVV-LVA is off |
| `see_quiescence_ordering_enabled` | quiescence | captures tried in descending swing order, ties by ascending action |
| `see_quiescence_pruning_enabled` | quiescence | skip a capture whose swing is below `see_threshold` (default `0.`, so strictly losing captures only) |

`compiled_see_enabled` (default true) selects the compiled kernel; false selects
the Python reference. Both are exercised in tests and must agree exactly.

Immediate wins, the preferred TT/PV move, previous root scores and corner
defence keep their existing priority over every SEE key, and the final action
tie-break stays deterministic. Proof and certificate kernels do not consult SEE.

## Delta pruning in quiescence

`delta_pruning_enabled` skips a capture when

```
stand_pat + gain(m) + delta_margin * allowance <= alpha
```

where `gain(m) = E_1 - E_0` is the **exact** material won by the capture itself
under the configured material mode — not a piece-table estimate, and in variable
mode not equal to the victim's own value, because removing the victim reprices
both armies.

The margin has to be in the evaluator's units; the weights here are not
centipawns and a margin copied from a chess engine would be meaningless. The
allowance is the non-material swing one capture can plausibly produce, built the
same way `selective.margin` builds its futility allowance and from the same
per-module scales:

```
allowance = |advantage_weight| * (max(predator_zero_bonus, predator_scarcity_bonus) + prey_bonus)
          + 3*|attack_weight|   if attack_enabled
          + 4*|defence_weight|  if defence_enabled
          + 2*|overload_weight| if overload_enabled
          + 8*|pressure_weight| if pressure_enabled
          + 3*|runner_weight|   if runner_enabled
```

The material part is deliberately absent: `gain(m)` already accounts for it
exactly. `delta_margin` is a finite multiplier in `0..16`; `0.` prunes on
material alone and is the aggressive setting, `1.` is the calibrated default.
With the adopted evaluator weights the allowance is about 237 evaluator units
against 100 per piece, so at `delta_margin = 1.` delta pruning fires rarely.
That is a measured property of these weights, not a claim that the margin is
correct; see the benchmark for what each setting actually does.

Delta pruning assumes one capture's worth of gain. A quiescence chain that wins
material over several captures can be pruned at its first link; that is the
technique's standard, accepted inaccuracy.

## Guards at the pruning sites

Neither SEE pruning nor delta pruning may drop a capture that ends the game.
Both are refused when either guard fails:

- **Corner.** A capture landing on the mover's own goal corner is never skipped.
- **Stalemate.** Skipping is refused for the whole node unless the opponent
  provably still has a move after any single reply, using the existing counting
  arguments `crowded_side_has_moves` / `open_side_has_moves` at depth one.

Quiescence itself remains what it was: a stand-pat leaf, then captures only.
Stand-pat still includes the ordinary bounded proof, so a proven position is
returned before any capture is considered, and a mate-range stand-pat short
circuits before pruning is reached.

## Costs, budget and identity

All of it is charged to the existing `Budget`. Per node the survey costs an
81-square guard scan when pruning is active, a bounded neighbourhood scan per
capture candidate, and at most nine constant-size valuations per candidate. No
site performs unbounded work, and no site allocates a game state.

Every switch enters `SearchConfig.to_dict()` and therefore
`SearchConfig.identity()`, the transposition table's configuration scope and the
tournament manifest. `search_version` is unchanged: these are additive Python
search options, and existing JSON configs load with all of them off. Tournament
manifests written before this change name a smaller set of shared search
settings and will no longer validate, which is the harness's stated rule that
changed settings need a new manifest, not a regression. The native Rust teacher
implements neither quiescence nor SEE; enabling any of these with a non-compact
backend raises, exactly as the existing selective options do.

`SearchResult.ordering` reports `see_*` counters, plus `cutoffs` and
`first_cutoffs` so first-move cutoff rate can be measured directly.
`SearchResult.selective` reports `quiescence_see_skips` and
`quiescence_delta_skips`.

## What this is not

An SEE verdict is a heuristic ranking, never a proof. Results from a search with
any of these switches on are labelled `selective_*` and report proof status
`unknown`, like every other selective feature. None of these switches changes
the evaluator, the genome contract, the rules, the action encoding or any
recorded outcome.

## Measured outcome

Every switch stays off. On seven contact midgame positions at depth 2, quiescence
costs 2.54× total work with variable material (5.50× on the worst position); the
exchange filter with `delta_margin=0` brings that to 1.67× total and 2.57× worst
while examining 62% fewer captures. Main-search exchange ordering does not pay
for itself: the first-move cutoff rate is unchanged and work rises about 1%. The
calibrated `delta_margin=1` allowance of 237.4 evaluator units against 100 per
piece is effectively inert. Equal-time paired play separates nothing, and the
predeclared rule needs both a work reduction and a strength result.

Measurements, the adoption decision and the re-measured quiescence multiplier are
in [the bounded benchmark](../benchmarks/exchange/README.md).
