# Issue #68 bounded validation

**Decision: every flag ships off.** Equal-time paired play cannot distinguish
any of the four from its reference — the noise floor of this harness is the
same size as every effect measured, established below by a technique that is
provably inert scoring 5–5. Node reduction alone is not acceptance, and that is
all any of them demonstrated.

Per technique:

- **Razoring — do not pursue.** It is inapplicable to this evaluator rather
  than mistuned; twelve times below its safe margin it still fires 44 times in
  48,765 eligible nodes, because the nodes `guarded()` admits are the balanced
  ones whose static score sits near the window.
- **Reverse futility and move-count pruning — plausible, unproven.** They fire,
  they reduce nodes by 2.2% and 1.5%, and at matched time they reach +0.70 and
  +0.93 ply deeper than their references. Deeper is not stronger, and 30 games
  a side cannot tell the difference.
- **Mate-distance pruning — correct and free, but currently pointless.** Zero
  disagreements with the bounded-proof oracle on all 100 certified tactics. It
  is exactly inert while the bounded leaf proof is enabled, which is how the
  engine ships, and removes 10% of nodes without it. **Worth enabling in any
  proof-free configuration**; there is no reason to make it a global default,
  since that would change every existing search identity for no gain in the
  shipped setup.

No live generator, trainer, checkpoint opponent or tournament protocol was
changed.

[Predeclared plan](plan.json) (fixed before any measurement below),
[implementation contract](../../heuristics/SELECTIVE_SEARCH.md).

## Reproduction

```sh
python -m unittest intransitive.tests.test_shallow_pruning -v
for stage in calibrate fixed sensitivity mate paired; do
  python -m intransitive.benchmarks.shallow_pruning.reproduce \
    --stage $stage --output intransitive/benchmarks/shallow_pruning/evidence/$stage.json
done
```

Use the repository Python environment (3.11, NumPy/Numba from the lock). Every
evidence file records the revision and the SHA-256 of each file that can change
a result — `heuristics/*.py`, this benchmark and the test suite — so stages run
at different times are comparable when those hashes match. A whole-tree diff
digest is deliberately **not** the identity: documentation was still being
written between stages.

Search settings throughout: the `selective.supported()` evaluator scale
(`count_weight=100`, `advantage_weight=25`, no route or pressure modules),
PVS on, compiled ordering on, bounded proof off except where stated. **PVS is
not optional here.** Issue #66 established that the shared non-PV guard never
opens without it, so a protocol without PVS cannot measure any of this.

**Host caveat.** These runs shared a busy machine (load average above 8 on 8
cores). Node counts, firing rates and divergences are unaffected — they are
deterministic. Wall-clock figures and the achieved depths in the paired stage
are not, and are reported as indicative only. There is no deployment latency
estimate here.

## Margins, measured rather than borrowed

[evidence/calibrate.json](evidence/calibrate.json). 480 random-play
positions (seed 680920, disjoint from every other corpus here). For each, the
static evaluation and the depth-1, 2 and 3 alpha-beta values, with the
difference expressed in `selective.allowance()` units — the only scale in which
a margin means anything, since the weights are not centipawns.

| scale | allowance (units/ply) | largest rise per ply | largest fall per ply |
|---|---:|---:|---:|
| `supported()` | 75.0 | 1.476 | 0.729 |
| adopted route genome | 287.4 | 0.413 | 0.162 |

Shipped defaults: `razoring_margin=1.5`, `reverse_futility_margin=1.0`. Both
cover every observed swing on **both** scales, and for both a larger multiple
fires less often.

Two findings worth recording:

- The issue text estimated the existing `futility_margin=1.0` at "roughly 1% of
  a piece". It is not: `allowance()` is `count_weight/2 + advantage_weight`, so
  the default futility margin is **75 evaluator units per ply, about 0.75 of a
  piece** at `count_weight=100`. The margin was never as tight as feared.
- `allowance()` sums weight *ceilings*, not the swing those modules actually
  produce, so it over-scales badly once the route modules are on: the adopted
  genome's allowance is 3.8× larger while its actual swings are 3.6× smaller.
  One multiplier therefore cannot be tight on both scales. The new margins
  accept fractional values so an evolved genome can be calibrated from its own
  measurement; a genome that adopts the shipped defaults gets a safe margin
  that fires even less often than it does here.

Median branching factor is 51 legal moves (mean 50.4, max 66), which sets
`move_count_base=12`: the first 13, 16 and 21 children at remaining depth 1, 2
and 3 are always searched in full.

## Does each technique fire, and what does it cost?

[evidence/fixed.json](evidence/fixed.json). 40 random-play positions
(seed 680921), depth 4, identical genome, one mode changed at a time. Razoring's
reference is quiescence, not the all-off baseline, because it cannot be enabled
without it.

| mode | reference | nodes | node ratio | seconds | eligible | fired | action changes |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | — | 418,305 | 1.000 | 12.4 | — | — | — |
| quiescence | baseline | 719,422 | 1.720 | 17.6 | — | — | 12 |
| razoring | quiescence | 719,232 | 1.000 | 28.4 | 85,672 | **2** | 0 |
| reverse futility | baseline | 409,271 | **0.978** | 33.3 | 85,890 | 9,034 | 0 |
| move count | baseline | 412,216 | **0.985** | 28.6 | 165,115 | 4,664 | 0 |
| mate distance | baseline | 418,305 | 1.000 | 15.8 | 111,062 | **0** | 0 |
| all four | baseline | 689,395 | 1.648 | 29.5 | 428,987 | 14,318 | 12 |

Read against quiescence, the four together are 0.958 — the 1.648 is quiescence's
own 1.720 partly recovered.

**Do not read the seconds column as a cost model.** This stage was run twice
on the same source; every node count, counter and divergence figure above
reproduced exactly, and every timing moved. The mate-distance row is the
control: it performs bit-identical work to the baseline — same nodes, zero
cuts — and its two runs differ by 28%. Treat the seconds only as evidence that
the guard and the extra static evaluation are *not free* in Python, which
issue #60 already measured for forward futility; the equal-time paired stage
below is where timing is actually controlled.

- **Reverse futility fires and reduces nodes**, 9,034 cutoffs on 10.5% of the
  nodes that reached its test, for a 2.2% node reduction and no change to any
  chosen move. 21% of nodes now pay for a static evaluation.
- **Move-count pruning fires** but returns 1.5%, and pays a per-child `quiet()`
  test on 165,115 children to do it.
- **Razoring effectively does not fire**: 2 applications in 85,672 eligible
  nodes.
- **Mate-distance pruning does not fire here at all** — ordinary midgame
  positions at depth 4 contain no mate within the window. Its own corpus is
  below.

## Sensitivity: is that just a bad parameter choice?

[evidence/sensitivity.json](evidence/sensitivity.json). 20 positions from the
same corpus. Settings below the calibrated margin are **measured-unsafe** and
recorded for information, not offered for adoption.

| parameter | value | eligible | fired | node ratio | action changes |
|---|---:|---:|---:|---:|---:|
| `razoring_margin` | 0.125 ‡ | 48,765 | 44 | 0.987 | 0 |
| | 0.25 ‡ | 48,765 | 44 | 0.987 | 0 |
| | 0.5 ‡ | 48,765 | 44 | 0.987 | 0 |
| | 1.0 ‡ | 49,027 | 38 | 0.990 | 0 |
| | **1.5** | 49,027 | 0 | 1.000 | 0 |
| `reverse_futility_margin` | 0.25 ‡ | 48,952 | 778 | 0.908 | 0 |
| | 0.5 ‡ | 48,952 | 778 | 0.908 | 0 |
| | **1.0** | 49,166 | 266 | 0.999 | 0 |
| | 2.0 | 49,166 | 0 | 1.000 | 0 |
| `move_count_base` | 2 | 119,749 | 2,326 | 0.985 | 0 |
| | 4 | 114,559 | 2,326 | 0.985 | 0 |
| | 8 | 104,179 | 2,325 | 0.985 | 0 |
| | **12** | 93,799 | 2,320 | 0.985 | 0 |
| | 24 | 62,659 | 2,068 | 0.987 | 0 |

‡ below the measured safe multiplier for this scale.

This is the central negative result. **Razoring is inapplicable to this
evaluator, not merely mistuned**: twelve times below its safe margin it still
fires 44 times in 48,765 eligible nodes and returns 1.3%. The eligible nodes are
exactly the ones `guarded()` admits — no available capture, nobody near a
corner, at least eight moves a side — which are balanced positions whose static
score sits close to the null window. The condition razoring tests for is one the
guard has already excluded.

Reverse futility does have a usable range, but it is **below** the calibrated
margin: 0.908 at 0.25–0.5 against 0.999 at the safe 1.0. Buying a 9% node
reduction means accepting a margin that the measured evaluation swing says is
too small. No action changed on this sample, which is evidence but not
permission — 20 positions cannot exclude a rare tactical failure.

Move-count pruning is insensitive to its own parameter across a 12× range: the
reviewed `quiet()` predicate, not the index threshold, decides what gets
skipped. That is the predicate working as intended, and it caps the technique.

## Mate-distance pruning against the oracle

[evidence/mate.json](evidence/mate.json). All 100 certified
tactics, each searched to depth 4 with the technique off and on, in two
configurations.

| configuration | cases | disagreements | oracle action | cuts | nodes on/off | ratio |
|---|---:|---:|---:|---:|---:|---:|
| bounded proof on (as shipped) | 100 | **0** | 100/100 | 0 | 735,680 / 735,680 | 1.000 |
| bounded proof off | 100 | **0** | 100/100 | 353 | 28,022 / 31,222 | **0.898** |

A disagreement means any difference in chosen action, score, or the bounded
proof's status or score. There were none.

The split matters. With the bounded proof enabled, a certified tactic is settled
by the leaf oracle at depth one, iterative deepening stops on the proven result,
and the narrowing is never reached — the technique is exactly inert. It only
acts on mates the tree has to find for itself, where it removes 10% of nodes.
So it is correct, free, and currently pointless in the shipped configuration;
it becomes useful in any configuration without leaf proofs.

## Equal-time paired play

[evidence/paired.json](evidence/paired.json). 15 held-out positions from
`tournament.spec.generate_positions(seed=6854)`, both colours, equal wall clock
per move, official wins only, a capped game is unfinished and never a draw.

The #54 harness could not be invoked directly: it varies the *genome* between
the two engines and gives them one shared search protocol, so it cannot pit two
search settings against each other. Its rules and position machinery are reused
here instead — the same generator, pool split, colour pairing, termination rule
and Wilson interval — and the deviation is stated rather than papered over.
Modelling draws stay inside the search; when a search declines a root that
official play considers live, the match plays the first legal move and the count
is recorded.

| candidate | reference | games | W | L | unfinished | decisive Wilson 95% | mean depth cand/ref | seconds cand/ref |
|---|---|---:|---:|---:|---:|---|---|---|
| razoring | quiescence | 30 | 3 | 4 | 23 | 0.16–0.75 | 3.20 / 3.23 | 253 / 254 |
| reverse futility | baseline | 30 | 7 | 5 | 18 | 0.32–0.81 | **3.91** / 3.21 | 243 / 242 |
| move count | baseline | 30 | 9 | 7 | 14 | 0.33–0.77 | **4.30** / 3.37 | 217 / 214 |
| mate distance | baseline | 30 | 5 | 5 | 20 | 0.24–0.76 | 3.71 / 3.72 | 249 / 248 |
| all four | baseline | 30 | 6 | 4 | 20 | 0.31–0.83 | **4.15** / 3.52 | 245 / 244 |

Wall clock per side matched to within 0.5% in every row, so these are equal-time
results. **No row establishes a strength difference**: every Wilson interval
spans 0.5 with room to spare, and most games hit the 80-ply cap unfinished.

The `mate distance` row is the useful one, and not for its win rate. That
technique is value preserving and, with the bounded proof enabled, provably
inert — its mean depth matches the reference to 0.01 ply. It still came out
**5–5 rather than all-drawn**, because an equal-*time* search is not
deterministic: the clock cuts iterative deepening at different points and the
games diverge. So ±5 decisive games is this harness's noise floor at 30 games
per row, which is the same size as every effect in the table. None of these
rows can distinguish a real change from timing jitter.

What the table does show clearly is **depth**: reverse futility reaches +0.70
ply over its reference, move-count pruning +0.93, all four together +0.63, at
matched time. That is the opposite sign to the fixed-depth wall-clock cost
above, and the regimes explain it — the fixed-depth stage ran with the bounded
leaf proof disabled to isolate the heuristic tree, while these games run with
it enabled, which makes each leaf far more expensive and the static-evaluation
overhead proportionally much smaller. **Node savings do convert into depth when
leaves are expensive.** Extra depth is not strength, and this sample cannot
show whether it becomes strength.

The `all four` row is confounded: it includes quiescence (which razoring
requires) while its reference does not, so it measures quiescence as much as
this issue's techniques. The other four rows each differ from their reference
by one flag only.

Modelling-draw fallbacks — moves where the search declined a root that official
play considers live — were 1 (reverse futility), 12 (move count), 12 (mate
distance), 31 (all) and **111 (razoring)** out of at most 2,400 moves per row.
Razoring's games are the quiescence-versus-quiescence pairing, which repeats
positions far more often; that row is the least interpretable of the five and
should not be read as evidence about razoring either way.

**Identity note.** The paired stage was launched before two later
non-behavioural edits — a docstring correction in `reproduce.py` and one added
test in `tests/test_shallow_pruning.py` — so its header records the earlier
hashes for those two files. Every `heuristics/*.py` hash and `plan.json` match
across all five evidence files: the stages measured the same search.

## What this does not establish

The measurements are deterministic for nodes and firing rates, and bounded and
noisy for everything about time and strength. They do not establish that any of
these techniques is safe on an evolved genome, at depths above 4, with the route
modules enabled, or in the native backend, which does not implement them. A
margin measured over 480 positions is a description of one corpus, not a bound;
issue #55 can move the evaluator underneath it at any time.

Nothing here was tuned on the paired positions or their symmetry variants, and
no pruning parameter is part of the weight-evolution genome.
