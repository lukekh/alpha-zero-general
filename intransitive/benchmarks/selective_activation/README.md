# Issue #66 activation measurement

**Decision: every technique stays disabled by default.** This is a diagnosis,
not an adoption. It establishes the first configuration in which NMP, futility
and LMR actually execute on ordinary positions, attributes a node count to each
one separately, and records what each one costs. It does not establish playing
strength, and node reduction alone is explicitly not acceptance.

[Predeclared plan](plan.json), [implementation contract](../../heuristics/SELECTIVE_SEARCH.md),
[raw evidence](evidence/activation.json).

## What the issue found, and why

Across a 37,299-move weight-evolution run whose frozen protocol declared
quiescence and LMR, `nmp_attempts`, `nmp_cutoffs`, `futility_eligible`,
`futility_pruned`, `lmr_reduced` and `lmr_researches` were all **zero**, and
every move reported `selective.disabled_reason = "unsupported evaluator scales"`.
Four independent causes, each verified here:

1. **The protocol searched at `depth=2`.** `nmp_min_depth` and `lmr_min_depth`
   are both 3, and the root never prunes, so a technique needing `d` remaining
   plies needs `max_depth > d`. Depth 2 cannot reach either.
2. **PVS was off.** NMP and futility are eligible only at a node entered with a
   null window, and only a scout search produces one. The support gate was
   irrelevant while this held.
3. **The evaluator-scale interlock rejected every evolved genome**, silently,
   while the configuration still claimed both techniques were enabled.
4. **MVV-LVA cannot reorder anything under flat material.** Every piece type is
   worth `BASE`, so the victim and attacker keys are constant.

A fifth cause only becomes visible once the first three are removed: at evolved
scales the futility allowance is roughly an order of magnitude too large to
prune, so the technique is eligible thousands of times per move and prunes
nothing.

## Reproduction and identity

Use the repository Python environment (Python 3.11; the evidence uses 3.11.4,
NumPy 2.4.6, Numba 0.67.0, llvmlite 0.49.0).

```sh
python -m unittest intransitive.tests.test_selective_search intransitive.tests.test_tournament -v
cargo test --release --offline --manifest-path intransitive/rust_teacher/Cargo.toml
python -m intransitive.benchmarks.selective_activation.reproduce \
  --output intransitive/benchmarks/selective_activation/evidence/activation.json
```

The revision, working-diff digest, platform, full base configuration, every
arm's options and each search's complete counters are recorded in the output.
Node counts are deterministic for a given revision and are the primary metric;
wall-clock seconds are one sample per cell on a shared host and are indicative
only. Proof leaves are disabled throughout so that only the heuristic tree is
measured.

## Arms and attribution

All arms share the **adopted default genome** — material 100, advantage
23.967050360966205, attack 25.714516982666414, defence 32.5643023919054, flat
material — that is, exactly what the evolution harness produces and what the
frozen protocol scored. Each arm differs from its named reference by a single
switch, so a node count belongs to that switch rather than to the regime:

- `pvs` is attributed against `baseline`, plain alpha-beta. It is the enabling
  precondition for two of the techniques, so its own cost is measured separately
  instead of being folded into theirs.
- `nmp`, `futility`, `futility_tuned`, `lmr`, `mvv_lva`, `quiescence` and `all`
  are attributed against `pvs`, the window regime they need to run at all.
- `variable_mvv_lva` is attributed against `variable_material`. Those two arms
  exist only so MVV-LVA has a reference in which capture values are not all
  equal; they are a control for that one question, not a second genome under
  test.

Positive `vs reference` means **fewer** nodes than the reference.

### Depth 4, six positions

| arm | nodes | vs reference | seconds | nmp att/cut | futility elig/pruned | lmr red/re | mvv captures | same move | same score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 50,672 | — | 6.3 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 6/6 | 6/6 |
| `pvs` | 44,605 | **+12.0%** | 5.5 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 6/6 | 6/6 |
| `nmp` | 44,837 | −0.5% | 5.5 | **68** / 12 | 0 / 0 | 0 / 0 | 0 | 6/6 | 6/6 |
| `futility` | 44,605 | **0.0%** | 6.0 | 0 / 0 | **7,085 / 0** | 0 / 0 | 0 | 6/6 | 6/6 |
| `futility_tuned` | 38,958 | **+12.7%** | 5.0 | 0 / 0 | 7,125 / **4,611** | 0 / 0 | 0 | 6/6 | 6/6 |
| `lmr` | 31,205 | **+30.0%** | 3.4 | 0 / 0 | 0 / 0 | **493** / 19 | 0 | 4/6 | 2/6 |
| `lmr_adaptive` | 31,205 | 0.0% | 3.5 | 0 / 0 | 0 / 0 | 493 / 19 | 0 | 6/6 | 6/6 |
| `mvv_lva` | 44,605 | **0.0%** | 5.4 | 0 / 0 | 0 / 0 | 0 / 0 | 1,299 | 6/6 | 6/6 |
| `quiescence` | 86,106 | −93.0% | 6.4 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 4/6 | 4/6 |
| `all` | 27,185 | **+39.1%** | 3.6 | 70 / 52 | 4,618 / 3,032 | 493 / 19 | 693 | 4/6 | 2/6 |
| `variable_material` | 44,377 | — | 6.0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 6/6 | 6/6 |
| `variable_mvv_lva` | 44,377 | **0.0%** | 6.0 | 0 / 0 | 0 / 0 | 0 / 0 | 1,297 | 6/6 | 6/6 |

### Depth 5, two positions

| arm | nodes | vs reference | seconds | nmp att/cut | futility elig/pruned | lmr red/re | mvv captures | same move | same score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 331,979 | — | 50.0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 2/2 | 2/2 |
| `pvs` | 260,276 | **+21.6%** | 37.3 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 2/2 | 2/2 |
| `nmp` | 239,702 | **+7.9%** | 35.4 | **334** / 112 | 0 / 0 | 0 / 0 | 0 | 2/2 | 2/2 |
| `futility` | 260,276 | **0.0%** | 40.9 | 0 / 0 | **21,726 / 0** | 0 / 0 | 0 | 2/2 | 2/2 |
| `futility_tuned` | 247,411 | +4.9% | 39.0 | 0 / 0 | 21,749 / **11,598** | 0 / 0 | 0 | 2/2 | 2/2 |
| `lmr` | 43,716 | **+83.2%** | 5.5 | 0 / 0 | 0 / 0 | **3,686** / 26 | 0 | 0/2 | 0/2 |
| `lmr_adaptive` | 40,668 | **+7.0%** | 6.1 | 0 / 0 | 0 / 0 | 3,836 / 30 | 0 | 1/2 | 0/2 |
| `mvv_lva` | 260,276 | **0.0%** | 43.4 | 0 / 0 | 0 / 0 | 0 / 0 | 8,428 | 2/2 | 2/2 |
| `quiescence` | 507,490 | −95.0% | 53.3 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 2/2 | 2/2 |
| `all` | 39,098 | **+85.0%** | 6.3 | 75 / 56 | 6,200 / 3,861 | 2,586 / 26 | 1,515 | 0/2 | 0/2 |
| `variable_material` | 253,361 | — | 44.0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 2/2 | 2/2 |
| `variable_mvv_lva` | 253,360 | **0.0%** | 51.1 | 0 / 0 | 0 / 0 | 0 / 0 | 10,815 | 2/2 | 2/2 |

`lmr_adaptive` is attributed against `lmr`, and the two variable arms against
`variable_material`; every other arm is attributed against `pvs`, and `pvs`
against `baseline`.

**The first acceptance criterion is met.** `nmp_attempts` 547, `futility_eligible`
68,503 and `lmr_reduced` 11,587 across the run, all on ordinary positions, with
per-arm counts above. The configuration is the `all` arm: the adopted genome,
`selective_evaluator_enabled`, `pvs_enabled`, depth 4 or 5, and
`futility_margin` at 1/16.

Read across the rows and each technique answers for itself:

- **PVS is the largest single win and is exact.** +12.0% at depth 4 and +21.6%
  at depth 5, with the same move and the same score on every position. It is
  also the precondition for two of the others.
- **Futility at the default margin is pure overhead.** Node counts identical to
  the reference to the node, at both depths, while 7,085 and 21,726 quiet moves
  passed every eligibility test and **none** was pruned. That is the issue's
  blocker 5, measured.
- **Futility at `futility_margin = 1/16` prunes and costs nothing.** +12.7% at
  depth 4 for 4,611 prunes, with the identical move and identical score on all
  six positions. The margin, not the technique, was the problem.
- **NMP is a depth effect.** −0.5% at depth 4 on 68 attempts; +7.9% at depth 5
  on 334. It needs remaining depth to pay for its probe and verification.
- **LMR is the largest reduction and the only one that changes answers.**
  +30.0% and +83.2%, but the score differs on four of six positions at depth 4
  and on both at depth 5. See the tactical check below before reading that as
  free.
- **MVV-LVA changes nothing here.** Identical node counts under flat material,
  which is structural — every piece type is worth `BASE`, so the capture keys
  are constant. Under variable material, where they are not constant, the depth-5
  count moved by **one node** in 253,361. On this corpus the ordering is inert
  either way.
- **Quiescence roughly doubles the tree** (−93% and −95%) in exchange for
  resolving captures at the horizon. It is a horizon-accuracy technique, not a
  reduction, and it is reported here only so it is not mistaken for one.

## The futility allowance

The experimental allowance is charged per remaining ply for the module terms and
from the second ply for material and advantage, so at the adopted genome
`futility_margin = 1` is worth 207.4 at depth 1 and 494.8 at depth 2. Sweeping
the multiplier over the same six positions at depth 4:

| `futility_margin` | depth-1 allowance | depth-2 allowance | nodes | eligible | pruned |
|---:|---:|---:|---:|---:|---:|
| 1 (default) | 207.4 | 494.8 | 44,605 | 7,085 | **0** |
| 1/2 | 103.7 | 247.4 | 43,319 | 7,085 | 1,286 |
| 1/4 | 51.9 | 123.7 | 43,117 | 7,085 | 1,488 |
| 1/8 | 25.9 | 61.8 | 39,987 | 7,085 | 3,582 |
| 1/16 | 13.0 | 30.9 | 38,958 | 7,125 | 4,611 |

Eligibility is flat at 7,085 across the sweep, which is the point: the guard and
the quiet test admit the same moves every time, and only the allowance decides
whether any of them is skipped. The default is roughly an order of magnitude too
generous, and the technique does not begin to prune until the multiplier is at
or below one half.

This is a measured sweep, not a calibration. The module terms are still the
conservative per-side feature ranges — a bound on the evaluation, not on one ply
of it — so the honest reading is that `futility_margin` is the knob and this
table is the range a tuning run should search. Nothing here says 1/16 is right.

## Paired play at fixed depth

Both colours from every start, both sides at depth 4, 24-ply cap. The candidate
is the `all` arm; the opponent is `baseline`. Equal depth, not equal time: the
question is whether pruning changes the game, with its cost reported separately.

| start | candidate colour | outcome | plies | candidate nodes | baseline nodes |
|---|---|---|---:|---:|---:|
| opening | first | unfinished | 24 | 118,401 | 257,123 |
| opening | second | unfinished | 24 | 95,973 | 159,431 |
| midgame | first | unfinished | 24 | 125,836 | 200,709 |
| midgame | second | unfinished | 24 | 119,146 | 244,765 |
| endgame | first | **win** | 9 | 1,277 | 808 |
| endgame | second | **loss** | 9 | 717 | 1,198 |

One win, one loss, four unfinished. The descriptive 95% Wilson interval on the
decisive score is 0.095–0.905, which excludes nothing. **This does not establish
equal strength and is not offered as though it did.** The two decisive games are
the same endgame fixture with the colours exchanged, so they are one position,
not two observations.

What the table does show is cost: at equal depth the candidate searched roughly
half the nodes of the baseline in every unfinished game. That is a cost result,
and cost is not strength.

Separately, on the 100 certified tactics at depth 5 with the adopted genome, the
reduced arms lose nothing:

| arm | tactics missed | nodes |
|---|---:|---:|
| `pvs` | 2 / 100 | 2,909,913 |
| `pvs + lmr` | **2 / 100** | 2,255,489 |
| `pvs + lmr_adaptive` | **2 / 100** | 1,963,547 |

The same two cases fail in all three arms, including the unreduced one, so they
belong to this configuration and not to the reduction. That is the reassurance
the divergent scores above need, and it is now a regression test
(`test_composed_reductions_cost_no_certified_tactic`), which asserts that no
tactic the plain scout search finds may be lost to a reduction.

## Limitations

- Six positions — opening, one seeded 24-ply midgame line and a synthetic sparse
  endgame, each in its original and colour-exchanged form — are enough to show
  that a technique executes and what it costs. They are nowhere near enough to
  establish tactical safety, and this report makes no such claim.
- Node counts are deterministic for a revision and are the metric. Wall-clock
  seconds are a single sample per cell on a shared host and are indicative only.
- Bounded proofs are disabled in every arm so that only the heuristic tree is
  measured. Harness games enable them, so absolute costs here are not harness
  costs.
- The paired games are fixed-depth and capped at 24 plies, so most end
  unfinished, and their outcome counts are descriptive, not a strength estimate.
  They also carry the whole `all` arm against plain alpha-beta rather than one
  technique at a time.
- The #54 paired harness cannot express this comparison: both sides of a match
  share one frozen protocol, and a candidate's identity is its genome, not its
  search settings, so selective-on versus selective-off is not schedulable there.
  The paired games here are built in the same shape — both colours from every
  start, fixed depth, complete move journals — but they are not that harness, and
  nothing here clears the adoption gate the issue sets. Every default stays off.
- `futility_margin = 1/16` is the smallest multiplier the configuration accepts.
  It was chosen before measurement as a lower end point for the sweep, not
  calibrated, and it is not a bound on what a quiet move can gain.
- Selective search remains heuristic. A completed selective depth is not an
  exhaustive depth, and the tables record where scores and chosen moves diverged
  from the unpruned reference rather than assuming they did not.
- No native arm. The Rust search has its own ordering, PVS and work-accounting
  policy, so its node counts are not comparable with these.
