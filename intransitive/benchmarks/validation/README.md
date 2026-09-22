# Held-out acceptance of the evolved defaults (#56)

The [acceptance protocol](../../validation/README.md) was frozen at revision
`518401a` and run on 21 September 2026 (Australia/Adelaide) on an otherwise
idle host: macOS 15.6.1, eight logical CPUs, 8 GiB RAM, Python 3.11.4, NumPy
2.4.6, Numba 0.67.0, llvmlite 0.49.0. Every process ran at `nice -n 19`.
Plan `ebab4c11…`, 186 official games on six held-out starts, all replayed.

**Outcome: retain existing defaults, for all three candidates.** Nothing was
promoted, nothing was reverted, and no default, generator, teacher or training
label was changed.

| Candidate | Comparison | Held-out margin (depth 3) | Equal time | Tactics | Outcome |
| --- | --- | ---: | ---: | --- | --- |
| `adopted-hour-search` — the configuration in force | prior incumbent | **+.389** (.556 vs .167), direct 8–2–2 | +.278 (.444 vs .167), direct 8–2–2 | 0 new failures, 3 fixed | retain-defaults |
| `endgame-provisional` | adopted default | −.028, direct 5–7–0 | −.056, direct 4–5–3 | **5 new failures** | retain-defaults |
| `variable-material-selected` | adopted default | **−.792** (.125 vs .917), direct 1–11–0 | −.708, direct 2–10–0 | **19 new failures, one of them a proof** | retain-defaults |

The shipped defaults are therefore *confirmed against the weights they
replaced*, not promoted: adoption additionally requires the two paired
intervals to be disjoint, and the plan said in advance that this schedule
cannot produce that. With six opening-line clusters the harness's conservative
Hoeffding radius is **.554**, and separating the declared .10 gain would need
about **738 clusters**. A run of this size can confirm, recommend reverting, or
be inconclusive; it cannot promote. The two challengers are rejected on
evidence that does not need separation: they lose their direct paired records
and they lose certified tactical fixtures their comparison solves.

## What was frozen before any engine started

Corpus seed 54, 24 generated lines, 73 positions, split into search, validation
and held-out **by generated line** before any selection, with symmetry families
deduplicated across the whole corpus. Six held-out starts, two per stage, each
from a different line. Entrants are #53 genomes; the three candidates come from
the #55 exports with their file digests recorded:

| Candidate | Export | Version |
| --- | --- | --- |
| `adopted-hour-search` | `benchmarks/evolution/hour-search-20260917/provisional-genome.json` | v1, re-expressed with `material: 100` |
| `endgame-provisional` | `benchmarks/evolution/endgame-search-20260917/provisional-genome.json` | v1, re-expressed with `material: 100` |
| `variable-material-selected` | `benchmarks/evolution/variable-material-20260917/selected-genome.json` | variable v3 |

Protocols: fixed completed depth **3** with a 30-second per-move safety
ceiling, and equal move time at **0.25 s**, both with a 192-ply and 150-second
game cap, the harness's .8 completion gate, and separate leaderboards.
Thresholds: gain .10, regression .10, disjoint intervals required, zero new
certified tactical failures, at most one median ply lost at equal time, at
least .2 of the comparison's label throughput for a teacher proposal.

Held-out starts were played once, by these games. Saturation preflights,
determinism, native parity and the cost ladder used search-pool positions.

## Held-out strength

Fixed completed depth 3, 60 games, 6 starts, 59 distinct trajectories:

| Entrant | W/L/unfinished | Completion | Lower win points | Eligible | Median move |
| --- | ---: | ---: | ---: | --- | ---: |
| `adopted-hour-search` | 20/14/2 | .94 | **.556** | yes | .317 s |
| `pressure-10-archive` | 13/10/1 | .96 | .542 | yes | .050 s |
| `endgame-provisional` | 19/15/2 | .94 | .528 | yes | .340 s |
| `prior-incumbent` | 4/17/3 | .88 | .167 | yes | .035 s |

Equal move time, 0.25 s, 60 games, 60 distinct trajectories, run single-worker
so the measurement is of search and not of CPU contention:

| Entrant | W/L/unfinished | Completion | Lower win points | Eligible | Completed depths |
| --- | ---: | ---: | ---: | --- | --- |
| `pressure-10-archive` | 16/6/2 | .92 | .667 | yes | mostly 3–4 |
| `adopted-hour-search` | 16/14/6 | .83 | .444 | yes | mostly 2–3 |
| `endgame-provisional` | 14/15/7 | .81 | .389 | yes | mostly 2–3 |
| `prior-incumbent` | 4/15/5 | .79 | .167 | **no** | mostly 4 |

The prior incumbent misses the .8 completion gate here, which blocks adoption
on this protocol without touching the fixed-depth comparison. It reaches two
plies deeper per move than the configuration that beats it, which is the point
of running the protocol at all: the evolved modules are not winning because
they search further.

The variable-material candidate has its own schedule, because material mode is
a fixed evaluator mode rather than a scale; it faces the configuration it would
displace and the fixed variable baseline the #55 runs used:

| Entrant | Depth 3 | Equal time |
| --- | --- | --- |
| `adopted-hour-search` | 11/1/0, lower .917 | 10/2/0, lower .833 |
| `variable-initial-material-baseline` | 10/2/0, lower .833 | 10/1/1, lower .833 |
| `variable-material-selected` | 3/21/0, lower .125 | 3/20/1, lower .125 |

## Attribution

Each changed module restored on its own to the prior incumbent value, against
the candidate itself, six games each on three held-out starts at depth 3:

| Restored gene | Restored entrant | `adopted-hour-search` | Carries the result |
| --- | --- | --- | --- |
| `attack` → 0 | 1/5/0 | 5/1/0 | yes |
| `defence` → 0 | 1/4/1 | 4/1/1 | yes |
| `advantage` → 25 | 3/1/2 | 1/3/2 | **no** — restoring it scored better |

The held-out gain comes from enabling the attack and defence modules, which the
prior incumbent had switched off, and not from the evolved advantage
coefficient: at 23.967 versus 25 the optimizer tuned a digit that the games
cannot see. The advantage pair is the one schedule where both entrants missed
the completion gate (four of six games decided), so that row is a direction and
not a measurement. Six games per gene cannot separate anything either; the
attack and defence rows agree with the tactical results below, where restoring
`defence` loses four certified fixtures.

## Tactical safety

All 112 independently certified fixtures — 100 puzzles and the 12 regressions
from the analysed game — for every entrant, at two profiles: the committed
suites' budget (depth 3, 2,000,000 work, 30 s) and the shipped default budget
(depth 3, 200,000 work, 1 s). "Strict" additionally requires the search to
finish inside the profile's budget, which is what the committed suites assert.

| Configuration | Certified pass | Strict | Budget-stopped | Shipped pass | Shipped strict | Shipped stopped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `prior-incumbent` | 107 | 107 | 0 | 107 | 107 | 0 |
| `adopted-hour-search` | **110** | 108 | 3 | 111 | **99** | **13** |
| `pressure-10-archive` | 109 | 109 | 0 | 109 | 105 | 5 |
| `endgame-provisional` | 106 | 104 | 3 | 107 | 99 | 13 |
| `variable-initial-material-baseline` | 94 | 91 | 5 | 94 | 83 | 21 |
| `variable-material-selected` | **93** | 89 | 5 | 97 | 83 | 24 |

Four of the committed game regressions already fail on `master` with the
shipped defaults — `red_14_preserve_material`, `red_18_prevent_paper_fork`,
`red_31_keep_goal_defender` and `blue_19_create_paper_fork`, the middle one by
exhausting even the 2,000,000-work budget. This run reproduces exactly those
four as the adopted configuration's strict failures and changes none of them;
they are the state of `test_game_blunders` before this work, not a finding of
it.

The adopted weights are tactically *better* than the ones they replaced when
the search can afford itself: they solve three fixtures the prior incumbent
misses (`red_18_prevent_paper_fork` and both colours of
`unique_last_defender`, and a fourth, `red_14_preserve_material`, at the
shipped budget) and newly fail none. They are tactically *worse* in
strict terms, because 13 fixtures exhaust the shipped work budget where the
prior incumbent exhausts none.

`endgame-provisional` newly fails five fixtures its comparison solves — both
colours of a forking capture, both colours of the last-defender regression, and
`red_14_preserve_material` — while fixing none. `variable-material-selected` newly fails 19,
including `blue_32_take_winning_run`, whose certificate is a proof of a forced
win: that is the failure mode #52 names, material judgement bought at the price
of a provable win.

## Correctness and parity

Every entrant round-trips its genome, re-derives the same evaluation identity,
returns identical actions, scores and work from two fresh engines, leaves its
input state unmodified, and passes #53's saturation preflight with no clipped
or saturated fixture. For the six entrants whose genes the native backend
supports, Python and Rust agree **exactly**: maximum score difference 0.0 and
100% action agreement at depths 1 and 2. The two entrants with a nonzero
overload gene are reported `unsupported` with the reason, never skipped
silently.

## Practical cost

Three search-pool positions per configuration, each profiled in its own spawned
process. The ladder stops a position at the first depth it cannot finish inside
120 seconds, and reports that rather than extrapolating.

| Configuration | Static eval | Work/eval | Depth 3 | Depth 4 | Depth 5 | Depth 6 | Labels/hour at depth 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prior-incumbent` | 9.1 µs | 22 | 0.13 s | 0.49 s | 7.31 s | 21.28 s | **492** |
| `adopted-hour-search` | 118.9 µs | 4,550 | 0.58 s | 2.44 s | 30.07 s | 118.34 s (1 of 3) | **120** |
| `endgame-provisional` | 162.2 µs | 4,032 | 0.68 s | 2.87 s | 32.81 s (2 of 3) | none | 110 |
| `variable-material-selected` | 183.6 µs | 5,302 | 0.88 s | 4.18 s | 42.40 s (2 of 3) | none | 85 |

Depth actually reached when every configuration is given the same move time:

| Configuration | 0.05 s | 0.25 s | 1 s | 5 s | Shipped limits |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prior-incumbent` | 2 | 3 | 4 | 5 | **3** |
| `adopted-hour-search` | 2 | 2 | 3 | 4 | **1** |
| `endgame-provisional` | 2 | 2 | 3 | 4 | 1 |
| `variable-material-selected` | 1 | 2 | 3 | 4 | **0** |

Startup was 8.3–13.2 s and peak process RSS 332–415 MiB per configuration.

### The finding that does not depend on which weights win

`SearchConfig()` ships `max_depth` 3 with `node_limit` 200,000, a work cap
sized for the core evaluator. Under those exact limits the adopted evaluation
is work-stopped at a **median completed depth of 1** on all three profiled
positions, in about 7 ms, and 13 of the 112 certified tactical fixtures
exhaust the budget. The weights it replaced complete depth 3 on every one of
them. Every default-configured bot — `intransitive --opponent alphabeta`,
`intransitive-analyze`, anything constructing `AlphaBetaPlayer()` — is
therefore searching one ply, not three. The committed tactical suites raise the
cap to 2,000,000 work, so they never see it, and the #54/#55 tournaments always
declared their own limits, so they never saw it either.

This run does not fix it, because #56 is not allowed to change a default. The
exported preset
[`heldout-adopted-hour-search.json`](../../heuristics/configs/heldout-adopted-hour-search.json)
is the opt-in path: the same evaluation with a work limit **measured** to
complete its own depth (`node_limit` 80,900,000, three times the worst
measured depth-3 search of 26.9 million work units, and a 5-second ceiling).
Its [record](../../heuristics/configs/heldout-adopted-hour-search.record.json)
carries the genome, the plan hash, the measured limits and the acceptance
outcome, so it cannot be mistaken for an accepted replacement.

## Teacher and checkpoint boundaries

The teacher labels at depth 5 (`supervised_minimax.TEACHER`). The adopted
evaluation produces **120 labels/hour** against the prior incumbent's 492 — a
ratio of .243, which clears the predeclared .2 floor but means four times the
wall clock for the same dataset. No teacher rollout is proposed or performed
here. If one is ever approved it additionally requires new labels to record the
evaluator/config identity that produced them, the dataset provenance boundary
to be written down, and the combination rule for old and new labels to be
stated; existing labels are never relabelled silently. No checkpoint-opponent
or promotion-gate change is proposed: that would require re-evaluating the
incumbent and current models under the same new suite first, with backups and
a rollback path.

## Compute, resume and evidence

186 games, all final, 5,210 seconds of known child CPU, 469 MiB parent peak
RSS. The run took **5,425 seconds** of wall clock in its final invocation;
`strength-depth` also carries a 570-second cancelled invocation from an earlier
run of the byte-identical schedule, whose 46 fsync'd journals this run resumed
rather than replayed, so the total measured wall clock across invocations is
about **100 minutes**. Per-experiment invocations, worker counts and reused
process handshakes are in each experiment's `invocations.json`.

Matches go through the unchanged #54 harness with the #55 resident engine pool
in front of it: a fresh player, evaluator and transposition table are still
built for every move, and only compiled machine code survives between games.
Without it each game paid about 11 seconds of Numba startup twice, which would
have cost more than the games. Equal-time experiments always run single-worker.

- [Frozen plan](evidence/plan.json) — corpus, splits, starts, entrants, protocols, thresholds, detectable effect
- [Decision](evidence/decision.json) — every margin, direct record, ablation, tactical comparison and cost verdict
- [Run summary](evidence/summary.json) and [replay audit](evidence/verification.json)
- [Tactical profiles](evidence/tactics.json), [cost profiles](evidence/cost.json) and [correctness/parity checks](evidence/parity.json)
- [Full archive](evidence/runs.tar.xz): plan, state, evidence, every experiment manifest, report, invocation record and complete move journal
- [Archive checksum](evidence/runs.sha256)

```sh
mkdir -p /tmp/acceptance-evidence
tar -xJf intransitive/benchmarks/validation/evidence/runs.tar.xz -C /tmp/acceptance-evidence
uv run --locked python -m intransitive.benchmarks.validation.verify \
  /tmp/acceptance-evidence/final3
```

The audit replays all 186 games through the rules engine, rebuilds the plan
from its own record, and re-applies the predeclared rule to the stored
evidence; a decision that no longer follows from its evidence is an error.

## Limitations

- Six opening-line clusters. The intervals are `[0, 1]` for every entrant and
  are conditional on this frozen corpus. No result here separates anything, and
  the direction of a margin is not a significance claim. This was stated in the
  plan before the run, not discovered in it.
- Attribution has six games per gene. It agrees with the tactical results and
  with the module semantics, and it is still six games.
- One host, one runtime, one corpus seed. Equal-time results depend on OS
  scheduling; the fixed-depth protocol is the reproducible one.
- A pilot of the same protocol at half the starts was run during development
  and discarded. No threshold was changed after seeing it; what changed was
  structural — the variable-material candidate had no schedule containing the
  configuration it must displace, "is this the configuration in force" was
  read from the declared comparison rather than from the genes, and the starts
  per stage went from one to two. The delivered run is the frozen plan above.
- Depths 5 and 6 are priced, not played. A single depth-5 move costs the
  adopted evaluation about 30 seconds and a depth-6 move about two minutes, so
  full games at the teacher's depth were not affordable; the ladder reports
  what they cost instead of extrapolating a result from cheaper games.
