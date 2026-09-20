# Issue #70 bounded validation

**Decision: keep `certificate_cutoff_enabled`, `certificate_guard_enabled` and
`race_reduction_enabled` disabled by default.** The cutoff is the only one of
the three with a measurable benefit, and only on the stage where forced runs
actually exist. The race reduction fires and changes nothing. The guard is the
one option these matches *do* separate from its baseline, and it separates the
wrong way: at equal time it loses 16–31 with a 95% interval entirely below even,
because refusing to prune costs more than the branches it protects are worth at
this time control. No live generator, trainer or checkpoint opponent was
changed.

The one change that is not conditional is a bug fix: `certificate_enabled`
raised `KeyError: 'pv'` at the first leaf that certified, so it had never run
inside the Python search at all. The `leaf` rows below are the first
measurement of that option working.

[Predeclared plan](plan.json), [implementation contract](../../heuristics/CERTIFICATE_SEARCH.md).
Every threshold was fixed before measurement. This is a bounded experiment on
one machine, not a strength certification.

## Reproduction and identity

Baseline: `8d22a1117f9fcb4fdc89057f13c5f48255ad5c26`. Both runs start from
revision `d831229ff00272d7aa479ab882628c91e03e9a1a`; each report records the
platform, that revision, the working diff digest and the full effective
configuration of every search.

The predeclared run ran on a clean tree (`e3b0c442…`, the SHA-256 of nothing).
The extended run ran with the `--extended` plumbing, the plan's `extended`
block, two test refinements and the predeclared report itself still
uncommitted, and records their digest as
`abb0ac0c0d7b4434193201369ca9e60f1802279c5f60df1dcbce2734ccdc90f0`. A digest of
a working diff is only checkable against the tree that produced it, and that
tree no longer exists, which is why the source manifest below exists: it hashes
each implementation file directly and does not depend on what was staged.

```sh
python -m unittest intransitive.tests.test_certificate_bound -v
python -m unittest intransitive.tests.test_clear_run intransitive.tests.test_rust_teacher
cargo build --release --offline --manifest-path intransitive/rust_teacher/Cargo.toml
cargo test --offline --manifest-path intransitive/rust_teacher/Cargo.toml
python -m intransitive.benchmarks.certificate.reproduce \
    --output intransitive/benchmarks/certificate/evidence/report-predeclared.json
python -m intransitive.benchmarks.certificate.reproduce --extended --sections oracle paired \
    --output intransitive/benchmarks/certificate/evidence/report-extended.json
```

Use the repository Python environment (3.11, NumPy/Numba from the lock).
Platform `macOS-15.6.1-arm64`. Python warmup was 11.79 s (12.32 s on the
extended run) and is excluded from every search timing. Predeclared section
wall times: probe 0.26 s, fixed 728.6 s, trigger 199.1 s, oracle 5.9 s, paired
742.0 s; extended: oracle 2.8 s, paired 1,063.9 s. Runs were sequential with no
other validation process running. Time limits are cooperative, not process
deadlines, and there is one timing sample per cell.

[evidence/source-manifest.json](evidence/source-manifest.json) hashes every
implementation file, which a diff digest cannot do for files that were
untracked when a run started. [evidence/tests.log](evidence/tests.log) is 303
Python tests after merging master's #71,
[evidence/rust-tests.log](evidence/rust-tests.log) is 30 Rust tests, and both
pass.

[evidence/clippy.log](evidence/clippy.log): `cargo clippy --offline
--all-targets -- -D warnings` under Rust 1.74.1 reports two lints, both on
lines this work does not touch (`clear_run.rs`'s starting-square comparison,
which predates the 60 lines inserted above it, and `routes.rs:452`, untouched
entirely). They predate the branch and were left alone rather than fixed in
passing.

## What each option is measured against

Every option plays the same search with that option removed and nothing else
changed. There is no cross-option comparison; the arms of different pairs do
not share an engine.

| Option | On | Off |
| --- | --- | --- |
| `leaf` | `certificate_enabled` | no certificate at all |
| `cutoff` | `certificate_cutoff_enabled` + leaf | leaf only |
| `guard` | `certificate_guard_enabled` + NMP/futility/LMR + PVS | the same without the guard |
| `race` | `race_reduction_enabled` + LMR, PVS off | LMR only, PVS off |

`guard` keeps PVS because narrow windows are what make null-move and futility
reachable at all; `race` turns it off because LMR is only consulted where PVS
is not already probing a child. Shared engine: depth target 3/5/6, 8-second
cap, `proof_depth=2`, `proof_nodes=64`, ordering, compiled ordering,
aspiration, 50,000 table entries.

## The probe's own cost

| Corpus | Positions | Gate passes | Certificates | Hidden |
|---|---:|---:|---:|---:|
| Reachable random legal play | 1,480 | 19 (1.28%) | 6 | **0** |
| Sparse synthetic endgames | 220 | 123 (55.91%) | 101 | **0** |

A gate call costs 0.627 µs; a `certify` call costs 8.131 µs on a 20-piece
board. Without the gate a probe is two `certify` calls, 16.262 µs. With it, one
gate plus a 1.28% chance of one `certify`: 0.731 µs, **22.2× cheaper**, and
bit-identical, since the gate never hides a certificate. Separately, 200,000
random boards and the whole reachable corpus were checked pairwise against
`certify` for both sides with no hidden certificate, and a Rust test does the
same on 20,000 boards.

This is the measurement that makes an interior probe possible at all. It is
also why `certificate_cutoff_min_depth` turns out to be the weaker of the two
cost controls.

## Fixed-depth searches

24 rows per option: four stages (opening, seeded midgame, sparse endgame, a
certified seven-ply run), each in its original and colour-exchanged form, at
depth targets 3, 5 and 6, under an 8-second cap. A capped row is not a
completed fixed-depth measurement.

| Option | Completed on/off | Nodes (all rows) | Work (all rows) | Nodes (jointly completed) | Work (jointly completed) |
|---|---:|---:|---:|---:|---:|
| `leaf` | 16 / 14 | −24.9% | −12.1% | **−26.0%** | **−55.3%** |
| `cutoff` | 16 / 16 | −3.6% | +5.2% | **−6.4%** | **−2.2%** |
| `guard` | 15 / 16 | +1.0% | +0.3% | +0.0% | +1.6% |
| `race` | 17 / 18 | +1.7% | +6.3% | −0.0% | +0.0% |

The leaf certificate is the large effect, and it is concentrated exactly where
it should be. On the `certified` fixture — a forced seven-ply run with every
piece more than three steps from its corner — the arm without it **never
finishes**: at a depth-5 target it spends the whole 8-second cap and reaches
depth 4, and at a depth-6 target it reaches depth 5. The arm with it resolves
at depth 1 in **64 nodes and under 0.01 s**, because the bounded proof reaches
the run from a leaf. The node counts of the unfinished arm (71,439 and 72,944)
are wall-clock artifacts and should not be read as a ratio; what is solid is
"did not finish in 8 s" against "finished immediately".

The cutoff then has little left to do on that fixture: the leaf term has
already ended the search before any interior node exists. Its 157 cutoffs come
from the sparse stage. On jointly completed rows it is a 6.4% node and 2.2%
work reduction with **no score change** — every one of its three score changes
is on a row where both arms hit the cap.

The guard costs about 1% of nodes and changes nothing measurable: it makes the
search more conservative, which is what it is for. Its 1,054 refusals are all
null-move/futility eligibility; `unreduced` is zero here because PVS is on and
LMR is therefore never consulted. The reduction half of the guard is
demonstrated in `test_no_child_of_a_certified_node_gets_a_shallower_look`
instead, where turning PVS off makes 27 prevented reductions visible and
`lmr_reduced` falls by exactly 27.

The race reduction fired — 221 quiet nodes, 210 reductions the index rule would
have refused — and produced no measurable change either way. Its single score
change is a row where the candidate ran out of time partway through depth 5
while the reference completed it; that is a cap artifact, not a demonstrated
tactical loss, and nothing here shows it losing a win.

## Where the cutoff is worth invoking

The sweep over `certificate_cutoff_min_depth`, each setting against the
leaf-only reference at the same target:

| `min_depth` | Nodes | vs reference | Work | vs reference | Cutoffs |
|---:|---:|---:|---:|---:|---:|
| 1 | 859,339 | −10.2% | 482,575,253 | −9.0% | 236 |
| 2 | 878,546 | −8.2% | 460,420,618 | −13.2% | 78 |
| 3 | 916,374 | −4.2% | 488,441,175 | −7.9% | 13 |
| 4 | 941,094 | −1.6% | 496,545,494 | −6.3% | 0 |

**Do not read that table.** `min_depth=4` produced *zero* cutoffs and still
shows a 1.6% node and 6.3% work reduction, which is impossible as an effect and
is entirely the 8-second cap. The opening and midgame stages contribute zero
cutoffs at every setting while swinging between −23.5% and +3.8%. Aggregated
over capped rows, this section measures the machine.

The sparse stage is the only one that completes within the cap and the only one
where cutoffs occur, so it is the whole of the usable signal:

| `min_depth` | Nodes | vs reference | Work | vs reference | Cutoffs |
|---:|---:|---:|---:|---:|---:|
| 1 | 664,937 | −10.3% | 22,687,915 | −4.4% | 236 |
| 2 | 693,944 | −6.4% | 22,778,961 | −4.0% | 78 |
| 3 | 719,671 | −3.0% | 23,329,747 | −1.7% | 13 |
| 4 | 741,612 | +0.0% | 23,742,663 | +0.0% | 0 |

Reference: 741,612 nodes, 23,734,644 work. Both node and work reduction shrink
monotonically as the threshold rises, and both vanish once the threshold is
high enough to stop the cutoff firing at all. The gate has already made the
probe cheap enough that raising the depth threshold only costs cutoffs; the
threshold is the weaker control and 1 is the better setting on this evidence.

The default was fixed at 2 before measurement and is left there. One stage's
uncapped rows is thin evidence for a default, and the setting that maximises
cutoffs is also the one that probes the most nodes in positions where no runner
is ever in range — which the opening and midgame rows are too noisy to price.
Anyone enabling the cutoff for endgame work should set it to 1 and say so.

## Oracle check

Every cutoff-derived mate score, re-searched full width to the claimed distance
with the certificate off. Alpha-beta returns the true minimax value at a fixed
depth, so a mate-range score confirms the claim and anything else refutes it.

| Run | Checked | Confirmed | Refuted |
|---|---:|---:|---:|
| Predeclared (220 positions, ≤4 plies) | 3 | 3 | **0** |
| Extended (700 positions, ≤5 plies) | 13 | 13 | **0** |

The predeclared corpus produced only three checks, which answers nothing, so
the extended run widens it. Widening a refutation search after it finds no
refutations strengthens the claim rather than tuning it, and both runs are
published. Thirteen is still thin, and it is thin for a structural reason:
verification is full width, so it stops at five plies, while most certified
runs are longer than that. This section checks the mate-distance arithmetic of
the cutoff, not the certificate; the certificate's own corpus — 1,244 confirmed
and 0 refuted — is in `test_clear_run.py`, and
`test_a_cutoff_result_survives_a_full_width_search` repeats this check in CI.

## Equal-time paired games

Colour-swapped pairs from distinct reachable starts, the candidate's option on
one side and off on the other, equal wall clock per move. Unfinished games are
excluded from the win-rate denominator and are never official draws. Intervals
are Wilson 95% on decisive games only.

Predeclared run — 0.1 s per move, 60-ply cap, 16 starts, 32 games per option:

| Option | W | L | Unfinished | Decisive Wilson 95% |
|---|---:|---:|---:|---|
| `leaf` | 4 | 4 | 24 | [0.215, 0.785] |
| `cutoff` | 4 | 3 | 25 | [0.250, 0.842] |
| `guard` | 4 | 4 | 24 | [0.215, 0.785] |
| `race` | 5 | 3 | 24 | [0.306, 0.863] |

Three quarters of the games hit the ply cap, so this settles nothing. The
extended run raises the cap and the number of starts:

Extended run — 0.05 s per move, 220-ply cap, 24 starts, 48 games per option:

| Option | W | L | Unfinished | Decisive Wilson 95% |
|---|---:|---:|---:|---|
| `leaf` | 26 | 18 | 4 | [0.444, 0.723] |
| `cutoff` | 25 | 15 | 8 | [0.470, 0.758] |
| `guard` | 16 | 31 | 1 | **[0.222, 0.483]** |
| `race` | 27 | 19 | 2 | [0.443, 0.717] |

Now that almost every game finishes, one result separates from its baseline:
**the guard is measurably weaker at equal time**, 16–31 over 47 decisive games
with a 95% interval entirely below even. The mechanism is not mysterious and is
not a defect — the guard refuses null-move and futility on certified nodes, so
it searches more per move, and at 0.05 s per move that thoroughness does not
pay for itself. This is evidence about this time control, not about the guard's
correctness.

`leaf`, `cutoff` and `race` all lean above even and none separates: every
interval contains 0.5. The cutoff's candidates took 446 cutoffs across their
games, so the option was live rather than dormant. Forty-eight games from
24 starts is a small sample and these intervals are wide.

Both sides reached a median completed depth of 2 at these time controls, so
these matches compare shallow searches; nothing here speaks to deeper play.

## Reproducibility after #71

These measurements were taken at `7c17569`, before #71's static exchange
evaluation merged into master. Every option #71 adds defaults to off, so the
engine configured here should behave identically — checked rather than
asserted, by
[`post_merge_check.py`](post_merge_check.py) against the recorded rows:

```sh
python -m intransitive.benchmarks.certificate.post_merge_check
```

[evidence/post-merge-check.log](evidence/post-merge-check.log): every completed
row reproduces to the node, and the one row that differs is the `certified`
depth-5 arm that hit the cap — which is the point made above about capped rows,
landing on this document's own numbers.

The merge also surfaced one genuine interaction. #71 repaired
`test_material.py`'s configuration loop so that it actually runs, and the
repaired loop flips every field in turn, including `race_reduction_enabled`,
whose interlock with `lmr_enabled` then rejects the flip. It now moves its
sibling with it, as `variable_material_linear` already did.

## Limitations

- One machine, one timing sample per cell, and an 8-second cap that most depth
  5 and 6 rows hit. The aggregate fixed-depth and trigger tables are dominated
  by that cap; only the jointly-completed columns and the sparse stage carry a
  usable signal.
- The sparse endgame corpus is synthetic. It is a sample of the positions the
  certificate is about, not a sample of play, and the reachable corpus shows
  why: 6 certificates in 1,480 positions of random legal play.
- Median completed depth 2 in the paired games, on both sides, at 0.05 s per
  move. The guard's loss is a statement about shallow equal-time play and
  nothing else; the three inconclusive intervals are 48-game samples and are
  wide enough to hide a real effect in either direction.
- `race_reduction_enabled` has no native counterpart, since the Rust search has
  no late move reductions. Its rows are Python only.

## Adoption

All three new options stay disabled.

The measured case for the cutoff is real but narrow: endgames with a runner in
range, where it is worth about 6% of nodes at no cost in accuracy, and nothing
anywhere else. Anyone enabling it for endgame analysis should also set
`certificate_cutoff_min_depth=1`.

The guard is refused on evidence rather than on caution. It behaves exactly as
designed — 1,054 refused pruning decisions, 27 refused reductions in the test
that can see them — and at equal time that thoroughness loses. If it is ever
wanted, it wants a time control where a deeper search is affordable, and that
is a different experiment.

The race reduction is refused for the opposite reason: it fires, 210 times in
the fixed-depth rows, and nothing moves. The admissibility argument behind it
holds; the reduction it licenses is simply not worth having at these depths.

The `pv` fix is unconditional, and `certificate_enabled` now works. Whether it
should be *on* is a separate question this run does not settle, though its
`certified`-fixture result — an 8-second cap reached without finishing, against
64 nodes and depth 1 — is the strongest evidence in this document, and its
paired result (26–18) is the least bad of the four.
