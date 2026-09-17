# Issue #60 bounded validation

**Decision: keep NMP and futility disabled by default.** NMP has promising native
throughput on this small sample, but these matches cannot establish strength
non-regression. Futility has substantial guard/evaluation overhead and the
Python both-flags paired sample includes an extra loss. No live jobs or checkpoint
opponents were changed.

[Predeclared plan](plan.json), [implementation contract](../../heuristics/SELECTIVE_SEARCH.md).
Parameters were fixed before measurement; no weight/margin tuning used these
positions or their symmetry variants. This is a bounded experiment, not a
production strength certification.

## Reproduction and identity

Baseline: `c31ab1625720a1dd0d256d56f83d048f218b3012`. Candidate file hashes are in
[evidence/source-manifest.json](evidence/source-manifest.json). The JSON results
also contain full effective configs, platform, starting revision and working
diff digest. That diff digest alone excludes then-untracked new files; the
source manifest is the complete implementation identity. No pruning parameters
are part of the weight-evolution genome.

```sh
cargo build --release --offline --manifest-path intransitive/rust_teacher/Cargo.toml
cargo test --offline --manifest-path intransitive/rust_teacher/Cargo.toml
cargo clippy --offline --manifest-path intransitive/rust_teacher/Cargo.toml --all-targets -- -D warnings
python -m unittest intransitive.tests.test_selective_search -v
python -m intransitive.benchmarks.selective.reproduce --reference --output reference.json
python -m intransitive.benchmarks.selective.reproduce --output native.json
python -m intransitive.benchmarks.selective.reproduce --backend python --output python.json
```

Use the repository Python environment (Python 3.11, NumPy/Numba from the lock).
For the old source, `git archive` the baseline into a temporary directory, copy
`reproduce.py` and `plan.json` there, and run the reference command with
`--archived-source c31ab1625720a1dd0d256d56f83d048f218b3012`. The baseline Rust
comparisons use its separately compiled binary and original wire protocol.

The final timing runs were sequential, with no other validation process running.
Python warmup was 17.21 seconds; the native run's Python fixture/kernel warmup
was 15.76 seconds and native process startup was 0.011 seconds. These startup
costs are separate from search timings. Time limits are cooperative, not hard
process deadlines. Shared-host noise and one timing sample per cell limit
interpretation; there is no confident latency estimate for deployment.

## Fixed-depth results

Opening, seeded 24-ply midgame and synthetic sparse endgame; original and
colour-exchanged symmetry; targets 3/5/6; four modes; 2-second per-search cap.
Proof leaves were disabled identically in these measurements to isolate the
heuristic tree. Tactical tests separately include bounded proofs. Each mode
has 18 searches per backend. A capped row is **not** a completed fixed-depth
measurement, and selective depth never means exhaustive coverage.

| Backend/mode | Completed / 18 | Median speedup on jointly completed rows | Total nodes, including capped rows | NMP cutoffs | Futility skips |
|---|---:|---:|---:|---:|---:|
| Native baseline | 17 | 1.000 | 1,590,572 | 0 | 0 |
| Native NMP | 18 | 1.224 | 1,170,814 | 4,103 | 0 |
| Native futility | 16 | 0.895 | 1,657,913 | 0 | 157,874 |
| Native both | 18 | 1.018 | 1,052,377 | 4,103 | 154,371 |
| Python baseline | 12 | 1.000 | 646,864 | 0 | 0 |
| Python NMP | 12 | 1.147 | 540,439 | 744 | 0 |
| Python futility | 10 | 0.808 | 307,295 | 0 | 49,505 |
| Python both | 12 | 0.989 | 353,927 | 744 | 62,883 |

The Python ratios have survivor bias: fewer hard cases completed with futility. They do **not** establish an overall speedup.
Across capped rows neither node totals nor different completed depths are
comparable fixed-depth costs. Raw [native](evidence/native.json) and
[Python](evidence/python.json) rows report each achieved depth, time, counters,
TT use and memory estimate for inspection.

Native NMP verification used 327,593 nodes (also 327,593 with both enabled).
Extra static evaluations numbered 4,357 / 280,812 / 40,473 for NMP / futility /
both. Python verification used 79,389 / 82,224 nodes for NMP / both, with
878 / 31,190 / 16,865 extra evaluations across the three modes. All work shares
the original move budget. Guard overhead is deliberately conservative and
particularly expensive in Python. Peak recorded TT payload estimates were
17.34 MB native and 14.28 MB Python; these exclude some allocation overhead
and are not RSS. Per-mode/cell memory is in the raw records.

Divergence classification: native had three and Python six cells with a
different completed depth under the cap. There were **no reported-score or
selected-action differences at equal completed depth** on this corpus. This
small, often material-balanced sample is weak evidence about tactics; it is
not an exhaustive equivalence claim. The explicit null-verification failure
fixture confirms fallback to ordinary search instead of accepting the probe.

## Equal-time paired play

Each candidate played baseline from all three stages, in both colours, with
20 ms per move and a 24-ply resource cap. Official-play rules continue beyond
modelling draws; budget caps are reported as unfinished. Each mode has six
games (three pairs), not six independent openings.

| Backend/mode | W / L / unfinished | Descriptive 95% Wilson interval, decisive score |
|---|---:|---:|
| Native NMP | 1 / 1 / 4 | 0.095–0.905 |
| Native futility | 1 / 1 / 4 | 0.095–0.905 |
| Native both | 1 / 1 / 4 | 0.095–0.905 |
| Python NMP | 1 / 1 / 4 | 0.095–0.905 |
| Python futility | 2 / 1 / 3 | 0.208–0.939 |
| Python both | 1 / 2 / 3 | 0.061–0.792 |

Intervals exclude unfinished games and are descriptive only: colour-pair and
opening dependence further weaken inference. None meets the predeclared 0.45
lower bound; we cannot assert equal-time strength or non-regression. No long
or unbudgeted tournament was run. Every move, achieved depth and node count is
archived. Retaining both defaults off is the issue's permitted rollout outcome.

## Correctness and known failures

- All 18 Python archived before/after comparisons match exactly in score,
  chosen move, PV, nodes, work, completed depth and TT hits, including reuse.
  See [before](evidence/reference-before.json) and [after](evidence/reference-after.json).
- All 18 native before/after comparisons likewise match score, move, PV, nodes,
  proof nodes, completion and TT usage: [native parity](evidence/native-baseline-parity.json).
- New tests exercise actual NMP/futility cutoffs, failed verification, both
  cancellation budgets and injected exceptions, state/history/key restoration,
  80-ply and repetition boundaries, official-win precedence, low mobility,
  sparse/race/retreat/capture exclusions, all four modes, PVS/aspiration/TT,
  pressure/unsupported scales, legal PVs/fallbacks and unpruned proof integrity.
- All 100 certified tactics pass in all four Python configurations. Native's
  existing 100-tactic and random-position parity suite passes. Targeted native
  tests exercise the same guarded cutoff fixture and incremental pressure undo.
- The issue-52 human-game roots and already-forced-loss control produce the
  same actions/scores in all four modes. **Five pre-existing depth-three hard
  assertions still fail**, identically at the archived baseline and candidate:
  `red_14_preserve_material`, `red_18_prevent_paper_fork`,
  `red_31_keep_goal_defender`, and `unique_last_defender` in both colours.
  Original tests were not modified or weakened. See
  [baseline log](evidence/human-before.log) and [candidate log](evidence/human-after.log).
  These failures prevent any broad claim of tactical safety; fixing them is
  separate from introducing opt-in pruning.

The final 73-test integration suite, seven teacher/pressure tests, two final
cancellation tests, ten native tests and Clippy passed. The checked-in logs
distinguish passing implementation/regression suites
from this known-failing historical suite. Rollback and compatibility are
specified in the implementation contract. No exhaustive teacher labels are
accepted from selective configurations.
