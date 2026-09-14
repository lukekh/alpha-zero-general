# Bounded compiled proof search (#38)

The default proof (depth two, 64 nodes) now stays inside Numba for its traversal.
Goal-first ordering uses two precomputed stable action orders. At the final ply,
the specialised path applies the move to a temporary piece plane and checks
corner/stalemate wins without constructing history that cannot affect the
remaining zero-ply result. Earlier plies still use the existing `Board`
transitions and exact draw histories. The public board API is unchanged; this
is independent of the compact-board work in #40.

## Semantic and cancellation contract

- Official wins precede modelling draws. Draws and unproved horizons both have
  neutral utility *inside* terminal-only proof search. The public proof reports
  only a certified win/loss or **unknown**, never an exact draw inferred from a
  horizon. Exhaustion discards the entire unfinished certificate.
- Mate scores, distances, stable tie ordering and principal variations are
  preserved. No heuristic weight, binary clear-run rule, main-search depth,
  pruning rule or transposition identity changes.
- Accounting is unchanged: one work per visited node, one per traversed edge,
  and one for the root no-win bound query. `nodes` includes proof nodes;
  `main_nodes = nodes - proof_nodes`. A specialised last-ply child still counts
  as a node and an edge although its history allocation is omitted. Exhaustion
  may charge the final edge before the node limit is observed, as before.
- A production native call is capped at 64 visited nodes / 129 work. The caller
  checks the deadline before and after, incorporates actual charged work and
  nodes, and raises before publishing a result if time expired. Work exhaustion
  never exceeds the shared cap. Completed main-search siblings remain available
  through the existing anytime selection policy.
- All configured proof depths 1–8 are supported. Budgets above 64 proof nodes
  use `prove_reference`, with its per-operation cancellation; they are **not**
  silently truncated. Zero depth/node budgets retain disabled/unknown behavior.
  Custom game adapters, cold kernels and nonstandard array layouts also use
  the reference. The implementation does not claim a speedup for these cases.
- Kernel compilation happens in preparation before the normal move budget is
  created. An externally supplied budget does not trigger the new compilation.
  The new recursive kernel is compiled per process: loading its disk cache
  crashed in the pinned Numba environment, so that cache is disabled. Stable
  helper kernels retain their existing cache. Setup time is reported separately.

## Reproduction

The clean baseline is `216cdc2d3580381fc72755cd42a0fb15286855c1` (after #37).
[`baseline.tar.gz`](evidence/baseline.tar.gz) archives its Intransitive Python
sources and root Python adapters, configurations, tactical fixtures, exact replay inputs, `pyproject.toml`
and `uv.lock`. [`baseline-manifest.json`](evidence/baseline-manifest.json) records
per-file and archive SHA-256 hashes. Large historical model/evidence artifacts
are excluded; the source commit retains them. The replay is the original
[`game79.pgn`](../search_budget/game79.pgn), with its verified final-state hash.

```sh
uv run --locked python intransitive/benchmarks/proof/reproduce.py \
  --repeats 3 --seconds 1 > results.json
uv run --locked python -m unittest \
  intransitive.tests.test_compiled_proof \
  intransitive.tests.test_tactics intransitive.tests.test_game_blunders \
  intransitive.tests.test_anytime_search intransitive.tests.test_heuristics \
  intransitive.tests.test_search_performance intransitive.tests.test_play \
  intransitive.tests.test_record
```

Four variants isolate the changes: retained Python reference, reference with
only its sorting replaced, compiled traversal with ordinary history transitions,
and compiled traversal with last-ply specialisation. Each run uses a fresh
player/table. Compilation is warmed before timing, and variant order rotates
between repetitions. Fixed-depth runs use depth four with sufficient time/work;
equal-time runs keep maximum depth 20 and allow one second. Proof depth, node
cap, weights, table capacity and all other settings are identical. The opening
and plies 27/35 are controls where existing proof queries are already cheap;
ply 61 is the proof-heavy late position.

[`results.json`](evidence/results.json) records full configurations, source
hashes, every elapsed/proof-module time, proof/main nodes, charged work, table
memory, timeout overshoot, completed/selected/partial depth, root coverage,
selected moves and independent tactical objectives. Whole-process peak RSS
includes compilation and all variants, and is not a per-variant allocation
measurement. Native cancellation intervals are sampled separately after the
search runs. These are single-machine measurements, not a strength claim.

## Measured results

Three repetitions on macOS 15.6.1 arm64 / Python 3.11.4, pinned dependencies in
`uv.lock`. Medians below include completed iterative depths 1–4 and final
explanation work. Every fixed-depth variant and repetition has **identical
score, move, PV, proof/main nodes, work and tactical outcome**.

| Position | Reference total / proof | Ordering only total / proof | Compiled total / proof | Specialised total / proof |
| --- | ---: | ---: | ---: | ---: |
| Opening | 0.370 / 0.027 s | 0.372 / 0.027 s | 0.370 / 0.016 s | 0.378 / 0.017 s |
| Before Red 14 | 0.506 / 0.033 s | 0.484 / 0.032 s | 0.465 / 0.019 s | 0.462 / 0.019 s |
| Before Red 18 | 0.546 / 0.049 s | 0.555 / 0.051 s | 0.547 / 0.031 s | 0.545 / 0.031 s |
| Before Red 31 | 2.567 / 2.316 s | 2.668 / 2.372 s | 0.740 / 0.485 s | 0.601 / 0.354 s |

The proof-heavy position improves **4.27× in total elapsed time** and **6.55×
in proof time**, at the same 185,459 proof nodes, 5,527 main nodes and 501,942
work. Traversal compilation supplies most of the improvement; last-ply
specialisation reduces total elapsed another 18.8% relative to compiled general
traversal. Sorting alone does not show a reliable improvement (its late-position
median is 3.9% slower); retain it as part of the native traversal, without
claiming an independent sorting speedup.

The cheap controls show limited benefit: specialised total elapsed changes by
+2.0% at the opening, −8.5% before Red 14 and −0.3% before Red 18. The small
opening regression and run-to-run variation remain visible; this is a targeted
late-position improvement, not a uniform engine speedup.

| Position | Main nodes | Proof nodes | Work | Approximate table bytes |
| --- | ---: | ---: | ---: | ---: |
| Opening | 6,107 | 4,325 | 265,937 | 1,196,340 |
| Before Red 14 | 7,742 | 5,096 | 309,264 | 1,721,469 |
| Before Red 18 | 10,023 | 8,170 | 319,654 | 2,027,503 |
| Before Red 31 | 5,527 | 185,459 | 501,942 | 924,916 |

At **one second, maximum depth 20**, the late position completes depth **3**
with the reference/ordering variants and depth **4** with both compiled
variants in every repetition. The specialised variant selects depth 4 while
depth 5 remains partial. Its median is 286,394 proof nodes / 9,659 main nodes /
808,837 work, versus reference 73,574 / 1,515 / 179,475. Units are identical;
the fixed-depth speedup, rather than this larger equal-time work count,
establishes the performance gain.

All variants complete depth four on the three cheap controls at equal time.
Selected depth is five at the opening and varies between four and five for the
middlegames; individual coverage and selection sources are in the JSON.
Independent tactical checks still reject Red 14 and Red 31 in every equal-time
run and accept Red 18 in every run. At fixed depth four those same outcomes hold
for all variants. Faster/deeper search **does not fix these defensive tactics**.
The original depth-three suite still has its five known failures, including
Red 18 and the last-defender cases in both colours.

The maximum observed end-to-end timeout overshoot across all 48 equal-time
runs is **3.64 ms**. Across 600 separately timed native calls at depths two and
eight, the largest observed cancellation interval is **0.286 ms**. These are
observations, not real-time scheduling guarantees. The static call cap remains
64 nodes / 129 work, including when callers request a large supported depth.

Separate setup: **5.091 s** for record/rules setup and **3.417 s** for the new
proof compilation in this process. Whole-process peak RSS is **309,051,392
bytes (294.7 MiB)**, including all variants and compilation. Per-search table
estimates are recorded above and in every JSON row; they do not include the
compiler/native allocator. Fresh processes pay the uncached proof setup again.

## Validation and adoption

The archived baseline suite has **177 tests: 172 pass, five known failures**
(the issue's older 175-test count predates #37). The first implementation run
has **184 tests: 179 pass, the same five failures**, plus the separately recorded
passing warm-signature test. No assertions were weakened or marked expected
failure. See [baseline log](evidence/baseline-tests.txt),
[focused validation](evidence/validation.txt) and
[warmed-call validation](evidence/warm-validation.txt).

New differential tests cover the 100 tactical fixtures, additional sparse
wins/losses/races/corner captures/stalemates, generated legal histories, both
colours, exact second/third repetitions, the capture-clock boundary and
win-over-draw precedence. They compare specialised and unspecialised native
proofs with the retained reference (including PV/work equality), independently
compare sufficient-budget scores to exhaustive oracle minimax, check all proof
depths 1–8, exercise each work cap 0–131 on representative trees, verify larger
budget fallback and deterministic deadline rejection, and compare completed
root selection under work interruption. A warm production call is forbidden
from compiling another Numba signature.

Adopt the specialised native path for the common bounded proof; retain the
reference for diagnosis, larger budgets and cold/nonstandard inputs. Keep the
small opening overhead, process setup cost and unchanged defensive failures
visible. This change neither depends on nor modifies the future compact-board
transition work (#40), material cache (#39), or search-window experiments (#41).

Final broader validation: **311 Intransitive tests, 306 pass and the same five
known defensive failures; all 7 repository tests pass**. The full run includes
all eight new compiled-proof tests, training/teacher integration and rule-engine
coverage. See [full validation log](evidence/full-validation.txt) and
[repository validation log](evidence/repository-validation.txt). Commands:
`uv run --locked python -m unittest discover -s intransitive/tests -v` and
`uv run --locked python -m unittest discover -s tests -v`.
