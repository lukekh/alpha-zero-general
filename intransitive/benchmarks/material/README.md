# Incremental material evaluation (#39)

Core scalar leaves now reuse material arithmetic from six immutable side/type
counts. Search counts the root once, shares the same tuple on quiet moves,
replaces exactly the captured entry on captures, and restores the parent tuple
in `finally` on returns, cutoffs, cancellation and unexpected exceptions. The
public board/history representation and transitions remain unchanged; #40 can
reuse `count_pieces` and `after_capture` with its later make/unmake board.

The cache belongs to an evaluator and its frozen `SearchConfig`. Replacing the
configuration replaces the cache before the next core score. Each cache has at
most 256 count entries, evicted by LRU; each entry contains 1,024 bytes of array
payload plus Python/NumPy bookkeeping. Neither positions nor completed scores
are cached by counts. Terminal, history draw and proof checks precede material;
optional positional modules and detailed explanations retain the reference
implementation and perform their own position-specific analysis.

## Exact arithmetic and bounded work

The original `Counter` implementation sums piece-type contributions in the
order each type first appears in board traversal. A quiet move can change that
order and its last floating-point bits. A fixed type order is therefore not a
numerically equivalent replacement. The cache stores all six possible sums
per side, using the exact original Python formula and summation operations.
A small Numba kernel reads each side's first-occurrence order and combines the
cached sums in the original arithmetic sequence, without fast-math. Weights,
clamping, perspective, signed zero, mate values, proof semantics, and tactical
tie ordering are preserved. The small board scan is intentional; quiet moves
reuse the arithmetic even when a different order variant must be selected.

Scalar core evaluation allocates no `Geometry`, `Piece`, `Counter`, feature
maps or explanations. Standalone `Evaluator.score` counts its input when no
search-owned counts are provided. Optional modules and public explanations
retain their reference path, including all disabled-module guards. Timing for
the fast path is reported as `material`; reference explanations still report
`piece_index`, `piece_count` and `piece_advantage`.

A completed core evaluation retains its previous logical allowance of one work
per piece plus two material-module work units, including cache hits. The fast
path reserves that allowance together, so an interrupted reservation can leave
unused work where the reference would charge a partial piece loop. It cannot
exceed the work cap or publish that unfinished leaf. Deadline checks surround
the bounded 81-square kernel. Existing completed-sibling/TT behavior is retained.
Compilation is warmed in search preparation before normal move timing.

## Reproduction

The clean baseline is `4a007f709ef3f8d81d2cc27d7a9c54efa2f69d2c`, after the merged
budget-policy and compiled-proof changes. [Source archive](evidence/baseline.tar.gz)
and [SHA-256 manifest](evidence/baseline-manifest.json) preserve its root and
Intransitive Python sources, configuration, tactical fixtures, dependency lock,
and exact [79-ply replay](../search_budget/game79.pgn). The original replay's
final-state hash is validated by the record loader. Historical model artifacts
are omitted from this archive and remain available at the baseline commit.

```sh
uv run --locked python intransitive/benchmarks/material/reproduce.py \
  --repeats 3 --seconds 1 --evaluations 2000 > results.json
uv run --locked python -m unittest \
  intransitive.tests.test_material intransitive.tests.test_compiled_proof \
  intransitive.tests.test_tactics intransitive.tests.test_game_blunders \
  intransitive.tests.test_anytime_search intransitive.tests.test_heuristics \
  intransitive.tests.test_search_performance intransitive.tests.test_play \
  intransitive.tests.test_record
uv run --locked python -m unittest discover -s intransitive/tests -v
uv run --locked python -m unittest discover -s tests -v
```

The benchmark loads the unmodified baseline evaluator and search directly from
the archive. All other rule/proof code is shared and unchanged. Variants use
fresh players/tables and alternate order across three repetitions. Both have
identical recorded configurations: depth four with ample limits, or maximum
depth 20 with one second and a billion-work safety cap. Compilation is warmed
before timing. The benchmark asserts identical fixed-depth scores, actions,
PVs, main/proof nodes, work and tactical objectives before publishing results.

[Raw results](evidence/results.json) include every repeated core/search time,
configuration, replay and implementation hash, nodes/second, module time,
completed/selected/partial depth, root coverage, timeout overshoot, independent
tactical outcome, table estimate, and material-cache payload/hit/miss counts.
Core timings include terminal checks with an already-unknown proof and warm
material entries. First-evaluation costs are recorded separately. Allocation
samples run separately under `cProfile` and `tracemalloc`: they count actual
`Geometry`, `Piece` and `Counter` constructor calls, traced peak/retained bytes
and net retained blocks. Constructor counts are not a count of all transient
allocations; net blocks do not count objects already freed. Traced samples are
excluded from throughput timings. Whole-process RSS includes compilation and
both variants, and is not attributed to one implementation.

## Validation

The archived [baseline focused run](evidence/baseline-tests.txt) has 177 tests,
172 passing with the same five documented defensive failures: original Red
14, 18 and 31, and the simplified last defender in both colours. These assertions
remain unchanged and are neither weakened nor marked expected failures.

The seven new material tests check independent recounts through legal generated
games, siblings and captures, all nine attacker/defender pairings in both
colours (including illegal pair rejection), type extinction, all 12 symmetries
and canonical forms. They compare float hex representations against detailed
recomputation with random weights and quiet type-order changes, check all eight
optional-module combinations, configuration replacement, clamping and bounded
cache eviction, and reject stale corner/draw/proof/positional values. Search
checks cover root initialization, warmed signatures, cutoffs, actual work caps,
injected time/work/errors at multiple depths, exact count restoration and
resumed-versus-fresh values/PVs.

[Full validation](evidence/full-validation.txt): 318 tests, 313 passing and the
same five known failures. [Repository validation](evidence/repository-validation.txt):
all seven tests pass. [Final focused validation](evidence/focused-validation.txt)
reruns the affected suites after the final cache-lifetime changes.
It reports 192 tests, 187 passing with the same five failures. The subsequent
[seven material checks](evidence/material-validation.txt) also pass, including
integer-valued configuration weights using the warmed float kernel signature.

## Measured results and adoption

Three paired repetitions on macOS 15.6.1 arm64 / Python 3.11.4 with the pinned
`uv.lock` environment. Medians are shown; the raw JSON retains every run.
Every fixed-depth score, move, full PV, main/proof-node count, logical work and
tactical objective matches the archived baseline exactly.

| Position | Core evaluation, baseline → incremental | Depth-four total, baseline → incremental | Total speedup | Main nodes/second, baseline → incremental |
| --- | ---: | ---: | ---: | ---: |
| Opening | 41.36 → 6.35 µs | 0.494 → 0.351 s | 1.41× | 12,359 → 17,381 |
| Before Red 14 (ply 27) | 43.54 → 8.61 µs | 0.627 → 0.454 s | 1.38× | 12,355 → 17,048 |
| Before Red 18 (ply 35) | 37.98 → 6.71 µs | 0.731 → 0.470 s | 1.55× | 13,714 → 21,303 |
| Before Red 31 (ply 61) | 33.84 → 6.27 µs | 0.799 → 0.671 s | 1.19× | 6,917 → 8,236 |

The core call improves **5.06–6.51×**; total depth-four elapsed improves
**16.0–35.6%**. The late position still spends most of its time in proof work.
These repeated single-machine timings demonstrate throughput, not playing
strength or a speed guarantee on other hardware.

| Position | Main / proof nodes | Work | Table bytes | Material entries / array bytes |
| --- | ---: | ---: | ---: | ---: |
| Opening | 6,107 / 4,325 | 265,937 | 1,196,340 | 4 / 4,096 |
| Before Red 14 | 7,742 / 5,096 | 309,264 | 1,721,469 | 12 / 12,288 |
| Before Red 18 | 10,023 / 8,170 | 319,654 | 2,027,503 | 17 / 17,408 |
| Before Red 31 | 5,527 / 185,459 | 501,942 | 924,916 | 15 / 15,360 |

For Red 18 the material cache has 8,152 hits and 17 misses. Its depth-four
constructor counts fall from **8,170 Geometry / 110,186 Piece / 32,680 Counter**
to **1 / 14 / 4**. The remaining constructors belong to the final detailed root
explanation. Across 100 warm core calls, all three constructor counts become
zero (baseline: 100 / 1,400 / 400). The other positions show the same removal
of per-leaf objects, with their exact counts in the JSON.

| Position | Traced depth-four peak bytes, baseline → incremental |
| --- | ---: |
| Opening | 1,679,090 → 1,600,236 |
| Before Red 14 | 2,203,333 → 2,197,310 |
| Before Red 18 | 2,566,300 → 2,522,225 |
| Before Red 31 | 1,378,010 → 1,314,035 |

Peak memory changes are modest because the retained search table dominates.
Material-cache array payload is 4–17 KiB on these trees, bounded at 256 KiB
plus container overhead in general. Memory rows include profiling/tracing
bookkeeping and are individual samples, not repeated memory medians. Peak RSS
for the entire benchmark process is **269,729,792 bytes (257.2 MiB)**, including
both variants and compilation.

At equal **one second**, all variants/repetitions complete depth **four**.
At the opening both select a completed root branch at partial depth five;
before Red 14 and Red 18, baseline selected depth is four and incremental
selected depth is five in all three repetitions. Before Red 31 both select
depth four. **Partial selected depth five is not a completed depth-five search.**
The independent Red 14 and Red 31 defensive objectives still fail, while Red
18 passes at depth four and at equal time in both variants. The original
five depth-three regression failures remain. No improvement in these tactical
objectives or overall playing strength is claimed.

Maximum observed timeout overshoot across the paired runs is **6.81 ms**,
including existing post-search bookkeeping. Common rules/record/proof setup
costs **13.588 s** in this process; loading already-compiled material kernels
costs **0.00493 s** separately. A [fresh temporary Numba-cache measurement](evidence/setup.json)
records **0.646 s** to compile the material kernels, excluding **0.244 s** of
imports. The warmed preparation call costs less than one microsecond. These
setup observations are separate from steady-state move timings.

Adopt the bounded core scalar path: it removes the measured allocations and
improves total time on all four controls with unchanged fixed-depth decisions.
Retain detailed/optional evaluation as the formula reference. Keep the existing
defensive failures visible and pursue #40/#41 independently; this change does
not alter their public-state, draw-identity or search-window contracts.
