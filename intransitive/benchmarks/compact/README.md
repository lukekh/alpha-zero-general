# Compact reversible search — issue #40

Alpha-beta now validates and imports public storage once, then traverses a
search-owned 81-byte piece plane using reversible moves. Bounded native proofs
also use reversible piece updates and compact history scratch space. The public
`Board`/`Game` methods, serialized states, records, training observations and
heuristic weights are unchanged. `AlphaBetaPlayer(use_compact=False)` retains
the public-copy implementation for diagnosis and comparisons.

## Baseline and reproduction

The exact clean baseline is `516b150` (full revision in the manifest).
[evidence/baseline.tar.gz](evidence/baseline.tar.gz) contains tracked Python
sources, configurations, lockfile and replay inputs captured before editing;
[evidence/baseline-manifest.json](evidence/baseline-manifest.json) records the
archive and individual SHA-256 hashes. There were no uncommitted baseline
fixtures. The benchmark verifies every archived file and the unchanged shared
rules/configuration/replay dependencies before loading the archived evaluator
and search. The loader also isolates the archived terminal helper and lazily imported
proof helper. It checks that all pre-existing native proof functions are
unchanged. The existing native proof implementation remains the reference;
the compact native entry point is separate.

The issue's original `(9,9,33)` / 30-turn description predates `8415b53`.
This work preserves the current version-2 `(9,9,84)` / **80-turn** contract,
including official-play history rollover. Tests explicitly check that 30 quiet
moves remain playable under the current rules.

From the repository root, using the pinned Python 3.11 environment:

```sh
python intransitive/benchmarks/compact/reproduce.py > results.json
NUMBA_NRT_STATS=1 python intransitive/benchmarks/compact/allocations.py > native-allocations.json
python -m unittest intransitive.tests.test_compact_search intransitive.tests.test_anytime_search intransitive.tests.test_compiled_proof intransitive.tests.test_material
python -m unittest discover -s intransitive/tests
python -m unittest discover -s tests
```

The full suite intentionally reports the five known defensive failures; it is
not an all-green command. No assertions were weakened, exemptions added or
weights changed. Two existing test probes now inspect compact exports / the
new count-initialization location, retaining their original assertions.

Measurements used Python 3.11.4 on macOS 15.6.1 arm64 and the repository's pinned
environment. The complete [results](evidence/results.json) include configuration,
source hashes, replay-state hashes and each repetition. Timing used three
repetitions with alternating variant order, four fixed-depth-four positions and
separate one-second searches. Each search starts with an empty table and warms
compilation before its move deadline. Profiling and native-allocation accounting
run separately from the timing samples. The two timings are not additive:
transition savings overlap proof-search savings.

## Ownership and correctness

`SearchPosition` owns its plane, six exact material counts, side, goal ownership,
move number, chronological immutable board-plus-turn history and occurrence
map. A quiet move appends one 82-byte history entry; a capture saves the old
history/map references and starts with one occurrence. Undo restores squares,
captured piece, material, turn, history/map, exact move number and cached key.
Official-mode rollover also saves and restores the dropped history entry.
Export reconstructs the full ordered packet and all zero padding, never lends
out the mutable search plane.

Transposition bytes exactly match the reference: board, turn, goals, capture
clock and sorted occurrence counts. A cached key is invalidated on make and
restored on unmake. History order and move number still do not obstruct reuse;
second and third occurrences remain distinct. Corner and stalemate wins are
checked before modelling draws.

Alpha-beta and Python proof recursion place every successful push under a
`finally`/pop. Ordering generators never hold a mutated position across a yield.
The native proof entry point copies only the piece plane and live history into
owned scratch storage; recursion appends history rows and advances its active
start after captures. No native return, budget stop or exception can modify the
main search position. Its production batch remains at most 64 nodes / 129 work,
with deadline checks before and after. Larger configured proofs use the
cancellable Python make/unmake path. Detailed/optional positional evaluation
exports a snapshot to the existing evaluator; core material evaluation consumes
the compact plane and incremental counts directly.

Coverage includes the complete issue-37 PGN, generated legal games against both
the independent rules and public engine, random nested undo, all capture types,
all 12 symmetries and colour relabelling, exact rewards/reasons/exports, initial
occurrence retention, capture resets, win-before-draw precedence, clock
boundaries, history rollover and total-ply overflow. Native proof certificates,
logical work, sufficient-budget oracle values and draw histories are checked
independently. Fixed-depth cached/uncached/copy/exhaustive values agree.
Deterministic time/work exceptions at depths 1–3, ordinary cutoffs and unexpected
errors restore the position and retain completed table/root results; a failed
analysis also retains its previous published result and can resume.

## Fixed-depth results

Median elapsed seconds, identical values/actions/PVs/nodes/work/TT hits across
all three repetitions and both variants:

| Position | Copy baseline | Compact | Reduction | Total nodes/s before → after |
| --- | ---: | ---: | ---: | ---: |
| Initial | 0.286 | 0.227 | 20.7% | 36,516 → 46,027 |
| Before Red 14 | 0.392 | 0.317 | 19.2% | 32,734 → 40,522 |
| Before Red 18 | 0.424 | 0.328 | 22.6% | 42,924 → 55,479 |
| Before Red 31 | 0.750 | 0.323 | 56.9% | 254,576 → 590,593 |

Total nodes include proof nodes; the JSON also publishes main nodes/s. Red 31
visits the same 5,527 main nodes and 185,459 proof nodes in both variants.

At equal one second, all variants complete depth four. For initial, Red 14 and
Red 18, all runs select a completed branch from partial depth five. For Red 31,
the baseline selects depth four in 3/3 runs and compact selects partial-depth-five
branches in 3/3. Tactical objective successes (baseline → compact): Red 14
**0/3 → 1/3**, Red 18 **3/3 → 3/3**, Red 31 **0/3 → 3/3**. These are repeated
fixture-specific improvements under this budget, not an estimate of general
playing strength. The five existing depth-three defensive regression failures
are still present. Maximum observed one-second overshoot is **4.43 ms** for
the baseline and **38.86 ms** for compact (one Red-18 outlier; the other 11
compact runs are within 4.9 ms). These are observed process timings, not a hard
real-time guarantee; the outlier is retained in the evidence.

## Transition, conversion and memory costs

Median microseconds per operation over three runs of 2,000 operations. The
public column makes a child snapshot; the compact column includes both making
and undoing the same generated legal move. Boundary costs are reported separately
and are included once in total search timing where used.

| Position | Public make | Compact make + unmake | Import | Export | Import + export |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial | 4.81 | 1.85 | 6.07 | 3.38 | 9.83 |
| Before Red 14 | 5.19 | 1.85 | 11.83 | 10.78 | 23.60 |
| Before Red 18 | 5.04 | 1.78 | 8.36 | 6.68 | 15.95 |
| Before Red 31 | 4.80 | 1.77 | 7.87 | 5.77 | 14.47 |

Per 1,000 transitions, Numba runtime allocation calls fall **7,000 → 0**.
Compact transitions still allocate Python history bytes/undo records; zero here
refers only to native allocations. [Separate native counts](evidence/native-allocations.json)
include matched frees and show depth-four allocation reductions:

| Position | Native allocations before → after | Traced search peak bytes before → after |
| --- | ---: | ---: |
| Initial | 200,021 → 184,381 | 1,641,478 → 1,544,249 |
| Before Red 14 | 247,693 → 236,824 | 2,229,768 → 2,138,939 |
| Before Red 18 | 288,738 → 272,807 | 2,551,855 → 2,468,346 |
| Before Red 31 | 1,564,990 → 1,125,753 | 1,348,510 → 1,247,202 |

Traced peaks/retained bytes and net retained block counts are reported per
variant; net retained blocks are not cumulative allocation counts. The trace
also confirms 5,523–10,019 public main-search transitions become the same number
of pushes/pops with one compact import and no core-search exports. TT memory is
identical. Whole-process peak RSS was **285.0 MiB**, including both variants,
compilation and profiling; it is not a per-variant RSS reduction. Python history
container overhead means a full 81-entry boundary object is not necessarily
smaller than one public packet; recursive search avoids copying that packet for
each child.

Warm setup was **10.791 s** for shared baseline/import work, **0.007 s** for
compact position signatures loaded from disk caches, and **1.925 s** for new
compact native proof compilation. These are separate from steady-state search
and are not claims about a fresh-machine cold install. Recursive proof disk
caching remains disabled, as in the previous proof optimization.

## Validation evidence

Baseline: **327/332 pass**. Final: **342/347 pass**, with exactly the same five
known defensive failures (original Red 14/18/31 and both last-defender colours).
All **42 focused tests** and **7 repository tests** pass. The final focused run
also includes the completed-PGN make/unmake replay added during review.
[Machine-readable validation](evidence/validation.json) checks the failure names
against the baseline.

- [Baseline full suite](evidence/baseline-tests.txt).
- [Final full suite](evidence/full-validation.txt).
- [Focused compact/material/proof/anytime suite](evidence/focused-validation.txt).
- [Repository suite](evidence/repository-validation.txt).
