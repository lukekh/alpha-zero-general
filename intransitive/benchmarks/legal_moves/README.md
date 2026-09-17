# Incremental legal moves — issue #62

Both Rust and the Python/Numba compact search now maintain six piece masks.
Rust uses six `u128` values; Numba uses two `uint64` lanes per mask (squares
0–63 and 64–80). Board arrays, state-v2 serialization, repetition history,
draw precedence, evaluation and action IDs are unchanged.

Each direction intersects the friendly pieces with inverse-shifted legal
destinations (empty squares or the enemy prey type), then applies its source
edge mask. This is equivalent to shifting pieces toward destinations and
retains source/direction identity. Enumeration stays in ascending action order;
multiple moves to the same destination remain distinct. Unused high bits never
become legal squares. `has_move` exits at the first nonempty direction.

Rust legality masks are separate from material's piece-location masks because
proof moves intentionally leave the material accumulator unchanged. Legality
masks update in `push_board`/`pop_board` and in history-free move-ordering probes.
Python maintains masks in `SearchPosition`, copies them once into a proof's
scratch storage, and updates them throughout that traversal. Move-ordering
probes also maintain scratch masks. Python returns a directly allocated legal
action array instead of allocating and then scanning a 648-entry boolean mask.
The public rules generator remains an independent reference.

The [subsequent search efficiencies](../search_efficiencies/README.md) are measured
separately against this bitboard implementation; that report archives its exact
baseline source overlay.

## Measurement

Measurements were made on an Apple M1, 8 GiB RAM, macOS 15.6.1 arm64, Python 3.11.4,
NumPy 2.4.6, Numba 0.67.0 and Rust 1.74.1. Rust uses the existing release profile
(thin LTO, one codegen unit); JIT/build warmup is excluded. Exact installed
versions are in [requirements.txt](evidence/requirements.txt).

The baseline is commit `032c6bb251f9f9d9adc8ac5bb3658dabc1492072`.
The corpus contains two opening, two middlegame and two endgame positions from
the existing incremental-native benchmark, plus the three simplified fixtures
from `heuristics/game_blunders.json`. Exact state bytes are archived in
[positions.tar.gz](evidence/positions.tar.gz); search results record their hashes,
source hashes and binary hashes.

Five trials alternate before/after engine order. Each worker warms its kernels;
each position also has a depth-2 search warmup. Every measured search starts
with a fresh table. Python uses depth 4; Rust uses depth 6. Both use material
evaluation, proof depth 2, proof allowance 64, 50,000 table entries, a one-billion
node/work ceiling and a 120-second fixed-depth deadline. Selective search and
optional pressure evaluation are disabled. Separate 250 ms searches target
depth 20. These measurements do not establish performance for every evaluator
or board density.

Existing background training jobs were left running. Both observed wall time
and process CPU time are reported: Python uses `process_time`; Rust uses reaped
child user+system CPU time with a fresh process per request. Rust wall times
include startup and IPC. Node counts include proof visits and must be compared
within a backend, since the two search implementations do different work.

All 90 paired fixed-depth searches matched actions, scores, principal
variations, completed depths and search/proof node counts exactly. There is no
changed tie-breaking in completed searches. Both implementations are retained.

| Backend | Nodes per corpus pass | Before CPU seconds | After CPU seconds | Before nodes/CPU-second | After nodes/CPU-second |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python/Numba, depth 4 | 272,070 | 2.155 | 1.706 | 126,263 | 159,515 |
| Rust release, depth 6 | 6,523,345 | 7.201 | 6.346 | 905,895 | 1,027,956 |

The table uses medians of whole-corpus totals. Median *paired* speedups were
**1.28× Python CPU / 1.28× wall** and **1.13× Rust CPU / 1.19× wall**.
Wall totals were 2.322 → 1.849 seconds for Python and 8.189 → 6.868 seconds for
Rust. These two aggregation methods need not produce precisely equal ratios.

Python's per-trial CPU speedups ranged from 0.85× to 1.86× (wall 0.72–1.97×);
Rust ranged from 1.06× to 1.32× CPU (wall 1.08–1.41×). Background load and M1
performance/efficiency-core placement make individual passes noisy; CPU time
removes descheduling time but cannot remove different core speeds. Rust improved
in every whole-corpus trial. Individual small workloads can still regress:
the Rust tactical subset's median paired CPU ratio was 0.91×. Aggregate gains
are not a claim of uniform acceleration.

To reduce that variability, a second Python experiment loaded the unchanged
baseline under a separate package name and alternated before/after searches
for every position in one interpreter/thread. It used a fresh cache-free copy
of the baseline to prevent Numba cache/module-name collisions, warmed both
implementations, and ran seven trials at the same depth/configuration. All
63 additional comparisons matched exactly.

This closer pairing measured **1.34× median CPU throughput** (range
1.23–1.40×) and **1.35× wall throughput** (range 1.18–1.41×). Every trial improved.
This is the more reliable Python estimate on this loaded machine. The raw
[paired results](evidence/python-paired.json) retain both timings and exact
search outputs; the independent-process results above are also retained.

### Queries and maintenance

Each Python query is called from Python 10,000 times per state per trial. Rust
uses an ignored release-mode test with 100,000 calls, `black_box` barriers, the
original board scanner as reference and alternating measurement order. Values
below are medians of per-trial, equal-weight corpus means. They include Python
call/loop overhead and exclude mask construction.

| Operation | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Python legal actions, CPU µs/call | 2.073 | 0.901 | 2.30× |
| Python has-move, CPU µs/call | 1.642 | 0.238 | 6.89× |
| Rust legal actions, wall ns/call | 193.2 | 103.3 | 1.87× |
| Rust has-move, wall ns/call | 15.56 | 4.07 | 3.82× |

Python's complete push/pop pair, including history maintenance, measured
2.660 → 2.649 CPU µs (wall 2.867 → 2.918 µs): mask maintenance has no clear net
cost at this resolution, because the update kernel also replaces Python-level
board writes. Rust's isolated capture-mask update+undo pair measured 5.81 ns;
this is included in full-search timings, not subtracted from them. Its quiet
move updates fewer bits. The `has_move` speedup varies by layout: an existing
scanner that immediately finds a movable piece can beat the new mask query.

Each Rust position adds 96 bytes of legality masks; the measured position size
is 432 bytes. Python adds a 96-byte array payload plus its NumPy wrapper, and
copies that payload once per compact proof. Neither undo records nor table keys
grow. Python legal arrays allocate exactly eight bytes per returned action;
a temporary 128-byte direction-mask array replaces the old 648-byte boolean
mask and its `flatnonzero` scan. Board/history storage is retained.

### Fixed-time search and move quality

At 250 ms, Python completed depth 4 on `position-5` in 5/5 optimized runs versus
2/5 baseline runs; Rust completed depth 5 there in 3/5 versus 1/5. Python
completed depth 4 on the last-defender fixture in 5/5 versus 4/5. Most other
completed-depth medians were unchanged, and there were isolated lower-depth
runs under load. Proof-complete roots can finish below the target depth.

All three certified tactical moves were selected in all five time-limited
trials by both versions of both backends: `G5-H6`, `F8-G8` and `F6-G7`.
Thus this corpus shows occasional extra completed depth, but no demonstrated
improvement in tactical pass rate. Faster generation does not fix evaluation
or remove horizon effects. Raw fixed-time results, including moves, depths,
node counts and scores, are in [search.json](evidence/search.json).

## Correctness and known failures

- All 11 native tests pass (the twelfth test is the opt-in microbenchmark).
  Clippy passes with warnings denied.
- The broad Python run executes 279 tests: 278 pass and one pre-existing test
  errors. An additional bitboard run passes all three tests, including the
  subsequently added empty/fully-blocked-board test.
- Differential tests cover every source, direction and piece matchup, both
  colours, edge wrapping, the 64-bit lane boundary, dense random boards,
  reachable games, captures and nested undo. Existing proof/search interruption
  tests now include masks in restoration snapshots. Native trajectory tests
  compare masks and legal actions with a fresh board scan.
- `test_game_blunders` has exactly the same five failures before and after:
  `red_14_preserve_material`, `red_18_prevent_paper_fork`,
  `red_31_keep_goal_defender`, and `unique_last_defender` in both colour variants.
  The other seven cases pass. This optimization does not change depth-3 choices.
- The unrelated existing error is
  `MaterialTests.test_configuration_identity_clamping_and_bounded_cache`:
  the test attempts to add `1` to a string configuration value. It reproduces
  unchanged on the baseline. No unrelated test expectations were weakened.

Logs are retained in [evidence](evidence/). The broad run includes rules, draws,
state handling, tactics, compact and compiled proofs, native parity, anytime
search, ordering optimizations, search windows, selective search, material,
pressure and heuristic evaluation.

## Reproduce

From the repository root, with the project's Python dependencies installed:

```sh
mkdir -p /tmp/issue62-before /tmp/issue62-corpus
git archive 032c6bb251f9f9d9adc8ac5bb3658dabc1492072 | tar -x -C /tmp/issue62-before
tar -xzf intransitive/benchmarks/legal_moves/evidence/positions.tar.gz -C /tmp/issue62-corpus
cargo build --release --offline --manifest-path /tmp/issue62-before/intransitive/rust_teacher/Cargo.toml
cargo build --release --offline --manifest-path intransitive/rust_teacher/Cargo.toml
python intransitive/benchmarks/legal_moves/benchmark.py \
  --before /tmp/issue62-before --after "$PWD" \
  --positions /tmp/issue62-corpus/positions --repeats 5 \
  --output /tmp/issue62-search.json
python intransitive/benchmarks/legal_moves/paired_python.py \
  --before /tmp/issue62-before --positions /tmp/issue62-corpus/positions \
  --repeats 7 --output /tmp/issue62-python-paired.json
ISSUE62_POSITIONS=/tmp/issue62-corpus/positions \
  cargo test --release --offline --manifest-path intransitive/rust_teacher/Cargo.toml \
  benchmark_legal_moves -- --ignored --nocapture
cargo test --release --offline --manifest-path intransitive/rust_teacher/Cargo.toml
cargo clippy --offline --manifest-path intransitive/rust_teacher/Cargo.toml --all-targets -- -D warnings
python -m unittest intransitive.tests.test_bitboard_moves \
  intransitive.tests.test_compact_search intransitive.tests.test_rust_teacher
```

The search benchmark aborts if a fixed-depth action, score, completed depth,
node count, proof count, principal variation or completion flag differs.
