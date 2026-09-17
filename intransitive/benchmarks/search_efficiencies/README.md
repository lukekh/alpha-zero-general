# Search hot-path follow-ups

These five changes build on the [incremental bitboard implementation](../legal_moves/README.md).
The comparison baseline is the working branch **after bitboards**, not the old
board-scanning engine. Measurements therefore show additional gains.

1. **Rust threat detection:** intersect a precomputed 3×3 neighbourhood with the
   enemy predator mask, excluding the temporarily removed square. This replaces
   two full-board `exposed` scans per candidate during move ordering. The centre
   square is included, matching the original distance test exactly.
2. **Compiled Python ordering by default:** `compiled_ordering_enabled=True`
   compiles the existing ranking and sorting. Enhanced ordering, killers/history,
   PVS, aspiration and selective pruning retain their existing defaults. Explicit
   `compiled_ordering_enabled=False` remains supported. The ordering differential
   test and historical compilation experiment now explicitly select that control.
3. **Rust history-key reuse:** construct the complete draw-safe key once per
   internal search node, change only its depth byte during hint lookups, then
   restore that byte before storing the current-depth result. Full history
   equality, lookup order and table replacement remain unchanged.
4. **Reusable Python proof storage:** each search position lazily owns private
   board, mask, history, counter and principal-variation buffers. The live board
   never aliases traversal scratch. Every readable buffer is initialized before
   reuse, including after an exception or interrupted proof. Direct callers get
   owned results; search consumes borrowed results immediately. History packing
   and copying inputs into scratch are still required.
5. **Skip the duplicate Python terminal query:** the recursive search passes an
   explicit `terminal_checked` flag into ordering after its existing terminal
   check. That path uses board-only legal generation. Public `legal()` and other
   ordering callers continue to enforce corner, stalemate and draw rules.

The usual depth-2 proof scratch payload is 7,727 bytes (83 history rows plus
board, masks, counters and PVs), allocated once per search position that needs a
proof, plus NumPy/Python wrapper overhead. Capacity grows if a later call needs
more history rows. Rust adds a shared 1,296-byte neighbour table, with no added
per-position fields. Transposition keys remain the same size; repeated temporary
key allocations are removed.

## Measurements

Across 63 Python pairs and 45 Rust pairs, completed moves, scores, principal
variations, depths and node/proof counts matched exactly. Python's logical work
counts also matched. These improvements come from less work per visited node.

| Backend | Median paired CPU throughput | Median paired wall throughput | CPU range across trials |
| --- | ---: | ---: | ---: |
| Python/Numba | **1.45×** (+45%) | **1.46×** | 1.41–1.48× |
| Rust release | **2.54×** (+154%) | **2.56×** | 2.07–2.59× |

Every trial improved. Median whole-corpus CPU times were 1.403 → 0.979 seconds
for Python (272,070 nodes) and 6.093 → 2.451 seconds for Rust (6,523,345 nodes).
Medians of separate totals need not give exactly the median paired ratio above.
Rust wall-time speedups ranged from 2.00× to 2.64× under background load.

At 250 ms, Rust completed depth 6 rather than depth 5 on `position-4` in all
five trials. It reached depth 7 on the last-defender fixture in 4/5 optimized
runs versus 1/5 baseline runs. The three certified tactical choices stayed
unchanged. This demonstrates extra completed depth on this corpus, not a
claim of universally stronger play.

A separate three-trial comparison on the final implementation changed only
`compiled_ordering_enabled`, holding scratch reuse and the terminal shortcut
constant. Compilation delivered **1.52× median CPU throughput** (range
1.32–1.52×), with all 27 result/work comparisons identical. This supports the
new default. Gains from separate experiments are not additive; scheduling and
interactions differ. See [compilation-only results](evidence/compiled-ordering.json).

Raw outputs: [Python paired results](evidence/python-paired.json),
[Rust fixed-depth and fixed-time results](evidence/rust-search.json).

The same nine saved opening, middlegame, endgame and tactical states are used as
in the bitboard report. Python measures depth 4, Rust depth 6, with proof depth 2,
64 proof nodes, 50,000 table entries, material evaluation, generous fixed-depth
limits and fresh tables. Python's compilation flag is the intentional default
configuration difference; scores, action order and logical work must still match.

Python alternates the two implementations per position within one interpreter,
using a cache-free copy of the baseline under a distinct package name. Seven
trials measure process CPU and wall time, excluding warmup. Rust alternates
engine order across five trials; each request uses a fresh release process.
Rust CPU time measures reaped-child user+system time, and wall time includes
startup and IPC. The Rust benchmark also records 250 ms searches targeting
depth 20. Background training jobs were left running.

Hardware/software are unchanged from the bitboard report: Apple M1, 8 GiB RAM,
macOS 15.6.1 arm64, Python 3.11.4, NumPy 2.4.6, Numba 0.67.0 and Rust 1.74.1.
Rust release mode uses thin LTO and one codegen unit. Source/binary hashes,
configurations, exact outputs and individual timings accompany the raw results.
These are workload-specific measurements, not a claim about every evaluator.

## Validation

- 12 native tests pass; one manual microbenchmark is ignored in ordinary runs.
  Clippy passes with warnings denied, and formatting checks pass.
- The broad Python run executes 295 tests: **289 pass**, with exactly the same
  five known depth-3 blunder failures and existing material configuration-test
  `TypeError` documented in the bitboard report.
- Exhaustive native exposure tests cover all squares, attacker/victim colours
  and types, edges and removed-square cases against the original scanner.
- New Python tests cover defaults, the internal-only terminal shortcut, scratch
  identity/reuse, independent search positions, retained result ownership,
  work-budget exits, overflow exceptions and deliberately poisoned scratch.
- Existing tests exercise exact legacy/compiled ranking, all search flag
  combinations, transpositions, repeated/capture histories, cancellation and
  Python/Rust parity. No test failure was suppressed or weakened.

See [test logs and benchmark evidence](evidence/).

## Reproduce

The baseline included uncommitted bitboard changes. Recreate it by applying the
saved [baseline overlay](evidence/baseline-overlay.tar.gz) over the base commit:

```sh
mkdir -p /tmp/issue62-followup-before /tmp/issue62-corpus
git archive 032c6bb251f9f9d9adc8ac5bb3658dabc1492072 | tar -x -C /tmp/issue62-followup-before
tar -xzf intransitive/benchmarks/search_efficiencies/evidence/baseline-overlay.tar.gz \
  -C /tmp/issue62-followup-before
tar -xzf intransitive/benchmarks/legal_moves/evidence/positions.tar.gz -C /tmp/issue62-corpus
cargo build --release --offline --manifest-path /tmp/issue62-followup-before/intransitive/rust_teacher/Cargo.toml
cargo build --release --offline --manifest-path intransitive/rust_teacher/Cargo.toml
python intransitive/benchmarks/legal_moves/paired_python.py \
  --before /tmp/issue62-followup-before --positions /tmp/issue62-corpus/positions \
  --repeats 7 --output /tmp/issue62-followup-python.json
python intransitive/benchmarks/legal_moves/benchmark.py --backend rust \
  --before /tmp/issue62-followup-before --positions /tmp/issue62-corpus/positions \
  --repeats 5 --output /tmp/issue62-followup-rust.json
# Isolate compiled ordering on the final implementation:
python intransitive/benchmarks/legal_moves/paired_python.py \
  --before "$PWD" --before-python-ordering --positions /tmp/issue62-corpus/positions \
  --repeats 3 --output /tmp/issue62-compiled-ordering.json
python -m unittest intransitive.tests.test_search_efficiencies \
  intransitive.tests.test_compact_search intransitive.tests.test_search_optimizations \
  intransitive.tests.test_anytime_search intransitive.tests.test_search_windows
cargo test --release --offline --manifest-path intransitive/rust_teacher/Cargo.toml
cargo clippy --offline --manifest-path intransitive/rust_teacher/Cargo.toml --all-targets -- -D warnings
```
