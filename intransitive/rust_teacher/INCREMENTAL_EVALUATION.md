# Incremental native evaluation

Quiet search moves now retain the six side/type counts and their material
contributions. Moving a piece updates its type's 81-square mask. Captures update
the captured count and refresh the six contributions; undo restores the same
state. Evaluation selects the original first-occurrence type order from the
masks, replacing the two board scans and repeated divisions at ordinary leaves.
It preserves the original floating-point operations and final pressure/clamping
operations. Material totals are combined from cached per-type contributions
at each evaluation.

The accumulator belongs to `Position`. Rust callers read the board through
`Position::board()` and mutate it through `apply`, keeping derived state coherent.
Board/history-only tactical proof moves leave the caller's material untouched
and undo all their temporary changes before returning. Proofs never query that
accumulator. This avoids maintaining an unused heuristic through millions of
proof nodes in some endgames.

Pressure-enabled search also maintains attacker and defender counts by square
and ring. Removing or adding a piece changes its own contribution and the
contributions of victims for which it acts as an attacker or defender. An
occupied-square mask limits updates to pieces on the board. Changing defensive
coverage for a victim with no attackers needs only a count update. Captures and
undo use the same remove/add operations. Proof-only moves skip this accumulator
as well.

For boards with at most 32 pieces, the pressure weights and defensive discounts
are binary fractions whose complete intermediate sums fit exactly in binary64
(denominator at most 2^33 and magnitude below 2^10). This permits exact delta
updates without drift or changes to pair-summation results. Larger edited
boards use the original pressure kernel. Disabled pressure allocates no
pressure accumulator. The root initializes the applicable cache from its actual
board and configuration before search.

These changes cover the native teacher. The Python implementation already has
its own incremental material path. Existing running dataset jobs retain their
pinned binary until explicitly migrated.

## Results — 16 September 2026

All **108 before/after pairs** matched exactly: scores, moves, principal variations,
depths, stop reasons, node/proof counts and table statistics. Ninety pairs completed
depth six; eighteen were the same endgame proved at depth one.

Times below are median totals for one six-position batch across six repetitions.
Paired speedups are the median of each trial’s before/after ratio; they can differ
from the ratio of the separately reported median times.

| Profile | CPU before → after | Paired CPU speedup | Wall before → after | Paired wall speedup |
| --- | ---: | ---: | ---: | ---: |
| Material only | 7.602 → 7.427 s | 1.031× | 11.194 → 11.069 s | 1.022× |
| Material + 7×7 pressure | 8.541 → 7.559 s | 1.122× | 12.365 → 12.058 s | 1.111× |
| Material + 9×9 pressure | 8.395 → 7.527 s | 1.134× | 11.612 → 10.306 s | 1.127× |

Material-only CPU savings are small (about 3% by paired ratios), with one of six
trials slower than baseline. Pressure-enabled CPU savings are about 11–12%; all
six CPU trials improved for each pressure radius. Whole-batch CPU speedup ranges
were 0.973–1.069× (material), 1.053–1.172× (7×7), and 1.098–1.157× (9×9).

Wall time was more variable under the concurrent training/generation workload:
paired ranges were 0.809–1.063×, 0.910–1.246× and 1.051–1.171× respectively.
These measurements support a useful pressure-search improvement and a modest
material-only improvement, rather than a large general minimax speedup. They do
not measure end-to-end dataset throughput or playing strength.

Per-position median paired CPU speedups show where the benefit occurs:

| Position | Completion | Material | 7×7 pressure | 9×9 pressure |
| --- | --- | ---: | ---: | ---: |
| 0: Opening | Depth 6 | 1.003× | 1.146× | 1.147× |
| 1: Midgame | Depth 6 | 1.040× | 1.136× | 1.150× |
| 2: Endgame | Proof at depth 1 | 1.066× | 1.063× | 0.976× |
| 3: Opening | Depth 6 | 1.018× | 1.131× | 1.148× |
| 4: Midgame | Depth 6 | 1.051× | 1.077× | 1.051× |
| 5: Endgame | Depth 6 | 1.003× | 0.962× | 1.006× |

Validation passed: nine native tests in both debug and release builds, eleven
Python/native integration tests, and Clippy with warnings denied.

[Summary JSON](../benchmarks/incremental_native/evidence/summary.json) ·
[Raw paired results](../benchmarks/incremental_native/evidence/results.json.gz) ·
[Validation and source evidence](../benchmarks/incremental_native/evidence/README.md)

## Measurement method

The benchmark compares separate release binaries built from the saved original
source and the incremental implementation with the same compiler and release
settings. Six saved positions span two openings, two midgames and two endgames.
Each of three profiles runs six times at target depth six: material only, 7x7
pressure at weight 10, and 9x9 pressure at weight 10. Every request uses two-ply,
64-node proofs and a fresh 50,000-entry search table. One endgame is proved early;
the other five positions must finish depth six.

Engine order alternates within position pairs, and profile order reverses on
alternate trials. A separate depth-three warmup precedes every pair. Both
engines must return exactly equal scores (including binary64 representation),
actions, principal variations, depths, stop reasons, search/proof node counts,
table hits and final table sizes. Incomplete or mismatched runs fail the
benchmark and retain the partial evidence.

Wall time includes process startup, request/response and process exit. Each
request runs in its own process so Python can measure its complete user+system
CPU time with `getrusage(RUSAGE_CHILDREN)` after reaping it. CPU time helps
separate computation from scheduling delays; wall time reports observed
latency. The benchmark uses one low-priority native search worker while the
existing training and generation jobs continue. Compilation, warmup, sampling,
ablations and corpus writes are excluded.

## Reproduction

The [evidence directory](../benchmarks/incremental_native/evidence/) contains
source archives for both measured versions, the six input positions, raw
results, compiler/source identities and validation logs. Extract and build the
measured sources separately, then run the same native harness against them:

```sh
mkdir -p checkpoints/native-delta-reproduction
tar -xzf intransitive/benchmarks/incremental_native/evidence/baseline.tar.gz -C checkpoints/native-delta-reproduction
tar -xzf intransitive/benchmarks/incremental_native/evidence/final-candidate.tar.gz -C checkpoints/native-delta-reproduction
tar -xzf intransitive/benchmarks/incremental_native/evidence/positions.tar.gz -C checkpoints/native-delta-reproduction
cargo build --release --offline --manifest-path checkpoints/native-delta-reproduction/baseline/Cargo.toml
cargo build --release --offline --manifest-path checkpoints/native-delta-reproduction/final-candidate/Cargo.toml
.venv/bin/python -m intransitive.rust_teacher.incremental_benchmark \
  --before checkpoints/native-delta-reproduction/baseline/target/release/intransitive-rust-teacher \
  --after checkpoints/native-delta-reproduction/final-candidate/target/release/intransitive-rust-teacher \
  --positions checkpoints/native-delta-reproduction/positions \
  --output checkpoints/native-delta-reproduction/results --repeats 6
```

Use a fresh output directory. The harness copies its exact input states and
records both binary hashes, configuration, execution order, raw timings and
complete search results.

## Correctness checks

The original evaluator remains a test-only reference. New native tests compare
scores bit for bit for both perspectives, both pressure radii, zero/default/large
pressure weights, random legal trajectories, sibling branches, captures, type
extinction and quiet moves that reorder piece types. They independently rebuild
the material state and fully unwind each trajectory. Additional tests exercise
search cutoffs, cancellation, resumed search and proof-budget exits.
Pressure checks rebuild all ring counts, test a defender moving away from a
stationary victim, and cover the 32/33-piece boundary and dense-board fallback.

The existing cross-language and backend suites cover 100 certified tactics,
shallow search parity, 160 trajectory transitions, official/modelling draw rules,
invalid states, label-family generation, configuration/cache isolation and
native-child cleanup.
