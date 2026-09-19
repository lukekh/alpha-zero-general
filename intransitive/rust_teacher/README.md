# Experimental Rust Minimax teacher

A dependency-free Rust library and persistent subprocess for complete-depth
policy labels. The standalone commands below are isolated experiments. The
million-position generator now opts into the native adapter with its unchanged
depth-6 material scoring; the fly-brain reinforcement trainer is independent.

Native code owns move generation, make/unmake, corner/stalemate wins, full
repetition history, modelling-only draw cutoffs, material/type-advantage scoring,
7x7/9x9 ring pressure, bounded terminal proofs, iterative alpha-beta/PVS,
move ordering and an exact-key transposition table. Python retains the existing
reachable-position sampler, symmetry transforms, split assignment and shard I/O.
There are no new Python packages or Cargo dependencies.

Evaluation now maintains material counts, piece-type location masks and local
pressure during search. See [incremental evaluation](INCREMENTAL_EVALUATION.md)
for the exactness checks and paired before/after benchmark.

## Build, verify, benchmark

Run from the repository root (Rust 1.74+):

```sh
cargo test --manifest-path intransitive/rust_teacher/Cargo.toml --offline
cargo build --release --manifest-path intransitive/rust_teacher/Cargo.toml --offline
.venv/bin/python -m unittest intransitive.tests.test_rust_teacher -v
.venv/bin/python -m intransitive.rust_teacher.benchmark \
  --output checkpoints/rust-comparison-new --positions 6 --depth 6 --repeats 2
```

The benchmark uses identical sampled positions for both engines and checks
completed-depth scores. Different tied choices are re-searched in Python to
check that the native action is equally good. It preserves input states, source
and binary hashes, per-search wall time, completion status and all results.
Compilation/warmup is excluded; Rust request/response overhead is included in
the reported wall-time ratio. Both programs run with low priority alongside
the live jobs. This is single-worker label-search throughput, not an estimate
of the existing multiworker pipeline with ablations and promoted roots.

## Create an experimental dataset

```sh
.venv/bin/python -m intransitive.rust_teacher.generate \
  --output checkpoints/rust-labels-new --positions 100 --depth 6 --seconds 60
```

Defaults: 7x7 pressure, weight 10, two-ply/64-node terminal proof, 50,000 table
entries. `--weight 0` disables pressure. Sampling balances openings, midgames
and endgames. Only full requested-depth or proven-result labels are saved;
timeouts are rejected. Output is gzip/pickle shards plus a manifest, with full
teacher-profile provenance. The existing `SymmetryExamples` class exposes 12
lazy variants per base position. Orbit deduplication prevents symmetries from
crossing train/validation/test splits within this isolated dataset.

This command refuses an existing output directory. It checkpoints completed
records in batches of 16 and on ordinary interruption; it is not a resumable
million-position service. Do not merge it into a live corpus without explicit
teacher-version/split migration. The experimental weight is not strength-tuned.

## Intentional limits and differences

- Material weights are fixed at the current defaults: count 100, advantage 25,
  predator-zero 1, scarcity 0.5 and prey 0.25. Legacy route attack/defence/overload
  modules are not implemented. Pressure radius/weight and depth are configurable.
- Terminal proofs support depths 0..2 and up to 64 nodes, matching current
  production defaults. They preserve goal-first ordering and local exhaustion
  semantics. No null-move pruning, late-move reductions or heuristic draws.
- Native search has PVS but no aspiration windows. Its bounded FIFO table
  uses the exact ordered active history, more conservative than Python's
  occurrence-count equivalence/depth-preferred eviction. These can change
  node counts and tied choices; completed score/action parity is tested.
- `node_limit` counts actual search/proof visits, **not** Python's charged-work
  units. A node-limited run is not a like-for-like benchmark. Compare completed
  fixed depths under generous limits. Wall time is separately bounded.
- Official and modelling termination are separate. Labels always use modelling
  rules; `apply(..., modelling=False)` continues actual games beyond draws.
- The standalone command has no ablations, tree-root promotion or parallel
  workers. The production adapter uses configurable Python sampling
  workers, verified ablations and promoted roots, with all Minimax searches in
  Rust. `search_reuse` keeps the bounded history-safe native table between related
  positions. The hot search loop is native; the whole data pipeline is not.

## Module interface

```python
from intransitive.rust_teacher import RustTeacher

with RustTeacher() as teacher:
    label = teacher.analyze(state, depth=6, radius=3, weight=10., seconds=60.)
    if label['complete']:
        action = label['action']
```

`state` is the existing validated `(9, 9, 84)` int8 state, including history.
The binary accepts one whitespace-delimited request per line and returns JSON.
The adapter provides `analyze`, `inspect` and `apply`; EOF cleanly shuts down
the persistent process. Native errors, invalid inputs and transport timeouts
are propagated rather than silently creating fallback labels.

## Durable generator integration

`adapter.RustAlphaBetaPlayer` validates supported scoring and a pinned binary
checksum, then maps complete native results to the label pipeline. It never
falls back to Python search. Root/descendant/ablation records carry backend,
binary hash, teacher config and explicit node-visit work units. Worker SIGTERM
and pool cleanup reap native children; a registry also covers a killed Python
worker. Backend-specific outbox namespaces prevent accidental cache mixing.

`expand_minimax_dataset.run(..., teacher_backend=..., allow_backend_change=True)`
permits an explicit backend-only migration. Existing corpus identity, scoring,
depth, time limit and numeric node ceiling must remain equal; the budget-unit
difference is recorded in migration history. Partial labels remain rejected.
The 16 September cutover preserved 45,500 committed positions and pending seeds;
the previous database/source pointer is recoverable from
`checkpoints/rust-generator-upgrade-20260916/`. The worker-count update freezes
Python orchestration in `source_backups/rust-workers-v2/`; the unchanged native
binary remains pinned under `source_backups/rust-teacher-v1/`.

`run(..., generation_workers=6)` (CLI `--workers 6`) explicitly selects 1–64
independent label workers. Each owns one single-threaded native search process
while labelling a family. The default remains automatic 2/4 workers. Worker
count is operational, not corpus identity: a restart preserves committed rows,
reserved seeds and durable results even when changing the count. More workers
can reduce throughput when competing with reinforcement training or memory.

Each label family shares one bounded Rust `HashMap` across its root, two
promoted descendants and ablation verification searches. Keys include exact
active history, side, orientation and remaining search depth; hash collisions
are checked with full equality. Bounds are respected and mate scores are
normalized when moving the root. Shallower entries supply move-ordering hints,
not falsely complete deeper labels. Configuration changes invalidate the table.
Unrelated roots start fresh; edited positions reuse only genuinely matching
entries. `parallel_benchmark.benchmark(binary, positions_dir, output_json)`
compares the same saved six-position workload at 4/6/8 workers in both orders,
checks result identity and separately compares a warm versus cold child search.

With the fly-brain trainer active, two trials per worker count averaged 11.70 s
(4 workers), 13.68 s (6), and 13.15 s (8) for 36 labels from 12 families.
Generation therefore resumed with four workers. All parallel results matched.
The paired child search used 864,117 nodes both warm and cold: cache hits alone
do not establish cross-root work savings. Evidence and the pre-restart corpus
backup are under `checkpoints/rust-workers-20260916/`. These short native-only
measurements exclude sampling, ablations and corpus writes, and are not a
universal scaling claim.

## Initial measurements — 16 September 2026

On this machine with both background jobs still active, six fixed positions
(two per stage), tested twice each with 7x7 pressure at weight 10:

| Target depth | Python total | Rust total, including IPC | Median paired speedup |
| --- | --- | --- | --- |
| 5 | 23.62 s | 2.76 s | 8.09x |
| 6 | 123.14 s | 16.93 s | 6.61x |

All 24 comparisons matched both selected action and score. One endgame was
proved at depth one by both implementations, so its two repetitions at each
target are proof-complete labels, not full-depth traversals. The other searches
completed their requested depths. These are small, contended single-worker
measurements, not a guaranteed production speedup.

Separately, the dataset command produced 30 accepted depth-6-target labels
(10 per stage, no rejected searches) in 69.63 s after imports/startup: 62.61 s
labelling, 6.97 s sampling, remainder symmetry keys and I/O. All 360 lazy symmetry
examples were materialized and checked for legal one-hot targets and disabled
value/Q supervision. This corpus remains isolated from both live jobs.

Validation: four Rust unit tests, six Python cross-language test groups including
100 certified tactics, 27 shallow search comparisons, 160 trajectory transitions,
static pressure/material checks, repetition, 80-ply draw boundaries, official-play
continuation, malformed input and cancellation. `cargo clippy --all-targets --
-D warnings` and formatting checks passed. Evidence is under
`checkpoints/rust-teacher-benchmark-20260916/{depth5,depth6,labels}/`.
