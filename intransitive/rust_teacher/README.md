# Native minimax teacher

A dependency-free Rust search library and persistent Python client for the
validated state-v2 board format. It implements move generation, make/unmake,
corner/stalemate wins, repetition and no-capture draws, bounded terminal proofs,
iterative alpha-beta/PVS and a history-aware transposition table.

Material evaluation maintains side/type counts, per-type contributions and
piece-location masks. Pressure evaluation updates attacker/defender rings and
affected victims after moves and captures. Both accumulators unwind with search;
proof-only moves avoid maintaining unused heuristic data.

The [incremental evaluation report](INCREMENTAL_EVALUATION.md) includes exactness
checks, before/after measurements, source snapshots and reproduction commands.
On the recorded workload, paired CPU savings were about 3% for material-only
search and 11–12% with pressure enabled. All 108 search comparisons matched
scores, selected moves, principal variations and node counts exactly.

Legal generation and terminal mobility checks use six incremental bitboards,
including temporary proof and ordering moves. The board array and action order
are preserved. See the [Python/Rust legality benchmark](../benchmarks/legal_moves/README.md)
for correctness checks, node throughput, memory cost and reproduction commands.

Neighbour bitboards also accelerate move-ordering threat checks, and depth hints
reuse history keys. See the [follow-up benchmark](../benchmarks/search_efficiencies/README.md).

## Build and validate

Run from the repository root with Rust 1.74+ and the project Python environment:

```sh
cargo build --release --offline --manifest-path intransitive/rust_teacher/Cargo.toml
cargo test --release --offline --manifest-path intransitive/rust_teacher/Cargo.toml
cargo clippy --offline --manifest-path intransitive/rust_teacher/Cargo.toml --all-targets -- -D warnings
.venv/bin/python -m unittest intransitive.tests.test_rust_teacher -v
```

The native tests check bit-exact reference evaluation, trajectories, captures,
undo, cancellation, pressure dependencies and dense-board fallback. The six
cross-language test groups cover 100 certified tactics, 27 shallow searches,
160 trajectory transitions, draw rules and malformed input.

## Python interface

```python
from intransitive.rust_teacher import RustTeacher

with RustTeacher() as teacher:
    result = teacher.analyze(state, depth=6, radius=3, weight=10., seconds=60.)
    if result['complete']:
        action = result['action']
```

`state` is the existing `(9, 9, 84)` int8 board including active history. The
client also provides `inspect` and `apply`. Set `weight=0` for material-only
search. `reuse=True` retains compatible history-aware table entries between
requests. Incomplete searches are explicitly marked and should not become
training labels. Rust library callers read a position through `board()` and
change it through `apply`, preserving derived evaluation state.

## Supported scoring

Material weights match the current defaults: count 100, advantage 25,
predator-zero 1, scarcity 0.5 and prey 0.25. Pressure supports radius 3 or 4 and
a configurable weight. Boards above 32 pieces use the reference pressure kernel
to preserve floating-point summation semantics. Legacy route-based attack,
defence and overload modules are outside this native implementation.

Proof depths 0–2 and up to 64 proof nodes are supported. Native node limits
count actual search/proof visits; Python's logical work allowance uses different
units. Compare completed fixed-depth results under generous limits. Native PVS,
FIFO table replacement and ordered-history keys can produce different tied
choices from the Python teacher while preserving optimal completed-depth scores.

Importing this module or building its binary does not select it for existing
browser or dataset processes. Callers explicitly instantiate `RustTeacher`.

## Experimental selective search

Independent `nmp_enabled` and `futility_enabled` options are available but remain
**off by default**. See [guards, verification, protocol and label semantics](../heuristics/SELECTIVE_SEARCH.md)
and the [bounded validation report](../benchmarks/selective/README.md).
A completed selective search is not an exhaustive label or mate certificate.
