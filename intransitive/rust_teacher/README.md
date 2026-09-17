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
client also provides `inspect` and `apply`. Set `weight=0, attack=0, defence=0` for material-only search. `reuse=True` retains compatible history-aware table entries between
requests. Incomplete searches are explicitly marked and should not become
training labels. Rust library callers read a position through `board()` and
change it through `apply`, preserving derived evaluation state.

## Supported scoring

The adopted defaults are material 100, advantage 23.967050360966205,
attack 25.714516982666414 and defence 32.5643023919054; pressure and overload
are off. Advantage bonuses remain 1/.5/.25. Attack and defence use the same
static occupied-board routes, capture safety, interception timing and goal
blocking as Python. Cross-language tests cover both goal orientations, legal
trajectories, signed coefficients and candidate changes. Overload is not
implemented in Rust and is rejected by the native genome contract.

`analyze` and `inspect` accept signed `material`, `advantage`, `attack`, `defence`
and pressure `weight` arguments; omitted arguments use the adopted defaults and
radius four. Explicit old coefficients remain usable. Library searches include
`Weights` in `Config`, and all coefficients participate in cache compatibility.
The wire protocol retains old requests (using the new defaults) and additionally
accepts material/advantage/attack/defence before the state hex. The Python adapter
sends all weights explicitly. Pressure supports radius three or four; boards
above 32 pieces retain the reference pressure kernel's summation order.

Proof depths 0–2 and up to 64 proof nodes are supported. Native node limits
count actual search/proof visits; Python's logical work allowance uses different
units. Compare completed fixed-depth results under generous limits. Native PVS,
FIFO table replacement and ordered-history keys can produce different tied
choices from the Python teacher while preserving optimal completed-depth scores.

Importing this module or building its binary does not select it for existing
browser or dataset processes. Callers explicitly instantiate `RustTeacher`.

The optional [variable-material replacement](../heuristics/VARIABLE_MATERIAL.md)
uses the same BASE=100, REG=0.25 formula as Python. Pass
`variable_material_enabled=True` to `RustTeacher.inspect` or `.analyze`.
Flat material remains the default; switching modes resets reused search state.

## Experimental selective search

Independent `nmp_enabled` and `futility_enabled` options are available but remain
**off by default**. See [guards, verification, protocol and label semantics](../heuristics/SELECTIVE_SEARCH.md)
and the [bounded validation report](../benchmarks/selective/README.md).
A completed selective search is not an exhaustive label or mate certificate.
