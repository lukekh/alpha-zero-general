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
**off by default**. Enabling either one with evaluator scales their margins are
not calibrated for is refused by `Config::validate`, which names
`selective_evaluator_enabled`; the previous behaviour, accepting the flags and
then pruning nothing, is gone. See
[guards, verification, protocol and label semantics](../heuristics/SELECTIVE_SEARCH.md),
the [bounded validation report](../benchmarks/selective/README.md) and the
[activation measurement](../benchmarks/selective_activation/README.md).
A completed selective search is not an exhaustive label or mate certificate.

Opt-in [MVV-LVA capture ordering](../heuristics/MVV_LVA.md) uses current variable
piece values via `RustTeacher.analyze(..., mvv_lva_enabled=True)`.

## Experimental branch-parallel search

`threads` enables a Young Brothers Wait Concept split and defaults to **one**,
which is the exact sequential search. Helpers hold private transposition tables,
split points depend only on the node and the configuration, and brothers are
joined in order, so a given thread count reproduces itself; two thread counts
agree on the score and the move but not on the node count. Proof and certificate
search stay sequential, a parallel mate reports `parallel_result` rather than
`proven_result`, and `Genome.native_arguments` pins `threads = 1` so no evolved
weights or tournament result can come from a parallel search.

See [the contract](PARALLEL_SEARCH.md) and the
[measured report](../benchmarks/ybwc/README.md) for the numbers and the decision.


## Route feature performance

`src/routes.rs` mirrors `intransitive/heuristics/{flood,geometry,features}.py`,
including the four structural changes those carry: a bitboard flood fill over
the crate's existing 81-square `u128`, one multi-source fill per predator type
so safety is a lookup rather than a scan of every enemy, each runner's
interception squares and deadlines built once instead of once per defender, and
an attack pass that walks the eight neighbours because both of its tests require
distance one.

Rust gains more than Python did, because the previous version recomputed and
reallocated each piece's shortest-path set on every call. Measured on ten
positions with fixed-depth searches, identical selected move, score and node
count at every depth:

| Depth | nodes | before | after | |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 549,391 | 8.90 s | 1.70 s | **5.23x** |
| 5 | 3,181,533 | 81.25 s | 15.02 s | **5.41x** |

That is 61,710 to 322,577 nodes per second at depth four.

Equivalence is checked three ways: `cargo test --release` keeps the replaced
queue search and per-enemy scan as reference implementations and asserts the
flood fill, threat maps and runner profiles reproduce them; the rebuilt binary
matches the previous one to exact float equality on both route terms across 500
random positions; and both match the Python evaluator with a worst absolute gap
of zero.


## Proof gate

`Search::proof_safe` mirrors `no_terminal_win_in_horizon`, including the
spacious-piece stalemate rule that clears sparse boards the pair rule cannot.
It cuts proof nodes by 34% at depth four and 50% at depth five on the benchmark
positions, with the selected move and score identical at both depths. Wall time
is essentially unchanged, because a native proof node already costs around 0.15
microseconds — the gain here is node budget, not seconds.

`cargo test --release` checks the gate against an exhaustive two-ply expansion:
every position it clears must genuinely contain no terminal result.


## Clear-run certificate

`src/clear_run.rs` mirrors `intransitive/heuristics/clear_run.py`, including the
starting-square check that the first Python version was missing. Enable it with
`certificate_enabled=True` on `RustTeacher.analyze`; the wire form appends one
optional field, so existing callers and binaries are unaffected.

Python and Rust certify the same 492 of 492 positions once both use the real
no-capture clock. `cargo test --release` checks short claims against an
exhaustive expansion of the same depth.
