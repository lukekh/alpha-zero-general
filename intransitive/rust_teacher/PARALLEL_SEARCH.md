# Branch-parallel search in the native teacher

Issue #65. An opt-in Young Brothers Wait Concept (YBWC) split for the Rust
search. **Off by default**: `threads = 1` is the exact single-threaded search,
including its node counts and the order it visits nodes in. The
[measured report](../benchmarks/ybwc/README.md) carries the decision and the
numbers; this file is the contract.

There is no parallel Python search and this does not propose one. Every njit
kernel in `intransitive/heuristics/` holds the GIL — there is no `nogil`,
`parallel=True` or `prange` anywhere in it — and `AlphaBetaPlayer._search` is
plain Python recursion over per-instance mutable state (`self.table`,
`self._material_counts`, `self._selective_stats`, the `SearchPosition`
make/unmake stack). Worker threads there would serialise and cost more than
they saved.

## Why not the obvious schemes

Lazy SMP is the simplest thing that works in a normal engine and is unusable
here. It shares one transposition table, so thread interleaving decides what is
in the table, which decides scores, principal variations and chosen moves. The
paired tournament (#54) and the weight-evolution harness (#55) both require
reproducible outcomes, and #55 carries a test asserting that a run's decisions
do not depend on `match_workers`. A search whose answer depends on scheduling
cannot sit underneath either of them.

Root splitting is reproducible but wastes the bound propagation that makes
alpha-beta cheap. YBWC keeps it: the eldest child is searched first and alone,
and only once it has returned a bound do its younger brothers start.

## What is actually parallel

Splitting happens on the master thread only, at nodes reached down the eldest
child. A helper never splits, so live threads are exactly `threads` and a
helper's subtree is an ordinary sequential search.

At a split node the master has already searched the eldest child. The remaining
brothers are then processed in chunks of at most `threads`:

1. The window is **frozen** at the chunk's current `(alpha, beta)`.
2. The master takes the first brother of the chunk itself, helpers take the
   rest, one each, assigned by position in the chunk.
3. `std::thread::scope` joins every helper before the chunk returns.
4. Results are applied in **brother order**, exactly as the sequential loop
   applies a child: best/alpha update, root score record, killer and history
   update on a cutoff, and stop as soon as `alpha >= beta`.
5. If brothers remain and no cutoff happened, the next chunk starts from the
   newly raised alpha.

Each helper runs the same principal-variation logic the sequential loop runs for
a non-eldest child: a null-window probe, then a full re-search if the probe
lands inside the window.

## Why a frozen window is sound

A frozen alpha is never above the live one, so a helper searches with a window
at least as wide as the sequential search would have used. Widening a window
never invalidates a result: an exact value stays exact, a value at or below the
frozen alpha is an upper bound and is therefore also below the higher live
alpha, and beta never moves inside a node, so a fail-high stays a fail-high.
Every outcome is usable as it stands, and no brother is ever re-searched merely
because another brother finished first.

The price is the standard parallel search overhead. Brothers start without the
bound their elder brothers would have handed them, so the tree is larger for the
same nominal depth. That is why node throughput is reported for diagnosis and
never used to accept the feature.

## Why a fixed thread count reproduces itself

Four things are pinned:

- **Fixed split points.** `Search::splits` looks only at the node's remaining
  depth, its unsearched brother count and the configuration. It never asks
  whether a helper happens to be idle.
- **Private tables.** Each thread owns its transposition table. Helper tables
  are fresh on every request, including under `search_reuse`, so a result never
  depends on the request history of a thread other than the master's.
- **Frozen windows and deterministic assignment.** A brother's window and the
  thread that searches it are fixed before any thread starts.
- **Deterministic join.** Results are applied in brother order and counters are
  summed in brother order, so neither the answer nor the reported work depends
  on which thread finished first.

Helpers inherit the master's killer moves and history at the start of a chunk
and their own updates are discarded. That is deterministic, and it is also why
determinism is a claim **per thread count**: two thread counts search different
trees and record different node counts. They agree on the answer, not the work.

A search that runs out of time is reproducible in neither mode, exactly as the
single-threaded search already was.

## Budget, time and cancellation

Node budgets are reserved, never raced. At the start of a chunk the remaining
allowance is divided between the participants and each one is handed its slice
before any of them runs. A chunk therefore cannot spend more than the search had
left, whatever order the threads finish in, and the reported work does not
depend on thread timing. Node exhaustion deliberately does **not** signal the
other threads: each participant stops on its own slice, which keeps node counts
reproducible even on an aborted search.

The time limit does signal. The first thread to see the deadline sets a shared
flag that every other thread reads on its next node, so a time-out reaches all
of them within one node instead of one clock read each. The flag is cleared at
the start of the next request, so a timed-out search cannot silently stop the
one after it.

`std::thread::scope` joins every helper before the split returns — on cutoff,
on early return, on time-out and on panic — so no worker outlives the split that
created it. A panicking helper is reported as a failed brother rather than
being allowed to poison the result.

## What stays sequential

- Proof search and the clear-run certificate. They carry exactness claims and
  are never split; each one is computed entirely inside a single thread.
- Null-move probes, their verification searches and any node reached with
  selective pruning disabled. Their recorded behaviour stays the behaviour #60
  measured.
- Any node below `split_min_depth`, and any node with fewer than
  `split_min_siblings` unsearched brothers.

A parallel search is exact minimax at its completed depth, but it is not the
single-threaded proof the teacher-label paths accept. So it never borrows that
name: a mate found with helpers running reports `stop_reason` of
`parallel_result` rather than `proven_result`, and the Python client reports
`score_bound` of `parallel_exact` rather than `exact`. `supervised_minimax` and
`defence_examples` accept `proven_result`, so existing teacher labels continue
to be generated single-threaded unless someone explicitly rebaselines them.
`Genome.native_arguments` pins `threads = 1`, so no evolved weight set and no
tournament result can be produced by a parallel search.

## Settings

| Setting | Default | Range | Meaning |
|---|---:|---|---|
| `threads` | 1 | 1..64 | Total live threads, master included. One is the exact sequential search. |
| `split_min_depth` | 4 | 2..32 | Remaining depth a node needs before its younger brothers may be split. |
| `split_min_siblings` | 2 | 2..64 | Unsearched brothers a node needs before a split is worth the position clones. |

Both interlocks are checked when the configuration is loaded, in
`Config::validate` and in `RustTeacher.parallel_settings`, and each refusal
states its reason:

- `threads * table_entries <= 1000000`. Each thread owns a private table, so the
  memory ceiling is the product. A shared table would be cheaper and would be
  nondeterministic.
- `split_min_depth > futility_max_depth`. A split node hands its whole tail to
  the helpers, which do not apply the futility margin, so overlapping the two
  would silently drop pruning the selective counters claim to be reporting.

## Wire protocol and rollback

`threads`, `split_min_depth` and `split_min_siblings` are three further
positional fields, appended after the certificate-as-a-bound group from #67, at
request lengths 28, 29 and 30. Because the table is positional, asking for
threads also sends that group at its own defaults. The client only sends any of
them when they differ from the defaults, so a default request is byte-identical
to the one older binaries already accept. Rolling back is deleting the argument:
`threads = 1` restores the frozen baseline exactly, which
`intransitive.benchmarks.ybwc.reproduce --stage equivalence` checks row by row
against a record made with the pre-change binary.

The search response gains a `parallel` object reporting the settings in force
and the observed split nodes, chunks, brothers handed to a helper and helper
nodes. `search_identity` records the three settings and
`deterministic_for_thread_count`.

### One fix this needed first

The positional length table in `main.rs` is a prefix table: a group must be
listed on every length at or above the one that introduced it. Two were not.
`certificate_enabled`'s length (24 tokens) appeared in none of the
`has_weights`, `has_mode` or `has_selective` sets, and the certificate-bound
group's length (27) inherited the same hole, so **any request enabling the
certificate, or using the certificate as a search bound, silently fell back to
the default weights, flat material and default selective settings.**

Confirmed against the pre-change binary: a depth-three opening search with
`material=1, advantage=0, attack=0, defence=0` scores 0, and the same request
with `certificate_enabled=True` scored 16.948 — the default weights' answer.

Both are fixed here, because the parallel fields sit behind those lengths and
would have inherited the hole a third time. The certificate and its bound are
off by default, and the clear-run certificate's decision does not depend on
evaluation weights, so the recorded parity and bound results stand; but any
future measurement that combines those options with non-default weights needs
this fix to mean what it says.
