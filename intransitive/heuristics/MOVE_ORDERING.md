# Move ordering v1: counter moves, continuation history, aging and IIR

Issue #69 adds four independently opt-in ordering mechanisms to the Python
alpha-beta search: `counter_move_enabled`, `continuation_enabled`,
`history_aging_enabled` and `iir_enabled`. **All four default to false**, all
four enter the search identity, and none of them changes evaluation, legal
moves, rules, the genome contract or recorded outcomes.

Ordering is not a pruning technique, so three of the four are held to a stricter
standard than the selective options: with the same requested depth completed,
they return the **same score** as plain alpha-beta. Only internal iterative
*reduction* is selective, and it declares itself as such.

## What each mechanism does

**Counter moves** (`counter_move_enabled`). One refutation per
`(side, opponent's previous action)`, written when a quiet move causes a beta
cutoff and read as an ordering key at the node that answers that move. The table
is `2 x 648` and holds an action or -1.

**Continuation history** (`continuation_enabled`, `continuation_plies` in 1..2).
History conditioned on the moves that led to the node rather than on the moving
square alone. `continuation_plies=1` indexes by the opponent's previous action;
`continuation_plies=2` adds a second table indexed by this side's own previous
action, and the two scores are **added to** the flat history score rather than
ranked above it. The tables are `(plies, 2, 648, 648)` `int32`, allocated only
for a configuration that reads them.

Both mechanisms require `ordering_enabled`; enabling them without it raises
rather than silently doing nothing, which is the interlock #66 asked for.

**History aging** (`history_aging_enabled`, `history_max` in 256..2^20).
Replaces the saturating `min(32767, h + depth^2)` update with the gravity update
`h += bonus - h*bonus/history_max`, whose fixed point is `history_max`, and
halves every history and continuation table between root iterations. Statistics
learned by the shallow iterations therefore order the deep ones without
outweighing what those find. Killers are per ply and are not aged.

**Internal iterative deepening/reduction** (`iir_enabled`, `iir_mode`,
`iir_min_depth` in 3..32, `iir_reduction` in 1..8 and at most `iir_min_depth-2`,
so a shallow pass always keeps at least one ply). Applies at a non-root node at
`depth >= iir_min_depth` that has no transposition-table or hint move, so its
children would otherwise be searched in whatever order the static keys give.

- `iir_mode='deepen'` searches the same node at `depth - 1 - iir_reduction` in
  the same window, **discards the value** and keeps only the move to try first.
  It costs nodes and leaves the completed value alone, so it is not selective.
- `iir_mode='reduce'` searches the node at `depth - iir_reduction` instead and
  stores the result at that reduced depth. It returns a shallower value than the
  caller asked for. It is therefore selective: `selective_mode_early` includes
  it, results report `selective_exact` and proof `unknown`, the transposition
  table uses the selective namespace, and a mate-range score does not end
  iterative deepening early.

Both modes are refused inside null-move and verification subtrees, at the root,
and where a move is already known.

## Ranking order

The compiled and reference ranking paths share one kernel. Enhanced ordering
(`ordering_enabled`) ranks, highest first:

1. immediate wins, then the preferred transposition-table/PV move, then previous
   root scores, then corner defence;
2. safe captures, escapes from an exposed square, captures, and — when
   `mvv_lva_enabled` — the victim/attacker values of [MVV-LVA](MVV_LVA.md);
3. **killers**, then the **counter move**, then **flat history plus the
   continuation scores**;
4. distance to the goal, then the action index as a deterministic tie-break.

A counter of -1 and absent continuation rows leave constant columns, so a
configuration with the new flags off produces exactly the previous order, the
same node and work counts and the same result. The reference `py_func` path and
the compiled kernel are the same function.

## Per-search state

Killers, history, counter moves, the continuation tables and the root path are
created or cleared by `_prepare()`, which runs at the start of every `analyze`
call. Nothing survives into the next move, and the match harness additionally
builds a fresh engine per move, so nothing can leak between candidates or games.
The path that supplies the continuation context stores -1 for a hypothetical
null turn, so a null-move probe's children borrow no context from their
grandparent. Every entry is appended and popped in a `finally`, so an
interrupted or cancelled search unwinds it.

Bounded proof search, the native proof kernels and the clear-run certificate
have their own move order and never read or write these tables.

## Diagnostics

`SearchResult.ordering` carries the switch positions, the per-mechanism counters
(`counter_move_available`, `counter_move_updates`, `continuation_updates`,
`history_updates`, `history_decays`, `iir_nodes`, `iir_reduced_plies`,
`iir_deepen_searches`) and the ordering quality measurements the issue asks for:

- `first_move_cutoff_rate` — beta cutoffs taken on the first move searched,
  over all beta cutoffs.
- `mean_cutoff_index` — mean 0-based index of the move that caused the cutoff.
- `cutoffs_by_depth` — cutoffs, first-move cutoffs and index sum per remaining
  depth, which is the distribution that predicts whether a reduction schedule at
  that depth is safe.
- `cutoff_from_preferred` / `cutoff_from_killer` / `cutoff_from_counter` /
  `cutoff_from_other` — which ordering key supplied the cutting move. These four
  sum to `beta_cutoffs`; attribution reads the killer and counter tables before
  the cutoff updates them. A move found by internal iterative deepening is
  counted under `cutoff_from_preferred`, because that is the slot it filled.

`lmr_reduced` and `lmr_researches` remain in `SearchResult.selective`; their
ratio is the re-search rate this issue is measured against.

## Cost

Each enabled key is one extra lookup per candidate move, charged to the work
budget from the configuration rather than from the data, so identical settings
always cost identical work. Continuation tables are 1.7 MiB per ply of context.
The per-iteration decay touches those tables once per root iteration.

## Measured result

The [ordering benchmark](../benchmarks/ordering/README.md) has the fixed-depth
ablation, the LMR re-search rates, the per-depth cutoff distributions and the
bounded equal-time games. The short version, and the reason every flag stays
false:

- Killers plus history already take **80.6%** of beta cutoffs on the first move
  searched at depth 4 and **97.8%** at depth 6. Counter moves, continuation
  history and aging all fire, and all change the node count by at most 0.05%.
- Internal iterative deepening removes **18.7%** of the tree at depth 6 without
  changing the completed value — but only once the tree outgrows the
  transposition table, which is the condition that leaves a node without a
  stored move. With a table large enough for the tree it never fires at all.
- The LMR re-search rate on this corpus was 0.29%–1.75% before any of this, so
  the reasoning that a high re-search rate implicates ordering does not apply
  here.
- Equal-time games: 2/14/0 in one bounded run and eight capped draws in another.
  No strength difference is established, so no default changes.

Node reduction alone is not acceptance; strength evidence is the equal-time
result with its interval.

## Native backend

The Rust teacher retains its established ordering policy, as it does for the
Python-only PVS, aspiration and selective switches. No native wire form, action
encoding, PGN notation or training-record format changes.
