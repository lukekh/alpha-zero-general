# Search under a deadline

The player now retains useful work from unfinished iterations and shares
completed results between equivalent positions. Heuristic weights, maximum
depth and bounded proof settings are unchanged.

## Retaining exploration

Each iteration rechecks the incumbent first, then orders alternatives by the
previous root results. Immediate wins retain top priority. Cached moves from
other depths guide ordering inside the tree too.

Every fully returned root child publishes its value, depth, score bound and
principal variation. On timeout, a better completed sibling can replace the old
iteration's move. Comparisons use completed results at the same depth. An
interrupted branch never publishes its optimistic intermediate score.

If the incumbent is proved losing, prefer an alternative not yet refuted. This
fallback is labelled explicitly and may have only a shallower bound or no score.
A proved loss for one move is not a proved loss for the position.

An iteration finished exactly at the deadline is retained. Root diagnostics run
after move search, using only remaining budget. One-time rule/search kernel
compilation is warmed before starting the timed search.

Time, work, maximum depth and a proven result are independent stopping limits.
The first limit observed ends search. Every charged operation checks time before
work, so `time` is the deterministic reason when both become observable at the
same check. A proof completed within budget takes precedence over maximum depth;
using exactly the allowed work is not exhaustion unless another operation is
needed. Zero time or zero work retains the existing legal-fallback behaviour.
Post-search diagnostics use only remaining budget and never change the recorded
search stop reason.

Search remains depth-first inside each candidate. An unfinished branch may use
the remaining time, but its completed descendants remain cached and completed
root siblings remain available for the final choice.

## Folding transpositions

Keys retain current pieces, player to move, goal ownership, the no-capture clock,
and occurrence counts of every historical board-plus-player position since the
last capture. Total move number and chronological history order do not affect
future play, so they no longer prevent reuse. Exact byte keys prevent accidental
merging from hash collisions.

Different repetition counts remain distinct. A second occurrence is playable;
the official third-occurrence draw stops exploration before a cache lookup.
Merging solely by the visible board would be incorrect.

Completed scores and alpha-beta bounds are reused at the same depth. Deeper
cached moves guide ordering, but their heuristic scores do not substitute for a
shallower horizon. Mate distances are normalised on storage/retrieval.

Post-capture leaf evaluations are cached too. Quiet leaves generally have
different repetition histories; constructing keys for every such leaf cost more
than it saved in timing checks. Internal nodes always use the safe key. Cache
capacity bounds the entries and associated move hints; configuration changes
clear both. Interrupted evaluations are never cached as completed values.

## Result fields

Search results, including those copied into game records, expose:

| Field | Meaning |
| --- | --- |
| `completed_depth` | Last fully completed root iteration, or an immediate win proving the optimal score. |
| `selected_depth` | Depth supporting the selected move; may exceed `completed_depth`. |
| `partial_depth` | Interrupted depth, or zero. |
| `root_moves_completed`, `root_moves_total` | Completed branches in the latest iteration and total legal moves. A whole-root cache hit requires no fresh children. |
| `selection_source` | `completed_iteration`, `partial_iteration`, `unrefuted_fallback`, or `legal_fallback`. |
| `score_bound` | Exact or upper-bound value for the selected move; null if unavailable. |
| `tt_hits` | Completed cache entries encountered. |
| `stop_reason` | `time`, `work`, `maximum_depth`, or `proven_result`. |
| `diagnostics_status` | `completed` or `skipped_budget`; separate from search termination. |
| `effective_limits` | The actual maximum depth, time limit and work limit used, including an explicitly supplied `Budget`. |

`explanation.search_scope` distinguishes a full position search from a
selected-move result. Partial iterations do not prove all alternatives worse.
A completed branch's heuristic evaluation is still limited by its horizon.

## Verification

```sh
uv run --locked python -m unittest intransitive.tests.test_anytime_search -v
```

Deterministic-clock tests interrupt before/after child completion, refute an
incumbent, expire at the end of a full iteration, identify simultaneous limits,
and keep skipped diagnostics separate. Real work-limited search values are
checked against exhaustive minimax. Other tests cover move-number
reuse, reordered histories, distinct repetition counts, third-occurrence draws,
bounded hints, leaf reuse and input preservation.

The five known defensive failures at depth three remain separate heuristic
weaknesses. These tests remain active; timeout changes do not suppress them.

## Reversible search positions

The default alpha-beta path imports validated public storage into a private
compact board and restores each searched move in `finally` blocks. Draw-safe
keys, exact material ordering, bounded proof work and root result publication
retain the semantics above. Public game/storage interfaces remain unchanged;
`AlphaBetaPlayer(use_compact=False)` selects the copy-based comparison path.
See the [compact-search report](../benchmarks/compact/README.md) for ownership,
validation, boundary costs and repeated fixed-depth/equal-time measurements.
