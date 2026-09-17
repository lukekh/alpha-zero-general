# Depth 4 after merging master: experimental pruning

This study merges master `032c6bb` and runs implementation `51ed837` with the
same eight legal eight-piece endgames and adopted coefficients as the
[previous depth-four comparison](../depth4/README.md). Each arm plays all eight
openings with colours swapped (16 scheduled games). Variable versus flat
material is the only evaluator difference within each arm.

Both Null Move Pruning and Futility Pruning are requested in the primary arm,
with the explicit experimental evaluator option. The control disables all three
flags. **PVS is enabled in both arms** to supply non-PV narrow windows. The old
comparison did not enable PVS, so it is not an isolated pruning baseline.
Neither material mode nor pruning is adopted as a new application default.

Both arms gave the same official and adjudicated match scores. This small
comparison found no strength improvement from enabling pruning.

## Outcomes

All results are from variable material's perspective. Unresolved means unfinished
for official scoring and inconclusive after the requested consensus rule.

| Arm | Scoring | Wins | Losses | Unresolved | Excluded failures | Unattempted |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| pruning-on | Official | 6 | 7 | 3 | 0 | 0 |
| pruning-on | Official + consensus | 7 | 7 | 2 | 0 | 0 |
| pruning-off | Official | 6 | 7 | 3 | 0 | 0 |
| pruning-off | Official + consensus | 7 | 7 | 2 | 0 | 0 |

Consensus awards an unfinished game only when both competing configurations give
the same strict nonzero sign to the final board, from the same player's
perspective. Disagreement and ties remain inconclusive. Incomplete searches are
excluded, never credited as wins. Official journals are retained unchanged.
These are eight reused opening clusters, not independent held-out validation.
Do not pool arms or earlier depths as independent strength samples.

## What pruning actually did

Seven starting boards already have fewer than four pieces on one side, which
excludes their entire continuations from either pruning method. The remaining
4-versus-4 opening can qualify until captures or other guards exclude a node.
Requested/effective settings and actual counters are recorded separately.

Played-search counters in the pruning-enabled arm:

| Material | Played moves | NMP attempts | Verified NMP cutoffs | Futility eligible | Futility pruned |
| --- | ---: | ---: | ---: | ---: | ---: |
| variable | 282 | 314 | 293 | 10457 | 0 |
| flat | 283 | 194 | 177 | 10881 | 0 |

These totals include only played searches; failed-response diagnostics remain
in the full journals. The experimental futility allowance uses absolute module
scales and the greatest current occupied piece-type value on either side in
variable mode. This is a local heuristic allowance, not a bound on nonlinear
future gains. All board guards, null verification, proof isolation and cache
separation from master are retained. See the [selective-search contract](../../../heuristics/SELECTIVE_SEARCH.md).

Selective mate-range scores must now finish the requested depth; they are not
certificates permitting an early stop. Unpruned proved results may still finish
earlier. Thus elapsed time and node totals also reflect this label distinction,
and games can follow different trajectories. Aggregate work is not a paired
speedup estimate.

A [matched-prefix diagnostic](paired-searches.json) compared completed, non-mate
depth-four searches on identical states. Logical work increased by about 23% in
each material mode with pruning enabled; the conservative guards themselves
charge work. This conditional sample is not a wall-time speedup estimate.

## Supplemental opening diagnostic

See [all diagnostic results](opening-diagnostic.json). Diagnostic status: **complete**. Eight searches compare both material modes with pruning on/off on the
initial board and the first 12 legal plies from the first frozen endgame history.
Positions are fixed without selecting for search outcomes. Each has the same
30-second / 100-million-work move ceiling. They supply no game wins or strength
ranking; input restoration and legal PVs are checked.

| Position | Material | Pruning | Completed depth | NMP cutoffs | Futility pruned | Complete |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 0 | flat | off | 4 | 0 | 0 | True |
| 0 | flat | on | 3 | 2 | 0 | False |
| 0 | variable | off | 4 | 0 | 0 | True |
| 0 | variable | on | 3 | 4 | 0 | False |
| 1 | flat | off | 3 | 0 | 0 | False |
| 1 | flat | on | 3 | 0 | 0 | False |
| 1 | variable | off | 3 | 0 | 0 | False |
| 1 | variable | on | 3 | 0 | 0 | False |

## Protocol and verification

Depth 4; 30 seconds / 100 million logical work units per move; proof depth/tokens
2/64; 192 plies / 180 active seconds per game. Each arm has a 1600-second wall
cap; the second also respects time reserved for audit. Pruning-on runs first,
then pruning-off on the shared machine. Two two-ply smoke games precede them.
The outer allocation is one hour, with a 3500-second supervisor deadline and
bounded cleanup. The opening diagnostic retains that original deadline.

The main supervisor took **2450.91 seconds**. Final cleanup
completed **2839.86 seconds** after the original launch.
Peak sampled process-tree RSS across phases was **345.5 MiB**.
Observed priorities: `[19]`; priority anomalies: **0**.
BLAS/OMP/MKL thread limits were one. No observed experiment descendants remained at cleanup.

All **34 records / 1162 played moves** passed legal replay and exact report reconstruction.
Every played selective move completed depth four; smoke games are excluded from
strength results. Integration validation passed 86 distinct focused Python
checks, 12 Rust tests and Clippy with warnings denied. Two Python label checks
initially failed to import absent dependencies; after installing those into this
worktree's local environment, both passed. The archive retains the initial
failures and successful reruns. No live teacher process or binary was changed.

## Evidence

- [Design](design.json), [per-opening results](comparison.json)
- [Consensus score pairs](consensus.json), [replay audit](verification.json)
- [Pruning counters and work](pruning-stats.json), [initial guard diagnostics](initial-guard-diagnostics.json)
- [Measurements](measurements.json), [cleanup](cleanup.json)
- [Full archive](runs.tar.gz), [SHA-256](runs.sha256)

The archive contains all manifests, journals, reports, diagnostic searches,
supervisor logs, test receipts and reproduction scripts. Run `study.py` through
`supervise.py` at the recorded implementation, after changing the original
`/tmp/issue55-variable-depth4-pruning-20260917` output root. The optional
`supervise-opening.py` uses the same original deadline; it does not start a new
hour. Saved journals can be replayed without starting engines.
