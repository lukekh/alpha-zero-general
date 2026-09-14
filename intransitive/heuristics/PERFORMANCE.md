# Search performance, 14 September 2026

This records the earlier evaluation optimisation. Subsequent
[deadline and transposition changes](ANYTIME_SEARCH.md) improve move ordering,
retain completed branches at interruption, and replace wire-format cache keys
with equivalent future-play identities.
The later [incremental-material report](../benchmarks/material/README.md) records
paired fixed-depth/equal-time measurements, allocations and numerical guards
from the merged compiled-proof baseline.

These changes preserve the v2 evaluation formulas, binary clear-run scoring,
move ordering, full game history and fixed-depth minimax results. A time-limited
search can now complete more depth and may consequently choose a different move.

## Measurements

Warm runs on the development Mac, using the project uv environment, empty search
tables, and the same positions and configurations before/after. The separate
neural training process continued running. These are individual local timings,
not a general speed guarantee. Fixed-depth cases used a 60-second/one-billion-work
safety cap; the timed case retained the user's 10 seconds and 20-million-work cap.

| Position / search | Before | After | Result |
| --- | ---: | ---: | --- |
| Opening, core, depth 2 | 0.479 s | 0.050 s | Same move and score |
| First supplied game, core, depth 3 | 9.688 s | 0.165 s | Same F6–E6, score 0, same PV |
| First supplied game, all modules, depth 2 | 1.798 s | 0.502 s | Same G6–H7 and score −95.6071 |
| First supplied game, 10-second core search | Depth 3 | Depth 5 | Both selected F6–E6 |

A separate comparison reproduced both user examples exactly at depth 3, including
scores and complete principal variations: first game 9.565 → 0.152 seconds;
second game 6.760 → 0.109 seconds. The second example also reached depth 5 under
its original 10-second budget. No claim about playing-strength improvement follows
from these latency/depth measurements alone.

## Changes

- Separate scalar scoring from route explanations. The old first-game search
  spent approximately 7.4 of 10 seconds generating diagnostic clear-run candidates
  whose scores were always zero. Explanations remain available through the analyser.
- Skip route maps when all optional positional modules are off. Piece count and
  type advantage require only the pieces, not routes across the board.
- Reuse shortest-route, safety and interception results within a position.
- Skip bounded proof expansion only when distance and a conservative mobility
  certificate exclude any terminal win inside that horizon. This does not prune
  ordinary minimax or award a win. See the proof in `kernels.py`.
- Preserve move ordering using compiled immediate-win checks, then construct full
  child history states lazily so cutoffs avoid unused transitions.

## Validation and reproduction

```sh
uv run --locked python -m unittest intransitive.tests.test_search_performance \
  intransitive.tests.test_heuristics intransitive.tests.test_play \
  intransitive.tests.test_record
uv run --locked intransitive-analyze position.pgn --last-ai --depth 3 --time 60 --work 1000000000
uv run --locked intransitive-analyze position.pgn --last-ai
```

The committed performance regression tests contain the exact move histories of
both supplied examples. They compare scalar and detailed evaluation across all
eight module combinations, verify unchanged original moves/scores, compare
immediate-win flags against full rules-engine transitions, enumerate complete
two-ply trees for the no-win certificate (including canonical colours), and check
that move ordering constructs only visited child states. Existing tests cover
draw precedence, mate scores, proof exhaustion, history-sensitive caching and
search/work limits. Timing thresholds are deliberately absent from unit tests.

A separate differential run against a snapshot of the pre-change code matched
428 bounded-proof outcomes and 216 evaluations, including raw features, terms and
route explanations, across replayed and seeded random positions. Both original
depth-three searches matched move, score and PV. Local raw measurements, profiles,
the comparison script and the pre-change source snapshot are retained in
`checkpoints/search-performance/`; the supplied records and candidate comparisons
are in `checkpoints/position-analysis/`.

## Second position: preserving the scissors on F4

The played `E7–D6` anticipates `E3×F4 D6×D5`: Red trades a scissors for Blue's paper.
At both depths 3 and 5, the existing core evaluator scores that choice **+1.9792**
for Red, while retreats `F4–G5` and `F4–F5` score **−0.6250**. The depth-five PV may
delay the recapture. This is the existing type-advantage preference for the trade,
not a failure to see the scissors can be captured. Performance work leaves that
preference unchanged; improving it is a separate heuristic decision.
