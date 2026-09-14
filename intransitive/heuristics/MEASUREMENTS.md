# Measured recommendation

Keep the core preset: piece count, clear-run analysis and piece-type advantage.
Leave attack, defence and overload disabled by default. Attack is the only
optional module that showed a promising diagnostic result, but the sample is too
small and its equal-time result did not improve. Defence and overload materially
reduced completed search at these budgets. No optional combination demonstrated
a practical improvement.

These are bounded acceptance measurements, not a broad strength claim. The
[readable summary](evidence/summary.json) and
[compressed full report](evidence/comparison.json.gz) preserve the configurations,
action traces, terminal reasons, held-out searches, timing and state hashes. See
the [evidence instructions](evidence/README.md) to replay it.

## Protocol

- Eight configurations: core, A, D, O, AD, AO, DO and ADO, where A is attack, D
  is defence and O is overload.
- Four fixed opponents: seeded random, greedy, the frozen accepted #16 neural
  baseline with eight full MCTS simulations, and core alpha-beta.
- Two paired seeds, both colours, for 16 games per configuration and mode: 256
  games total. Blue started every physical game.
- Diagnostic mode: depth 2, 200,000 work units per move and a 60-second safety
  deadline. Work includes search/proof nodes, move ordering, transitions, routes
  and interceptor checks.
- Practical mode: equal 100 ms per candidate move; alpha-beta variants and the
  core alpha-beta opponent received the same limit. Random, greedy and neural
  opponents retained their fixed definitions.
- The exact game engine supplied all outcomes. There was no adjudication: 110
  corner wins, 16 stalemates, 32 repetition draws and 98 no-capture-limit draws.
- The neural checkpoint SHA-256 is
  `c428e867ee3a1cebe24355d1e4f5e526cd8449b8af8d96a76a1f4bb77ab9a22c`.
  It was read from the committed #16 artifact archive; no active training run was
  read or changed.
- Conservative 95% Hoeffding score intervals use the two colour-paired seed
  units. Their radius is 0.960, so every interval is wide. Deterministic opponents
  also repeat trajectories. Treat score differences as observations from these
  fixtures, not statistically established general strength.

Weights were frozen before the final comparisons. Tuning fixtures are distinct
from the eight held-out tactical positions in
[positions.json](positions.json). The material tuning fixture scored 160.05 with
the zero-threshold-only matchup formula and 180.89 after adding the selected 0.5
gradual scarcity bonus. The predator fixture changed from -21.69 to -25.85. The
targeted tests verify that the gradual term changes smoothly while the explicit
last-predator threshold remains, and that no bonus is assigned when the relevant
piece type is gone.

## Results

W/D/L is from the candidate's perspective across all four opponents. Score gives
one point for a win and half for a draw. Depth and fallback counts cover candidate
moves only. A fallback is legal but has no completed iterative-deepening pass.

| Mode | Configuration | W/D/L | Score | Moves | Mean completed depth | Fallbacks | p50 / p95 move latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 200k work | core | 12/0/4 | 0.750 | 268 | 1.205 | 3 | 179.1 / 198.2 ms |
| 200k work | A | 12/2/2 | 0.812 | 415 | 0.860 | 87 | 178.9 / 198.9 ms |
| 200k work | D | 7/5/4 | 0.594 | 450 | 0.758 | 168 | 224.9 / 256.7 ms |
| 200k work | O | 8/2/6 | 0.562 | 329 | 0.781 | 109 | 203.8 / 230.5 ms |
| 200k work | AD | 3/7/6 | 0.406 | 310 | 0.355 | 207 | 231.1 / 252.8 ms |
| 200k work | AO | 3/5/8 | 0.344 | 299 | 0.391 | 197 | 203.4 / 228.7 ms |
| 200k work | DO | 5/7/4 | 0.531 | 465 | 0.619 | 223 | 247.4 / 276.4 ms |
| 200k work | ADO | 7/7/2 | 0.656 | 435 | 0.605 | 236 | 245.0 / 275.0 ms |
| 100 ms | core | 1/11/4 | 0.406 | 374 | 0.778 | 335 | 100.1 / 100.1 ms |
| 100 ms | A | 0/12/4 | 0.375 | 360 | 0.742 | 345 | 100.1 / 100.1 ms |
| 100 ms | D | 0/12/4 | 0.375 | 362 | 0.735 | 348 | 100.1 / 100.1 ms |
| 100 ms | O | 0/12/4 | 0.375 | 362 | 0.738 | 347 | 100.1 / 100.1 ms |
| 100 ms | AD | 0/12/4 | 0.375 | 362 | 0.732 | 349 | 100.1 / 100.1 ms |
| 100 ms | AO | 0/12/4 | 0.375 | 362 | 0.738 | 347 | 100.1 / 100.1 ms |
| 100 ms | DO | 0/12/4 | 0.375 | 362 | 0.732 | 349 | 100.1 / 100.1 ms |
| 100 ms | ADO | 0/12/4 | 0.375 | 362 | 0.732 | 349 | 100.1 / 100.1 ms |

At equal work, attack scored 0.812 overall and beat core alpha-beta 4/0/0. Its
95% interval is still [0, 1] overall and [0.040, 1] against core, so this is a
hypothesis for a larger experiment. It completed less depth than core and
generated 87 fallback moves. Enabling overload alone lost 0/0/4 to core; AO and
AD also lost 0/0/4. ADO split 2/0/2 with core. These interactions do not support
enabling defence or overload.

At equal time, core scored 0.406 overall versus 0.375 for every optional variant.
All configurations drew 0/4/0 head-to-head with core. The 100 ms budget is below
the reliable first-iteration cost in many opening/middlegame positions: core used
fallback on 89.6% of its moves, while optional variants used it on 95.8% to 96.4%.
That is itself the practical result for this short latency target. Use the default
one-second limit for interactive play and raise it when completed depth matters.

## Cost and memory

The tables below divide accumulated module time by candidate moves. Shared route
construction supports the core clear-run term. Optional-module columns show only
their own additional analysis; later core work also changes because configurations
produce different games and reach different portions of interrupted iterations.

| Mode/configuration | Routes | Proof | Clear run | Attack | Defence | Overload |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 200k/core | 4.2 ms | 50.6 ms | 100.7 ms | 0 | 0 | 0 |
| 200k/A | 4.2 ms | 44.9 ms | 105.5 ms | 10.4 ms | 0 | 0 |
| 200k/D | 4.1 ms | 42.6 ms | 90.9 ms | 0 | 59.4 ms | 0 |
| 200k/O | 3.9 ms | 39.2 ms | 99.4 ms | 0 | 0 | 34.3 ms |
| 100 ms/core | 2.5 ms | 20.0 ms | 73.5 ms | 0 | 0 | 0 |
| 100 ms/A | 2.3 ms | 18.1 ms | 69.8 ms | 5.9 ms | 0 | 0 |
| 100 ms/D | 1.7 ms | 12.6 ms | 49.1 ms | 0 | 33.0 ms | 0 |
| 100 ms/O | 2.0 ms | 15.4 ms | 59.0 ms | 0 | 0 | 19.8 ms |

Piece-count and matchup evaluation together remained below 1 ms per move in all
aggregates. The maximum shallow transposition-table estimate observed was 1.49
MiB (ADO diagnostic); process peak RSS was 441.1 MiB and includes Python, NumPy,
Numba, Torch and ONNX. The per-table estimate covers Python table, key bytes,
entries and PV tuples but does not recursively attribute every interpreter object.

The held-out searches correctly selected/proved the immediate quiet win and the
faster opposing race, kept an intercepted route heuristic, and distinguished the
blocked, capturable, overloaded, paper-supported and exposed-rock cases. Full
details and score explanations for all 128 searches remain in the evidence file.

## Decision

Ship core-only. The default configuration keeps all optional booleans false.
Attack is available for further measurement; its equal-work result is promising
but it gave up depth and did not improve the equal-time result. Defence is the
most expensive optional term here. Overload correctly fixes the targeted tactical
examples, but its measured cost and interactions do not justify enabling it.

The main engineering limitation is route-analysis cost. A future performance
iteration should cache immutable-board distance maps or replace repeated Python
route/interceptor traversal with a compiled batch. Any such optimization must
retain history-aware terminal/proof semantics and rerun the exhaustive-equivalence
suite before the optional modules are reconsidered.
