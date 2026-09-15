# Experimental local RPS pressure

The follow-up [search optimisations](OPTIMIZATIONS.md) provide independent PVS,
aspiration, learned/native ordering, table replacement/capacity and pressure
caching experiments without changing the pressure heuristic.

This experiment is disabled by default (`pressure_enabled=false`). It does not
modify/restart the live trainer or generators, their frozen source snapshots,
or existing dataset labels. `pressure_weight` is separately configurable and
participates in the search/transposition configuration identity.
`pressure_radius` selects the window: 4 preserves the original 9x9 experiment,
and 3 enables the 7x7 experiment. Other radii are rejected. Neither is enabled
by default, and each has a separate opt-in configuration file.

## Exact scoring

Use a 9x9 window centred on each piece, with empty off-board padding and square
(Chebyshev) distance. Ignore blockers, side-to-move tempo and defender exposure.
The 7x7 variant drops distance four entirely, retaining weights 2, 1 and 1/2;
it does not rescale the remaining rings. Its defender discounts are otherwise
identical and still centred on the victim.

| Distance | Window boundary | Undiscounted pressure |
| --- | --- | --- |
| 1 | 3x3 | 2 |
| 2 | 5x5 | 1 |
| 3 | 7x7 | 1/2 |
| 4 | 9x9 | 1/4 |

Same-type enemies, friendly pieces and enemies beyond distance four contribute
no direct attack/threat. Friendly pieces can still contribute defence.

For an opposing attacker/victim pair at distance `r`, count friendly defenders
of the **victim**, within distance `r` of that victim, which can capture the
attacker. Multiply pressure by `2 ** (-defender_count)`.

For example, friendly paper defends scissors against rock. Adjacent paper halves
all the scissors' threat penalties. Paper at distance four halves only its
distance-four penalties. Two qualifying papers divide by four, even if either
paper is itself threatened. Defender counts are unconditional and inclusive of
the ring boundary. The attacker receives the same discounted credit that the
victim receives as a penalty; defenders are centred on the victim, not attacker.

We sum Blue's attack credits and Red's attack credits separately, then subtract
opponent from own. This equals summing signed pressure over your own pieces;
each opposing interaction is counted **once**, not doubled. Explanations expose
both nonnegative attack totals and their weighted difference as `local_pressure`.
Pressure supplements material/type advantage; ordinary scores retain their
existing clamp, and terminal/proven wins take precedence over any pressure weight.

## Implementation and speed

`heuristics/pressure.py` implements the equivalent sparse operation with Numba:
scan occupied squares, use a precomputed 81x81 distance table, accumulate three/four
defender counts per victim, then evaluate each opposing pair once. No general
tensor-convolution framework, route calculation, or compact-state export is
needed. Both contiguous and public strided-board signatures are warmed before
timed searches. A bounded work charge precedes each native evaluation, and the
existing deadline is checked before/after. Disabled or zero-weight configurations
do no pressure work.

Initial M1 measurements, with live training/generation left running:

- Compiled pressure kernel: about **1.0 microsecond** per opening board.
- Scalar evaluator, excluding terminal proof search: median **9.7 -> 15.1 us**
  over five groups of 2,000 calls. These include terminal checks and budgeting.
- Full depth-five opening search: baseline **6.3 seconds / 332,057 nodes**;
  weight 10 **18.2 seconds / 820,962 nodes**. The scoring change alters the search
  tree, so a fast kernel does not imply unchanged end-to-end generation speed.
  These are initial observations under contention, not stable throughput claims.

## Calibration controls

The user's ten-ply game is replayed and checked against its exported SHA-256.
Comparing each forced move plus four searched plies gives:

| Pressure weight | Bad `F6-E5`, Red score | Safer `F6-F5`, Red score |
| --- | --- | --- |
| 0 | -0.625 | -0.625 |
| 5 | -13.750 | 1.094 |
| 10 | -26.875 | 2.812 |
| 25 | -66.250 | 7.969 |

Thus the feature distinguishes the original sacrifice at five plies rather
than leaving it tied. It does **not** establish globally optimal play: fresh
full-root searches selected F6-F5 at weight 0, E7-D7 at 5/10, and E7-D6 at 25.
The latter moves need broader strength evaluation; changing an action is not
automatically an improvement. Each search completed all five plies.

All four weights passed **100/100 independently certified tactical fixtures**
at depth three (immediate wins, runs, goal defence, saving pieces, safe captures).
Separate depth-five retreat and immediate-win controls passed at all four weights.
These are known calibration positions, not held-out evaluation. Weight 10 is
an illustrative experimental preset, **not** a tuned or recommended production
weight. Evidence is in `checkpoints/pressure-experiment-20260915-v1/`:
`results.json` contains timings/root searches; `calibration.json` contains forced
alternatives, all tactical decisions and source hashes. Live jobs remain unchanged.

## Run it

```sh
.venv/bin/python -m unittest intransitive.tests.test_pressure
.venv/bin/python -m intransitive.benchmarks.pressure.reproduce \
  --output checkpoints/pressure-new-run --weights 0 5 10 25 --depth 5 --seconds 120
.venv/bin/python -m intransitive.benchmarks.pressure.calibrate \
  --output checkpoints/pressure-new-run/calibration.json
.venv/bin/python -m intransitive.play --opponent alphabeta \
  --ab-config intransitive/heuristics/configs/pressure-experimental.json --port 8767
```

Compare window sizes at weight 10, with three interleaved full depth-five
opening searches each, microbenchmarks, forced opening alternatives and 100
certified tactical controls per configuration:

```sh
.venv/bin/python -m intransitive.benchmarks.pressure.compare_windows \
  --output checkpoints/pressure-window-comparison/results.json
.venv/bin/python -m intransitive.play --opponent alphabeta \
  --ab-config intransitive/heuristics/configs/pressure-7x7-experimental.json --port 8767
```

Both `reproduce` and `calibrate` also accept `--radius 3` for independent 7x7
weight sweeps. Their default remains 4 to preserve the original experiment.

The play preset permits up to 120 seconds per move to complete depth five;
it is deliberately an analysis preset, not a low-latency default. Existing PGN
analysis also accepts it with `--config`. Change `pressure_weight` in a separate
experimental config when comparing strengths.

Known limitations to test next: blocked predators still exert pressure,
trapped/overloaded defenders still count, and tempo or a counter-race can make
an attractive static attack strategically wrong. Evaluate on held-out games
and equal-time matches before using these labels in supervised training.

## 7x7 follow-up (2026-09-15)

Three interleaved repeats on the same opening, with weight 10 and complete
depth-five searches, gave the following medians. Live training/generation
continued throughout; compilation was warmed and excluded.

| Variant | Search seconds | Nodes | Static evaluator, us |
| --- | --- | --- | --- |
| Material baseline | 6.765 | 332,057 | 12.09 |
| 9x9 pressure | 19.373 | 820,962 | 15.32 |
| 7x7 pressure | 15.213 | 618,917 | 14.45 |

7x7 reduced measured search time by **21.5%** relative to 9x9 on this opening,
with **24.6% fewer nodes**. It still took about 2.25x the baseline time. Kernel
microtimings were noisy and around one microsecond for both sizes; the main
observed gain is in the changed search tree, not a demonstrated convolution
kernel speedup. This is one position, not a general throughput/strength claim.

All three configurations passed 100/100 certified tactical controls. The 7x7
forced five-ply opening comparison still distinguished the sacrifice:
`F6-E5 = -25.9375`, versus `F6-F5 = +2.1875` from Red's perspective. Its full-root
selection was E7-D6, versus E7-D7 for 9x9; neither alternative is thereby proved
globally better. A smaller window discards information and can change play.

Raw timings, every tactical decision, configurations and source hashes are in
`checkpoints/pressure-7x7-20260915-v1/results.json`. The original 9x9 preset/default
radius remains available; the separate `pressure-7x7-experimental.json` preset
opts into 7x7. No live teacher, dataset or checkpoint was changed.
