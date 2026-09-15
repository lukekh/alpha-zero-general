# Exact-search optimisation experiments

These switches change search organisation or reuse, not the 7x7 pressure
formula, its weight, terminal proofs or modelling-draw rules. All new switches
default off. Live training/generation retains its frozen source and was not
restarted or modified. No dataset labels are rewritten.

## Implemented options

| Configuration | Implementation |
| --- | --- |
| `pvs_enabled` | Existing principal-variation probing, with full re-search for improving alternatives. |
| `aspiration_enabled` | Existing narrow initial score window, with bounded widening and full-window recovery. |
| `ordering_enabled` | Prioritise safe captures and escapes from adjacent predators; retain two quiet cutoff moves per ply and bounded side/action history scores. |
| `compiled_ordering_enabled` | Numba move-ranking and sorting loop. Can accelerate the original ordering without enabling the new ordering heuristic. |
| `depth_replacement_enabled` | OrderedDict table with bounded eviction sampling: prefer deeper/exact records among the eight oldest candidates. |
| `table_entries` | Existing configurable capacity; this experiment tests 10,000, 50,000 and 100,000. |
| `pressure_cache_entries` | Bounded evaluator-owned LRU of board-local pressure totals; 0 disables, experimental capacity 8,192. |

### Ordering and compilation

Immediate wins, the transposition/PV move, previous root scores and urgent
goal replies retain precedence. Enhanced ordering then prefers safe captures,
safe escapes, other captures, quiet cutoff moves, history scores and progress.
History updates occur only on quiet beta cutoffs; scores saturate at 32,767.
Killer/history arrays reset on each `analyze`, making fresh runs reproducible.
No candidate is removed and no depth reduction is introduced. Tied minimax
actions/PVs can change; fully completed same-depth scores must agree.

`ordering.py` compiles rank construction and bounded sorting, including the
adjacent-predator checks. With enhanced ordering disabled it exactly reproduces
the old ranking, including fractional previous-root scores and deterministic
action ties. A Python reference path remains available for comparison.
Contiguous and strided signatures are warmed before timed search. Kernel calls
are bounded to one node's legal move list, with work/deadline checks before and
after, plus checks while yielding children.

This is **not** a fully compiled recursive search engine. Recursion, root
progress, table bounds, push/pop history and cancellation remain in Python.
That isolates compilation from the history-sensitive controller; a complete
native-engine rewrite is not part of this experiment.

### Cache correctness

The transposition table still keys exact depth and the complete future-play
history identity. Different-depth records remain ordering hints, not substitute
evaluations. Eviction keeps hint references valid and examines at most eight
candidates. Configuration changes clear the search table.

The pressure cache stores only `(blue_attack_total, red_attack_total)` under
`(pressure_radius, exact_board_bytes)`. Blocker-blind pressure does not depend
on turn, goals, history, or its final weight. Terminal/draw checks and terminal
proof searches happen outside this cache, so repeating a board cannot reuse an
old win/draw result. Raw pressure is scaled by the current weight after lookup.
Hits still consume the same logical work charge and obey the deadline. Radius
changes cannot alias; lowering capacity prunes the LRU. This implements caching,
not incremental update/undo of each contribution.

## Reproduce

```sh
.venv/bin/python -m unittest intransitive.tests.test_search_optimizations
.venv/bin/python -m intransitive.benchmarks.pressure.optimize_search \
  --output checkpoints/optimization-screen-new/results.json
.venv/bin/python -m intransitive.benchmarks.pressure.optimize_search \
  --output checkpoints/optimization-repeat-new/results.json \
  --variants baseline pvs all --repeats 3
.venv/bin/python -m intransitive.play --opponent alphabeta \
  --ab-config intransitive/heuristics/configs/pressure-7x7-all-optimizations.json \
  --port 8767
```

The all-options preset is an experiment, not a claim that every switch helps
every position. The benchmark isolates each option, tests their combination,
checks completed depth-five scores and runs the 100 independently certified
tactical controls. Output includes source hashes, configurations, counters,
completion status and table-size estimates. The latter are not full process
RSS or pressure-cache memory. Larger tables consume more RAM per worker.

Tests include all 64 flag combinations against an exhaustive small-tree score,
original/native ordering equality on reached positions, Python/native enhanced
ordering equality, complete legal candidate coverage, compact/public search
equivalence, table/hint eviction, cache capacity and radius separation,
history-sensitive draws, bounded work and exception/cancellation rollback.
Timing runs are warmed and low-priority; live jobs remain active, so small
timing differences under contention are not sufficient evidence of a win.

## Results and suggested trial configuration (2026-09-15)

220 targeted unit/regression tests passed. Every screened configuration passed
all 100 certified tactical controls. These are not the entire repository suite
or evidence of general playing strength.

The opening screen isolated 11 configurations. All completed depth five with
the identical score, move and PV. Initial times in seconds:

| Configuration | Seconds |
| --- | --- |
| Existing 7x7 search | 13.58 |
| PVS only | 4.83 |
| Aspiration only | 7.82 |
| Both windows | 5.48 |
| Compiled original ordering | 12.80 |
| Enhanced ordering, Python reference | 25.14 |
| Enhanced ordering, compiled | 5.48 |
| 50k/depth-aware table | 8.00 |
| 100k/depth-aware table | 7.35 |
| Pressure cache only | 12.93 |
| All options | 2.97 |

The 50k/100k runs retained identical entries/node counts; their small timing
difference is not evidence that 100k helps. Table capacity and replacement are
combined in those two variants, not independently attributed speedups. Native
and Python enhanced ordering visited exactly the same 268,430 nodes, versus
618,917 for the original rank: compilation removes the reference implementation's
Python sorting cost. Prefer enabling enhanced ordering with compilation.

Three rotated-order repetitions followed, including a second recorded position
before Red's 18th move (35 plies played). Medians in seconds:

| Position | Existing 7x7 | All options | All except pressure cache |
| --- | --- | --- | --- |
| Reported opening blunder | 13.206 | 3.133 | 3.071 |
| Recorded midgame | 6.821 | 2.078 | 2.036 |

Every repeated search completed depth five with identical score, move and PV
within its position. Without pressure caching, the measured speedup was about
4.3x on the opening and 3.35x on the midgame. Main+proof nodes decreased from
618,917 to 165,726 and from 332,202 to 121,264 respectively. No scoring terms,
proof horizons, rules or repetition identities were weakened to obtain this.

Pressure caching had 47% and 26% hit rates in the combined runs, but its lookup,
copy and LRU overhead did not demonstrate a runtime benefit. It is implemented
and available in `pressure-7x7-all-optimizations.json`, but the suggested trial
preset `pressure-7x7-optimized.json` leaves that cache off, retaining all other
optimisations and a 50k table. The roughly 2% elapsed difference between the two
combined variants is too small to treat as a robust speed claim; disabling the
cache also avoids its additional memory. The recursive controller is still Python.

```sh
.venv/bin/python -m intransitive.play --opponent alphabeta \
  --ab-config intransitive/heuristics/configs/pressure-7x7-optimized.json --port 8767
```

Raw evidence lives in `checkpoints/search-optimizations-20260915-v1/`:
`screen.json`, `repeated.json` and `midgame.json`. The reproduction command
accepts `--position opening|midgame|late_game` and `--variants` selections,
including `all_no_cache`. All new options remain off in ordinary defaults and
in the untouched live training/generation jobs. Broader held-out/equal-time
matches are still needed before calling this a generally stronger player.
