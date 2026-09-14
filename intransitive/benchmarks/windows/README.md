# Principal-variation search and aspiration windows (#41)

Two independent `SearchConfig` comparison switches retain full-width main search
and every legal move: `pvs_enabled` and `aspiration_enabled`. Both default to
false. `aspiration_window` is a finite positive half-width in existing evaluator
units (25 by default). Configuration files work in browser play, terminal play
and the replay analyzer; exported records preserve the switches and counters.
Weights, proof horizon, maximum search depth and legal ordering inputs are held
fixed. There are no reductions or selective pruning.

## Window representation and interrupted searches

The evaluator produces finite binary64 scores. PVS searches the first candidate
normally, then probes alternatives with `(alpha, nextafter(alpha, +inf))`.
There is no representable score strictly inside this interval. A fractional or
one-ULP improvement challenges the incumbent just like any other improvement;
negative zero compares equal to zero, and `nextafter(-0., +inf)` is the smallest
positive subnormal. Infinite endpoints are sentinels, not evaluator scores;
nonfinite alpha uses ordinary search. A probe already spanning the available
window also uses ordinary search. Mate distances retain the existing finite
representation and table normalization.

An improving probe below beta requires a re-search at the same depth with the
full available window. A cutoff at beta remains a lower bound. Root progress
publishes a challenger only after that re-search returns. An interrupted probe
or re-search cannot replace a completed sibling. Exact, lower and upper TT
entries retain their full draw-safe keys and exact-depth identity; deeper
entries only supply move ordering hints.

Aspiration starts after the first completed iteration, centred on its score.
Fail-low expands the lower endpoint; fail-high expands the upper endpoint.
The half-width doubles on recovery, with representable outward progress even
for subnormal widths. After eight failures, the next attempt uses `(-inf, inf)`.
Every attempt uses the original target depth and the same work/deadline budget.
A failed window never completes an iteration or publishes an exact root score.

Completed root siblings survive retries. A later weaker bound cannot overwrite
an exact sibling. On interruption, exact siblings take priority; failed-high
lower bounds and optimistic failed-low upper bounds cannot displace them. If
there is no new verified candidate, the last completed iteration is retained.
An incumbent already proved losing yields to an unrefuted alternative, which
may have only a shallower bound or no score. `completed_depth`, `selected_depth`,
`partial_depth`, `selection_source` and `score_bound` preserve this distinction.
Finishing an exact iteration at the deadline retains it; finishing a failed
window at the deadline does not.

`SearchResult`/record JSON add `pvs_probes`, `pvs_researches`,
`aspiration_researches`, `aspiration_fail_highs` and `aspiration_fail_lows`.
These count started probes/re-searches and observed root recovery events;
interrupted attempts can contribute work without completing a result.

The organization follows the non-PV probing and aspiration-recovery ideas in
[Stockfish search](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp).
Its integer score increment, effective-depth adjustments and chess pruning
constants are not used here.

## Reproduction and baseline

The exact clean baseline is `516b150e967dd6a40cd5824ba2bc982c36f0fe73`, including
merged #38/#39 optimizations and the subsequent 80-turn modelling-draw history
and official-play separation. [Archive](evidence/baseline.tar.gz) and
[SHA-256 manifest](evidence/baseline-manifest.json) preserve its Python source,
configuration, dependency lock, tactical fixtures and exact
[79-ply replay](../search_budget/game79.pgn). The historical replay is loaded with
its original format/hash validation and current supported history representation.
This is the current anytime-search baseline, not the older 33-plane snapshot.

```sh
uv run --locked python intransitive/benchmarks/windows/reproduce.py \
  --repeats 3 --seconds 1 --depth 4 > results.json
uv run --locked python intransitive/benchmarks/windows/verify_pvs.py
uv run --locked python -m unittest intransitive.tests.test_search_windows -v
uv run --locked python intransitive/benchmarks/windows/validate_variants.py
uv run --locked python -m unittest discover -s intransitive/tests -v
uv run --locked python -m unittest discover -s tests -v
```

The benchmark imports archived search/config/budget directly from the checked
archive, verifies the shared implementation hashes, and runs that baseline plus
current full-window, PVS only, aspiration only and both. Fresh players/tables and
rotated order across three repetitions compare depth four and equal one-second
budgets (maximum depth 20, billion-work ceiling). Rule/proof/material compilation
and representative positions are warmed before measurement; setup is separate.
The window code adds no compilation or native setup.

All repeated fixed-depth scores must agree. Current full-window must also match
archived action, PV, node and work counts exactly. Each variant's selected move
is independently re-searched with the full window at the same horizon; every
returned PV is replayed through the independent rules. A [separate full-window verification](evidence/pv-validation.txt) checks every
distinct measured main-PV suffix, including exact partial-iteration selections.
Small-tree tests compare
all variants against exhaustive minimax, including each main-PV suffix. Tied
moves or PVs may differ while remaining valid minimax choices.

[Raw results](evidence/results.json) retain all 120 search runs, configurations,
source/state hashes, work, main/proof nodes, re-search counts, elapsed time,
completed/selected/partial depths, root coverage, bounds, TT/material memory,
module time and independent tactical objectives. Separate `tracemalloc` runs
sample peak/retained Python allocations for each fixed-depth position/variant;
these timings are excluded. Whole-process RSS includes setup and all variants,
so it cannot be attributed to one variant. Neither traced memory nor the table
estimate is a total native allocation count.

## Validation

[Baseline affected suites](evidence/baseline-tests.txt): **176/181 pass**, with
five ordinary defensive failures: original Red 14, 18 and 31, and the simplified
last defender in both colours. No assertions, weights or limits are weakened.

[Window and anytime tests](evidence/window-tests.txt): **27/27 pass**. They exercise fractional/one-ULP ties,
signed zero, infinite endpoints, subnormal aspiration widths, both colours,
forced wins/losses, mate normalization, proof leaves, playable repeated positions,
draw clocks, TT bound recovery, exact-depth reuse, real work caps and deterministic
deadlines. They interrupt probes, full re-searches, fail-high/fail-low recovery,
and the end of exact/failed windows, checking completed-sibling retention and
unrefuted fallback. Generated legal positions extend depth-three minimax/PV
comparisons beyond the fixed fixtures. The final failed-low recovery check also
keeps already visited upper-bound alternatives available when the incumbent is
proved losing.

[Full validation](evidence/full-validation.txt): **342/347 pass**, with exactly
the same five defensive failures. [All three window variants](evidence/variant-tactics.txt)
run the unchanged tactical/blunder assertions: **107/112 pass per variant**,
each with those same five failures. These are ordinary failures, not exemptions;
the commands exit nonzero. [Repository checks](evidence/repository-validation.txt):
**7/7 pass**. The 15 new window tests and 12 existing anytime tests all pass.

## Measurements and recommendation

Three paired repetitions on macOS 15.6.1 arm64 / Python 3.11.4, pinned `uv.lock`.
Times are medians in seconds; each raw repetition remains available. The opening
has stable iteration values; Red 14 and Red 31 trigger fail-high and fail-low
recoveries. This four-position sample is not an estimate of general strength.

| Position | Archived | Current full | PVS | Aspiration | Both |
| --- | ---: | ---: | ---: | ---: | ---: |
| Opening | 0.2722 | 0.2743 | 0.2724 | 0.2709 | 0.2711 |
| Before Red 14 | 0.3690 | 0.3756 | 0.3745 | 0.3787 | 0.3815 |
| Before Red 18 | 0.4200 | 0.3979 | 0.3795 | 0.3243 | 0.3183 |
| Before Red 31 | 0.7191 | 0.7181 | 0.7049 | 0.4882 | 0.4780 |

PVS alone changes median time versus the archived reference by **+0.1%, +1.5%, -9.6%, -2.0%** across these positions.
Aspiration changes median time versus the archived reference by **-0.5%, +2.6%, -22.8%, -32.1%** across these positions.
Combined changes median time versus the archived reference by **-0.4%, +3.4%, -24.2%, -33.5%** across these positions.
Timing variation is visible in the per-run records. Substantial deterministic
work reductions support the large aspiration gains on Red 18/31; small
single-digit time differences should not drive adoption alone.

| Position / variant | Main nodes | Proof nodes | Work | PVS probes / re-searches | Aspiration retries (high / low) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Opening / full | 6,107 | 4,325 | 265,937 | 0 / 0 | 0 (0 / 0) |
| Opening / pvs | 6,035 | 4,253 | 264,065 | 365 / 0 | 0 (0 / 0) |
| Opening / aspiration | 6,035 | 4,253 | 264,065 | 0 / 0 | 0 (0 / 0) |
| Opening / combined | 6,035 | 4,253 | 264,065 | 365 / 0 | 0 (0 / 0) |
| Red 14 / full | 7,742 | 5,096 | 309,264 | 0 / 0 | 0 (0 / 0) |
| Red 14 / pvs | 7,647 | 5,097 | 304,286 | 452 / 2 | 0 (0 / 0) |
| Red 14 / aspiration | 7,793 | 5,054 | 309,863 | 0 / 0 | 3 (1 / 2) |
| Red 14 / combined | 7,796 | 5,055 | 309,965 | 485 / 2 | 3 (1 / 2) |
| Red 18 / full | 10,023 | 8,170 | 319,654 | 0 / 0 | 0 (0 / 0) |
| Red 18 / pvs | 9,371 | 7,495 | 303,370 | 447 / 4 | 0 (0 / 0) |
| Red 18 / aspiration | 7,253 | 5,495 | 260,769 | 0 / 0 | 0 (0 / 0) |
| Red 18 / combined | 7,130 | 5,351 | 255,000 | 303 / 3 | 0 (0 / 0) |
| Red 31 / full | 5,527 | 185,459 | 501,942 | 0 / 0 | 0 (0 / 0) |
| Red 31 / pvs | 5,513 | 178,923 | 489,623 | 439 / 5 | 0 (0 / 0) |
| Red 31 / aspiration | 4,383 | 115,233 | 345,356 | 0 / 0 | 3 (1 / 2) |
| Red 31 / combined | 4,398 | 110,178 | 336,502 | 439 / 4 | 3 (1 / 2) |

The combined Red 14 search uses **701 more work** than full-window: three
aspiration retries and two PVS re-searches erase its potential savings. This
is an explicit reject case for blanket enablement. On Red 31, aspiration alone
saves **31.2% of work** and combined saves **33.0%**. PVS adds only modest savings
beyond aspiration here, and slightly more traced memory.

At equal **one second**, every variant/repetition completes depth **four**.
All select completed branches at partial depth five for opening/Red 14/Red 18;
Red 31 retains selected depth four. Aspiration/combined cover eight root siblings
at Red 18 versus four for current full-window, and two at Red 31 versus zero. Those Red 31
branches do not supply a verified improvement, so the completed depth-four
candidate survives. More root coverage is not another completed iteration.

Independent tactical objectives are unchanged in every fixed-depth and equal-time
run: Red 14 fails, Red 18 passes at depth four, Red 31 fails. This does not repair
the five documented depth-three defensive failures or demonstrate stronger play.

| Position | Peak traced bytes, full | PVS | Aspiration | Both |
| --- | ---: | ---: | ---: | ---: |
| Opening | 1,525,393 | 1,526,138 | 1,526,394 | 1,525,622 |
| Before Red 14 | 2,139,393 | 2,109,086 | 2,144,636 | 2,144,898 |
| Before Red 18 | 2,462,305 | 2,290,935 | 2,265,462 | 2,216,024 |
| Before Red 31 | 1,257,845 | 1,271,966 | 1,160,487 | 1,189,505 |

Table estimates range from 0.85 to 2.03 MB at depth four; the raw JSON reports
each table and material-cache payload. Window counters are constant-sized;
root progress retains at most one row per legal root move across retries.
Whole-process peak RSS was **317,521,920 bytes** (302.8 MiB),
including all variants and setup. Common import/replay/kernel setup took
**10.363 s**, outside move timing. The new Python window logic has no
compilation cost. Maximum observed timeout overshoot was **8.70 ms**
(aspiration, ply 35); every actual work count stays within the shared cap.
These are single-machine samples, not scheduling or speed guarantees.

**Retain both independent switches for controlled experiments; reject blanket
default enablement.** Aspiration is the more promising option for this sample,
with large savings on Red 18/31 and small mixed changes on the controls.
PVS is a smaller experiment, and combining it with aspiration is not uniformly
better. Keep the existing core/time-first presets and full-window reference as
the default until a larger held-out equal-time evaluation supports promotion.
This preserves the measured improvement as an available configuration without
turning throughput observations into a claim of strength.

For example, add the following fields to an existing config and pass its path
via `--ab-config` (player) or `--config` (replay analyzer):

```json
{"pvs_enabled": false, "aspiration_enabled": true, "aspiration_window": 25.0}
```

Set both flags false to diagnose a regression against the full-window path.
