# Search-budget policy measurement

Issue [#37](https://github.com/lukekh/alpha-zero-general/issues/37) compares the
existing 200,000-work ceiling with the selected opt-in time-first preset. The
preset uses maximum depth 20, five seconds and a 1,000,000,000-work safety cap.
Existing `SearchConfig`, `core.json`, `all.json`, and `wall-time.json` defaults
remain unchanged for compatibility. Browser users can apply the same values
with **Time first · 5 seconds**, followed by **New game**; a manually entered
lower work cap remains supported.

Zero is deliberately not an unlimited sentinel: zero work or zero time still
returns a legal fallback. The high safety cap makes time the effective limit in
these measured unsolved positions while retaining protection against abnormal
work. This is a budget-policy comparison, not a per-node speedup or a strength
claim.

## Reproduction contract

[`game79.pgn`](game79.pgn) is the exact 79-ply replay from the issue, including
configuration and final-state hash. The pre-change source revision is
`3bdf9333329b1499bc7c35bc625e6a314d0084b3`. Each measured run uses the same
replayed position and configuration except for `node_limit`, creates a fresh
player/transposition table, and warms compiled kernels before timed budget
creation. Reproduce the three repetitions with:

```sh
uv run --locked python intransitive/benchmarks/search_budget/reproduce.py --repeats 3
```

The script writes JSON to standard output. The checked-in
[`evidence/results.json`](evidence/results.json) records Python 3.11.4 on
macOS 15.6.1 arm64, 1.138 seconds of separate warm setup, and a whole-process
peak RSS of 249,249,792 bytes. Search `elapsed` includes result assembly but not
the explicitly warmed preparation.

## Results

Medians from three fresh-table runs on 2026-09-14:

| Position | Policy / work cap | Elapsed | Completed depth | Selected depth | Partial depth | Stop reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Before Red 18 (ply 35) | 200,000 | 0.364 s | 3 | 4 | 4 | work |
| Same position | 1,000,000,000 | 5.004 s | 5 | 5 | 6 | time |
| Before Red 31 (ply 61) | 200,000 | 1.030 s | 3 | 3 | 4 | work |
| Same position | 1,000,000,000 | 5.002 s | 4 | 4 | 5 | time |

All repetitions produced the displayed depths and stop reasons. At ply 35 the
time-first runs retained the completed depth-five result while depth six was
partial. At ply 61 they retained the depth-four unrefuted fallback while depth
five was partial. `diagnostics_status` was `skipped_budget` in every capped run,
independently of whether search stopped for work or time.

The time-first policy used approximately 2.85 million mean work at ply 35 and
1.05 million at ply 61, well below its safety cap. The 200,000-work policy is
useful when deterministic compute is more important than using the allotted
wall time; the time-first policy yields deeper coverage here at higher work and
latency. These two positions do not establish a general playing-strength gain.

Fixed-depth equivalence against exhaustive minimax, exact work interruption,
zero budgets, early proofs, maximum-depth completion, simultaneous-limit
precedence and retained partial roots are covered by `test_anytime_search` and
`test_heuristics`. Browser/config round trips and exported `LastAI` fields are
covered by `test_play` and `test_record`.
