# Minimax audit — 2026-09-17

The Python and Rust search code correctly implement the alternating objective in the cases checked. Both use negamax: evaluate for the side to move, recurse with `(-beta, -alpha)`, negate the returned value, and maximize it. Maximizing the negative of the opponent's best result is minimizing that opponent's result. No reversed-player or missing-negation defect was found. This is a bounded audit, not a proof for every position or configuration.

The weak moves are reproducible. Five existing tactical regression tests fail even with completed depth-3 searches; the other 95 tests in the selected suite pass. All nine native Rust unit tests pass. Production code and settings were left unchanged, including pre-existing workspace edits.

## Findings

1. **High impact: default leaf evaluation cannot distinguish many positional choices.** `heuristics/config.py:28` disables attack, defence, overload, and pressure. The ordinary leaf path in `heuristics/evaluation.py:257` consequently scores piece counts and type matchups, without a positional term. Clear-run credit requires a terminal proof. All 36 legal moves from the initial position have static score zero with default settings and proof explicitly disabled for this measurement. Equal search scores retain the first searched move (`heuristics/search.py:443`), so ordering becomes the deciding factor. This is a playing-strength limitation rather than an incorrect minimax recurrence.

2. **High impact: the default horizon misses demonstrated threats.** `SearchConfig` caps search at three plies, with a separately bounded two-ply terminal proof of at most 64 nodes per leaf. Three plies means own move, opponent reply, own move—not three full turns. Proof extensions do not guarantee a complete five-ply search: they may exhaust their allowance and do not prove material gains. Existing failures are listed below. In the simplified last-defender position, exhaustive max/min and production Python both return `348.4375` at depth 3, selecting different tied moves. Rust also chooses the bad `G7-F6` at depth 3; at depths 4, 5, and 6 it chooses the certified defence `F8-G8`, with pressure weight zero. See `horizon.log`.

3. **Medium impact: configured depth and node limits can misrepresent practical strength.** Time/work interruption can stop well before maximum depth. In an opening measurement, default Python completed depth 3 in about 0.060 seconds and stopped at its depth cap; the existing time-first preset targeted depth 20 but completed depth 5 before its five-second deadline. Python's `node_limit` actually limits work including ordering, evaluation, and proof (`heuristics/budget.py`), whereas the Rust browser adapter passes a node-visit limit. Identical numeric settings therefore do not provide equivalent search budgets. Rust retains the last completed iteration on interruption; Python can also select completed branches from a partial iteration. These are intentional policies, but the Rust policy can retain a shallow move after discovering adverse information in an unfinished iteration. That latter risk was identified by inspection, not reproduced as an additional failure here.

## Reproduced tactical failures

All five searches finished depth 3 without exhausting their test budgets (30 seconds and 2,000,000 work units).

| Existing regression | Selected move | Problem |
| --- | --- | --- |
| `red_14_preserve_material` | `E6-F6` | Allows the capture/recapture material loss |
| `red_18_prevent_paper_fork` | `G7-F6` | Allows the paper fork |
| `red_31_keep_goal_defender` | `F8-E8` | Abandons the last timely goal defender |
| `unique_last_defender_original_colours` | `G7-F6` | Misses the unique defence `F8-G8` |
| `unique_last_defender_exchanged_colours` | `C3-D4` | Misses the colour-exchanged defence `B4-B3` |

The fixtures check good and bad controls using the independent tactical oracle before checking the engine. Full assertions, scores, and principal variations are in `python-tests.log`.

## Validation

Reviewed side changes, terminal-score perspective, leaf-score differences, alpha/beta sign changes, PVS re-search, depth-specific transposition entries, mate-distance normalization, root selection on interruption, and browser configuration forwarding. Existing tests exercise exhaustive max/min agreement, cache and window behavior, compact-state restoration, both colours, and Python/Rust parity, including 100 certified tactical positions within one test method.

```sh
.venv/bin/python -m unittest \
  intransitive.tests.test_heuristics \
  intransitive.tests.test_search_windows \
  intransitive.tests.test_anytime_search \
  intransitive.tests.test_compact_search \
  intransitive.tests.test_search_optimizations \
  intransitive.tests.test_game_blunders \
  intransitive.tests.test_rust_teacher \
  intransitive.tests.test_rust_browser -q

cargo test --manifest-path intransitive/rust_teacher/Cargo.toml --offline -q
```

Python: 100 tests, five failures, no skips reported. Rust: nine tests, all passed. Python/Rust integration tests used the existing release binary; Cargo tests compiled the current native source. Runtime measurements are illustrative local observations, not controlled performance benchmarks. The exhaustive reference shares the production leaf evaluator/proof, so its agreement validates search selection relative to that evaluator, not the evaluator's strategic quality.

## Recommended next changes

Use the five failing tactical tests as acceptance criteria. First evaluate greater completed depth and carefully bounded extensions through forcing captures and goal threats. Then measure positional evaluation changes against the same tests and fixed-time matches; simply enabling every expensive heuristic may reduce depth and make play worse. Report completed/selected depth and stop reason when diagnosing a specific game. Do not rewrite the max/min recurrence on the evidence from this audit.
