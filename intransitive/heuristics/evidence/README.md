# Acceptance evidence

`summary.json` contains report metadata, budgets, configurations, tuning results
and all 64 aggregate rows in a reviewable form. `comparison.json.gz` is the full
report: it additionally contains all 128 held-out search results and every action,
candidate-move metric, terminal reason and final-state hash for 256 games.

```sh
gzip -dc intransitive/heuristics/evidence/comparison.json.gz | jq . > /tmp/issue34-comparison.json
python -m intransitive.heuristics.benchmark \
  --verify-report intransitive/heuristics/evidence/comparison.json.gz
```

The verifier replays every action through `IntransitiveGame`, checks legality at
each ply, and compares the terminal score, reason and final-state SHA-256. It also
checks the complete mode/configuration/opponent/seed/colour matrix and held-out
search and summary counts. The committed verification result is:

```text
{"games": 256, "positions": 128, "summaries": 64}
```

- `benchmark.log`: progress for the complete bounded run.
- `tests.log`: all 152 Intransitive tests.
- `repository-tests.log`: all seven shared repository tests.
- `browser.log`: headless desktop/mobile opponent, module, colour, move and undo checks.
- `pit.log`: real terminal alpha-beta versus greedy invocation.
