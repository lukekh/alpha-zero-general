# Held-out acceptance for evolved heuristic scales (#56)

This package decides whether an evolved configuration should be adopted. It
consumes the [module-scale contract](../heuristics/TUNING.md) from #53, the
[paired match harness](../tournament/README.md) from #54, the
[optimizer exports](../evolution/README.md) from #55, and the
[certified tactical fixtures](../heuristics/TACTICS.md) and
[game regressions](../heuristics/GAME_REGRESSIONS.md) from #52.

It never writes an application default, the generator, the teacher or existing
training labels. Its output is a decision, a reproduction report, and an opt-in
preset a person may choose to load. **"Retain existing defaults" is a valid
outcome and is the declared default one.**

The configuration currently shipped as the Python and Rust default is itself an
evolved candidate, [adopted in 9424a35 at the user's explicit request](../heuristics/TUNING.md)
without held-out acceptance. This protocol therefore treats it as a candidate
like any other, with the weights it replaced as its comparison, and can
recommend reverting to those weights.

## What is frozen before anything runs

`prepare` writes one `plan.json` containing every input and every rule. A run
rebuilds the plan from its own record and refuses to continue if the corpus,
the entrants, the design or the thresholds have moved.

| Frozen | Content |
| --- | --- |
| Corpus | The complete generated corpus, split into search/validation/held-out **by generated line** before any selection, with symmetry families deduplicated across the whole corpus |
| Starts | The held-out starts, at most one per line per stage, with exact state, history, opening actions and hashes |
| Entrants | Every candidate, its ablations, the prior incumbent and the fixed pressure-10 archive, each as a #53 genome with its effective evaluation and config hash |
| Protocols | Fixed completed depth and equal move time, with identical safety limits, on separate leaderboards |
| Thresholds | The gain, regression, eligibility, tactical-safety and cost rules below |
| Provenance | Repository revision, this package's source digest, the evaluator/rules/harness digest and runtime versions |
| Detectable effect | The Hoeffding radius each schedule can produce, published with the plan rather than discovered afterwards |

Held-out starts are played **once**, by the matches. Saturation preflights,
determinism checks, native parity and the cost ladder all use search-pool
positions: they ask questions about an evaluator, not about a position, so they
cost the held-out set nothing. A set reused to choose between candidates has
stopped being held out.

## Predeclared decision rule

| Threshold | Default | Meaning |
| --- | ---: | --- |
| `practical_gain` | .10 | Paired difference of the harness's conservative lower win-point bound that a candidate must clear against its declared comparison |
| `practical_regression` | .10 | The mirror image; an established loss of this size by the current default recommends reverting |
| `require_disjoint_intervals` | true | Adoption also needs the two paired intervals to be disjoint. A point margin is an observation; separation is the claim |
| `min_completion` | .8 | The harness's eligibility rule: every scheduled game attempted, no failures, and this share decided by the rules engine |
| `max_new_tactical_failures` | 0 | A candidate may not newly fail any independently certified fixture |
| `max_proven_win_failures` | 0 | It may never fail one whose certificate is a proof |
| `max_median_depth_loss` | 1 | Median completed depth at equal move time, against the comparison |
| `min_label_throughput_ratio` | .2 | Share of the comparison's measured labels per hour required before a teacher proposal is even considered |

`adopt` requires all of them, on every protocol the candidate played, with
correctness and parity passing. An established loss by the current default is
`revert-recommended`. Anything else is `retain-defaults`, or `inconclusive`
when a protocol is ineligible, incomplete or unsafe. The rule is applied by
[report.py](report.py) to the recorded evidence, and `decide` re-applies it
without starting an engine.

Each candidate declares the configuration it must displace: the shipped default
is measured against the weights it replaced, and everything else is measured
against the shipped default. No candidate is measured only against an opponent
it was evolved beside.

## Evidence each run produces

- **Held-out matches.** Paired colour-swapped games from the same held-out
  starts, at fixed completed depth and at equal move time, against the prior
  incumbent and the fixed archive. W/L/unfinished, completion, distinct starts
  and trajectories, achieved depths, stopped searches, latency, work, CPU and
  memory all come from the unchanged #54 harness, as does the clustered
  interval. Unfinished is never an invented draw.
- **Attribution.** One experiment per changed module, restoring that module's
  baseline coefficient alone against the candidate itself on held-out starts at
  fixed depth. A module that can be restored without cost did not produce the
  result. Attribution is scheduled for the configuration installed as the
  default; another candidate earns one by first clearing the gain threshold.
- **Tactical safety.** All 112 certified fixtures for every entrant, at two
  profiles: the committed suites' generous budget, and the shipped default
  budget. Results separate proofs from bounded obligations, and record which
  searches ran out of budget rather than choosing badly.
- **Correctness and parity.** Genome round-trip, evaluation identity, repeated
  searches, input immutability, #53's saturation preflight, and a Python/Rust
  score comparison for genomes the native backend supports.
- **Practical cost.** Static evaluation seconds and work, a depth ladder at
  depths 3–6, achieved depth at equal move time, the depth actually reached
  under the shipped default limits, teacher labels per hour at the teacher's
  own depth, startup time and peak RSS — each candidate profiled in its own
  spawned process.

## Reproduce

```sh
uv run --locked python -m intransitive.validation smoke \
  --output /tmp/acceptance-smoke --revision "$(git rev-parse --short HEAD)"
uv run --locked python -m intransitive.validation prepare \
  --output /tmp/acceptance-plan.json --revision "$(git rev-parse --short HEAD)"
nice -n 19 uv run --locked python -m intransitive.validation run \
  --plan /tmp/acceptance-plan.json --output /tmp/acceptance-run --workers 1
uv run --locked python -m intransitive.validation verify \
  --plan /tmp/acceptance-plan.json --output /tmp/acceptance-run
```

The smoke uses the same code and the same stages at depth one with two-ply
games and four tactical fixtures. It tests the machinery; two-ply games cannot
establish strength. Re-run `run` with the same plan and output to resume:
completed games, tactical profiles, parity checks and cost profiles are never
repeated, and SIGINT/SIGTERM stops the run through the harness with its
journals fsync'd. `--max-seconds` adds a cooperative wall budget, `--stages`
runs part of a plan, and `--skip-native` drops the Rust comparison when the
binary is not built. `verify` replays every recorded game through the rules
engine without starting an engine.

Exporting a preset, after a run has produced a decision:

```sh
uv run --locked python -m intransitive.validation preset \
  --plan /tmp/acceptance-plan.json --output /tmp/acceptance-run \
  --candidate adopted-hour-search \
  --preset-output intransitive/heuristics/configs/evolved-hour-search.json
```

The preset is a plain `SearchConfig` record, loadable with
`SearchConfig.from_file` or `intransitive --ab-config`. It carries a work and
time limit **measured** to complete its own depth for that evaluation, because
the shipped pair of a depth and a work cap was sized for a much cheaper
evaluator. The sidecar record states the acceptance outcome, so a preset can
never be mistaken for an accepted replacement.

Tests:

```sh
uv run --locked python -m unittest intransitive.tests.test_validation -v
```

## Boundaries

- No default, generator, trainer or checkpoint gate is changed here. Adoption
  is a separate, explicit decision by a person, informed by this report.
- A teacher rollout additionally requires that new labels record the
  evaluator/config identity that produced them, that the dataset provenance
  boundary is written down, and that the combination rule for old and new
  labels is stated. Existing labels are never relabelled silently.
- A checkpoint-opponent change additionally requires re-evaluating the
  incumbent and current models under the same new suite before comparing
  scores, with backups and a rollback path. The official termination rules and
  the existing checkpoint gate stand unless separately agreed.
- The intervals are conditional on the frozen corpus. Repeated deterministic
  trajectories are not independent evidence, and the plan publishes the
  detectable effect so a null result is read as a null result and not as proof
  of equivalence.

See the [measured acceptance run](../benchmarks/validation/README.md) for the
delivered evidence and its limitations.
