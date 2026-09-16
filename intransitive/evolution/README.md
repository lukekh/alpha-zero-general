# Evolution of Minimax module scales (#55)

This opt-in optimizer uses the [v1 genome](../heuristics/TUNING.md) and
[paired official-game harness](../tournament/README.md). Python's five effective
module coefficients are the only genes. Material stays at 100; module signs,
zero/enable semantics, clipping, decisive scores, proof settings, search depth,
time limits, official rules, MCTS and training configuration are fixed. Rust is
not supported by the match harness. Exported candidates are **unaccepted** and
must go through separate held-out acceptance in #56. No defaults are updated.

## Reproduce a bounded run

Use the harness's minimal environment (Python 3.11):

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r intransitive/tournament/requirements.txt
.venv/bin/python -m intransitive.evolution smoke --output /tmp/evolution-smoke
.venv/bin/python -m intransitive.evolution prepare \
  --config intransitive/evolution/smoke-benchmark.json \
  --output /tmp/evolution-input.json
.venv/bin/python -m intransitive.evolution run \
  --manifest /tmp/evolution-input.json --smoke /tmp/evolution-smoke \
  --output /tmp/evolution-run
```

The smoke has two candidates, one generation, a common search start and one
fresh validation start: at most 16 games, 3 million reserved work units and 180
seconds total wall time, including input validation, preflights and child
startup. Each game has at most two plies. These very short games test machinery;
unfinished games cannot establish strength. Operational smoke failures require
diagnosis, not automatically increasing limits. Work/depth failures are expected
candidate rejection data; they are retained and cannot qualify a candidate.
`smoke --mode wall` exercises the separate
equal-time protocol. A run beyond the smoke envelope requires a successful smoke
receipt for the same code, backend/runtime and protocol mode, including through
the Python `run()` API.

The example benchmark configuration runs two generations. `prepare` starts no
engines: it freezes the exact complete legal corpus, splits, candidate bounds,
fixed incumbent/archive, settings, revision, optimizer source digest and
evaluator/rules/runtime digest. For a different experiment, copy the JSON and
change its explicit settings. An insufficient quota of distinct validation lines
is an error; increase corpus generation limits before freezing the experiment.
Held-out positions are recorded to establish the split but never evaluated.

Repeat the same command/output to resume. SIGINT/SIGTERM stops active children
through the harness, retains their fsync'd move journals and writes the optimizer
checkpoint. Only one run can write an output directory. Changing any frozen
setting, source digest or runtime requires a new manifest and output directory.
Preparation is outside the run budget; run-time validation is charged to it.

## Published algorithm and fitness

`strategy.Settings` serializes every optimizer control into the manifest:

| Setting | Default | Meaning |
| --- | ---: | --- |
| seed | 55 | One local `random.Random` stream for initialization, selection and variation |
| population / generations | 4 / 3 | Unique population size and generation ceiling |
| elites / tournament_size | 1 / 2 | Retained top candidates; select the best of a uniform sample without replacement |
| mutation_rate / log_sigma | .5 / .5 | Independent per-gene mutation probability; normal standard deviation in natural log space |
| toggle_rate | .15 | Conditional on mutation, turn a positive gene off or reactivate zero uniformly within its bounds |
| recombination_rate | .5 | Uniform per-gene crossover between two selected parents before mutation |
| random_off_rate | .5 | Atom at zero for random initialization/immigrants; otherwise uniform within bounds |
| near_default_fraction | .5 | Initial fraction obtained by mutating the unchanged default genome |
| hall_size | 2 | Maximum additional historical winners; fixed pressure-10 archive remains present |
| search_positions / validation_positions | 2 / 1 | Common search starts; fresh validation lines per generation |
| min_completed_depth | 1 | Below this, a non-proven move makes the candidate ineligible, including in wall mode |
| resident_engines | 4 | Bounded LRU cache of isolated candidate processes (2–8); only one game/two bots are active |
| max_games / max_nodes / max_seconds | 200 / 100000000 / 300 | Explicit per-run attempt/work/wall ceilings |

Positive mutation is multiplicative/log-normal and clipped to legal bounds.
Zero has its own probability mass, so disabled modules can activate. Near-default
initialization and breeding fall back to random immigrants after 100 proposals;
10,000 proposals without a full unique population fail. Incumbent and fixed
archive duplicates are excluded from the population; duplicate elites/children
are rejected. Every proposal goes through `Genome`, including the final bounds
check after floating-point exponentiation. Bounds come directly from the shared
contract and are frozen in the manifest. No static evaluator score is fitness.

Every population pair plays both colours on each common search start. Every
population member also faces the unchanged default, a fixed pressure-10 archive,
and each distinct historical hall entry. An identical historical/population
genome occupies one identity and never plays itself. Names and roles do not
create additional evidence. The exact opponent matrix is retained in each
generation's report. The initial fixed archive provides historical opposition
even before a validation-eligible winner exists.

Ranking follows the harness: eligibility, then official wins divided by
scheduled games, then stable config identity. Unfinished games, crashes,
illegal moves, infrastructure timeouts and incomplete depth searches earn no
win points. The harness's completion threshold (default .8), failure exclusion
and all-games-attempted rule apply. The optimizer additionally rejects measured
search-fixture saturation, failed preflights and low completed search depths.
Proof-terminated exact results are exempt from the minimum-depth check. The
preflight uses the run's common search positions, never validation or held-out
positions, and is cached once per candidate within the immutable run.

The apparent winner is re-evaluated each generation on fresh validation starts
from previously unused generation-line seeds. The winner and unchanged default
face each other plus the same frozen archive/history slate. Training and
validation eligibility are both required before the winner enters the FIFO hall.
Hall membership is historical opposition, not a claim of improvement over the
default. Winning one parent matchup cannot replace the population/opponent
mixture or validation. No significance claim or automatic adoption is made.
The repeatedly inspected validation pool is development data, not held-out data.

## Resume, caching and budgets

`checkpoint.json` atomically stores the population, complete RNG state (including
the Gaussian cache), phase, generation, match ledger, fitness component reports,
preflights, historical hall, candidate exports and resource accounting. Each
match has a canonical two-candidate manifest and the harness's complete move
journal. The checkpoint ledger points to the journal and records its digest.
A process death between final journal fsync and checkpoint commit is reconciled
by replaying that journal, not repeating the game.

The cache key covers complete effective candidate and opponent identities,
backend/rules/runtime version, exact position and history, opening actions,
seed, colour assignment, protocol and all search/safety/eligibility settings.
Canonical pair manifests remove changing population roles and task indices from
the evidence identity. Cache hits can inform another generation's ranking but
are never added to unique match/trajectory counts or pooled as independent
samples. Reports retain colour-paired, opening-line-clustered uncertainty from
the harness. Convergence points share search evidence and are correlated.

There is one active game; only one bot searches at a time. Up to
`resident_engines` isolated candidate processes stay resident, avoiding repeated
Numba compilation. Python/NumPy seeds reset and the candidate/config handshake
is verified at the start of every game. Each move still constructs a fresh search
engine, evaluator and transposition table through the unchanged harness worker.
Only compiled code and immutable configuration survive a game. Cache identity
includes candidate/backend and complete protocol. Inactive engines use LRU
eviction, engines outside the next opponent slate are removed, and failed,
cancelled and completed runs reap all children. This does not increase concurrent
searches. Report resident limits and process RSS; do not infer a higher worker
count recommendation from CPU count. The work budget counts the Python search's logical
`Budget.work`, including evaluation/proof/order work, rather than only tree
nodes. Before each attempt the optimizer durably reserves
`2 * 10000 + max_plies * node_limit`, including both child warmups. Preflight
reserves `search_positions * node_limit`. Warmup is conservatively reserved even
on a process-cache hit. Reservations are deliberately not
refunded: interrupted attempts and killed/unreported work cannot reset a limit.
Thus `max_games` limits attempts, while unique matches count evidence. A cached
final game costs no new reservation. Reports include both reserved work and
measured search work/known CPU, which exclude unreported work.

A timer enforces remaining cumulative wall time through cooperative cancellation;
child termination/reaping and final fsync may take a short cleanup interval.
After a hard crash, an unreleased wall lease conservatively charges elapsed time
including downtime. Clean cancellation preserves remaining time. Exact resume
means the optimizer state, proposals, schedule, committed games and deterministic
completed-depth trajectories; wall-limited searches remain dependent on timing
and host load, as in the underlying harness. Budget exhaustion retains partial
journals and exports but cannot finish another generation without a new run.

## Equal-budget control and exports

After a successful smoke, on resources available for experiments:

```sh
.venv/bin/python -m intransitive.evolution benchmark \
  --manifest /tmp/evolution-input.json --smoke /tmp/evolution-smoke \
  --seeds 55 56 --output /tmp/evolution-benchmark
```

This runs evolution and random search serially for each distinct seed, with
identical initial populations, frozen corpus, opponent/validation policy, search
settings, and game/work/wall **ceilings**. The random control draws fresh uniform
plus zero-atom populations after the shared initialization; it does not select
parents or retain elites. Both controls retain their own validation-eligible
history. Realized resource consumption may differ because cache hits, early
termination and hall membership differ; the comparison publishes that difference
and does not describe it as equal measured compute. Convergence, unique
candidates/matches, diversity, uncertainty, failures, W/L/unfinished, work, CPU,
wall time and default validation results are retained per optimizer seed. Each
protocol has its own manifest and benchmark; depth and wall scores never mix.
The benchmark contract publishes both per-run caps and number of runs, bounding
the total allocation by their product. Run it explicitly; implementation of this
issue does not authorize a day-long experiment or competing with live training.

`candidates.json` contains each generation's apparent winner, effective config,
genome/config manifests, provenance, search and fresh validation components,
eligibility and validation match keys. Even ineligible winners are exported as
diagnostics and prominently marked unaccepted. `report.json` contains convergence
and unique compute/evidence totals; `comparison.json` contains the multi-seed
control comparison. These files are inputs for review and #56, never application
configuration, generator or trainer updates.

See [measured smoke and control results](../benchmarks/evolution/README.md) for
the bounded two-seed experiment and its limitations. Audit saved records without
starting any engines:

```sh
.venv/bin/python -m intransitive.benchmarks.evolution.verify /tmp/evolution-run
```

Tests:

```sh
.venv/bin/python -m unittest intransitive.tests.test_evolution \
  intransitive.tests.test_tournament intransitive.tests.test_tuning -v
```
