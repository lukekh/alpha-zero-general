# Paired Minimax fitness matches

This opt-in harness implements issue #54. It plays official games between frozen
Python alpha-beta evaluators. It does not optimize weights, change the teacher,
train a network, or use modelling draws as physical outcomes.

## Bounded reproduction

From a fresh checkout on macOS/Linux, use Python 3.11 (the evidence uses 3.11.4):

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r intransitive/tournament/requirements.txt
.venv/bin/python -m intransitive.tournament prepare \
  --output /tmp/fitness-input.json --positions 2 --max-plies 4
.venv/bin/python -m intransitive.tournament run \
  --manifest /tmp/fitness-input.json --output /tmp/fitness-run --workers 1
.venv/bin/python -m intransitive.tournament verify \
  --manifest /tmp/fitness-input.json --output /tmp/fitness-run
```

This schedules exactly **16 games**: one population candidate versus the default
incumbent and one frozen archive opponent, two starting positions, two colours,
and two protocols. Each game has at most four plies, 30 seconds of active play,
and 120 seconds per child startup. The archive reproduces the evaluation
parameters of `heuristics/configs/pressure-experimental.json` at revision
`225dc40` (9x9 pressure, weight 10). Its search limits are replaced by the same
declared limits as all other candidates. The population example changes
advantage from 25 to 50; this is an example configuration, not a tuned winner.

Run the same `run` command to resume. SIGINT/SIGTERM cancels active games and
reaps child processes. Completed games, including recorded failures, are never
repeated in that output. Interrupted games resume from their last fsync'd move;
an in-flight move may be recomputed, but is never appended twice. Use a new run
to retry a terminal failure. An exclusive output lock rejects simultaneous
writers. Changing code, runtime versions, candidates, corpus, settings or quotas
requires a new manifest and output directory.

The defaults deliberately produce mostly **unfinished** games. They test the
machinery and do not establish playing strength. Longer experiments require
explicitly larger `--max-plies`, `--game-seconds`, `--positions` and, if wanted,
`--depth`. No tuning loop or automatic follow-on tournament starts.

## Candidate and protocol inputs

`prepare --candidates candidates.json` accepts a JSON list, for example:

```json
[
  {"name": "candidate-A", "weights": {"advantage_weight": 50}},
  {"name": "candidate-B", "weights": {"pressure_weight": 2}},
  {"name": "default", "role": "incumbent"},
  {"name": "pressure-archive", "role": "archive", "weights": {"pressure_weight": 10}}
]
```

Every population pair is played, plus every population member against every
incumbent/archive member. Reference-versus-reference games are omitted. A
candidate also accepts `genome` instead of `weights`: the complete versioned
JSON record from #53 (`intransitive-module-scales-v1`, `backend: python`, and all
five genes `advantage`, `attack`, `defence`, `overload`, `pressure`). The reader
uses the shared `heuristics.tuning.Genome` validator and `to_config()` mapping
from #53 directly, so bounds and enable flags have a single implementation:

- Material is fixed at 100. Advantage/attack/defence/overload are in [0,100];
  pressure is in [0,20]. Zero disables an optional module while retaining its
  dormant legacy coefficient. Default evaluation equals `SearchConfig()`.
- Race/proof/mate scores, geometry, internal bonuses, depth and budgets cannot
  be candidate genes. Unknown fields, NaN, infinity, unsupported versions and
  Rust candidates fail before games start. Native arbitrary-weight parity is
  not claimed.
- The manifest freezes the full mapped evaluation, genome, evaluator version,
  implementation/runtime digest, and effective shared search settings. Names
  and role labels cannot disguise duplicate deterministic candidates.

`spec.py` exposes the same functions for #55/#56 callers. `protocol()` creates
common settings. Other existing search switches can be set in its `search`
dictionary **before** calling `manifest()`. All candidates receive those exact
settings, confirmed by a child handshake. Raw manifest edits are rejected.

The `depth` protocol requires a completed iteration at the requested depth;
an exact proven result may finish earlier and is recorded as `proven_result`.
A budget stop becomes `depth_incomplete`, never a result on the completed-depth
leaderboard. `--depth-seconds` is a safety ceiling (default 10 seconds), not the
target of this comparison. The `wall` protocol uses equal search time per move
(`--seconds`, default 0.05) and a maximum depth of 64. Stopped/partial searches,
selected depths, work, latency and stop reasons remain in the records/reports.
The two leaderboards are always separate. Wall-time trajectories depend on OS
scheduling and load; the fixed-depth protocol is the reproducibility check.

## Positions and independence

The corpus includes the official opening and bounded capture-biased legal
playouts. Each generated line contributes up to one opening (after eight plies),
midgame, and endgame (at most 12 pieces). Generation may finish before finding
all stages for a particular seed; it never creates an endgame by deleting pieces.
The manifest stores the full opening action sequence, exact binary state,
history, side to move, corner ownership, generation seed, stage and hashes.
Preparation replays every opening and validates every stored byte.

Each line is assigned to search/validation/heldout by a stable hash **before**
selection. Positions on the same line stay together. The minimum exact-state
hash over all 12 symmetries identifies each family; duplicate families, including
cross-pool duplicates, are rejected. The official opening belongs to search.
`--pool` selects one pool while the manifest retains the complete frozen corpus.
Positions are sorted deterministically, with the official opening first. An
unavailable exact position quota is an error. Both colours use the identical
physical start and seed; only bot assignment changes.
Recorded colour `0` is Blue and `1` is Red.

The parent uses official transitions. Before each search it creates the engine's
existing modelling observation and canonical current-player perspective. This
may rebase a *search copy* after a modelling repetition/no-capture cutoff; the
physical history and replay record stay intact. Canonicalization never changes
action coordinates. Move legality is checked again against the physical state.

## Outcomes, ranking and uncertainty

Only a rules-engine corner/stalemate win is a win or loss. Safety limits produce
`unfinished`. Crashes, illegal moves, child timeouts and incomplete depth searches
have separate statuses and responsible colours. Cancellation is resumable and
does not count as a game loss. There is no evaluator adjudication or artificial
draw score.

The ranking policy is frozen before execution: lower win points are wins divided
by **scheduled** games; unresolved games contribute zero. The upper bound counts
all unresolved games as possible wins. Eligibility additionally requires every
scheduled game attempted, no failures, and at least 80% official completion by
default. Ranking uses eligibility, then the lower bound, then a stable identity
tie-break. Turning a loss into a cycle cannot increase the ranking score and can
only hurt completion eligibility. Upper bounds are uncertainty, not fitness.
Reference entries are included for context. Evolutionary callers select only
`role: population`; references face a different opponent slate.

Reports include each opponent and colour, W/L/unfinished, failures, pending
games, completion, completed/selected depths, stopped searches, latency, work,
search/startup CPU and wall time, and process peak memory. Complete JSON records
include all moves and state hashes.

A protocol may declare the opt-in selective techniques — `nmp_enabled`,
`futility_enabled`, `lmr_enabled`, `quiescence_enabled`, `mvv_lva_enabled`, with
`pvs_enabled` and the margins that govern them — and every one of them is part of
the frozen protocol record and therefore of each candidate's search identity.
`manifest` builds every candidate's effective configuration at freeze time, so a
protocol that declares a technique those evolved scales cannot support fails
before a single game is played, naming the candidate and
`selective_evaluator_enabled`.

`prepare --probe` answers the same question before a run rather than after it.
`activation` reports what the configuration forbids outright; it cannot report
the other half, which is a protocol that declares a technique with no blocker at
all and still never fires it because no search under its time limit completes an
iteration deep enough. The probe searches each selected position once per
protocol and says so: a wall protocol at 0.05 seconds a move reports `nmp needs
a completed depth of 4 and the deepest probe reached 2`. It distinguishes a
configuration blocker, a shortfall on every probe position, and a shortfall on
only some, and it reports rather than rejects — a technique that fires unevenly
is a real protocol, just a less informative one than it looks.

`report.json` then carries `selective_firing` per protocol: the totalled
counters, which declared techniques actually fired, which stayed silent, and any
precondition that made a declared technique unreachable. A declared technique
that never fires across the whole run is listed in the report's top-level
`warnings`, because a parameter governing code that executes zero times cannot
be tuned and must not be mistaken for a measured one (issue #66).

Known child CPU covers worker warmup and searches; interpreter imports and work lost before a response are unavailable.
Parent wall time includes waiting, and the benchmark also measures total CLI
wall time including parent initialization and input validation.

Colour pairs form one observation. Multiple starts from one generated line and
multiple opponents remain clustered by generation seed. A weighted Hoeffding
interval uses those clusters, includes unfinished-outcome uncertainty, and is
conditional on the frozen corpus. It is not a calibrated claim about all legal
positions. Reports separately count distinct starts, trajectories and duplicate
trajectories; repeating a deterministic game under another seed is not a way to
increase the sample size. Duplicate exact/symmetric starts and duplicate
candidates are rejected before scheduling.

## Process isolation and measured concurrency

`--workers` bounds concurrent games (1–8), each with two independent spawned bot
processes. Both bots run single-threaded Numba. Compilation/warmup precedes the
move clock and is reported separately. Each move gets fresh mutable search and
evaluation caches, including on resume; compiled machine code may use Numba's
on-disk cache. Match IDs and exact quotas never depend on completion order.

Measure your machine before increasing workers:

```sh
.venv/bin/python -m intransitive.benchmarks.tournament.reproduce \
  --output /tmp/fitness-concurrency
```

This runs the same 16-game smoke with one worker and then two, samples aggregate
process-tree RSS every 200 ms, reports throughput/CPU/memory, verifies every
record, and asserts equal fixed-depth trajectories. See
`../benchmarks/tournament/README.md` for the delivered measurements. One worker
is the conservative default; CPU count alone is not a throughput measurement.

Validation:

```sh
.venv/bin/python -m unittest intransitive.tests.test_tournament -v
```

## Signed scales and heuristic MCTS

The current v2 genome includes signed material and module scales in [−100,100].
Python and Rust defaults adopt the measured endgame candidate by explicit user
request. Historical archives require their original source revisions.

The `mcts` protocol freezes `simulations`, `cpuct` and `value_scale` and uses the
repository's PUCT implementation with uniform legal priors and heuristic leaf
values. It records simulation completion and observed tree depth separately;
incomplete simulation counts are `simulation_incomplete` failures. Source/cache
identity includes MCTS.py. See [the tuning contract](../heuristics/TUNING.md).
