# Issue #54: bounded paired tournament evidence

The [harness guide](../../tournament/README.md) documents inputs, official outcome
semantics, ranking, process isolation, resume and the serialized #53 genome
contract. This benchmark checks the harness; it is not a strength experiment.

## Reproduce

Use Python 3.11.4 and `../../tournament/requirements.txt` from the repository root:

```sh
.venv/bin/python -m intransitive.benchmarks.tournament.reproduce \
  --output /tmp/intransitive-fitness-benchmark
```

Preparation generates 28 legally replayable positions across search (13),
validation (9), and heldout (6), including all four stages: official opening,
generated opening, midgame and endgame. The bounded smoke selects the official
opening and a generated opening from search. It schedules 16 games per run:
material-50 versus the default and the archived 9x9 pressure-10 configuration,
both colours, two positions, completed depth 1 and 50 ms search protocols.
Each game is capped at four plies and 30 active seconds. Startup has a separate
120-second limit per candidate process.

## Measured results

Measured on macOS 15.6.1 arm64, Python 3.11.4, NumPy 2.3.5, Numba 0.65.1 and
llvmlite 0.47.0. Runs were sequential, one trial per worker count; other machine
load and warm filesystem caches were not controlled.

| Concurrent games | CLI wall seconds | Games/second | Sampled process-tree peak RSS | Known child CPU seconds |
| --- | ---: | ---: | ---: | ---: |
| 1 | 703.25 | 0.02275 | 367.2 MiB | 489.13 |
| 2 | 596.91 | 0.02680 | 518.7 MiB | 558.75 |

Two workers delivered only **1.18x throughput**, with **1.41x sampled peak RSS**.
One worker remains the conservative default. This single, startup-dominated
trial does not establish the optimum for longer games or another machine.
RSS is sampled every 200 ms and sums the tournament parent and descendants;
shared pages may be counted more than once. Known CPU includes worker warmup
and reported searches, excluding interpreter imports and unreported work.
The end-to-end CLI clock includes initialization, compilation and validation.

All **32 records replayed legally**, with **zero crashes, illegal moves,
infrastructure timeouts or incomplete-depth failures**. Every fixed-depth game
had the same trajectory with either worker count. The serial run contains 10
distinct trajectories and six duplicates across its 16 games; those repeated
paths are explicitly reported, not additional independent observations.

All games reached the four-ply safety cap: **0 wins, 0 losses, 32 unfinished**,
0% official completion. Every candidate is ineligible for strength ranking,
with conservative score bounds [0,1]. No artificial draws or evaluator-based
adjudications were used. Unit tests separately exercise official Blue/Red wins,
including a win on the safety-limit ply, and cycling past modelling cutoffs.

The measured serial run spent 688.83 seconds waiting for candidate startup and
only 3.20 seconds in active game play. This exposes the cost of fresh spawned
processes and compilation per game. Fresh mutable caches per move make partial
game resume independent of previous search state. A future persistent-process
optimization would need to retain that isolation and repeat these comparisons.

## Evidence and checks

- `evidence/measurements.json`: throughput, sampled memory, CPU and environment.
- `evidence/tournament-smoke.tar.gz`: immutable input, all 32 complete game
  journals, reports by opponent/colour/protocol, environment and invocation logs.
- `evidence/audit.json`: configuration routing, depth equality, complete quotas
  and replay/resume checks.
- `evidence/tests.log`: 96 passing focused tournament, rule, replay, heuristic and
  search regression tests, including 16 tournament-specific tests.
- `evidence/SHA256SUMS`: checksums of delivered artifacts.

The completed serial run was resumed with the same manifest: all 16 match files
remained byte-for-byte identical, with no new matches. Unit tests also interrupt
a partially played game, resume it and compare its trajectory with a fresh game,
and prove crashed, hung and cancelled child processes are reaped.

Extract the archive and verify records using the same source and runtime:

```sh
mkdir -p /tmp/fitness-evidence
tar -xzf intransitive/benchmarks/tournament/evidence/tournament-smoke.tar.gz \
  -C /tmp/fitness-evidence
.venv/bin/python -m intransitive.tournament verify \
  --manifest /tmp/fitness-evidence/input.json \
  --output /tmp/fitness-evidence/workers-1
.venv/bin/python -m intransitive.tournament verify \
  --manifest /tmp/fitness-evidence/input.json \
  --output /tmp/fitness-evidence/workers-2
```

The manifest fingerprints code and runtime versions. A different checkout or
runtime should prepare a new manifest; silently relabelling old results with
new evaluator semantics is rejected.
