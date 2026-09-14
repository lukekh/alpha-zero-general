# Independent game-process benchmark (#48)

This experiment reproduces the version-2 continuation against
`ReferenceGreedyPlayer`. It measures independent spawned game processes, with one
PyTorch optimizer thread in the coordinator. It does not use Coach's shared
inference batching. The existing training job was neither stopped nor switched.

## Results — 2026-09-14

All **nine trials** exited successfully. Matched trajectories, complete replay,
final network weights, optimizer update counts and selection/best-model outcomes
are **exactly equal across worker counts for all three seed offsets**. Every
worker was reaped. The measurements comprise **432 physical modelling games**
(72 training rollouts, 180 frozen selection games and 180 candidate selection
games); paired repeats are not independent playing-strength observations.

| Seed offset | Game processes | Rollouts s | Optimization s | Candidate selection s | Complete iteration s | Iterations/hour |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [0](evidence/w1-seed0.json) | 1 | 11.809 | 26.683 | 28.093 | 66.720 | 53.96 |
| [0](evidence/w2-seed0.json) | 2 | 6.612 | 26.725 | 15.374 | 48.848 | 73.70 |
| [0](evidence/w4-seed0.json) | 4 | 4.341 | 26.605 | 9.904 | 40.988 | 87.83 |
| [10000](evidence/w1-seed10000.json) | 1 | 9.679 | 23.714 | 29.968 | 63.479 | 56.71 |
| [10000](evidence/w2-seed10000.json) | 2 | 5.486 | 25.108 | 18.940 | 49.672 | 72.48 |
| [10000](evidence/w4-seed10000.json) | 4 | 3.760 | 24.255 | 10.214 | 38.364 | 93.84 |
| [20000](evidence/w1-seed20000.json) | 1 | 10.221 | 25.617 | 29.063 | 65.019 | 55.37 |
| [20000](evidence/w2-seed20000.json) | 2 | 6.384 | 25.050 | 15.864 | 47.421 | 75.92 |
| [20000](evidence/w4-seed20000.json) | 4 | 3.702 | 25.367 | 9.625 | 38.824 | 92.73 |

Medians across the three trials, except RSS which shows the range of sampled
per-trial peaks. Process counts exclude the separate optimizer coordinator.

| Processes | Cold setup s | Frozen selection games/s | Rollout games/s | Candidate selection games/s | Complete-cycle games/s | Complete CPU % | Peak aggregate RSS MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 11.552 | 0.689 | 0.783 | 0.688 | 0.431 | 100.7 | 917.5–1267.6 |
| 2 | 12.589 | 1.191 | 1.253 | 1.261 | 0.573 | 141.8 | 1140.0–1344.7 |
| 4 | 14.056 | 1.738 | 2.128 | 2.019 | 0.721 | 190.6 | 1736.6–2086.0 |

Matched complete-iteration speedups relative to one game process:

| Seed offset | Two processes | Four processes |
| ---: | ---: | ---: |
| 0 | 1.366× | 1.628× |
| 10000 | 1.278× | 1.655× |
| 20000 | 1.371× | 1.675× |

Rollout responses carried **1.176–1.423 MiB** per trial; parent
receive/decode took **2.79–9.55 ms**. Durable bundle publication took
**0.053–0.069 s**. Parent interpreter bootstrap/shutdown added
**1.530–1.814 s** outside the timed trial body. Raw reports
also separate per-worker snapshot loading/export, startup, replay ordering,
candidate checkpoint writing and reload verification. These costs are included
in the relevant cold/complete totals; they are not treated as free.

**Recommendation:** on this measured M1 workload, opt into **four game workers**
for throughput when roughly **2.1 GiB of aggregate benchmark-process RSS** is
acceptable. Two workers are a useful lower-memory option (up to **1.32 GiB**).
The gains are repeated and substantially smaller than linear: single-thread
optimization remains the largest serial cost. Keep the portable **one-worker
default**, and leave the active 24-hour job unchanged. These results justify the
optional worker counts on comparable hardware; they do not establish isolated
performance, system-wide memory headroom or scaling on other machines.

The complete timings above are selection-bearing iterations. The normal
every-fifth-iteration selection schedule remains unchanged; extrapolating these
numbers directly to an entire 24-hour run would overstate selection frequency.

## Reproduction

From the repository root, with the Python 3.11 environment in `uv.lock`:

```sh
uv sync --locked
uv run python -m unittest intransitive.tests.test_greedy_process -v
uv run python -m intransitive.benchmark_process --output results/process-benchmark
```

The suite runs nine fresh trial coordinators, using worker orders **1/2/4,
2/4/1, 4/1/2** for seed offsets **0, 10000, 20000**. Use a fresh output directory
when changing sources; saved reports can be reused only with matching source and
fixture hashes. A single trial is available with `--trial --workers 2
--seed-offset 0 --output results/trial.json`. The default timeout is 1,800 seconds
per trial. The supervisor terminates the trial process group on timeout or
cancellation; each coordinator also owns and reaps its workers.

[fixture.pt](fixture.pt) is a complete, trusted local PyTorch continuation bundle,
not an external download dependency. Its [provenance](fixture.json) records the
read-only capture at completed iteration **84**, replay generations **83/84**
(3,000/2,580 examples), source artifact hashes and the unchanged status before and
after capture. Current/best network weights, AdamW moments/update count and
selection state are included. Redundant `full_model` objects were omitted.
The fixture uses the learned current model from the active continuation, not a
random network or the old version-1 benchmark. The benchmark gives its **copy** a
fresh measurement deadline so it remains reproducible after the original run's
deadline. Production continuation never extends that deadline.

Each trial performs:

1. Start the coordinator, load/validate the fixture and publish an immutable
   inference snapshot. Spawn 1/2/4 interpreters, load/export each inference model,
   create a private ONNX session, and warm MCTS and the greedy evaluator.
2. Measure **20 fixed selection games** on the unchanged fixture model separately.
3. Measure a complete **iteration 85**: distribute its snapshot, generate **eight
   alternating-colour training games**, collect replay in episode-index order,
   retain the previous replay generation, optimize with persistent AdamW, export
   the candidate and play **20 fixed selection games**. Apply strict best-score
   promotion and durably publish the complete continuation file.
4. Reload/validate the result and reap every worker.

All comparisons use **32 full MCTS simulations per model move**, batch **64**, one
epoch, learning rate **0.0001**, zero search-Q mixing, the original two-generation
57,600-example replay bounds, and all existing distinct symmetry triples (up to
12). Training seeds are `settings.seed + offset + iteration*100 + episode_index`.
Selection always uses seeds `800000..800019` with alternating colours. Each game
owns its state, MCTS, NumPy/Python/Torch seeds and greedy RNG. Model turns alone
supply MCTS policy targets; values are the **actual opponent-rollout result** in
model-relative player order. The original full-state symmetry deduplication is
retained.

The suite asserts exact matched action/full-state trajectory hashes, replay
bytes (therefore policies, values, masks and symmetry ordering), final trained
weights, update counts, selection outcomes and best-model state across worker
counts. Legal actions and unchanged transition inputs are checked during every
played move. These paired repetitions establish deterministic equivalence for
the measured seeds, not a playing-strength sample.

## Measurement definitions

- **Cold setup** includes fixture validation/distribution plus spawned interpreter
  imports, per-worker model load/export, ONNX setup and JIT/search warmup. Pool
  startup is also reported separately. Existing Numba disk caches are allowed;
  this is fresh-process startup, not a claim that every disk cache was erased.
  `spawned_command_seconds` includes parent interpreter startup and shutdown;
  `parent_bootstrap_and_exit_seconds` separates it from the timed trial body.
- **Warm iteration** uses initialized worker interpreters after the frozen
  selection probe. New checkpoint exports/sessions required for changing weights
  remain in its timing. Rollout and selection phase wall times include
  scheduling, snapshot loading, result serialization, pipe transfer and decoding.
- `result_bytes`, `parent_send_seconds` and `parent_receive_decode_seconds` expose
  replay/metadata transport size and parent-side IPC cost. These are not isolated
  bandwidth measurements. Worker-side serialization and scheduling overhead are
  included in phase and complete-iteration wall time. Replay ordering and durable
  commit times are reported separately.
- **Complete iteration** spans input checkpoint distribution through fsynced
  atomic continuation publication, including optimization and selection. It
  excludes initial worker setup, the frozen-model selection probe, and the later
  verification reload. It contains **28 games**; `28/seconds` is complete-cycle
  game throughput and `3600/seconds` is iterations/hour. Normal iterations without
  the every-fifth-iteration selection will have a different mix of costs.
- **CPU %** for the complete iteration is coordinator CPU plus reported worker
  task CPU divided by wall time: 100% means one busy core, 200% two. Worker IPC
  serialization outside the task timer is omitted from this approximate phase
  utilization. Whole-trial CPU includes reaped child CPU and coordinator CPU.
- **Memory** is the sampled sum of coordinator and descendant RSS every 250 ms.
  Shared pages are counted once per process, so this is a conservative aggregate,
  not private physical memory or a claim about system-wide peak memory. The
  supervising suite and competing training process are outside that sum.

Timing sources are preserved at [18c9728](https://github.com/lukekh/alpha-zero-general/commit/18c9728).
The subsequent legacy-capture boundary guard is outside the timed path and is
validated separately.

The hardware and package versions, Python workload inventory, exact source
hashes, raw timing phases, per-game records and memory samples are in
[evidence](evidence/manifest.json). The Apple M1 has eight logical CPUs (four
performance/four efficiency cores) and 8 GiB RAM. The original one-core training
job remained active throughout; desktop activity was uncontrolled. These are
**contended-machine measurements**, not an isolated hardware capacity estimate.

## Safe continuation

Create a *new* continuation from a compatible persistent-optimizer checkpoint:

```sh
uv run python -m intransitive.greedy_training \
  --initialize-from checkpoints/my-model.pt \
  --state checkpoints/greedy/state.pt --duration 86400
```

This performs the original 20-game baseline selection and creates a fresh replay
history. To preserve an existing run's replay and deadline, use its **bundle**
instead; `--initialize-from` is not a resume operation.

```sh
uv run python -m intransitive.greedy_training \
  --state checkpoints/greedy/state.pt --workers 1
```

Workers are opt-in (`1`, `2`, `4`); the default remains **1**. Resume preserves
weights, BatchNorm buffers, AdamW moments and update count, replay order, iteration
number, best checkpoint/score, last selection and the **original absolute
deadline**. `--duration` applies only to initialization. An expired bundle starts
no games. The coordinator handles SIGINT/SIGTERM and the deadline alarm; a failed
or cancelled game batch is discarded and workers are terminated and joined.
Optimizer moments are copied before updates to prevent PyTorch's CPU tensor
aliasing from mutating the committed or best checkpoint in memory.

A single file contains current/best checkpoints, replay and all continuation
metadata. Publication writes a unique same-directory temporary file, flushes and
fsyncs it, atomically replaces the committed bundle, then fsyncs its directory.
A digest binds replay content and order; generation and optimizer/selection
metadata are validated on publication and load. Interruptions before replacement
retain the previous complete iteration; interruptions after replacement expose
the new complete iteration. Replay collection, optimization and scheduled
selection are committed together. Work interrupted mid-iteration is regenerated
from the committed seed/iteration on resume; partial work does not consume a new
iteration or extend the deadline. The bundle is the source of truth if stdout
logging is interrupted after a successful commit. Immutable export snapshots live
in temporary directories and are removed after their users finish.

The local legacy script published weights/replay separately. A guarded **read-only
snapshot** path is provided:

```sh
uv run python -m intransitive.greedy_training \
  --capture-legacy-run checkpoints/greedy-training-20260914-215558 \
  --state checkpoints/greedy-captured.pt
```

Capture exits without starting or controlling training. It accepts only a stable
`opponent_rollouts` boundary, when the previous iteration and scheduled selection
are complete, with fewer than eight finished rollouts in **both** status reads.
The legacy script can publish replay immediately after its eighth-game status,
before changing phase; that window is explicitly rejected even if replay sizes
happen to agree. Capture checks status before/after the reads, checkpoint/progress/update
agreement, replay-generation sizes, exact logged game quotas/seeds/colours and
selection metadata. If the source advances through publication while being read,
or is in optimization/selection, capture fails and must be retried at a valid
boundary. Ambiguous legacy interruption states are rejected rather than guessed.
The captured bundle retains the original absolute deadline. Starting it while the
old job runs would create another training run; that is outside this experiment.

All repetition and **80-noncapture-ply** cutoffs here are **modelling-only**.
Neither official-play termination nor unfinished official-game reporting changes.
No architecture, game count, search budget or selection policy changes are made.

## Validation

All **nine new tests** pass. They cover inline/spawn trajectory and target parity,
colour-balanced exact quotas, startup failures/timeouts, worker exceptions and
SIGKILL, deadline/keyboard cancellation and cleanup, optimizer isolation,
interruption before/after publication, exact AdamW continuation, strict best-model
promotion and legacy snapshot consistency. The final legacy-capture guard also
passed the separate **six-test continuation suite**.

The full Intransitive suite ran **372 tests: 367 passed, five failed**. The same
five failures reproduce on untouched base commit `8aabd7a`, with **identical
selected moves, scores, completed depths, work counts and principal variations**.
They are the explicitly documented existing `test_game_blunders` defences: three
original-game cases (red 14/18/31) and both colours of the simplified last-defender
case. The reference-greedy/MCTS path measured here does not use that alpha-beta
player. All **seven repository tests** pass. No hosted CI workflow is configured.

[Validation manifest](evidence/validation.json),
[full Intransitive log](evidence/intransitive-tests.log),
[untouched-base comparison](evidence/base-blunders.log),
[repository tests](evidence/repository-tests.log),
[focused process tests](evidence/focused.log),
[final continuation tests](evidence/continuation-tests.log).

`uv build --wheel` succeeded. The resulting wheel was inspected to verify that
all three new modules and the complete checkpoint/replay fixture are included
with bytes identical to the source checkout. `git diff --check` passes.
