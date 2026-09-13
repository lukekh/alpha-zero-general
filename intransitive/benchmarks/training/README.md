# Measured CPU/ONNX training budget (#14)

Use **one inference worker, CPU training, ONNX CPU inference, all 12 symmetries,
32 simulations/move, batch 64, eight games/iteration and four iterations** for the
first #15 baseline. Retain two replay histories, at most 57,600 examples each.
Allow **600 seconds for training and candidate arenas, plus 300 seconds for
48 post-training evaluation games**. These are explicit stop budgets, not a
promise of completion or playing strength. The complete recommendation and its
memory calculation are in [baseline.json](baseline.json).

## Results — 2026-09-13

Seven fresh processes exited **0**, completing **64 physical games, 960 original
self-play positions, 11,520 augmented examples and 177 optimizer updates**.
Every position produced 12 distinct full-state examples. No replay was truncated.
The 116-test Intransitive suite and seven repository tests passed before trials;
the two accounting tests also passed after the auditor correction described below.
[Validation manifest](evidence/validation.json), [test log](evidence/focused.log),
[repository log](evidence/repository.log), [accounting log](evidence/accounting.log).

Hardware: **Apple M1, 8 logical CPUs, 8 GiB RAM, macOS 15.6.1 arm64**;
Python 3.11.4, one PyTorch CPU thread, ONNX `CPUExecutionProvider` with one
intra-op thread and sequential execution. The wrapper leaves the inter-op
setting at 0 (automatic; unused for sequential execution), as verified in the
validation manifest. Exact installed packages match the
[smoke requirements](../../smoke/requirements.txt); every report also embeds them.
State, network and checkpoint format versions are **1**. Rules, draw limits,
network architecture and acceptance threshold are unchanged.

### One versus two workers

Each trial starts from scratch: one iteration, four self-play games, 16 MCTS
simulations/move, full-search probability 1, no Dirichlet noise, one epoch,
batch 64, learning rate 0.0003 and four candidate arena games. Trials run
sequentially; worker order alternates across seeds 140–142. Each produces
120 original positions, 1,440 examples and 22 updates (1,408 sampled examples;
the trainer uses `floor(examples / batch_size)` updates per epoch).

| Seed | Workers | Self-play s | Training s | Arena s | Coach total s | Self-play sims/s | Peak child RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [140](evidence/final-w1-seed140.json) | 1 | 2.666 | 4.413 | 2.346 | 9.625 | 720 | 536.8 |
| [140](evidence/final-w2-seed140.json) | 2 | 2.595 | 4.366 | 2.283 | 9.345 | 740 | 634.9 |
| [141](evidence/final-w1-seed141.json) | 1 | 2.689 | 4.707 | 2.302 | 9.923 | 714 | 559.2 |
| [141](evidence/final-w2-seed141.json) | 2 | 2.705 | 4.866 | 2.319 | 10.012 | 710 | 546.5 |
| [142](evidence/final-w1-seed142.json) | 1 | 2.651 | 4.246 | 5.111 | 12.173 | 724 | 595.3 |
| [142](evidence/final-w2-seed142.json) | 2 | 2.650 | 4.283 | 3.328 | 10.365 | 724 | 604.6 |

The complete sets of self-play action/full-state trajectory hashes match across
worker counts for **all three seeds**. Two-worker self-play speedups are
1.0275×, 0.9939× and 1.0001×; there is no consistent useful gain in this small
sample. One worker is the chosen baseline. RSS ranges overlap and include
setup and the later batch-32 inference probe; they are not isolated worker-memory
measurements or a claim that every two-worker run consumes more memory.

Compare **self-play**, where weights and trajectories match. Replay reaches the
trainer in episode-completion order. For seed 142, that order differs, changing
sampled training batches and the candidate: one worker's candidate wins 2 and
draws 2 (accepted), while two workers draw all 4 (rejected). The other four
candidate arenas draw all 4 and reject. Their total-cycle/arena differences
cannot establish a worker speedup or a strength difference. The selected
one-worker configuration avoids this scheduling-dependent replay order.

### Larger-budget calibration

[Seed 143](evidence/calibration-w1-seed143.json) measures the recommended
per-iteration search/game/batch settings: one worker, eight self-play games,
32 simulations/move, one epoch, batch 64, and eight candidate arena games.

| Metric | Measurement |
| --- | ---: |
| Original positions / augmented examples | 240 / 2,880 |
| Training updates / sampled examples | 45 / 2,880 |
| Self-play / training / arena wall seconds | 8.736 / 8.615 / 11.326 |
| Coach total / other Coach overhead seconds | 28.868 / 0.191 |
| Setup including warmup / whole child-process seconds | 7.940 / 39.853 |
| Self-play root simulations / simulations per wall second | 7,680 / 879 |
| Self-play inferred states / states per wall second | 7,427 / 850 |
| ONNX session throughput during self-play | 1,025 states/s |
| Training throughput | 5.22 updates/s |
| Mean policy KL / value loss | 0.5519 / 0.004887 |
| Peak child-process RSS | 503.8 MiB |
| Candidate W/D/L | 0 / 8 / 0 — rejected |

On the calibration's final replay state, after three warmups and 30 timed calls:

| ONNX batch | Median call ms | p95 call ms | States/s |
| --- | ---: | ---: | ---: |
| 1 | 0.949 | 0.976 | 1,050 |
| 2 | 1.853 | 1.900 | 1,075 |
| 32 | 29.191 | 29.707 | 1,094 |

These session-only measurements exclude conversion, queuing and export. They
explain the limited batching benefit on this machine. They do not imply the same
rates on another device or network. All seven supervised processes together took
161.16 seconds, excluding the test suite and intervals between invocations.

### Legal actions and termination

Legal actions are counted **once per original self-play position**, not once per
symmetry or every MCTS leaf. The paired seeds' means are 46.55, 48.33 and 47.98;
medians 47, 48 and 49; p95 values 54, 59 and 59.05; overall range 35–63.
Calibration mean/median/p95/max are 48.19 / 49 / 57 / 60.

| Physical games | Count | Median plies | p95 plies | Max plies | Corner | Stalemate | Repetition | No-capture limit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Self-play, all seven trials | 32 | 30 | 30 | 30 | 0 | 0 | 0 | 32 |
| Candidate arenas, all seven trials | 32 | 30 | 82.25 | 123 | 2 | 0 | 0 | 30 |
| Calibration self-play only | 8 | 30 | 30 | 30 | 0 | 0 | 0 | 8 |
| Calibration arena only | 8 | 43 | 56 | 56 | 0 | 0 | 0 | 8 |

Percentiles use NumPy's default linear interpolation; raw per-game rows are in
the reports. The pairs deliberately repeat the same self-play trajectories, so
these are counts of work performed, **not 64 independent strength observations**.
Zero observed stalemates/repetitions does not establish zero probability; existing
rule tests cover those reasons and official-win precedence explicitly.

All self-play games reach 30 plies without a capture, explaining the draw-heavy
training labels. Longer candidate games show that captures can occur after
training. Six of seven candidates are rejected for all-draw arenas. This is a
useful diagnostic for #15's opponent evaluations, not evidence to change the
rules or exclude draws differently. Official corner/stalemate wins remain
separate from the modelling-only repetition and 30-noncapture draws.

## Replay cost and chosen limits

The complete `(state, policy, outcome, valid_mask, Q)` tuple is measured. State
history lengths span 1–30; terminal states are not themselves training positions.
There are 2,673 state bytes and 5,921 total NumPy array bytes per example; Q's
Python list/scalars are included in the decoded object estimate below.

| Storage measurement | Four-game trials | Eight-game calibration |
| --- | ---: | ---: |
| Mean compressed bytes/example | 613.43–622.05 | 632.96 |
| Calibration compressed median / p95 / max bytes | — | 654 / 722 / 743 |
| Maximum compressed bytes across all trials | 747 | 743 |
| Original uncompressed pickle bytes/example | 6,326 | 6,326 |
| Decoded owned Python/NumPy bytes/example estimate | 6,723 | 6,723 |
| Compressed replay resident bytes, including deque/references | 0.900–0.911 MiB | 1.853 MiB |
| Decoded replay resident upper estimate | 9.244 MiB | 18.488 MiB |
| Persisted `checkpoint.examples` size | recorded per report | 1,840,504 bytes |
| Encode / decode whole calibration replay | — | 0.199 / 0.0358 s |

Pickle sizes are measured on Coach's original bytes. Decoded sizes use
`sys.getsizeof`, ndarray ownership and within-example reference deduplication;
summing across examples conservatively ignores sharing between examples. They
exclude allocator fragmentation. Process RSS also contains PyTorch/ONNX, JIT,
models, tensors and the inference microbenchmark. Replay is inspected one example
at a time, so this audit does not inflate RSS by holding an uncompressed copy of
the whole replay. Training decodes only its sampled minibatch.

The chosen **128 MiB replay planning allowance** uses the observed maximum
747-byte compressed blob, 33 bytes of Python `bytes` overhead and eight bytes
per reference, with 25% headroom: `ceil((747 + 33 + 8) * 1.25) = 985` bytes/example.
With two histories this allows **68,124 examples/history**, rounded down to a
multiple of 12. Eight games can generate at most `8 * 600 * 12 = 57,600`
examples: at most 19 captures, each after at most 29 noncaptures, then 30 final
noncaptures bound a game to 600 plies. Set the queue to the smaller limit,
**57,600**, keeping **all** symmetries even at that rule-derived game-length bound.
Two full queues reserve **108.22 MiB** using the planning estimate, below 128 MiB.
The comparable decoded estimate would be **738.61 MiB**. Observed two-history
cost at calibration-like lengths is only about **3.71 MiB compressed**.

This is an observed-cost planning margin, not a guaranteed byte limit for unseen
histories/policy distributions. Re-measure before raising search/game/replay
budgets. The configuration itself has strict example/history caps and the run
has a separate time cap. Generic repository estimates are not used.

Materialized augmentation is affordable at this budget, so it is retained.
It multiplies stored examples and optimizer work by 12, but the actual compressed
replay is small and no symmetry coverage is dropped. No minibatch-augmentation
redesign is justified by these measurements. If a larger run exceeds its replay
allowance or materialization dominates memory, a follow-up should store original
positions, transform full histories/policies/masks when sampling, verify all
12 transformations, and compare equal-update and equal-wall-time runs. #16
remains the planned controlled symmetry comparison.

## Budget for #15

[baseline.json](baseline.json) records every setting and the derivation above:

- Four iterations; eight self-play and eight candidate-arena games per iteration:
  **32 self-play + 32 arena games**, always with Blue first.
- One worker; 32 simulations/move; full search; no Dirichlet noise; unchanged v1
  network; CPU optimization; ONNX CPU inference; all 12 symmetries.
- One epoch, batch 64, learning rate 0.0003, Q weight 0.5; two replay histories.
  Keep threshold 0.6 and retain rejected candidates. `stop_after_N_fail=4` permits
  the four-budgeted-iteration diagnostic even if all candidates draw.
- Seed Python, NumPy and PyTorch with 150; deterministic PyTorch algorithms.
  Set `selfplay_seed` to 150/151/152/153 before each iteration, with independent
  `SeedSequence([iteration_seed, episode_id])` generators. Evaluation seeds are
  151/152/153 for random/greedy/earlier-checkpoint opponents, including their
  private generators.
- Enforce **600 seconds** for the training invocation. Stop if the wall or
  iteration/game budget is exhausted; a timeout is an incomplete run, not a draw.
- Reserve **300 seconds** for **16 games/opponent**, against random, deterministic
  greedy and an earlier available checkpoint: eight model games per colour,
  **48 games total**, search capped at 32. Report actual counts if interrupted.
  Total wall allowance: **900 seconds**. #15 must retain reloadable weights,
  report uncertainty and investigate all-draw rejection without promising strength.

For planning only, if game lengths stay near calibration, two-history training
roughly doubles the 8.615-second training phase after iteration one. Four cycles
then cost about 141 seconds plus setup; this arithmetic is an extrapolation,
**not a measured four-iteration result**. The 600-second ceiling leaves room for
longer games, reloads and tails. Long-game behavior can still exhaust it. At the
600-ply rule bound the specified 64 training/arena games allow at most 1,228,800
root simulations and, with these replay caps, 6,300 optimizer updates. Do not
replace this bounded configuration with `main.py`'s generic defaults.

## Reproduce and audit

From the repository root, use Python 3.11.4 and the pinned environment:

```sh
python3.11 -m venv /tmp/intransitive-benchmark-venv
/tmp/intransitive-benchmark-venv/bin/python -m pip install -r intransitive/smoke/requirements.txt
BENCH_PY=/tmp/intransitive-benchmark-venv/bin/python \
BENCH_ROOT=checkpoints/issue14-reproduction \
sh intransitive/benchmarks/training/reproduce.sh
```

The recorded runs used `/tmp/intransitive-issue10-venv/bin/python` with those
packages. The script first requires both suites to pass, then runs all trials
sequentially in the recorded order. Each benchmark refuses an existing output
directory and has a **600-second subprocess deadline** (maximum configurable
3,600 seconds). The CLI also bounds games, simulations, batch size and probe
repetitions. An individual calibrated command is:

```sh
python -m intransitive.benchmark_training --output checkpoints/calibration \
  --workers 1 --seed 143 --games 8 --arena-games 8 --simulations 32 --timeout 600
```

Reports and logs are in [evidence/](evidence/). Each run records effective
settings, package versions, command, hardware, state/network/checkpoint metadata,
source SHA-256 manifest, checkpoint/replay hashes and local artifact locations.
Each output directory retains source/settings snapshots, replay, pre-training
`temp.pt`, and `candidate_1.pt`; accepted candidates also have `best.pt` and
`checkpoint_1.pt`. These are measurement artifacts, not the delivered #15 model.

`git_revision` is checkout base `91a4daa` plus this issue's changes. All seven
source manifests are identical and match the final Python files and retained
snapshots. Documentation, shell reproduction commands and evidence were added
after measurements. Checkpoints remain in the recorded local directories; the
commands regenerate them. JSON numbers, not rounded tables, are authoritative.

The benchmark wraps real `Coach.learn`, MCTS calls, CPU optimizer updates and
ONNX session calls. Warmup compiles search/transitions/symmetries and exports
ONNX before self-play timing; setup is reported separately. Training timing
includes optimizer creation and minibatch decoding; arena timing includes its
necessary ONNX exports. Coach overhead covers persistence, shuffle, checkpoint
handling and acceptance. Search simulations count completed **root loop**
iterations, not recursive nodes. Simulations/s uses phase wall time, never the
sum of overlapping worker times. `search_worker_seconds` includes inference
waits; session throughput and phase throughput are reported separately.

Fixed worker quotas remove polling-driven game overshoot. Finished and zero-quota
workers clear their slots; the server infers only active observations. Tests cover
one and three requested games on two real ONNX workers, delayed startup, complete
legal histories, distinct mutable Boards, replay draining and clean shutdown.
The benchmark checks parent/root byte stability on every move/search and validates
every replay state. Matching full-state trajectory hashes across workers provide
additional measured evidence against history contamination. NumPy/Torch/Python
seeds are explicit; MCTS's private unseeded generator cannot change a full-search,
zero-noise decision. Training replay completion order is the documented multiworker
exception above; floating-point decisions can also differ across environments.

Review corrected one auditor assumption: decoding/re-encoding NumPy pickles can
change dtype alias memoization (6,326 to 6,360 bytes) without changing values.
The preliminary trial therefore failed its final audit and is excluded from all
measurements. Final runs compare decoded values, count original wire bytes and
verify the persisted replay equals Coach's compressed records exactly. All final
runs passed and exited 0; the successful-process marker is written only by the
supervisor after child exit. ONNX telemetry is disabled in this command, as in
the previously validated smoke gate, to avoid the observed macOS shutdown race.
