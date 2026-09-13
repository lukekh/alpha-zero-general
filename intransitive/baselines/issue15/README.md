# Reproducible Intransitive baseline (#15)

This baseline uses the [measured #14 configuration](../../benchmarks/training/baseline.json)
without changing the official opening, either modelling draw limit, the v1 network,
all 12 symmetries, or the 0.6 candidate acceptance threshold. The results below
measure this small run; training losses alone do not establish playing strength.

## Results — 2026-09-13

**The delivered model is `candidate_4.pt`, a rejected diagnostic baseline.**
All four candidate arenas were 0 wins / 8 draws / 0 losses. There is no accepted
`best.pt`; Coach restored the original random incumbent after every rejection.
Each candidate therefore trains from that incumbent on its current replay window,
rather than accumulating four accepted updates. Opponent evaluation found no wins
and a clear practical weakness against greedy. This run does not demonstrate an
improvement over random or the earlier candidate.

Training process: **150.121 / 600 seconds**. Evaluation process:
**59.464 / 300 seconds**. Combined: **209.585 / 900 seconds**.
Both supervised processes exited 0. Training completed 32 self-play + 32 candidate
games; the primary evaluation completed all 48 games.

Hardware: Apple M1, eight logical CPUs, 8 GiB RAM, macOS 15.6.1 arm64, Python
3.11.4, one Torch CPU thread and ONNX CPUExecutionProvider. State, network and
checkpoint formats are v1. Source revision: `e34a1cf7b3ee3275c5a609ef3f0a085b7b9bb348`.
The archived Python source hashes match this revision. Delivery revision
`bd4914e` additionally sets `ORT_DISABLE_TELEMETRY=1` before importing ORT;
training/evaluation behavior and retained weights are unchanged. Reproduction
uses that pre-import setting to prevent the native shutdown race. Installed
packages match the pinned smoke requirements.

### Training and throughput

| Iteration | Original positions | New augmented examples | Replay used | Updates | Policy KL mean | Value loss mean | Train seconds | Candidate W/D/L | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 325 | 3900 | 3900 | 60 | 0.527346 | 0.004960 | 11.437 | 0/8/0 | Rejected |
| 2 | 240 | 2880 | 6780 | 105 | 0.442458 | 0.003669 | 18.368 | 0/8/0 | Rejected |
| 3 | 301 | 3612 | 6492 | 101 | 0.427034 | 0.003760 | 18.657 | 0/8/0 | Rejected |
| 4 | 273 | 3276 | 6888 | 107 | 0.447085 | 0.003694 | 19.335 | 0/8/0 | Rejected |

Totals: **1,139 original positions → 13,668 augmented examples,
373 optimizer updates / 23,872 sampled examples**. All 12 symmetries
are present for every position; replay was not truncated. Policy and value loss
arrays are retained, not just averages. The shared objective weights value loss
by 0.25; its targets mix outcomes and search Q with Q weight 0.5.

| Phase | Wall seconds | Root simulations | Simulations/s | Inferred states/s |
| --- | ---: | ---: | ---: | ---: |
| self_play | 44.106 | 36,448 | 826.4 | 803.4 |
| training | 67.797 | 0 | 0.0 | 0.0 |
| candidate_arena | 32.681 | 30,720 | 940.0 | 904.7 |

Whole Coach time: 145.430s. Optimization throughput: 5.50 updates/s.
Unlike #14's warmed microbenchmark, these phase times include first-use compilation
and ONNX exports when they occur. Inference rates above divide inferred states by
whole phase wall time; JSON also retains session-only time and call counts.
Peak training process RSS: **412.84 MiB**.
Final replay: **6,888 examples**, **4.42 MiB** compressed resident estimate,
**44.22 MiB** decoded upper estimate. The two histories are below both
their example caps and the measured 128 MiB compressed planning allowance.

### Opponent evaluation

All counts are from the delivered model's perspective. Score is `(wins + draws/2)/n`.

| Opponent | Games | W/D/L | Score | Conservative 95% score interval | Approximate 95% win interval |
| --- | ---: | --- | ---: | --- | --- |
| random | 16 | 0/16/0 | 0.5000 | [0.160, 0.840] | [0.000, 0.194] |
| greedy | 16 | 0/1/15 | 0.0312 | [0.000, 0.371] | [0.000, 0.194] |
| earlier_checkpoint | 16 | 0/15/1 | 0.4688 | [0.129, 0.808] | [0.000, 0.194] |

| Opponent | Model colour | Games | W/D/L |
| --- | --- | ---: | --- |
| random | Blue | 8 | 0/8/0 |
| random | Red | 8 | 0/8/0 |
| greedy | Blue | 8 | 0/1/7 |
| greedy | Red | 8 | 0/0/8 |
| earlier_checkpoint | Blue | 8 | 0/7/1 |
| earlier_checkpoint | Red | 8 | 0/8/0 |

Each opponent produced 16 distinct action/full-state trajectories. The model's
seeded maximum-visit tie breaking makes games against deterministic greedy vary.
The score interval uses Hoeffding for independent bounded game scores; it covers
the mean for this balanced colour schedule without modelling draws as half a
binomial success. Wilson win/draw intervals are also retained per colour in JSON.
Pooled Wilson intervals are descriptive approximations for the two fixed colour
strata. Intervals are marginal, not simultaneous across opponents. The small
samples and one training seed do not establish performance across training seeds,
other openings, opponents or larger budgets. Zero wins does not imply a zero
underlying win probability. Verification games are excluded from strength counts.

### Termination and rejection investigation

| Phase/opponent | Games | Median/p95/max plies | Captures | Corner | Stalemate | Repetition | No-capture limit |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| self_play | 32 | 30/61.35/94 | 9 | 0 | 0 | 0 | 32 |
| candidate_arena | 32 | 30/30/30 | 0 | 0 | 0 | 0 | 32 |
| random | 16 | 30/114.75/117 | 10 | 0 | 0 | 0 | 16 |
| greedy | 16 | 19/33.5/41 | 63 | 15 | 0 | 1 | 0 |
| earlier_checkpoint | 16 | 59.5/116/125 | 36 | 1 | 0 | 0 | 15 |

All 32 self-play outcomes are modelling draws, although nine captures occurred.
Thus the outcome component of training has no positive/negative win signal; search
Q and policy targets still vary. Candidate arenas have **zero captures and end
at exactly 30 plies**, directly explaining every all-draw rejection. Evaluation
against greedy yields 15 official corner losses and one threefold draw, showing
that neither the engine nor the evaluation path forces every game to draw.
The earlier-checkpoint comparison also contains an official corner result.
The no-capture draws against random do not establish useful winning play.
Rules, draw rewards, search budget and rejection threshold were not changed.

These observations motivate controlled follow-up work on exploration and training
signal, but this single run does not establish a remedy. #16 can use the retained
opponents and seeds for its symmetry comparison; it must report weak/all-draw
results too and compare both original-position and compute efficiency.

## Validation evidence

- [Training report](evidence/training.json), [evaluation report](evidence/evaluation.json), and [fresh-process reproduction](evidence/verification.json).
- [119-test Intransitive suite](evidence/tests.log), [seven repository tests](evidence/repository-tests.log), and [evaluation accounting checks](evidence/accounting.log).
- [Training log](evidence/training.log), [evaluation log](evidence/evaluation.log), [verification log](evidence/verification.log), and [real pit CLI log](evidence/pit.log).
- [Verification after archive extraction](evidence/archive-verification.json) and [its log](evidence/archive-verification.log).
- [Artifact/source/parameter audit](evidence/validation.json) and [archive manifest](artifact-manifest.json).

Before training, two suite attempts passed the worker assertions but one child
segfaulted during shutdown. The [crash frames](evidence/preliminary-crash.json)
identify ONNX Runtime's telemetry uploader; [initial](evidence/preliminary-tests.log)
and [late-disable](evidence/preliminary-telemetry-tests.log) logs retain the failures.
API disabling initially allowed a clean suite, but two short verification
processes still aborted after reproducing their games ([original log](evidence/preliminary-verification.log),
[extracted-artifact log](evidence/preliminary-archive-verification.log)). The final
runner and reproduction shell set `ORT_DISABLE_TELEMETRY=1` **before import**.
The installed binary contains this setting; upstream [environment handling](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/platform/telemetry_environment.h)
and [POSIX initialization](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/platform/posix/telemetry.cc)
show that it prevents uploader initialization, whereas the API only disables
event collection after initialization. Final verification and the full suite
exit cleanly. Failed attempts remain recorded and are excluded from strength and
successful validation counts. This local workaround changes no inference or rules.

## Budget and reproducibility

Four iterations each have eight self-play games and eight candidate games, with
one CPU training/ONNX inference worker, 32 full MCTS simulations per move, no
Dirichlet noise, one epoch, batch 64, learning rate 0.0003 and Q weight 0.5.
Replay retains two histories of at most 57,600 examples each. The 128 MiB replay
allowance is a planning estimate; the example/history caps are enforced.

Training's subprocess deadline is **600 seconds**, including startup, persistence
and audits. Opponent evaluation has a separate **300-second deadline** for 48
games, including reload checks. Timeouts fail the invocation and preserve progress
JSON; they never become draws. The separate two-game reproduction check is a
validation invocation, outside the 900-second experiment budget.

Python, NumPy and Torch start at seed 150 with deterministic Torch algorithms
and one CPU thread. Self-play iteration seeds are 150–153, with independent
`SeedSequence([iteration_seed, episode_id])` generators. Random/greedy/earlier
opponent base seeds are 151/152/153. Each evaluation game derives four seeds from
`SeedSequence([opponent_seed, game_index])` for NumPy/Python/Torch, the model's
private MCTS RNG, random opponent RNG and earlier checkpoint's MCTS RNG. All
individual seeds are retained in the evaluation JSON.

Each evaluation starts at the official Blue-first setup with fresh trees.
Assignments follow Blue/Red/Red/Blue, yielding eight games per model colour per
opponent. Evaluation chooses maximum-visit actions with seeded random tie
breaking (`temp=0`), without sampling nonmaximal moves. Greedy remains the exact
deterministic two-ply heuristic. Candidate acceptance uses the existing Coach
arena and its temperature schedule. Results for these two protocols are reported
separately. `pit.py` uses its checkpoint temperature schedule, so its casual play
command is not an exact reproduction of the seeded evaluation protocol.

The selection rule was implemented before training/evaluation: deliver the latest
accepted candidate; if every candidate is rejected, retain the final trained
candidate as an explicitly rejected diagnostic baseline. Never promote rejected
weights to `best.pt`. The earlier trained opponent is always iteration one's
`candidate_1.pt`, regardless of acceptance. It is retained as `earlier.pt` for
future controlled comparisons. Opponent results do not affect model selection.

## Reproduce from scratch

From the repository root, use Python 3.11.4 and the pinned environment:

```sh
export ORT_DISABLE_TELEMETRY=1
python3.11 -m venv /tmp/intransitive-baseline-venv
/tmp/intransitive-baseline-venv/bin/python -m pip install -r intransitive/smoke/requirements.txt
BASELINE_PY=/tmp/intransitive-baseline-venv/bin/python \
BASELINE_ROOT=checkpoints/issue15-reproduction \
sh intransitive/baselines/issue15/reproduce.sh
```

The script requires both test suites, then runs training, evaluation and the
independent two-game reproduction check. Every output must be new. Commands can
also be run separately:

```sh
python -m intransitive.baseline train --output checkpoints/issue15-reproduction
python -m intransitive.baseline evaluate --model-folder checkpoints/issue15-reproduction --output checkpoints/issue15-reproduction/evaluation
python -m intransitive.baseline verify --model-folder checkpoints/issue15-reproduction --output checkpoints/issue15-reproduction/verification
```

`verify` reloads both checkpoints in a fresh process, checks CPU/ONNX predictions
on a full-history replay state, and exactly matches the first two random-opponent
games (one per model colour): seeds, outcomes, terminal reasons, capture counts,
plies and complete action/state trajectory hashes. Floating-point behavior may
vary on different hardware or package versions; failures must be investigated.

## Artifact and human play

The committed [baseline-artifacts.tar.gz](baseline-artifacts.tar.gz) contains the reloadable `baseline.pt`,
`earlier.pt`, initial weights, every candidate (including rejected candidates),
accepted checkpoints if any, compressed replay, effective settings, budget,
requirements, reports, and the complete shared/game Python source snapshot.
[artifact-manifest.json](artifact-manifest.json) records the archive checksum and every retained file's
size and SHA-256. The archive is 13,991,070 bytes and contains 70 files. All extracted file hashes
and 44 Python source files were verified. Checkpoint SHA-256:

- `baseline.pt`: `c4eca4769284b57ee17dfee7e22b83e9a8315f0a7b7311af5f1c027faa8e4e7b`
- `earlier.pt`: `40b8fc6eecc606e4cfaaa45b0af23cf7f5d1fc89d2ad21cfb27aa80dd0b50e14`

Both representative games matched exactly before and after archive extraction.
Maximum CPU/ONNX reload policy/value differences were **7.08e-8 / 1.72e-7**.
All four candidates reload with valid metadata and finite, changed trainable
parameters; `temp.pt` exactly retains the initial incumbent. A real `pit.py`
checkpoint-versus-greedy invocation completed two games and exited 0.

Extract from the repository root:

```sh
mkdir -p checkpoints/issue15-delivered
tar -xzf intransitive/baselines/issue15/baseline-artifacts.tar.gz -C checkpoints/issue15-delivered
ORT_DISABLE_TELEMETRY=1 python pit.py intransitive human checkpoints/issue15-delivered/baseline.pt -n 2 -m 32
```

Enter coordinate pairs such as `B5 C5`. Two games assign the human each colour;
Blue moves first in both. Put the checkpoint before `human` to play Red first.
For a fresh reproducibility check on the delivered artifact:

```sh
python -m intransitive.baseline verify --model-folder checkpoints/issue15-delivered --output checkpoints/issue15-delivered/new-verification
```

The source snapshot also includes `main.py` and checkpoint/replay continuation
support. See [checkpoint continuation](../../smoke/README.md) for the bounded
resume path; optimizer/scheduler state is recreated, as the checkpoint metadata
explicitly records. The issue #16 symmetry comparison should start from the
fixed settings/seeds/opponents here and account for compute, not augmented
example count alone.
