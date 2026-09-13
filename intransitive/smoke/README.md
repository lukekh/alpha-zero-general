# Bounded training smoke gate (#13)

This is a pipeline check, not a strength benchmark. It runs the real `Coach.learn`
with one inference worker, unchanged official rules, and the confirmed repetition
and 30-noncapture modelling draws. Candidate acceptance still excludes draws;
an all-draw arena rejects the candidate.

## Recorded result — 2026-09-13

All three final processes exited **0**. The focused suite passed **113 tests**
and the repository suite passed **7 tests**, before the smoke. Source hashes
were checked against every final Python source and every restored snapshot.

| Cycle / seed | New positions | New examples | Training examples | Updates | Candidate W/D/L | Decision | Seconds |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| [initial-onnx](evidence/initial-onnx.json) / 13 | 60 | 720 | 720 | 11 | 1/1/0 | accept | 12.94 |
| [resume-cpu](evidence/resume-cpu.json) / 14 | 60 | 720 | 1440 | 22 | 0/2/0 | reject | 16.06 |
| [resume-onnx](evidence/resume-onnx.json) / 15 | 151 | 1812 | 2532 | 39 | 0/1/1 | reject | 18.35 |

Total: **271 original positions, 3,252 new augmented examples, 72 optimizer
updates, and 24 completed games**. Every original self-play position produced
12 distinct full-state examples; maximum recorded history length was 30.
Both continuations restored one replay history and trained with two histories.
The initial candidate was accepted; both resumed candidates were rejected and
remained reloadable. All 12 checkpoint-versus-random games completed across
the two inference paths, including games using both rejected candidates.

The largest observed parameter change per cycle was 0.0014601 / 0.0031274 /
0.0050082. All policy/value losses and parameters were finite. Maximum reload
policy/value differences across the runs were 3.43e-7 / 1.64e-7, within the
recorded tolerance. These small arenas are completion evidence, not strength
estimates. Game decisions may differ across inference backends near search ties
even when prediction parity passes.

## Reproduce

Use Python 3.11.4. The recorded machine is an Apple M1 with 8 GiB RAM, macOS,
one PyTorch CPU thread, CPU optimization, and CPUExecutionProvider for ONNX.
Exact package versions are in [requirements.txt](requirements.txt).
From the repository root:

```sh
python3.11 -m venv /tmp/intransitive-smoke-venv
/tmp/intransitive-smoke-venv/bin/python -m pip install -r intransitive/smoke/requirements.txt
export SMOKE_PY=/tmp/intransitive-smoke-venv/bin/python
export ORT_DISABLE_TELEMETRY=1
export SMOKE_ROOT="$PWD/checkpoints/issue13"
mkdir -p "$SMOKE_ROOT/logs"
"$SMOKE_PY" -m unittest discover -s intransitive/tests -v > "$SMOKE_ROOT/logs/focused.log" 2>&1
"$SMOKE_PY" -m unittest discover -s tests -v > "$SMOKE_ROOT/logs/repository.log" 2>&1
"$SMOKE_PY" -m intransitive.smoke --checkpoint "$SMOKE_ROOT/initial-onnx" --seed 13 --backend onnx > "$SMOKE_ROOT/logs/initial-onnx.log" 2>&1
```

Require both test commands to succeed before starting the smoke. The recorded run
used the already installed `/tmp/intransitive-issue10-venv/bin/python` with exactly
the pinned packages. Each smoke command refuses an existing output directory.

Each cycle has **one iteration, two self-play games, four MCTS simulations per
move, full search probability 1, no Dirichlet noise, one optimization epoch,
batch size 64, learning rate 0.0003, and two arena games** with the candidate
assigned each colour once. Every physical game starts with Blue. Compression is
on, replay retains at most two iterations, and each iteration's queue holds
14,400 examples. At most 19 captures and 30 noncaptures per segment bound a game
to 600 plies, so two games and all 12 transformations fit without truncation.
Four additional candidate-versus-random checkpoint games run per cycle: two on
PyTorch and two on ONNX, swapping model colours. Thus the three cycles below
complete six self-play, six candidate arena, and twelve checkpoint games.

Python `random`, NumPy and PyTorch are seeded with 13/14/15 for the initial and
resumed cycles. Checkpoint games reseed Python/NumPy and the random opponent’s private generator
with cycle seed + 10.
PyTorch deterministic algorithms are enabled. MCTS's private unseeded generator
cannot affect these settings: every search is full and Dirichlet noise is off;
compiled action selection and Intransitive transitions are deterministic.
Same-environment reruns should reproduce decisions; floating-point parity across
hardware is subject to the documented tolerances.

## Restore and resume both inference paths

Read the exact snapshot path from the initial report; run the next commands from
that snapshot to prove the source backup is executable. All training/search
settings are loaded from its JSON; the output path, input checkpoint, backend and
seed are explicit overrides. Resume loads both the saved weights and adjacent
`checkpoint.examples`. AdamW and OneCycleLR are recreated, as documented in #12.

```sh
export SMOKE_BACKUP=$("$SMOKE_PY" -c 'import json,os; print(json.load(open(os.environ["SMOKE_ROOT"] + "/initial-onnx/report.json"))["source_backup"])')
cd "$SMOKE_BACKUP"
"$SMOKE_PY" -m intransitive.smoke --settings settings.json --resume "$SMOKE_ROOT/initial-onnx/candidate_1.pt" --checkpoint "$SMOKE_ROOT/resume-cpu" --seed 14 --backend cpu > "$SMOKE_ROOT/logs/resume-cpu.log" 2>&1
"$SMOKE_PY" -m intransitive.smoke --settings settings.json --resume "$SMOKE_ROOT/initial-onnx/retained.pt" --checkpoint "$SMOKE_ROOT/resume-onnx" --seed 15 --backend onnx > "$SMOKE_ROOT/logs/resume-onnx.log" 2>&1
```

The CPU run intentionally resumes the trained candidate even if rejected; the
ONNX run resumes the retained incumbent. These are independent one-iteration
continuations of the initial run. This distinguishes candidate inspection from
promotion and demonstrates both useful restart choices.

## Artifacts and assertions

Each directory contains `report.json`, exact `settings.json` and legacy
`settings.txt`, `requirements.txt`, full shared/game Python `source_backups/`,
`checkpoint.examples`, `temp.pt` (pre-update incumbent), `candidate_1.pt` (trained
candidate), and `retained.pt` (post-arena incumbent). Acceptance also creates
`best.pt` and `checkpoint_1.pt`. Candidate files are saved before arena and before
rollback/early stopping. Iteration numbers restart on a resumed invocation; use
a new directory to preserve earlier candidates.

The auditor checks every played move and full state, every augmented state's
valid mask and policy, all finite policy/value losses, actual parameter changes,
replay round trips without truncation, arena-derived acceptance, and exact
post-arena weights. A replay position with real history is reloaded through
`nn_version=-1` and compared on PyTorch/ONNX with `atol=2e-6, rtol=1e-5`.
Reports include raw positions, augmentation multiplicities/history lengths,
optimizer updates/deltas/losses, game lengths/reasons, W/D/L, package versions,
checkpoint metadata and SHA-256 hashes, and hashes for every backed-up Python
source. The focused suite separately verifies all 12 transformations exactly,
dynamic-batch inference, multiworker isolation, and accepted/rejected arenas.

Reports and logs are committed in [evidence/](evidence/). Checkpoints, replay and
source snapshots remain under the recorded local `checkpoints/issue13/` directory;
the commands recreate them. They are deliberately tiny-budget gate outputs and
are not the trained baseline requested by #15. Throughput, memory measurement,
worker comparisons and an appropriate larger budget remain #14.

## Review findings

The first resumed ONNX process completed all assertions and wrote its report,
then exited with SIGABRT. The macOS crash stack showed
`onnxruntime::PosixTelemetry::Shutdown` / Microsoft telemetry request teardown
and `recursive_mutex lock failed: Invalid argument`. The smoke command now calls
`onnxruntime.disable_telemetry_events()` before inference; the final evidence
requires successful process exits as well as `passed: true` reports. This change
is confined to the smoke command. The preliminary log is retained as
[evidence/preliminary-onnx-shutdown.log](evidence/preliminary-onnx-shutdown.log).
Review also replaced a hard-coded metadata offset in the history-length metric
with `META_HISTORY_LENGTH`; the final runs report the actual history length.
The game's state and training code were unaffected by that reporting correction.
The final checkpoint opponents also receive an explicit seed for their private
NumPy generator, independently of global NumPy seeding.

`git_revision` records the checkout base at invocation. The final smoke ran on
base `b3646c5` plus this issue's changes; `source_sha256` identifies the exact
Python sources used, including the auditor, and the source-backed resumes verify
the same files. Documentation and evidence were added after the runs.
