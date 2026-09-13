# Controlled symmetry efficiency comparison (#16)

This experiment compares materializing all 12 symmetries (`on`) with retaining
only the identity example (`off`). Both use the real Coach, v1 network, official
Blue-first opening, exact repetition history, modelling draws, 32 full search
simulations, and #15's fixed random/greedy/earlier-checkpoint opponents.

## Finding and recommendations

**This bounded comparison does not establish a strength or learning-efficiency
benefit from augmentation.** Both final candidates draw all 16 games against
random and lose all 16 to greedy. Their scores against the fixed earlier
checkpoint are 0.5000 on versus 0.4688 off; the paired difference's conservative
95% interval is **[-0.648, 0.710]**. That uncertainty does not establish equivalence
either. Across 325–1,139 cumulative original positions, random and greedy scores
remain flat in both arms. There is only one paired training seed.

The enabled final candidate **was accepted** after 2 wins / 5 draws / 1 loss
against its incumbent; `on/best.pt` and `on/baseline.pt` contain candidate 4.
Only three arena games were decisive, and fixed-opponent results do not support
a strong-model claim. All disabled candidates were rejected; `off/baseline.pt`
is the explicitly rejected candidate 4 and there is no `off/best.pt`. The first
three rejections in both arms mean all four candidates trained from identical
initial incumbent parameters on their respective replay windows. The enabled
fourth acceptance occurs after the final self-play collection, so it does not
change the matched original data. #15's original baseline remains rejected;
this new experiment resets optimizer seeds as documented below.

At matched 373 updates, enabled training took **172.163s vs 164.988s** (4.3% more)
and final compressed replay estimates were **4.421 MiB vs 0.367 MiB** (about 12×).
Total training plus all checkpoint evaluations was **364.665s vs 362.489s**.
Materializing twelve examples did not provide twelve times as much independent
data, strength, or throughput. The peak RSS estimates and phase measurements
below also show why replay size alone is not whole-process memory or runtime.

- Keep the existing all-12 production default pending broader strength evidence;
  this run does not justify a strength-based switch in either direction.
- For a replay-memory-constrained comparison, the identity arm demonstrably
  stores less replay. Preserve the update budget when using it; equal epoch
  counts would spend much less optimization compute without augmentation.
- Measure additional independent training seeds and more evaluation games
  before choosing a strength/compute tradeoff. All 32 original self-play games
  per arm ended in modelling draws, so the outcome component supplies no win/loss
  signal in this corpus. Investigate training signal and exploration in separate
  controlled experiments; this comparison does not establish a remedy.

## Measurements — 2026-09-13

Both arms completed **373 optimizer updates / 23,872 sampled examples**. **All self-play trajectories and complete original training examples matched exactly.** These checks compare every original board, policy, outcome, legal mask and Q target. Original positions count occurrences collected by self-play, not globally unique states.

The measured experiment comprises **64 self-play + 64 candidate-arena + 384 fixed-opponent games**. The four additional fresh-process verification games are excluded from strength counts. Hardware: Apple M1, eight logical CPUs, 8 GiB RAM, macOS 15.6.1 arm64, Python 3.11.4, one Torch CPU thread and ONNX CPUExecutionProvider. Packages match the pinned smoke environment. Source revision: `b748cc3`; complete per-file source hashes are retained.

| Arm | Original positions | Materialized examples | Train process seconds / CPU seconds | Evaluation process seconds | Total experiment seconds / 1,800 |
| --- | ---: | ---: | --- | ---: | ---: |
| on | 1,139 | 13,668 | 172.163 / 164.387 | 192.502 | 364.665 |
| off | 1,139 | 1,139 | 164.988 / 160.182 | 197.500 | 362.489 |

### Candidate quality per original position and compute

These are evaluations of every saved candidate, including rejected candidates. Cumulative updates and positions include work spent on earlier rejected candidates; the replay column reports the original positions actually available to that candidate. After rejection Coach restores the incumbent, and each optimizer/scheduler is recreated. This is a closed-loop Coach comparison, not four uninterrupted accepted training epochs. The score is `(wins + draws/2)/games` for 16 games per opponent at each point.

| Iteration | Arm | Cumulative originals / replay originals | Cumulative updates | Coach wall seconds | Random score | Greedy score | Earlier score | Candidate arena W/D/L | Decision |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | on | 325 / 325 | 60 | 41.176 | 0.5000 | 0.0000 | 0.5312 | 0/8/0 | Rejected |
| 1 | off | 325 / 325 | 60 | 39.020 | 0.5000 | 0.0000 | 0.5000 | 0/8/0 | Rejected |
| 2 | on | 565 / 565 | 165 | 79.304 | 0.5000 | 0.0000 | 0.5312 | 0/8/0 | Rejected |
| 2 | off | 565 / 565 | 165 | 78.830 | 0.5000 | 0.0000 | 0.5000 | 0/8/0 | Rejected |
| 3 | on | 866 / 541 | 266 | 119.065 | 0.5000 | 0.0000 | 0.5000 | 0/8/0 | Rejected |
| 3 | off | 866 / 541 | 266 | 123.142 | 0.5000 | 0.0000 | 0.4688 | 0/8/0 | Rejected |
| 4 | on | 1139 / 574 | 373 | 166.976 | 0.5000 | 0.0000 | 0.5000 | 2/5/1 | Accepted |
| 4 | off | 1139 / 574 | 373 | 159.673 | 0.5000 | 0.0000 | 0.4688 | 0/8/0 | Rejected |

The full learning measurements, including per-update loss arrays and parameter hashes, are in [on training](evidence/on-training.json) and [off training](evidence/off-training.json). Training losses describe different replay distributions and are not an independent strength metric.

| Iteration | On policy KL / value loss | Off policy KL / value loss |
| ---: | --- | --- |
| 1 | 0.523729 / 0.004932 | 0.225007 / 0.004162 |
| 2 | 0.445334 / 0.003808 | 0.202094 / 0.003379 |
| 3 | 0.424209 / 0.003728 | 0.205859 / 0.003186 |
| 4 | 0.447746 / 0.003662 | 0.200814 / 0.003152 |

### Final candidates against fixed opponents

| Arm | Opponent | W/D/L (16 games) | Score | Conservative marginal 95% score interval |
| --- | --- | --- | ---: | --- |
| on | random | 0/16/0 | 0.5000 | [0.160, 0.840] |
| on | greedy | 0/0/16 | 0.0000 | [0.000, 0.340] |
| on | earlier_checkpoint | 0/16/0 | 0.5000 | [0.160, 0.840] |
| off | random | 0/16/0 | 0.5000 | [0.160, 0.840] |
| off | greedy | 0/0/16 | 0.0000 | [0.000, 0.340] |
| off | earlier_checkpoint | 1/13/2 | 0.4688 | [0.129, 0.808] |

| Opponent | Paired final score difference (on − off) | Conservative marginal 95% interval |
| --- | ---: | --- |
| random | +0.0000 | [-0.679, 0.679] |
| greedy | +0.0000 | [-0.679, 0.679] |
| earlier_checkpoint | +0.0312 | [-0.648, 0.710] |

| Arm | Opponent | Model Blue W/D/L (8 games) | Model Red W/D/L (8 games) |
| --- | --- | --- | --- |
| on | random | 0/8/0 | 0/8/0 |
| on | greedy | 0/0/8 | 0/0/8 |
| on | earlier_checkpoint | 0/8/0 | 0/8/0 |
| off | random | 0/8/0 | 0/8/0 |
| off | greedy | 0/0/8 | 0/0/8 |
| off | earlier_checkpoint | 1/5/2 | 0/8/0 |

Full per-colour counts, seeds, trajectory hashes, Wilson win/draw intervals and reasons are retained for every checkpoint: [on 1](evidence/on-evaluation-1.json), [on 2](evidence/on-evaluation-2.json), [on 3](evidence/on-evaluation-3.json), [on 4](evidence/on-evaluation-4.json); [off 1](evidence/off-evaluation-1.json), [off 2](evidence/off-evaluation-2.json), [off 3](evidence/off-evaluation-3.json), [off 4](evidence/off-evaluation-4.json).

### Termination, throughput and memory

Official corner/stalemate outcomes remain separate from modelling repetition and no-capture draws. The opponent rows below describe the final candidates; the linked reports retain every earlier checkpoint.

| Arm | Phase/opponent | Games | Captures | Corner | Stalemate | Repetition | No-capture limit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| on | self_play | 32 | 9 | 0 | 0 | 0 | 32 |
| on | candidate_arena | 32 | 33 | 3 | 0 | 0 | 29 |
| on | random | 16 | 15 | 0 | 0 | 0 | 16 |
| on | greedy | 16 | 43 | 16 | 0 | 0 | 0 |
| on | earlier_checkpoint | 16 | 23 | 0 | 0 | 0 | 16 |
| off | self_play | 32 | 9 | 0 | 0 | 0 | 32 |
| off | candidate_arena | 32 | 25 | 0 | 0 | 0 | 32 |
| off | random | 16 | 5 | 0 | 0 | 0 | 16 |
| off | greedy | 16 | 58 | 16 | 0 | 0 | 0 |
| off | earlier_checkpoint | 16 | 35 | 3 | 0 | 0 | 13 |

| Arm | Phase | Wall seconds | Root simulations | Simulations/s | Inferred states/s |
| --- | --- | ---: | ---: | ---: | ---: |
| on | self_play | 46.902 | 36,448 | 777.1 | 755.5 |
| on | training | 70.745 | 0 | 0.0 | 0.0 |
| on | candidate_arena | 48.368 | 42,432 | 877.3 | 853.3 |
| off | self_play | 43.705 | 36,448 | 834.0 | 810.8 |
| off | training | 68.171 | 0 | 0.0 | 0.0 |
| off | candidate_arena | 47.557 | 43,488 | 914.4 | 884.7 |

| Arm | Original positions / training-process second | Updates / optimization second | Peak train RSS MiB | Final replay entries | Compressed resident estimate MiB | Decoded upper estimate MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| on | 6.62 | 5.27 | 523.91 | 6,888 | 4.421 | 44.218 |
| off | 6.90 | 5.47 | 439.44 | 574 | 0.367 | 3.686 |

Peak RSS covers the entire training process, including replay audits and source/checkpoint persistence. Replay estimates exclude allocator overhead. No replay history was truncated. Inference throughput divides states by whole phase wall time, including compilation/export when applicable. Process times include audits, so phase totals need not equal process time. The single serial on-then-off training order is not enough to estimate timing variation or attribute every wall-time difference to augmentation; arena lengths and host load also matter.

## Validation evidence

- [124-test Intransitive gate](evidence/tests.log), [seven repository tests](evidence/repository-tests.log),
  and [eight focused accounting/optimization checks](evidence/focused-tests.log) pass.
- The [machine-readable comparison](evidence/comparison.json), [predeclared protocol](evidence/protocol.json)
  and [complete run log](evidence/comparison.log) retain the successful serial run.
  No measured training or evaluation invocation was discarded or retried.
- All eight candidates reload with finite, changed parameters and valid metadata.
  Maximum CPU/ONNX policy/value differences across their evaluations are
  **3.80e-7 / 5.44e-7**. [On](evidence/on-verification.json) and
  [off](evidence/off-verification.json) representative games reproduce exactly.
- The [artifact audit](evidence/artifact-validation.json) verifies all **154 archived
  files**, all **46 Python source hashes per arm**, checkpoint selection, initial
  parameter identity and candidate updates. The archive is **26,787,075 bytes**.
  Reloads after extraction reproduce the same games for
  [on](evidence/on-archive-verification.json) and [off](evidence/off-archive-verification.json).
- The documented [initial smoke](evidence/quickstart-initial.json) and
  [CPU continuation](evidence/quickstart-resumed.json) pass, including restored
  weights/replay. The unchanged [#15 artifact still reproduces](evidence/issue15-verification.json),
  and a real [pit checkpoint-versus-greedy invocation](evidence/pit.log) completes
  both colour assignments. These are validation games, excluded from strength counts.

The [preliminary focused test log](evidence/preliminary-focused-tests.log) retains
an incorrect test expectation (`AssertionError` instead of the existing
`require` helper's `RuntimeError`); the expectation was corrected before the full
gate and measurements. Local source, evidence, archive, documentation-link and
whitespace review was performed; no independent review is claimed.

## Protocol fixed before measurement

Both arms start from seed 150 and identical initial parameters. Four iterations
each play eight self-play and eight candidate-arena games. Both optimize with
batch 64, learning rate 0.0003, one epoch, AdamW and OneCycleLR, using exactly
**60/105/101/107 updates** (373 total, 23,872 sampled examples). These counts come
from #15, before this comparison. `batches_per_epoch` overrides only the epoch's
batch count; without it the shared wrapper retains its replay-length behavior.
Sampling is without replacement within each batch and with resampling across
batches. The identity arm therefore revisits its smaller replay more often.
Equal epochs alone would give augmentation roughly 12 times the optimization
budget and would confound this comparison.

Replay retains two complete iteration histories. Capacity is 4,800 **original
positions** per history in either arm: 57,600 augmented entries on versus 4,800
identity entries off. All actual replay bytes are measured. The identity hook
does not compute and discard the other transformations. Enabled augmentation
uses the unchanged production hook; every position must emit 12 distinct triples.
The focused tests verify IDs 0–11 against an independent full-history oracle,
with both defended-corner assignments and 1/5/31 history entries. MCTS search
identity and inference remain unchanged.

Python, NumPy and Torch initialize at 150. Self-play uses iteration seeds
150–153 and private `SeedSequence([iteration_seed, episode_id])` generators.
Before each optimizer phase all three RNGs reset to the integer produced by
`SeedSequence([150, iteration_1_based, 16])`. Torch uses deterministic algorithms
and one CPU thread. Replay shuffling remains Coach's seeded Python shuffle;
different replay lengths consume different streams. Exact optimizer seeds,
initial/final parameter hashes, and full identity-example hashes are recorded.
Equal seeds alone are not assumed to give identical self-play data: trajectory
and original board/policy/outcome/mask/Q hashes are compared explicitly.

Each of the four candidates, including rejected candidates, is evaluated in a
fresh process against random, the unchanged deterministic two-ply greedy player,
and **#15's fixed `earlier.pt`** (SHA-256
`40b8fc6eecc606e4cfaaa45b0af23cf7f5d1fc89d2ad21cfb27aa80dd0b50e14`). The earlier
opponent is never replaced with the new arm's first candidate. Sixteen games per
opponent use Blue/Red/Red/Blue assignments, eight per model colour; physical Blue
always moves first. Opponent base seeds are 151/152/153, each expanded using
`SeedSequence([opponent_seed, game_index])` into the same four streams as #15.
Search uses 32 full simulations, maximum visits and seeded random tie breaking,
with fresh trees. No evaluation result selects a checkpoint. Candidate promotion
still requires 0.6 of decisive arena games; all-draw arenas reject.

Training arms run serially, on then off, with a 600-second subprocess deadline
each. Eight isolated 48-game evaluation processes alternate arm order by
iteration; each has a 300-second deadline. Thus the experiment's maximum is
3,600 seconds (1,800 per arm). Two additional two-game fresh-process verification
invocations, each capped at 300 seconds, are validation outside that budget.
Timeouts and nonzero exits fail the run and preserve progress; they never count
as draws. Whole-process wall time includes startup, persistence and audits.
Per-iteration wall points start at Coach.learn and include self-play, training,
checkpoint writes and candidate arenas through the decision; process CPU points
include process startup. Evaluation cost is reported separately. Updates are a
controlled compute proxy, not a claim of identical total computation or time.

One paired training seed is a deliberately bounded diagnostic. Per-opponent
score intervals use Hoeffding for bounded game scores; paired on-minus-off
differences use range [-1,1]. Wilson win/draw intervals remain in raw JSON.
Intervals are marginal and conditional on these fixed models/opponents/opening,
not uncertainty across training seeds. Checkpoints reuse evaluation seeds, so
their games cannot be pooled as independent replicates. No correction is made
for multiple comparisons. Zero observed wins cannot establish zero win probability.

## Reproduction

From the repository root, use Python 3.11.4 and the pinned smoke environment:

```sh
python3.11 -m venv /tmp/intransitive-symmetry-venv
/tmp/intransitive-symmetry-venv/bin/python -m pip install -r intransitive/smoke/requirements.txt
SYMMETRY_PY=/tmp/intransitive-symmetry-venv/bin/python \
SYMMETRY_ROOT=checkpoints/issue16-reproduction \
sh intransitive/benchmarks/symmetry/reproduce.sh
```

The script requires both suites, extracts the committed #15 artifact, checks the
fixed opponent hash, runs both training arms, evaluates all eight candidates,
verifies final-candidate reload trajectories, and produces `comparison.json`.
Every output must be new. With an extracted #15 baseline, the main command is:

```sh
export ORT_DISABLE_TELEMETRY=1
python -m intransitive.symmetry_efficiency run --baseline-folder checkpoints/issue15-delivered --output checkpoints/issue16
```

Individual phases can be repeated in new directories:

```sh
python -m intransitive.symmetry_efficiency train --arm off --baseline-folder checkpoints/issue15-delivered --output checkpoints/issue16-off
python -m intransitive.symmetry_efficiency evaluate --model-folder checkpoints/issue16-off --iteration 4 --output checkpoints/issue16-off/evaluation-4
python -m intransitive.symmetry_efficiency verify --model-folder checkpoints/issue16-off --iteration 4 --output checkpoints/issue16-off/verification
python -m intransitive.symmetry_efficiency report --output checkpoints/issue16
```

`verify` exactly reproduces the first two random-opponent games, one per colour,
against that iteration's `evaluation-N/evaluation.json`, including full action
and state trajectory hashes. It also checks CPU/ONNX prediction parity on real
replay history. Different hardware/packages may change floating-point ties.
`report` checks settings, source/package parity, update counts, initial weights,
opponent/candidate hashes, original data, game counts and seed/colour pairing.

Each arm retains `initial.pt`, every `candidate_N.pt`, `temp.pt`, fixed `earlier.pt`,
`checkpoint.examples`, settings, packages, budget and a complete Python source
snapshot. `baseline.pt` follows #15's latest-accepted/final-rejected fallback
rule; its status is recorded. `best.pt` exists only after actual acceptance.
See the [operating instructions](../../README.md#setup-training-evaluation-and-human-play)
for setup, bounded smoke, continuation and human play. Continuing through the
general smoke command restores its normal all-12 augmentation; it is not an
identity-arm continuation or an extension of this fixed comparison.

## Delivered artifacts and reload checks

The committed [symmetry-artifacts.tar.gz](symmetry-artifacts.tar.gz) contains both
arms' checkpoints, replay, reports, effective configuration, pinned packages and
complete Python source snapshots. The [manifest](artifact-manifest.json) records
the archive checksum and each member's size/SHA-256. Check the archive directly
without loading a model:

```sh
python - <<'PY'
import hashlib, json, tarfile
from pathlib import Path
folder = Path('intransitive/benchmarks/symmetry')
manifest = json.loads((folder / 'artifact-manifest.json').read_text())
archive = folder / manifest['archive']
assert archive.stat().st_size == manifest['bytes']
assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest['sha256']
with tarfile.open(archive) as stream:
    assert len(stream.getmembers()) == len(manifest['files'])
    for member in stream.getmembers():
        data = stream.extractfile(member).read()
        expected = manifest['files'][member.name]
        assert len(data) == expected['bytes']
        assert hashlib.sha256(data).hexdigest() == expected['sha256']
print('Archive and every member verified')
PY
```

Extract into a new directory, then verify the final candidates independently:

```sh
export ORT_DISABLE_TELEMETRY=1
mkdir -p checkpoints/issue16-delivered
tar -xzf intransitive/benchmarks/symmetry/symmetry-artifacts.tar.gz -C checkpoints/issue16-delivered
python -m intransitive.symmetry_efficiency verify --model-folder checkpoints/issue16-delivered/on --output checkpoints/issue16-delivered/on/fresh-verification
python -m intransitive.symmetry_efficiency verify --model-folder checkpoints/issue16-delivered/off --output checkpoints/issue16-delivered/off/fresh-verification
python pit.py intransitive human checkpoints/issue16-delivered/on/candidate_4.pt -n 2 -m 32
```

The same `evaluate` command above can evaluate any delivered `candidate_N.pt`
using `--iteration N` and a fresh output directory. The off hook affects training
examples only; both checkpoints play through the standard engine and inference.
The [#15 archive](../../baselines/issue15/README.md) remains the authoritative
fixed earlier-opponent source and retains its original rejected-baseline status.

## Optional future experiments — not measured here

- Symmetry-shared MCTS: share equivalent continuation searches while preserving
  exact full-history state identity and correctly mapping actions and values.
- Inference ensembles: average inverse-mapped predictions, measuring the added
  inference cost against a comparable search/compute budget.
- Equivariant architectures: enforce symmetry structure in the model, with
  controlled parameter counts and optimization/evaluation budgets.
- Minibatch augmentation: transform examples during sampling to reduce stored
  replay, auditing transform coverage and original-position exposure over time.

None of these is implemented or established as an improvement by this report.
They are separate experiments, not explanations inferred from a small score
difference in this comparison.
