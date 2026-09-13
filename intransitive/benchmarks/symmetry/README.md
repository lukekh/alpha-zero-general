# Controlled symmetry efficiency comparison (#16)

This experiment compares materializing all 12 symmetries (`on`) with retaining
only the identity example (`off`). Both use the real Coach, v1 network, official
Blue-first opening, exact repetition history, modelling draws, 32 full search
simulations, and #15's fixed random/greedy/earlier-checkpoint opponents.

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
