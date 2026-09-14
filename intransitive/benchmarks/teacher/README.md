# Teacher learning and training-efficiency pilot

This is the bounded experiment for issues [#33](https://github.com/lukekh/alpha-zero-general/issues/33)
and [#35](https://github.com/lukekh/alpha-zero-general/issues/35). It implements a
history-correct heuristic-teacher data path, masked target semantics, resumable
optimizer state, two self-play continuation strategies, and separate model-only,
neural-MCTS, heuristic-only, and hybrid evaluation. The result is a measured
negative result, not a claim that the short pilot produced a strong network.

## Result

The run completed all 12 trained arm/seed combinations plus three matched
untrained initializations in **72.57 wall seconds** and **419,528,704 bytes peak
RSS**, below the declared 1,800-second and 1 GiB caps.
Seeds 410, 411, and 412 were used for every arm. Scratch self-play received eight
updates, pretrained-only four, and both teacher-plus-self-play arms twelve. The
larger totals and the one-time teacher cost are reported explicitly rather than
presented as matched-compute strength evidence.

| Arm | Updates per seed | Held-out teacher top-1, mean | Tactical top-1, mean | vs greedy | vs alpha-beta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random initialization | 0 | 2.40% | 12.7% | 0/0/6 | 0/0/6 |
| Scratch + self-play | 8 | 2.92% | 18.0% | 0/0/6 | 0/0/6 |
| Teacher pretrained only | 4 | 1.98% | 15.7% | 0/0/6 | 0/0/6 |
| Teacher pretrain, then self-play | 12 | 2.68% | 14.7% | 0/0/6 | 0/0/6 |
| Teacher pretrain + 50% then 25% teacher mix | 12 | 2.92% | 17.0% | 0/0/6 | 0/0/6 |

W/D/L above combines six model-only games per arm, balanced across physical
colours and three training seeds. Each arm also played random and the fixed issue
#16 checkpoint. Scratch scored 0/6/0 against each; the annealed arm scored 1/5/0
against random and 0/5/1 against the frozen network. Seed-pair Hoeffding intervals,
colour splits, draw causes, trajectories, capture counts, move latency, and every
game are retained in `report.json` inside the artifact. These small conditional
samples do not establish a difference.

The 120 model-only games above are supplemented by 60 equal four-simulation MCTS
games against greedy and alpha-beta. Every untrained and trained arm also went
0/0/6 against each opponent with MCTS, so added neural search did not change the
bounded strength conclusion.

The selected annealed checkpoint used validation imitation accuracy only (seed
412; SHA-256 `bb9f39d02d3afe9c8a5891dccbb71b50a40391728c0f46ccc4748004431a606d`).
It has 3.39% held-out teacher top-1 and 14/100 independent tactical accuracy. It
lost both model-only games to greedy and alpha-beta. Four-simulation MCTS did not
change those outcomes. The tactical hybrid, which adds exact immediate-win and
two-ply immediate-loss filtering to the policy, drew one and lost one against
greedy; its mean candidate-move latency was 10.90 ms versus 1.02 ms model-only.
The heuristic-only player won both games against greedy at 11.09 ms. Those are
two-game deployment samples, useful for separating mechanisms but not ranking
strength.

## Dataset and target semantics

Nine complete teacher/self, teacher/random, random/teacher, teacher/greedy and
greedy/teacher games produced **192 original positions and 2,304 materialized
examples**. Every game ended under the unchanged rules (Blue won four and Red
five); no terminal state is a move-selection example. Coverage includes both
physical colours, 80 captures, 12 goal threats, five decisions avoiding an
immediate loss, and seven immediate wins.

The teacher is the #34 core-only alpha-beta player at a fully completed fixed
depth one, including its bounded two-ply proof check. It scored 100/100 on the
independent tactical suite in this run, but its outputs remain imitation targets,
not ground truth. The stored policy is a one-hot selected legal action. Value is
the eventual exact terminal outcome in the canonical current-player order.
Teacher heuristic scores never become outcomes. Teacher Q has a zero storage
placeholder and an all-false Q mask, so it contributes no value loss.

Trajectory IDs are assigned to train/validation/test before augmentation. Shared
opening prefixes and any other equal symmetry orbit are then owned by one split
and dropped from the others. The verifier proves disjoint trajectory and orbit
sets. Every retained original is expanded with IDs 0–11; state, complete history,
policy and legal mask are transformed together. The deterministic compressed
dataset is 756/840/708 examples across train/validation/test. Its SHA-256 and all
trajectory actions, terminal reasons, final-state hashes, teacher work, target
construction and drop counts are in `dataset-manifest.json`.

## Exploration and plateau diagnosis

The paired root-noise pilot held the initial weights, four MCTS simulations,
temperature, seeds and game count fixed. No-noise produced one capture and 114
unique positions; automatic Dirichlet noise produced zero captures and 90 unique
positions. Both arms produced three distinct trajectories and zero decisive games.
Entropy rose from 0.995 to 1.079, but this tiny sample shows neither useful data
diversity nor strength, so root noise is deferred.

This agrees with the previously frozen plateau diagnosis rather than replacing
it. The issue #33 run observed one accepted candidate followed by 87 rejected
all-draw comparisons and a replay window whose outcome labels were all draws.
`Coach.learn()` restores the incumbent after rejection, so those updates do not
accumulate; the default wrapper also recreates AdamW and OneCycleLR each iteration.
The new continuation path is opt-in: `persist_optimizer=true` saves AdamW moments
and the cumulative update count, reloads them before the next phase, and uses a
constant learning rate because an already-consumed OneCycle horizon cannot safely
be extended. The established recreate behaviour remains the default and its
regression tests remain unchanged.

## Recommendations

- **Adopt** masked value/Q targets and opt-in persistent AdamW checkpoints for
  explicitly labelled continuous-learner experiments.
- **Reject** this pretrained-only configuration as a replacement for the current
  baseline. It did not improve held-out imitation or model-only strength.
- **Defer** teacher + self-play and annealed mixing. Neither transferred to a
  repeatable fixed-opponent gain; more updates/data should be tried only in a
  focused follow-up with a matched-compute design.
- **Defer** root noise based on this pilot. It changed entropy but not decisive
  outcomes and reduced the measured unique positions/captures.
- **Defer** the tactical hybrid for deployment. It prevents immediate blunders by
  construction but did not establish strength and costs more per move.
- **Keep** #34's heuristic-only core preset as the stronger playable opponent.
  Search throughput work remains independently scoped in #37–#41.

No draw was relabelled, no reward shaping or adjudication was added, and the
official rules plus existing modelling-only limits are unchanged.

## Reproduce and inspect

Run the full protocol in a fresh directory:

```sh
uv sync --frozen
TEACHER_ROOT=checkpoints/issue33-reproduction \
  sh intransitive/benchmarks/teacher/reproduce.sh
```

Or run and verify the experiment directly:

```sh
uv run --locked python -m intransitive.teacher_learning run \
  --output checkpoints/issue33-reproduction
uv run --locked python -m intransitive.teacher_learning verify \
  --output checkpoints/issue33-reproduction
```

The committed artifact contains the exact dataset, selected checkpoint, complete
machine-readable report, requirements and source snapshot:

```sh
mkdir -p checkpoints/issue33-published
tar -xzf intransitive/benchmarks/teacher/teacher-artifacts.tar.gz \
  -C checkpoints/issue33-published
uv run --locked python -m intransitive.teacher_learning verify \
  --output checkpoints/issue33-published
uv run --locked python pit.py intransitive \
  checkpoints/issue33-published/selected.pt greedy -m 4 -n 2
uv run --locked python -m intransitive.play \
  --checkpoint checkpoints/issue33-published/selected.pt --simulations 4
```

`evidence/summary.json` is a compact review copy. `report.json.gz` retains the
complete report without extracting the artifact. The source snapshot is the
authority for the measured run; its recorded Git revision identifies the base
commit while per-file hashes identify the experiment sources.

The repository validation baseline is 301 Intransitive tests: 296 pass and the
five already documented defensive search regressions remain active failures
(original Red moves 14, 18 and 31, plus the simplified last-defender position in
both colours). No assertion is weakened or marked expected-failure.
