# Variable material versus flat material

Measured on 17 September 2026 on the authorized shared machine, at low priority.
This compares the user's [variable piece-value formula](../../heuristics/VARIABLE_MATERIAL.md)
with the current adopted flat-material heuristic. **The only candidate difference
is `variable_material_enabled`.** Both use material 100, advantage
23.967050360966205, attack 25.714516982666414, defence 32.5643023919054,
and disabled overload/pressure. There is no coefficient evolution in this test.

**This test does not establish an improvement over the original flat-material
heuristic.** Official results favour flat material at depths one and two, while
the deeper/Monte Carlo results need to be read with unresolved games and the
small sample in mind. The new mode remains selectable; flat material stays the
default. Results below are from **variable material's** perspective.

| Setting | Official wins / losses / unfinished | After consensus wins / losses / inconclusive | Main wall (s) |
| --- | --- | --- | ---: |
| minimax-d1 | 6 / 7 / 3 | 8 / 8 / 0 | 55.17 |
| minimax-d2 | 5 / 9 / 2 | 6 / 9 / 1 | 100.16 |
| minimax-d3 | 7 / 7 / 2 | 8 / 7 / 1 | 270.82 |
| mcts-s128 | 5 / 7 / 4 | 6 / 9 / 1 | 149.03 |
| mcts-s32 | 3 / 4 / 9 | 5 / 8 / 3 | 91.23 |

Official paired 95% intervals (including uncertainty from unresolved games):

- minimax-d1: [0.0, 1.0]
- minimax-d2: [0.0, 0.9176613956599604]
- minimax-d3: [0.0, 1.0]
- mcts-s128: [0.0, 1.0]
- mcts-s32: [0.0, 1.0]

No method/depth totals are pooled into a single win rate. Per-opening and
per-colour outcomes, actual completed depths/simulations and work are available
in [comparison.json](comparison.json).


## Design

Implementation is frozen at `2228bc6305bc0d5e73dedbf5a44f24bcd49d4a02`.
All strength games use Python. Rust implements the same mode and is separately
covered by formula/parity/search tests; this is not a Rust tournament result.

The same **eight fresh legal eight-piece endgames** are used at every setting,
each with colours swapped: 16 games per setting, 80 main games total. Two
capture-biased generation batches, seeds 1601 and 2601, produce 80 lines each
with up to 240 plies per line. Validation lines that reach 12 pieces are sorted
by state hash and legally continued, with a 160-move ceiling, to eight pieces.
The first four distinct symmetry families from each batch are frozen before any
candidate game. Their line seeds are 1643, 1680, 1652, 1675, 2616, 2669, 2657
and 2639. Held-out lines are not used. The full histories, discarded continuation
attempts and original splits are archived.

Minimax depths are 1, 2 and 3. MCTS uses 32 or 128 simulations, uniform legal
priors, `tanh(heuristic / 400)` leaf values, `cpuct=1`, no noise and a fresh tree
per move. A simulation budget is not a fixed tree depth; observed depths are
recorded separately. Every move has eight seconds and 20 million logical work
units, with proof depth/tokens 2/64. Games stop after 192 plies or 30 active
seconds. Each setting has a 580-second wall ceiling and a 16-game limit. A
supervisor enforces a 3500-second deadline plus bounded cleanup within the
user's one-hour allocation. Per-setting reserved-work ceilings are derived from
all scheduled games and their per-move caps, and saved in `budget.json`.

Before the main games, real-engine smoke checks play two games per mode at a
two-ply ceiling, 5 million work/four seconds per move, with proof disabled and
four MCTS simulations. These exercise execution, not playing strength. Only
one game runs at a time, using at most two resident candidate engines.

## Official and consensus outcomes

Only exact official terminal outcomes count as official wins. Unfinished games
remain unresolved. Following the user's instruction, a separate consensus
analysis scores each unfinished final board with **both competing effective
configurations**, from the same player-zero perspective. Two strictly positive
scores award player zero a consensus win; two strictly negative scores award
player one. Disagreement, ties or evaluation-budget exhaustion are inconclusive.
Static adjudication disables tree/proof search and has two seconds / five million
work units per evaluator. Crashes, illegal moves and incomplete requested search
remain excluded. Original game journals and official reports are retained.

Inconclusive games are excluded from the alternative decided-game win fraction;
conservative wins over all scheduled games are also reported. Consensus is an
agreed evaluation convention, not an independent oracle. Two related heuristics
can agree and both be wrong. No defaults are automatically replaced.

## Interpretation limits

This is an endgame comparison under fixed weights and compute limits. It does
not establish performance in openings, midgames, long unrestricted games, or
with separately tuned weights. Repeated positions across methods are not
independent samples; results are reported by method/depth rather than pooled.
The harness's 95% uncertainty intervals use paired outcomes clustered by the
opening-generation line seed and retain unresolved outcomes. Eight lines leave
substantial uncertainty, especially when differences are small.

## Measurements and audit

The full supervised corpus generation, smoke, 80-game comparison, replay audit
and consensus pass took **12m22.91s** (742.91 seconds).
The supervisor used nice 19 and OPENBLAS/OMP/MKL thread counts of one. Sampled
peak descendant RSS was **372.6 MiB**, at most
4 processes, sampled every three seconds. Observed priorities:
`[19]`; priority anomalies: **0**.
These are shared-host measurements, not isolated throughput estimates.
All observed experiment descendants were absent at cleanup. There were no
incomplete-depth/simulation rejections, crashes, illegal moves or infrastructure
timeouts. All 80 main games were attempted.

All **84 records / 6042 moves** passed replay and exact report
reconstruction. Played moves completed their requested search. Recorded outcome
counts, including the four smoke games, were `{'unfinished': 24, 'win': 60}`. The separate
consensus analysis classified them as `{'consensus': 18, 'official': 60, 'inconclusive': 6}`. Evaluator completion
statuses: `{'complete': 48}`. These accounting totals include repeated starts
and smoke games and are not an aggregate strength result.

[Initial-board evaluations](initial-evaluations.json) record raw module terms and
MCTS value calibration for the frozen positions. They were computed after the
games, without selecting positions or changing settings in response. Neither
mode clipped on the eight initial boards. With the fixed MCTS value scale of
400, variable material produced `abs(tanh(score / 400)) >= 0.99` on **3/8**
initial boards, versus **0/8** for flat material. This identifies a calibration
difference worth separating from move quality; it does not prove the cause of
the MCTS results.


## Verification and reproduction

Before this benchmark, 62 focused Python tests passed, followed by the expanded
nine-test variable-material suite (two additional cases, for 64 distinct checks).
Coverage includes the exact formula and cyclic piece types, extinction/zero
counts, rare-type balance, reference/cached scores, captures and parent-state
restoration, decisive-score precedence, clipping, Minimax/MCTS integration,
configuration identity, consensus mode handling and native parity in both player
orientations. Rust's ten tests and Clippy with warnings denied also passed.

- [Frozen design and candidates](design.json)
- [Results by setting, side and opening](comparison.json)
- [Consensus scores and alternative rankings](consensus.json)
- [Replay/report audit](verification.json)
- [Resource measurements](measurements.json), [cleanup](cleanup.json)
- [Full archive](runs.tar.gz), [SHA-256](runs.sha256): manifests, journals,
  complete reports, corpus histories, scripts, tests and monitor logs


The archived `study.py` recreates the frozen corpus, runs the paired games and
audits/consensus-scores the journals. Use its `supervise.py` with the recorded
implementation, after changing the original `/tmp/issue55-variable-material-20260917`
output root if needed. A new run requires its own bounded allocation and
available or explicitly authorized shared resources. Earlier archived studies
must be read with their recorded revisions because evaluator fingerprints differ.


## Depth-four follow-up

A [Minimax depth-four comparison](depth4/README.md) reused these eight openings
with larger search/game limits. Variable material scored 6–6 officially, with
three unfinished games and one incomplete-depth failure; after consensus it
scored 6–7, with two inconclusive games and the failure excluded. A separate
larger-budget retry of the affected opening in both colours completed without
failures and split 1–1 after consensus. This provides no clear evidence of an
improvement; the different caps also prevent attributing changes solely to depth.


## Pruning follow-up

After merging master, the [depth-four pruning comparison](depth4-pruning/README.md)
played the same eight openings with PVS enabled in both arms. With pruning on or
off, variable material scored 6–7 officially with three unfinished, and 7–7 with
two inconclusive after consensus. All searches used for moves completed their
required depth. Experimental pruning produced 470 verified null-move cutoffs but
no futility skips; it did not improve the match score. Opening diagnostics and
work counters are retained separately.


## Material scale and MVV-LVA follow-up

The [depth-four scale/ordering comparison](scaled-mvv-lva/README.md) crossed
full variable material versus one-twentieth scale with MVV-LVA off/on, against
flat material on four reused colour-swapped openings. On the seven pairs
attempted by all settings, consensus results were identical: 3 wins, 2 losses,
2 inconclusive. The three commonly completed decisive games also had identical
winners. No clear improvement emerged. All 31 records / 1,057 moves passed
replay, and the low-priority run and cleanup finished within 56 minutes.
The optional coefficient-5 preset gives the requested BASE/20 effect; defaults
remain unchanged.
