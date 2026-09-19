# Evolutionary tuning with variable piece values

Merged master `2ac15df` (move-generation and search hot-path improvements).
The experiment uses implementation `c20fdb31a03eb82f98ef93eaa8e1648f6b1502bb`. Variable v3 genomes keep
variable material enabled through initialization, mutation, crossover,
preflight, native export, match/cache identity and checkpoint resume.
The existing flat v2 format remains supported. No application defaults changed.

**No replacement for the flat default was established.** The primary eligible
candidate lost all eight decided games in its interrupted depth-two screen.
The two completion-rejected candidates received separate fresh depth-two checks.
Raw A scored 2–5 with 1 inconclusive against each baseline after consensus.
Raw B scored **5–2 with 1 inconclusive against the scaled-variable baseline**, but
**2–5 with 1 inconclusive against flat defaults**. Its official results were
3–2–3 and 1–5–2 respectively (W–L–unfinished). This is a small, post-hoc signal
worth retaining, not a validated replacement. Raw B's full-precision
[diagnostic genome](diagnostic-raw-b-genome.json) remains provisional.

Depth four supplied no official wins or losses: primary had four incomplete
searches and four inconclusive unfinished games. Raw A had three incomplete
searches, three unfinished games, one interrupted game and one unstarted game;
consensus gave 2–1 with 1 inconclusive among eligible prefixes. These tiny,
truncated results do not establish depth-four strength.

## Selected candidate

Selection was **eligible in optimizer search and development validation**. This is not held-out acceptance.

| Coefficient | Scaled baseline | Selected |
| --- | ---: | ---: |
| advantage | 23.9670504 | 69.7326144546 |
| attack | 25.714517 | 42.3376227762 |
| defence | 32.5643024 | 29.3111788191 |
| material | 5 | -36.6972817723 |
| overload | 0 | 82.7175611082 |
| pressure | 0 | 13.1401166692 |

The [full-precision genome](selected-genome.json) is provisional. The material
coefficient multiplies army totals normalized by BASE=100, so 5 corresponds
to the requested BASE/20 scale. REG=0.25 and feature geometry remain fixed.

The separate follow-up candidates retain material 5 and alter the other coefficients:

| Coefficient | Raw A | Raw B |
| --- | ---: | ---: |
| material | 5 | 5 |
| advantage | 23.967050361 | 33.8161201879 |
| attack | 25.7145169827 | 25.7145169827 |
| defence | 32.5643023919 | 32.5643023919 |
| overload | 17.0887000884 | 9.0153495528 |
| pressure | 11.9620645102 | 14.4748455964 |

All three diagnostic genomes enable overload, which the Rust backend does not
implement. These are Python genomes; do not silently drop that coefficient for
a native run. Flat application defaults remain compatible with both backends.

## Fresh deeper checks

Every result below is from the candidate named by the setting. The original
screens used the primary selected candidate. `focused/raw-a` and `raw-b` are
separate post-hoc diagnostics of the two completion-rejected 5–1 candidates;
`focused/primary` retains the original selection. Exact genomes are in
[focus-design.json](focus-design.json). Opponents are
the adopted flat default and the untuned coefficient-5 variable baseline.
The original depth-two and depth-four screens use separate fresh development-validation lines,
selected before training; their outcomes cannot isolate a pure depth effect.
These checks evaluate transfer of depth-one weights, not independently optimized
depth-specific weights. Held-out positions are never played.

| Setting | Opponent | Official W–L–unfinished | Failures | Interrupted | Unstarted | Consensus W–L–inconclusive |
| --- | --- | --- | ---: | ---: | ---: | --- |
| screen-depth2 | flat-default | 0–3–3 | 0 | 0 | 2 | 0–3–3 |
| screen-depth2 | scaled-variable | 0–5–1 | 0 | 1 | 1 | 0–5–2 |
| screen-depth4 | flat-default | 0–0–0 | 0 | 0 | 4 | 0–0–0 |
| screen-depth4 | scaled-variable | 0–0–0 | 0 | 1 | 3 | 0–0–0 |
| focused/primary-depth4/screen-depth4 | flat-default | 0–0–2 | 2 | 0 | 0 | 0–0–2 |
| focused/primary-depth4/screen-depth4 | scaled-variable | 0–0–2 | 2 | 0 | 0 | 0–0–2 |
| focused/raw-a-depth2/screen-depth2 | flat-default | 2–4–2 | 0 | 0 | 0 | 2–5–1 |
| focused/raw-a-depth2/screen-depth2 | scaled-variable | 2–2–4 | 0 | 0 | 0 | 2–5–1 |
| focused/raw-a-depth4/screen-depth4 | flat-default | 0–0–2 | 1 | 0 | 1 | 1–0–1 |
| focused/raw-a-depth4/screen-depth4 | scaled-variable | 0–0–1 | 2 | 1 | 0 | 1–1–0 |
| focused/raw-b-depth2/screen-depth2 | flat-default | 1–5–2 | 0 | 0 | 0 | 2–5–1 |
| focused/raw-b-depth2/screen-depth2 | scaled-variable | 3–2–3 | 0 | 0 | 0 | 5–2–1 |

Consensus preserves official wins. For a nonempty unfinished/interrupted game,
both competing static evaluators must agree on a strict nonzero sign for the
final board. Disagreement/ties are inconclusive. Failed searches and empty or
missing games are excluded. Consensus can adjudicate interrupted prefixes, so
the interruption column is not additive to consensus totals. Official journals,
evolutionary fitness and eligibility are not rewritten. Different candidate
weights also change adjudication, so read both outcome measures.

## User-requested compute reallocation

At the user's explicit request during the original depth-two screen, SIGTERM
was sent to the competing dataset generator and the benchmark's active screens.
The generator confirmed `stopped`, with 254,430 saved positions, 3 pending seeds,
and no worker processes. The original depth-two screen retained 13 attempted
games (8 losses, 4 unfinished, 1 interrupted), with 3 never started; the original
depth-four screen retained one interrupted attempt. These are user-interrupted
results, not ordinary wall-cap exhaustion. Their audit completed before new work.

Four independent validation jobs then ran at normal priority 0, with one active
search and at most 3 resident bots per job. Raw A/B were selected post hoc from
seed 171 generation 0's completion-rejected candidates by wins/scheduled,
completion and stable hash, and received four further fresh depth-two validation
lines against both baselines. Primary and Raw A were checked at depth four on
the two originally reserved positions. No held-out lines were played.

Each focused job had at most 650 seconds, further limited by the original
3400-second game deadline. Numerical-library threads remained one. These results
form a separate compute phase and are not pooled with earlier time-capped games.
The initial focused launcher resolved the virtual-environment symlink to the
system interpreter, failed to import Numba before any game, and was corrected;
its failed-launch logs are retained in the archive. The focused workers then
flagged a report-reconstruction mismatch in summed CPU seconds (~2e-13 seconds)
because filesystem order differed from play order. A read-only recovery replayed
every journal in the declared interleaved play order and matched all reports
exactly. Original worker exit codes and the [audit recovery](focus-audit-recovery.json)
are preserved; no game was rerun or outcome changed. Original search rankings,
eligibility and selected genome are unchanged.

## Evolution versus random search

Each seed shares the same initial population between algorithms. Random search
draws fresh populations after that initialization, without elite retention or
parent selection. Both algorithms have identical game/work/wall ceilings,
opponent policy and corpus. Actual work and cache reuse differ. Repeated
validation and cached matches are correlated, not independent extra samples.

| Seed | Algorithm | Status | Completed generations | Unique matches | Eligible exports | Wall seconds |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 171 | evolution | wall_budget | 2 | 75 | 1 | 550.19 |
| 171 | random | wall_budget | 2 | 59 | 1 | 550.11 |
| 172 | evolution | wall_budget | 2 | 55 | 0 | 550.15 |
| 172 | random | wall_budget | 2 | 57 | 0 | 550.07 |

The [comparison](comparison.json) retains per-generation official rankings,
opening-clustered intervals, diversity, failures, reservations, actual work,
CPU and wall time. It does not establish a general advantage for evolution.
The [selection record](selection.json) includes all evolutionary exports used
for ranking, eligibility and provenance; random-search exports were excluded
from selection of the evolved candidate. Eligibility precedes unique validation
candidate-game wins/scheduled, completion and stable config identity. When no
export qualifies, the best available export is explicitly diagnostic only.
The selected genome is frozen before deeper checks. The sole eligible genome
came from the shared initial population (generation 0), and was also exported
by the matching random control. No newly bred candidate passed both eligibility
stages in these runs; this is not evidence that evolution improved on random
search.

The completion gate materially affected selection. In seed 171's first
population, two candidates scored 5–1 with 2 unfinished games, but their 75%
completion fell below the fixed 80% threshold. The eligible candidate scored
2–6 with all games finished. Therefore the selected export is the best under
this constrained eligibility rule, not the candidate with the highest raw
training win rate and not evidence of optimal coefficients. Short game caps
and one common training opening limit what this search can establish.

## Frozen search design

All six coefficients range independently over **−100 to 100**, including
material. Variable material is fixed on for every population member. Initial
near-baseline candidates start from material 5 and the adopted other weights;
uniform immigrants still sample the full interval. BASE 100, REG .25, internal
advantage bonuses, clipping and proof rules are fixed.

Population 3, up to 3 generations, one elite, tournament size 2, mutation probability .7,
additive Gaussian sigma .05 of the interval width (10 coefficient units),
conditional zero-toggle .1, crossover .5, random zero atom .2, near-baseline
fraction 2/3, and hall size 2. One common search position and one fresh validation
line per generation are frozen by stable hash. Fixed opponents are flat adopted
defaults and the scaled-variable baseline; validation-eligible history joins
the opponent slate. Fitness remains official wins/scheduled after eligibility,
with unfinished games worth zero and failures disqualifying candidates.

Corpus seed 3457 generates 120 capture-biased legal lines up to 240 plies; retain
the first endgame (at most 12 pieces) from each line with its full legal history,
line split and symmetry identity. Two seeds 171/172 each run evolution and random
search. Each arm has 550 seconds, 120 attempts and 100 billion reserved logical
work units. There is one active game, at most 4 resident candidate processes.
The matching smoke allows 180 seconds and 16 two-ply games with 50,000 work/move.
Its work-limited results are machinery checks, never strength evidence.

Training uses Minimax depth 1, 3 seconds / 3 million work per move, 12 active seconds
per game. Fresh checks use depth 2 with 10 seconds / 30 million work per move and 30
active seconds per game (4 openings,16 scheduled games,600-second arm cap), and
depth 4 with 30 seconds / 100 million work per move and 90 active seconds per game
(2 openings,8 scheduled games,400-second cap). All main games cap at 192 plies.
PVS, MVV-LVA, NMP/futility and the explicit experimental-evaluator option are
fixed on for all candidates; the latest master's compiled ordering is retained.
Proof budgets are 2/64; selective mate-range scores must complete requested depth.
Depth 1 cannot exercise NMP's minimum depth 3. Actual pruning/ordering counters
are in [summary.json](summary.json); enabled flags alone do not prove activity.

## Verification and resource bounds

All **323 records / 26,554 played moves** passed replay and saved-evidence checks.
Every move used a completed requested search or an unpruned proven result;
incomplete requested searches are retained as failures, never played.

Supervised elapsed **2660.09s**, cleanup check **3432.69s**
after launch. Initial-phase peak sampled process-tree RSS **567.7MiB**;
priorities `[0, 19]`, **1 anomaly**.
The monitor retained one transient exiting zero-RSS process reported at nice 0;
it vanished before correction. Initial-phase active workers inherited nice 19. See measurements
for the original process record and attempted correction.

Focused phase elapsed **562.12s**, ending **3406.75s** after original launch; peak sampled RSS **1271.4MiB**, priorities `[0]`.

BLAS/OMP/MKL limits were one. The initial phase ran one active search; the
focused phase ran four independent searches. Both supervisors retain the original
3500-second deadline plus bounded cleanup inside the user's one-hour allocation.
No observed experiment descendants remained at cleanup. The dataset generator
and its workers were stopped at the explicit user request described above; their
data was preserved and binaries were untouched.

Validation passed 106 distinct Python checks covering evolution/resume/cache,
variable genomes, evaluator/native parity, MVV-LVA, merged bitboards and search
optimizations, selective pruning and search windows. Rust passed 15 tests with
one manual benchmark ignored; Clippy passed with warnings denied. Native parity
was rerun against the rebuilt local release binary after the master merge.

## Evidence and reproduction

- [Design](design.json), [comparison](comparison.json), [summary](summary.json)
- [Selection](selection.json), [consensus](consensus.json)
- [Replay verification](verification.json)
- [Initial measurements](measurements.json), [focused measurements](focus-measurements.json), [cleanup](cleanup.json)
- [Compute reallocation](focus-design.json), [stop signals](user-stop-request.json), [generator shutdown receipt](shutdown-receipt.json)
- [Complete run archive](runs.tar.gz), [SHA-256](runs.sha256)

The archive includes full manifests, optimizer checkpoints, per-game journals,
reports, corpus histories, test logs and reproduction scripts. Use its recorded
implementation, change the original `/tmp/issue55-variable-evolution-20260917`
output root, and run `study.py` through `supervise.py` under a bounded allocation.
The separate `focus.py` controller launches the four later validation jobs; its
original-launch deadline and frozen `focus-design.json` must be retained when
reproducing that phase. `recover_focus_audit.py` performs read-only replay and
report reconstruction without rerunning games.
Source/runtime fingerprints intentionally reject stale resumes and earlier
archives require their original code revisions.
