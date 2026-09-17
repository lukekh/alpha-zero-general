# Signed coefficients across Minimax depths and heuristic MCTS

This follow-up on 17 September 2026 searches all six Python coefficients in
**[−100, 100], including material**, under the user's one-hour, low-priority
allocation. It compares Minimax depths 1/2/3 and heuristic MCTS with 32/128
simulations. The user separately instructed adoption of the prior measured
candidate in both Python and Rust; that change is already implemented:

| Material | Advantage | Attack | Defence | Overload | Pressure |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 23.967050360966205 | 25.714516982666414 | 32.5643023919054 | 0 | 0 |

Rust now has matching static attack/defence routes and signed material scaling.
It supports five coefficients; overload remains unsupported. All games in this
new study use **Python**, so the native implementation is covered by parity
tests, not a Rust strength tournament. Running teachers were not restarted or
reconfigured.

**No stronger replacement was established. Keep the adopted defaults.** The
same challenger A was selected at Minimax depth one and MCTS32; the deeper
Minimax selections used B, but were training-only or ineligible. Neither
MCTS128 seed completed an export. These differences do not establish optimal
weights that vary with depth.

| Vector | Material | Advantage | Attack | Defence | Overload | Pressure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 62.641359 | 23.967050 | 25.714517 | 18.275587 | 52.472203 | −84.631747 |
| B | −56.672198 | 53.275799 | 0 | −10.287148 | −39.597764 | 92.016227 |

Values above are rounded for display; exact genomes are in the archive and
[comparison](comparison.json). Primary fresh-screen outcomes below are from the
challenger's perspective. Consensus results use the user's later requested rule.

| Main setting | Vector | Official W / L / unfinished | After consensus W / L / inconclusive | Limit |
| --- | --- | --- | --- | --- |
| Minimax depth 1 | A | 0 / 7 / 1 | 0 / 8 / 0 | Four fresh paired lines |
| Minimax depth 2 | — | — | — | No completed export before wall cap |
| Minimax depth 3 | B | — | — | All eight games rejected for incomplete depth |
| MCTS 32 simulations | A | 0 / 3 / 5 | 0 / 5 / 3 | Diagnostic: no eligible training export |
| MCTS 128 simulations | — | — | — | No completed export before wall cap |

Both depth-one and both depth-three arms completed two generations. Each MCTS32
arm completed one; depth two and MCTS128 completed none. Generation completion
is not search eligibility: depth three's generations contain mostly rejected
moves. Observed MCTS tree depths across the optimizer runs were 1–8 at 32
simulations and 1–9 at 128; every played MCTS move completed its full requested
simulation count.

| Arm | Completed generations | Configurations¹ | Games | Official finishes² | Unfinished | Incomplete depth | Cancelled | Work (million) | Known child CPU (s) | Wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| minimax-d1-91 | 2 | 5 | 32 | 25 | 7 | 0 | 0 | 141.42 | 129.41 | 183.51 |
| minimax-d1-92 | 2 | 5 | 32 | 28 | 4 | 0 | 0 | 102.82 | 118.52 | 160.07 |
| minimax-d2-91 | 0 | 4 | 13 | 7 | 5 | 0 | 1 | 214.14 | 218.44 | 240.07 |
| minimax-d2-92 | 0 | 4 | 10 | 2 | 7 | 0 | 1 | 185.52 | 212.48 | 240.09 |
| minimax-d3-91 | 2 | 5 | 28 | 0 | 2 | 26 | 0 | 217.41 | 195.72 | 213.46 |
| minimax-d3-92 | 2 | 5 | 28 | 0 | 2 | 26 | 0 | 210.08 | 193.58 | 208.60 |
| mcts-s128-91 | 0 | 4 | 10 | 2 | 7 | 0 | 1 | 177.01 | 219.59 | 240.06 |
| mcts-s128-92 | 0 | 4 | 9 | 1 | 7 | 0 | 1 | 133.46 | 202.89 | 240.09 |
| mcts-s32-91 | 1 | 5 | 22 | 4 | 17 | 0 | 1 | 214.91 | 192.77 | 240.12 |
| mcts-s32-92 | 1 | 5 | 21 | 7 | 13 | 0 | 1 | 223.11 | 193.38 | 240.17 |

¹ Includes two fixed reference configurations. Only **nine distinct registered
configurations** occur across all ten arms; shared initialization and short runs
make this a very sparse sample of the permitted six-dimensional box. Recorded
per-gene ranges are in `comparison.json`; permitting −100…100 does not mean this
small search thoroughly covers that range.

² Official finishes count either side winning, not challenger wins. The ten arms
contain 205 journal records, 195 distinct full match identities and 147 distinct
physical trajectories. Timing changes can produce distinct records for the same
logical match across arms; these are not independent evidence. Complete
per-generation populations, diversity, preflights, rankings, uncertainty and
per-opponent/colour statistics remain in the archive.


## Frozen design and limits

The main study uses implementation commit `9424a35`; subsequent commit
`ef40662` changes regression tests only. Each manifest records backend and
optimizer source hashes, including the root MCTS implementation. The v2 genome
has six signed coefficients. Optional zero terms are disabled; nonzero values
activate them. Mutation adds a Gaussian step with standard deviation 30
(0.15 times the 200-wide range), clips to bounds, and can cross zero. Mutation
probability is 0.7, explicit toggle probability 0.1, and uniform initialization
has a 0.2 off probability. Material is tunable in initialization and mutation.

Each of five settings runs seeds **91 and 92**, serially, with population two,
two planned generations, one common search line, and one fresh optimizer
validation line per generation. Half the initial population is near the adopted
default and half is uniform. Each arm receives at most 240 seconds, 64 attempted
games and 80 billion reserved work units. Four resident engines amortize startup;
only one game runs at a time. There is no random-search control in this expanded
allocation. It cannot establish evolution's superiority over random search.

The corpus is generated from seed 901 across 120 capture-biased legal lines,
with at most 240 plies per line. The first nonterminal position with at most
12 pieces is retained per line. Original search/validation/held-out splits are
preserved. Four further validation lines are frozen before games for the final
paired screens. Held-out games are not used.

Main Minimax searches require depth 1, 2 or 3. MCTS uses the repository's PUCT,
uniform legal priors, `tanh(heuristic / 400)` leaf values, `cpuct=1`, no noise,
and a fresh tree per move. MCTS simulation count is **not fixed search depth**;
actual maximum visited depth is logged. Each move has four seconds and five
million logical work units; proof depth/tokens are 2/64. Games allow 192 plies
or 12 active seconds. Incomplete requested Minimax depth or MCTS simulations are
explicit rejections; unfinished games are unresolved, never synthetic draws.

Before the ten main arms, two bounded real-engine smokes exercise each mode.
The Minimax smoke's tiny 50,000-work ceiling rejects all 16 games for incomplete
depth; all 16 four-simulation MCTS smoke games are unfinished at the two-ply cap.
These check execution and rejection paths, not strength.

Final-screen selection considers completed exports, prefers a candidate with
any eligible training export, then ranks its direct-default validation wins over
all unique full match keys, completion, fewer optional modules and stable hash.
All validation observations are retained, including from ineligible exports.
If no training-eligible candidate exists, the screen is diagnostic only; if
there is no completed export, the main screen is absent. Each main screen has
130 seconds for eight games on the same four fresh lines, with both colours.

## Remaining-budget diagnostic

After the main study, the remaining allocation was used for one common unused
validation line (seed **915**), legally continued to **eight pieces** without
consulting candidate outcomes. The continuation is bounded to 160 moves per
source line; the full original history and split are preserved. The choices
were frozen before these games: A at depth one/MCTS32; B at depth two from its
completed seed91 training ranking and at depth three from the main diagnostic
selection. With no MCTS128 export, A was transferred from MCTS32 without retuning.

Each setting has two games with colours swapped, at most 100 wall seconds,
eight seconds / 20 million work units per move, and 30 active game seconds /
192 plies. These are different caps and a different corpus from the primary
screens; their results are not pooled with those screens.

| Diagnostic setting | Vector | Official W / L / unfinished | After consensus W / L / inconclusive |
| --- | --- | --- | --- |
| Minimax depth 1 | A | 0 / 1 / 1 | 0 / 2 / 0 |
| Minimax depth 2 | B | 1 / 1 / 0 | 1 / 1 / 0 |
| Minimax depth 3 | B | 0 / 1 / 1 | 1 / 1 / 0 |
| MCTS 32 simulations | A | 0 / 1 / 1 | 0 / 1 / 1 |
| MCTS 128 simulations | A (transfer) | 0 / 1 / 1 | 0 / 2 / 0 |

All ten diagnostic games used completed requested searches; none was rejected
for incomplete depth/simulations. One paired opening cannot establish a strength
gain or a depth-specific optimum. Its 95% paired interval is [0, 1] at each setting.

## Consensus adjudication requested during the run

Both **competing weight configurations** score the same final legal board from
player zero's perspective, using static weighted evaluation with tree/proof
search disabled. If both scores are strictly positive, player zero wins by
adjudication; if both are negative, player one does. Ties, disagreement and
budget exhaustion stay inconclusive. Scores use a two-second / five-million-work
cap per evaluator. Official wins are retained. Nonempty legal prefixes stopped
by the game or overall wall cap may be adjudicated; crashes, illegal moves,
incomplete-depth failures and empty cancellations are excluded.

All **271 records** were analyzed, including the smokes for audit completeness:
92 official wins, **56 consensus wins**, 44 inconclusive and 79 excluded. Ten
consensus wins are two-ply smoke artifacts, leaving **46 additional non-smoke
adjudications**. This is an accounting total, not a pooled strength estimate.
All score pairs, original statuses, final-state hashes and alternative rankings
are in [consensus.json](consensus.json). No evaluator exceeded its scoring budget.

The alternative rankings exclude inconclusive games from the decided-game win
fraction, with conservative wins/scheduled also shown. Re-ranking the 11
completed training batches changes the first-ranked genome in two batches;
[full details](consensus-training-ranking.json) retain the original ranking.
This analysis does **not** recreate the breeding sequence of an optimizer that
used consensus from the start. Journals, original fitness, eligibility and
exports were not rewritten; no further defaults were adopted. Agreement between
related heuristics is an adjudication convention, not proof of the official
winner or an independent held-out assessment.


## Interpretation

This is a small, compute-bounded search, not an estimate of optimal weights.
A few population members, one common training line, shared initial populations,
reused trajectories and timing-dependent truncation limit inference. A different
selected vector may reflect a tie break, sampling or incomplete games. Results
from different methods, depths or corpora must not be pooled as independent
wins. MCTS also has a nonlinear value scale: changing all coefficients changes
its value calibration even where Minimax's score ordering would be preserved.

## Resource measurements and cleanup

The main supervisor ran **2527.22 seconds (42m07s)**; the diagnostic ran
**258.54 seconds (4m19s)**. Audits and adjudication took **33.42 seconds** in the
final supervised phase (adjudication itself 10.30 seconds). Including the gaps
between phases, cleanup confirmed no recorded experiment PID remained at
**2898.80 seconds (48m19s)** after the original launch, within the one-hour cap.
Report preparation and archiving follow this receipt.

Supervisors used nice 19, with OPENBLAS/OMP/MKL threads fixed to one. Three-second
sampling observed peak simultaneous descendant RSS of **539.7 MiB** in the main
study, **490.9 MiB** in the diagnostic and **274.6 MiB** during audit/analysis.
These are sampled process-tree sums, not exact peaks. Known child CPU excludes
work lost when children are killed; wall time includes startup and waiting on
the authorized shared machine.

The primary monitor observed one transient row at 481.62 seconds with nice 0,
zero RSS, state `?E` and command `(python3.11)`. Its attempt to lower that own
descendant's priority got `ProcessLookupError`: the PID was already absent.
All other sampled descendants were nice 19; the later phases sampled only 19.
The exact anomaly is preserved rather than claiming every sample had nice 19.
No crashes, illegal moves or infrastructure timeouts occurred. Six overall-cap
cancellations and 76 incomplete-depth records remain explicit in primary
outcomes. Training jobs were not changed.


## Validation and evidence

The implementation passed 68 focused Python checks and 52 material, pressure
and search regressions; Rust's nine unit tests and Clippy passed. Native parity
covers signed weights, both goal orientations and switching coefficients in a
reused engine. Additional parity checks cover 160 legal transitions and all 100
certified tactics. Material-only regression fixtures now disable the newly
adopted route terms explicitly.

All **271 journals / 16,496 played moves** replayed successfully (261 primary,
10 diagnostic), with zero played-search completion violations. The consensus
analysis adds four passing tests for common-perspective scoring, immutable
journals, abstention and failure exclusion. Its source hash and backend fingerprint
are recorded in the analysis output.

- [Compact comparison](comparison.json), [frozen design](design.json)
- [Full primary summary](summary.json), [primary replay audit](verification.json)
- [Main-screen audit](screen-verification.json), [diagnostic audit](supplement-verification.json)
- [Consensus scores and rankings](consensus.json), [training re-ranking](consensus-training-ranking.json)
- [Measurements and cleanup receipt](cleanup.json)
- [Full archive](runs.tar.gz) and [SHA-256](runs.sha256): frozen scripts, inputs,
  complete journals/manifests/checkpoints/exports, tests and monitor logs

To replay/audit the archived data without starting engines, extract the archive
and use the saved `audit.py` and `audit-supplement.py` scripts after updating their
output-root paths. The primary evaluator fingerprint is unchanged by the
separate benchmark-analysis module. The original implementation is `9424a35`;
the consensus tool was introduced in `05fdaee` and includes its final source hash.


Historical v1 archives require their original revisions; v2 intentionally
rejects their schema and stale evaluator fingerprints. Reproduce this study
from its archived scripts on the recorded implementation, using available
resources or explicitly authorized shared resources. The scripts contain the
original `/tmp` output root; edit that path for another location. The supplemental
script also contains a deadline tied to the original launch and needs a new,
explicitly bounded supervisor for a new experiment.
