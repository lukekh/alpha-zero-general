# Longer endgame search: no confirmed improvement

A follow-up search on 17 September 2026 replaced the two-ply smoke games with
longer legal endgames. The selected provisional genome **tied unchanged defaults
3–3 on three fresh validation positions**, with all six games completed. The
paired interval remains [0, 1]. This does not establish a stronger heuristic.
No defaults, generator settings, trainer settings or live training processes were
changed. Held-out positions were not played.

## Experiment

The same committed optimizer (`ac30d8e`) ran at requested nice 19. A new corpus
used seed 155, 36 capture-biased legal lines and a 240-ply generation limit. The
first position with 12 remaining pieces from each line formed the endgame corpus:
13 search, 11 validation and 12 held-out positions. Original pool assignments,
action histories, corner ownership and symmetry identities were retained.
Positions were selected by stable hash before candidate evaluation.

Two calibration pilots compared a provisional 25/12/10/0/2 genome
(advantage/attack/defence/overload/pressure) against defaults on two search starts:

| Pilot | Wins | Losses | Unfinished | Run seconds |
| --- | ---: | ---: | ---: | ---: |
| Depth two | 2 | 1 | 1 | 71.45 |
| Depth one | 3 | 0 | 1 | 56.03 |

Neither passed the unchanged 80% completion threshold. Replay found repeated
board/turn states in the unfinished games; those games remained scoreless.
Depth one was chosen for the bounded optimizer run, and the ply allowance was
increased from 96 to 192. The calibration genomes were not injected into the
optimizer population or treated as accepted improvements.

The comparison used optimizer seeds 73 and 74, population two, two generations,
mutation probability .7 and conditional toggle probability .3. Other strategy
controls remain published in [design.json](design.json). Every arm used the same
185-second, 40-attempt and 16-billion **reserved** logical-work ceilings, four
resident engines, one common search start and one fresh validation line per
generation. Individual searches had depth one, two seconds and two million work
units; proof depth/nodes were fixed at 2/64. Games allowed 192 plies or 12 active
seconds. Conservative work reservations cover the worst case, not measured work.

| Arm | Finished generations | Game records | Official finishes¹ | Unfinished | Cancelled | Measured search work | Run seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Evolution 73 | 2 | 28 | 21 | 7 | 0 | 72,124,435 | 161.62 |
| Random 73 | 1 | 27 | 22 | 4 | 1 | 67,723,194 | 185.10 |
| Evolution 74 | 2 | 28 | 25 | 3 | 0 | 37,942,214 | 127.81 |
| Random 74 | 1 | 19 | 17 | 1 | 1 | 24,182,701 | 185.09 |

¹ A finished game has an official winner; this column is not an optimizer's
win count. All 85 official finishes were corner or stalemate outcomes. There
were no incomplete-depth, crash, illegal-move or infrastructure-timeout records.
The two cancellations occurred at cumulative wall-budget exhaustion.

Random 74 also experienced an external monitoring interruption after its first
10 games: the monitor raised a priority alert but did not log the offending
process. Its cause is undetermined. The optimizer saved cleanly and resumed at
nice 19; [live parent/child priorities](resume-priorities.json) were verified at
19. Its budget was not reset. Repeating process startup consumed remaining time,
so that arm's compute comparison is confounded. Random 73 independently exhausted
its wall budget during second-generation validation. No superiority claim is made
from comparing completed evolution arms with these partial controls.

## Candidate selection and fresh screen

All completed optimizer exports failed validation eligibility, generally with
one unresolved game out of four. The seed-73 training leader scored 5–0–1, but
its two fresh validation batches scored 2–1–1 and 1–2–1. Training success did not
establish generalization.

For one final development-validation screen, candidates with eligible training
results were ranked by conservative validation wins over unique match keys,
then completion, fewer active optional modules and stable identity. Duplicated
records across arms were not counted again. This selected the seed-74 evolved
genome with pressure disabled:

| Module | Effective scale |
| --- | ---: |
| Material count (fixed) | 100 |
| Advantage | 0 |
| Attack | 44.59984687779922 |
| Defence | 0 |
| Overload (subtracted) | 0.30123493773565047 |
| Pressure | 0 |

Its earlier training result was 4–2–0; its earlier validation was 2–1–1. The
additional screen used three previously unused validation lines, both colours,
and unchanged defaults as the sole opponent. The genome and selection rule were
frozen before those games. The original completion policy remained unchanged.

The screen completed in 43.90 seconds within a 70-second ceiling: **3 wins,
3 losses, no unfinished games**. Both configurations passed this screen's
completion requirement but neither outperformed the other. This was development
validation, not held-out acceptance. The [provisional genome](provisional-genome.json)
is saved for reproducibility and further investigation; it is **not accepted**.

## Budget and evidence

There were 116 game attempts including calibration and the final screen.
Recorded run time totaled 830.99 seconds. Fresh-log creation/last-write timestamps
estimate 864.60 seconds of total CLI time (about 14m25s), within the 15-minute
allocation; preparation and read-only audits are excluded. The interrupted
monitor did not produce a complete process-tree memory measurement, so none is
claimed. Live training was left running, and only one match was active at once.

The 102 optimizer records contain 66 distinct match identities, 57 trajectories
and 12 configurations across arms. Repeated initial populations, reference games,
opening lines and trajectories are not independent evidence. The final screen
has only three paired opening-line clusters; its wide interval must not be
presented as a calibrated general-strength advantage.

- [Summary, actual budgets and per-generation outcomes](summary.json)
- [Selection rule and source export](selection.json)
- [Optimizer replay audit](verification.json)
- [Pilot/final-screen replay audit](additional-verification.json)
- [Pilot repetition audit](pilot-cycle-audit.json)
- [Full manifests, journals, checkpoints, scripts and logs](runs.tar.gz)
- [Archive checksum](runs.sha256)

All 116 records were replayed. Source/runtime identities are frozen in the
manifests; the optimizer implementation was unchanged. The raw original
`benchmark/comparison.json` retains the monitoring-stop snapshot; `summary.json`
uses the final reports after resume. The archive preserves both histories.
