# Variable material at one twentieth scale, with and without MVV-LVA

Implementation `c10a530ffee5e7c61a26ed0bc264a433a70256b7` follows the [pruning comparison](../depth4-pruning/README.md).
Four settings cross variable material at full/one-twentieth scale with MVV-LVA
off/on. Each setting plays the same flat-material opponent on four openings,
with colours swapped: **eight scheduled games per setting**, 32 in total.
All use Minimax depth 4, PVS, both pruning methods and the explicit experimental
evaluator option. No evaluator or pruning defaults are adopted by this study.

## Results

**Neither scaling nor MVV-LVA showed a clear strength improvement.** On the
seven opening/colour pairs attempted by all four settings, consensus outcomes
were identical: **3 wins, 2 losses, 2 inconclusive**. The three pairs that
finished decisively in all settings also had identical winners (1 win, 2 losses).
Only full-scale/MVV-off reached the eighth game, whose interrupted prefix was
adjudicated a loss; its aggregate 3–3 versus 3–2 elsewhere is therefore not
evidence that scaling or ordering improved strength. See the explicit
[paired outcomes](paired-outcomes.json). The scaled configuration did change
play: for example, the loss on seed 2669 as colour 1 took 59 moves instead of 27.

All scores are from the variable-material candidate's perspective. Official
unfinished games and consensus-inconclusive games are different categories.

| Material scale | MVV-LVA | Official W–L–unfinished | Consensus W–L–inconclusive | Failures | Official pending |
| --- | --- | --- | --- | ---: | ---: |
| Full | off | 2–2–3 | 3–3–2 | 0 | 1 (1 interrupted, 0 unstarted) |
| Full | on | 1–2–3 | 3–2–2 | 0 | 2 (1 interrupted, 1 unstarted) |
| 1/20 | off | 1–2–3 | 3–2–2 | 0 | 2 (1 interrupted, 1 unstarted) |
| 1/20 | on | 1–2–3 | 3–2–2 | 0 | 2 (1 interrupted, 1 unstarted) |

Official pending includes interrupted games, not just games never started. A
nonempty interrupted prefix can still receive a consensus result; do not add
that column to the consensus totals.

For an unfinished or interrupted game, both competing configurations must agree on a strict
nonzero sign for the final board to adjudicate a winner. Disagreement/ties remain
inconclusive. Incomplete-search failures are excluded and never become wins.
Official journals are not rewritten. Changing material scale also changes one
adjudicator, so consensus scores must be read alongside official outcomes.

These are **four reused opening clusters**, not held-out strength validation.
The positions are frozen subset indices 0,2,5,7 of the previous eight endgames:
seeds 1643,1652,2669,2639, two from each original corpus seed, including the sole
4-versus-4 opening where pruning can be eligible. This subset was selected before
MVV-LVA/scale results to fit the hour and retain that coverage. Do not pool arms
or previous experiments as independent samples. Per-opening outcomes and paired
uncertainty are retained in [comparison.json](comparison.json).

## What changed

Full variable material uses material coefficient **100**; one-twentieth uses
**5**. Flat opponents always use **100**. Since variable army totals are
normalized by a fixed 100, coefficient 5 is exactly equivalent to changing BASE
from 100 to 5. Changing BASE in both the numerator and normalization would cancel
the intended reduction. All other adopted coefficients are identical:
advantage 23.967050360966205, attack 25.714516982666414, defence 32.5643023919054,
with overload/pressure disabled.

In the saved five-versus-three-piece example, variable material drops from
+2009.68 to +100.48, while advantage stays +111.85, attack +45.53 and defence
−6.51. Total static evaluation drops from +2160.55 to +251.35. See
[module scales](../../../heuristics/MODULE_SCALE.md) and the
[selectable preset](../../../heuristics/configs/variable-material-scaled.json).
This changes the evaluator's balance, not merely a reporting unit.

MVV-LVA uses pre-move per-type values from each army's own counts. Within existing
tactical priority groups, it prefers the most valuable victim and then the
least valuable attacker. Wins, TT/PV preferences, previous root scores and
corner defence retain priority. No move is discarded. A positive uniform scale
leaves these relative capture keys unchanged. With flat material the keys tie,
so the original order is retained. The flag is shared by both players per arm;
its accounted work/time is included, including flat-mode ordering overhead.

Python reference and compiled ordering and Rust implement the option. It enters
search/cache identity, and statistics record actual ordering work. See the
[MVV-LVA contract](../../../heuristics/MVV_LVA.md). Full unpruned scores are
preserved by tests; selective search and finite caps can make ordering affect
moves and completion. This benchmark uses selective search throughout.

## Observed search activity

Played-search totals (failed-response details remain in the journals):

| Setting | Evaluator | Moves | MVV-LVA nodes | Capture candidates ranked | NMP cutoffs | Futility skips |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| full-mvv-off | variable | 134 | 0 | 0 | 293 | 0 |
| full-mvv-off | flat | 135 | 0 | 0 | 177 | 0 |
| scaled-mvv-on | variable | 141 | 145734 | 19680 | 293 | 0 |
| scaled-mvv-on | flat | 142 | 135997 | 18439 | 177 | 0 |
| scaled-mvv-off | variable | 129 | 0 | 0 | 293 | 0 |
| scaled-mvv-off | flat | 129 | 0 | 0 | 177 | 0 |
| full-mvv-on | variable | 121 | 133452 | 17947 | 293 | 0 |
| full-mvv-on | flat | 122 | 112136 | 21768 | 177 | 0 |

## Limits and verification

Each move has **30 seconds / 100 million work units**, requested depth 4,
proof depth/tokens 2/64. Games have 192 plies / 180 active seconds. Selective
mate-range scores must also complete requested depth; they are not certificates
for early termination. All arms use the same settings except material coefficient
and MVV-LVA. Arm order is full/off, scaled/on, scaled/off, full/on; each has an
800-second wall cap, reduced if needed to reserve audit time before 3350 seconds.
Two two-ply smoke games precede the arms. The supervisor deadline is 3500 seconds
plus bounded cleanup, within the one-hour allocation.

Supervised elapsed time: **3247.88 seconds**; cleanup completed
**3337.89 seconds** after launch. Peak sampled process-tree
RSS: **394.1 MiB**; priorities `[19]`,
**0 priority anomalies**, at most 4 processes.
BLAS/OMP/MKL limits were one. No observed experiment descendants remained.

All **31 records / 1057 played moves** passed legal replay and
exact report reconstruction. Smoke outcomes are not strength evidence.
Before the run, 102 focused Python tests, 13 Rust tests and Clippy with warnings
denied passed. Tests cover lexicographic victim/attacker priorities, both army
perspectives, capture/pop restoration, reference/compiled/compact ordering,
complete legal move sets, exact unpruned scores, cancellation, native protocol
and cache isolation, and the exact one-twentieth material contribution with other
terms unchanged. Live teacher processes/binaries were not changed.

## Evidence

- [Frozen design](design.json), [results and uncertainty](comparison.json)
- [Consensus scores](consensus.json), [replay audit](verification.json)
- [Paired outcomes](paired-outcomes.json), [search counters and work](search-stats.json)
- [Measurements](measurements.json), [cleanup](cleanup.json)
- [Full archive](runs.tar.gz), [SHA-256](runs.sha256)

The archive includes all manifests, complete journals, reports, test receipts,
monitor logs and reproduction scripts. Run its `study.py` through `supervise.py`
at the recorded implementation after changing the original
`/tmp/issue55-variable-mvv-scale-20260917` output root, with a bounded resource
allocation. Earlier archives require their recorded implementation because
search/evaluator fingerprints have changed.
