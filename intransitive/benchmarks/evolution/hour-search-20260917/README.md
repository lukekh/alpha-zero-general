# Promising endgame weights from the extended search

The selected evolved candidate beat unchanged defaults **31–3 with six
unfinished games at depth one**, then **23–8 with one unfinished game at depth
two**, on separate fresh development-validation positions. Both screens passed
the completion and search-depth requirements. These are promising endgame
results, not held-out acceptance or evidence about opening/midgame strength.
Application defaults and live training were unchanged.

The user authorized up to one hour at low priority. The supervised experiment
used **2,375.18 seconds (39m35s)** including corpus preparation, process startup
and the additional depth-two screen. All **696 game records** passed replay
checks. There were no crash, illegal-move, infrastructure-timeout or
incomplete-depth records, and all observed experiment processes were reaped.

## Selected coefficients and search space

| Term | Search range | Unchanged default | Selected effective scale |
| --- | ---: | ---: | ---: |
| Material | Fixed | 100 | 100 |
| Advantage | 0–100 | 25 | 23.967050360966205 |
| Attack | 0–100 | 0 | 25.714516982666414 |
| Defence | 0–100 | 0 | 32.5643023919054 |
| Overload penalty | 0–100 | 0 | 0 |
| Pressure | 0–20 | 0 | 0 |

Coefficients are continuous and nonnegative; zero disables a term. Material,
module signs, geometry and evaluator clipping remain fixed. The
[full-precision genome](provisional-genome.json) is **UNACCEPTED** pending the
separate acceptance process in #56. Its config hash is
`d3d2d4d4fbaae355f8ee2a385ff84c38d734190b7c8041916a8a4fcfa4fa9ef6`.
The search considered 26 distinct proposed configurations plus two fixed
references across all arms; the published bounds are not an exhaustive grid.

## Frozen experiment

The optimizer implementation was unchanged from the prior measured runs. Input
provenance records revision `9998e79` and the complete implementation and runtime
digests. The prior successful depth-mode smoke receipt was revalidated against
those digests before the comparison.

Corpus seed 257 generated 120 capture-biased legal lines, each limited to 240
plies. The first 12-piece endgame from each line supplied 42 search, 42 validation
and 36 held-out positions, preserving legal action histories, corner ownership,
line splits and symmetry identities. Two common search positions and six
validation positions (two fresh lines per generation) were selected by stable
hash. Twenty further validation lines were frozen for the final depth-one screen
before any search games ran. The remaining 16 were used only for the later
depth-two screen. No held-out positions were played.

Both algorithms ran serially with seeds 81 and 82, population three, up to three
generations, mutation probability .7, conditional toggle probability .3 and
log-space sigma .5. Elites, crossover, off-rate, initialization, historical
opponents and eligibility follow the published optimizer contract; exact
settings are in [design.json](design.json). There was one active game, with at
most four resident candidate engines during optimization and two during each
final screen. Numerical-library thread limits were set to one.

Each comparison arm had identical ceilings of 650 seconds, 180 game attempts
and 75 billion **reserved logical-work units**. Reservations include conservative
worst-case game and warmup work; they are not measured consumption. Per-move
search was fixed at depth one, two seconds and two million work units, with
proof depth/nodes 2/64. Games allowed 192 plies or 15 active seconds. Completion
required at least 80%, all scheduled games attempted and no disqualifying
failures. Search preflights and minimum completed depth remained mandatory.

## Comparison outcomes

| Arm | Completed generations | Attempts | Official finishes¹ | Unfinished | Measured search work | Run seconds | Stop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Evolution 81 | 3 | 152 | 138 | 14 | 204,310,613 | 558.84 | Complete |
| Random 81 | 2 | 180 | 169 | 11 | 128,382,768 | 569.29 | Game ceiling during third validation |
| Evolution 82 | 3 | 128 | 111 | 17 | 303,825,826 | 441.80 | Complete |
| Random 82 | 3 | 164 | 153 | 11 | 245,801,556 | 479.80 | Complete |

¹ This counts games with official winners, not wins for a particular candidate.
There were 551 corner wins and 20 stalemate wins. No arm exhausted its wall or
reserved-work ceiling. Random 81's partial third-generation evidence remains in
its checkpoint and journals; it did not yield a completed candidate export.

The 624 comparison records contain 496 distinct match keys and 235 distinct
trajectories. Shared initial populations, reference games, deterministic
trajectories and repeatedly used search starts are not independent observations.
Evolution and random search received equal ceilings, not equal realized compute;
the partial control also precludes a clean three-generation superiority claim.
Full per-generation fitness, diversity, candidate counts, CPU, memory and
convergence records are retained in the archive.

## Selection and fresh screens

Only completed candidate exports with eligible training entered the ranking.
Their direct validation games against unchanged defaults were deduplicated by
full match key. Candidates were ranked by conservative wins/scheduled games,
then completion, fewer active optional modules and stable config identity. This
rule was recorded before the comparison. Validation against other opponents did
not replace direct-default evidence for this selection.

The third-generation evolution-82 winner ranked first: 3–1 against defaults on
two validation lines, and 6–1–1 across its full validation slate. Two random-82
candidates also scored 3–1 directly against defaults; the evolved candidate won
the preset tie-break by using two optional modules instead of three. This is a
selection decision, not a statistically established difference between those
candidates. The full [selection record](selection.json) includes every ranked
candidate and the source export.

The genome was frozen before the 20-line depth-one screen. After that screen's
positive result, the **same genome** received an additional depth-two transfer
screen on all 16 remaining validation lines, chosen by stable hash without
examining their game outcomes. The extra screen had a 670-second run limit and
700-second supervisor deadline, comfortably inside the remaining hour budget.
It used depth two, four seconds and five million work units per move, 192 plies
or 20 active seconds per game, unchanged proof settings and explicit rejection
of non-proven moves that failed to complete depth two.

| Fresh screen | Paired starts | Wins | Losses | Unfinished | Completion | Candidate win-point interval² | Run seconds³ |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Depth one | 20 | 31 | 3 | 6 | 85% | [0.471, 1.000] | 98.93 |
| Depth two | 16 | 23 | 8 | 1 | 96.875% | [0.379, 1.000] | 199.19 |

² The harness's conservative 95% weighted Hoeffding intervals cluster paired
colours by opening-generation line and span unresolved outcomes. They remain
wide and include 0.5. These development results do not satisfy a claim of
statistically established general strength. The two depths use different
positions and budgets; their outcomes must not be pooled or used to estimate a
causal depth effect. Held-out acceptance and opening/midgame testing remain
separate.

³ Includes imports in each screen process. Supervisor measurements additionally
include launch and polling overhead. No result was scored from a truncated
search depth; the one unfinished depth-two game remains scoreless.

## Resource accounting and audit

The supervisor and its children were launched at nice 19, with process-tree
sampling every three seconds. The largest sampled subtree was 618,856,448 bytes
(about 590 MiB), including the CLI, resource tracker and four resident engines;
only one game was active. This is sampled aggregate RSS, not a continuous peak
or total machine memory. Existing live teacher processes were not modified.

One sample reported PID 41704 as `(python3.11)`, state `?E`, zero RSS and nice 0.
It no longer existed when the monitor attempted to lower that descendant's
priority. The complete event is retained in [measurements.json](measurements.json).
All other sampled entries were nice 19, including both fresh screens. The run
was not interrupted or resumed, and no successful priority correction was
needed. The cause of that transient entry is not asserted.

The total 39m35s sums measured supervised phases, including preparation and
polling. Read-only replay audits and report writing are outside that experiment
time. All 111 observed experiment process IDs, including the supervisors, were
absent at cleanup. Verification covers 624 optimizer records, 40 depth-one
records and 32 depth-two records, including every unfinished outcome.

- [Full summary and per-arm exports](summary.json)
- [Frozen design](design.json) and [selected genome](provisional-genome.json)
- [Comparison replay audit](verification.json)
- [Depth-one replay audit](additional-verification.json)
- [Depth-two design](depth-two-design.json), [report](depth-two-report.json) and [replay audit](depth-two-verification.json)
- [Process measurements](measurements.json), [depth-two measurements](depth-two-measurements.json) and [cleanup](cleanup.json)
- [Complete manifests, journals, checkpoints, scripts and logs](runs.tar.gz), with [SHA-256 checksum](runs.sha256)

To inspect or replay the archive, extract it under `/tmp`; it contains
`issue55-hour-20260917/`. Use the repository's documented Python 3.11 tournament
environment. Replay starts no engines:

```sh
nice -n 19 .venv/bin/python -m intransitive.benchmarks.evolution.verify \
  /tmp/issue55-hour-20260917/benchmark/evolution-81 \
  /tmp/issue55-hour-20260917/benchmark/random-81 \
  /tmp/issue55-hour-20260917/benchmark/evolution-82 \
  /tmp/issue55-hour-20260917/benchmark/random-82
PYTHONPATH="$PWD" nice -n 19 .venv/bin/python \
  /tmp/issue55-hour-20260917/audit.py
PYTHONPATH="$PWD" nice -n 19 .venv/bin/python \
  /tmp/issue55-hour-20260917/audit_depth_two.py
```

The archived `prepare.py`, `supervise.py`, `final_screen.py` and
`depth-two/run.py` contain the exact experimental commands and selection logic.
For a new experiment, use fresh output directories, revalidate the matching
smoke receipt and set the overall allocation before launching engines.
