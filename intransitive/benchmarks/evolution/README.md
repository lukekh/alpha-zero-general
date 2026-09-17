# Bounded evolution versus random search (#55)

The [optimizer](../../evolution/README.md) was exercised on 17 September 2026
(Australia/Adelaide), after explicit authorization to share the machine with live
training at low priority. Every benchmark process ran at **nice 19**, with one
active game and at most four resident candidate engines. Training processes were
not changed. Host: macOS 15.6.1, eight logical CPUs, 8 GiB RAM; Python 3.11.4,
NumPy 2.3.5, Numba 0.65.1, llvmlite 0.47.0.

**No candidate qualified, and these results do not establish that evolution is
better than random search or unchanged defaults.** Two-ply games and a small
search-work ceiling test the machinery and rejection policy, not playing
strength. All exports remain unaccepted inputs for #56; no defaults, generator
or trainer settings were updated.

A [follow-up with longer endgames](endgame-search-20260917/README.md) produced
official outcomes, but its selected provisional genome tied defaults 3–3 on
three fresh validation positions. That experiment also found no confirmed gain.

## Configuration and measurements

Both algorithms used population two, two generations, one common search start,
one fresh validation line per generation, and identical initialization for each
seed. All five supported Python module scales were eligible to vary. Opponents
included contemporaries, unchanged defaults and the fixed pressure-10 archive.
No winner qualified for the historical hall. Search was fixed depth one, 50,000
work units and five seconds per move, proof disabled, at most two physical plies.
Each arm had identical **150-second, 32-attempt, 6-million reserved-work ceilings**.
The archived `config.json` differs from the
[example](../../evolution/smoke-benchmark.json) only by reducing its wall cap
from 180 to 150 seconds. All four arms completed both generations:

| Method / seed | Unique configs¹ | Unique games | Unfinished | Incomplete depth | Measured work | Known child CPU (s) | Run wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Evolution / 55 | 5 | 28 | 8 | 20 | 1,340,265 | 79.77 | 115.97 |
| Random / 55 | 6 | 32 | 8 | 24 | 1,569,390 | 90.49 | 115.53 |
| Evolution / 56 | 5 | 28 | 16 | 12 | 1,266,492 | 71.59 | 81.89 |
| Random / 56 | 6 | 32 | 8 | 24 | 1,569,010 | 94.93 | 132.29 |

¹ Includes the two fixed reference configurations. Evolution evaluated three
population genomes per seed; random search evaluated four.

Each evolution arm reused four games in generation two without counting them
as new evidence. Reserved work was 3,610,000 per evolution arm and 4,140,000 per
random arm. This compares **equal allocated budgets**, not equal realized compute.
Timings include startup/compilation on the authorized shared host; they are not
isolated throughput estimates or recommendations for more parallel workers.

The comparison CLI took **456.57 seconds**. One-second sampling observed peak
summed process-tree RSS of **672.4 MiB**, with at most six processes (parent,
resource tracker and four candidate engines). The successful smoke took 68.93
run seconds / 76.41 CLI seconds and peaked at 386.1 MiB. It completed 16 games:
six unfinished and ten incomplete-depth rejections, with 28 reused-process
handshakes. There were **no crashes, illegal moves or infrastructure timeouts**
in the successful smoke or comparison.

An initial fresh-process smoke was stopped after 109.27 run seconds: two
unfinished games, one depth rejection and one safely cancelled attempt. It
exposed repeated Numba startup cost. Its records and resource use are retained.
The bounded process cache was tested before the successful smoke and comparison.
Across both smokes and all four arms: **140 attempts**, **18.3 million reserved
work units**, and about **642 seconds** of measured CLI/run time (including the
retired attempt, excluding preparation/audits). This stayed within the authorized
144-attempt, 27-million-work, 900-second allocation.

## Convergence, diversity and uncertainty

Every generation's apparent winner and unchanged-default validation opponent had
zero official wins, zero official completion, a lower win bound of zero, a
paired/line-clustered interval of **[0, 1]**, and failed eligibility. Incomplete
searches remained explicit failures with work/achieved-depth records; they did
not become wins or artificial draws. There is no positive fitness convergence
or accepted winner to report.

Normalized mean pairwise genome distance and off-gene fraction, generation 0 → 1:

| Method / seed | Distance | Fraction off |
| --- | --- | --- |
| Evolution / 55 | .3997 → .1781 | .50 → .20 |
| Random / 55 | .3997 → .3281 | .50 → .40 |
| Evolution / 56 | .2736 → .2157 | .60 → .50 |
| Random / 56 | .2736 → .2813 | .60 → .50 |

The 120 comparison games contain **80 distinct full match keys, 15 distinct
physical trajectories and 12 distinct configurations across arms**. Shared
initial populations, fixed reference games, repeated trajectories and the common
corpus are not independent strength evidence. No outcomes or generation intervals
are pooled to manufacture a more precise estimate. Per-opponent/colour breakdowns,
depth/latency/work telemetry, preflights, fitness and exports are archived.

## Evidence and reproduction

- [Summary and compute](evidence/summary.json)
- [Per-seed convergence and default comparison](evidence/comparison.json)
- [Replay audit](evidence/verification.json): all 136 successful-run records
- [Real resume audit](evidence/resume-verification.json): no new games; identical
  population, RNG, phase, generation, hall, ledger, fitness, exports and reservations
- [Fresh/reused-process parity](evidence/process-reuse-parity.json): all three
  finished initial games matched status, winner, trajectory and final state
- [Full tests](evidence/tests.log): 37 passed, one optional native test skipped;
  [focused rerun](evidence/evolution-tests.log): all 15 passed after final cache changes
- [Full archive](evidence/runs.tar.gz): manifests, checkpoints, game journals,
  exports, logs, measurements, exact optimizer source/digests and config
- [Archive checksum](evidence/runs.sha256)

Provenance is the repository base commit plus the optimizer and backend source
digests in each manifest. The retired smoke has the earlier optimizer digest and
cannot resume under the later implementation; its canonical matches still replay
against the unchanged harness/backend. All candidate children were confirmed
reaped after the comparison.

Reproduce with the documented `smoke`, `prepare` and `benchmark --seeds 55 56`
commands, using the archive's `config.json` for the exact measured caps. Use
available resources or explicitly authorized shared resources; this measurement
used `nice -n 19`. Audit saved evidence without starting engines:

```sh
mkdir -p /tmp/evolution-evidence
tar -xzf intransitive/benchmarks/evolution/evidence/runs.tar.gz -C /tmp/evolution-evidence
.venv/bin/python -m intransitive.benchmarks.evolution.verify \
  /tmp/evolution-evidence/smoke-pooled \
  /tmp/evolution-evidence/benchmark/evolution-55 \
  /tmp/evolution-evidence/benchmark/random-55 \
  /tmp/evolution-evidence/benchmark/evolution-56 \
  /tmp/evolution-evidence/benchmark/random-56
```

A [longer search under a one-hour allocation](hour-search-20260917/README.md) found provisional endgame weights that scored 31–3–6 at depth one and 23–8–1 at depth two on separate fresh validation sets. It used 39m35s at low priority; all 696 records passed replay. Held-out acceptance remains separate.

At the user's request, the [signed depth/MCTS follow-up](signed-depth-20260917/README.md)
also reports **consensus adjudication** of unfinished games. The two competing
weight configurations evaluate the same final legal board from player zero's
perspective, without tree search or proof. Matching strictly positive scores
award player zero an adjudicated win; matching negative scores award player one.
Ties, disagreement and budget exhaustion remain inconclusive. Nonempty prefixes
cancelled by an overall time cap may be scored; crashes, illegal moves and
incomplete-search failures are excluded. Official journals, optimizer fitness
and acceptance eligibility are preserved. This post-hoc analysis is not an
independent strength oracle, especially when two similar heuristics agree.

```sh
nice -n 19 .venv/bin/python -m intransitive.benchmarks.evolution.adjudicate \
  /path/to/optimizer-run /path/to/fresh-screen \
  --max-seconds 120 --output /tmp/consensus.json
```

The report gives official and adjudicated wins separately. Its alternative
ranking excludes inconclusive games from the decided-game win fraction and also
shows conservative wins over scheduled games. Full score pairs and final-state
hashes make each adjudication inspectable.
