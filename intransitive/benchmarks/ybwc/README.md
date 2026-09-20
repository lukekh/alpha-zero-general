# Issue #65 branch-parallel search

**Decision: keep YBWC branch-parallel search implemented and disabled.**
`threads` defaults to one, and one is the exact sequential search. Equal-time
paired play at two and four threads scored 0.490 and 0.510, with 95% decisive
intervals of 0.307-0.660 and 0.335-0.700; the predeclared rule required a lower
bound above 0.5 to adopt. The search is exact, reproducible for a fixed thread
count and 1.3-1.8x faster in wall clock at depths five and six, and none of that
converted into measurable strength at the time control tested. An exploratory
arm at four times the clock went 4-1 with twice the depth advantage, on five
decisive games, which is a reason to re-test at a longer control rather than a
reason to adopt. No live job, teacher label or checkpoint opponent was changed.

[Predeclared plan](plan.json),
[implementation contract](../../rust_teacher/PARALLEL_SEARCH.md).
Parameters were fixed before any measurement was taken. This is a bounded
experiment on one machine, not a production strength certification.


## Reproduction and identity

Baseline: `8d22a1117f9fcb4fdc89057f13c5f48255ad5c26`. The frozen record was made
with that revision's separately compiled binary; its hash and the candidate
file hashes are in [evidence/source-manifest.json](evidence/source-manifest.json)
and in each JSON's own header. Every report also carries the platform, the
starting revision and a working-diff digest; the source manifest is the complete
implementation identity, because the diff digest alone omits then-untracked new
files.

```sh
cargo build --release --offline --manifest-path intransitive/rust_teacher/Cargo.toml
cargo test --release --offline --manifest-path intransitive/rust_teacher/Cargo.toml
python -m unittest intransitive.tests.test_parallel_search -v

E=intransitive/benchmarks/ybwc/evidence
python -m intransitive.benchmarks.ybwc.reproduce --stage equivalence --baseline $E/baseline.json --output $E/equivalence.json
python -m intransitive.benchmarks.ybwc.reproduce --stage overhead --output $E/overhead.json
python -m intransitive.benchmarks.ybwc.reproduce --stage paired --output $E/paired.json
```

To rebuild the frozen baseline, `git archive` the baseline revision into a
temporary directory, build its binary there, and run

```sh
python -m intransitive.benchmarks.ybwc.reproduce --stage baseline \
  --binary <archived>/intransitive/rust_teacher/target/release/intransitive-rust-teacher \
  --revision 8d22a1117f9fcb4fdc89057f13c5f48255ad5c26 --output $E/baseline.json
```

The baseline request uses no parallel wire fields, so the old binary accepts it
unchanged — which is itself the compatibility check.

Test and lint output is archived beside the data, refreshed after master was
merged in: [rust-tests.log](evidence/rust-tests.log) (36 native tests),
[parallel-tests.log](evidence/parallel-tests.log) (10 tests),
[integration-tests.log](evidence/integration-tests.log) (153 tests across
`test_rust_teacher`, `test_certificate_bound`, `test_selective_search`,
`test_mvv_lva`, `test_clear_run`, `test_tuning`, `test_variable_material`,
`test_signed_search`, `test_exchange` and `test_shallow_pruning`).

The measurements below were taken before #67's certificate bound and #68's
shallow pruning were merged in. Both are off by default and neither touches the
default search path: after the merge the same three searches returned the same
node counts to the unit — 541,905 at one thread, 564,118 at two and 700,102 at
four — so the numbers still describe this code.

`cargo clippy --all-targets -- -D warnings` **fails, and failed identically
before this work**: `clear_run.rs:67` (`int_plus_one`) and `routes.rs:452`
(`needless_range_loop`). [clippy.log](evidence/clippy.log) and
[clippy-baseline.log](evidence/clippy-baseline.log) are the candidate and the
baseline revision run side by side; the two errors are the same two. They are
untouched by this change and are left for whoever owns those files rather than
folded into a search-cost diff.

## The #54 harness could not run this

The acceptance criteria name the #54 paired harness. It cannot run a native
candidate: `tournament.spec.candidate()` raises on any `backend` but `python`,
and `tournament.runner.engine_worker` constructs an `AlphaBetaPlayer` from a
`SearchConfig` that has no thread count. Adding a native backend would change
the hashed candidate identity that every existing manifest is built on, which is
well outside a search-cost issue.

What is reused instead is everything about #54 that carries the measurement:
its position generator (`generate_positions`), its opening/midgame/endgame stage
definitions, its both-colour pairing, its treatment of an unfinished game as
unresolved and worth zero, and its Wilson interval. The engines differ only in
`threads`; evaluation, limits, proof settings and table size are identical.

## Threads = 1 reproduces the frozen baseline

84 rows — seven starts, proof leaves on and off, depths one to six — replayed
against the record made with the baseline revision's own binary. **All 84 match
exactly**: action, score, principal variation, completed depth, node count,
proof nodes, transposition hits, reported work and stop reason.

The baseline request carries no parallel wire fields, so the pre-change binary
accepted it unaltered. That is the rollback path as well as the check:
`threads = 1` is not a reimplementation of the sequential search, it is the
sequential search.

Raw rows: [baseline.json](evidence/baseline.json),
[equivalence.json](evidence/equivalence.json).

## A fixed thread count reproduces itself

Seven starts at depth six, three runs each, at one, two, three and four threads.
**All 28 cells returned identical action, score, principal variation, completed
depth and node count across their three runs.** Node counts are identical, not
merely close: the tree a thread count searches is fixed, not sampled.

Thread counts do not reproduce each other, and the report never claims they do.
The same 28 cells give the size of that difference:

| Threads | Node ratio vs one thread (median) | Range |
| ---: | ---: | --- |
| 2 | 1.139 | 1.014 – 1.343 |
| 3 | 1.174 | 1.039 – 1.270 |
| 4 | 1.235 | 1.064 – 1.301 |

## Parallel search overhead

Fixed-depth searches on the same seven starts, proof leaves disabled to isolate
the heuristic tree. Every cell completed its target depth.

| Depth | Threads | Nodes vs 1 thread | Wall speedup (median) | Speedup range |
| ---: | ---: | ---: | ---: | --- |
| 4 | 2 | 1.001 | 0.87 | 0.57 – 1.17 |
| 4 | 3 | 1.001 | 1.04 | 0.69 – 1.22 |
| 4 | 4 | 1.012 | 1.09 | 0.45 – 1.47 |
| 5 | 2 | 1.094 | 1.32 | 1.16 – 1.48 |
| 5 | 3 | 1.118 | 1.51 | 1.45 – 1.90 |
| 5 | 4 | 1.181 | 1.77 | 1.55 – 2.09 |
| 6 | 2 | 1.139 | 1.35 | 0.80 – 2.30 |
| 6 | 3 | 1.174 | 1.74 | 1.05 – 3.48 |
| 6 | 4 | 1.235 | 1.70 | 1.26 – 4.26 |

**Depth four loses money.** At two threads the median is 0.87 and the worst case
0.57: the position clones and the join cost more than the brothers save when the
subtrees are that small. Gains only appear from depth five, which is why
`split_min_depth` defaults to four rather than two and why the shallow rows are
reported rather than dropped.

**The ranges matter more than the medians.** There is one timing sample per
cell. The 4.26x at depth six is above the thread count and is measurement
spread, not superlinear scaling; the 0.80x in the same row is the same spread in
the other direction. This machine has four performance and four efficiency
cores, so a four-thread search is not guaranteed four comparable cores.

Effective branching factor, as the median ratio of nodes between consecutive
completed depths, was 7.30 at one thread and 8.48 at four. The parallel tree is
wider per ply, which is the overhead stated directly. The three-thread figure
(6.91) is below the one-thread figure, which is a median over only fourteen
ratios and should be read as noise rather than as three threads searching a
smaller tree than one.

### Exactness survived every comparison

Across all 63 cross-thread fixed-depth comparisons the score was **bit-identical**
to the single-threaded score and the selected move was identical. Two
comparisons — depth four at two and three threads, from the official start —
returned a different principal variation *tail*: score `3.0519755004679845` and
root move 241 in both cases, with the continuation after the root move broken
differently. That is the design showing through. The root value and the root
move are provably preserved by a frozen, lower alpha; an equally-valued
continuation deeper in a child subtree is not, because the child was searched
with a different window and a different table.

## Depth reached in a fixed time

The mechanism by which parallel search could buy strength is extra depth inside
the same clock. Mean completed depth over the seven starts:

| Budget | 1 thread | 2 threads | 3 threads | 4 threads |
| --- | ---: | ---: | ---: | ---: |
| 0.5 s | 4.000 | 4.000 | 4.143 | 4.143 |
| 2.0 s | 4.286 | 4.571 | 4.714 | 5.000 |

At half a second, four threads reached a deeper ply on **one of seven** starts.
At two seconds it did so on five of seven, and reached depth six on two starts
that one thread could not take past depth five.

This is the most important table in the report for reading the paired result
below, and it was measured before those games were played.

## Equal-time paired play

The predeclared measurement. 25 starts — the official opening plus the first
eight opening, midgame and endgame starts by state hash — each played from both
colours, 50 games per thread count. Both engines use identical evaluation,
identical limits and identical proof settings; **only `threads` differs**. Half a
second per move, 100-ply cap, an unfinished game scored unresolved and worth
zero, exactly as #54 scores one.

| Threads | Games | Wins | Losses | Unresolved | Score | Decisive rate, 95% Wilson |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 50 | 13 | 14 | 23 | 0.490 | 0.307 – 0.660 |
| 4 | 50 | 13 | 12 | 25 | 0.510 | 0.335 – 0.700 |

Both intervals contain 0.5 with room to spare. The decision rule required a
95% lower bound above 0.5 to adopt; neither arm is close, and neither arm shows
a regression either. By stage, four threads went 6–4 in endgames, 4–3 in
openings, 3–5 in midgames and 0–0 from the official start.

**The mechanism explains the result, and it was measured first.** Mean completed
depth in these games was 3.969 against 3.917 at two threads and 4.322 against
4.136 at four: an advantage of **0.05 and 0.19 of a ply**. The fixed-time table
above predicted exactly this, before the games were played. A 1.7x search at
half a second per move does not reliably buy a whole extra ply on this game, and
without an extra ply there is very little for the extra threads to convert.

**This sample could not have detected a small effect.** 48 of 100 games hit the
100-ply cap and are unresolved, leaving 27 and 25 decisive games. An interval
that wide rules out only large effects. Detecting a genuine 0.55 score at 95%
confidence would need several hundred decisive games, and because both engines
are deterministic those games must come from distinct starts — replaying a start
from the same colour reproduces the same game exactly.

#54's consensus adjudication is not applicable to rescue the capped games. That
policy decides a capped game only when **both competing evaluations** agree on a
sign, and its value comes from the two candidates disagreeing. Here both sides
run the same evaluator, so it would collapse to a single evaluator's sign of the
final position. Reporting that as consensus adjudication would overstate it, so
the capped games stay unresolved.

Raw games, including every move with its depth, node count and time:
[paired.json](evidence/paired.json). The run took 1.07 hours.

## Exploratory: the same comparison at four times the clock

**Not predeclared, and not part of the decision.** The fixed-time table shows
that at half a second per move the mechanism barely operates, so a null result
there could mean either that the feature does not help or that it was never
given a control where it could. This arm separates those two readings.

Four threads against one at **two seconds per move**, seven starts (the official
start plus the first two of each stage) from both colours, 14 games:

| | Games | Wins | Losses | Unresolved | Score | Decisive rate, 95% Wilson |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4 threads, 2.0 s/move | 14 | 4 | 1 | 9 | 0.607 | 0.376 – 0.964 |

The depth advantage roughly doubled, from 0.19 of a ply at half a second to
**0.37** here (5.41 against 5.04), and the record went from 13–12 to 4–1. The
interval still contains 0.5, on five decisive games, so this establishes
**nothing** about strength on its own. What it does show is that the null at the
predeclared control is consistent with too little extra depth rather than with
the parallel search being worthless, and it says where a real test would have to
be run: a longer clock, several hundred decisive games, and many more distinct
starts.

Reproduce with the overrides, which mark the record exploratory:

```sh
python -m intransitive.benchmarks.ybwc.reproduce --stage paired \
  --seconds-per-move 2.0 --paired-threads 4 --per-stage 2 --output $E/paired-long.json
```

Raw games: [paired-long.json](evidence/paired-long.json). The `starts` string
inside that record is the plan's default text and describes 25 starts; the run
used `--per-stage 2`, so the true count is the seven starts and 14 games above.
`reproduce.py` now records `per_stage` alongside it so a later record cannot
carry the same stale description.

## What this does not establish

- **One machine, one time control.** Four performance cores and four efficiency
  cores. Thread counts above four are neither measured nor claimed, and the
  measured speedups will not transfer to a machine with a different core mix.
- **Not an equal-memory comparison.** Each thread owns a table of
  `table_entries`, so the four-thread engine holds four times the entries. That
  is the natural configuration for private tables and it is disclosed, not
  controlled for.
- **Deterministic engines make repeats worthless.** A start replayed from the
  same colour is the same game, so the sample size is the number of distinct
  starts, not a number of repeats. Wider intervals cannot be bought by replaying.
- **Shared tables were not measured.** Measuring them would mean shipping a
  nondeterministic search path, which the #54 and #55 constraints rule out
  before any measurement could be taken. The comparison here is YBWC with
  private tables against the sequential search, not against Lazy SMP.
- **A time-limited search is not reproducible** in either mode. The determinism
  claim covers searches that finish inside their depth and node limits.
