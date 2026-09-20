# Issue #67 bounded validation

**Decision: every exchange and delta switch stays off by default.** The
quiescence filters do what they were built to do — with variable material they
cut total quiescence-inflated work from 2.54× to 1.67× and examine 62% fewer
captures — but the equal-time paired play cannot establish strength
non-regression, and the predeclared rule requires both. Main-search exchange
ordering is a clear negative result and should not be revisited without a
different idea: it leaves the first-move cutoff rate unchanged and costs work.
No live generator, trainer, tournament protocol or checkpoint opponent changed.

[Predeclared plan](plan.json), [implementation contract](../../heuristics/EXCHANGE.md).
Modes, positions, depths, margins and the adoption rule were fixed before any
measurement was taken, and no weight, margin or threshold was chosen using these
positions. This is a bounded experiment, not a strength certification.

## Reproduction and identity

```sh
python -m unittest intransitive.tests.test_exchange -v
python -m intransitive.benchmarks.exchange.reproduce \
  --output intransitive/benchmarks/exchange/evidence/results.json
```

Use the repository Python environment (Python 3.11.4, NumPy 2.4.6, Numba 0.67.0
from the lock). Sections run separately with `--sections`.

[evidence/results.json](evidence/results.json) carries the plan, platform,
runtime versions, every position's opening actions and hashes, and every
measured row. It records the starting revision `8d22a11` and a working-diff
digest, both taken **before** this change was committed; the complete
implementation identity is the `backend_version` digest in the same file, which
hashes every `heuristics/*.py` (including `exchange.py`), `Intransitive*.py` and
`tournament/*.py` file by content.

Corpus generation took 68.4 s, kernel warmup 0.05 s (Numba's on-disk cache) and
the measurements 276.5 s.

**The host was shared.** Other search benchmarks ran on the same machine
throughout. Work, node, capture, cutoff and outcome counts are deterministic and
unaffected; wall-clock numbers, and the depth each engine reached inside a 0.2 s
move, are not. The paired table therefore reports per-move nodes and completed
depth for both sides so a reader can check that equal time bought comparable
search — it did, within 2%.

## What was measured, and on which positions

Issue #66's "quiescence multiplies per-move work 4–14× at depth 2" came from the
weight-evolution run, and those five positions are not archived in this
repository. The multiplier is therefore re-established from scratch: every ratio
below compares rows measured in the same run, on the same position, in the same
material mode, so the before/after comparison the issue asks for is internally
valid even though the absolute figure is not comparable with #66's.

The corpus needed rebuilding too. `tournament.spec.generate_positions` spends its
capture bias on the ply it samples, so **all eight positions it selects have no
legal capture for the side to move at all** — quiescence resolves nothing there
and no filter can be observed. The corpus here keeps that generator's identity
record, opening-action replay, orbit hash, pool split and one-per-stage-per-line
rule, and changes only which ply is sampled: a short seeded capture-biased
prefix, then baseline-engine self-play, sampling the first ply from ply 8 that
carries at least two legal captures.

Seven positions survived that filter and the search/validation pool split, all
of them midgame. Contact this dense simply does not occur in the sampled opening
and endgame plies, so **nothing here says what these switches do in an endgame**.

| position | pool | captures at root | plies played |
|---|---|---:|---:|
| `01e82622aee2` | validation | 2 | 19 |
| `0c9d0609b0fa` | search | 2 | 52 |
| `3269ffde0e48` | search | 2 | 61 |
| `44e58be75c1f` | search | 4 | 28 |
| `ac4425c592bf` | validation | 2 | 36 |
| `c308cdc277f2` | validation | 2 | 16 |
| `cf8385a289f5` | search | 2 | 25 |

## Compiled and reference agree

48 captures compared across flat, variable and variable-linear valuations on
these positions, **zero mismatches**. `intransitive.tests.test_exchange` compares
a further corpus of more than 200 captures from random legal play, also exactly.

`heuristics/exchange.py` carries a compiled kernel and a readable Python twin,
selected at runtime by `compiled_see_enabled`. They share
`variable_material_total` deliberately — the valuation already has its own
reference and tests — so a parity failure can only mean the series or the
backward induction diverged, which is the part this issue actually invented.

## Quiescence cost

Depth 2, per-move work against the same position with quiescence off. "Same
move" counts positions where the mode chose what the quiescence-off baseline
chose.

| mode | median | min | max | total | captures examined | SEE skips | delta skips | same move |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flat / quiescence | 1.75 | 0.85 | 2.58 | 1.69 | 1156 | 0 | 0 | 5/7 |
| flat / + SEE | 1.61 | 0.87 | 2.47 | 1.63 | 962 | 0 | 0 | 5/7 |
| flat / + delta, margin 0 | 1.84 | 0.87 | 2.34 | 1.68 | 1035 | 0 | 121 | 5/7 |
| flat / + delta, margin 1 | 1.84 | 0.87 | 2.65 | 1.74 | 1156 | 0 | 0 | 5/7 |
| flat / + SEE + delta 0 | 1.61 | 0.87 | 2.19 | **1.56** | 841 | 0 | 121 | 5/7 |
| variable / quiescence | 2.00 | 0.84 | **5.50** | **2.54** | 1839 | 0 | 0 | 6/7 |
| variable / + SEE | 2.00 | 0.82 | 4.61 | 2.33 | 1462 | 101 | 0 | 6/7 |
| variable / + delta, margin 0 | 1.76 | 0.85 | 3.33 | 1.85 | 907 | 0 | 495 | 6/7 |
| variable / + delta, margin 1 | 2.00 | 0.85 | 5.52 | 2.56 | 1733 | 0 | 106 | 6/7 |
| variable / + SEE + delta 0 | 1.76 | 0.82 | **2.57** | **1.67** | 698 | 78 | 373 | 6/7 |

The headline the issue asked for: with variable material quiescence costs 2.54×
total work and 5.50× on the worst position; with the exchange filter and
material-only delta pruning that becomes 1.67× total and 2.57× worst, with 62%
fewer captures examined. Per-position ratios for those two rows are
1.69/2.00/0.84/2.29/1.23/5.50/2.80 before and
1.76/2.00/0.82/1.57/1.23/2.57/1.80 after: the gain is concentrated on the
expensive positions, and one position is slightly worse because the survey is
paid at every quiescence node whether or not it finds anything to skip. One
position costs *less* than 1.0 with quiescence on, which is not an anomaly —
better leaf values cut the tree above them.

Three results worth naming:

- **SEE pruning never fires with flat material** (0 skips, every row). That is
  not a tuning failure, it is the two-valued swing proved in EXCHANGE.md: every
  exchange is one piece for one piece, so no capture is ever losing. The flat
  SEE rows measure the *ordering* effect and the survey's cost, nothing else —
  and the ordering alone still drops captures examined from 1156 to 962 by
  cutting off sooner.
- **The calibrated delta margin is inert.** `delta_margin=1` is an allowance of
  237.4 evaluator units against 100 per piece with the adopted weights, so it
  prunes 0 captures flat and 106 variable and leaves the multiplier unchanged.
  Only `delta_margin=0`, which prunes on material alone and has no allowance for
  the positional terms at all, does real work. That is an honest negative result
  about the margin's calibration, and mirrors #66's finding about the futility
  margin on evolved weights.
- Quiescence changed the chosen move on 2 of 7 positions (flat) and 1 of 7
  (variable); the filters did not change it on any further position.

## Ordering quality

Depth 3, quiescence off, killers and history on in every mode.

| mode | nodes | work | cutoffs | first-move cutoffs | first-move rate |
|---|---:|---:|---:|---:|---:|
| flat / killers+history | 37,289 | 74,507,942 | 1,325 | 939 | 0.709 |
| flat / MVV-LVA | 37,289 | 74,695,074 | 1,325 | 939 | 0.709 |
| flat / SEE | 37,283 | 74,886,575 | 1,325 | 939 | 0.709 |
| flat / SEE + MVV-LVA | 37,283 | 75,073,707 | 1,325 | 939 | 0.709 |
| variable / killers+history | 36,068 | 71,594,587 | 1,310 | 1,054 | 0.805 |
| variable / MVV-LVA | 36,063 | 71,767,297 | 1,309 | 1,054 | 0.805 |
| variable / SEE | 36,479 | 72,690,489 | 1,313 | 1,059 | 0.807 |
| variable / SEE + MVV-LVA | 36,479 | 72,875,869 | 1,313 | 1,059 | 0.807 |

**Main-search exchange ordering does not pay for itself.** The first-move cutoff
rate moves by 0.002 with variable material and not at all with flat; nodes are
within 0.1% flat and 1.1% *worse* variable; work is 0.5–1.5% higher in every
case because the survey runs at every ordered node. MVV-LVA is equally inert,
which is consistent with the reason it was left disabled: with a cyclic capture
rule, material value does not decide who wins an exchange.

The likely explanation is that these positions carry two to four captures among
roughly fifty legal moves, so the capture keys rarely decide anything the
existing win / TT / corner-defence / safe-capture keys have not already decided.
A corpus with more captures per node might show something different; this one
does not.

## Equal-time paired play

Each arm plays the candidate against a baseline that differs only in the switch
under test, from all seven positions in both colours, 0.2 s per move, depth
ceiling 20, 40-ply cap, official rules. Wilson intervals are descriptive.

Games run in process, as the issue #60 benchmark's paired play does. The #54
harness freezes **one shared search configuration for every candidate** and
confirms it with a child handshake, so an SEE-versus-no-SEE match cannot be
expressed in its manifest at all. This reuses #54's position generator and
identity record, its Wilson convention and its capped-game adjudication rule.

| arm | material | W/L/unfinished | adjudicated W/L | decisive Wilson 95% | with adjudicated | candidate / baseline median nodes | median depth |
|---|---|---:|---:|---:|---:|---:|---:|
| see-ordering | flat | 3 / 2 / 9 | 5 / 4 | 0.231–0.882 | 0.326–0.786 | 1,590 / 1,625 | 2 / 2 |
| quiescence-see | variable | 4 / 4 / 6 | 4 / 2 | 0.215–0.785 | 0.326–0.786 | 2,269 / 2,237 | 2 / 2 |
| quiescence-delta | flat | 2 / 2 / 10 | 4 / 6 | 0.150–0.850 | 0.214–0.674 | 3,018 / 2,979 | 2 / 2 |

Fourteen games per arm, most of them capped, is not enough to separate anything:
no lower bound reaches the predeclared 0.45, raw or adjudicated. Both sides saw
comparable search — median nodes within 2% and the same median completed depth
in every arm — so equal time was equal, and the intervals are wide because the
sample is small, not because the match was unfair.

The adjudication scores a capped final position with the **shared** static
evaluator, proof and search disabled. In #54 the rule requires two competing
evaluators to agree; here both engines carry the same one, so this is a single
evaluator's opinion about positions its own search steered toward. It is
reported beside the raw outcome, never instead of it.

## What this does not establish

- Seven midgame positions, one machine, one run per cell. Nothing about
  openings, endgames, or positions with few captures.
- No playing-strength claim in either direction. The intervals are consistent
  with a meaningful gain and with a meaningful loss.
- `delta_margin=0` has no allowance for the positional terms at all and is
  unsound with respect to them by construction. It is measured because the
  calibrated margin is inert, not because it is safe.
- Nothing here is tuned. `see_threshold=0` and the two margins are the
  predeclared values; a search over them might find better ones, and would need
  its own held-out positions.
- Python only. Quiescence does not exist in the Rust teacher and neither does
  this; no native claim is made or implied.
- Ordering changes can change which move a capped or selectively pruned search
  returns. Equality of completed unpruned scores is asserted by
  `test_exchange.OrderingTests`, not by this benchmark.

## Adoption

The predeclared rule was: a switch is adopted only if equal-time paired play
gives a decisive-score Wilson lower bound of at least 0.45 **and** the quiescence
work multiplier falls. No arm meets the first condition, so **all six switches
keep their `False`/inert defaults** and the issue's stated permitted outcome —
keeping it disabled — is what happened.

What a follow-up would need: many more paired games per arm at a longer time
control, a corpus that includes dense endgame contact, and a delta margin chosen
by measurement rather than by analogy with the futility allowance. The quiescence
filters are the part worth that effort; main-search exchange ordering is not.
