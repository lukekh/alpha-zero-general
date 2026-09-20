# Move-ordering measurements (issue #69)

Bounded measurements for the four opt-in ordering mechanisms described in the
[move-ordering guide](../../heuristics/MOVE_ORDERING.md): counter moves,
continuation history, history aging and internal iterative deepening/reduction.
All four default to false. **Nothing here supports changing a default.**

## Summary

| mechanism | does it fire? | fixed-depth effect | recommendation |
| --- | --- | --- | --- |
| counter moves | yes: 648 cutoffs attributed at depth 4, 5,849 at depth 6 | ±0.03% nodes; 0.2% *worse* under LMR at depth 6 | keep off |
| continuation history (1 or 2 plies) | yes: 90k updates at depth 4, 1.9M at depth 6 | ±0.02% nodes | keep off |
| history aging | yes | ±0.05% nodes | keep off |
| IIR, `deepen` | only once the tree outgrows the table | **−18.7% nodes at depth 6**, value unchanged | keep off, but the one worth revisiting |
| IIR, `reduce` | same condition | −77% nodes, but returns shallower values | keep off; selective |

Two things are worth taking away. First, the ordering this issue proposes to
strengthen is already strong: killers plus history take **80.6%** of beta
cutoffs on the first move searched at depth 4 and **97.8%** at depth 6, so
counter moves, continuation history and aging have almost nothing left to
recover, and the measurements say so. Second, the premise that a high LMR
re-search rate points at weak ordering does not reproduce here — the rate was
0.3% to 1.8% before any change was made, not the 8.3% the issue quotes.

The single mechanism that pays is internal iterative deepening, and what it
actually repairs is a transposition table too small for the tree rather than a
weak move order. The equal-time games do not establish a strength difference
either way, so nothing here supports changing a default.

## Reproduce

```sh
uv sync --locked
ORT_DISABLE_TELEMETRY=1 uv run python -m intransitive.heuristics.ordering_benchmark \
  --output /tmp/ordering-depth-4.json fixed --depth 4 --positions 16 \
  --games 8 --plies 40 --seed 911 --genome adopted --schedules plain lmr lmr-pvs
ORT_DISABLE_TELEMETRY=1 uv run python -m intransitive.heuristics.ordering_benchmark \
  --output /tmp/ordering-depth-6.json fixed --depth 6 --positions 8 \
  --games 8 --plies 40 --seed 911 --genome core --schedules plain lmr
ORT_DISABLE_TELEMETRY=1 uv run python -m intransitive.heuristics.ordering_benchmark \
  --output /tmp/ordering-small-table.json fixed --depth 4 --positions 16 \
  --games 8 --plies 40 --seed 911 --genome adopted --schedules plain \
  --table-entries 256 --rows iir-reduce iir-deepen all-ordering counter-move
ORT_DISABLE_TELEMETRY=1 uv run python -m intransitive.heuristics.ordering_benchmark \
  --output /tmp/ordering-equal-time.json equal-time --genome core \
  --challenger all-ordering --incumbent killers-history \
  --seconds 2 --openings 8 --max-plies 120 --seed 2026
ORT_DISABLE_TELEMETRY=1 uv run python -m intransitive.heuristics.ordering_benchmark \
  table /tmp/ordering-depth-4.json
```

Reports are saved under [evidence/](evidence/). `table` renders any saved report
as the markdown published below. The fixed-depth runs use the default 10,000
transposition-table entries unless the command says otherwise.

The corpus is generated from the seed, so it needs no data file. Positions are
recorded by state hash in every report. The fixed-depth reports are exactly
reproducible: identical settings give identical actions, scores, nodes and work.
The equal-time report is not, because a wall-clock deadline decides how much
search each move completes.

## Fixed-depth ablation, adopted genome, depth 4

One search per position per row, fresh engine per position, 16 positions, the
adopted default evaluator (`advantage=23.97`, `attack=25.71`, `defence=32.56`,
flat material 100, attack and defence enabled). Every row shares that evaluator;
only ordering and search settings differ. `killers-history` is the baseline the
issue names. `fmc` is the first-move cutoff rate and `idx` the mean 0-based
cutoff index.

| row | mechanisms | nodes | vs base | fmc | idx | cutoffs | from TT/killer/counter/other | LMR reduced | re-searched | rate | inert |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `killers-history` | — | 577,245 | 1.0000 | 0.8062 | 0.3412 | 59,912 | 2134/48088/0/9690 | 0 | 0 | — | none |
| `counter-move` | counter_move | 577,169 | 0.9999 | 0.8071 | 0.3390 | 59,939 | 2136/48000/648/9155 | 0 | 0 | — | none |
| `continuation-1` | continuation, continuation_plies | 577,209 | 0.9999 | 0.8061 | 0.3410 | 59,914 | 2134/48078/0/9702 | 0 | 0 | — | none |
| `continuation-2` | continuation, continuation_plies | 577,131 | 0.9998 | 0.8061 | 0.3402 | 59,912 | 2134/48026/0/9752 | 0 | 0 | — | none |
| `history-aging` | history_aging | 576,983 | 0.9995 | 0.8066 | 0.3399 | 59,915 | 2134/48017/0/9764 | 0 | 0 | — | none |
| `iir-reduce` | iir, iir_min_depth, iir_mode | 577,245 | 1.0000 | 0.8062 | 0.3412 | 59,912 | 2134/48088/0/9690 | 0 | 0 | — | iir_enabled |
| `iir-deepen` | iir, iir_min_depth, iir_mode | 577,245 | 1.0000 | 0.8062 | 0.3412 | 59,912 | 2134/48088/0/9690 | 0 | 0 | — | iir_enabled |
| `counter+continuation` | continuation, continuation_plies, counter_move | 577,191 | 0.9999 | 0.8065 | 0.3393 | 59,984 | 2136/48055/647/9146 | 0 | 0 | — | none |
| `all-ordering` | continuation, continuation_plies, counter_move, history_aging, iir, iir_min_depth, iir_mode | 576,919 | 0.9994 | 0.8065 | 0.3395 | 59,906 | 2136/47970/650/9150 | 0 | 0 | — | iir_enabled |
| `killers-history+lmr` | — | 336,150 | 1.0000 | 0.8490 | 0.2382 | 52,752 | 1422/44244/0/7086 | 1,373 | 24 | 0.0175 | none |
| `counter-move+lmr` | counter_move | 335,458 | 0.9979 | 0.8497 | 0.2358 | 52,651 | 1424/44114/529/6584 | 1,374 | 24 | 0.0175 | none |
| `continuation-1+lmr` | continuation, continuation_plies | 335,774 | 0.9989 | 0.8486 | 0.2381 | 52,698 | 1422/44193/0/7083 | 1,373 | 23 | 0.0168 | none |
| `continuation-2+lmr` | continuation, continuation_plies | 336,031 | 0.9996 | 0.8491 | 0.2370 | 52,751 | 1422/44263/0/7066 | 1,373 | 24 | 0.0175 | none |
| `history-aging+lmr` | history_aging | 335,687 | 0.9986 | 0.8498 | 0.2366 | 52,705 | 1423/44196/0/7086 | 1,373 | 24 | 0.0175 | none |
| `iir-reduce+lmr` | iir, iir_min_depth, iir_mode | 336,150 | 1.0000 | 0.8490 | 0.2382 | 52,752 | 1422/44244/0/7086 | 1,373 | 24 | 0.0175 | iir_enabled |
| `iir-deepen+lmr` | iir, iir_min_depth, iir_mode | 336,150 | 1.0000 | 0.8490 | 0.2382 | 52,752 | 1422/44244/0/7086 | 1,373 | 24 | 0.0175 | iir_enabled |
| `counter+continuation+lmr` | continuation, continuation_plies, counter_move | 335,881 | 0.9992 | 0.8491 | 0.2366 | 52,752 | 1422/44232/523/6575 | 1,373 | 24 | 0.0175 | none |
| `all-ordering+lmr` | continuation, continuation_plies, counter_move, history_aging, iir, iir_min_depth, iir_mode | 335,512 | 0.9981 | 0.8498 | 0.2359 | 52,703 | 1423/44170/528/6582 | 1,373 | 24 | 0.0175 | iir_enabled |
| `killers-history+lmr-pvs` | — | 436,224 | 1.0000 | 0.8141 | 0.3226 | 51,825 | 2405/40665/0/8755 | 1,264 | 21 | 0.0166 | none |
| `counter-move+lmr-pvs` | counter_move | 436,503 | 1.0006 | 0.8139 | 0.3222 | 51,839 | 2407/40673/599/8160 | 1,265 | 22 | 0.0174 | none |
| `continuation-1+lmr-pvs` | continuation, continuation_plies | 436,103 | 0.9997 | 0.8140 | 0.3225 | 51,828 | 2404/40653/0/8771 | 1,265 | 21 | 0.0166 | none |
| `continuation-2+lmr-pvs` | continuation, continuation_plies | 436,283 | 1.0001 | 0.8134 | 0.3229 | 51,859 | 2404/40705/0/8750 | 1,264 | 21 | 0.0166 | none |
| `history-aging+lmr-pvs` | history_aging | 435,267 | 0.9978 | 0.8148 | 0.3242 | 51,775 | 2396/40564/0/8815 | 1,265 | 21 | 0.0166 | none |
| `iir-reduce+lmr-pvs` | iir, iir_min_depth, iir_mode | 436,224 | 1.0000 | 0.8141 | 0.3226 | 51,825 | 2405/40665/0/8755 | 1,264 | 21 | 0.0166 | iir_enabled |
| `iir-deepen+lmr-pvs` | iir, iir_min_depth, iir_mode | 436,224 | 1.0000 | 0.8141 | 0.3226 | 51,825 | 2405/40665/0/8755 | 1,264 | 21 | 0.0166 | iir_enabled |
| `counter+continuation+lmr-pvs` | continuation, continuation_plies, counter_move | 436,294 | 1.0002 | 0.8138 | 0.3213 | 51,839 | 2406/40689/591/8153 | 1,265 | 22 | 0.0174 | none |
| `all-ordering+lmr-pvs` | continuation, continuation_plies, counter_move, history_aging, iir, iir_min_depth, iir_mode | 435,078 | 0.9974 | 0.8150 | 0.3219 | 51,760 | 2397/40558/577/8228 | 1,265 | 22 | 0.0174 | iir_enabled |

Nothing here moves the tree. Every mechanism lands within 0.06% of the baseline
on the plain schedule; the best combination saves 0.26% under LMR+PVS. The
counter move does what it is supposed to do — 648 cutoffs are attributed to it
and `cutoff_from_other` falls from 9,690 to 9,155 — and the tree is the same
size anyway, because the cutoffs it claims were already being found a fraction
of a move later. Killers alone account for about four cutoffs in five.

Both IIR rows are byte-identical to the baseline with `iir_nodes = 0`, even
after the ladder lowered `iir_min_depth` to 3 so the depth is reachable below
the root. See [When IIR can fire](#when-iir-can-fire).

## Fixed-depth ablation, core genome, depth 6

The depth-4 tree never outgrows the transposition table, so it cannot reach
internal iterative deepening, and it is too shallow for ordering to compound.
This run uses the documented core preset —
material and piece-type advantage only, no route analysis — which is an order of
magnitude cheaper per node, so depth 6 is affordable on 8 positions.

| row | mechanisms | nodes | vs base | fmc | idx | cutoffs | from TT/killer/counter/other | LMR reduced | re-searched | rate | inert |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `killers-history` | — | 11,182,964 | 1.0000 | 0.9780 | 0.0451 | 1,530,563 | 20581/1086222/0/423760 | 0 | 0 | — | none |
| `counter-move` | counter_move | 11,180,004 | 0.9997 | 0.9780 | 0.0445 | 1,530,651 | 20583/1085803/5849/418416 | 0 | 0 | — | none |
| `continuation-1` | continuation, continuation_plies | 11,182,493 | 1.0000 | 0.9780 | 0.0451 | 1,530,563 | 20638/1086159/0/423766 | 0 | 0 | — | none |
| `continuation-2` | continuation, continuation_plies | 11,182,115 | 0.9999 | 0.9780 | 0.0451 | 1,530,547 | 20638/1086130/0/423779 | 0 | 0 | — | none |
| `history-aging` | history_aging | 11,182,882 | 1.0000 | 0.9780 | 0.0451 | 1,530,575 | 20547/1086293/0/423735 | 0 | 0 | — | none |
| `iir-reduce` | iir, iir_min_depth, iir_mode | 2,533,309 | 0.2265 | 0.9615 | 0.0773 | 149,278 | 21276/88184/0/39818 | 0 | 0 | — | none |
| `iir-deepen` | iir, iir_min_depth, iir_mode | 9,091,098 | 0.8129 | 0.9762 | 0.0512 | 1,247,641 | 42156/899441/0/306044 | 0 | 0 | — | none |
| `counter+continuation` | continuation, continuation_plies, counter_move | 11,180,182 | 0.9998 | 0.9780 | 0.0445 | 1,530,650 | 20637/1085700/5836/418477 | 0 | 0 | — | none |
| `all-ordering` | continuation, continuation_plies, counter_move, history_aging, iir, iir_min_depth, iir_mode | 9,090,221 | 0.8129 | 0.9762 | 0.0512 | 1,247,638 | 42123/899519/4766/301230 | 0 | 0 | — | none |
| `killers-history+lmr` | — | 2,889,789 | 1.0000 | 0.9431 | 0.1010 | 174,592 | 8223/116494/0/49875 | 49,923 | 147 | 0.0029 | none |
| `counter-move+lmr` | counter_move | 2,895,666 | 1.0020 | 0.9430 | 0.1011 | 174,363 | 8216/116101/849/49197 | 49,782 | 147 | 0.0030 | none |
| `continuation-1+lmr` | continuation, continuation_plies | 2,889,888 | 1.0000 | 0.9431 | 0.1010 | 174,592 | 8222/116495/0/49875 | 49,923 | 147 | 0.0029 | none |
| `continuation-2+lmr` | continuation, continuation_plies | 2,889,897 | 1.0000 | 0.9431 | 0.1010 | 174,597 | 8221/116480/0/49896 | 49,923 | 147 | 0.0029 | none |
| `history-aging+lmr` | history_aging | 2,889,745 | 1.0000 | 0.9431 | 0.1010 | 174,545 | 7953/116710/0/49882 | 49,923 | 147 | 0.0029 | none |
| `iir-reduce+lmr` | iir, iir_min_depth, iir_mode | 1,474,517 | 0.5103 | 0.9625 | 0.0560 | 103,856 | 8652/65749/0/29455 | 31,355 | 20 | 0.0006 | none |
| `iir-deepen+lmr` | iir, iir_min_depth, iir_mode | 2,698,015 | 0.9336 | 0.9446 | 0.1005 | 168,550 | 20908/102614/0/45028 | 44,905 | 136 | 0.0030 | none |
| `counter+continuation+lmr` | continuation, continuation_plies, counter_move | 2,895,765 | 1.0021 | 0.9431 | 0.1011 | 174,370 | 8217/116106/847/49200 | 49,782 | 147 | 0.0030 | none |
| `all-ordering+lmr` | continuation, continuation_plies, counter_move, history_aging, iir, iir_min_depth, iir_mode | 2,697,274 | 0.9334 | 0.9446 | 0.1005 | 168,511 | 20617/102859/738/44297 | 44,900 | 136 | 0.0030 | none |

At depth 6 the baseline ordering is better still: **97.8%** of beta cutoffs are
taken on the first move searched, at a mean cutoff index of 0.045. Counter
moves, continuation history and aging again change nothing measurable — the
largest effect in the whole table is 0.03%, and `counter-move+lmr` is 0.2%
*worse* than the baseline.

Internal iterative deepening is the exception. It fires 2,055 times here,
because an 11-million-node tree evicts a 10,000-entry transposition table
heavily enough that nodes genuinely arrive without a stored move, and it
removes **18.7%** of
the tree while leaving the completed value exactly where it was. Under LMR the
saving is 6.6%. The reduction mode removes far more — 77% of the tree — but it
returns shallower values and is selective, so that number is not comparable.

## Beta-cutoff distribution by depth

The aggregate first-move cutoff rate is dominated by the shallowest nodes, which
are the great majority. Split by remaining depth (`cutoffs` / first-move rate /
mean index), baseline against the full ladder:

| run | remaining depth | `killers-history` | `all-ordering` |
| --- | ---: | --- | --- |
| depth 4, adopted | 1 | 56,632 / 0.812 / 0.329 | 56,622 / 0.812 / 0.328 |
| depth 4, adopted | 2 | 2,481 / 0.689 / 0.533 | 2,485 / 0.689 / 0.533 |
| depth 4, adopted | 3 | 799 / 0.787 / 0.585 | 799 / 0.787 / 0.588 |
| depth 6, core | 1 | 1,431,912 / 0.983 / 0.039 | 1,168,813 / 0.980 / 0.047 |
| depth 6, core | 2 | 68,731 / 0.902 / 0.128 | 54,749 / 0.907 / 0.120 |
| depth 6, core | 3 | 28,422 / 0.938 / 0.143 | 22,670 / 0.946 / 0.118 |
| depth 6, core | 4 | 1,091 / 0.811 / 0.357 | 999 / **0.946** / 0.193 |
| depth 6, core | 5 | 407 / 0.708 / 0.585 | 407 / **0.978** / 0.081 |

This is the one place the mechanisms show a clear effect, and it is exactly
where internal iterative deepening operates. At remaining depths 4 and 5 — the
deep, TT-missing nodes IIR exists for — the first-move cutoff rate rises from
0.81 to 0.95 and from 0.71 to 0.98, and the mean cutoff index falls from 0.59 to
0.08. Those are the nodes whose subtrees are expensive, which is why an 18.7%
node saving comes out of a distribution that barely moves in aggregate. At
remaining depths 1 to 3 the ordering is unchanged to three decimal places.

## When IIR can fire

Internal iterative deepening exists for a node that has no move to try first.
In this engine that is rare by construction: iterative deepening visits every
node near the root at every shallower depth, and the transposition table and its
hints then hand the deeper iteration a move. The condition only arises once the
tree outgrows the table and entries are evicted — which is why IIR is completely
inert in the depth-4 run above (10,000 entries against a 577,000-node tree) and
fires 2,055 times per corpus in the depth-6 run (10,000 entries against an
11,000,000-node tree).

The same effect is reproducible on the shallow corpus by shrinking the table.
Depth 4, adopted genome, 16 positions, **256 table entries**:

| row | mechanisms | nodes | vs base | fmc | idx | cutoffs | from TT/killer/counter/other | LMR reduced | re-searched | rate | inert |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `killers-history` | — | 734,463 | 1.0000 | 0.8485 | 0.2797 | 76,880 | 370/56386/0/20124 | 0 | 0 | — | none |
| `counter-move` | counter_move | 730,418 | 0.9945 | 0.8482 | 0.2780 | 76,905 | 370/56328/739/19468 | 0 | 0 | — | none |
| `iir-reduce` | iir, iir_min_depth, iir_mode | 398,101 | 0.5420 | 0.6019 | 0.6866 | 7,483 | 373/4312/0/2798 | 0 | 0 | — | none |
| `iir-deepen` | iir, iir_min_depth, iir_mode | 575,701 | 0.7838 | 0.8065 | 0.3449 | 55,810 | 1095/44301/0/10414 | 0 | 0 | — | none |
| `all-ordering` | continuation, continuation_plies, counter_move, history_aging, iir, iir_min_depth, iir_mode | 573,352 | 0.7806 | 0.8082 | 0.3389 | 55,734 | 1095/44234/642/9763 | 0 | 0 | — | none |

With a 256-entry table the baseline costs 734,463 nodes, against 577,245 with
the default 10,000 entries: the undersized table is worth about 27% extra work.
Internal iterative deepening brings that back to **575,701**, which is within
0.3% of the well-sized-table baseline. That is the honest description of what it
buys here — it recovers most of what a transposition table too small for the
tree loses, rather than improving on a table that is already large enough.

The reduction mode's 0.54 node ratio in that table is not a like-for-like
saving: it returns values from a shallower search, its beta cutoffs fall from
76,880 to 7,483, and its first-move cutoff rate drops from 0.85 to 0.60. It is
selective, and it is priced as such.

## LMR re-search rate

The issue names 71 re-searches per 859 reductions (8.3%) on a depth-4 sample as
the baseline to beat, and takes the high rate as evidence that ordering is weak.
That rate does not reproduce on this corpus, before or after the change:

| run | schedule | reductions | re-searches | rate |
| --- | --- | ---: | ---: | ---: |
| depth 4, adopted | `killers-history+lmr` | 1,373 | 24 | **0.0175** |
| depth 4, adopted | `all-ordering+lmr` | 1,373 | 24 | 0.0175 |
| depth 4, adopted | `killers-history+lmr-pvs` | 1,264 | 21 | 0.0166 |
| depth 4, adopted | `all-ordering+lmr-pvs` | 1,265 | 22 | 0.0174 |
| depth 6, core | `killers-history+lmr` | 49,923 | 147 | **0.0029** |
| depth 6, core | `all-ordering+lmr` | 44,900 | 136 | 0.0030 |
| depth 6, core | `iir-deepen+lmr` | 44,905 | 136 | 0.0030 |

The rate was already between 0.3% and 1.8% before any ordering change, and the
ordering changes leave it there. LMR also reduces the tree rather than enlarging
it: 577,245 to 336,150 nodes at depth 4 (−41.8%) and 11,182,964 to 2,889,789 at
depth 6 (−74.2%). The measurement that motivated this issue differed in genome,
positions and PVS setting, so the numbers are not directly comparable — but on
the corpus measured here there is no re-search problem for better ordering to
solve.

## Equal-time games

Paired official games, core genome, fresh engine per move, both engines given
the same per-move deadline. One opening played from both colours is one paired
unit; the interval is a conservative 95% Hoeffding bound over those units and is
wide by construction. Unresolved games (the ply cap) score half and are reported
separately.

### `all-ordering` vs `killers-history`, 2 s per move, 8 openings

| challenger | incumbent | games | W/D/L | unresolved | score | 95% interval |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `all-ordering` | `killers-history` | 16 | 2/14/0 | 0 | 0.562 | [0.082, 1.000] |

| side | moves | mean completed depth | fallbacks | nodes | iir nodes | counter updates | continuation updates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `all-ordering` | 245 | **4.282** | 0 | 68,520,156 | 1,597 | 1,039,436 | 2,050,745 |
| `killers-history` | 243 | 4.070 | 0 | 58,855,357 | 0 | 0 | 0 |

Two wins, no losses, and every mechanism fired. The interval includes 0.5, so
this does **not** establish a strength difference; what it does establish is
that the ladder is not a regression and that it buys **+0.21 completed depth**
at the same deadline. Both wins came from the two longest games (51 and 75
plies); the remaining games are short repetition draws, which is what two
deterministic engines of near-equal strength produce here.

### `iir-deepen` vs `killers-history`, 5 s per move, 4 openings

A deeper control, to check whether the node saving converts into strength when
the searches are long enough for IIR to matter.

| challenger | incumbent | games | W/D/L | unresolved | score | 95% interval |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `iir-deepen` | `killers-history` | 8 | 0/8/0 | **8** | 0.500 | [0.000, 1.000] |

| side | moves | mean completed depth | fallbacks | nodes | iir nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `iir-deepen` | 320 | 4.812 | 0 | 327,465,852 | 19,483 |
| `killers-history` | 320 | 4.875 | 0 | 336,521,191 | 0 |

**Every one of these eight games hit the 80-ply cap**, so the 0.500 score
carries no strength information at all — at this deadline both engines search
deeply enough to avoid the early repetitions that decide the 2 s games, and 80
plies is not enough to finish. The diagnostics are still meaningful: IIR fired
19,483 times and saved 2.7% of nodes, and the mean completed depth came out
**0.06 plies lower**, not higher. A conclusive run at this control needs a ply
cap of 200 and many more openings, which is several hours of wall time.

### What the equal-time evidence supports

Nothing beyond "not a regression". One run gives 2/14/0 with +0.21 completed
depth, the other gives eight capped draws with −0.06 completed depth. Two small
samples pointing in opposite directions on the depth margin is exactly the state
in which a default must not change, and none does.

## Firing rates

Every row records `inert_mechanisms`: techniques that the configuration enables
and that then execute zero times. This is #66's fifth proposal applied to this
benchmark, and it caught a real instance during development — with the default
`iir_min_depth=4`, internal iterative deepening can never fire in a depth-4
search, because the deepest non-root node has only three plies remaining. The
ladder therefore sets `iir_min_depth=3` for its IIR rows, and the reports carry
the firing counts rather than leaving a zero to be found by hand later.

## Validation

```sh
ORT_DISABLE_TELEMETRY=1 uv run python -m unittest intransitive.tests.test_move_ordering -v
ORT_DISABLE_TELEMETRY=1 uv run python -m unittest discover -s intransitive/tests -v
```

`test_move_ordering` covers defaults and bounds, the `ordering_enabled`
interlock, exact inertness when the flags are off, the kernel's ranking of each
new key, that no legal move is ever dropped, that each mechanism fires and
updates its own counters, that gravity bounds every statistic, that both IIR
modes fire and that only the reduction declares itself a non-certificate, that
the whole sound ladder returns plain alpha-beta's completed score and depth,
that the tables are cleared per search and survive concurrent processes, and
that `prove` (including its proof node count) and `certificate` are byte-for-byte
unchanged by every mechanism.

The broad run's non-passing tests are the repository's existing known failures —
four `test_game_blunders` depth-3 cases, the `test_greedy_process` spawn
comparison, two `test_search_performance` cases and the
`test_compiled_proof` work-interruption case, whose `prove` patch does not
accept the `stats` keyword that issue #72's certificate statistics added to the
leaf call. Every one of them reproduces identically on an untouched
`git archive` of `origin/master`, with the same test names and the same
messages, so this change adds no failure and fixes none.

`test_material.variants()` gained entries for `counter_move_enabled`,
`continuation_enabled`, `iir_reduction` and `iir_mode`, on the same terms as the
`razoring_enabled` and `race_reduction_enabled` entries already there: a field
whose validity depends on a sibling moves that sibling with it, and a two-valued
enum moves within its own range. The measurements above were re-run after merging
master's exchange, shallow-pruning and certificate work and reproduce
bit-for-bit: the new ranking keys sit beside the exchange key without disturbing
it, and every default path is untouched.

Inertness was additionally checked outside the test suite, by running one probe
against the merge-base checkout and the same probe against this branch: 48
paired searches over 12 positions and four configurations (plain, ordering,
ordering + MVV-LVA, ordering + LMR + PVS) at depth 4, matching exactly on
action, score, nodes, work, completed depth, principal variation and both LMR
counters.

## Limitations

- One corpus, one seed, two genomes, one machine. These are bounded
  observations on these fixtures, not general strength or throughput claims.
- Both engines in an equal-time game are deterministic, so the opening pool is
  the only source of variation and the intervals are conservative Hoeffding
  bounds over a small number of paired openings. They are wide by construction.
- #54's paired harness gives every candidate the same search settings by
  design — candidates differ by genome, not by search switches — so it cannot
  play an ordering A/B directly. The equal-time runs here use the same paired
  colours, the same conservative interval and the same official rules, in a
  self-contained runner. `tournament.spec.protocol()` now accepts the new
  ordering switches, so a future #54 run can freeze them run-wide for every
  candidate; pitting two ordering settings against each other would need a
  per-candidate search override that #54 deliberately does not have.
- The search is Python/Numba only. The Rust teacher keeps its established
  ordering policy, exactly as it does for PVS, aspiration and the selective
  switches.
- The two equal-time runs shared the machine with one other benchmark process.
  Both engines in a game alternate inside one process, so contention affects
  them equally and the comparison stays fair, but the absolute completed depths
  would be higher on an idle machine.
- Apple M1, 8 GiB RAM, macOS 15.6.1 arm64, Python 3.11.4, NumPy 2.4.6,
  Numba 0.67.0. The exact environment is recorded in every report.
