# The corner-run certificate as a search bound, v1

Issue #70 promotes the forced corner-run certificate from an evaluation term to
an interior search bound, and adds admissible race/horizon tests for branches
where nothing decisive is in reach. Four independently opt-in flags —
`certificate_cutoff_enabled`, `certificate_guard_enabled`,
`race_reduction_enabled` and the existing `certificate_enabled` — **all default
to false.** Nothing here changes the rules, the evaluator's weights, the genome
contract or any recorded outcome, and the bounded proof search remains
unpruned and authoritative.

The [certificate itself](clear_run.py) is unchanged. Everything below is about
when it is asked, what its answer is allowed to mean inside the search, and
what the search may do with it.

## The option had never run

`prove()` returned a certificate result without a `pv` key and `_leaf` reads
`proof['pv']` unconditionally, so `certificate_enabled=True` raised
`KeyError: 'pv'` at the first leaf that certified. The existing tests call
`prove()` directly and never through a search, so this was not covered.
`prove()` now publishes `pv: []`: the certificate argues that a run cannot be
stopped, not which squares it walks, so it has no line to offer and says so.
Read the pre-#70 measurements of `certificate_enabled` with that in mind —
inside the Python search there were none to read.

## Admissibility comes first

Everything here is built on **free-board Chebyshev distance**. One king step
changes each coordinate by at most one, so a square `d` steps away needs at
least `d` moves whatever occupies the board, whatever is captured and whoever
moves. That is an admissible lower bound on arrival, and it is already what
`clear_run._soonest_enemy` and `kernels.no_terminal_win_in_horizon` use.

The route maps in `geometry.py` are **not** admissible for this and nothing
here uses them. `distance_map` and `nearest_distance` read the board as it
stands and allow only the moving code's own captures, so a route blocked by a
piece that will step aside next ply reports a distance *longer* than the one
actually available. That is an overestimate of arrival time. Pruning a branch
because a runner "cannot get there" on a number that can only be too large is
exactly the error the certificate was written to avoid; route distances stay
where they belong, in the weighted evaluation.

## Where the certificate is asked: the race gate

`certify()` sweeps the whole board once per enemy piece and then floods once
per runner. Asking it at every node is the reason it was not already a cutoff.
`clear_run.race_gate` reads the board once and answers a cheaper question:
*which side, if either, could hold a certificate here at all?*

Write `d` for the fewest king steps any of a side's pieces needs to reach its
corner, ignoring the board. Every run of that side takes `moves >= d`, and
`_arrival` is monotone in moves, so `arrival(d)` is the earliest ply any run of
that side could finish on. The gate evaluates, at that most permissive run,
each of the five grounds on which `certify()` already refuses:

- more than `certificate_plies` moves are needed;
- the run does not finish strictly inside the no-capture clock;
- the run does not arrive strictly before the other side's own fastest
  free-board arrival at *its* corner — the same `own_goal_race` comparison,
  over the same pieces, on the same bound;
- the corner is occupied, by anyone: `_safe_run` steps only onto empty squares
  and the corner is a square like any other;
- an enemy could stand on the corner no later than the run arrives, which is
  `soonest[goal] > ply` in `_safe_run`, read at the earliest `ply` available.

Each refusal here is a refusal there, so **the gate never hides a certificate**;
it declines to look for one that cannot exist. The race comparison is strict
and both sides read it from the same two numbers, so at most one side can pass,
and the caller certifies one side rather than two.

The two board conditions do most of the work. The race comparison alone is
close to a coin flip on reachable positions; a held or covered corner is
common. Measured pass rates, per-call costs and the resulting speedup are in
[the bounded validation](../benchmarks/certificate/README.md).

## The certificate as a bound

At an interior node with `ply > 0`, outside a null-move subtree, and with
`depth >= certificate_cutoff_min_depth`, a certified run returns immediately:

```
value = MATE - certified_plies - ply
```

measuring the distance from this node exactly as `terminal_value` measures a
real win, and converted by the existing `from_table`/`to_table` pair. The
score is in the mate range, above `MATE_THRESHOLD`, so every existing
mate-distance comparison in the search treats it as the win it is.

**It is stored as a bound and never as an exact score.** The certificate proves
that the runner cannot be stopped; it does not prove that no faster win exists,
because a stalemate or a capture route could decide the position sooner. The
node is therefore stored `lower` when the side to move holds the run and
`upper` when its opponent does. A transposition-table read of either keeps the
ordinary bound discipline, so no later search can mistake a certified distance
for a proved shortest distance.

What the cutoff returns is not the depth-limited minimax value of the node, and
is not claimed to be. It is a statement about the game, independent of depth,
substituted for an estimate that was bounded by the horizon. That is the point:
it is never contradicted by the true game, and it is not required to agree with
what a shallower search would have guessed.

Three restrictions:

- **The root never cuts.** A score without a move is useless there, so ply zero
  always searches. The leaf term (`certificate_enabled`) is what carries a run
  into a root score; the cutoff is what stops the tree below it.
- **Null-move subtrees never probe.** The side-to-move flip in a null-move
  probe never happened, and `certify()` reads the turn, so a certificate taken
  there would describe a position that does not exist.
- **The PV line ends at a cutoff.** The returned line is empty, as the leaf
  certificate's already is.

A certificate cutoff is a proof, so it does **not** set selective mode: with
only the cutoff and the guard enabled, a mate-range root score is still
reported as `proven` with `score_bound = exact`. The
`"selective search is not a certificate"` distinction is preserved exactly —
`race_reduction_enabled` *is* a selective feature and does join
`selective_mode_early`, so a mate score produced under it reports `unknown`
with that reason and a `selective_` bound, and iterative deepening does not
stop early on it.

## The guard

`certificate_guard_enabled` makes a node where **either** side certifies
ineligible for null-move and futility pruning, and refuses a reduction to every
one of its children.

This is not redundant with the existing board guard. `selective.guarded`
excludes a position with any runner within `max(3, ceil(depth/2))` steps of its
own corner — a proxy for "something is happening near a corner". A certified
run can be seven plies long with every piece four or more steps out, and such a
position passes the board guard today while the game is already decided. The
certificate is the precise version of the test the proxy was approximating.

There is no move-count pruning in this search, so the guard covers null-move,
futility and LMR; the hook is a single `certified` flag on `_reducible` and on
the selective eligibility, so a later move-count rule inherits it by reading the
same flag.

The guard and the cutoff are independent. With the cutoff on, a certified node
returns before the move loop, so the guard's visible effect is the nodes where
the cutoff is not enabled or not yet deep enough to probe.

Measurement narrowed where the reduction half of the guard actually bites.
With `certificate_enabled` on, the children of a certified node mostly come
back as mate scores, and LMR was already refusing them through its existing
`abs(best) < MATE_THRESHOLD` condition. The guard earns its place in the
opposite case: the certificate off at the leaves and the run beyond the
horizon, so the children score as ordinary quiet positions and reductions
apply freely to a branch whose outcome is already settled. `unreduced` counts
the reductions actually prevented, not every refusal, so a search with
reductions off reports zero rather than a flattering number.

## Race and horizon reductions

`race_reduction_enabled` (which requires `lmr_enabled`, and is rejected without
it rather than sitting inert) replaces LMR's `lmr_min_index` guess at nodes
where two admissible arguments both hold for the remaining `depth` plies:

- `kernels.no_terminal_win_in_horizon` — no corner win and no stalemate. The
  existing kernel, unchanged.
- `kernels.no_capture_in_horizon` — no capture. A capture landing at ply `t` is
  a step onto the victim's square, so the pair stood one square apart after
  `t-1` plies; a ply changes a pair's Chebyshev distance by at most one and
  pieces are never created, so both pieces are on the board now and no more
  than `t` apart. Keeping every ordered predator/prey pair strictly further
  apart than `depth` therefore rules out every capture inside the horizon,
  whoever moves and wherever they move. Same-colour, same-type and
  reversed-type pairs are skipped: they cannot take each other.

Modelling draws are excluded separately, since a repetition or the eightieth
noncapture would end the branch inside the horizon.

Both true means no tactic in the branch can land inside the horizon. **It does
not mean the branch has no value**: the positional terms keep moving, and the
leaves still evaluate. So it licenses a reduction — with the re-search on an
alpha improvement that LMR always does — and never a cutoff. What changes is
the justification: `lmr_min_index` is a statistical guess that late moves
matter less, and here it is replaced by an argument about what is reachable.
The first move of a node is still searched in full.

The test is only consulted where LMR is consulted, at `depth >= lmr_min_depth`
and `ply > 0`, and it only ever grants a reduction the index rule would have
refused. In practice it therefore does nothing below a five-ply search: at
shallower targets the only nodes deep enough cut off at their first move.

## Work accounting

Every probe is charged before it runs, so a search under a work limit pays for
its certificates rather than getting them free:

| Step | Charge |
| --- | --- |
| Race gate | `81` — one board pass |
| `certify` after a gate pass | `81 * pieces` — a board sweep per piece |
| Race draw check | the occurrence map, before it is scanned |
| `no_terminal_win_in_horizon` | `3 * 81` — up to three board passes |
| `no_capture_in_horizon` | `81` plus the piece pairs it actually examined |

The old flat `2 * 81` for a certificate under-counted a pass that is linear in
the piece count and charged it whether or not the argument ran. Because the
gate refuses the large majority of nodes, the new scheme costs less on average
as well as being closer to the truth. `no_capture_in_horizon` returns the pair
count it examined so the caller charges what was spent rather than a constant,
following `features.interception_plies`.

`certificate_cutoff_min_depth` (integer 1..32, default 2) is the second cost
control: it stops the tree paying for probes at nodes whose subtrees are too
small to be worth cutting. The sweep behind the default is in the validation.

## Configuration, diagnostics and the native wire form

`SearchConfig` gains `certificate_cutoff_enabled`, `certificate_cutoff_min_depth`,
`certificate_guard_enabled` and `race_reduction_enabled`. All four enter
`to_dict()`/`identity()`, so they scope the transposition table, invalidate
reused search state and appear in checkpoint and tournament manifests.
`search_version` stays `intransitive-selective-v1` and every existing JSON
configuration still loads with the new flags off.

`SearchResult.certificate` reports `probes`, `gated`, `certified`, `cutoffs`,
`guards`, `unreduced`, `race_probes`, `race_quiet` and `race_reductions`, plus
the effective settings, the seconds spent in the certificate module and its
call count. `probes`/`gated`/`certified` count every certificate probe, at
leaves under `certificate_enabled` as well as at interior nodes, so a leaf-only
configuration reports its probe cost too; `cutoffs` counts interior returns
only. The analysis CLI accepts `--certificate/--no-certificate`,
`--certificate-cutoff/--no-certificate-cutoff` and
`--certificate-guard/--no-certificate-guard`.

The native wire form appends `certificate_cutoff_enabled
certificate_cutoff_min_depth certificate_guard_enabled` after the existing
certificate flag, as one group of three. The group travels whole, so an
incomplete request is rejected rather than run with the binary's own defaults,
and an old binary rejects the longer form instead of ignoring the tail.
`RustTeacher.analyze` takes the three matching keyword arguments, and the
search response gains a `certificate` object with the same counters.

## Parity, and one asymmetry

`clear_run.py` and `clear_run.rs` stay line-for-line comparable: `race_gate`
mirrors `race_gate`, and `certify` is untouched in both. The native search
gains the same interior cutoff, with the same bound qualification, the same
root and hypothetical-position exclusions, and the same guard over null-move
and futility.

The asymmetry is deliberate: **the native search has no late move reductions**,
so `race_reduction_enabled` has nothing to attach to there and is not sent over
the wire. The two `kernels` tests it depends on are Python-only for the same
reason. If reductions are ever added natively, the race test is what should
gate them.

Cross-backend, a certified run is an integer distance both implementations
either find or do not, and it does not depend on tree shape, ordering or the
table; that is what `test_the_certificate_decides_the_same_positions_in_both_backends`
compares. Heuristic leaf scores are not compared, and no node-count equivalence
is promised.

## Evidence and rollback

[Bounded validation](../benchmarks/certificate/README.md), with the
[predeclared plan](../benchmarks/certificate/plan.json). Tests live in
`intransitive/tests/test_certificate_bound.py`, with the cross-backend pair in
`test_rust_teacher.py` and the certificate's own soundness corpus still in
`test_clear_run.py`.

Cutoff soundness is checked the way the certificate's own tests check it: every
cutoff-derived mate score is re-searched full width to the claimed distance
with the certificate off, where alpha-beta returns the true minimax value, so a
mate-range score confirms the claim and anything else refutes it.
`no_capture_in_horizon` is checked against an exhaustive `depth`-ply search
rather than against fixtures alone.

Rollback: set the flags false and restart the opponent, or load any earlier
preset. No rules, action encoding, PGN notation or training-record format
changes, and no live generator, trainer or checkpoint opponent is touched.
