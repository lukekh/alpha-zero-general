# Guarded selective search v1

Issue #60 adds independently opt-in `nmp_enabled` and `futility_enabled` to
Python and Rust. **Both default to false.** Enabled searches are heuristic:
completed selective depth is not exhaustive depth, and their scores/choices can
differ from pruning-disabled alpha-beta. The bounded measurements do not justify
changing any live generator, trainer or checkpoint opponent.

## Eligibility and parameters

Ordinary terminal and legal-move checks run first, preserving official
corner/stalemate precedence over modelling draws. Only non-root, non-PV
(adjacent binary64 window **on entry**, before TT tightening) nodes can prune.
Both window endpoints must be finite and strictly inside the evaluator's
±10,000 clipping range. Proof kernels are never selectively pruned.

Shared Intransitive exclusions:

- Fewer than four pieces or eight legal actions on either side.
- Any currently available capture, on either side. This protects threatened
  retreats and forced capture/defence situations, even if the proposed move
  itself is quiet.
- Any runner within `max(3, ceil(depth/2))` king steps of its own target corner.
- Any repeated board/turn in active history, or a no-capture clock which could
  reach 80 within the current depth plus the hypothetical ply.

These are conservative heuristics, not a theorem excluding every zugzwang.

`nmp_min_depth=3` (integer 3..32), `nmp_reduction=1` (integer 1..8, at most
minimum depth minus two): require static evaluation ≥ beta, flip the side and
search at `depth - 1 - reduction` with `[-beta, nextafter(-beta,+inf)]`.
Both techniques are disabled throughout the null subtree; consecutive null
moves are impossible. Every non-mate fail-high is verified from the **original
position** at `depth - reduction`, same original ply, with
`[nextafter(beta,-inf), beta]`. Both techniques and TT reads/writes are disabled
throughout verification. Only a non-mate verified fail-high cuts off, returning
beta as a selective lower bound and a legal verification line. Reduced-depth
verification can still miss deeper zugzwangs and tactics.

`futility_max_depth=2` (1..2), `futility_margin=1` (finite multiplier
1/16..16): after searching at least one move, skip an eligible quiet child when
`static + depth * multiplier * (count_weight/2 + advantage_weight + 8*pressure_weight)
<= alpha`. Disabled pressure contributes zero. Thus the default material-only
margins are 75/150 evaluator units, versus 100 per piece; pressure 10 yields
155/310. These are experimental margins, not upper bounds on future gains.
Keep every fastest runner (including retreating moves), all captures, moves
within three steps of the goal, and moves with an enemy within two steps of
source or destination. The shared guard also protects immediate quiet wins,
corner defence and low-mobility forced replies. No reverse futility, razoring
or weight tuning is included.
source or destination. Issue #68 additionally refuses any move that vacates or
occupies a square on a shortest enemy king route to the enemy's corner; see
*The reviewed quiet predicate* below. The shared guard also protects immediate
quiet wins, corner defence and low-mobility forced replies.

The experimental allowance separates the terms a quiet ply can move from the
terms it cannot. The shared guard excludes every position where either side has
a capture available, and `quiet` excludes capture moves, so across the first
skipped ply the piece counts are fixed: material and advantage contribute
exactly nothing to that ply's change and are charged only from the second ply,
where the skipped subtree can capture. The module terms stay at the
conservative per-side feature ranges, which bound the evaluation rather than one
ply of it, so `futility_margin` remains the tuning knob and the multiplier is
symmetric around one. At the adopted genome the original allowance left the
technique eligible thousands of times per move and pruning nothing at all. See
the [activation measurement](../benchmarks/selective_activation/README.md) for
the sweep that locates the allowance at which it starts to prune. The original
non-experimental branch is unchanged, because its margins are what #60
validated.

Without the experimental evaluator option below, the conservative margins are
calibrated only for flat material, count=100, advantage=25 and the standard
predator/prey bonuses (1, .5, .25), no attack/defence/overload modules, and
pressure disabled or weight in 0..20 (radius 3 or 4). Enabling either pruning
method with any other scales is a **configuration error** in both backends:
`SearchConfig.__post_init__` and the native `Config::validate` refuse it and
name `selective_evaluator_enabled` as the deliberate opt-in. Neither backend
accepts the flags and then prunes nothing. Issue #66 found that the previous
silent interlock disabled both methods on **every** move of a 37,299-move
tuning run whose frozen protocol declared them, which is indistinguishable from
a permanent off switch. Unsupported scales remain loadable while both methods
are off, and ordinary evaluation is unchanged. Python's legacy
expanded-state/custom-game backend explicitly rejects selective options.
Native's unsupported arguments raise errors.

## Internal iterative reduction (issue #69)

`iir_enabled` with `iir_mode='reduce'` searches a deep node that has no
transposition-table move at `depth - iir_reduction` and stores the result at
that reduced depth, so it returns a shallower value than the caller asked for
and joins the selective family: `selective_mode_early` covers it, results report
`selective_exact` and proof `unknown`, the table uses the selective namespace,
and a mate-range score no longer ends iterative deepening early. It is refused
at the root, inside null and verification subtrees, and — like a late-move
reduction — on a node holding a clear-run certificate under
`certificate_guard_enabled`. `iir_mode='deepen'` discards the shallow value and
keeps only the move, so it is **not** selective. Both, and the three ordering
tables that accompany them, are documented in [MOVE_ORDERING.md](MOVE_ORDERING.md).

## Shallow-depth cutoffs v1 (issue #68)

Four more independently opt-in techniques, **all default false**. Three are
heuristic and join the selective family; the fourth is value preserving and
deliberately does not.

| flag | parameters | eligibility | effect |
|---|---|---|---|
| `razoring_enabled` | `razoring_max_depth` 1..4 (2), `razoring_margin` (0,64] (1.5) | shared guard, plus `quiescence_enabled` | `static + margin <= alpha` runs quiescence; a value still at or below alpha is returned as a selective **upper** bound |
| `reverse_futility_enabled` | `reverse_futility_max_depth` 1..6 (2), `reverse_futility_margin` (0,64] (1.0) | shared guard | `static - margin >= beta` returns beta as a selective **lower** bound, with no probe search |
| `move_count_pruning_enabled` | `move_count_max_depth` 1..8 (3), `move_count_base` 1..64 (12) | shared guard | children from index `base + depth²` onward that are quiet are **skipped**, not reduced |
| `mate_distance_pruning_enabled` | none | non-root, not inside a null or verification search | narrows to `[-MATE+ply, MATE-ply-1]` and cuts when the window closes |

`razoring_enabled` without `quiescence_enabled` is a configuration error, not a
silent downgrade: razoring returns a quiescence value, and without quiescence
it would return the bare static score — a different and far more aggressive
technique that has not been measured.

The first three reuse the issue #60 eligibility block unchanged — non-root,
non-PV on entry, finite window inside the ±10,000 clipping range, a supported
evaluator scale and `selective.guarded()` — so they inherit every Intransitive
exclusion listed above rather than restating it. They are covered by the
certificate guard on the same terms as NMP and futility: a node with a
certified forced run is exempt from all five, and enabling any of them is
enough to make the guard probe for one. Mate-distance pruning is outside that
too, and runs *before* the certificate probe, because narrowing the window
costs nothing while the probe costs board passes. A node eligible only for
move-count pruning pays for no static evaluation, because move-count pruning
asks the evaluator nothing. Counters are `*_eligible` (the node or child
reached the test) and `*_applied` / `*_pruned` (the test fired), plus
`razoring_nodes` for quiescence work charged to razoring. All of it shares the
one move budget.

Every node keeps at least one searched child. Move-count pruning requires a
finite incumbent, which only exists once a child has returned, and forward
futility requires a nonzero index. **Stalemate loses in Intransitive, so a node
emptied of moves would be scored as a draw rather than the loss it is**; that
is why the property is tested directly rather than argued from the thresholds.

### Margins are measured, not inherited

`selective.allowance(config)` is one ply of evaluator units in the current
scales, and every margin in the family — futility, razoring and reverse
futility — is `depth × multiplier × allowance`. For both new margins a **larger
multiplier fires less often** and is the conservative direction.

The multipliers were fixed from the measured distribution of
(depth-*d* alpha-beta value − static evaluation) ÷ allowance over 480 random
positions, at *d* = 1, 2, 3:

| scale | allowance | largest rise per ply | largest fall per ply | shipped razoring / reverse |
|---|---:|---:|---:|---|
| `supported()` | 75.0 | 1.476 | 0.729 | 1.5 / 1.0 |
| adopted route genome | 287.4 | 0.413 | 0.162 | 1.5 / 1.0 |

Both defaults cover every observed swing on **both** scales. They are not
equally *tight* on both: because `allowance()` sums weight ceilings rather than
the swing those modules actually produce, the adopted genome needs roughly a
quarter of the supported scale's multiplier. The margins therefore admit
fractional values, and an evolved genome that wants these techniques to fire at
all should set them explicitly from its own measurement. This is a heuristic
allowance measured on one corpus, **not a bound**; it says nothing about weight
ranges outside the measured ones, and #55 can move the evaluator underneath it.

### The reviewed quiet predicate

Issue #68 required `selective.quiet()` to be re-examined before more techniques
depended on it. Captures, threat creation and the mover's own fastest runners
were already excluded, and `guarded()` already refuses any node with an
available capture or a piece within three steps of its own corner, which covers
immediate corner threats and forced defensive replies. The gap was
corner-threat *prevention*: a blocker could be walked off the enemy's route, or
a blocking square declined, and still count as quiet. `selective.blocks()` now
tests whether a square lies on a shortest enemy king route to the enemy corner —
king distance is additive exactly along such a route — and a move that vacates
or occupies one is no longer quiet. An occupied route square is a real
obstruction here precisely because `guarded()` has already excluded positions
where it could simply be captured.

The predicate is strictly narrower than the issue #60 one: it can only refuse
pruning. Measured consequence — it, not the index threshold, is what limits
move-count pruning: sweeping `move_count_base` from 2 to 24 changes the number
of skipped children by under 12%.

### What this does not change

Mate-distance pruning is excluded from `selective_mode_early()` and from
`SearchConfig.selective_pruning()`. It narrows the window to bounds the true
value already satisfies, so a search using it still reports ordinary `exact`
bounds and may still publish a proven result. It is verified against the
bounded-proof oracle rather than assumed: see the measurements. The other three
withdraw the certificate exactly as NMP and futility do.

These four are **Python only**. `Genome.native_arguments()` raises rather than
returning a native search that ignores them, and the exhaustive-label pipelines
(`teacher_config`, the supervised trainer, the dataset expander) reject the
three heuristic members through `SearchConfig.selective_pruning()`. The
analysis CLI exposes `--razoring`, `--reverse-futility`, `--move-count` and
`--mate-distance`; the browser option whitelist is unchanged.

See [bounded validation](../benchmarks/shallow_pruning/README.md) for firing
rates, node costs, the mate-distance oracle check and the equal-time paired
result, including which of the four are recommended for adoption.

## Experimental evolved and variable-material evaluators

`selective_evaluator_enabled=True` explicitly permits pruning with the adopted
route weights, signed coefficients, and variable material. It defaults to false
and does not enable either pruning method by itself. The full configuration,
including this flag, enters search/cache identity and result diagnostics.
Board eligibility, non-PV windows, null verification and proof isolation remain
unchanged. Enable PVS to create the non-PV narrow windows used by these guards.

The experimental futility allowance per remaining ply is:

```
piece_scale/2
+ abs(advantage_weight) * (max(predator_zero_bonus, predator_scarcity_bonus) + prey_bonus)
+ 3*abs(enabled attack_weight) + 4*abs(enabled defence_weight)
+ 2*abs(enabled overload_weight) + 8*abs(enabled pressure_weight)
```

Multiply by remaining depth and `futility_margin`. `piece_scale` starts at
`abs(count_weight)`; in variable mode it is the larger of that and the greatest
current value of an occupied piece type on either side, multiplied by
`abs(count_weight)/BASE`. Both sides use the same local scale. Native uses its
fixed predator/prey bonuses and has no overload module. This is a heuristic
allowance, **not an upper bound**: variable values can change nonlinearly after
captures. It has not been calibrated to establish general strength or tactical
safety. The original margin remains unchanged when this option is false.

A selective mate-range score no longer ends iterative deepening early: the
requested depth must finish, since such a score is not a mate certificate.
Unpruned proven results can still finish early. The opt-in
[corner-run certificate bound](CERTIFICATE_SEARCH.md) is one of those: it is a
proof, so it does not make a search selective, while the race reduction it ships
alongside does and joins this list. That document also describes the guard which
exempts a certified branch from both methods above, replacing the board guard's
`max(3, ceil(depth/2))` corner proxy with the certificate itself. Tournament depth validation
continues to exclude incomplete requested selective searches.

The combined native wire form appends four coefficients, a `0/1` variable-mode
flag, the six selective parameters, then a `true/false` experimental-evaluator
flag before state. Existing shorter weight-only and selective-only forms remain
accepted. `RustTeacher.analyze(..., selective_evaluator_enabled=True)` uses the
combined form. All evaluator and selective settings invalidate reused search
state when changed.

## Hypothetical state and caches

A synthetic pass is not an action. Side-to-move flips, goal ownership and board
stay fixed; real total ply, no-capture clock, history and occurrence counts do
not advance. The probe's search ply advances one for internal mate-distance
arithmetic, but no synthetic mate value/certificate is accepted as a cutoff.
Null leaves use ordinary evaluation without bounded proof calls; verification
retains the ordinary unpruned proof settings. All modelling draws are disabled
in the entire hypothetical subtree: neither
a repeated board nor the 80th noncapture manufactures a draw. Official wins
still apply. Hypothetical descendants temporarily append ordinary moves;
Python `finally` blocks restore board, counts, history, occurrences, keys,
clock, side and rule mode on returns, exceptions and budget cancellation.
Rust probes clone the complete position, including incremental material and
pressure; normal and verification move stacks unwind on every Result error.
Neither implementation exports a hypothetical state or null action.

Null probes bypass TT reads and writes entirely, and neither they nor
verification searches update killers/history ordering. Verification keeps its
own `verified-v1` namespace, described under **Reductions** below. Python's
table is scoped by the full configuration identity, with a selective namespace;
Rust recreates its cached Search whenever any protocol setting changes. Even
ordinary ancestors of selectively pruned nodes are thus isolated from
pruning-disabled results, because the namespaces never meet. `exact` internal TT
bounds mean exact only within that configuration's search procedure. Public
Python/native adapter bounds use `selective_exact` etc. Enabled Python results
report proof `unknown`; native uses `selective_result`, never `proven_result`,
for a mate-range search result. Bounded terminal proof APIs remain unchanged.

## Activation preconditions

A technique can be declared, supported and still never execute, because each one
also needs a search shape. `heuristics.config.activation(config)` reports those
preconditions from the configuration alone, before a node is searched, and
`blocked(config)` summarises the ones that apply:

| Technique | Precondition | Why |
| --- | --- | --- |
| NMP | `pvs_enabled` | Eligibility requires a null window on entry, and only a PVS scout search reliably creates one. TT tightening is excluded on purpose. |
| NMP | `max_depth > nmp_min_depth` | The root never prunes, so the deepest eligible node has `max_depth - 1` plies left. |
| futility | `pvs_enabled` | The same null-window eligibility. |
| futility | `max_depth >= 2` | There must be a non-root frontier node. |
| LMR | `ordering_enabled` or `compiled_ordering_enabled` | Reductions need a move order worth trusting. |
| LMR | `max_depth > lmr_min_depth` | As for NMP. |
| MVV-LVA | `variable_material_enabled` | Flat material values every type at BASE, so the victim/attacker keys are constant and the compiled sort returns exactly the old order. |
| NMP, futility, quiescence | compact Python backend | The legacy expanded-state backend rejects them. |

Every one of these was a live blocker in the tuning runs audited by issue #66:
the protocol ran at `depth=2` with PVS off, so NMP, futility and LMR could not
fire even once, and its flat-material candidates would not have been reordered
by MVV-LVA either. Each search result now reports `selective.declared`,
`selective.fired`, `selective.unreachable` and a `disabled_reason` summary, and
`effective` means *every declared technique can execute*, not merely that the
evaluator scales are supported. The paired-match harness carries the same
settings in its frozen protocol and flags a declared-but-silent technique in the
run report; see [the harness README](../tournament/README.md).

A depth-mode protocol that raises `max_depth` to clear those minimums must raise
the optimizer's `min_completed_depth` with it, or a move that completes below the
old minimum silently costs the candidate its eligibility. A selective result is
never exempt from that rule: its `stop_reason` is `selective_result`, never
`proven_result`, and only a proof is exempt.

## Reductions, and what they cost

Late move reductions are a separate opt-in (`lmr_enabled`), and they compose
with the scout searches rather than competing with them. A scout search and a
reduction used to be alternatives, so whenever PVS applied — every non-first
move of every PV node — the reduction was skipped, and LMR only ever ran below
an already-null window. It now reduces whichever narrow search happens first:
the scout is searched at `depth - 1 - R`, re-searched at full depth if it beats
alpha, and only then re-searched at full width if it lands inside the window.
With PVS off the behaviour is unchanged, which is the path the existing
reduction tests cover.

`nmp_reduction` and `lmr_reduction` are fixed plies, and `nmp_reduction` may not
exceed `nmp_min_depth - 2`, so at the default minimum depth a probe removes a
single ply and costs almost what the search it replaces would. Three optional
divisors let a reduction grow instead, all defaulting to zero, which is exactly
the fixed behaviour:

| Setting | Effect |
| --- | --- |
| `nmp_depth_divisor` | Adds `(depth - nmp_min_depth) // divisor` probe plies. |
| `lmr_depth_divisor` | Adds `(depth - lmr_min_depth) // divisor` plies. |
| `lmr_index_divisor` | Adds `(index - lmr_min_index) // divisor` plies for later moves. |

Every total is clamped so at least one ply survives below the reduction: a
reduced search is never a static leaf, and a null probe always retains a probe
ply. The native teacher has a fixed schedule and no LMR, so
`Genome.native_arguments` refuses a genome carrying a nonzero divisor rather
than passing a parameter Rust would ignore.

Verification searches keep their own transposition namespace, `verified-v1`,
instead of running with the table switched off. A verification search is
ordinary alpha-beta on the real position, so repeated verifications may reuse
each other's work, while a pruned ancestor still cannot read any of it: the two
namespaces never meet. Null probes remain entirely un-tabled, because they
search a hypothetical position with the modelling draw rules suspended and
nothing they compute is a value for any real key.

Two costs were removed without changing a single answer. The shared board
guard and the quiet-move test no longer scan in Python: distances are table
lookups, the capture test is an indexed occupancy read, and the part of the
quiet test that depends only on the position is hoisted out of the per-move
loop instead of being recomputed for every candidate. Their work charges are
unchanged, so node and work counts are identical either way, and a reference
implementation of the original loops is asserted equal to them across every
legal move of nine positions. The node's static evaluation is also deferred:
NMP needs it before the move loop, but a futility node computes it only once a
candidate quiet child actually appears, instead of at every guarded node.

### Calibrating the allowance instead of tuning it

`heuristics.calibration` measures the quantity the allowance is supposed to
bound, rather than leaving `futility_margin` to taste. For every position the
guard admits it takes every move `quiet` admits and records the signed change in
static score from the mover's view. That is exactly the right quantity at depth
one, where the skipped child is evaluated statically; from depth two the child
gets a search that may capture, which is why material and advantage are charged
from that ply and sit outside the measurement.

On the adopted genome, over 202 quiet moves from 183 guarded positions: mean
gain **−16.49** and a largest observed gain of **−6.43** against a charged
allowance of **207.4**. Every admitted quiet move loses ground, because a quiet
move cedes the tempo, so on this sample no multiplier is needed to cover a gain
that never occurs and the derived margin is **0**.

That is a stronger result than it was before issue #68 tightened `quiet`. Under
the looser test the same corpus admitted 3,618 moves with a largest gain of
**+12.81**, giving a derived multiplier of 0.0617 — which is why the 1/16 used
elsewhere on this branch is close to right. Adding the route-blocking clause
removed precisely the moves that could gain: a move that vacates or occupies a
square on a shortest enemy route is corner-threat prevention, and that is what a
gaining quiet move was. The two changes were made independently and agree.

A margin of zero is not a recommendation. Two hundred moves from one genome
cannot license removing the allowance, the configuration floor is 1/16 in any
case, and the argument for keeping headroom is the tail this sample does not
contain.

`calibrate(config, states, quantile=…)` returns the multiplier covering a given
fraction of observed gains. A quantile is not a bound: futility is a heuristic
and skipping a move in the tail is the risk it exists to take, so the function
reports coverage and never safety. Recompute it per genome — these numbers
belong to the route weights they were measured on, which is the whole reason the
original allowance, carried over from a genome it was never calibrated for,
pruned nothing.

### The probe guard, and what it is for

`nmp_relaxed_guard_enabled` (default off) drops exactly one clause of the shared
guard, and only for a null probe: the captures available to the side to move.
The clause exists to protect threatened retreats and forced capture and defence
situations, which is a *futility* concern — futility skips one quiet move while
a tactic is pending. A null probe's risk runs the other way. It is unsound where
passing beats every legal move, and a side holding a capture is precisely a side
that is not in zugzwang, so its own captures argue against the failure the guard
exists to prevent. The opponent's captures still refuse the probe, because those
are the threats a pass declines to answer. Futility reaches at most depth two and
NMP starts at three, so the two never share a node and each gets the guard it
needs.

On sampled positions the shared guard refuses 58.3% for an available capture and
10.4% for a runner near a goal, and admits 31.2%. Relaxing the capture clause
doubles NMP's attempts at depth 4 and raises its cutoffs 14% at depth 6, with no
change of move or score anywhere — and moves the node count within noise at every
depth measured. The clause gated how often the technique fires, not whether it
pays. What the measurement points at instead is verification, which is 6.1% of
the whole tree at depth 6 and is paid on every non-mate fail-high before any
cutoff is allowed.

### What a quiescence chain costs

A chain resolves captures the search would never have evaluated, and each one
runs the same bounded proof an ordinary leaf runs. `quiescence_proof_nodes` and
`module_seconds['quiescence_proof']` charge that separately from the horizon leaf
the search would have reached anyway, so the question is answerable rather than
assumed. Measured at depth 4 with proofs at their default, it is one proof node
per resolved capture and under 1% of wall time: the proof gate rejects
immediately at these positions. A chain's cost is its extra search nodes, which
were +58% in the same measurement, and not its proofs.

## Configuration, diagnostics and labels

`SearchConfig.to_dict()/identity()` includes `search_version=intransitive-selective-v1`
and every parameter. Existing JSON configs load with both flags off. Example:

```python
config = SearchConfig(nmp_enabled=True, futility_enabled=True)
# Native: RustTeacher.analyze(state, nmp_enabled=True, futility_enabled=True)
```

The analysis CLI accepts `--nmp/--no-nmp` and `--futility/--no-futility`, plus
`--pvs/--no-pvs` and `--selective-evaluator/--no-selective-evaluator` for the
two preconditions without which those flags cannot do anything; `--config`
supplies numeric parameters. Browser settings expose the same four switches, so
a game using evolved weights can turn pruning on rather than only being refused.
Last AI analysis and exported PGN diagnostics retain effective settings,
selective depth, counters and identity. Existing JSON configuration plumbing
also carries this identity into checkpoint/tournament manifests. Changing a
checkpoint opponent requires a separate comparable rebaseline, not reuse of
old win rates.

Counters report attempts/cutoffs/skips, verification searches/failures and
nodes, null nodes, eligible/pruned quiet moves and additional static evaluations.
Python records static-evaluation seconds plus ordinary module costs, work,
nodes, TT payload bytes and wall time. Rust reports nodes (its existing budget
unit, including charged extra evaluations), TT hits/payload byte estimate and
wall time. Rust memory excludes allocator/hash-control overhead; Python's
estimate likewise is not process RSS. Work units differ across backends and
must not be compared numerically. All recursive probes and verification use
the original move deadline/node budget. Python preserves its legal fallback
and partial/completed iteration policy; native retains its existing no-label
result when no iteration completes. A caller needing a move then uses the
ordinary legal fallback (as the benchmark does).

The native wire protocol still accepts the original eight search parameters
plus state. The extended form appends `nmp_enabled nmp_min_depth nmp_reduction
futility_enabled futility_max_depth futility_margin` before state (booleans
`true/false`). Old binaries reject extended requests instead of ignoring them.
Native retains its established PVS/ordering policy, not Python's independent
optimization switches. No cross-backend node-count equivalence is promised.

The existing exhaustive-label pipeline rejects enabled selective settings and
selective results. This change does **not** enable a selective-label experiment.
Such a future dataset needs a separate label contract and complete search
identity; it must not call depth-N selective choices exhaustive optimal labels.
Rollback: set both flags false/restart the opponent, or load any old preset.
Config changes invalidate cached search results. No rules, action encoding,
PGN move notation or training-record action format changes.

Quiescence's own selective filters — the cyclic exchange evaluation and delta
pruning — are specified separately in [EXCHANGE.md](EXCHANGE.md). They share this
document's rules: opt-in, off by default, part of the search identity, and never
a certificate.

## Evidence and references

See [bounded validation](../benchmarks/selective/README.md), including known
baseline tactical failures, divergences, uncertainty and adoption decision, and
the [issue #66 activation measurement](../benchmarks/selective_activation/README.md),
which records the first configuration in which these techniques actually run and
attributes a node count to each of them separately.
The conceptual references are [Verified Null-Move Pruning, v1](https://arxiv.org/abs/0808.1125v1)
and [Stockfish search.cpp at 031dfeb4](https://github.com/official-stockfish/Stockfish/blob/031dfeb437fa6b06cdbdf4ef89dfb82f6b83c4d3/src/search.cpp).
The paper's chess results do not establish performance or zugzwang safety in
Intransitive. No chess constants, material tests or check predicates were copied.
