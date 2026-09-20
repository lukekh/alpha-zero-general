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

`futility_max_depth=2` (1..2), `futility_margin=1` (finite multiplier 1..16):
after searching at least one move, skip an eligible quiet child when
`static + depth * multiplier * (count_weight/2 + advantage_weight + 8*pressure_weight)
<= alpha`. Disabled pressure contributes zero. Thus the default material-only
margins are 75/150 evaluator units, versus 100 per piece; pressure 10 yields
155/310. These are experimental margins, not upper bounds on future gains.
Keep every fastest runner (including retreating moves), all captures, moves
within three steps of the goal, and moves with an enemy within two steps of
source or destination. Issue #68 additionally refuses any move that vacates or
occupies a square on a shortest enemy king route to the enemy's corner; see
*The reviewed quiet predicate* below. The shared guard also protects immediate
quiet wins, corner defence and low-mobility forced replies.

Without the experimental evaluator option below, selection is effective only with flat material, count=100, advantage=25 and the standard
predator/prey bonuses (1, .5, .25), no attack/defence/overload modules, and
pressure disabled or weight in 0..20 (radius 3 or 4). Unsupported evolved scales
keep the requested flags in provenance but disable **both** pruning methods;
`effective=false` explains this in diagnostics. Ordinary evaluation continues
unchanged. Python's legacy expanded-state/custom-game backend explicitly
rejects selective options. Native's unsupported arguments raise errors.

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
exclusion listed above rather than restating it. A node eligible only for
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
Unpruned proven results can still finish early. Tournament depth validation
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

Null and verification searches bypass TT reads/writes, and do not update
killers/history ordering. Python's table is scoped by the full configuration
identity, with a selective namespace; Rust recreates its cached Search whenever
any protocol setting changes. Even ordinary ancestors of selectively pruned
nodes are thus isolated from pruning-disabled results. `exact` internal TT
bounds mean exact only within that configuration's search procedure. Public
Python/native adapter bounds use `selective_exact` etc. Enabled Python results
report proof `unknown`; native uses `selective_result`, never `proven_result`,
for a mate-range search result. Bounded terminal proof APIs remain unchanged.

## Configuration, diagnostics and labels

`SearchConfig.to_dict()/identity()` includes `search_version=intransitive-selective-v1`
and every parameter. Existing JSON configs load with both flags off. Example:

```python
config = SearchConfig(nmp_enabled=True, futility_enabled=True)
# Native: RustTeacher.analyze(state, nmp_enabled=True, futility_enabled=True)
```

The analysis CLI accepts `--nmp/--no-nmp` and `--futility/--no-futility`;
`--config` supplies numeric parameters. Browser settings expose both flags;
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

## Evidence and references

See [bounded validation](../benchmarks/selective/README.md), including known
baseline tactical failures, divergences, uncertainty and adoption decision.
The conceptual references are [Verified Null-Move Pruning, v1](https://arxiv.org/abs/0808.1125v1)
and [Stockfish search.cpp at 031dfeb4](https://github.com/official-stockfish/Stockfish/blob/031dfeb437fa6b06cdbdf4ef89dfb82f6b83c4d3/src/search.cpp).
The paper's chess results do not establish performance or zugzwang safety in
Intransitive. No chess constants, material tests or check predicates were copied.
