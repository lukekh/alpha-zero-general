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
source or destination. The shared guard also protects immediate quiet wins,
corner defence and low-mobility forced replies. No reverse futility, razoring,
late-move reductions or weight tuning is included.

Selection is effective only with count=100, advantage=25 and the standard
predator/prey bonuses (1, .5, .25), no attack/defence/overload modules, and
pressure disabled or weight ≤20 (radius 3 or 4). Unsupported evolved scales
keep the requested flags in provenance but disable **both** pruning methods;
`effective=false` explains this in diagnostics. Ordinary evaluation continues
unchanged. Python's legacy expanded-state/custom-game backend explicitly
rejects selective options. Native's unsupported arguments raise errors.

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
