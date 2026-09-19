# Reading the heuristic: per-square attribution and corpus audit

Two instruments answer different questions about the same evaluator.

| Question | Instrument |
| --- | --- |
| What is this position worth, and *where* on the board does that come from? | `attribution.py`, shown in the browser by `play.html` |
| Which replies does the search actually rate, and how confident is each number? | the same panel's reply list, fed by `SearchResult.root_moves` |
| Is a module too big, too small, dead, or pinned at a cap — and does it ever change a decision? | `audit.py` |

## Per-square attribution

`attribute(game, state, side, budget, config)` returns, for each module, an
81-cell array of signed contributions **already multiplied by the module's
coefficient**, whose sum is that module's term in the total score. Positive
cells favour the perspective player.

```python
from intransitive.heuristics.attribution import attribute
from intransitive.heuristics.budget import Budget
from intransitive.heuristics.config import SearchConfig
from intransitive.IntransitiveGame import IntransitiveGame

game, config = IntransitiveGame(), SearchConfig()
report = attribute(game, game.getInitBoard(), 0, Budget(config.node_limit, 30.), config)
report['modules']['attacking_position']['squares']   # 81 signed contributions
report['modules']['attacking_position']['term']      # the evaluator's own term
report['modules']['attacking_position']['residual']  # term - sum(squares); must be ~0
```

### Where each module's credit lands

| Module | Credited to | Notes |
| --- | --- | --- |
| `piece_count` | every piece's own square | flat: ±`count_weight`. Variable material: the piece's type value ÷ `BASE`. |
| `clear_run` | nothing | a proof or zero, never a weighted estimate; a proven position has no decomposition at all. |
| `piece_advantage` | every piece's own square | all pieces of one type carry the same number — this module has no spatial content. |
| `attacking_position` | own pieces (progress) and **enemy** squares (capture opportunities) | two independently capped halves, each rescaled proportionally. |
| `defensive_position` | the **defender** squares doing the covering | each threat's inner sum is capped at 1 and split by each defender's `1/(1+ply)` share, then the side total is capped at 4. |
| `overload` | the overloaded defender squares, split evenly | subtracted, so its cells are negative at a positive coefficient. |
| `local_pressure` | the attacking piece's square | matches `pressure_totals` exactly (`pressure_squares` is the per-square kernel). |

Because the opponent's contributions are subtracted, a square can carry a large
negative number while holding one of *your* pieces — that is the enemy's credit
for attacking it, which is usually what you want to see.

### Residuals are the bug detector

Attribution recomputes what `evaluation` sums; it does not borrow the running
total. The two can therefore disagree, and the report says so rather than
hiding it: every module carries `residual = term - sum(squares)`, and the
browser paints a red `residual` badge on any module that drifts.
`tests/test_attribution.py` asserts the residual is zero across a corpus of
random reachable positions for four configurations (core, all modules on,
variable material, and signed coefficients), for both perspectives. A new or
edited module that forgets to keep the two paths in step fails there.

Inactive modules are **previews**, not residual failures: they are computed at
their configured coefficient so you can see what a module would say before
enabling it, they report `term: null`, and they are excluded from the board
total.

### Caps are reported, never hidden

`attacking_position` caps broad progress at 2 and captures at 1;
`defensive_position` caps each covered threat at 1 and the side total at 4;
`overload` caps distinct defenders at 2. Each module reports `raw` (the
uncapped feature), `used`, and whether the cap was `binding`. A module sitting
on its ceiling still produces a number but has stopped responding to the board,
which a single score cannot show you.

## Browser panel

`uv run --locked python -m intransitive.play` serves the heatmap under the board.

- **Module checkboxes** add modules to the board tint. The cell numbers are
  score points; the sum of the selected modules is printed under the legend.
  Untick Material and Piece advantage to make the positional modules legible —
  at the default coefficients they are two orders of magnitude smaller.
- **Colour** is gold toward the perspective player, violet toward the opponent,
  with a neutral midpoint. The pair is validated for colour-vision separation
  against each other *and* against the Blue/Red piece colours, so a tint never
  reads as an owner. The scale is per-report: the legend prints its maximum.
- **Search replies** lists every legal reply with its score, and draws the
  selected line on the board with numbered steps. Alpha–beta proves an exact
  score only for the reply it selects; the others come back as upper bounds
  (`≤`). **Exact score per reply** searches every reply on a full window so all
  of them are exact, at several times the cost. Rows from an interrupted deeper
  pass keep their own depth and the note warns that mixed depths are not
  comparable.
- Analysis runs on its own budget, never the opponent's, and is capped at depth
  8, 30 seconds and 50,000,000 work. The report is discarded from the board as
  soon as the position moves on.

`POST /api/analyze` is read-only and takes `{revision, modules, perspective,
search, exact_root, depth, time, work}`. `overload` replays every legal reply
and builds a child geometry for each, so it is computed only when selected.

## Corpus audit

```
uv run --locked python -m intransitive.heuristics.audit \
    --games 30 --plies 200 --capture-bias 0.55 \
    --pressure --search-depth 3 --output audit.json
```

Positions come from legal play biased toward captures. Unbiased random play
almost never trades, so a plain random walk never leaves the opening bucket;
`--capture-bias` is a corpus-shaping knob only, and the positions it produces
are ordinary legal positions. They are bucketed by material remaining
(opening ≥ 16 pieces, middlegame 8–15, endgame < 8) rather than by ply, and
each module reports:

| Column | Meaning |
| --- | --- |
| `|term| med / p90 / max` | how large the module's contribution actually is in that phase |
| `share` | its fraction of the phase's total absolute term mass |
| `zero` | positions where it contributed exactly nothing |
| `at cap` | positions where a side's feature was pinned at the module's ceiling |
| `flips` | positions where **removing** it changes which reply the static evaluation prefers |

`flips` is the one that says a module matters. A module with a large term that
never flips a choice is a constant offset; a module with a small term that
flips often is doing real work at a coefficient that may be under-scaled.

Two modules are structurally prone to a low flip rate, and the audit is the
place that shows it: `piece_count` and `piece_advantage` are both functions of
the six type counts alone. Those counts are identical for every *quiet* child
of a position, so both modules contribute the same constant to every quiet
reply and cancel out of the comparison entirely. They can only separate moves
that capture. A large `share` next to a near-zero `flips` is that structure,
not a bug — but it does mean the coefficient is buying very little move
selection.

### Per-module cost

```
uv run --locked python -m intransitive.heuristics.audit --timing \
    --games 14 --plies 200 --capture-bias 0.55 --pressure
```

Two measurements, because they answer different questions and disagree:

* **Instrumented timers** come from the evaluator's own `module_seconds`. They
  show where wall time goes in one configuration, but they are *not* marginal
  costs. `Geometry` memoizes interception and safety answers, so whichever
  module asks first pays for the shared work and later modules look cheap.
  `overload` is the clearest case: it bills a fraction of its standalone cost
  when it runs after `defensive_position` has filled the interception cache.
* **Ablation deltas** re-time the whole evaluation with each module enabled
  against a material-only baseline. This is the honest cost of turning a module
  on, because enabling *any* of attack/defence/overload switches the evaluator
  off its cheap material-only path onto the full piece index with king-route
  maps — a shared setup no single module's timer contains.

Both positional modules and the route maps are compiled; see the shared route
maps section of [the heuristics README](README.md) for what each one does
differently from its reference, and `--timing` here for what it costs.

The `proof` row is the bounded clear-run proof, not a weighted module. It is
part of a leaf's real cost, and it dominates the endgame; set `proof_depth` to
zero in a `--config` preset to measure the weighted modules alone.

### Comparing against depth

`--search-depth N` additionally searches each position to depth N and reports
how often the static evaluation already prefers the reply the search chooses,
plus where the search's choice sits in the static ordering. The per-module
`agree+` column is the leave-one-out version: how much agreement with the
deeper search that module adds. A module with a negative `agree+` actively
points away from what depth concludes.

Read `agree+` together with the headline agreement number. When the search
already agrees with the static pick in ~95% of positions, `agree+` is close to
`flips` times that agreement and adds little; it earns its keep at depths where
the two diverge. Note also that a capture-biased random walk still rarely
reaches fewer than eight pieces — games end first — so the endgame bucket needs
either a much higher `--capture-bias` or a corpus seeded from recorded endgames
rather than from the opening.

This is the direct measurement of "what does the evaluation miss that depth has
to fix". Where agreement is low, the engine's strength at that phase is coming
from search, and a position where the static rank of the search's choice is
poor is a candidate for a new feature — or for accepting that depth is the
right tool. A worked example: in the `red_31_keep_goal_defender` regression the
static evaluation ranks the losing move *below* the correct defence, depth 3
chooses it anyway, and depth 4 scores it as a proven loss. That failure is a
four-ply horizon, not a mis-scaled module, and no coefficient change fixes it.

Influence is measured one ply deep deliberately. It isolates what the
evaluation contributes *before* search compensates for it, so the gap between a
module's one-ply influence and its effect on played strength is a measure of
how much work depth is doing in its place.

Terminal and proven positions are excluded from both the term statistics and
the influence pass: their value replaces the weighted sum, so there is nothing
to decompose or attribute a decision to.
