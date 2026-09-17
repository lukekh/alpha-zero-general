# Variable piece values

Select `SearchConfig(variable_material_enabled=True)` to replace flat material
with composition-dependent piece values in Python Minimax or heuristic MCTS.
The Rust teacher exposes the same `variable_material_enabled=True` option on
`inspect` and `analyze`. The default is `False`, allowing direct comparison with
the previously tuned evaluator.

With **BASE = 100**, **REG = 0.25**, and **N equal to this player's remaining
pieces**, each rock is worth:

```text
100 * (opponent_scissors + 0.25) / (opponent_paper + 0.25)
    * sqrt((N / 3 + 0.25) / (own_rocks + 0.25))
```

The formula rotates for the other types:

| Piece valued | Opponent prey (numerator) | Opponent predator (denominator) | Own count in balance term |
| --- | --- | --- | --- |
| Rock | Scissors | Paper | Rocks |
| Scissors | Paper | Rock | Scissors |
| Paper | Rock | Scissors | Papers |

Compute each side's army value as the sum of its **piece count × per-piece
value**, then subtract opponent value from own value. Each side uses its own
`N`. An absent type contributes zero, even though its hypothetical per-piece
value is finite. REG keeps every denominator positive when a type goes extinct.

The balance multiplier is one when the player's army has equal type counts.
With the opponent matchup held fixed, a rarer own type receives a larger
per-piece value. For example, with own counts `(rock=1, scissors=4, paper=4)`
and balanced opposing counts, rock is worth about **161.25**, while scissors
and paper are each worth about **87.45**.

The existing `count_weight` scales the material difference by
`count_weight / BASE`; at its default 100 the formula is used at full strength.
This replaces the flat-count term; it is not added on top of flat material.
Other enabled heuristic modules continue to contribute their own terms.
`Evaluator.explain` includes `piece_values.own` and `piece_values.opponent` for
rock, scissors and paper. Its `piece_count` feature is now army value divided
by BASE, so multiplying by `count_weight` reconstructs the material term.

Terminal and proved wins retain their decisive scores. Ordinary evaluation
keeps the existing ±10,000 clamp; strong composition imbalances can saturate it.
This mode has formula, search and parity tests, but has not yet had a measured
strength benchmark or newly tuned weights.

## Use and compare

```python
from intransitive.heuristics import AlphaBetaPlayer, SearchConfig

config = SearchConfig(variable_material_enabled=True)
player = AlphaBetaPlayer(config=config)
result = player.analyze(state)
```

The [minimal config preset](configs/variable-material.json) also works with
`SearchConfig.from_file` and the existing analysis CLI's `--config` option.

To compare the two modes with otherwise identical coefficients in the paired
match harness:

```python
from intransitive.tournament.spec import candidate

flat = candidate('flat')
variable = candidate('variable', variable_material_enabled=True)
```

The mode is fixed per candidate, included in its evaluation/cache identity,
and checked when loading manifests. Consensus adjudication uses each candidate's
actual mode, and keeps their configuration identities separate. Rust includes
the mode in its search configuration, resetting a reused tree/table when it
changes. The native wire protocol accepts an optional `0`/`1` mode flag after
the four material/advantage/attack/defence coefficients and before the state;
older request formats retain flat material.

The existing v2 signed **genome** evolves coefficients with flat material fixed;
it does not encode or evolve this mode. Its validation rejects a variable-mode
base instead of silently dropping the setting. Use the per-candidate comparison
API above for this new mode. Earlier benchmark archives require their recorded
implementation because the evaluator source fingerprint has changed.
