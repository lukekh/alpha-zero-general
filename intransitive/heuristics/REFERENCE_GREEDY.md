# Reference greedy

`ReferenceGreedyPlayer` ports the move evaluation in
[rps2.py](https://github.com/charbelkassab/flybrain-intransitive/blob/f22c07ad64a78b9e1423c376dbc289380ef6f6fc/rps2.py).
The existing two-ply `GreedyPlayer` retains its separate name and behaviour.
Choose `reference-greedy` in `pit.py` or `--opponent reference-greedy` in the
browser CLI. The browser menu labels it **Reference greedy**.

## Scoring

All features are binary and evaluated for each legal candidate move:

| Feature | Weight | Definition |
| --- | ---: | --- |
| goal | +1000 | Destination is the enemy base. |
| capture | +10 | Destination was occupied. |
| progress | +2 | Chebyshev distance to the enemy base decreases. |
| danger | -8 | An adjacent enemy can capture the moved piece after the move. |
| base_threat | -500 | After the move, an enemy has a legal move onto our base. |

Goal moves suppress danger and base threat. Threat detection accounts for
captures and base occupants. The player maximizes the weighted sum and chooses
uniformly among scores within `1e-9` of the maximum, using a local NumPy RNG.
Pass `seed` to the Python constructor for repeatable choices. Action enumeration
differs from the upstream dictionary order, so identical seeds need not select
identical moves across the two implementations; their tied move sets agree.

This is a move heuristic with no minimax continuation or additional stalemate
bonus. A risky capture can beat a quiet escape: `10 - 8 = 2`. Scoring intentionally
ignores hypothetical draw cutoffs, as the reference does. Actual move legality
and game termination remain governed by the supplied `IntransitiveGame`.
`intransitive.reference_greedy.features(state, action)` and `reference_value`
expose the score components; `features` expects a legal action.

## Rules comparison

Reviewed against upstream commit `f22c07ad64a78b9e1423c376dbc289380ef6f6fc`:

| Rule | Comparison |
| --- | --- |
| Board and first mover | Same 9×9 board, Blue first, A1/I9 bases. |
| Setup | Same 10 pieces per side, including Red's anti-diagonal reflection. |
| Movement | Same eight king directions; destination-only blocking; no passing. |
| Captures | Same R > S > P > R cycle; optional; attacker keeps its type. |
| Bases | Same occupancy/capture rules; reaching enemy base immediately wins. |
| No legal move | Same loss, including elimination of the last piece. |
| Draws | Upstream assumes 200 quiet plies and has no repetition cutoff. Our modelling mode now uses 80 quiet plies or threefold board-plus-turn repetition. These are modelling assumptions, separate from the movement and win rules. |
| Win precedence | Both check corner and no-legal-move wins before draw limits. |

The upstream comment explicitly calls its 200-ply limit an assumption because
server-side draw handling was not in the extracted client. This comparison
does not verify whether the live server has additional draw rules.

The following measurements predate the increase from 30 to 80 quiet plies.

A direct differential run over 12 seeded trajectories (six random and six
reference-greedy, capped at 250 plies) compared 1,764 positions and 82,553 legal
candidate feature dictionaries/scores. All matched, including canonical Red
positions. Opening boards, legal action sets and resulting boards matched;
six trajectories ended in matching official wins. Draw handling was compared
as the explicit difference above, not treated as equivalent.

## Tactical regression results

Measured with a fresh `ReferenceGreedyPlayer(game, seed=0)` for each fixture,
using the existing independent tactical certificates and game-blunder objectives.
No fixtures, expected moves, weights or alpha-beta tests were changed.

| Category | Passed | Total |
| --- | ---: | ---: |
| Immediate goal | 20 | 20 |
| Clear run | 16 | 20 |
| Goal defence | 20 | 20 |
| Save piece | 16 | 20 |
| Safe capture | 20 | 20 |
| **Basic tactics** | **92** | **100** |
| Original game positions | 2 | 6 |
| Simplified game positions, both colours | 2 | 6 |
| **Hard game regressions** | **4** | **12** |

All 100 basic results are independent of tie randomness. For the harder cases,
four always pass, five have a mixture of correct/incorrect tied best moves, and
three always fail. Summing the fraction of correct moves among each position's
tied maxima gives an expected **5.1833/12** passes (43.2%); seed 0 scores 4/12.
This is exact tie-set enumeration, not a Monte Carlo estimate. The original
fork prevention case always fails; the earlier material-preservation, defender
retention and attacking-fork cases pass with probabilities 1/4, 1/6 and 1/10.

The heuristic handles immediate threats well but misses some forced runs,
poisoned captures and multi-move forks/defences. These scores measure the supplied
positions, not head-to-head win rates. Local per-position results are saved in
`checkpoints/reference-greedy-regressions.json`.

Implementation checks:

```sh
uv run python -m unittest intransitive.tests.test_reference_greedy intransitive.tests.test_players intransitive.tests.test_rules intransitive.tests.test_draws intransitive.tests.test_play -q
```
