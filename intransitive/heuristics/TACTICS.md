# Tactical move regressions

The suite now also includes [12 harder regressions from the analysed game](GAME_REGRESSIONS.md).
Run all 112 with:

```sh
uv run --locked python -m unittest intransitive.tests.test_tactics intransitive.tests.test_game_blunders -v
```

Run all **100 individually named tests** from the repository root:

```sh
uv run --locked python -m unittest intransitive.tests.test_tactics -v
```

Run a category or one position by matching its name:

```sh
uv run --locked python -m unittest intransitive.tests.test_tactics -k goal_defence -v
uv run --locked python -m unittest intransitive.tests.test_tactics -k 079_ -v
```

These test the **alpha-beta heuristic bot** used by the browser. They do not load
a neural checkpoint. Each test asserts one exact selected action, with no list
of acceptable alternatives and no expected-failure exemptions.

## Positions and guarantees

[tactics.json](tactics.json) contains 50 hand-authored layouts, each represented
as both a Blue-to-move and a reflected, colour-exchanged Red-to-move position:
100 different boards and 100 move assertions. Both goals, all three piece types,
edge and central tactics, poisoned captures, double runners, and capture bait
are covered. These are synthetic puzzles with fresh draw histories, not claims
that the positions arose in a recorded game.

| Category | Tests | Independently checked guarantee |
| --- | ---: | --- |
| `immediate_goal` | 20 | Exactly one move wins immediately, by entering or capturing on the goal. |
| `clear_run` | 20 | No immediate win; exactly one move forces a win within three plies. After **every** enemy reply, the runner can finish on the corner. |
| `goal_defence` | 20 | Exactly one move prevents loss on the next reply. Every alternative allows an immediate enemy goal win. Includes capturing, blocking, and blocking two runners together. |
| `save_piece` | 20 | Exactly one quiet escape prevents immediate elimination. Every alternative lets the opponent capture the last friendly piece, including tempting poisoned captures. |
| `safe_capture` | 20 | Exactly one move guarantees a net gain of one piece after the opponent's best reply. Other moves guarantee less; losing the game ranks below material. No forced win within three plies competes with the capture. |

The win cases prove a unique fastest finish within their horizon. The defence
cases prove a unique way to avoid immediate loss, and the capture cases prove a
unique immediate material advantage. **The latter two categories do not prove
the eventual whole-game result.** Unknown continuations are not treated as
proven draws. This distinction lets the suite use obvious tactical obligations
without inventing game-theoretic claims about unsolved positions.

Each JSON entry records its ID, category, layout, side to move, square-to-piece
mapping, expected move, and rationale. Pieces use the game encoding:
`1 = rock`, `2 = scissors`, `3 = paper`; positive is Blue and negative is Red.
Player `0` is Blue, defending A1 and targeting I9; player `1` is Red, defending
I9 and targeting A1. Unlisted squares are empty. The layout name identifies the
original Blue pattern; its paired Red case has transformed coordinates.

## How the assertions work

Before invoking the bot, every test runs
[tactical_oracle.py](../tests/tactical_oracle.py) over **all legal alternatives**.
It uses the existing independent, immutable Python
[reference rules](../tests/reference_rules.py), not the production evaluator,
move ordering, search scores, or proof helper. Terminal-only AND/OR search
certifies wins; exhaustive replies certify survival and material. Ambiguous
fixtures fail even if the bot happens to select the written answer.

Then a fresh `AlphaBetaPlayer` analyzes the full history-bearing state. The test
asserts its action equals the expected action, it completed a search without
exhausting the budget, and it did not mutate the input. A failure prints the
board mapping, expected and actual moves, score, completed depth, work, and PV.

All positions use the same configuration: maximum depth **3**, default core
weights, optional attack/defence/overload modules disabled, and the existing
bounded proof settings of depth **2** and **64** nodes. Only the test work/time
ceilings are raised to 2,000,000 work units and 30 seconds to avoid timeout
dependence on machine speed. A proven mate may stop iterative deepening early.
No per-position tuning or extra search depth is used to obtain a passing result.

Initial result: **100/100 passed**, approximately **8 seconds** for the suite on
the development machine. This is a basic tactical regression baseline; passing
it does not establish that the bot's quiet positional evaluations are good.

## Reviewing or extending fixtures

Edit the explicit JSON position and expected move, then run the suite. The
independent certificate must still establish uniqueness. Do not update an
expected move from the bot's output, accept tied alternatives, or suppress a
failure to accommodate an evaluator change. The suite also checks the fixture
count, unique boards and IDs, category counts, and colour balance.
