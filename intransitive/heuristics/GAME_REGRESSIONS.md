# Regressions from the 79-ply game

Run the 12 harder tests:

```sh
uv run --locked python -m unittest intransitive.tests.test_game_blunders -v
```

Run these together with the original 100 tactical puzzles:

```sh
uv run --locked python -m unittest intransitive.tests.test_tactics intransitive.tests.test_game_blunders -v
```

The [fixture file](game_blunders.json) includes the complete original PGN and
hashes identifying each tested state. The tests replay all 79 plies through both
the production loader and independent Python rules, comparing full states
including repetition and capture history. Nothing is loaded from an ignored
checkpoint directory or a live browser game.

## Six original positions

| Position | Assertion on the bot's selected move | Checked continuation after that move |
| --- | --- | --- |
| Red 14, before E6–F6 | Do not allow Blue to force a net material improvement. F7–F6 is a verified defensive example. | All replies, four plies; captures and recaptures on D7 explain the played move's failure. |
| Red 18, before G7–F6 | Do not allow G5–H6 to force a material gain through the paper fork. G6–H6 prevents this particular tactic. | Fix Blue's initial G5–H6, then examine all replies through five plies. |
| Red 31, before F8–E8 | Do not allow a forced win within five plies. Both F8–G7 and F8–G8 pass this bounded check. | All legal attacking moves and defensive replies, five plies. |
| Blue 19, before G5–F4 | Force a net gain of at least one piece. G5–H6 passes; the played retreat does not. | All replies, four plies after the selected move. |
| Blue 32, before H6×H7 | Choose a move that forces a win within five plies overall. H6×H7 passes; the unrelated D7–E6 does not. | All replies, four plies after the selected move. |
| Blue 34, before C5–D4 | Win on the corner immediately. H8–I9 passes. | Actual terminal result of the selected move. |

The original positions can have multiple valid moves. Their assertions check
the tactical consequence of **the move the bot actually selects**, rather than
requiring one arbitrarily chosen defence. Each also checks known good and bad
controls independently before running the bot.

The Red 18 test is deliberately specific to the G5–H6 fork. Preventing that fork
does not prove that every other material threat has been neutralised. Likewise,
preventing a five-ply forced win is not a proof of an eventual draw or victory.
Material checks count pieces equally and prioritise a terminal win over material;
they do not use the heuristic's piece-advantage formula.

## Three simplified layouts, tested for both colours

These six tests retain exact-move assertions. The independent proof enumerates
every legal root move and requires the qualifying set to contain exactly the
written answer before comparing it with the bot's choice.

| Derived from | Original-colour answer | Unique tactical obligation |
| --- | --- | --- |
| Blue 19's paper fork | G5–H6 | Fork the two papers and force a material gain within three plies. |
| Red 31's last defender | F8–G8 | Prevent a forced five-ply loss. Friendly papers on G7/G9 remove the alternative rock defences. |
| Blue 32's winning run | F6×G7 | Begin the only five-ply forced win, followed by H8 and I9. The runner and blocking paper are moved to make the shortest route unique. |

Each fixture records precisely how it differs from the original game. The
colour-exchanged version also transforms the coordinates and expected action.

## Search and proof settings

The bot uses the same settings as the basic suite: maximum depth **3**, the
default core heuristic, default proof depth **2** / **64** nodes, and test ceilings
of 2,000,000 work units and 30 seconds. Tests fail if the search exhausts its
budget. Input states must remain unchanged.

The **independent test oracle** looks farther where necessary to certify a
tactic. Its bounded AND/OR traversal uses terminal outcomes or a material
threshold, with ordering and memoization for speed. It has no heuristic score,
time cutoff, or node cutoff, and never calls the production search or evaluator.
This extra verification depth does not increase the bot's search depth.

These are ordinary assertions, including the currently failing defensive
positions. They are not skipped or marked as expected failures. A red result is
a target for improving the heuristic at the existing search depth.

Initial result: **7 pass, 5 fail**, about 14 seconds on the development machine.
All independent fixture checks pass. The failures are Red's original moves 14,
18 and 31, plus the simplified last-defender position in both colours. The bot
repeats E6–F6, G7–F6 and F8–E8 in the original positions, and moves a paper instead
of the sole effective rock defender in the simplified versions. No heuristic or
production search changes were made to obtain these results.
