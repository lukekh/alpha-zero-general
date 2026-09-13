# Intransitive

Confirmed rules for implementing the game and training a model.

See the [state contract](#state-contract-version-1) below for storage, action
encoding, and tests, and the [implementation plan](IMPLEMENTATION_PLAN.md) for
the delivery sequence.

## Board and players

- Play on a 9×9 grid with columns A–I and rows 1–9, labelled A1 through I9.
- The two players are Blue and Red. Blue always moves first.
- Players alternate turns, moving one piece per turn.
- Passing is not allowed: a player must make a legal move if one is available.
- Only one piece may occupy a square.
- Each player starts with 10 pieces: 3 scissors, 3 rocks, and 4 paper.

## Starting positions

Blue defends corner A1 and starts with the following pieces:

| Piece | Squares |
| --- | --- |
| Paper | B5, C4, D3, E2 |
| Rock | B4, C3, D2 |
| Scissors | C5, D4, E3 |

Red's setup is Blue's setup reflected across the diagonal from A9 to I1.
Red defends corner I9 and starts with the following pieces:

| Piece | Squares |
| --- | --- |
| Paper | E8, F7, G6, H5 |
| Rock | F8, G7, H6 |
| Scissors | E7, F6, G5 |

## Movement and captures

- Every piece moves like a chess king: one square horizontally, vertically, or
  diagonally, remaining on the board.
- A piece may move onto an empty square.
- Diagonal movement depends only on the destination: a piece on D4 may move to
  E5 even if E4 and D5 are occupied.
- A friendly piece blocks movement onto its square.
- An opposing piece can be captured only according to standard rock–paper–scissors
  rules:

  | Moving piece | Can capture |
  | --- | --- |
  | Rock | Scissors |
  | Paper | Rock |
  | Scissors | Paper |

- Captures are optional: a player may choose any legal move even when a capture
  is available.
- A capture permanently removes the opposing piece and places the moving piece
  on its square. The attacker retains its type; pieces never transform, respawn,
  or get promoted.
- An opposing piece that cannot be captured blocks movement onto its square,
  including an opposing piece of the same type.
- Corners follow the same movement and capture rules as other squares. A player
  may occupy and leave their own corner. Entering an occupied opponent's corner
  requires a legal capture of its defender.

## Winning

- A player wins as soon as any of their pieces reaches the opponent's defended
  corner: Blue targets I9, and Red targets A1. Any piece type can win this way.
- If the player whose turn it is has no legal move (stalemate), the other player
  wins. This includes having no pieces remaining. Losing all pieces of a single
  type has no special effect.
- There are no other termination or draw conditions in the official game.

## Appendix: modelling-only termination rules

**These are not official Intransitive rules.** The official game has no
repetition or move-limit rules. The following additions are solely for modelling
and self-play training, to trim the model's state space and bound play that
cycles or continues without captures.

“3 move repetition” means threefold repetition of a position, not repeating an
individual move three times. A move means one player's turn (not a pair of
turns). Both limits automatically end the game as a draw:

- **Threefold repetition:** end the game as a draw when the same position occurs
  for the third time. A position includes every piece's type, colour, and square,
  together with the player whose turn it is. Occurrences need not be consecutive.
  The initial position counts as the first occurrence of that position. Pieces
  of the same type and colour are interchangeable; their individual identities
  do not affect position equality.
- **30 moves without a capture:** end the game as a draw after 30 consecutive
  moves by either player without a capture. The counter starts at zero and resets
  to zero whenever a capture takes place. This is 30 total player turns (15 per
  player), not 30 turns each.

An official win (reaching the opponent's corner or leaving the opponent with no
legal move) takes precedence over either modelling-only draw condition.
There are no additional modelling draw conditions: play continues even if a
position appears hopeless or neither player seems able to force a win, until an
official win or one of these two limits is reached.

---

## State contract (version 1)

This foundation implements issue #1: coordinates, action slots, official setup,
serialized storage, and snapshot ownership. Movement/official wins (#2), draw
detection (#3), and the Game adapter (#5) are separate work. Reversible
symmetries (#4) are implemented in the module documented below.
The rules authority is the [project overview](https://github.com/users/lukekh/projects/1).

### Coordinates and actions

Arrays use `board[y, x]`. Internal coordinates are zero based: A1 is `(0, 0)`
and I9 is `(8, 8)`. Printed row 9 is at the top; north increases `y`.
Players are Blue = 0 and Red = 1. Positive pieces belong to player 0, negative
pieces to player 1: rock = 1, scissors = 2, paper = 3; empty = 0.
The initial Blue paper occupies B5/C4/D3/E2, rock B4/C3/D2, and scissors C5/D4/E3.
Red is reflected by `(x, y) -> (8-y, 8-x)`. Blue moves first and defends A1;
Red defends I9. Transformed states may change player labels and corner ownership.

`encode_action(x, y, direction) = 8 * (9*y + x) + direction`. Directions are
N, NE, E, SE, S, SW, W, NW, numbered 0 through 7. All 648 slots are stable,
including the 104 with off-board destinations. `action_stays_on_board` rejects
those destinations; an on-board destination alone does **not** imply a legal move.
`decode_action` returns `(x, y, direction)`; `action_destination` returns `(x, y)`
even outside the board. Coordinate formatting/parsing is a Python display helper;
numeric action helpers compile with Numba.

### Serialized layout

The state is an `int8` array of shape `(9, 9, 33)`, exactly 2,673 bytes in C order.
There is no external header or implicit Python state. The byte offset of
`state[y, x, plane]` is `33 * (9*y + x) + plane`.

| Plane | Contents |
| --- | --- |
| 0 | Current signed piece board |
| 1–31 | Up to 31 historical boards, oldest first, **including the current position** |
| 32 | Metadata; offsets below are `9*y + x` within this plane |

| Metadata offsets | Meaning |
| --- | --- |
| 0 | State version, currently 1 |
| 1 | Next player (0 or 1) |
| 2 | A1 defender (0 or 1); the other player defends I9 |
| 3 | Consecutive noncapture plies (0–30) |
| 4 | Valid history length (1–31) |
| 5–9 | Total ply count: five little-endian base-128 digits |
| 10–40 | Next player for each corresponding history slot |
| 41–80 | Reserved, always zero |

The total ply value is `sum(int(meta[5+i]) * 128**i for i in range(5))`.
Every digit is 0–127, avoiding signed-byte overflow and host endianness concerns.
The supported range is 0–34,359,738,367; overflow raises before any mutation.
For example, 128 is `[0, 1, 0, 0, 0]`. Total plies never reset on a capture.
No parity relationship with absolute player labels is required, since future
canonicalization can exchange player labels without changing total plies.

Initialization records one position with next player Blue. A capture discards
earlier history and records its resulting position as slot 0. Each noncapture
appends one resulting position. This fits the capture/initial position plus all
30 subsequent noncapture plies. The current board and next player equal the last
valid history entry, historical players alternate, and history length equals
the noncapture clock plus one. Unused boards, unused historical player bytes,
and reserved bytes must be zero; historical Blue (0) is distinguished from
padding using the valid length. Total plies must be at least the capture clock.

`serialize_state` validates and writes exact C-order bytes. `deserialize_state`
requires the exact byte count, returns an owned writable array, and rejects
unsupported versions, invalid codes/metadata, inconsistent history, and nonzero
padding. `validate_state` checks these same storage invariants in compiled code.
It does not validate legal reachability, inventory, corner winners, or repetition.

### Board ownership and extension points

`Board(2)` is a Numba jitclass in `IntransitiveLogicNumba.py`, consistent with the
repository's compiled Board integration. `get_state`, `get_board`, and
`get_history(index)` always return owned copies. Queries never change storage.
`copy_state(state, True)` validates and copies all planes.
`copy_state(state, False)` borrows the array for queries; callers must not modify
that input while it is borrowed. All provided mutations detach before writing,
so future updates do not affect the borrowed input or previously returned arrays.
`state` is internal mutable storage exposed by the jitclass; direct writes bypass
validation and ownership protections and are reserved for engine internals.

`record_position(pieces, next_player, captured)` is a storage primitive for an
already-validated move result, **not** a public move API. It checks board shape,
piece codes, alternating player, capacity, and overflow before changing storage.
It updates current/history boards, turns, capture clock, and total plies atomically.
The caller supplies the capture flag after validating the move; this function
does not infer or enforce captures, stop on repetition, or determine terminal wins.
At full history, a further noncapture is rejected; terminal enforcement belongs
to the game engine. A capture resets history even at that storage boundary.

Future transition methods must preserve these invariants and detach borrowed
storage before writes. Goal/player transformations must transform the whole
history and its next-player metadata consistently. `get_total_ply()` is the
monotonic source for the future adapter's `getRound()`.

### Validation

From the repository root, using Python 3.11 with NumPy and Numba installed:

```sh
python -m unittest discover -s intransitive/tests -v
```

Validated with Python 3.11.4, NumPy 2.4.6, Numba 0.67.0, and llvmlite 0.49.0.
An isolated environment can be prepared with `python3.11 -m venv /tmp/intransitive-venv`
and `/tmp/intransitive-venv/bin/python -m pip install numpy==2.4.6 numba==0.67.0`.
Use that environment's Python executable for the test command above.

The tests exhaust all coordinates and action slots, compare the setup to explicit
Blue and Red fixtures, assert exact metadata/padding, exercise history capacity
and capture reset, round-trip mutable and immutable serialized buffers, check
128+ total plies and the maximum value, reject malformed states atomically, and
run branch isolation through an actual `njit` caller. Storage fixtures deliberately
do not substitute for legal-move or draw-detection tests in the dependent issues.

## Reversible symmetries

`IntransitiveSymmetries.py` implements issue #4. These are **continuation
equivalences**: they preserve the future game from a transformed state, but may
change the initial type inventory or starting player. Actual games still use
`Board.init_game()` with the official 3/3/4 inventory and Blue first.

The commuting generators are C (rock → scissors → paper → rock), D
(`(x,y) → (y,x)`), and E (`(x,y) → (8-y,8-x)` plus swapping player labels).
Only their 12 combinations are included; arbitrary two-type exchanges reverse
the capture cycle and quarter-turn rotations move the defended corners.

IDs are permanently `k + 3*d + 6*e` for `C^k D^d E^e`, with `k=0..2` and
`d,e=0..1`. Identity/C/D/E have IDs 0/1/3/6. `symmetry_id` reduces integer
exponents modulo their orders; `symmetry_components` decodes IDs.
`compose_symmetries(first, second)` applies first then second;
`inverse_symmetry` reverses a transform. All numeric helpers are callable from
Numba `njit` code; permutation tables are read-only NumPy arrays.

`transform_state(state, id)` validates storage and returns an owned copy,
transforming current and every populated historical board with the same element.
E also exchanges the current mover and every recorded historical mover. History
ordering, counters, version, padding, and exact repetition equality are preserved.
No historical entry is independently canonicalized. D fixes A1/I9, while E swaps
both the corners and player labels: its new A1 defender is
`1 - (1 - old_A1_defender)`, so the stored defender byte stays unchanged for
all 12 elements, even when player 0 initially defends I9.

`ACTION_PERMUTATIONS[id, a]` gives the forward action slot and
`INVERSE_ACTION_PERMUTATIONS[id, transformed_a]` recovers the original. Both
cover all 648 slots, including off-board destinations. `transform_action` maps
one slot; `transform_coordinate` also accepts off-board destinations.
D maps direction `(dx,dy)` to `(dy,dx)` and E maps it to `(-dy,-dx)`;
C leaves action indices unchanged. For example, D sends B5→C5 to E2→E3;
E sends it to E8→E7.

`transform_action_vector(vector, id)` maps a NumPy policy or mask using
`out[p[a]] = vector[a]`, preserving dtype and policy mass.
`transform_player_vector(vector, id)` maps a two-entry **absolute-player** value
or Q vector, swapping entries exactly when E occurs. Relative current-player
training targets belong to the later canonicalization/augmentation integration.
Apply the inverse ID to undo any mapping. Every transform returns fresh storage,
including identity, and never mutates its input.

The test command above also runs exhaustive group laws (all 12³ triples), all
12×648 action mappings and inverses, independent coordinate fixtures, full-state
composition/round trips with 1/5/31 history entries and both corner assignments,
policy/mask/value mapping, unchanged initialization, and an actual compiled caller.
