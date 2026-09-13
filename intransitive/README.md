# Intransitive state contract (version 1)

This foundation implements issue #1: coordinates, action slots, official setup,
serialized storage, and snapshot ownership. Movement/official wins (#2), draw
detection (#3), symmetries (#4), and the Game adapter (#5) are separate work.
The rules authority is the [project overview](https://github.com/users/lukekh/projects/1).

## Coordinates and actions

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

## Serialized layout

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

## Board ownership and extension points

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

## Validation

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
