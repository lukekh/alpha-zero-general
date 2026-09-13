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

The compiled foundation implements coordinates, action slots, official setup,
serialized storage, snapshot ownership, legal movement, captures, and official
wins, plus the modelling-only repetition and noncapture draws (#3).
Reversible symmetries (#4) are implemented in the module documented below.
The Game adapter (#5) supports canonical player perspective and compiled MCTS.
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
earlier history and records its resulting position as slot 0. This is safe:
each legal capture permanently removes exactly one piece, so no position before
that capture can recur. Each noncapture
appends one resulting position. This fits the capture/initial position plus all
30 subsequent noncapture plies. The current board and next player equal the last
valid history entry, historical players alternate, and history length equals
the noncapture clock plus one. Unused boards, unused historical player bytes,
and reserved bytes must be zero; historical Blue (0) is distinguished from
padding using the valid length. Total plies must be at least the capture clock.

`serialize_state` validates and writes exact C-order bytes. `deserialize_state`
requires the exact byte count, returns an owned writable array, and rejects
unsupported versions, invalid codes/metadata, inconsistent history, and nonzero
padding. `validate_state` checks these same storage invariants in compiled code
and rejects the malformed case where both players already occupy their winning
corners. It does not validate legal reachability, inventory, a single corner
winner, or repetition.

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

`raw_movement_mask(pieces, player)` computes piece mobility without consulting
terminal state. `Board.valid_moves(player)` uses that helper to return all legal
one-square moves and returns an all-false mask after a corner win or when the
current player is stuck, or after either modelling draw.
`Board.make_move(action, player, random_seed=0)` rejects
out-of-range, out-of-turn, and illegal actions before mutation, moves the attacker
without changing its type, and records whether the defender was captured.
`Board.check_end_game(next_player)` checks corner wins, next-player stalemate,
threefold repetition, and the 30-noncapture limit, in that order. It returns
`float32` absolute-player vectors: `[1, -1]` or `[-1, 1]` for wins, `[0, 0]`
for ongoing play, and `[1e-4, 1e-4]` for either draw (`DRAW_VALUE` in constants).
This small equal nonzero sentinel makes `result.any()` distinguish terminal draws
from ongoing play, as required by MCTS, Coach, and Arena. It approximates zero
draw utility; it is neither a win nor a reward based on material. The Game adapter
and compiled MCTS preserve these vectors; game registration and full training
pipeline integration remain tracked in #11.

`Board.get_repetition_count()` counts the current exact signed piece array and
next player among valid history slots, including the latest slot. Same-type,
same-colour pieces have no individual identity. Physical positions related only
by symmetry do not match; neither counters nor total ply participate in position
equality. No history or repetition cache exists outside the serialized state.

`Board.get_terminal_reason()` returns `"ongoing"`, `"corner"`, `"stalemate"`,
`"repetition"`, or `"no-capture limit"`. It shares the same terminal decision as
rewards and legal masks, so official wins take precedence and simultaneous draws
consistently report repetition. Queries do not mutate state, and public moves
from any terminal state are rejected before mutation.

`Board.get_score(player)` (the source for the adapter's `getScore`)
reports remaining piece count for diagnostics only. It is not reward shaping,
an official score, or an additional win condition.

All transitions preserve the storage invariants and detach borrowed storage
before writes. Goal/player transformations must transform the whole history and
its next-player metadata consistently. `get_total_ply()` is the monotonic source
for the adapter's `getRound()`.

### Game adapter and canonical search

`IntransitiveGame` implements the shared `Game` interface with player IDs 0/1,
`num_players = 2`, observation shape `(9, 9, 33)`, and 648 actions. It exposes
initialization, transitions, legal masks, absolute-player reward vectors,
remaining-piece diagnostics, total plies, canonicalization, state keys, a training
triple hook, and basic coordinate display. `getInitBoard()` always restores the
official Blue-first opening. `getBoardSize()` describes the complete serialized
observation, including history and metadata, rather than only the 9×9 piece plane.

`getCanonicalForm(state, player)` requires the state's next player and returns an
owned snapshot whose mover is player 0. For player 1, compiled
`Board.swap_players(1)` negates every current and populated historical piece
plane, exchanges the next player, A1 defender, and each populated historical
player label. Coordinates, piece types, history order, clocks, total plies,
version, and zero padding stay intact. A second swap restores every byte.
Canonical player 0 may therefore defend I9 and target A1. Labels in a canonical
state are relative to its mover, not necessarily physical Blue/Red.

Actions keep their source square and direction through canonicalization. Coach
and Arena can apply a selected action directly to the original board. The shared
MCTS helper calls `copy_state`, `make_move`, `swap_players`, and `get_state` on the
Numba Board without a Python fallback. Its next canonical state agrees byte for
byte with applying the action physically and then canonicalizing the next mover.
All mutations detach borrowed states, and returned snapshots survive sibling
searches and later adapter queries.

`stringRepresentation` uses all 2673 versioned serialized bytes in C order,
including history, turns, goals, and counters. This search key is deliberately
separate from physical repetition's exact piece-array-plus-turn comparison.
`getRound()` decodes the five base-128 total-ply digits; captures reset the draw
clock but never reset the round used for MCTS memory cleanup.

`getSymmetries(state, policy, valid_actions)` accepts a player-0 canonical state
and materializes all 12 transformations through Coach's triple interface. Each
output owns its full state, float32 policy, and boolean mask. After transforming
the full history, it colour-only canonicalizes the transformed next player back
to 0. E's reflected coordinates and action permutation survive this step; its
canonical A1 defender flips. Policy and mask use the identical action permutation,
preserving probability mass and zero probabilities on illegal actions.

Deduplication is local to one call and compares the complete serialized state,
policy, and mask. The first occurrence in symmetry-ID order is retained. An
asymmetric complete example yields 12 triples; the official opening with a
uniform legal policy yields six because D leaves the complete example unchanged.
Equal current boards with different histories, policies, or masks remain distinct.
Every populated historical position receives the same transform in a common
player frame, preserving exact repetitions, history order, and capture clocks.

Coach already supplies Q in current-player order and rolls the eventual absolute
outcome into that order. All augmented triples retain that same relative outcome
and Q without an additional E swap. In contrast, `transform_player_vector` is an
absolute-player API and swaps both entries for E. Augmentation only creates
training examples; self-play still starts from the official setup with Blue first.

Materializing up to 12 examples increases replay storage and training work.
Augmentation does not guarantee exact CNN equivariance or a 12× speedup; throughput
and learning efficiency need measurement at matched compute budgets. Minibatch
augmentation and group-averaged inference remain possible later optimizations.

`moveToString` and `printBoard` delegate to `IntransitiveDisplay` for fixed
coordinates, rows 9 through 1, physical colours, piece types, and goal ownership.
Registration, neural inference, and full self-play integration remain in #10/#11.

### Human play and baseline opponents

`IntransitivePlayers` provides `RandomPlayer(game, seed=None)`,
`HumanPlayer(game)`, and `GreedyPlayer(game)`. Each exposes the Arena/pit callback
`play(canonical_state, nb_moves=0)` and returns an integer action in the original
physical coordinates. Callbacks require a nonterminal canonical state with
player 0 to move. Arena checks termination before calling a player; direct calls
on terminal positions raise `ValueError` (human play does not prompt).

The display uses tokens such as `BR` (Blue Rock), `BS` (Blue Scissors), `BP`
(Blue Paper), and `RR`/`RS`/`RP` for Red. Empty squares are `..`. Both defended
corners and their attackers are labelled below the board, even when occupied.
Blue always defends A1 in physical games, so A1 defender metadata recovers
physical colour after canonicalization swaps the player labels. A canonical Red
turn therefore shows positive pieces as Red and prompts `Red move`. This colour
interpretation applies to physical games and label-only canonical states;
spatially transformed training examples are continuation equivalents, not
physical game records.

Human input consists of source and destination, for example `B5 B6`. Parsing also
accepts `B5 C5` as coordinate syntax, but that move is illegal in the official
opening because C5 contains a friendly piece. Input is case insensitive and
allows surrounding whitespace. Malformed coordinates, nonadjacent destinations,
empty sources, and illegal moves report an error and retry without changing any
state/history bytes. EOF and Ctrl-C exit through their normal exceptions.

Random samples uniformly from legal actions using its own NumPy generator; pass
a seed for reproducible evaluation. Greedy uses the exact compiled rule engine
and complete draw history to rank moves in this order:

1. An immediate official win (corner or opponent stalemate).
2. No legal opponent reply that immediately wins.
3. A capture.
4. Smaller Chebyshev distance from the moved piece's destination to its target.
5. Lowest encoded action ID for deterministic ties.

Terminal draws have no opponent reply. This is a bounded two-ply baseline, not
a strength guarantee. Its ranking never changes rewards, rules, draw limits, or
the official opening. Greedy evaluates branches in separate Board storage and
does not alter the callback state or the Game adapter's state through lookahead.

Run human versus greedy now through the shared Arena (install `numpy`, `numba`,
and `tqdm` in the Python environment first):

```sh
python - <<'PY'
from Arena import Arena
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitivePlayers import HumanPlayer, GreedyPlayer

game = IntransitiveGame()
arena = Arena(HumanPlayer(game).play, GreedyPlayer(game).play,
              game, display=game.printBoard)
arena.playGame(verbose=True)  # Human is Blue; use other_way=True to play Red.
PY
```

For seeded baseline evaluation, replace `HumanPlayer(game)` with
`RandomPlayer(game, seed=8)` (import it from the same module), then call
`arena.playGames(4)`. Agent assignments alternate while every physical game
starts with Blue. The callbacks match `pit.create_player`; `pit.py --game
intransitive` discovery and checkpoint opponents still require #10/#11.

### Validation

From the repository root, using Python 3.11 with NumPy, Numba, and tqdm installed:

```sh
python -m unittest discover -s intransitive/tests -v
```

Validated with Python 3.11.4, NumPy 2.4.6, Numba 0.67.0, llvmlite 0.49.0,
and tqdm 4.70.1 (Arena).
An isolated environment can be prepared with `python3.11 -m venv /tmp/intransitive-venv`
and `/tmp/intransitive-venv/bin/python -m pip install numpy==2.4.6 numba==0.67.0 tqdm==4.70.1`.
Use that environment's Python executable for the test command above.

The tests exhaust all coordinates and action slots, compare the setup to explicit
Blue and Red fixtures, assert exact metadata/padding, exercise history capacity
and capture reset, round-trip mutable and immutable serialized buffers, check
128+ total plies and the maximum value, and reject malformed states atomically.
Rule fixtures cover edges, diagonal clearance, every type pairing for both
colours, optional captures, conservation, occupied goals, own corners, corner
wins, empty and blocked armies, terminal masks, and invalid actions. Both storage
and legal transitions run through actual `njit` callers.

Display/player fixtures verify both physical colours, coordinate round trips,
human retries and interrupts, seeded legal random choices, greedy tactical
priorities and deterministic ties, and `pit.create_player` callback compatibility
with game discovery supplied by the fixture. Four seeded greedy/random games
(seeds 8 and 80, each colour assignment) run through the real Arena, checking
official Blue-first setup, legal actions, immutable callback snapshots, and
termination. Restored terminal corner/stalemate/repetition/noncapture fixtures
verify Arena never calls a player after the game ends.

Draw fixtures cover legal nonconsecutive repetitions, exchange of same-type
pieces, exact type/colour/square/turn comparisons, symmetry distinctions, all
30 individual noncaptures, captures on moves 29/30, terminal precedence and
diagnostics, and rejection of moves after draws. Serialized midgame histories
retain their future draw decisions, and all 12 uniform symmetries preserve both
draw reasons. Compiled sibling transitions exercise copying and borrowing the
same parent while retaining independent histories, counters, and results.

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
training targets instead stay unchanged through the augmentation hook above.
Apply the inverse ID to undo any mapping. Every transform returns fresh storage,
including identity, and never mutates its input.

The test command above also runs exhaustive group laws (all 12³ triples), all
12×648 action mappings and inverses, independent coordinate fixtures, full-state
composition/round trips with 1/5/31 history entries and both corner assignments,
policy/mask/value mapping, unchanged initialization, and an actual compiled caller.
Augmentation tests cover all 12 full-state outputs with 1/5/31 history entries,
both goal assignments, E's explicit B5→C5 / E8→E7 action fixture, inverse policies,
legal masks, exact repetitions, independent output ownership, and deduplication
that distinguishes histories, policies, and masks. Real `Coach.executeEpisode`
runs with scripted legal search policies verify both winner perspectives, nonzero
relative Q targets, and compressed draw examples from an official Blue-first
episode. Network-backed training integration remains in #10/#11.
