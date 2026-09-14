# Intransitive

Confirmed rules for implementing the game and training a model.

See the [state contract](#state-contract-version-1) below for storage, action
encoding, and tests, and the [implementation plan](IMPLEMENTATION_PLAN.md) for
the delivery sequence.

## Trained baseline

The [issue #15 baseline report](baselines/issue15/README.md) provides measured
results against random, greedy and an earlier checkpoint, split by model colour,
with draw diagnostics and uncertainty. It includes the committed model/replay/source
archive, checksum manifest, bounded reproduction commands and `pit.py` human-play
instructions. Candidate rejection remains explicit; losses alone are not strength
evidence. The official rules and modelling-only appendix below still apply.

The [controlled symmetry comparison](benchmarks/symmetry/README.md) fixes the
optimizer-update budget for augmentation on/off and reports strength against
original positions and compute, with per-colour results and sample limitations.

## Setup, training, evaluation and human play

Run commands from the repository root. Use Python 3.11.4 and the pinned packages:

```sh
python3.11 -m venv /tmp/intransitive-venv
. /tmp/intransitive-venv/bin/activate
python -m pip install -r intransitive/smoke/requirements.txt
export ORT_DISABLE_TELEMETRY=1
python -m unittest discover -s intransitive/tests -v
python -m unittest discover -s tests -v
```

Keep the telemetry environment setting before Python imports ONNX Runtime;
it prevents the documented native macOS shutdown race. Training uses CPU and
ONNX CPU inference; no GPU is needed. The recorded machine has 8 GiB RAM.

For a bounded one-iteration smoke (two self-play games, two candidate games,
four simulations per move), then a continuation with restored weights/replay:

```sh
python -m intransitive.smoke --checkpoint checkpoints/quickstart/initial --seed 13 --backend onnx
python -m intransitive.smoke --settings checkpoints/quickstart/initial/settings.json --resume checkpoints/quickstart/initial/candidate_1.pt --checkpoint checkpoints/quickstart/resumed --seed 14 --backend cpu
```

Use new output directories. Resume requires the adjacent `checkpoint.examples`;
it recreates AdamW and OneCycleLR and restarts iteration numbering. The command
above deliberately resumes the saved candidate, whose acceptance is in
`report.json`; use `retained.pt` for the post-arena incumbent. Both use all 12
symmetries. The [smoke guide](smoke/README.md) also verifies continuation from
the saved source snapshot and through ONNX. The smoke has bounded game/update
counts; the full baseline below additionally enforces wall-clock deadlines.

For the four-iteration measured baseline (600 seconds training, 300 evaluation):

```sh
python -m intransitive.baseline train --output checkpoints/my-baseline
python -m intransitive.baseline evaluate --model-folder checkpoints/my-baseline --output checkpoints/my-baseline/evaluation
python -m intransitive.baseline verify --model-folder checkpoints/my-baseline --output checkpoints/my-baseline/verification
```

To evaluate or play the delivered #15 checkpoint without retraining:

```sh
mkdir -p checkpoints/issue15-delivered
tar -xzf intransitive/baselines/issue15/baseline-artifacts.tar.gz -C checkpoints/issue15-delivered
python -m intransitive.baseline evaluate --model-folder checkpoints/issue15-delivered --output checkpoints/issue15-delivered/new-evaluation
python pit.py intransitive human checkpoints/issue15-delivered/baseline.pt -n 2 -m 32
```

Enter moves such as `B5 C5`. Two games assign the human each colour; physical
Blue always starts. Swap `human` and the checkpoint to play Red first. `pit.py`
uses its casual-play temperature schedule, so use the evaluator for the seeded
strength protocol. The delivered #15 `baseline.pt` is a rejected diagnostic
candidate, with no accepted `best.pt` and no demonstrated improvement.

Model and replay locations, checksums and source snapshots are in the
[baseline artifact guide](baselines/issue15/README.md#artifact-and-human-play)
and [symmetry comparison guide](benchmarks/symmetry/README.md). Local run outputs
are under `checkpoints/`; reports distinguish every candidate from the incumbent.
The comparison's [reproduction script](benchmarks/symmetry/reproduce.sh) runs
both arms, all per-iteration evaluations and independent reload checks.

## Play in a browser

From the repository root, with Python 3.11, NumPy, and Numba installed:

```sh
python -m intransitive.play
```

Open <http://127.0.0.1:8765>. Use `--port 8766` if that port is busy.
The first launch compiles the rules engine before printing the ready URL.
Click a piece, then a highlighted destination. You control both Blue and Red;
this command starts a local game with both sides controlled by you. Undo restores
the complete position and draw history. New game restores the official setup.

To play against a trained model, use the pinned training environment and pass a
checkpoint (for an active run, use its `retained.pt`):

```sh
ORT_DISABLE_TELEMETRY=1 python -m intransitive.play --checkpoint checkpoints/my-run/retained.pt
```

Choose your colour and click **New game**; Blue moves first. The AI replies
automatically using 32 MCTS simulations per move (`--simulations` changes this).
**Undo turn** takes back your move and the AI's reply. New game reloads the latest
checkpoint, while an ongoing game keeps its model fixed. The model label identifies
the saved iteration when available. Training can continue in its separate process.
Use **Retry AI** if an AI request fails. Model loading requires the training packages
above; local play without `--checkpoint` still needs only NumPy and Numba.

The board uses the existing compiled engine, including its modelling-only
threefold repetition and 30-noncapture draw rules, which are explained in the UI.
Each server has one shared game across browser tabs. Refresh keeps the current
game; stopping the server discards it. The server listens only on localhost.
Press Ctrl+C in its terminal to stop it.

To prepare an isolated environment:

```sh
python3.11 -m venv /tmp/intransitive-ui-venv
/tmp/intransitive-ui-venv/bin/python -m pip install numpy numba
/tmp/intransitive-ui-venv/bin/python -m intransitive.play
```

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
and compiled MCTS preserve these vectors throughout the shared training pipeline.

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
The shared training entry points and history-aware network are documented below.

### Shared training, Arena, and replay

`import_game('intransitive')` resolves `IntransitiveGame`, `NNetWrapper`,
`IntransitivePlayers`, and `NUMBER_PLAYERS=2`; `import_logicnumba` resolves its
compiled `Board`. The existing entry points accept the game directly:

```sh
python main.py intransitive --checkpoint ./temp/intransitive -V 1 -P 2
python pit.py intransitive random greedy -n 4
python pit.py intransitive human ./temp/intransitive/best.pt -n 1
```

The training command uses the normal training defaults. For the verified bounded
cycle, exact dependencies, seeds, budgets, logs and source-backed resume commands,
use the [training smoke gate](smoke/README.md). For measured CPU/ONNX throughput,
replay costs, worker comparisons and the budget recommended for #15, see the
[training benchmark](benchmarks/training/README.md).
Every new physical game uses the official setup and Blue moves first. Arena
alternates which agent controls Blue; augmentation never changes initialization.
Sequential self-play starts each episode with a new MCTS tree. Parallel self-play
constructs a separate Game, compiled Board, and search tree for each worker
episode; the ONNX server batches copied observations, with no shared draw history.
Parallel workers receive fixed episode quotas, so `numEps` is an exact game
budget, including targets smaller than the worker count. Finished workers clear
their inference slots; the server only batches active observations. Coach drains
every completed episode before joining all workers and the inference server.
An optional `selfplay_seed` supplies an independent NumPy generator per episode
(`SeedSequence([selfplay_seed, episode_id])`), including temperature-zero ties.

Coach ends an episode on any nonzero terminal vector. Outcome labels are rolled
into each example's mover frame, including equal nonzero draw labels; Q labels
retain that frame. Replay stores complete `(state, policy, outcome, valid, q)`
tuples. Both `--no-compression` and compressed replay retain all 33 int8 planes,
including every historical board and metadata byte. `checkpoint.examples`
round-trips those tuples and can convert either compression mode at load time.

Arena returns absolute player 0's reward and maps it to agent wins/losses after
accounting for the colour assignment. The existing candidate gate accepts when
`new_wins / (new_wins + previous_wins) >= updateThreshold` (default **0.60**).
Draws are excluded from the denominator; an all-draw comparison **rejects** the
candidate and restores the previous network. For example, 3 wins, 2 losses, and
5 draws meet 0.60. This integration does not change the threshold or scoring.
Coach saves each trained `candidate_<iteration>.pt` before arena so a rejected
candidate remains inspectable and reloadable. `temp.pt` retains the pre-update
incumbent; `best.pt` and `checkpoint_<iteration>.pt` are written on acceptance.
Use a new checkpoint directory for resumed invocations to preserve prior files.

To save a position for `pit.py --state`, use
`Arena.serialize_state(state, next_player, turn)`. It retains the existing raw
DEFLATE/base64 format: all C-order int8 board bytes, one absolute-player byte,
and a two-byte big-endian turn count (0–65535). `arena.restore_state(text)`
returns an owned writable state plus player and turn; `playGame(initial_state=...)`
uses that path. For Intransitive, pass the complete `(9,9,33)` state and its
`getRound(state)` value, not only the piece plane. A restored penultimate
repetition or noncapture-limit fixture reaches the same draw on its next move.

Run the focused integration suite with:

```sh
python -m unittest intransitive.tests.test_pipeline -v
```

It covers registry and CLI selection, real Coach/MCTS/network self-play, both
replay formats and compression conversions, real training from replay, restored
draw decisions, terminal MCTS without inference, Arena colour assignment,
candidate acceptance, two actual ONNX workers with independent legal game
trajectories, and an unrelated Santorini import/move/serialization smoke path.
The `main.py` test bounds learning to one real self-play episode. Full checkpoint,
resume/export tests are in `test_checkpoints`; the complete training lifecycle and
resumed CPU/ONNX runs are recorded in the [smoke evidence](smoke/README.md).

### History-aware policy and value network (version 1)

`IntransitiveNNet.py` and `NNet.py` implement #10 through `GenericNNetWrapper`.
Construct `NNetWrapper(game, nn_args)` with `nn_version=1` for new training or
`nn_version=-1` before loading a checkpoint. Version -1 is an empty placeholder
that the shared loader replaces with the saved full model. Other positive
versions are rejected. The wrapper consumes raw canonical float32 batches shaped
`(batch, 9, 9, 33)` and boolean legal masks shaped `(batch, 648)`.

`extract_features` is the single mapping inside `forward`, used by training,
single predictions, batched ONNX inference, and export. It expects validated
state-v1 observations from the Game adapter. No Python preprocessing or spatial
convolution of the packed metadata plane is used. Piece codes are categorical,
never ordinal magnitudes. All history remains in the current canonical frame.

| Current feature channels | Meaning |
| --- | --- |
| 0–5 | Own rock/scissors/paper, opponent rock/scissors/paper (codes 1,2,3,-1,-2,-3) |
| 6–7 | Own and opponent defended-corner indicator planes, decoded from A1 defender |
| 8 | Noncapture clock divided by 30, broadcast spatially |
| 9 | Exact current board-plus-turn occurrences among valid slots, divided by 3 |
| 10 | `log1p(total ply) / log1p(34359738367)`, broadcast spatially |

Total ply is decoded from the five base-128 digits. Its float32 logarithmic
feature is an approximate elapsed-game diagnostic; draw decisions continue to
use exact engine history and counters. The version tag and reserved bytes are
excluded from learned inputs. Feature order, normalizations, state/network
versions, architecture sizes, and action/value order are recorded in
`FEATURE_CONFIG`, the saved full model, and the checkpoint's
`intransitive_config` key.

Each of the 31 historical slots has eight input channels: six categorical piece
planes in the same order, a broadcast validity flag, and a broadcast historical
next-player label (0 own, 1 opponent). All eight channels are masked before a
shared 3×3 convolution maps them to four ReLU channels. The output is masked
again so convolution biases cannot leak padding. The four channels for slot 0,
then slot 1, and so on are concatenated oldest first, including the current
position. Padding is always zero, distinct from a valid empty board or player 0.

The 11 current plus 124 history channels project through a 1×1 convolution to
64 channels, followed by four residual blocks with two 3×3 convolutions each.
The policy head emits eight direction logits per square, explicitly transposes
to `(batch,y,x,direction)`, and flattens to `8*(9*y+x)+direction`. Illegal slots
receive finite `-1e8` log probabilities and exponentiate to zero. Legal slots
normalize to one. An all-false terminal mask yields an all-zero probability
vector; it is not a playable policy. Finite masking keeps the existing KL loss
and zero-target illegal entries numerically safe.

The value head returns two independent tanh values in canonical player order
`[mover, opponent]`. Shared policy and outcome/Q losses are unchanged; Coach's
relative labels require no extra swap for E augmentation and no material,
distance, or other heuristic reward is added. Batch normalization supports
single-position training because each convolution retains the 9×9 grid.

The baseline has **326,706 trainable parameters**, including **292** in the
shared history encoder. Historical convolution costs **723,168 multiply-adds
per state**. CPU measurements on macOS 15.6.1 arm64, Python 3.11.4, PyTorch 2.8.0,
one thread, inference mode, 100 timed iterations after warmup:

| Batch | Feature decoding | History encoding | Complete forward |
| --- | --- | --- | --- |
| 1 | 0.131 ms | 0.089 ms | 1.081 ms |
| 32 | 1.785 ms | 2.618 ms | 34.811 ms |

These are batch latencies, not training or MCTS throughput guarantees. The
fixed-shape encoder computes all 31 slots even for the one-slot opening used
in this benchmark. Decoded current/history tensors occupy 83,916 bytes per
state; encoded history occupies another 40,176 bytes (excluding intermediates,
autograd, and backend workspaces). Sizes remain the proposed baseline; no
width/depth increase is justified by this measurement alone. Raw measurements
are in [network-v1-cpu.json](benchmarks/network-v1-cpu.json). Reproduce with:

```sh
python -m intransitive.benchmark_network --repeats 100
python -m unittest intransitive.tests.test_network -v
```

Network tests cover categorical/goal/counter fixtures, 1/5/31-slot histories,
exact repetition with historical turns, padding and order, all 648 action slots,
finite losses/gradients and parameter changes, canonical value/Q labels, actual
shared-wrapper training, version-1 and version-minus-1 checkpoint loading, and
single/batched ONNX parity within `2e-6` absolute tolerance. The tested inference
stack uses ONNX 1.22.0 and ONNX Runtime 1.30.0 with the shared wrapper's legacy
PyTorch 2.8 exporter. Persistence/resume/export verification is covered by
`test_checkpoints`; shared self-play integration is covered by `test_pipeline`.

### Checkpoint compatibility, resume, and export

Intransitive checkpoint format **1** retains the shared `state_dict` and
`full_model` keys and adds the following metadata:

| Key | Contents |
| --- | --- |
| `intransitive_checkpoint` | Format version, game identifier, `(9,9,33)` input shape, 648 actions, two players, and optimizer/scheduler recreation policy |
| `intransitive_config` | State/network versions, complete feature and action ordering, normalization definitions, and architecture settings |
| `nn_args` | Saved training settings, with `nn_version` set to the actual loaded architecture |
| `nn_version` | Actual network version (1), including when the wrapper was initialized with -1 |

Coach's additional run/search settings remain at the top level, including
temperature and cpuct. The loader reconstructs version 1 from its validated
configuration and strictly loads all weights and BatchNorm statistics. It checks
the fixed piece/history/corner/ply-normalization buffers too. `full_model` is
retained for compatibility but is not the source of Intransitive reconstruction.
Earlier issue-#10 checkpoints without the format envelope are accepted only
with their matching `intransitive_config` and complete compatible `state_dict`.
Other formats need an explicit migration: incompatible metadata, missing weights,
or corrupt files raise descriptive errors, and a failed Intransitive load leaves
the current network intact. Missing paths raise `FileNotFoundError` through the
shared wrapper. Checkpoints use PyTorch pickle loading and must be trusted files.

Both `nn_version=1` and the `pit.py`/inference placeholder `nn_version=-1` work.
Reloading invalidates an existing ONNX session so inference uses the new weights.
A bare wrapper checkpoint can be played using pit's default search settings;
Coach checkpoints retain their stored settings (CLI overrides still apply):

```sh
python pit.py intransitive checkpoints/best.pt random -n 2
python chkpt_to_onnx.py -i checkpoints/best.pt -o /tmp/intransitive.onnx
python -m unittest intransitive.tests.test_checkpoints -v
```

The exported inputs are float32 `board` of shape `(batch,9,9,33)` in canonical
player perspective and boolean `valid_actions` of shape `(batch,648)`. Outputs
are `pi` **log probabilities** `(batch,648)` and `v` values `(batch,2)` in canonical
player order. Exponentiate `pi` to obtain probabilities. Illegal actions remain
exactly zero, legal policies sum to one, and terminal all-false masks yield zero
policy mass. The standalone converter uses opset 16; the shared wrapper uses its
existing runtime export path. Both explicitly use the legacy TorchScript exporter.

The checkpoint acceptance suite compares CPU PyTorch single and batched
predictions, the reloaded model, shared ONNX inference, and the standalone CLI
export at dynamic batch sizes **1, 2, 5**, with `atol=2e-6, rtol=1e-5` for policy
probabilities and values. Fixtures include the opening, 5/29/31-slot histories,
both canonical corner assignments, large total-ply counters, and terminal masks.
Weights and decoded features round-trip exactly. Real pit checkpoint opponents
complete games as both colours, starting from the official Blue-first position,
with every chosen action checked for legality.

Resume restores **network parameters and BatchNorm statistics only**. Each
`GenericNNetWrapper.train()` call constructs a fresh AdamW optimizer and OneCycleLR
scheduler; optimizer moments, step counts, scheduler progress, and RNG state are
not restored. Current caller training arguments remain in effect; saved `nn_args`
are provenance rather than automatic overrides. This is continued training, not
bit-exact continuation. The acceptance suite performs one optimizer update,
saves/reloads, performs one further update after ONNX inference, verifies fresh
optimizer state, finite gradients/parameters and changed weights, then checks
the regenerated ONNX predictions against PyTorch. Replay restoration is separate
and remains the responsibility of Coach's `loadTrainExamples()` path.

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
starts with Blue. The callbacks match `pit.create_player`; `pit.py
intransitive` discovery and checkpoint opponents use the registered game and network.

### Validation

From the repository root, using Python 3.11 with the engine and training
dependencies installed:

```sh
python -m unittest discover -s intransitive/tests -v
```

Validated with Python 3.11.4, NumPy 2.4.6, Numba 0.67.0, llvmlite 0.49.0,
tqdm 4.70.1, PyTorch 2.8.0, torchvision 0.23.0 (Santorini smoke import),
ONNX 1.22.0, ONNX Runtime 1.30.0, coloredlogs, and colorama.
An isolated environment can be prepared with:

```sh
python3.11 -m venv /tmp/intransitive-venv
/tmp/intransitive-venv/bin/python -m pip install -r intransitive/smoke/requirements.txt
export ORT_DISABLE_TELEMETRY=1
```

Use that environment's Python executable for the test command above.

### Independent engine-equivalence gate

Issue #7 adds `tests/reference_rules.py`, an immutable tuple-based Python model
with no production imports. It implements the explicit capture table, legal
transitions, corner ownership, stalemate, exact board-plus-turn repetition,
capture clock, complete history, and all 12 transforms independently. NumPy is
used only to encode/decode the documented state storage. It is a test oracle for
legal play, not a second public engine or a malformed-input validator.

Run the focused rules and bounded generated/metamorphic gate in compiled mode:

```sh
NUMBA_DISABLE_JIT=0 python -m unittest \
  intransitive.tests.test_rules \
  intransitive.tests.test_draws \
  intransitive.tests.test_equivalence -v
```

For just the independent gate, run
`NUMBA_DISABLE_JIT=0 python -m unittest intransitive.tests.test_equivalence -v`.
It rejects disabled JIT and asserts actual nopython signatures for inspection,
transitions, and the shared MCTS helper. No trained network, GPU, or training
performance is involved; MCTS uses eight simulations and a uniform legal policy
with zero value predictions.

Three complete random games use Python `random.Random` seeds **4, 7, 11** and
sorted legal action IDs. Every played transition is checked against the oracle.
Every legal action at plies **0, 7, 19**, plus each game's first capture parent,
is checked against the reference and commuted through **all 12 transforms**.
The fixed seeds include captures, and coverage assertions prevent silently
losing that case. Two complete MCTS games use NumPy `default_rng` seeds
**17, 170** for both the search RNG and action sampling (temperature 1, no
Dirichlet noise). They start from the official Blue-first setup, compare every
physical/canonical state with the oracle, and reject any illegal policy support.

Each comparison includes full serialized bytes, masks for both players, capture
flags, goal ownership/coordinates, next players, terminal vectors/reasons,
repetition counts, complete ordered histories, clocks, total plies, and padding.
Borrowed/copied parents and retained sibling/game snapshots must stay unchanged
after transformations, queries, moves, and actual MCTS searches.

Independent hand-built fixtures cover quiet/capturing moves 29/30, the total-ply
carry 127→128, every piece entering empty/occupied goals with both goal assignments
and player labels, canonical player 0 defending I9, last-piece elimination,
blocked-army stalemate, legal second/third repetitions, and official-win/draw
precedence. Synthetic threshold histories are explicitly separate from reachable
generated games. Every nonidentity symmetry supplies an unequal orbit member
that must not count as an exact physical repetition; the entire resulting
history is then checked under every uniform transform.

All games must terminate within **600 plies**: at most 19 captures can remove the
initial 20 pieces, each capture can follow at most 29 quiet plies, and a final
30 quiet plies forces a modelling draw. The bound is an assertion, not an extra
game rule. Failures include seed, ply, serialized `state_hex`, and action/symmetry
subtest diagnostics. Reconstruct a failing state with
`np.frombuffer(bytes.fromhex(state_hex), dtype=np.int8).reshape(9, 9, 33).copy()`.
The gate prints actual game lengths/reasons and generated branch counts.

### Focused feature coverage

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
episode. Network-backed shared pipeline checks are in `test_pipeline`.


## Standalone alpha–beta opponent

Play without a trained network using `python pit.py intransitive human alphabeta`
or `python -m intransitive.play --opponent alphabeta`. The browser also retains
local, random, greedy and saved-model choices. Independent attack, defence and
overload switches start disabled in the core preset. See the
[configuration, evaluation explanations and commands](heuristics/IMPLEMENTATION.md)
and [controlled comparisons](heuristics/MEASUREMENTS.md).
