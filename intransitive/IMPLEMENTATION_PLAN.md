# Intransitive implementation plan

Implement the confirmed rules in README.md using Santorini's integration pattern.
Symmetry support is part of the initial implementation and its acceptance tests.

This document specifies proposed implementation decisions, not completed code.
The [rules README](README.md) remains the authority for gameplay. Parameters such
as network width and training batch size below are starting configurations to
measure, not claims of optimal performance.

## Delivery order

1. Define state serialization, 648 actions (81 source squares × 8 directions),
   and reversible symmetry mappings together.
2. Implement IntransitiveConstants.py, IntransitiveLogicNumba.py, and
   IntransitiveGame.py, including exact draw history and Numba-compatible Board
   methods called directly by MCTS.
3. Verify rules, all symmetry combinations, and canonical player perspective.
4. Add IntransitiveDisplay.py and IntransitivePlayers.py for coordinate-based
   human play and random/heuristic evaluation.
5. Add IntransitiveNNet.py and NNet.py: a compact 9×9 residual network with piece,
   goal, and draw-history inputs, masked policy output, and two player values.
6. Register the game in GameSwitcher.py; fix main.py's Santorini-specific source
   backup; verify draw encoding, serialization, replay memory sizing, and the
   existing checkpoint acceptance policy's treatment of draws.
7. Complete a small self-play/training/evaluation/checkpoint-reload run, including
   ONNX inference. Benchmark before scaling training; the wrapper currently uses
   CPU training and ONNX inference.

## Symmetry group

Use zero-based coordinates x = column A–I, y = row 1–9. Implement these generators:

| Generator | Coordinates | Piece type | Colour and side to move |
| --- | --- | --- | --- |
| C: cyclic relabelling | unchanged | rock → scissors → paper → rock | unchanged |
| D: reflection across A1–I9 | (x, y) → (y, x) | unchanged | unchanged |
| E: reflection across A9–I1 with colour swap | (x, y) → (8−y, 8−x) | unchanged | swap Blue and Red |

C has order three; D and E each have order two. They commute, giving 12
transformations C^k D^d E^e, with k in {0,1,2} and d,e in {0,1}. D composed
with E is a 180-degree board rotation combined with colour swap.

These preserve the capture cycle, king movement, and corner objectives. Arbitrary
piece-type swaps reverse the capture cycle and are invalid. A 90-degree rotation
does not preserve the fixed goal corners and is not included.

These are equivalences of positions and their continuations, not necessarily
of the prescribed starting setup. For example, cyclic relabelling changes which
type has four pieces, and E changes the side to move. Every real self-play game
must still start from the specified setup with Blue first; transformed examples
teach the same continuation rules without changing the game's initialization.

## Transform the complete example

- Transform every occupied square and piece category, goal ownership, and current
  player. Swap all player-indexed metadata under E.
- Apply the same transformation to every stored historical position and its side
  to move. Preserve occurrence counts, history order, the capture counter, and
  elapsed move count. Never transform history entries independently.
- Encode action = 8 × source_square + direction, with a documented square and
  direction order. Precompute forward and inverse permutations for all actions.
  D maps direction (dx,dy) to (dy,dx); E maps it to (−dy,−dx). C leaves action
  indices unchanged because actions refer to squares, not piece identities.
- Permute policy probabilities and valid-action masks with the same action map.
- In absolute Blue/Red indexing, E swaps both value and Q target entries. In
  current-player indexing, the relative values remain unchanged after consistent
  canonicalization. Verify the entire Coach target conversion, not just the board.

Repetition means exact equality of physical positions and side to move. Two
different positions related by symmetry must NOT count as repetitions of one
another. Symmetry transformations preserve repetition only when applied uniformly
to the entire trajectory.

## Canonical perspective and training

Coach and Arena apply actions selected on canonical states directly to the
original board. Keep coordinates fixed for player canonicalization, swapping
ownership, goal metadata, and history consistently. Do not introduce a spatial
canonicalization without an explicit inverse action mapping at every caller.

Generate all 12 symmetry combinations for training, then convert them into the
network's canonical perspective. Verify that canonicalization does not accidentally
discard the reflected colour-swap variants. Deduplicate identical complete
training examples so symmetric positions are not unintentionally overweighted.

Initially use the existing getSymmetries interface to supply augmented examples.
Measure replay-memory and training costs; if materialization becomes expensive,
move augmentation to minibatch sampling while preserving coverage of all 12
transformations. Twelve equivalents do not imply a twelvefold speed improvement.

Keep exact history-aware MCTS keys initially. Sharing search nodes or inference
results between symmetric states is a later optimization requiring full-state
canonicalization and inverse policy/value mappings. It must never change physical
repetition detection.

## Required verification

- Test the starting setup, every capture pairing, blocking, boundaries, diagonal
  movement, optional captures, corners, elimination, and stalemate.
- Test initial-position repetition counting, nonconsecutive repetitions, side to
  move, interchangeable pieces, capture resets, the 30-move boundary, and official
  win precedence. Test independent histories in sibling MCTS branches.
- For all 12 transformations, test inverses, closure, C³ = D² = E² = identity,
  and commutativity. Check action permutations are bijections over all 648 slots.
- For reachable states and explicit edge cases, verify that transforming a legal
  move's result equals moving in the transformed state using its mapped action.
- Verify legal masks, captures, winner vectors, draws, history, and counters agree
  under every transformation, including moves that trigger terminal conditions.
- Round-trip policies and targets; preserve policy mass and keep illegal actions
  at zero. Check transformed histories preserve exact repetition counts.
- Test canonical action application for both players through Coach, Arena, and
  MCTS's compiled Board path, and ensure transformations never mutate inputs.
- Run complete random and MCTS games, then an end-to-end training smoke test.
  Compare augmentation enabled/disabled at matched compute budgets, tracking draw
  reasons, game lengths, throughput, and playing strength as both colours.

## 1. Scope and repository contracts

The first deliverable is a correct, playable, trainable implementation with all
12 symmetry transformations exercised. A longer training run follows successful
integration and performance measurements. Existing Santorini checkpoints, god
powers, worker identities, building logic, and network architecture versions do
not carry across.

The implementation must follow the running code rather than older interface
comments or Santorini documentation:

| Existing component | Contract relevant to Intransitive |
| --- | --- |
| GameSwitcher.py | Dynamically imports the game class, NNetWrapper, player classes, and NUMBER_PLAYERS |
| SantoriniGame.py | Reference adapter around a mutable Board with serialized NumPy states |
| MCTS.py | Calls copy_state, make_move, swap_players, and get_state directly in a Numba-compiled helper |
| Coach.py | Starts player 0; obtains canonical actions; applies them in original coordinates; trains on symmetry triples |
| Arena.py | Uses the same canonical-action convention and interprets terminal reward vectors |
| GenericNNetWrapper.py | Consumes fixed-shape float-convertible states and boolean valid masks; handles training, checkpoints, and ONNX |
| main.py | Creates the game and network and runs training; currently backs up Santorini code for every game |
| pit.py | Loads random, human, greedy, or checkpoint-based players for evaluation |

Use player IDs 0 and 1. The original game has Blue = 0 and Red = 1. Canonical
states use 0 for the player to move, which need not be physical Blue. All methods
must agree on whether their input uses original or canonical player labels.

## 2. Coordinates, pieces, and actions

Store squares as board[y, x], where A1 is [0, 0], I1 is [0, 8], and I9 is
[8, 8]. Display row 9 at the top and row 1 at the bottom, with columns A–I from
left to right. Use source_square = 9*y + x.

Use signed piece codes in compact storage:

| Code magnitude | Piece | Captures |
| --- | --- | --- |
| 1 | Rock | Scissors |
| 2 | Scissors | Paper |
| 3 | Paper | Rock |

Zero means empty; positive codes belong to player 0 and negative codes to player
1 in the state's current labelling. Implement a capture lookup table rather than
scattered comparisons. Cyclic relabelling advances the magnitude 1 → 2 → 3 → 1
and preserves the sign. Neural inputs use categorical planes rather than treating
these magnitudes as ordered strengths.

Fix direction indices as follows, with positive dy pointing toward row 9:

| Index | Name | dx | dy |
| --- | --- | --- | --- |
| 0 | North | 0 | 1 |
| 1 | Northeast | 1 | 1 |
| 2 | East | 1 | 0 |
| 3 | Southeast | 1 | -1 |
| 4 | South | 0 | -1 |
| 5 | Southwest | -1 | -1 |
| 6 | West | -1 | 0 |
| 7 | Northwest | -1 | 1 |

Encode action = 8*(9*y + x) + direction. There is no pass action, placement
action, or separate capture action. A capture is determined by the destination.
Keep all 648 slots, including directions that leave the board; the legal mask
disables those slots. This gives fixed output dimensions and bijective symmetry
permutations even for currently illegal actions.

The initial position must exactly match README.md. Include explicit assertions
for 20 occupied squares, 10 pieces per player, the 3/3/4 inventories, no overlap,
empty goal corners, and Blue to move.

## 3. Proposed serialized state

Keep every fact that can affect future legality or termination in the NumPy
state returned by get_state. Do not keep repetition history solely in a Python
object, global variable, or mutable MCTS cache.

A concrete initial layout is an int8 array of shape (9, 9, 33):

| Channel | Contents |
| --- | --- |
| 0 | Current signed piece board |
| 1–31 | Up to 31 chronological board snapshots since the latest capture, including the current position |
| 32 | Packed metadata, using the flattened square order defined above |

The metadata plane uses these flattened indices:

| Index | Meaning |
| --- | --- |
| 0 | Current player ID in this state's labelling |
| 1 | Player ID defending A1; the other player defends I9 |
| 2 | Consecutive moves without capture, 0–30 |
| 3 | Number of occupied history slots, 1–31 |
| 4–5 | Total ply count as low/high base-128 digits |
| 6 | State-format version, initially 1 |
| 7 | Reserved, zero |
| 8–38 | Side to move for each corresponding history snapshot |
| 39–80 | Reserved, zero |

Index 1 is necessary because colour-only canonicalization changes which labelled
player owns each corner. The two fixed corner locations suffice; arbitrary goal
coordinates are unnecessary for the selected symmetry group.

All unused history slots and reserved bytes must be zero. Two copies of the same
serialized state must produce identical keys. There are 2,673 raw bytes per state
before policies, targets, Python overhead, and compression. Measure actual replay
memory; the existing queue-size assumptions are not a reliable budget for this
game, especially after augmentation.

The history capacity follows from the modelled 30-move rule: one snapshot at
initialization or immediately after a capture, plus at most 30 noncapture moves.
On a capture, prior positions can be discarded because piece count strictly
decreases and never increases. Repeating any of those prior positions is impossible.

Maintain total ply separately from the noncapture counter. MCTS uses getRound
for age-based cleanup, so that value must not reset after a capture. Base-128
digits avoid int8 overflow; test beyond 127 plies. getRound returns the decoded
individual-move count, documented explicitly despite its inherited name.

copy_state(state, True) must isolate subsequent writes. copy_state(state, False)
may borrow storage for read-only queries, but those queries must not modify it.
Rebind all views whenever a new state is loaded. get_state may expose Board
storage only under the same ownership conventions as the existing adapter;
callers must receive stable snapshots across subsequent transitions.

## 4. Transition and termination algorithms

For a move from a live state:

1. Decode the source and direction; check that the source belongs to the mover.
2. Require an on-board destination that is empty or holds a capturable enemy.
3. Move the piece, removing any captured defender permanently.
4. Increment total ply and switch the player to move.
5. On capture, reset the noncapture counter and history; otherwise increment it.
6. Append the resulting board and next player to history.
7. Evaluate the terminal conditions in the order below.

The public adapter should reject invalid actions clearly. The compiled search
path may rely on its legal mask for speed, but must use the same transition
semantics. Terminal states expose no playable actions and must never be expanded
by MCTS. Factor a raw movement-mask helper so stalemate checks do not recursively
call terminal checking through getValidMoves.

Terminal priority is:

1. A piece occupies its opponent's defended corner: its owner wins.
2. The next player has no legal move: that player loses.
3. The current exact board and next player have occurred three times: draw.
4. The noncapture counter is 30: draw.
5. Otherwise the game continues.

If both modelling draws occur together, report repetition as the diagnostic
reason; either produces the same result. No new gameplay tie-break is introduced.
Malformed synthetic states with both players already occupying winning corners
should be rejected by test/setup validation rather than assigned arbitrary results.

Return reward vectors in the state's player ordering: [1, -1] or [-1, 1] for a
win, [0, 0] for ongoing play, and a documented small nonzero equal vector such as
[1e-4, 1e-4] for a terminal draw. The current framework tests result.any(), so an
exact zero draw would incorrectly continue play. Verify this sentinel through
Arena classification, Coach labels, and MCTS backup; it introduces only a small
numerical approximation to zero draw utility.

getScore is diagnostic, not a source of rewards. Use remaining piece count and
document that it is not an official score. Provide a separate terminal-reason
helper for logs: corner, stalemate, repetition, or no-capture limit.

## 5. Symmetry implementation details

Create IntransitiveSymmetries.py to own transformation definitions, permutation
tables, and Python-facing augmentation. Put any arrays/functions needed by the
compiled Board path in Numba-compatible form. Keep coordinate and action mapping
logic in one place; display, tests, and training must not invent their own maps.

Number transformations deterministically, for example id = 4*k + 2*d + e.
Composition adds k modulo 3 and XORs d and e. The inverse changes k to −k modulo
3 and retains d and e. Identity has ID zero.

For a physical absolute state, E swaps colours, swaps every recorded player to
move, and reflects all boards. Reflection swaps the corner locations while the
colour swap swaps their owners, preserving the physical rule that Blue defends
A1 and Red defends I9. On states with canonical labels, transform the actual goal
ownership metadata rather than assuming label 0 always means Blue.

For each action permutation p, use destination-index assignment:

```text
transformed_policy[p[a]] = policy[a]
transformed_valids[p[a]] = valids[a]
```

Test the convention explicitly; applying an inverse permutation accidentally can
produce plausible-looking but incorrect targets. The C generator changes piece
planes but leaves both arrays unchanged. Map all eight direction offsets using
the coordinate transform and look up their new direction indices.

As an example, D maps B5 to E2, and maps the move B5 → C5 to E2 → E3.
E maps B5 to E8, and maps B5 → C5 to E8 → E7 while swapping colours.
Both examples must be covered by direct coordinate tests.

Transform history as a single trajectory. Do not independently canonicalize each
snapshot relative to its own player to move: that would erase the shared identity
frame needed to compare repetitions correctly. Each history snapshot retains an
explicit side-to-move label in the current state's common frame.

### Canonical augmentation through the current Coach interface

getSymmetries receives a state already canonicalized to player 0. Its returned
triples must also use player 0 as the player to move, because Coach reuses the
same relative Q values and later supplies the same relative outcome for them.

For each transformation T:

1. Apply T to the full canonical state, policy, and legal mask.
2. If T changed the next player to 1, apply colour-only canonicalization to the
   transformed state, including every history entry and corner ownership.
3. Return the resulting state, transformed policy, and transformed mask.

The second step has no coordinate effect. Relative value/Q targets therefore
remain unchanged, including for E. In an absolute-state testing API, E still swaps
the two vector entries. Test both conventions independently and end to end.

After these steps E still produces a spatially reflected example with transformed
goal metadata. It is not discarded merely because player labels are canonical.
There are up to 12 distinct examples, fewer when a complete example is invariant
under some transformations. Deduplicate only equal state/policy/mask triples;
equal boards with different histories or policies are not duplicates.

Augmentation teaches equivalence but does not guarantee an ordinary CNN makes
exactly symmetric predictions. Group-averaged inference is a possible later
experiment: predict transformed states, map policies back, and average relative
values. It costs additional inference and is not required for the initial model.
An architecture enforcing exact equivariance is also outside the first milestone.

## 6. Network and replay design

Implemented baseline (#10): `IntransitiveNNet.py` and `NNet.py`, network version 1.
The [network contract in README.md](README.md#history-aware-policy-and-value-network-version-1)
records exact feature order, normalizations, checkpoint metadata, and validation.
The fixed 64-channel/four-block model has 326,706 parameters; measured history
encoding and total inference costs are recorded in
`benchmarks/network-v1-cpu.json` and reproducible with
`python -m intransitive.benchmark_network`. The design below describes this
baseline; full pipeline integration and expanded persistence checks remain
#11 and #12.

Keep compact game serialization separate from the network's feature meaning.
Implement deterministic feature extraction inside the network so training,
single predictions, batched inference, and ONNX export all use the same mapping.
Do not feed the packed metadata plane to spatial convolutions as if its cell
locations had gameplay meaning.

The network receives the complete serialized state and the legal-action mask.
Extract current-player/opponent categorical piece planes, defended-corner planes,
the normalized noncapture count, and current-position occurrence count. Also
provide every valid historical snapshot and its relative side to move; the
current repetition count alone cannot predict which future move causes a draw.

A practical initial architecture to benchmark is:

- Encode each historical board as six piece planes plus broadcast validity and
  side-to-move planes. Invalid slots have a zero validity flag and are masked out.
- Apply a shared small convolutional encoder to the 31 history slots, initially
  producing four channels per slot. Preserve slot order when concatenating the
  resulting features with the current-board features.
- Project to 64 channels and use four residual blocks on the 9×9 grid.
- Produce eight direction logits per source square. Reorder to the documented
  square-major action layout, flatten to 648, mask illegal logits, and return
  log-softmax as required by the existing wrapper.
- Produce two tanh value outputs in canonical player order. Use the existing
  policy/value/Q losses initially; no heuristic distance reward is introduced.

Treat these sizes as a baseline. Measure the cost of history encoding before
increasing width or depth. Counter normalization, feature order, network version,
and state-format version must be recorded with checkpoint configuration.

Use materialized getSymmetries triples first for compatibility with Coach. Report
raw self-play positions separately from augmented examples; twelve transformations
do not create twelve independent games. If memory is excessive, use randomized
minibatch transformations with coverage across epochs, testing that each selected
sample's board, mask, and policy undergo the same transform.

Checkpoint loading must work with pit.py's nn_version=-1 initialization convention.
Verify that architecture reconstruction, inference, optimizer use during resumed
training, and ONNX export all use the recorded Intransitive dimensions.

## 7. Integration changes and operational behaviour

Add 'intransitive': 'Intransitive' to GameSwitcher.py. Expose NUMBER_PLAYERS = 2
and implement the complete adapter contract: initialization, dimensions, next
state, legal mask, terminal result, score, round, canonicalization, symmetries,
state key, number of players, move text, and board display.

Use full serialized bytes for the initial MCTS key, including counters, history,
and goal metadata. This conservatively distinguishes some future-equivalent
histories but avoids incorrect merges. Do not confuse this key with the smaller
board-plus-turn comparison used to count physical repetitions.

Replace main.py's fixed Santorini backup with selected-game source copying using
Python filesystem operations. Preserve the game directory structure and record
the effective training settings. Keep unrelated game behaviour unchanged.

Review Arena's int8 state-loading path against the selected layout. Verify
serializing and restoring a midgame state preserves future draws. Exercise the
parallel inference path with separate game/history storage for each worker;
begin smoke tests with one inference worker to make failures easier to diagnose.

Keep the existing acceptance rule initially, but report its implications: draws
are excluded from the decisive-game ratio and an all-draw arena rejects the new
checkpoint. Save enough candidate/run information to inspect these cases. If
early training is dominated by draws, assess draw-aware evaluation separately
before changing acceptance thresholds or the confirmed modelling limits.

Provide random and human players plus a deterministic heuristic baseline. The
heuristic should prioritize immediate wins and avoiding immediate opponent wins,
then use captures and distance to the target as explicit tie-break criteria.
These are opponent heuristics, not rule changes or training rewards.

Human input should accept moves such as B5 C5, display the physical board labels,
and report illegal input without altering state. Display actual colour identity
correctly even when the player callback receives a canonical state.

## 8. Verification matrix and completion criteria

Add focused tests under intransitive/tests/. Prefer tests of observable rules and
equivalence over tests that merely duplicate implementation expressions. Use a
small, independent Python reference transition function in tests to cross-check
the optimized Board on generated legal positions.

| Test area | Required evidence |
| --- | --- |
| Setup | Exact coordinates, inventories, reflection, starting player, empty corners |
| Movement | Each direction, every edge/corner, friendly blockers, diagonal clearance |
| Capture | All nine attacking/defending type pairs for both colours; optional captures |
| Win | Every type reaches target; friendly-corner movement; defended target; no pieces; blocked army |
| Draw | Initial snapshot counted; third occurrence; different next player; nonconsecutive repeats; move 29/30; capture reset |
| Priority | Corner/stalemate wins on the same move as either draw threshold |
| History | No branch leakage; copies round-trip; unused storage zeroed; total ply survives captures and exceeds 127 |
| Symmetry algebra | All 12 transforms, inverses, composition, direction/action permutations |
| Symmetry semantics | Transitions, legal masks, terminal rewards, counters, histories, and goal ownership commute with transforms |
| Training labels | Canonical player remains 0; policy mass preserved; relative Q/value order correct; duplicates handled |
| Network | Correct shapes, finite losses/gradients, illegal probability near zero, categorical/history features meaningful |
| Persistence | Checkpoint reload reproduces predictions; restored history reproduces draw decisions |
| Inference | PyTorch/ONNX agreement within recorded tolerance; single/batched predictions agree |
| Pipeline | Complete self-play, optimizer update, arena, candidate handling, and resumed run |

Use deterministic seeds for generated tests. Retain explicit hand-built fixtures
for rare termination cases rather than hoping random play encounters them. For
each generated live state, test every legal action under all 12 transformations
within a bounded test sample. Include nontrivial histories and canonical states
where player 0 defends I9.

Compare policies with numerical tolerances, but compare compact states, masks,
piece counts, capture flags, and action permutations exactly. Test transforms
against saved input copies to catch mutation.

## 9. Milestones and training rollout

### Milestone A: rule engine and symmetries

Deliver constants, state serialization, compiled Board, adapter, symmetry module,
and rule/symmetry tests. Completion requires all 12 transformations to preserve
transitions and terminal outcomes, and full games to terminate under the confirmed
modelling rules. This milestone includes symmetry correctness, not just a stub
that returns the unmodified board.

### Milestone B: playable training integration

Deliver display, players, network, wrapper, registration, selected-game backup,
and integration tests. Completion requires a small run that generates examples,
updates weights, evaluates a candidate, saves/reloads a checkpoint, and predicts
through both PyTorch and ONNX. A rejected candidate is a valid smoke-test outcome;
failure to save or reload the relevant checkpoint is not.

### Milestone C: measured baseline

Begin with one inference worker, deterministic game transitions, a small search
budget, and a dedicated checkpoint directory. Increase game count and search
budget after measuring time per move and per episode. Choose actual batch sizes,
iteration counts, and run duration from measured hardware throughput; do not start
an unbounded training run using repository defaults.

Record at least:

- Self-play games, original positions, augmented examples, and training updates.
- Median and tail game length; corner, stalemate, repetition, and capture-limit
  termination frequencies.
- Legal-move counts, MCTS simulations per second, inference throughput, replay
  memory, and elapsed time split between self-play/training/evaluation.
- Policy and value losses, candidate acceptance results, and win/draw/loss results
  against random, heuristic, and earlier checkpoints, separately as Blue and Red.
- Effective parameters, random seeds, code revision, state/network versions, and
  checkpoint paths.

Run an augmentation ablation using the same opening rules, model size, evaluation
opponents, and comparable total compute. Report both learning per original
self-play position and learning per wall-clock time. More stored examples alone
is not evidence of better efficiency. Evaluate with agents assigned to both
colours while keeping Blue as the first mover in every physical game.

The initial project is complete when a reproducible baseline checkpoint and its
evaluation results exist, with commands documented in README.md. Longer runs,
symmetry-shared MCTS, inference ensembles, and architecture tuning can then be
judged against that baseline.
