# Intransitive

Confirmed rules for implementing the game and training a model.

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
