"""Fixed physical coordinates and colours for original or canonical play states."""

from .IntransitiveConstants import (
    CURRENT_PLANE, DIRECTIONS, META_A1_DEFENDER, META_NEXT_PLAYER,
    META_TOTAL_PLY, METADATA_PLANE, TOTAL_PLY_DIGITS, action_destination,
    decode_action, encode_action, format_coordinate, on_board, parse_coordinate,
)
from .IntransitiveLogicNumba import validate_state


def player_colour(state, player):
    """Recover physical colour after label-only canonicalization.

    Actual play always has Blue defending A1. Spatial training symmetries are
    continuation equivalents, not physical game records.
    """
    defender = int(state[:, :, METADATA_PLANE:].flat[META_A1_DEFENDER])
    return "Blue" if player == defender else "Red"


def move_to_str(action, current_player=None):
    x, y, _ = decode_action(action)
    nx, ny = action_destination(action)
    destination = format_coordinate(nx, ny) if on_board(nx, ny) else "off-board"
    return f"{format_coordinate(x, y)}->{destination}"


def parse_move(text):
    """Parse two adjacent coordinates; board legality is checked by the player."""
    coordinates = text.split()
    if len(coordinates) != 2:
        raise ValueError("Enter source and destination, for example B5 C5")
    x, y = parse_coordinate(coordinates[0])
    nx, ny = parse_coordinate(coordinates[1])
    delta = (nx - x, ny - y)
    if delta not in DIRECTIONS:
        raise ValueError("Move exactly one square horizontally, vertically, or diagonally")
    return int(encode_action(x, y, DIRECTIONS.index(delta)))


def format_board(state):
    """Render text with explicit colour/type tokens, including occupied goals."""
    validate_state(state)
    meta = state[:, :, METADATA_PLANE:]
    mover = int(meta.flat[META_NEXT_PLAYER])
    ply = sum(int(meta.flat[META_TOTAL_PLY + i]) * 128**i
              for i in range(TOTAL_PLY_DIGITS))
    lines = ["     " + "  ".join("ABCDEFGHI")]
    for y in range(8, -1, -1):
        cells = []
        for x in range(9):
            piece = int(state[y, x, CURRENT_PLANE])
            if piece:
                colour = player_colour(state, 0 if piece > 0 else 1)
                cells.append(colour[0] + {1: "R", 2: "S", 3: "P"}[abs(piece)])
            else:
                cells.append("..")
        lines.append(f"{y + 1}   " + " ".join(cells) + f"   {y + 1}")
    lines.extend([
        "     " + "  ".join("ABCDEFGHI"),
        "B = Blue; R = Red. Piece suffix: R = Rock, S = Scissors, P = Paper.",
        "Goals: A1 defended by Blue (Red's target); I9 defended by Red (Blue's target).",
        f"{player_colour(state, mover)} to move (player {mover}); ply: {ply}",
        f"Player 0: {player_colour(state, 0)}; player 1: {player_colour(state, 1)}",
    ])
    return "\n".join(lines)


def print_board(state):
    print(format_board(state))
