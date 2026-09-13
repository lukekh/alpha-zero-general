"""Versioned storage and fixed coordinates; see README.md for the wire format."""

from numba import njit

BOARD_SIZE = 9
NUMBER_PLAYERS = 2
BLUE, RED = 0, 1
EMPTY, ROCK, SCISSORS, PAPER = 0, 1, 2, 3
ACTION_SIZE = 648
HISTORY_CAPACITY = 31
NO_CAPTURE_LIMIT = 30
# Equal nonzero terminal utility: shared search/self-play checks result.any().
DRAW_VALUE = 1e-4
STATE_VERSION = 1
STATE_SHAPE = (9, 9, 33)
STATE_BYTES = 2673
CURRENT_PLANE = 0
HISTORY_START = 1
METADATA_PLANE = 32

# Flat row-major offsets within the metadata plane.
META_VERSION = 0
META_NEXT_PLAYER = 1
META_A1_DEFENDER = 2
META_NO_CAPTURE = 3
META_HISTORY_LENGTH = 4
META_TOTAL_PLY = 5
TOTAL_PLY_DIGITS = 5
MAX_TOTAL_PLY = 128**TOTAL_PLY_DIGITS - 1
META_HISTORY_PLAYERS = 10
META_RESERVED = 41

N, NE, E, SE, S, SW, W, NW = range(8)
# (dx, dy): north increases the printed row number.
DIRECTIONS = ((0, 1), (1, 1), (1, 0), (1, -1),
              (0, -1), (-1, -1), (-1, 0), (-1, 1))


@njit(cache=True)
def on_board(x, y):
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


@njit(cache=True)
def encode_action(x, y, direction):
    if x != int(x) or y != int(y) or direction != int(direction):
        raise ValueError("Action coordinates and direction must be integers")
    if not on_board(x, y) or not 0 <= direction < 8:
        raise ValueError("Action source or direction out of range")
    return 8 * (9 * int(y) + int(x)) + int(direction)


@njit(cache=True)
def decode_action(action):
    if action != int(action) or not 0 <= action < ACTION_SIZE:
        raise ValueError("Action must be an integer in [0, 648)")
    action = int(action)
    square = action // 8
    return square % 9, square // 9, action % 8


@njit(cache=True)
def action_destination(action):
    x, y, direction = decode_action(action)
    dx, dy = DIRECTIONS[direction]
    return x + dx, y + dy


@njit(cache=True)
def action_stays_on_board(action):
    x, y = action_destination(action)
    return on_board(x, y)


def parse_coordinate(label):
    """Convert A1..I9 (case insensitive) to zero-based (x, y)."""
    if not isinstance(label, str) or len(label) != 2:
        raise ValueError("Expected a coordinate A1 through I9")
    column, row = label.upper()
    if column not in "ABCDEFGHI" or row not in "123456789":
        raise ValueError("Expected a coordinate A1 through I9")
    return ord(column) - ord("A"), int(row) - 1


def format_coordinate(x, y):
    if x != int(x) or y != int(y) or not on_board(x, y):
        raise ValueError("Coordinate out of range")
    return "ABCDEFGHI"[int(x)] + str(int(y) + 1)
