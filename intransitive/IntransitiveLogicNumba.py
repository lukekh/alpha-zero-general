"""Compiled state, official rules, and modelling-only termination."""

import numpy as np
from numba import int8, njit
from numba.experimental import jitclass

from .IntransitiveConstants import (
    ACTION_SIZE, DRAW_VALUE, HISTORY_CAPACITY, HISTORY_START, MAX_TOTAL_PLY,
    METADATA_PLANE, META_A1_DEFENDER, META_HISTORY_LENGTH,
    META_HISTORY_PLAYERS, META_NEXT_PLAYER, META_NO_CAPTURE,
    META_RESERVED, META_TOTAL_PLY, META_VERSION, NO_CAPTURE_LIMIT, NUMBER_PLAYERS,
    STATE_BYTES, STATE_SHAPE, STATE_VERSION, TOTAL_PLY_DIGITS,
    DIRECTIONS, decode_action, on_board,
)


@njit(cache=True)
def observation_size():
    return STATE_SHAPE


@njit(cache=True)
def action_size():
    return ACTION_SIZE


@njit(cache=True)
def raw_movement_mask(pieces, player):
    """Return moves allowed by pieces alone, without terminal-state checks."""
    if player != 0 and player != 1:
        raise ValueError("Player must be 0 or 1")
    actions = np.zeros(ACTION_SIZE, dtype=np.bool_)
    sign = 1 if player == 0 else -1
    for y in range(9):
        for x in range(9):
            source = int(pieces[y, x])
            if source * sign <= 0:
                continue
            attacker = abs(source)
            for direction in range(8):
                dx, dy = DIRECTIONS[direction]
                destination_x = x + dx
                destination_y = y + dy
                if not on_board(destination_x, destination_y):
                    continue
                defender = int(pieces[destination_y, destination_x])
                if defender == 0 or (
                        defender * sign < 0
                        and abs(defender) == attacker % 3 + 1):
                    actions[8 * (9 * y + x) + direction] = True
    return actions


@njit(cache=True)
def _corner_winner(pieces, a1_defender):
    """Return winner, -1 for none, or -2 for a malformed double winner."""
    a1_piece = int(pieces[0, 0])
    i9_piece = int(pieces[8, 8])
    a1_winner = -1
    i9_winner = -1
    if a1_piece != 0:
        owner = 0 if a1_piece > 0 else 1
        if owner != a1_defender:
            a1_winner = owner
    if i9_piece != 0:
        owner = 0 if i9_piece > 0 else 1
        if owner == a1_defender:
            i9_winner = owner
    if a1_winner >= 0 and i9_winner >= 0:
        return -2
    return a1_winner if a1_winner >= 0 else i9_winner


@njit(cache=True)
def _repetition_count(state):
    """Count exact current board-plus-turn occurrences, including the latest."""
    meta = state[:, :, METADATA_PLANE]
    count = 0
    for i in range(int(meta.flat[META_HISTORY_LENGTH])):
        if meta.flat[META_HISTORY_PLAYERS + i] != meta.flat[META_NEXT_PLAYER]:
            continue
        equal = True
        for y in range(9):
            for x in range(9):
                if state[y, x, HISTORY_START + i] != state[y, x, 0]:
                    equal = False
                    break
            if not equal:
                break
        if equal:
            count += 1
    return count


@njit(cache=True)
def _terminal_status(state):
    """Return (winner or -1, reason) without consulting terminal-aware legality."""
    pieces = state[:, :, 0]
    meta = state[:, :, METADATA_PLANE]
    next_player = int(meta.flat[META_NEXT_PLAYER])
    winner = _corner_winner(pieces, int(meta.flat[META_A1_DEFENDER]))
    if winner == -2:
        raise ValueError("Both players cannot occupy their winning corners")
    if winner >= 0:
        return winner, "corner"
    if not raw_movement_mask(pieces, next_player).any():
        return 1 - next_player, "stalemate"
    if _repetition_count(state) >= 3:
        return -1, "repetition"
    if meta.flat[META_NO_CAPTURE] >= NO_CAPTURE_LIMIT:
        return -1, "no-capture limit"
    return -1, "ongoing"


@njit(cache=True)
def validate_state(state):
    """Reject malformed version-1 storage, without changing any bytes."""
    if state.shape != STATE_SHAPE:
        raise ValueError("Expected state shape (9, 9, 33)")
    if not (state.dtype == np.dtype(np.int8)):
        raise ValueError("Expected int8 state")
    meta = state[:, :, METADATA_PLANE]
    if meta.flat[META_VERSION] != STATE_VERSION:
        raise ValueError("Unsupported state version")
    if not 0 <= meta.flat[META_NEXT_PLAYER] <= 1:
        raise ValueError("Invalid next player")
    if not 0 <= meta.flat[META_A1_DEFENDER] <= 1:
        raise ValueError("Invalid A1 defender")
    length = int(meta.flat[META_HISTORY_LENGTH])
    if not 1 <= length <= HISTORY_CAPACITY:
        raise ValueError("Invalid history length")
    if meta.flat[META_NO_CAPTURE] != length - 1:
        raise ValueError("Capture clock must equal history length minus one")
    total = 0
    for i in range(TOTAL_PLY_DIGITS - 1, -1, -1):
        digit = int(meta.flat[META_TOTAL_PLY + i])
        if digit < 0:
            raise ValueError("Invalid total-ply digit")
        total = total * 128 + digit
    if total < length - 1:
        raise ValueError("Total ply is less than the capture clock")
    for i in range(HISTORY_CAPACITY):
        turn = int(meta.flat[META_HISTORY_PLAYERS + i])
        if i < length:
            if not 0 <= turn <= 1:
                raise ValueError("Invalid historical player")
            if i > 0 and turn == meta.flat[META_HISTORY_PLAYERS + i - 1]:
                raise ValueError("Historical players must alternate")
        elif turn != 0:
            raise ValueError("Unused historical players must be zero")
        for y in range(9):
            for x in range(9):
                piece = state[y, x, HISTORY_START + i]
                if i >= length and piece != 0:
                    raise ValueError("Unused history must be zero")
                if not -3 <= piece <= 3:
                    raise ValueError("Invalid historical piece code")
    if meta.flat[META_HISTORY_PLAYERS + length - 1] != meta.flat[META_NEXT_PLAYER]:
        raise ValueError("Latest historical player must match next player")
    for y in range(9):
        for x in range(9):
            if state[y, x, 0] != state[y, x, length]:
                raise ValueError("Latest history must match current board")
    if _corner_winner(state[:, :, 0], int(meta.flat[META_A1_DEFENDER])) == -2:
        raise ValueError("Both players cannot occupy their winning corners")
    for i in range(META_RESERVED, 81):
        if meta.flat[i] != 0:
            raise ValueError("Reserved metadata must be zero")


@jitclass([("state", int8[:, :, :])])
class Board:
    def __init__(self, num_players=NUMBER_PLAYERS):
        if num_players != NUMBER_PLAYERS:
            raise ValueError("Intransitive requires two players")
        self.state = np.zeros(STATE_SHAPE, dtype=np.int8)
        self.init_game()

    def init_game(self):
        self.state = np.zeros(STATE_SHAPE, dtype=np.int8)
        # (x, y, piece), in the official Blue setup order.
        setup = ((1, 4, 3), (2, 3, 3), (3, 2, 3), (4, 1, 3),
                 (1, 3, 1), (2, 2, 1), (3, 1, 1),
                 (2, 4, 2), (3, 3, 2), (4, 2, 2))
        for x, y, piece in setup:
            self.state[y, x, 0] = piece
            self.state[8 - x, 8 - y, 0] = -piece
        self.state[:, :, HISTORY_START] = self.state[:, :, 0]
        meta = self.state[:, :, METADATA_PLANE]
        meta.flat[META_VERSION] = STATE_VERSION
        meta.flat[META_HISTORY_LENGTH] = 1
        # Blue moves first and defends A1; both IDs are zero.

    def get_state(self):
        """Return an owned snapshot, safe across all later Board operations."""
        return self.state.copy()

    def copy_state(self, state, copy_or_not=True):
        validate_state(state)
        if copy_or_not:
            self.state = state.copy()
        else:
            # Borrow for queries. Mutating methods detach before writing.
            self.state = state

    def get_board(self):
        return self.state[:, :, 0].copy()

    def swap_players(self, player):
        """Relabel player as 0 at fixed coordinates, including the full history."""
        if player != int(player) or not 0 <= player <= 1:
            raise ValueError("Player must be 0 or 1")
        if player == 0:
            return
        # Detach borrowed states, just as make_move/record_position do.
        state = self.state.copy()
        meta = state[:, :, METADATA_PLANE]
        length = int(meta.flat[META_HISTORY_LENGTH])
        for plane in range(length + 1):
            state[:, :, plane] = -state[:, :, plane]
        meta.flat[META_NEXT_PLAYER] = 1 - meta.flat[META_NEXT_PLAYER]
        meta.flat[META_A1_DEFENDER] = 1 - meta.flat[META_A1_DEFENDER]
        for i in range(length):
            meta.flat[META_HISTORY_PLAYERS + i] = 1 - meta.flat[META_HISTORY_PLAYERS + i]
        self.state = state

    def get_history(self, index):
        if index != int(index) or not 0 <= index < self.get_history_length():
            raise ValueError("History index out of range")
        return self.state[:, :, HISTORY_START + int(index)].copy()

    def get_history_player(self, index):
        if index != int(index) or not 0 <= index < self.get_history_length():
            raise ValueError("History index out of range")
        return int(self.state[:, :, METADATA_PLANE].flat[META_HISTORY_PLAYERS + int(index)])

    def get_history_length(self):
        return int(self.state[:, :, METADATA_PLANE].flat[META_HISTORY_LENGTH])

    def get_next_player(self):
        return int(self.state[:, :, METADATA_PLANE].flat[META_NEXT_PLAYER])

    def get_a1_defender(self):
        return int(self.state[:, :, METADATA_PLANE].flat[META_A1_DEFENDER])

    def get_no_capture_count(self):
        return int(self.state[:, :, METADATA_PLANE].flat[META_NO_CAPTURE])

    def get_total_ply(self):
        meta = self.state[:, :, METADATA_PLANE]
        total = 0
        for i in range(TOTAL_PLY_DIGITS - 1, -1, -1):
            total = total * 128 + int(meta.flat[META_TOTAL_PLY + i])
        return total

    def valid_moves(self, player):
        if player != int(player) or not 0 <= player <= 1:
            raise ValueError("Player must be 0 or 1")
        player = int(player)
        _, reason = _terminal_status(self.state)
        if reason != "ongoing":
            return np.zeros(ACTION_SIZE, dtype=np.bool_)
        return raw_movement_mask(self.state[:, :, 0], player)

    def make_move(self, move, player, random_seed=0):
        """Apply one legal action atomically and return the next player."""
        if player != int(player) or not 0 <= player <= 1:
            raise ValueError("Player must be 0 or 1")
        player = int(player)
        if player != self.get_next_player():
            raise ValueError("Player is not next to move")
        if move != int(move) or not 0 <= move < ACTION_SIZE:
            raise ValueError("Action must be an integer in [0, 648)")
        move = int(move)
        legal = self.valid_moves(player)
        if not legal[move]:
            raise ValueError("Illegal action")
        source_x, source_y, direction = decode_action(move)
        dx, dy = DIRECTIONS[direction]
        destination_x = source_x + dx
        destination_y = source_y + dy
        pieces = self.state[:, :, 0].copy()
        captured = pieces[destination_y, destination_x] != 0
        pieces[destination_y, destination_x] = pieces[source_y, source_x]
        pieces[source_y, source_x] = 0
        next_player = 1 - player
        self.record_position(pieces, next_player, captured)
        return next_player

    def check_end_game(self, next_player):
        """Return absolute-player wins, zero ongoing, or equal DRAW_VALUE draws."""
        if next_player != int(next_player) or not 0 <= next_player <= 1:
            raise ValueError("Player must be 0 or 1")
        next_player = int(next_player)
        if next_player != self.get_next_player():
            raise ValueError("Player does not match state")
        winner, reason = _terminal_status(self.state)
        result = np.zeros(2, dtype=np.float32)
        if winner >= 0:
            result[winner] = 1.0
            result[1 - winner] = -1.0
        elif reason != "ongoing":
            result[:] = DRAW_VALUE
        return result

    def get_repetition_count(self):
        """Exact board-plus-turn count; piece identities/symmetry are not keys."""
        return _repetition_count(self.state)

    def get_terminal_reason(self):
        """Return ongoing, corner, stalemate, repetition, or no-capture limit."""
        _, reason = _terminal_status(self.state)
        return reason

    def get_score(self, player):
        """Return remaining piece count for diagnostics; this is not a reward."""
        if player != int(player) or not 0 <= player <= 1:
            raise ValueError("Player must be 0 or 1")
        sign = 1 if int(player) == 0 else -1
        count = 0
        for piece in self.state[:, :, 0].flat:
            if piece * sign > 0:
                count += 1
        return count

    def record_position(self, pieces, next_player, captured):
        """Store one already-validated transition; this is not a legal-move API."""
        if pieces.shape != (9, 9) or not (pieces.dtype == np.dtype(np.int8)):
            raise ValueError("Expected int8 pieces with shape (9, 9)")
        if next_player != 1 - self.get_next_player():
            raise ValueError("Players must alternate")
        for y in range(9):
            for x in range(9):
                if not -3 <= pieces[y, x] <= 3:
                    raise ValueError("Invalid piece code")
        if _corner_winner(pieces, self.get_a1_defender()) == -2:
            raise ValueError("Both players cannot occupy their winning corners")
        length = self.get_history_length()
        if not captured and length == HISTORY_CAPACITY:
            raise ValueError("Noncapture history is full")
        total = self.get_total_ply()
        if total == MAX_TOTAL_PLY:
            raise ValueError("Total ply overflow")
        # Detach even when loaded in borrowed/query mode. Check errors first.
        state = self.state.copy()
        meta = state[:, :, METADATA_PLANE]
        if captured:
            state[:, :, HISTORY_START:METADATA_PLANE] = 0
            for i in range(HISTORY_CAPACITY):
                meta.flat[META_HISTORY_PLAYERS + i] = 0
            length = 0
        state[:, :, 0] = pieces
        state[:, :, HISTORY_START + length] = pieces
        meta.flat[META_HISTORY_PLAYERS + length] = next_player
        meta.flat[META_HISTORY_LENGTH] = length + 1
        meta.flat[META_NO_CAPTURE] = length
        meta.flat[META_NEXT_PLAYER] = next_player
        total += 1
        for i in range(TOTAL_PLY_DIGITS):
            meta.flat[META_TOTAL_PLY + i] = total % 128
            total //= 128
        self.state = state


def serialize_state(state):
    """Exact version-1 C-order wire bytes, including history and padding."""
    validate_state(state)
    return state.tobytes(order="C")


def deserialize_state(data):
    if len(data) != STATE_BYTES:
        raise ValueError("Expected exactly 2673 serialized bytes")
    state = np.frombuffer(data, dtype=np.int8).reshape(STATE_SHAPE).copy()
    validate_state(state)
    return state
