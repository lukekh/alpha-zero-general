"""Compiled Board storage foundation. Movement and terminal rules follow separately."""

import numpy as np
from numba import int8, njit
from numba.experimental import jitclass

from .IntransitiveConstants import (
    ACTION_SIZE, HISTORY_CAPACITY, HISTORY_START, MAX_TOTAL_PLY,
    METADATA_PLANE, META_A1_DEFENDER, META_HISTORY_LENGTH,
    META_HISTORY_PLAYERS, META_NEXT_PLAYER, META_NO_CAPTURE,
    META_RESERVED, META_TOTAL_PLY, META_VERSION, NUMBER_PLAYERS,
    STATE_BYTES, STATE_SHAPE, STATE_VERSION, TOTAL_PLY_DIGITS,
)


@njit(cache=True)
def observation_size():
    return STATE_SHAPE


@njit(cache=True)
def action_size():
    return ACTION_SIZE


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
