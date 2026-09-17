"""Trusted, search-owned position. Only generated legal moves may be pushed.

Public storage is validated once and never borrowed. History entries are
immutable board+turn bytes; captures replace the active history/count map and
undo retains their previous references. Export reconstructs every public byte,
including ordered history, zero padding and the base-128 total move number.
"""
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import numpy as np

from ..IntransitiveConstants import (
    DIRECTIONS, HISTORY_CAPACITY, MAX_TOTAL_PLY, METADATA_PLANE,
    NO_CAPTURE_LIMIT, STATE_SHAPE, STATE_VERSION, TOTAL_PLY_DIGITS,
)
from ..IntransitiveLogicNumba import validate_state, raw_movement_mask, _corner_winner
from .material import count_pieces, after_capture


@dataclass(slots=True)
class Undo:
    source: int
    target: int
    captured: int
    counts: tuple
    history: list | None
    occurrences: dict | None
    dropped: bytes | None
    key: bytes | None


class SearchPosition:
    def __init__(self, state, *, modelling_draws=True, counts=None):
        validate_state(state)
        self.pieces = state[:, :, 0].copy()
        # A compact contiguous view accepted by the existing material kernel.
        self.material_state = self.pieces[:, :, None]
        meta = state[:, :, METADATA_PLANE:].ravel()
        self.side, self.a1 = int(meta[1]), int(meta[2])
        self.total = sum(int(meta[5+i]) * 128**i for i in range(TOTAL_PLY_DIGITS))
        self.history = [state[:, :, i+1].tobytes() + bytes((int(meta[10+i]),))
                        for i in range(int(meta[4]))]
        self.occurrences = dict(Counter(self.history))
        self.counts = count_pieces(state) if counts is None else counts
        self.modelling_draws = modelling_draws
        self.stack = []
        self._key = None

    @contextmanager
    def null_turn(self):
        """Synthetic side flip: no public move, history or clock increment.

        Modelling draws are disabled throughout this hypothetical subtree.
        Ordinary descendants still make/unmake normally. Never export it.
        """
        side, modelling, key = self.side, self.modelling_draws, self._key
        self.side = 1 - side
        self.modelling_draws = False
        self._key = None
        try:
            yield self
        finally:
            self.side, self.modelling_draws, self._key = side, modelling, key

    @property
    def clock(self):
        return len(self.history) - 1

    def key(self):
        if self._key is None:
            self._key = (self.history[-1][:81] + bytes((self.side, self.a1, self.clock)) +
                         b''.join(item + bytes((count,))
                                  for item, count in sorted(self.occurrences.items())))
        return self._key

    def terminal(self):
        winner = _corner_winner(self.pieces, self.a1)
        if winner >= 0:
            return winner, 'corner'
        if not raw_movement_mask(self.pieces, self.side).any():
            return 1 - self.side, 'stalemate'
        if self.modelling_draws:
            if self.occurrences[self.history[-1]] >= 3:
                return -1, 'repetition'
            if self.clock >= NO_CAPTURE_LIMIT:
                return -1, 'no-capture limit'
        return -1, 'ongoing'

    def legal(self):
        if self.terminal()[1] != 'ongoing':
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(raw_movement_mask(self.pieces, self.side))

    def push(self, action):
        """Apply a generated legal action; pop in a caller-owned finally block."""
        if self.total == MAX_TOTAL_PLY:
            raise ValueError('Total ply overflow')
        source = action // 8
        dx, dy = DIRECTIONS[action % 8]
        target = source + 9 * dy + dx
        captured = int(self.pieces.flat[target])
        if self.modelling_draws and not captured and len(self.history) == HISTORY_CAPACITY:
            raise ValueError('Noncapture history is full')
        dropped = self.history[0] if not captured and len(self.history) == HISTORY_CAPACITY else None
        self.stack.append(Undo(source, target, captured, self.counts,
                               self.history if captured else None,
                               self.occurrences if captured else None, dropped, self._key))
        self.pieces.flat[target] = self.pieces.flat[source]
        self.pieces.flat[source] = 0
        self.side = 1 - self.side
        self.total += 1
        self._key = None
        if captured:
            self.history, self.occurrences = [], {}
            self.counts = after_capture(self.counts, captured)
        elif dropped is not None:
            self.history.pop(0)
            self._remove_occurrence(dropped)
        item = self.pieces.tobytes() + bytes((self.side,))
        self.history.append(item)
        self.occurrences[item] = self.occurrences.get(item, 0) + 1
        return captured

    def _remove_occurrence(self, item):
        count = self.occurrences[item] - 1
        if count:
            self.occurrences[item] = count
        else:
            del self.occurrences[item]

    def pop(self):
        undo = self.stack.pop()
        if undo.history is not None:
            self.history, self.occurrences = undo.history, undo.occurrences
        else:
            self._remove_occurrence(self.history.pop())
            if undo.dropped is not None:
                self.history.insert(0, undo.dropped)
                self.occurrences[undo.dropped] = self.occurrences.get(undo.dropped, 0) + 1
        self.pieces.flat[undo.source] = self.pieces.flat[undo.target]
        self.pieces.flat[undo.target] = undo.captured
        self.side = 1 - self.side
        self.total -= 1
        self.counts, self._key = undo.counts, undo.key

    def export(self):
        state = np.zeros(STATE_SHAPE, dtype=np.int8)
        state[:, :, 0] = self.pieces
        meta = np.zeros(162, dtype=np.int8)
        meta[:5] = STATE_VERSION, self.side, self.a1, self.clock, len(self.history)
        for i in range(TOTAL_PLY_DIGITS):
            meta[5+i] = self.total // 128**i % 128
        for i, item in enumerate(self.history):
            state[:, :, i+1] = np.frombuffer(item, dtype=np.int8, count=81).reshape(9, 9)
            meta[10+i] = item[-1]
        state[:, :, METADATA_PLANE:] = meta.reshape(9, 9, 2)
        return state


@lru_cache(maxsize=1)
def warm_position_kernels():
    from .kernels import winning_actions, no_terminal_win_in_horizon
    from .material import ordered_score
    pieces = np.zeros((9, 9), dtype=np.int8)
    raw_movement_mask(pieces, 0)
    _corner_winner(pieces, 0)
    winning_actions(pieces, np.empty(0, dtype=np.int64), 0, 80)
    no_terminal_win_in_horizon(pieces, 0, 0, 2)
    ordered_score(pieces[:, :, None], 0, (0,)*6, np.zeros((2, 64)), 100., 25.)
