"""Play the native minimax teacher as an opponent.

The teacher is otherwise a label-generation backend. Wrapping it as a player
makes the same search that produces labels directly playable, which is the
only way to feel what a time budget buys: at equal wall clock the native
search reaches several plies deeper than the Python evaluator.

One subprocess is held for the whole session and reused between moves.
"""

from pathlib import Path

import numpy as np

from . import BINARY, RustTeacher

# Mirrors the Python time-first preset: a depth ceiling well above what a few
# seconds reaches, so ordinary positions stop on time and the node cap stays a
# safety ceiling rather than the binding limit.
DEFAULTS = dict(depth=20, seconds=5.0, node_limit=1_000_000_000)
SETTING_NAMES = ('depth', 'seconds', 'node_limit', 'radius', 'weight',
                 'table_entries', 'threads')


class RustTeacherPlayer:
    """Arena- and browser-compatible opponent backed by the native search."""

    kind = 'rust'
    # The browser's analysis panel decomposes the Python evaluator's modules.
    # A native result does not carry them, so no analysis is offered rather
    # than a plausible-looking partial one.
    last_result = None

    def __init__(self, game=None, binary=None, **settings):
        unknown = set(settings) - set(SETTING_NAMES)
        if unknown:
            raise ValueError(f'unknown native search settings: {sorted(unknown)}')
        self.game = game
        self.binary = Path(binary) if binary else BINARY
        self.settings = {**DEFAULTS, **settings}
        self.teacher = None
        self.reload()

    @property
    def label(self):
        threads = self.settings.get('threads', 1)
        suffix = f" · {threads} threads" if threads and threads > 1 else ''
        return (f"Native minimax · depth {self.settings['depth']}"
                f" · {self.settings['seconds']:g}s{suffix}")

    def reload(self):
        """Restart the search process; called when a new game begins."""
        self.close()
        if not self.binary.is_file():
            raise FileNotFoundError(
                f'Native teacher binary not found: {self.binary}. Build it with '
                'cargo build --release --offline --manifest-path '
                'intransitive/rust_teacher/Cargo.toml')
        self.teacher = RustTeacher(self.binary)

    def analyze(self, state):
        if self.teacher is None:
            self.reload()
        return self.teacher.analyze(np.ascontiguousarray(state), **self.settings)

    def _action(self, state):
        result = self.analyze(state)
        # An exhausted budget still yields the last completed iteration's move;
        # only a result with no action at all is a failure.
        if result.get('action') is None:
            raise RuntimeError(f"Native search returned no move: {result.get('stop_reason')}")
        return int(result['action'])

    def choose(self, state, player):
        if player != int(state[:, :, 82:84].flat[1]):
            raise ValueError('Player does not match state')
        return self._action(state)

    def play(self, board, nb_moves=0):
        if int(board[:, :, 82:84].flat[1]) != 0:
            raise ValueError('Expected canonical current player zero')
        return self._action(board)

    def close(self):
        teacher, self.teacher = getattr(self, 'teacher', None), None
        if teacher is not None:
            teacher.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
