"""Thread-safe browser opponent: native search with no Python search fallback."""
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from . import BINARY, RustTeacher
from .adapter import validate_backend
from ..heuristics.search import SearchResult
from ..IntransitiveGame import IntransitiveGame


class RustMinimaxOpponent:
    kind = 'rust-minimax'
    label = 'Minimax · Rust'

    def __init__(self, config, binary=BINARY):
        self.config = config
        if not 1 <= config.max_depth <= 32 or config.table_entries > 1000000:
            raise ValueError('Rust minimax supports depths 1–32 and at most 1,000,000 table entries')
        self.binary = Path(binary).resolve()
        if not self.binary.is_file():
            raise ValueError('Build Rust minimax first: cargo build --release '
                             '--manifest-path intransitive/rust_teacher/Cargo.toml --offline')
        self.backend = dict(implementation='rust-v1', node_limit_unit='node_visits',
                            binary=str(self.binary), sha256=hashlib.sha256(self.binary.read_bytes()).hexdigest())
        validate_backend(config, self.backend)
        self.game = IntransitiveGame()
        self.last_result = None

    def reload(self):
        validate_backend(self.config, self.backend)
        self.last_result = None

    def choose(self, state, player):
        # Validate the player and full state at the same game boundary as Python.
        canonical = self.game.getCanonicalForm(state, player)
        if self.game.getGameEnded(canonical, 0).any():
            raise ValueError('Cannot select a move from a terminal position')
        legal = np.flatnonzero(self.game.getValidMoves(canonical, 0))
        validate_backend(self.config, self.backend)
        c = self.config
        started = perf_counter()
        # Each move owns a short-lived native process: no signal handlers,
        # shared pipe state, cross-game hints, or children left after restart.
        with RustTeacher(self.binary) as client:
            row = client.analyze(state, depth=c.max_depth, seconds=c.time_limit,
                radius=c.pressure_radius, weight=c.pressure_weight if c.pressure_enabled else 0.,
                proof_depth=c.proof_depth, proof_nodes=c.proof_nodes,
                table_entries=c.table_entries, node_limit=c.node_limit)
        elapsed = perf_counter() - started
        fallback = row['action'] is None
        if fallback and (row['completed_depth'] or row['stop_reason'] not in ('time', 'nodes')):
            raise RuntimeError('Rust minimax returned no move without exhausting its budget')
        action = int(legal[0]) if fallback else row['action']
        if type(action) is not int or action not in legal:
            raise RuntimeError('Rust minimax returned an illegal move')
        self.last_result = SearchResult(action=action, score=row['score'],
            completed_depth=row['completed_depth'], selected_depth=row['completed_depth'],
            pv=row['pv'], nodes=row['nodes'], work=row['nodes'], proof_nodes=row['proof_nodes'],
            elapsed=elapsed, stopped=not row['complete'], tt_hits=row['tt_hits'],
            stop_reason=row['stop_reason'], score_bound=None if fallback else 'exact',
            selection_source='legal_fallback' if fallback else 'completed_iteration',
            diagnostics_status='native_search_only',
            effective_limits=dict(max_depth=c.max_depth, time_limit=c.time_limit,
                                  node_limit=c.node_limit, node_limit_unit='node_visits'),
            explanation=dict(backend='Rust', native_search_seconds=row['seconds'],
                             binary_sha256=self.backend['sha256'],
                             timing='elapsed includes native process and request overhead'))
        return action
