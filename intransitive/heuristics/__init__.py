"""Standalone, history-aware Intransitive heuristic search."""
from .config import SearchConfig
from .search import AlphaBetaPlayer, exhaustive_minimax

__all__ = ['AlphaBetaPlayer', 'SearchConfig', 'exhaustive_minimax']
