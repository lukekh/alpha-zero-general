"""Analyse a copied game: uv run python -m intransitive.heuristics.analyze game.pgn."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

from ..IntransitiveDisplay import format_board, move_to_str, player_colour
from ..IntransitiveGame import IntransitiveGame
from ..IntransitiveLogicNumba import search_observation
from ..record import OFFICIAL_RULES, load_record, parse_record_move, state_hash
from .budget import Budget, BudgetExpired
from .config import SearchConfig
from .evaluation import Evaluator, terminal_value
from .search import AlphaBetaPlayer


def evaluate_position(state, config=None, *, perspective=None):
    """Call the actual heuristic; scores/terms are from the named player's side.

    A fresh budget prevents diagnostics from consuming the move search budget.
    Budget exhaustion is explicit rather than a misleading zero score.
    """
    config = config or SearchConfig()
    side = int(state[:, :, 82:84].flat[1]) if perspective is None else perspective
    if type(side) is not int or side not in (0, 1):
        raise ValueError('Perspective must be player 0 or 1')
    budget = Budget(config.node_limit, config.time_limit)
    try:
        explanation = Evaluator(IntransitiveGame(), config).explain(state, side, budget)
    except BudgetExpired:
        explanation = dict(status='budget exhausted before evaluation completed', score=None)
    return dict(perspective=player_colour(state, side), **explanation,
                work=budget.work, elapsed=budget.clock() - budget.start)


def analyze_record(text, *, ply=None, last_ai=False, config=None, moves=()):
    """Replay any ply and compare candidate moves with the production evaluator.

    Static candidate scores share the root perspective. Search scores describe
    searched continuations, and are intentionally separate from static scores.
    """
    record = load_record(text)
    if last_ai:
        if ply is not None or record.last_ai is None:
            raise ValueError('Use --last-ai only when LastAI exists, without --ply')
        ply = record.last_ai['ply']
    ply = len(record.actions) if ply is None else ply
    if type(ply) is not int or not 0 <= ply <= len(record.actions):
        raise ValueError(f'Ply must be between 0 and {len(record.actions)} (moves already played)')
    state = record.states[ply]
    physical_hash = state_hash(state)
    if record.tags['Rules'] == OFFICIAL_RULES:
        state = search_observation(state)
    config = config or record.config
    side = int(state[:, :, 82:84].flat[1])
    game = IntransitiveGame()
    # Warm compiled validation before starting a timed search.
    terminal = terminal_value(game, state, side)
    if terminal is None:
        game.getValidMoves(state, side)
    report = dict(ply=ply, perspective=player_colour(state, side),
                  board=format_board(state), state_sha256=physical_hash,
                  search_state_sha256=state_hash(state),
                  recorded_config=record.config.to_dict(), config=config.to_dict(),
                  evaluation=evaluate_position(state, config),
                  recorded_ai=record.last_ai if record.last_ai and record.last_ai['ply'] == ply else None,
                  search=None, candidates=[],
                  note='Positive scores favour the perspective player. Candidate evaluations are static, '
                       'not minimax values. Fresh searches have an empty cache; time limits, cache history '
                       'and engine changes can change the result. Recorded AI data is the original decision.')
    candidates = [parse_record_move(move) for move in moves]
    if ply < len(record.actions):
        candidates.insert(0, record.actions[ply])
    if terminal is None:
        result = AlphaBetaPlayer(game, config).analyze(state)
        report['search'] = dict(asdict(result), move=move_to_str(result.action),
                              line=[move_to_str(a) for a in result.pv])
        candidates.append(result.action)
    for action in dict.fromkeys(candidates):
        child, _ = game.getNextState(state, side, action)
        report['candidates'].append(dict(move=move_to_str(action),
            played=ply < len(record.actions) and action == record.actions[ply],
            selected=report['search'] is not None and action == report['search']['action'],
            evaluation=evaluate_position(child, config, perspective=side)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', help='Copied .pgn file, or - to read stdin')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--ply', type=int, help='Analyse after this many moves (0 = opening)')
    selection.add_argument('--last-ai', action='store_true', help='Analyse immediately before the last AI move')
    parser.add_argument('--move', action='append', default=[], help='Compare a candidate, e.g. C5-D5; repeatable')
    parser.add_argument('--config', type=Path, help='Override the exported heuristic configuration')
    parser.add_argument('--depth', type=int, help='Override maximum search depth')
    parser.add_argument('--time', type=float, help='Override seconds per search/evaluation')
    parser.add_argument('--work', type=int, help='Override work per search/evaluation')
    args = parser.parse_args()
    try:
        text = sys.stdin.read() if args.record == '-' else Path(args.record).read_text()
        config = SearchConfig.from_file(args.config) if args.config else load_record(text).config
        overrides = {k: v for k, v in dict(max_depth=args.depth, time_limit=args.time,
                                          node_limit=args.work).items() if v is not None}
        report = analyze_record(text, ply=args.ply, last_ai=args.last_ai,
                                config=replace(config, **overrides), moves=args.move)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
