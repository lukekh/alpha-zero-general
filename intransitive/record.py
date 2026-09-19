"""Replayable Intransitive PGN-style records (coordinate moves, not chess SAN)."""

from dataclasses import dataclass
import hashlib
import json
import re

import numpy as np

from .IntransitiveDisplay import parse_move
from .IntransitiveLogicNumba import Board
from .heuristics.config import SearchConfig

FORMAT = 'Intransitive-PGN-2'
RULES = 'threefold;80-plies-without-capture'
LEGACY_FORMAT = 'Intransitive-PGN-1'
LEGACY_RULES = 'threefold;30-plies-without-capture'
OFFICIAL_RULES = 'official;modelling-draws-only-in-search'
TAG = re.compile(r'^\[([A-Za-z][A-Za-z0-9]*) ("(?:[^"\\]|\\.)*")\]$')
MOVE = re.compile(r'^([A-I][1-9])(?:->|[-x×→]|\s+)([A-I][1-9])$')
RESULTS = ('*', '1-0', '0-1', '1/2-1/2')


def parse_record_move(text):
    match = MOVE.fullmatch(text.strip().replace(' ', ''))
    if match is None:
        # Also accept the engine's usual two-coordinate input.
        return parse_move(text)
    return parse_move(' '.join(match.groups()))


def state_hash(state):
    return hashlib.sha256(state.tobytes()).hexdigest()


def legacy_state_hash(state):
    """Verify a PGN-1 hash using its original 31-position storage window.

    This projection is only for authenticating old records, never for search.
    """
    meta = state[:, :, 82:84].ravel()
    length = min(int(meta[4]), 31)
    start = int(meta[4]) - length
    legacy = np.zeros((9, 9, 33), dtype=np.int8)
    legacy[:, :, 0] = state[:, :, 0]
    legacy[:, :, 1:length + 1] = state[:, :, start + 1:start + length + 1]
    old_meta = legacy[:, :, 32]
    old_meta.flat[:10] = meta[:10]
    old_meta.flat[0] = 1
    old_meta.flat[3] = length - 1
    old_meta.flat[4] = length
    old_meta.flat[10:10 + length] = meta[10 + start:10 + start + length]
    return state_hash(legacy)


def result_token(board):
    result = board.check_end_game(board.get_next_player())
    if result[0] == 1:
        return '1-0'
    if result[1] == 1:
        return '0-1'
    return '*' if board.get_terminal_reason() == 'ongoing' else '1/2-1/2'


def export_record(moves, board, config, *, opponent='local', human_player=0, last_ai=None, players=None):
    """Export the full active line, including history-dependent draw state."""
    result = result_token(board)
    tags = dict(Variant='Intransitive', Format=FORMAT,
                Rules=RULES if board.modelling_draws else OFFICIAL_RULES,
                Blue='Human' if opponent == 'local' or human_player == 0 else opponent,
                Red='Human' if opponent == 'local' or human_player == 1 else opponent,
                Result=result, PlyCount=str(len(moves)),
                StateSHA256=state_hash(board.get_state()),
                HeuristicConfig=json.dumps(config.to_dict(), separators=(',', ':')))
    if players is not None:
        tags.update(Blue=players[0], Red=players[1])
    if last_ai is not None:
        tags['LastAI'] = json.dumps(last_ai, separators=(',', ':'), allow_nan=False)
    lines = [f'[{key} {json.dumps(value)}]' for key, value in tags.items()]
    tokens = []
    for ply, move in enumerate(moves):
        if ply % 2 == 0:
            tokens.append(f'{ply // 2 + 1}.')
        # The browser distinguishes captures; legality comes from replay.
        tokens.append(move.replace(' → ', '-').replace(' × ', 'x').replace('->', '-'))
    tokens.append(result)
    lines.append('')
    lines.extend(' '.join(tokens[i:i + 12]) for i in range(0, len(tokens), 12))
    return '\n'.join(lines) + '\n'


@dataclass
class Record:
    tags: dict
    config: SearchConfig
    actions: list
    states: list
    last_ai: dict | None


def load_record(text):
    """Validate and replay a copied record without trusting embedded state data."""
    if len(text) > 2_000_000:
        raise ValueError('Record is too large')
    tags, movelines = {}, []
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith('[') and not movelines:
            match = TAG.fullmatch(line)
            if match is None or match[1] in tags:
                raise ValueError('Malformed or duplicate record tag')
            tags[match[1]] = json.loads(match[2])
        elif line:
            movelines.append(line)
    legacy = tags.get('Format') == LEGACY_FORMAT
    supported_rules = (LEGACY_RULES, OFFICIAL_RULES) if legacy else (RULES, OFFICIAL_RULES)
    if (tags.get('Variant') != 'Intransitive'
            or tags.get('Format') not in (FORMAT, LEGACY_FORMAT)
            or tags.get('Rules') not in supported_rules):
        raise ValueError('Expected a copied Intransitive record with supported format and draw rules')
    if 'FEN' in tags or 'SetUp' in tags:
        raise ValueError('Only full games from the official opening are supported')
    try:
        config_data = json.loads(tags['HeuristicConfig'])
        if not isinstance(config_data, dict):
            raise ValueError('HeuristicConfig must be an object')
        config = SearchConfig(**config_data)
        last_ai = json.loads(tags['LastAI']) if 'LastAI' in tags else None
    except (KeyError, TypeError) as exc:
        raise ValueError('Invalid heuristic configuration or AI record') from exc
    # Exported notation is deliberately narrow: numbered coordinate moves and a result.
    tokens = ' '.join(movelines).split()
    if not tokens or tokens[-1] not in RESULTS or tokens[-1] != tags.get('Result'):
        raise ValueError('Missing or inconsistent game result')
    board, actions = Board(modelling_draws=tags['Rules'] != OFFICIAL_RULES), []
    hash_position = legacy_state_hash if legacy else state_hash

    def recorded_result():
        result = result_token(board)
        if (result == '*' and tags['Rules'] == LEGACY_RULES
                and board.get_no_capture_count() >= 30):
            return '1/2-1/2'
        return result

    states = [board.get_state()]
    for token in tokens[:-1]:
        if re.fullmatch(r'\d+\.', token):
            if len(actions) % 2 or int(token[:-1]) != len(actions) // 2 + 1:
                raise ValueError('Incorrect move number')
            continue
        if MOVE.fullmatch(token) is None:
            raise ValueError(f'Invalid coordinate move: {token}')
        try:
            if recorded_result() != '*':
                raise ValueError('The recorded game has already ended')
            action = parse_record_move(token)
            board.make_move(action, board.get_next_player())
        except ValueError as exc:
            raise ValueError(f'Illegal move at ply {len(actions) + 1}: {token}') from exc
        actions.append(action)
        states.append(board.get_state())
    if tags.get('PlyCount') != str(len(actions)) or recorded_result() != tokens[-1]:
        raise ValueError('Move history does not match the recorded ply count or result')
    if tags.get('StateSHA256') != hash_position(states[-1]):
        raise ValueError('Replayed position does not match the recorded state hash')
    if last_ai is not None:
        if not isinstance(last_ai, dict):
            raise ValueError('Invalid LastAI tag')
        ply = last_ai.get('ply')
        if (type(ply) is not int or not 0 <= ply < len(actions)
                or last_ai.get('action') != actions[ply]
                or last_ai.get('state_sha256') != hash_position(states[ply])):
            raise ValueError('LastAI does not match the move history')
    return Record(tags, config, actions, states, last_ai)
