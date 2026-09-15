"""Verified counterexamples to defending a corner with the attacker's prey.

Counterfactual positions, not claimed reachable from the official opening.
Positive labels come from the existing depth-five teacher. Rejected moves stay
legal: their tactical refutations are metadata, never fabricated value targets.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from .IntransitiveConstants import STATE_SHAPE, STATE_VERSION
from .IntransitiveDisplay import parse_move
from .IntransitiveGame import IntransitiveGame
from .IntransitiveLogicNumba import validate_state
from .heuristics import AlphaBetaPlayer, SearchConfig
from .heuristics.evaluation import MATE_THRESHOLD
from .supervised_minimax import TEACHER, orbit_key, save_shard, split_for, write_json

TEMPLATES = {
    'capturable_corner_block': dict(runner='B2', weak='B1', strong='C1',
        bad='B1 A1', capture='B2 A1', good='C1 B2'),
    'capturable_route_block': dict(runner='B3', weak='B1', strong='C2',
        bad='B1 A2', capture='B3 A2', good='C2 B3'),
    'race_instead_of_defending': dict(runner='B3', weak='B1', attack='G7',
        bad='B1 A2', capture='B3 A2', good='G7 H8'),
    'strictly_shorter_race': dict(runner='C3', weak='B1', attack='H8', interceptor='G7',
        bad='B1 A2', capture='G7 H8', good='H8 I9'),
}


def candidate(seed, template, noise_count=3):
    """Scissors runner, paper decoy and rock counter; random distant context."""
    spec = TEMPLATES[template]
    rng = np.random.default_rng(seed)
    pieces = np.zeros((9, 9), dtype=np.int8)
    def put(square, code):
        pieces[int(square[1])-1, ord(square[0])-ord('A')] = code
    put(spec['runner'], -2)
    put(spec['weak'], 3)
    if 'strong' in spec:
        put(spec['strong'], 1)
    else:
        put(spec['attack'], 3)
    if 'interceptor' in spec:
        put(spec['interceptor'], -2)
    limit = 6 if 'attack' in spec else 7
    squares = [(x, y) for y in range(3, limit) for x in range(3, limit)]
    if not 1 <= noise_count <= len(squares):
        raise ValueError('Noise must leave the tactical area untouched')
    selected = rng.choice(len(squares), noise_count, replace=False)
    for index, selected_square in enumerate(selected):
        x, y = squares[int(selected_square)]
        # At least one additional opponent piece; capturing the runner must not
        # trivialize the task by immediately eliminating the only enemy.
        sign = -1 if index == 0 else int(rng.choice((-1, 1)))
        kinds = (2, 3) if template == 'strictly_shorter_race' and sign == 1 else (1, 2, 3)
        choices = [kind for kind in kinds
                   if np.count_nonzero(pieces == sign*kind) < (4 if kind == 3 else 3)]
        if not choices:
            return None
        pieces[y, x] = sign * int(rng.choice(choices))
    state = np.zeros(STATE_SHAPE, dtype=np.int8)
    state[:, :, 0] = state[:, :, 1] = pieces
    meta = state[:, :, 82:84]
    meta.flat[0], meta.flat[4] = STATE_VERSION, 1
    validate_state(state)
    return state, dict(template=template, bad_action=parse_move(spec['bad']),
                      attacker_capture=parse_move(spec['capture']),
                      good_action=parse_move(spec['good']), seed=seed)


def refutation(state, bad_action, attacker_capture):
    """Exact official-play proof: capture wins now or after EVERY defender reply.

    This is stronger than finding just one cooperative losing variation. Neither
    repetition nor the quiet-move modelling limit counts as a win in this proof.
    """
    game = IntransitiveGame(modelling_draws=False)
    if game.getGameEnded(state, 0).any() or not game.getValidMoves(state, 0)[bad_action]:
        return None
    child, side = game.getNextState(state, 0, bad_action)
    if game.getGameEnded(child, side).any() or not game.getValidMoves(child, side)[attacker_capture]:
        return None
    captured, side = game.getNextState(child, side, attacker_capture)
    result = game.getGameEnded(captured, side)
    if result.any():
        if result[1] != 1:
            return None
        return dict(prefix=[bad_action, attacker_capture], replies=[],
                    loss_within_plies=2, terminal_reason=game.board.get_terminal_reason())
    replies = []
    for defence in np.flatnonzero(game.getValidMoves(captured, 0)):
        after, next_side = game.getNextState(captured, 0, int(defence))
        if game.getGameEnded(after, next_side).any():
            return None
        winning = None
        for response in np.flatnonzero(game.getValidMoves(after, 1)):
            terminal, turn = game.getNextState(after, 1, int(response))
            if game.getGameEnded(terminal, turn)[1] == 1:
                winning = int(response)
                break
        if winning is None:
            return None
        replies.append(dict(defender_action=int(defence), winning_response=winning))
    if not replies:
        return None
    return dict(prefix=[bad_action, attacker_capture], replies=replies,
                loss_within_plies=4, terminal_reason='corner_or_stalemate')


def attack_proof(state, action):
    """After our advance, every opponent reply permits an immediate corner win."""
    game = IntransitiveGame(modelling_draws=False)
    if game.getGameEnded(state, 0).any() or not game.getValidMoves(state, 0)[action]:
        return None
    child, side = game.getNextState(state, 0, action)
    terminal = game.getGameEnded(child, side)
    if terminal.any():
        return dict(prefix=[action], replies=[], win_within_plies=1) if terminal[0] == 1 else None
    replies = []
    for reply in np.flatnonzero(game.getValidMoves(child, 1)):
        after, next_side = game.getNextState(child, 1, int(reply))
        if game.getGameEnded(after, next_side).any():
            return None
        winning = None
        for response in np.flatnonzero(game.getValidMoves(after, 0)):
            terminal, turn = game.getNextState(after, 0, int(response))
            if game.getGameEnded(terminal, turn)[0] == 1:
                winning = int(response)
                break
        if winning is None:
            return None
        replies.append(dict(opponent_action=int(reply), winning_response=winning))
    return dict(prefix=[action], replies=replies, win_within_plies=3) if replies else None


def planned_refutation(state, bad_action, attacking_line):
    """Verify a fixed enemy continuation against every intervening legal reply."""
    game = IntransitiveGame(modelling_draws=False)
    if not game.getValidMoves(state, 0)[bad_action]:
        return None
    child, side = game.getNextState(state, 0, bad_action)
    counts = Counter()
    def enemy_turn(board, index):
        if game.getGameEnded(board, 1).any():
            return False
        attack = attacking_line[index]
        if not game.getValidMoves(board, 1)[attack]:
            return False
        after, turn = game.getNextState(board, 1, attack)
        terminal = game.getGameEnded(after, turn)
        if terminal.any():
            counts['terminal_leaves'] += 1
            return terminal[1] == 1
        if index+1 == len(attacking_line):
            return False
        for reply in np.flatnonzero(game.getValidMoves(after, 0)):
            counts['defender_replies'] += 1
            next_board, next_side = game.getNextState(after, 0, int(reply))
            if game.getGameEnded(next_board, next_side).any() or not enemy_turn(next_board, index+1):
                return False
        return True
    if not enemy_turn(child, 0):
        return None
    return dict(bad_action=bad_action, attacking_line=list(attacking_line),
                loss_within_plies=2*len(attacking_line), **dict(counts),
                proof='Fixed enemy continuation checked against every intervening legal defender reply')


def label(state, details, config=None):
    strict = details['template'] == 'strictly_shorter_race'
    line = [parse_move(m) for m in ('G7 H8', 'C3 B2', 'B2 A1')]
    proof = (planned_refutation(state, details['bad_action'], line) if strict else
             refutation(state, details['bad_action'], details['attacker_capture']))
    if proof is None:
        return None
    race = 'attack' in TEMPLATES[details['template']]
    winning = attack_proof(state, details['good_action']) if race else None
    rejected_defences = []
    if race:
        if winning is None:
            return None
        for bad, reply in [('B1 A1', 'B3 B2'), ('B1 A2', 'B3 A2'), ('B1 B2', 'B3 B2')]:
            refuted = (planned_refutation(state, parse_move(bad), line) if strict else
                       refutation(state, parse_move(bad), parse_move(reply)))
            if refuted is None:
                return None
            rejected_defences.append(dict(action=parse_move(bad), proof=refuted))
    game = IntransitiveGame()
    result = AlphaBetaPlayer(game, config or SearchConfig(**TEACHER)).analyze(state)
    if result.completed_depth < 5 and result.stop_reason != 'proven_result':
        return None
    # Require the teacher to actually remove the threat with the correct piece,
    # not merely choose another doomed defence or an unrelated winning race.
    if result.action != details['good_action'] or result.score <= -MATE_THRESHOLD:
        return None
    legal = game.getValidMoves(state, 0)
    key = orbit_key(state, result.action, legal)
    # Distant-piece variants share a split, including their symmetry variants.
    # Hash a fixed background independently from the selected template so the
    # two related tactical arrangements cannot cross held-out boundaries.
    background = state[:, :, 0].copy()
    background[:3, :3] = 0
    if race:
        background[6, 6] = 0
    if strict:
        background[7, 7] = 0
    neutral = state.copy()
    neutral[:, :, 0] = neutral[:, :, 1] = background
    from .IntransitiveSymmetries import transform_state
    family = hashlib.sha256(b'vulnerable-defence-v1:' + min(
        transform_state(neutral, s)[:, :, 0].tobytes() for s in range(12))).hexdigest()
    count = int(np.count_nonzero(state[:, :, 0]))
    return dict(state=state, legal=legal, action=result.action, orbit=key, family=family,
        split=split_for(family), teacher_depth=result.completed_depth,
        teacher_reason=result.stop_reason, teacher_score=result.score,
        teacher_work=result.work, teacher_seconds=result.elapsed,
        provenance=dict(source='tactical_defence', template=details['template'],
            seed=details['seed'], stage='endgame' if count <= 8 else 'midgame',
            pieces=count, ply=0, physical_player=0, counterfactual=True,
            history='fresh one-position history; reachability not asserted'),
        defence_lesson=dict(attacker='scissors', vulnerable_defender='paper',
            effective_defender=None if race else 'rock', rejected_action=details['bad_action'],
            good_action=details['good_action'], refutation=proof,
            winning_attack=winning, rejected_defences=rejected_defences,
            race_distances=dict(own=1, opponent=2) if strict else dict(own=2, opponent=2) if race else None,
            explanation=('H8-I9 wins immediately; our runner is one square away and theirs is two. '
                'Any of the three paper blocks wastes the winning tempo, loses our runner, '
                'and permits a forced loss within six plies.' if strict else
                'Continue G7-H8-I9: a forced win in three plies, ahead of the '
                'opponent. Each paper block at A1, A2 or B2 instead forces a loss in four plies.'
                if race else 'Paper is capturable by scissors; this block allows a forced loss. '
                'The depth-five teacher instead captures the runner with rock.')))


def run(output, positions=192, seed=202609151350, seconds=1200, templates=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'dataset').mkdir(exist_ok=True)
    if (output / 'status.json').exists():
        raise ValueError('Output already used; refusing to overwrite')
    import torch
    torch.set_num_threads(1)
    templates = list(TEMPLATES) if templates is None else list(templates)
    if not templates or any(name not in TEMPLATES for name in templates):
        raise ValueError('Unknown or empty template selection')
    n_templates = len(templates)
    target = {name: positions//n_templates + int(i < positions%n_templates) for i, name in enumerate(templates)}
    counts, seen, records, shards = Counter(), set(), [], []
    started = time.time()
    attempts = 0
    shard_number = 0
    def flush():
        nonlocal shard_number
        if records:
            shard_number += 1
            path = output / 'dataset' / f'shard-{shard_number:05d}.pkl.gz'
            save_shard(path, records)
            shards.append(dict(file=str(path.relative_to(output)), positions=len(records),
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            records.clear()
    while len(seen) < positions and time.time()-started < seconds:
        template = templates[attempts % n_templates]
        current_seed = seed + attempts//n_templates
        noise = 3 if (attempts//n_templates) % 2 == 0 else 9
        attempts += 1
        if counts[template] >= target[template]:
            continue
        write_json(output / 'status.json', dict(state='running', unique_positions=len(seen),
            target_positions=positions, by_template=dict(counts), attempts=attempts,
            updated_epoch=time.time()))
        item = candidate(current_seed, template, noise)
        if item is None:
            continue
        state, details = item
        record = label(state, details)
        if record is None or record['orbit'] in seen:
            continue
        seen.add(record['orbit'])
        counts[template] += 1
        records.append(record)
        if len(records) >= 24:
            flush()
            print(json.dumps(dict(positions=len(seen), templates=dict(counts))), flush=True)
    flush()
    manifest = dict(schema='synthetic-minimax-policy-v1', shards=shards,
        unique_positions=len(seen), target_positions=positions, by_template=dict(counts),
        symmetry_variants=12, augmented_examples=12*len(seen), teacher=TEACHER,
        value_targets=False, counterfactual=True, separate_supplement=True,
        currently_used_by_running_training=False,
        supervision='One-hot completed teacher policy; bad moves remain legal. '
                    'Official terminal refutations are metadata, not Q/value labels.',
        split_policy='Related templates with symmetry-equivalent distant context share a family.',
        seed=seed, attempts=attempts, elapsed_seconds=time.time()-started)
    write_json(output / 'dataset-manifest.json', manifest)
    write_json(output / 'status.json', dict(state='complete', target_reached=len(seen)>=positions,
        **{k:v for k,v in manifest.items() if k != 'shards'}))
    print(json.dumps({k:v for k,v in manifest.items() if k != 'shards'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--positions', type=int, default=192)
    parser.add_argument('--seconds', type=float, default=1200)
    args = parser.parse_args()
    if args.positions < 2 or args.seconds <= 0:
        parser.error('Need at least two positions and a positive time budget')
    run(args.output, args.positions, seconds=args.seconds)
