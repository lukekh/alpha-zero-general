"""Bounded eight-way ablation: python -m intransitive.heuristics.benchmark."""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import gzip
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import platform
import random
import resource
import sys
import tarfile
from time import perf_counter
import numpy as np

from . import AlphaBetaPlayer, SearchConfig
from .budget import Budget
from .evaluation import Evaluator
from ..IntransitiveGame import IntransitiveGame
from ..IntransitiveLogicNumba import Board
from ..IntransitivePlayers import GreedyPlayer, RandomPlayer
from ..play import ModelOpponent


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def fixture_state(fixture):
    # Same version-2 format as the independently validated rule fixtures.
    state = np.zeros((9, 9, 84), dtype=np.int8)
    for square, piece in fixture['pieces'].items():
        state[int(square[1]) - 1, ord(square[0]) - 65, 0] = piece
    state[:, :, 1] = state[:, :, 0]
    meta = state[:, :, 82:84]
    meta.flat[0] = 2
    meta.flat[1] = fixture.get('turn', 0)
    meta.flat[4] = 1
    meta.flat[10] = fixture.get('turn', 0)
    return state


def variants(base):
    for flags in product((False, True), repeat=3):
        name = ''.join(letter for letter, active in zip('ADO', flags) if active) or 'core'
        yield name, replace(base, **dict(zip(('attack_enabled', 'defence_enabled', 'overload_enabled'), flags)))


def stats(moves):
    latencies = [m['elapsed'] for m in moves]
    module_seconds = {
        name: sum(m['module_seconds'].get(name, 0.) for m in moves)
        for name in ('routes', 'proof', 'piece_count', 'clear_run',
                     'piece_advantage', 'attacking_position',
                     'defensive_position', 'overload')
    }
    return dict(moves=len(moves), p50_seconds=float(np.percentile(latencies, 50)) if moves else 0,
                p95_seconds=float(np.percentile(latencies, 95)) if moves else 0,
                mean_depth=float(np.mean([m['depth'] for m in moves])) if moves else 0,
                max_table_bytes=max((m.get('table_bytes', 0) for m in moves), default=0),
                fallback_moves=sum(m['depth'] == 0 for m in moves),
                nodes=sum(m['nodes'] for m in moves), work=sum(m['work'] for m in moves),
                proof_nodes=sum(m['proof_nodes'] for m in moves),
                module_seconds=module_seconds,
                module_seconds_per_move={name: seconds / len(moves) if moves else 0.
                                         for name, seconds in module_seconds.items()})


def summarize(games):
    groups = defaultdict(list)
    for game in games:
        groups[(game['mode'], game['variant'], game['opponent'])].append(game)
    result = []
    for (mode, variant, opponent), rows in groups.items():
        scores = [r['score'] for r in rows]
        # Paired colours form one seed unit; deliberately conservative with tiny n.
        n = len({r['seed'] for r in rows})
        radius = math.sqrt(math.log(40) / (2 * n))
        score = float(np.mean(scores))
        result.append(dict(mode=mode, variant=variant, opponent=opponent,
                           wins=scores.count(1.), draws=scores.count(.5), losses=scores.count(0.),
                           score=score, score_interval=[max(0., score-radius), min(1., score+radius)],
                           by_colour={str(c): [sum(r['score'] == s for r in rows if r['colour'] == c)
                                              for s in (1., .5, 0.)] for c in (0, 1)},
                           **stats([m for r in rows for m in r['moves']])))
    return result


def verify_report(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as handle:
            report = json.load(handle)
    else:
        report = json.loads(path.read_text())
    expected_variants = {'core', 'A', 'D', 'O', 'AD', 'AO', 'DO', 'ADO'}
    if set(report['budgets']) != {'nodes', 'wall'}:
        raise ValueError('Expected node and wall-time comparison modes')
    if len(report['positions']) != 128 or len(report['summaries']) != 64:
        raise ValueError('Incomplete held-out searches or summaries')
    game = IntransitiveGame()
    groups = defaultdict(list)
    for record in report['games']:
        key = (record['mode'], record['variant'], record['opponent'],
               record['seed'], record['colour'])
        groups[key].append(record)
        state, side = game.getInitBoard(), 0
        for action in record['actions']:
            if game.getGameEnded(state, side).any():
                raise ValueError('Action recorded after terminal state')
            if type(action) is not int or not game.getValidMoves(state, side)[action]:
                raise ValueError('Illegal recorded action')
            state, side = game.getNextState(state, side, action)
        outcome = game.getGameEnded(state, side)
        if not outcome.any():
            raise ValueError('Recorded game is incomplete')
        board = Board()
        board.copy_state(state, False)
        colour = record['colour']
        score = 1. if outcome[colour] == 1 else 0. if outcome[1-colour] == 1 else .5
        if (score != record['score']
                or board.get_terminal_reason() != record['reason']
                or hashlib.sha256(state.tobytes()).hexdigest() != record['final_state_sha256']):
            raise ValueError('Recorded result, reason or final state does not replay')
    if len(groups) != 256 or any(len(records) != 1 for records in groups.values()):
        raise ValueError('Comparison matrix is incomplete or duplicated')
    if {key[1] for key in groups} != expected_variants:
        raise ValueError('Expected all eight optional-module combinations')
    if len({record['stream'] for record in report['games']}) != 8:
        raise ValueError('Unexpected paired-seed streams')
    return dict(games=len(report['games']), positions=len(report['positions']),
                summaries=len(report['summaries']))


def run(args):
    import torch
    torch.set_num_threads(1)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint
    if checkpoint is None:
        # Read just the frozen delivered checkpoint; never touch a live training run.
        checkpoint = output / 'frozen.pt'
        archive = Path(__file__).parents[1] / 'benchmarks/symmetry/symmetry-artifacts.tar.gz'
        with tarfile.open(archive) as tar:
            checkpoint.write_bytes(tar.extractfile('on/baseline.pt').read())
    model_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    neural = ModelOpponent(checkpoint, args.simulations)
    game = IntransitiveGame()
    initial = game.getInitBoard()
    # Compile rules/routes/greedy/MCTS before timing (no compilation in move limit).
    warm = AlphaBetaPlayer(config=SearchConfig(max_depth=1, time_limit=10, node_limit=10**7))
    warm.analyze(initial)
    GreedyPlayer(game).play(initial)
    neural.choose(initial, 0)
    specs = json.loads(Path(__file__).with_name('positions.json').read_text())
    tuning = []
    for scarcity in (0., .5):
        config = SearchConfig(predator_scarcity_bonus=scarcity)
        for spec in specs['tuning']:
            tuning.append(dict(position=spec['name'], scarcity=scarcity,
                               explanation=Evaluator(game, config).explain(fixture_state(spec), 0, Budget(10**9, 60))))
    report = dict(schema=1, platform=platform.platform(), python=sys.version,
                  numpy=np.__version__, torch=torch.__version__, checkpoint_sha256=model_hash,
                  neural_simulations=args.simulations, seeds=args.seeds,
                  seed_policy='SeedSequence([3400, seed, opponent_index]); same stream for both colours and every variant/mode; fresh trees per game',
                  rules='Official Blue opening; exact engine threefold/80-ply draw limits; no adjudicated/truncated draws',
                  uncertainty='95% Hoeffding score intervals using colour-paired seed units; tiny conditional sample only. Deterministic opponents repeat trajectories; these are not independent strength evidence or opening/training-seed generalization.',
                  budgets={}, configurations={}, tuning=tuning, positions=[], games=[], summaries=[])
    start = perf_counter()
    for mode in args.modes:
        base = SearchConfig(max_depth=args.depth if mode == 'nodes' else 64,
                            node_limit=args.nodes if mode == 'nodes' else 10**9,
                            time_limit=60. if mode == 'nodes' else args.seconds)
        report['budgets'][mode] = base.to_dict()
        for variant, config in variants(base):
            report['configurations'][mode + '/' + variant] = config.to_dict()
            for spec in specs['heldout']:
                result = AlphaBetaPlayer(config=config).analyze(fixture_state(spec))
                report['positions'].append(dict(mode=mode, variant=variant, position=spec['name'], **asdict(result)))
            for opponent_index, opponent in enumerate(('random', 'greedy', 'neural', 'core')):
                for seed in args.seeds:
                    stream = int(np.random.SeedSequence([3400, seed, opponent_index]).generate_state(1)[0])
                    for colour in (0, 1):
                        random.seed(stream)
                        np.random.seed(stream)
                        torch.manual_seed(stream)
                        candidate = AlphaBetaPlayer(config=config)
                        baseline = (RandomPlayer(game, stream) if opponent == 'random' else
                                    GreedyPlayer(game) if opponent == 'greedy' else
                                    AlphaBetaPlayer(config=base) if opponent == 'core' else neural)
                        state, side, moves, actions = game.getInitBoard(), 0, [], []
                        # 19 captures and at most 29 quiet turns between them bound
                        # this game below 601 turns under the unchanged rules.
                        for ply in range(601):
                            outcome = game.getGameEnded(state, side)
                            if outcome.any():
                                break
                            if side == colour:
                                result = candidate.analyze(state)
                                action = result.action
                                moves.append(dict(
                                    elapsed=result.elapsed,
                                    depth=result.completed_depth,
                                    table_bytes=result.table_bytes,
                                    nodes=result.nodes,
                                    work=result.work,
                                    proof_nodes=result.proof_nodes,
                                    module_seconds=result.module_seconds,
                                    module_calls=result.module_calls,
                                ))
                            elif opponent == 'neural':
                                action = baseline.choose(state, side)
                            elif opponent == 'core':
                                action = baseline.choose(state, side)
                            else:
                                action = baseline.play(game.getCanonicalForm(state, side))
                            if not game.getValidMoves(state, side)[action]:
                                raise AssertionError('Opponent selected an illegal move')
                            actions.append(action)
                            state, side = game.getNextState(state, side, action)
                        else:
                            raise AssertionError('Game exceeded the rules-derived bound')
                        board = Board()
                        board.copy_state(state, False)
                        score = 1. if outcome[colour] == 1 else 0. if outcome[1-colour] == 1 else .5
                        report['games'].append(dict(mode=mode, variant=variant, opponent=opponent,
                                                    seed=seed, stream=stream, colour=colour, score=score,
                                                    reason=board.get_terminal_reason(), actions=actions, moves=moves,
                                                    final_state_sha256=hashlib.sha256(state.tobytes()).hexdigest()))
                print(mode, variant, opponent, len(report['games']), 'games', flush=True)
                report['summaries'] = summarize(report['games'])
                report['elapsed_seconds'] = perf_counter() - start
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                report['process_peak_rss_bytes'] = int(rss if sys.platform == 'darwin' else rss * 1024)
                write_json(output / 'comparison.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('checkpoints/issue34-benchmark'))
    parser.add_argument('--checkpoint', type=Path, help='Frozen neural opponent; default archived #16 accepted checkpoint')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--modes', nargs='+', choices=('nodes', 'wall'), default=['nodes', 'wall'])
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--nodes', type=int, default=200000)
    parser.add_argument('--seconds', type=float, default=.1)
    parser.add_argument('--simulations', type=int, default=8)
    parser.add_argument('--verify-report', type=Path,
                        help='Replay and validate an existing comparison, then exit')
    args = parser.parse_args()
    if args.verify_report:
        print(json.dumps(verify_report(args.verify_report), sort_keys=True))
        return
    if args.seconds <= 0 or args.nodes <= 0 or args.depth < 1 or args.simulations < 2:
        parser.error('Use positive budgets and at least two neural simulations')
    run(args)


if __name__ == '__main__':
    main()
