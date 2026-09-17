"""Prepare, smoke-test, run/resume and benchmark opt-in heuristic evolution."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess

from ..tournament.runner import atomic_json
from ..tournament.spec import generate_positions, protocol
from .runner import prepare, run, smoke_receipt, validate
from .strategy import Settings


def revision():
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()


def smoke_spec(positions, mode='depth'):
    return prepare(Settings(population=2, generations=1, search_positions=1,
                            max_games=16, max_nodes=3_000_000, max_seconds=180.), positions,
                   protocol(mode, depth=1, seconds=5. if mode == 'depth' else .05, node_limit=50_000,
                            proof_depth=0, proof_nodes=0, max_plies=2, game_seconds=15.),
                   revision=revision())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--config', type=Path, required=True,
                      help='JSON {settings, protocol, corpus: {seed, lines, max_plies}}')
    prep.add_argument('--output', type=Path, required=True)
    smoke = commands.add_parser('smoke')
    smoke.add_argument('--output', type=Path, required=True)
    smoke.add_argument('--mode', choices=('depth', 'wall', 'mcts'), default='depth')
    for name in ('run', 'benchmark'):
        command = commands.add_parser(name)
        command.add_argument('--manifest', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--smoke', type=Path, required=True,
                             help='Successful smoke directory; no automatic long runs')
        if name == 'benchmark':
            command.add_argument('--seeds', type=int, nargs='+', default=[55, 56])
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        if args.output.exists():
            raise ValueError('Output already exists')
        config = json.loads(args.config.read_text())
        if set(config) != {'settings', 'protocol', 'corpus'}:
            raise ValueError('Expected exactly settings, protocol and corpus')
        spec = prepare(Settings(**config['settings']), generate_positions(**config['corpus']),
                       protocol(**config['protocol']), revision=revision())
        atomic_json(args.output, spec)
        print(json.dumps(dict(manifest=str(args.output), settings=spec['settings'])))
        return
    if args.command == 'smoke':
        path = args.output / 'manifest.json'
        spec = json.loads(path.read_text()) if path.exists() else smoke_spec(generate_positions(54, 9), args.mode)
        if spec['limits']['mode'] != args.mode:
            raise ValueError('Existing smoke has a different protocol mode')
        result = run(spec, args.output)
        print(json.dumps({k: result[k] for k in ('status', 'unique_matches', 'failures', 'eligible_exports', 'wall_seconds')}))
        return
    spec = json.loads(args.manifest.read_text())
    validate(spec)
    receipt = smoke_receipt(args.smoke, spec)
    if args.command == 'run':
        result = run(spec, args.output, smoke=args.smoke)
    else:
        if len(set(args.seeds)) < 2 or len(set(args.seeds)) != len(args.seeds):
            raise ValueError('Benchmark requires at least two distinct optimizer seeds')
        settings = Settings(**spec['settings'])
        contract = dict(template=spec['sha256'], seeds=args.seeds, smoke=receipt,
                        budget_per_run={k: spec['settings'][k] for k in ('max_seconds', 'max_games', 'max_nodes')},
                        total_run_count=2*len(args.seeds))
        path = args.output / 'benchmark.json'
        if path.exists() and json.loads(path.read_text()) != contract:
            raise ValueError('Incompatible benchmark resume')
        atomic_json(path, contract)
        results = []
        # Serial runs only; each arm receives identical caps, corpus and shared
        # initialization. Cache savings are reported, not counted as new games.
        for seed in args.seeds:
            for algorithm in ('evolution', 'random'):
                trial = prepare(replace(settings, seed=seed, algorithm=algorithm), spec['positions'],
                                spec['limits'], revision=spec['revision'])
                folder = args.output / f'{algorithm}-{seed}'
                row = run(trial, folder, smoke=args.smoke)
                results.append(dict(seed=seed, algorithm=algorithm, report=str(folder / 'report.json'),
                                    **{k: row[k] for k in ('status', 'generations', 'unique_candidates', 'unique_matches',
                                                          'games_reserved', 'nodes_reserved', 'measured_search_work',
                                                          'wall_seconds', 'known_child_cpu_seconds', 'failures', 'outcomes', 'eligible_exports')},
                                    convergence=[dict(generation=g['generation'], diversity=g['diversity'],
                                                      search=[{k: e[k] for k in ('config_hash', 'role', 'eligible', 'win_points_lower', 'paired_interval_95')}
                                                              for e in g['search']['leaderboards'][spec['limits']['mode']]],
                                                      validation=[{k: e[k] for k in ('config_hash', 'role', 'eligible', 'wins', 'losses', 'unfinished',
                                                                                     'completion_rate', 'win_points_lower', 'paired_interval_95')}
                                                                  for e in g['validation']['leaderboards'][spec['limits']['mode']]])
                                                 for g in row['convergence']]))
                atomic_json(args.output / 'comparison.json', dict(contract=contract, runs=results,
                    conclusion='Equal wall/game/work ceilings, not equal measured consumption. Defaults are '
                    're-evaluated on the same validation slate each generation. Report all outcomes and '
                    'uncertainty; this benchmark does not establish that evolution is superior.'))
                if row['status'] == 'cancelled':
                    print(json.dumps(dict(status='cancelled', completed_runs=len(results))))
                    return
        result = dict(status='complete', runs=len(results), comparison=str(args.output / 'comparison.json'))
    print(json.dumps({k: result[k] for k in ('status', 'runs', 'comparison', 'unique_matches', 'eligible_exports', 'wall_seconds') if k in result}))


if __name__ == '__main__':
    main()
