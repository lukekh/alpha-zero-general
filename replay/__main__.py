"""Command line for recording and replaying AlphaZero search pools.

    python -m replay record  --game intransitive --checkpoint <file> --games 8 --out <dir>
    python -m replay measure --pool <dir>
    python -m replay check   --pool <dir> --game intransitive
    python -m replay label   --pool <dir> --depth 6
    python -m replay sweep   --pool <dir> --game intransitive --grid numMCTSSims=100,200,400,800
"""

import argparse
import json
import os
import sys

os.environ.setdefault('ORT_DISABLE_TELEMETRY', '1')

from pathlib import Path

import numpy as np

from GameSwitcher import import_game

from . import labels as labels_module
from .bench import DEFAULT_SIZES, batch_latency
from .record import (load_positions, record_games, save_positions, temperature_schedule)
from .store import EvalStore
from .sweep import BASE_ARGS, expand_grid, make_args, self_check, sweep


def load_game_and_net(game_name, checkpoint, backend=None):
    Game, NNet, _players, _players_count = import_game(game_name)
    game = Game()
    net = NNet(game, dict(lr=None, dropout=0., epochs=None, batch_size=None, nn_version=-1))
    folder, filename = os.path.split(checkpoint)
    keys = net.load_checkpoint(folder or '.', filename)
    if keys is None:
        raise SystemExit(f'could not load checkpoint {checkpoint}')
    if backend:
        net.device['inference'] = backend
    return game, net, keys


def load_game(game_name):
    Game, _NNet, _players, _count = import_game(game_name)
    return Game()


def base_from_checkpoint(keys):
    cpuct = keys.get('cpuct', BASE_ARGS['cpuct'])
    cpuct = float(cpuct[0]) if isinstance(cpuct, (list, tuple)) else float(cpuct)
    return dict(
        numMCTSSims=int(keys.get('numMCTSSims', BASE_ARGS['numMCTSSims'])),
        cpuct=cpuct,
        fpu=float(keys.get('fpu', BASE_ARGS['fpu'])),
        universes=int(keys.get('universes', BASE_ARGS['universes'])),
        forced_playouts=bool(keys.get('forced_playouts', BASE_ARGS['forced_playouts'])),
    )


def parse_value(text):
    lowered = text.lower()
    if lowered in ('true', 'false'):
        return lowered == 'true'
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_grid(entries):
    grid = {}
    for entry in entries or []:
        if '=' not in entry:
            raise SystemExit(f'grid entry {entry!r} must look like name=value,value')
        name, values = entry.split('=', 1)
        grid[name.strip()] = [parse_value(v.strip()) for v in values.split(',') if v.strip()]
    return grid


# --------------------------------------------------------------------- record

def command_record(args):
    game, net, keys = load_game_and_net(args.game, args.checkpoint, args.backend)
    base = base_from_checkpoint(keys)
    for name in ('numMCTSSims', 'cpuct', 'fpu', 'universes'):
        value = getattr(args, name.lower() if name != 'numMCTSSims' else 'sims')
        if value is not None:
            base[name] = value
    if args.forced_playouts:
        base['forced_playouts'] = True
    search_args = make_args(base, {})

    def progress(games_done, positions, states, inferences):
        print(f'  game {games_done}/{args.games}: {positions} positions, '
              f'{states} pool states, {inferences} inferences', flush=True)

    print(f'recording {args.games} games at {search_args.numMCTSSims} simulations '
          f'(cpuct={search_args.cpuct}, fpu={search_args.fpu}, '
          f'universes={search_args.universes}, forced_playouts={search_args.forced_playouts})')
    store, positions, summaries = record_games(
        game, net, search_args, games=args.games, seed=args.seed,
        temperature=temperature_schedule(args.temp_begin, args.temp_end, args.temp_halflife),
        tree_reuse=args.tree_reuse, max_plies=args.max_plies,
        split_seed=args.split_seed, select_fraction=args.select_fraction,
        progress=progress)

    pool_manifest = store.save(args.out, shard_entries=args.shard_entries)
    position_manifest = save_positions(args.out, positions, summaries, metadata=dict(
        game=args.game, checkpoint=os.path.abspath(args.checkpoint),
        backend=net.device['inference'], search_args=store.metadata['reference_args']))
    print(json.dumps(dict(pool={k: v for k, v in pool_manifest.items() if k != 'shards'},
                          positions={k: v for k, v in position_manifest.items()
                                     if k not in ('summaries',)},
                          measurement=store.measure()), indent=2))
    if store.mask_violations > 0.01 * max(len(store), 1):
        print('warning: many evaluations put probability outside the legal mask, so the '
              'pool is being stored densely. Pass --mask-tolerance to store sparsely.',
              file=sys.stderr)


# -------------------------------------------------------------------- measure

def command_measure(args):
    store, pool_manifest = EvalStore.load(args.pool, verify=not args.no_verify)
    positions, position_manifest = load_positions(args.pool, verify=not args.no_verify)
    measurement = store.measure()
    per_position = measurement['unique_states'] / max(len(positions), 1)
    print(json.dumps(dict(
        pool={k: v for k, v in pool_manifest.items() if k != 'shards'},
        positions={k: v for k, v in position_manifest.items() if k != 'summaries'},
        measurement=measurement,
        projection=dict(
            unique_states_per_position=per_position,
            resident_bytes_per_position=measurement['bytes_per_state'] * per_position,
            note='multiply by the positions in a full iteration (numEps x plies) to '
                 'project a whole-iteration pool',
        ),
    ), indent=2))


# ---------------------------------------------------------------------- check

def command_check(args):
    game = load_game(args.game)
    store, _ = EvalStore.load(args.pool, verify=not args.no_verify)
    positions, _ = load_positions(args.pool, verify=not args.no_verify)
    base = store.metadata.get('reference_args', {})
    if store.metadata.get('tree_reuse'):
        print('note: this pool was recorded with tree reuse, so a fresh-tree replay is '
              'not expected to reproduce the recorded decisions exactly.', file=sys.stderr)
    result = self_check(game, store, positions, base)
    print(json.dumps(result, indent=2))
    return 0 if result['exact'] or store.metadata.get('tree_reuse') else 1


# ---------------------------------------------------------------------- label

def command_label(args):
    positions, _ = load_positions(args.pool, verify=not args.no_verify)
    settings = dict(engine=args.engine, depth=args.depth, seconds=args.seconds,
                    node_limit=args.node_limit, accept_depth=args.accept_depth)

    def progress(done, total, stats):
        if done % 25 == 0 or done == total:
            print(f'  labelled {done}/{total} (complete: {stats["labelled"]})', flush=True)

    labels, records, stats = labels_module.label_positions(
        positions, binary=args.binary, progress=progress, **settings)
    manifest = labels_module.save_labels(args.pool, labels, records, stats, settings)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'records'}, indent=2))


# ---------------------------------------------------------------------- bench

def command_bench(args):
    game, net, _keys = load_game_and_net(args.game, args.checkpoint, 'onnx')
    positions, _ = load_positions(args.pool, verify=not args.no_verify)
    sample = positions[:max(args.positions, 1)]
    boards = [p['state'] for p in sample]
    masks = [np.asarray(game.getValidMoves(p['state'], 0)).astype(bool) for p in sample]

    curve = batch_latency(net, boards, masks, sizes=tuple(args.sizes),
                          repeats=args.repeats, warmup=args.warmup)
    report = dict(checkpoint=os.path.abspath(args.checkpoint), game=args.game,
                  positions=len(sample), repeats=args.repeats, curve=curve)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
        print(f'wrote {args.out}')

    print(f'{"batch":>6} {"call ms":>10} {"per pos ms":>12} {"pos/s":>10} {"speedup":>9}')
    print('-' * 51)
    for row in curve:
        print(f'{row["batch"]:>6} {row["median_seconds"] * 1e3:>10.3f} '
              f'{row["per_position_seconds"] * 1e3:>12.4f} '
              f'{row["positions_per_second"]:>10.0f} {row["speedup_vs_batch1"]:>8.2f}x')
    print()
    best = max(curve, key=lambda r: r['speedup_vs_batch1'])
    print(f'best throughput at batch {best["batch"]}: {best["speedup_vs_batch1"]:.2f}x '
          f'the per-position cost of batch 1.')
    print('Batch width today comes from --parallel-inferences (one in-flight leaf per')
    print('concurrent game), not from any swept search setting.')


# ---------------------------------------------------------------------- sweep

def command_sweep(args):
    game = load_game(args.game)
    store, _ = EvalStore.load(args.pool, verify=not args.no_verify)
    positions, _ = load_positions(args.pool, verify=not args.no_verify)
    base = dict(store.metadata.get('reference_args', {}))

    configs = []
    if args.config_file:
        configs.extend(json.loads(Path(args.config_file).read_text()))
    configs.extend(expand_grid(parse_grid(args.grid)))
    if not configs:
        raise SystemExit('nothing to sweep: pass --grid or --config-file')

    references = None
    if args.reference == 'teacher':
        references, labels_manifest = labels_module.load_labels(args.pool,
                                                                verify=not args.no_verify)
        print(f'teacher reference: {len(references)} labelled positions '
              f'({labels_manifest["stats"]["coverage"]:.1%} coverage)')

    fallback = None
    if args.on_miss == 'delegate':
        _game, fallback, _keys = load_game_and_net(args.game, args.checkpoint, args.backend)

    def progress(done, total, overrides, report):
        row = report['all']
        print(f'  [{done}/{total}] {overrides} -> agreement {row["agreement"]:.3f}, '
              f'{row["evals_per_move"]:.0f} evals/move, miss {row["miss_rate"]:.1%}',
              flush=True)

    latency_curve = None
    if args.latency:
        latency_curve = json.loads(Path(args.latency).read_text())['curve']
        print(f'cost model: measured latency at collection {args.collection}')
        if args.collection > 1:
            print('warning: the wall-clock model prices the TIME of gathering several '
                  'leaves per\n         network call but not its COST in decision '
                  'quality, which needs a\n         virtual-loss search to measure. '
                  'Read ms/move, not V, at collection > 1.', file=sys.stderr)

    results = sweep(game, store, positions, configs, base=base, on_miss=args.on_miss,
                    fallback=fallback, references=references, beta1=args.beta1,
                    latency_curve=latency_curve, collection=args.collection,
                    progress=progress)

    report = dict(pool=str(args.pool), reference=args.reference, beta1=args.beta1,
                  on_miss=args.on_miss, latency=args.latency, collection=args.collection,
                  base=base, results=results)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
        print(f'wrote {args.out}')
    print()
    print_table(results)


def print_table(results):
    timed = bool(results and results[0]['report']['all'].get('seconds_per_move'))
    header = (f'{"config":<40} {"agree":>7} {"confirm":>8} {"TV":>7} {"evals":>7}'
              + (f' {"ms/move":>9}' if timed else '') + f' {"miss":>6} {"V":>8}')
    print(header)
    print('-' * len(header))
    for row in results:
        overall, confirm = row['report']['all'], row['report'].get('confirm')
        label = ', '.join(f'{k}={v}' for k, v in sorted(row['config'].items()))
        timing = f' {overall["seconds_per_move"] * 1e3:>9.2f}' if timed else ''
        print(f'{label[:40]:<40} {overall["agreement"]:>7.3f} '
              f'{(confirm["agreement"] if confirm else float("nan")):>8.3f} '
              f'{overall["policy_tv"]:>7.3f} {overall["evals_per_move"]:>7.0f}'
              f'{timing} {overall["miss_rate"]:>5.1%} {overall["value"]:>8.3f}')
    skipped = results[0]['report'].get('skipped_unlabelled', 0) if results else 0
    print()
    print('agree:   top-1 match with the chosen reference, over every scored position.')
    print('confirm: the same, over the held-out games only. A gap means the selection')
    print('         split was overfitted.')
    print('TV:      total-variation distance from the RECORDED visit distribution, whichever')
    print('         reference scores agreement.')
    if timed_note(results):
        print('ms/move: modelled wall clock from the measured latency curve. V normalises')
        print('         against the reference at the SAME collection size, so read ms/move')
        print('         for absolute regressions.')
    print('evals:   network evaluations per move. miss: share served from outside the pool,')
    print('         so a high value means a partial rerun, not a free evaluation.')
    cost = 'modelled wall clock' if results and results[0]['report']['all'].get(
        'seconds_per_move') else 'evals / reference evals'
    print(f'V:       agreement - beta1 x relative cost, priced in {cost}.')
    if skipped:
        print(f'{skipped} recorded positions had no reference label and were skipped.')


def timed_note(results):
    return bool(results and results[0]['report']['all'].get('seconds_per_move'))


def main(argv=None):
    parser = argparse.ArgumentParser(prog='replay', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--no-verify', action='store_true', help='skip shard checksums')
    sub = parser.add_subparsers(dest='command', required=True)

    rec = sub.add_parser('record', help='play games and record a replay pool')
    rec.add_argument('--game', default='intransitive')
    rec.add_argument('--checkpoint', required=True)
    rec.add_argument('--out', required=True)
    rec.add_argument('--games', type=int, default=8)
    rec.add_argument('--seed', type=int, default=0)
    rec.add_argument('--sims', type=int, default=None, help='override numMCTSSims')
    rec.add_argument('--cpuct', type=float, default=None)
    rec.add_argument('--fpu', type=float, default=None)
    rec.add_argument('--universes', type=int, default=None)
    rec.add_argument('--forced-playouts', action='store_true')
    rec.add_argument('--tree-reuse', action='store_true',
                     help='reuse the tree across moves; breaks the exact self-check')
    rec.add_argument('--max-plies', type=int, default=400)
    rec.add_argument('--temp-begin', type=float, default=1.0)
    rec.add_argument('--temp-end', type=float, default=0.1)
    rec.add_argument('--temp-halflife', type=float, default=10.0)
    rec.add_argument('--select-fraction', type=float, default=0.5)
    rec.add_argument('--split-seed', type=int, default=0)
    rec.add_argument('--shard-entries', type=int, default=50_000)
    rec.add_argument('--backend', choices=('onnx', 'cpu'), default=None)
    rec.set_defaults(func=command_record)

    mea = sub.add_parser('measure', help='report pool size and a whole-iteration projection')
    mea.add_argument('--pool', required=True)
    mea.set_defaults(func=command_measure)

    chk = sub.add_parser('check', help='replay the recording configuration; must be exact')
    chk.add_argument('--pool', required=True)
    chk.add_argument('--game', default='intransitive')
    chk.set_defaults(func=command_check)

    lab = sub.add_parser('label', help='label recorded roots with the native minimax teacher')
    lab.add_argument('--pool', required=True)
    lab.add_argument('--engine', choices=('python', 'rust'), default='python')
    lab.add_argument('--depth', type=int, default=6)
    lab.add_argument('--seconds', type=float, default=5.0)
    lab.add_argument('--node-limit', type=int, default=200_000)
    lab.add_argument('--accept-depth', type=int, default=None,
                     help='keep a partial result that reached at least this depth '
                          '(default: require the full target depth)')
    lab.add_argument('--binary', default=None, help='rust engine only')
    lab.set_defaults(func=command_label)

    bch = sub.add_parser('bench', help='measure ONNX latency against inference batch size')
    bch.add_argument('--pool', required=True, help='source of realistic input positions')
    bch.add_argument('--checkpoint', required=True)
    bch.add_argument('--game', default='intransitive')
    bch.add_argument('--sizes', type=int, nargs='*', default=list(DEFAULT_SIZES))
    bch.add_argument('--positions', type=int, default=32)
    bch.add_argument('--repeats', type=int, default=25)
    bch.add_argument('--warmup', type=int, default=5)
    bch.add_argument('--out', default=None)
    bch.set_defaults(func=command_bench)

    swp = sub.add_parser('sweep', help='score search configurations against the pool')
    swp.add_argument('--pool', required=True)
    swp.add_argument('--game', default='intransitive')
    swp.add_argument('--grid', nargs='*', default=[], metavar='NAME=V,V')
    swp.add_argument('--config-file', default=None, help='JSON list of override dicts')
    swp.add_argument('--reference', choices=('recorded', 'teacher'), default='recorded')
    swp.add_argument('--on-miss', choices=('uniform', 'strict', 'delegate'), default='uniform')
    swp.add_argument('--checkpoint', default=None, help="required for --on-miss delegate")
    swp.add_argument('--backend', choices=('onnx', 'cpu'), default=None)
    swp.add_argument('--beta1', type=float, default=0.0,
                     help='cost weight in V = agreement - beta1 x relative cost')
    swp.add_argument('--latency', default=None, metavar='BENCH.JSON',
                     help='price cost in measured wall clock instead of evaluation count')
    swp.add_argument('--collection', type=int, default=1,
                     help='leaves gathered per network call, for the wall-clock model')
    swp.add_argument('--out', default=None)
    swp.set_defaults(func=command_sweep)

    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == '__main__':
    sys.exit(main())
