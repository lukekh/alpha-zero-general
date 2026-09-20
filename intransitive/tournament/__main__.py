"""Prepare, run/resume, and verify bounded tournaments."""
import argparse
import json
from pathlib import Path
import platform
import sys

from .runner import atomic_json, replay, run, validate_manifest
from .spec import candidate, digest, effective_config, generate_positions, manifest, protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--candidates', type=Path, help='JSON list of {name, weights, role, backend}')
    prepare.add_argument('--seed', type=int, default=54)
    prepare.add_argument('--lines', type=int, default=9)
    prepare.add_argument('--pool', choices=('search', 'validation', 'heldout'), default='search')
    prepare.add_argument('--positions', type=int, default=2, help='Exact distinct-position quota')
    prepare.add_argument('--depth', type=int, default=1)
    prepare.add_argument('--depth-seconds', type=float, default=10.)
    prepare.add_argument('--seconds', type=float, default=.05)
    prepare.add_argument('--max-plies', type=int, default=8)
    prepare.add_argument('--game-seconds', type=float, default=30.)
    prepare.add_argument('--selective', action='store_true',
                         help='Declare NMP, futility, LMR and MVV-LVA, together with the PVS windows, '
                              'the experimental evaluator opt-in and the depth they need to fire')
    prepare.add_argument('--futility-margin', type=float, default=1.,
                         help='Allowance multiplier in 1/16..16; the default prunes nothing at evolved scales')
    prepare.add_argument('--variant', action='store_true',
                         help='Schedule the selective settings as a per-candidate search variant '
                              'against the same genome with them off, instead of giving them to '
                              'every entrant through the protocol')
    prepare.add_argument('--probe', action='store_true',
                         help='Search each selected position once per protocol and report whether the '
                              'declared techniques can reach the depth they need under these limits')
    for name in ('run', 'verify'):
        command = commands.add_parser(name)
        command.add_argument('--manifest', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        if name == 'run':
            command.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        inputs = json.loads(args.candidates.read_text()) if args.candidates else [
            dict(name='material-50', weights={'advantage_weight': 50.}),
            dict(name='default', role='incumbent'),
            dict(name='pressure-9x9-archive-225dc40', weights={'pressure_weight': 10.}, role='archive')]
        candidates = [candidate(**item) for item in inputs]
        positions = generate_positions(args.seed, args.lines)
        # The techniques come as a set because they share preconditions: without
        # PVS neither NMP nor futility ever sees a null window, and without the
        # experimental opt-in evolved candidate scales refuse both outright.
        selective = dict(nmp_enabled=True, futility_enabled=True, lmr_enabled=True,
                         mvv_lva_enabled=True, pvs_enabled=True,
                         selective_evaluator_enabled=True) if args.selective else {}
        if args.variant:
            # The A/B the protocol cannot express: one genome, two search
            # policies, every other setting held equal by the shared protocol.
            if not selective:
                raise ValueError('--variant needs --selective to have a variant to schedule')
            candidates.append(candidate('selective-variant', role='population',
                                        search=dict(selective, futility_margin=args.futility_margin)))
            selective = {}
        limits = dict(depth=args.depth, max_plies=args.max_plies, game_seconds=args.game_seconds,
                      **({} if args.variant else dict(futility_margin=args.futility_margin)),
                      **selective)
        spec = manifest(candidates, positions, [protocol('depth', seconds=args.depth_seconds, **limits),
                                               protocol('wall', seconds=args.seconds, **limits)],
                        pool=args.pool, position_limit=args.positions)
        if args.output.exists():
            raise ValueError('Manifest already exists; use it unchanged or choose a new path')
        atomic_json(args.output, spec)
        summary = dict(manifest=str(args.output), matches=len(spec['tasks']),
                       corpus_positions=len(positions), stages=sorted({p['stage'] for p in positions}))
        if args.probe:
            from .preflight import reachability
            from .spec import unpack
            selected = [unpack(p['state']) for p in spec['positions']
                        if p['sha256'] in spec['selected_positions']]
            # One probe per distinct search policy: the depth a search reaches
            # is a property of the limits, but which techniques want that depth
            # is a property of the candidate, and a variant carries its own.
            summary['reachability'] = {}
            for limits in spec['protocols']:
                seen, rows = set(), {}
                for item in spec['candidates']:
                    policy = digest(item.get('search', {}))
                    if policy in seen:
                        continue
                    seen.add(policy)
                    rows[item['name']] = reachability(effective_config(item, limits), selected)
                summary['reachability'][limits['mode']] = rows
            summary['warnings'] = [f'{mode} ({name}): {text}'
                                   for mode, rows in summary['reachability'].items()
                                   for name, row in rows.items() for text in row['warnings']]
        print(json.dumps(summary))
    else:
        spec = json.loads(args.manifest.read_text())
        if args.command == 'run':
            result = run(spec, args.output, workers=args.workers)
            atomic_json(args.output / 'environment.json', dict(python=sys.version, platform=platform.platform()))
            print(json.dumps(dict(report=str(args.output / 'report.json'), final=result['final_matches'],
                                  scheduled=result['scheduled_matches'])))
        else:
            validate_manifest(spec)
            starts = {p['sha256']: p for p in spec['positions']}
            checked = 0
            for task in spec['tasks']:
                path = args.output / 'matches' / (task['id'] + '.json')
                if path.exists():
                    row = json.loads(path.read_text())
                    if row['task'] != task or row['manifest_sha256'] != spec['sha256'] or row['colours'] != task['colours']:
                        raise ValueError('Mismatched record identity')
                    replay(starts[task['position']], row)
                    checked += 1
            print(json.dumps(dict(verified_matches=checked, scheduled_matches=len(spec['tasks']))))


if __name__ == '__main__':
    main()
