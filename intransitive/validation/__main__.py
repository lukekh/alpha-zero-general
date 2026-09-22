"""Prepare, run/resume, verify and decide a held-out acceptance experiment."""
import argparse
import json
from pathlib import Path

from ..tournament.runner import atomic_json
from ..tournament.spec import generate_positions
from . import report as report_module
from .preset import export
from .runner import STAGES, run, verify
from .spec import Design, Thresholds, configs, plan, validate

SMOKE = dict(starts_per_stage=1, ablation_starts_per_stage=1,
             ablation_stages=('endgame',), fixed_depth=1, depth_seconds=10.,
             wall_seconds=.05, max_plies=2, game_seconds=30., cost_depths=(1, 2),
             cost_seconds=15., cost_positions=1, equal_time_budgets=(.05,),
             teacher_depth=1, teacher_positions=1, tactical_cases=4)


def build(args, **overrides):
    design = Design(**{**dict(corpus_seed=args.seed, corpus_lines=args.lines), **overrides})
    candidates = json.loads(args.candidates.read_text()) if args.candidates else None
    corpus = generate_positions(design.corpus_seed, design.corpus_lines)
    return plan(corpus, candidates=candidates, design=design,
                thresholds=Thresholds(), revision=args.revision)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'smoke'):
        command = commands.add_parser(name)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--revision', required=True)
        command.add_argument('--candidates', type=Path,
                             help='JSON list of {name, genes, variable, baseline, source, '
                                  'source_sha256, source_version, note}')
        command.add_argument('--seed', type=int, default=Design().corpus_seed)
        command.add_argument('--lines', type=int, default=Design().corpus_lines)
        if name == 'smoke':
            command.add_argument('--workers', type=int, default=1)
    for name in ('run', 'verify', 'decide', 'preset'):
        command = commands.add_parser(name)
        command.add_argument('--plan', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        if name == 'run':
            command.add_argument('--workers', type=int, default=1)
            command.add_argument('--max-seconds', type=float, default=None)
            command.add_argument('--skip-native', action='store_true')
            command.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
        if name == 'preset':
            command.add_argument('--candidate', required=True)
            command.add_argument('--preset-output', type=Path, required=True)
            command.add_argument('--depth', type=int, default=None)
    args = parser.parse_args(argv)
    if args.command in ('prepare', 'smoke'):
        spec = build(args, **(SMOKE if args.command == 'smoke' else {}))
        if args.command == 'prepare':
            if args.output.exists():
                raise ValueError('Plan already exists; use it unchanged or choose a new path')
            atomic_json(args.output, spec)
            print(json.dumps(dict(plan=str(args.output), sha256=spec['sha256'],
                                  experiments={e['name']: e['games'] for e in spec['experiments']},
                                  configurations=sorted(configs(spec)),
                                  detectable={e['name']: e['power'] for e in spec['experiments']},
                                  corpus=len(spec['corpus']), held_out_starts=len(spec['starts']))))
            return
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output / 'plan.json', spec)
        result = run(spec, args.output, workers=args.workers, skip_native=True)
        print(json.dumps(dict(status=result['summary']['status'],
                              scheduled=result['summary']['total_scheduled'],
                              final=result['summary']['total_final'],
                              outcomes=result['decision']['outcomes'])))
        return
    spec = json.loads(args.plan.read_text())
    validate(spec)
    if args.command == 'run':
        result = run(spec, args.output, workers=args.workers, max_seconds=args.max_seconds,
                     skip_native=args.skip_native, stages=tuple(args.stages))
        print(json.dumps(dict(status=result['summary']['status'],
                              scheduled=result['summary']['total_scheduled'],
                              final=result['summary']['total_final'],
                              elapsed_seconds=result['summary']['elapsed_seconds'],
                              outcomes=result['decision']['outcomes'],
                              defects=[row['kind'] for row in result['decision']['defects']])))
    elif args.command == 'verify':
        print(json.dumps(verify(spec, args.output)))
    elif args.command == 'decide':
        evidence = json.loads((args.output / 'evidence.json').read_text())
        decision = report_module.verdict(spec, evidence)
        decision['defects'] = report_module.defects(spec, evidence)
        atomic_json(args.output / 'decision.json', decision)
        print(json.dumps(dict(outcomes=decision['outcomes'],
                              defects=[row['kind'] for row in decision['defects']])))
    else:
        evidence = json.loads((args.output / 'evidence.json').read_text())
        decision = json.loads((args.output / 'decision.json').read_text())
        row = configs(spec)[args.candidate]
        record = export(args.candidate, row, plan=spec, decision=decision,
                        profile=evidence['cost'].get(args.candidate), depth=args.depth)
        atomic_json(args.preset_output, record['preset'])
        atomic_json(args.preset_output.with_suffix('.record.json'), record)
        print(json.dumps(dict(preset=str(args.preset_output), outcome=record['acceptance']['outcome'],
                              limits=record['measured_limits'])))


if __name__ == '__main__':
    main()
