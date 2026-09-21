"""Audit an archived acceptance run without starting a single engine.

Replays every recorded game through the rules engine, rebuilds the frozen plan
from its own record, and re-applies the predeclared decision rule to the stored
evidence. A decision that no longer follows from its evidence is an error, not
a note in a report.

    uv run --locked python -m intransitive.benchmarks.validation.verify RUN [RUN ...]
"""
import argparse
import json
from pathlib import Path
import sys

from ...validation import report as report_module
from ...validation.runner import verify
from ...validation.spec import validate


def audit(path):
    path = Path(path)
    plan = json.loads((path / 'plan.json').read_text())
    validate(plan)
    result = verify(plan, path)
    evidence = json.loads((path / 'evidence.json').read_text())
    recorded = json.loads((path / 'decision.json').read_text())
    decision = report_module.verdict(plan, evidence)
    decision['defects'] = report_module.defects(plan, evidence)
    if decision != recorded:
        raise ValueError(f'{path}: the recorded decision does not follow from its evidence')
    return dict(run=str(path), plan_sha256=plan['sha256'], revision=plan['revision'],
                verified_games=result['verified_games'], experiments=result['experiments'],
                outcomes=recorded['outcomes'],
                defects=[row['kind'] for row in recorded['defects']],
                decision_reproduced=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args(argv)
    rows = [audit(run) for run in args.runs]
    json.dump(rows, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == '__main__':
    main()
