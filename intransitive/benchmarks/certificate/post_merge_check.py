"""Do the recorded fixed-depth rows still reproduce after merging master?

Issue #70's measurements were taken before #71's static exchange evaluation
landed. Every option #71 adds defaults to off, so the engine this benchmark
configures should behave identically; this checks that claim against the
recorded numbers instead of asserting it.

A row that hit the eight-second cap is wall-clock dependent and is expected to
differ. Only the rows that completed are reproducible, and those must match
exactly.

    python -m intransitive.benchmarks.certificate.post_merge_check
"""
import json
from pathlib import Path

from intransitive.IntransitiveGame import IntransitiveGame
from .reproduce import PAIRS, fixtures, search

HERE = Path(__file__).resolve().parent


def main():
    report = json.loads((HERE / 'evidence/report-predeclared.json').read_text())
    recorded = {(row['stage'], row['depth'], row['arm']): row
                for row in report['fixed']
                if row['option'] == 'leaf' and row['symmetry'] == 0}
    game = IntransitiveGame()
    stages = dict(fixtures(game))
    candidate, reference = PAIRS['leaf']
    for stage, depth in (('certified', 5), ('sparse', 3), ('sparse', 6), ('opening', 3)):
        for arm, settings in (('off', reference), ('on', candidate)):
            was = recorded[stage, depth, arm]
            now = search(game, stages[stage], settings, depth=depth, seconds=8.)
            capped = was['stopped'] or now['stopped']
            agrees = was['nodes'] == now['nodes'] and was['score'] == now['score']
            print(f"{stage:10s} d={depth} {arm:3s} recorded {was['nodes']:>7,d} nodes "
                  f"now {now['nodes']:>7,d} "
                  f"{'CAPPED, not reproducible' if capped else 'MATCH' if agrees else 'DIFFERS'}")
            if not capped and not agrees:
                raise SystemExit('a completed row changed after the merge')


if __name__ == '__main__':
    main()
