"""Replay and audit saved evolution runs without launching any search engines."""
import argparse
import json
from pathlib import Path

from ...evolution.runner import validate
from ...tournament.runner import FINAL, replay, validate_manifest
from ...tournament.spec import digest


def verify(output):
    output = Path(output)
    spec = json.loads((output / 'manifest.json').read_text())
    checkpoint = json.loads((output / 'checkpoint.json').read_text())
    summary = json.loads((output / 'report.json').read_text())
    validate(spec)
    if checkpoint['manifest'] != spec['sha256'] or summary['manifest'] != spec['sha256']:
        raise ValueError('Run/checkpoint/report identity mismatch')
    if summary['convergence'] != checkpoint['history']:
        raise ValueError('Checkpoint/report fitness mismatch')
    if json.loads((output / 'candidates.json').read_text()) != checkpoint['exports']:
        raise ValueError('Checkpoint/export mismatch')
    trajectories, moves, statuses = set(), 0, {}
    for key, entry in checkpoint['ledger'].items():
        path = output / entry['record']
        row = json.loads(path.read_text())
        match_spec = json.loads((path.parent / 'manifest.json').read_text())
        validate_manifest(match_spec)
        expected = digest(dict(manifest=match_spec['sha256'], task=row['task']['id']))
        if (expected != key or digest(row) != entry['record_sha256']
                or row['manifest_sha256'] != match_spec['sha256']
                or row['task'] not in match_spec['tasks'] or row['colours'] != row['task']['colours']
                or entry['final'] != (row['status'] in FINAL) or entry['status'] != row['status']
                or entry['trajectory'] != row['trajectory_sha256']):
            raise ValueError('Match ledger/journal identity mismatch')
        replay(match_spec['positions'][0], row)
        trajectories.add(row['trajectory_sha256'])
        moves += len(row['moves'])
        statuses[row['status']] = statuses.get(row['status'], 0) + 1
    if (summary['unique_matches'] != len(checkpoint['ledger'])
            or summary['unique_trajectories'] != len(trajectories)
            or {s: n for s, n in summary['outcomes'].items() if n} != statuses):
        raise ValueError('Evidence totals disagree with replayed journals')
    for count, limit in (('games_reserved', 'max_games'), ('nodes_reserved', 'max_nodes')):
        if summary[count] != checkpoint[count] or summary[count] > spec['settings'][limit]:
            raise ValueError('Invalid budget accounting')
    return dict(run=str(output), verified_matches=len(checkpoint['ledger']), verified_moves=moves,
                distinct_trajectories=len(trajectories), status=summary['status'], outcomes=statuses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args()
    print(json.dumps([verify(path) for path in args.runs], indent=2))


if __name__ == '__main__':
    main()
