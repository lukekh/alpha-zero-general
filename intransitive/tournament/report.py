"""Conservative official win bounds; paired/line-clustered uncertainty."""
from collections import Counter, defaultdict
import math

import numpy as np

from .runner import FINAL


def report(spec, rows):
    by_id = {r['task']['id']: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) - {t['id'] for t in spec['tasks']}:
        raise ValueError('Duplicate or unknown match record')
    entries = []
    for task in spec['tasks']:
        row = by_id.get(task['id'])
        for colour, identity in enumerate(task['colours']):
            status = row['status'] if row else 'pending'
            entries.append(dict(candidate=identity, opponent=task['colours'][1-colour], colour=colour,
                                mode=spec['protocols'][task['protocol']]['mode'],
                                seed=task['seed'], pair=task['pair_id'], position=task['position'],
                                status=status, win=int(status == 'win' and row['winner'] == colour),
                                loss=int(status == 'win' and row['winner'] != colour),
                                row=row))

    def summarize(items, threshold):
        n = len(items)
        wins, losses = sum(i['win'] for i in items), sum(i['loss'] for i in items)
        statuses = Counter(i['status'] for i in items)
        unfinished = statuses['unfinished']
        failures = sum(count for status, count in statuses.items() if status in FINAL - {'win', 'unfinished'})
        pending = n - wins - losses - unfinished - failures
        moves, startups = [], []
        for item in items:
            if item['row']:
                moves.extend(m for m in item['row']['moves'] if m['side'] == item['colour'])
                if item['row'].get('failed_response') and item['row'].get('responsible_colour') == item['colour']:
                    moves.append(item['row']['failed_response'])
                startups.extend(s for s in item['row']['startups'] if s['colour'] == item['colour'])
        depths = Counter(str(m['result']['completed_depth']) for m in moves)
        latencies = [m['latency_seconds'] for m in moves]
        # Treat a colour pair as one observation. Correlated positions generated
        # along one line remain a single seed cluster, even across opponents.
        clusters = defaultdict(list)
        pairs = defaultdict(list)
        for item in items:
            clusters[item['seed']].append(item)
            pairs[item['pair']].append(item)
        lower, upper = wins / n, (n - losses) / n
        # Weighted Hoeffding radius: unequal numbers of positions per line do
        # not falsely buy more independent observations. This is conditional
        # on sampled opening lines, not a claim of population-wide calibration.
        radius = math.sqrt(math.log(40) / 2 * sum((len(group) / n)**2 for group in clusters.values()))
        completion = (wins + losses) / n
        return dict(scheduled=n, wins=wins, losses=losses, unfinished=unfinished,
                    failures=failures, pending=pending, statuses=dict(statuses),
                    completion_rate=completion, win_points_lower=lower, win_points_upper=upper,
                    eligible=pending == 0 and failures == 0 and completion >= threshold,
                    paired_interval_95=[max(0., lower-radius), min(1., upper+radius)],
                    seed_clusters=len(clusters), pairs=len(pairs),
                    completed_pairs=sum(len(group) == 2 and all(i['status'] == 'win' for i in group)
                                        for group in pairs.values()),
                    distinct_starts=len({i['position'] for i in items}),
                    distinct_trajectories=len({i['row']['trajectory_sha256'] for i in items if i['row']}),
                    completed_depths=dict(depths),
                    stopped_searches=sum(m['result']['stopped'] for m in moves),
                    stop_reasons=dict(Counter(m['result']['stop_reason'] for m in moves)),
                    selected_depths=dict(Counter(str(m['result']['selected_depth']) for m in moves)),
                    latency_p50_seconds=float(np.percentile(latencies, 50)) if latencies else None,
                    latency_p95_seconds=float(np.percentile(latencies, 95)) if latencies else None,
                    search_wall_seconds=sum(latencies),
                    search_cpu_seconds=sum(m['cpu_seconds'] for m in moves),
                    work=sum(m['result']['work'] for m in moves),
                    startup_wall_seconds=sum(s['parent_seconds'] for s in startups),
                    startup_cpu_seconds=sum(s['cpu_seconds'] for s in startups),
                    peak_candidate_rss_bytes=max([m['peak_rss_bytes'] for m in moves + startups], default=0))

    boards = {}
    for limits in spec['protocols']:
        board = []
        for candidate in spec['candidates']:
            items = [i for i in entries if i['candidate'] == candidate['sha256'] and i['mode'] == limits['mode']]
            if not items:
                continue
            summary = summarize(items, limits['completion_required'])
            summary.update(name=candidate['name'], candidate=candidate['sha256'], role=candidate['role'],
                           by_colour={str(c): summarize([i for i in items if i['colour'] == c], limits['completion_required']) for c in (0, 1)},
                           by_opponent={opponent: dict(summarize([i for i in items if i['opponent'] == opponent], limits['completion_required']),
                                                       by_colour={str(c): summarize([i for i in items if i['opponent'] == opponent and i['colour'] == c], limits['completion_required']) for c in (0, 1)})
                                        for opponent in sorted({i['opponent'] for i in items})})
            board.append(summary)
        boards[limits['mode']] = sorted(board, key=lambda r: (not r['eligible'], -r['win_points_lower'], r['candidate']))
    return dict(schema=spec['schema'], manifest_sha256=spec['sha256'], leaderboards=boards,
                protocols=spec['protocols'], scheduled_matches=len(spec['tasks']),
                final_matches=sum(r['status'] in FINAL for r in rows),
                distinct_starts=len(spec['selected_positions']),
                distinct_trajectories=len({r['trajectory_sha256'] for r in rows}),
                duplicate_trajectories=len(rows)-len({r['trajectory_sha256'] for r in rows}),
                active_game_seconds=sum(r['active_seconds'] for r in rows),
                startup_wait_seconds=sum(r['startup_wait_seconds'] for r in rows),
                known_child_cpu_seconds=sum(m['cpu_seconds'] for r in rows for m in r['moves'])
                    + sum(s['cpu_seconds'] for r in rows for s in r['startups'])
                    + sum(r.get('failed_response', {}).get('cpu_seconds', 0.) for r in rows),
                child_peak_rss_bytes=max([m['peak_rss_bytes'] for r in rows for m in r['moves'] + r['startups']], default=0),
                uncertainty='95% weighted Hoeffding bounds on colour-paired outcomes clustered by '
                            'opening-generation seed. Unresolved outcomes span [0,1]. Conditional '
                            'on the frozen corpus; repeated deterministic trajectories are not independent evidence.',
                compute_note='CPU/search/startup totals exclude work lost to hard-killed children; '
                             'parent-observed active and invocation wall time include that waiting. '
                             'RSS is process high-water memory, not simultaneous aggregate RSS.')
