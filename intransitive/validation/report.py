"""Apply the predeclared decision rule to the measured evidence.

Nothing here chooses a threshold: the plan froze them before any game ran.
This module reads the harness's conservative bounds, the tactical comparison,
the attribution matches and the cost profile, and states which predeclared
outcome the evidence supports. 'Retain existing defaults' is the outcome when
nothing else is established, and it is not a failure of the run.
"""
from . import cost as cost_module
from . import tactics as tactics_module

OUTCOMES = ('adopt', 'retain-defaults', 'revert-recommended', 'inconclusive')


def board(report, mode):
    """One protocol's leaderboard, keyed by entrant name."""
    return {row['name']: row for row in report['leaderboards'][mode]}


def head_to_head(candidate, baseline):
    """Only the games these two played against each other.

    The leaderboard margin also contains each entrant's games against the
    shared references, which is the right basis for a ranking and the wrong
    one for 'did this configuration beat that one'. The direct record is
    colour-paired on the same starts and is what a regression is judged on.
    """
    row = candidate.get('by_opponent', {}).get(baseline.get('candidate'))
    if not row:
        return None
    return dict(scheduled=row['scheduled'], wins=row['wins'], losses=row['losses'],
                unfinished=row['unfinished'], failures=row['failures'],
                completion_rate=row['completion_rate'],
                win_points_lower=row['win_points_lower'],
                loss_points_lower=row['losses'] / row['scheduled'] if row['scheduled'] else 0.,
                net=(row['wins'] - row['losses']) / row['scheduled'] if row['scheduled'] else 0.,
                by_colour={colour: dict(wins=item['wins'], losses=item['losses'],
                                        unfinished=item['unfinished'], scheduled=item['scheduled'])
                           for colour, item in sorted(row.get('by_colour', {}).items())})


def margin(candidate, baseline):
    """Paired difference of the conservative lower win-point bound.

    Both entrants played the same starts, both colours and the same opponent
    slate inside one experiment, so the difference is paired by construction.
    Disjoint intervals are what separation means here; the point difference
    alone is an observation.
    """
    low, high = candidate['paired_interval_95']
    other_low, other_high = baseline['paired_interval_95']
    return dict(candidate=candidate['name'], baseline=baseline['name'],
                head_to_head=head_to_head(candidate, baseline),
                candidate_lower=candidate['win_points_lower'],
                baseline_lower=baseline['win_points_lower'],
                difference=candidate['win_points_lower'] - baseline['win_points_lower'],
                candidate_interval=[low, high], baseline_interval=[other_low, other_high],
                disjoint=low > other_high or other_low > high,
                candidate_eligible=candidate['eligible'], baseline_eligible=baseline['eligible'],
                candidate_record=dict(wins=candidate['wins'], losses=candidate['losses'],
                                      unfinished=candidate['unfinished'],
                                      failures=candidate['failures'],
                                      scheduled=candidate['scheduled'],
                                      completion_rate=candidate['completion_rate'],
                                      distinct_trajectories=candidate['distinct_trajectories']),
                baseline_record=dict(wins=baseline['wins'], losses=baseline['losses'],
                                     unfinished=baseline['unfinished'],
                                     failures=baseline['failures'],
                                     scheduled=baseline['scheduled'],
                                     completion_rate=baseline['completion_rate'],
                                     distinct_trajectories=baseline['distinct_trajectories']))


def strength(plan, results, name, baseline_name):
    """Every strength protocol this candidate and its comparison both played."""
    rows = {}
    for item in plan['experiments']:
        if not item['name'].startswith('strength-'):
            continue
        report = results['experiments'].get(item['name'], {}).get('report')
        if report is None:
            continue
        mode = item['manifest']['protocols'][0]['mode']
        table = board(report, mode)
        if name in table and baseline_name in table:
            rows[item['name']] = dict(mode=mode, **margin(table[name], table[baseline_name]))
    return rows


def attribution(plan, results, name, thresholds):
    """Did restoring one changed module take the candidate's result with it?

    The candidate is the incumbent inside its own ablation experiment, so each
    restore entrant is measured against it on identical starts. A restore that
    does at least as well says the module is not what produced the difference.
    """
    prefix = f'ablation-{name}-'
    items = [e for e in plan['experiments'] if e['name'].startswith(prefix)]
    if not items:
        return dict(status='not-scheduled',
                    reason='No attribution experiment is scheduled for this candidate')
    rows, mode = {}, None
    for item in items:
        gene = item['name'][len(prefix):]
        report = results['experiments'].get(item['name'], {}).get('report')
        if report is None:
            rows[gene] = dict(status='pending')
            continue
        mode = item['manifest']['protocols'][0]['mode']
        table = board(report, mode)
        restored = f'{name}-restore-{gene}'
        if name not in table or restored not in table:
            rows[gene] = dict(status='pending')
            continue
        row = margin(table[restored], table[name])
        # The module 'carries' the candidate when removing it costs more than
        # the same margin a default change would have to clear.
        row['module_carries_result'] = row['difference'] <= -thresholds['practical_gain']
        rows[gene] = row
    if not any(row.get('module_carries_result') is not None for row in rows.values()):
        return dict(status='pending', by_gene=rows,
                    reason='No attribution experiment has a completed report yet')
    return dict(status='measured', mode=mode, by_gene=rows,
                attributed=sorted(gene for gene, row in rows.items()
                                  if row.get('module_carries_result')),
                unattributed=sorted(gene for gene, row in rows.items()
                                    if row.get('module_carries_result') is False))


def tactical(results, name, baseline_name):
    rows = {}
    for profile in ('certified', 'shipped'):
        candidate = results['tactics'].get(name, {}).get(profile)
        baseline = results['tactics'].get(baseline_name, {}).get(profile)
        if candidate and baseline:
            rows[profile] = tactics_module.compare(candidate, baseline)
    return rows


def practical(results, name, baseline_name, thresholds):
    candidate = results['cost'].get(name)
    baseline = results['cost'].get(baseline_name)
    if not candidate or not baseline:
        return dict(status='pending')
    return dict(status='measured', **cost_module.compare(candidate, baseline, thresholds))


def verdict(plan, results):
    """The predeclared outcome for every candidate, with the reasons for it."""
    thresholds = plan['thresholds']
    primary = 'strength-depth'
    rows = {}
    for item in plan['candidates']:
        name, baseline_name = item['name'], item['baseline']
        strengths = strength(plan, results, name, baseline_name)
        variable = next((e['name'] for e in plan['experiments']
                         if e['name'].startswith('strength-depth-') and e['name'].endswith(name)), None)
        head = strengths.get(variable or primary)
        tact = tactical(results, name, baseline_name)
        cost = practical(results, name, baseline_name, thresholds)
        parity = results['parity'].get(name, {})
        reasons, blocking = [], []
        if head is None:
            blocking.append('The primary fixed-depth comparison has no completed report')
        if not parity.get('passed', False):
            blocking.append('Correctness/parity checks did not pass for this candidate')
        for key, row in sorted(strengths.items()):
            if not (row['candidate_eligible'] and row['baseline_eligible']):
                blocking.append(f'{key}: an entrant failed harness eligibility '
                                f'(completion, failures or unattempted games)')
        proven = [row for row in tact.values() if row['new_proven_failures']]
        if proven:
            blocking.append('New failures on fixtures whose certificate is a proof')
        new_failures = sum(len(row['new_failures']) for row in tact.values())
        if new_failures > thresholds['max_new_tactical_failures']:
            blocking.append(f'{new_failures} newly failed certified tactical fixtures')
        gain = head['difference'] if head else None
        separated = bool(head and head['disjoint'])
        established = bool(gain is not None and gain >= thresholds['practical_gain']
                           and (separated or not thresholds['require_disjoint_intervals']))
        # A regression is a safety finding, so it is judged on the point margin
        # and on the direct paired record, not on interval separation that the
        # affordable schedule cannot produce in either direction.
        direct = head['head_to_head'] if head else None
        regressed = bool(gain is not None and gain <= -thresholds['practical_regression']
                         and direct is not None
                         and direct['net'] <= -thresholds['practical_regression'])
        if cost.get('status') == 'measured':
            if not cost['depth_within_allowance']:
                blocking.append(f'Equal-time achieved depth falls '
                                f'{cost["worst_median_depth_loss"]} below the comparison')
            if not cost['shipped_reaches_default_depth']:
                reasons.append('Under the shipped default limits this evaluation does not '
                               'complete the shipped default depth')
        if established and not blocking:
            outcome = 'adopt'
        elif regressed and not blocking:
            outcome = ('revert-recommended' if item['current_default'] else 'retain-defaults')
        elif blocking or head is None:
            outcome = 'inconclusive'
        else:
            outcome = 'retain-defaults'
        if outcome == 'revert-recommended' and not item['current_default']:
            outcome = 'retain-defaults'
        rows[name] = dict(candidate=name, baseline=baseline_name, outcome=outcome,
                          is_current_default=item['current_default'],
                          primary=head, strength=strengths,
                          gain=gain, separated=separated, established=established,
                          regressed=regressed, blocking=blocking, notes=reasons,
                          attribution=attribution(plan, results, name, thresholds),
                          tactical=tact, cost=cost, parity_passed=parity.get('passed', False),
                          provenance=dict(source=item['source'], source_sha256=item['source_sha256'],
                                          source_version=item['source_version'], note=item['note']))
    return dict(schema=plan['schema'], plan_sha256=plan['sha256'], thresholds=thresholds,
                outcomes={name: row['outcome'] for name, row in rows.items()},
                candidates=rows, legal_outcomes=list(OUTCOMES),
                separation='Adoption requires disjoint paired intervals. The plan publishes the '
                           'cluster count that would be needed to separate the declared gain '
                           'under the harness\'s conservative bound; a schedule below it cannot '
                           'promote anything, and a null result there is a null result and not '
                           'evidence of equivalence.',
                rule='A candidate is adopted only when its paired held-out margin clears the '
                     'predeclared gain, its interval is separated from its comparison, every '
                     'protocol it played is eligible, correctness and tactical safety hold, and '
                     'the practical cost stays inside the declared allowance. Anything else '
                     'retains existing defaults or is inconclusive. A regression is judged on '
                     'the point margin together with the direct paired head-to-head record, '
                     'because a safety finding must not need the evidence a promotion needs. '
                     'No export changes a default, the generator or the trainer.')


def defects(plan, results):
    """Facts a reader must not miss, whatever the adoption outcome was.

    These are configuration findings rather than candidate verdicts: they are
    true of the shipped defaults themselves, so they belong outside the
    adopt/retain decision and in front of it.
    """
    rows = []
    for name, profile in sorted(results.get('cost', {}).items()):
        shipped = profile['shipped']
        if shipped['reached_requested_depth'] < shipped['positions']:
            rows.append(dict(kind='shipped-budget-starves-search', configuration=name,
                             limits=shipped['limits'],
                             median_completed_depth=shipped['median_completed_depth'],
                             reached_requested_depth=shipped['reached_requested_depth'],
                             positions=shipped['positions'],
                             detail=f'{name} cannot complete the shipped default depth '
                                    f'{shipped["limits"]["max_depth"]} inside the shipped default '
                                    f'work limit {shipped["limits"]["node_limit"]}; the search '
                                    f'stops at a median completed depth of '
                                    f'{shipped["median_completed_depth"]}.'))
    for name, profiles in sorted(results.get('tactics', {}).items()):
        shipped = profiles.get('shipped')
        if shipped and shipped['budget_stopped']:
            rows.append(dict(kind='tactical-fixtures-budget-stopped', configuration=name,
                             limits=shipped['limits'], stopped=len(shipped['budget_stopped']),
                             cases=shipped['cases'],
                             detail=f'{len(shipped["budget_stopped"])} of {shipped["cases"]} '
                                    f'certified tactical fixtures exhaust the shipped default '
                                    f'work budget for {name}.'))
    return rows
