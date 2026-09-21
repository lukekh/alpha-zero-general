"""Frozen acceptance plans, the predeclared decision rule, and replayable runs."""
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path
import tempfile
import unittest

from intransitive.heuristics.config import SearchConfig
from intransitive.heuristics.tuning import DEFAULTS, Genome
from intransitive.tests.test_tournament import FakeEngine
from intransitive.tournament.runner import play_match
from intransitive.tournament.spec import generate_positions
from intransitive.validation import preset as preset_module
from intransitive.validation import report as report_module
from intransitive.validation import runner as runner_module
from intransitive.validation import tactics as tactics_module
from intransitive.validation.spec import (Design, PRIOR_INCUMBENT, Thresholds,
                                          VARIABLE_BASELINE, ablation_baseline,
                                          ablation_genes, configs, normalize_candidate,
                                          plan, validate)

CANDIDATE = dict(name='candidate', genes={'advantage': 40., 'attack': 0., 'defence': 0.},
                 baseline='prior-incumbent', source='unit-test', source_version='test-v1',
                 source_sha256='0' * 64, note='A single changed gene keeps the schedule small.')
# The shipped configuration, which is the only entrant a run may recommend
# reverting. Decision tests use it so that path is actually exercised.
SHIPPED = dict(CANDIDATE, genes=dict(DEFAULTS), note='The configuration in force today.')


@lru_cache(maxsize=1)
def corpus():
    """One generated corpus for the whole module; replaying it is the slow part."""
    return generate_positions(54, 9)


def small_design(**overrides):
    return Design(**dict(dict(corpus_lines=9, starts_per_stage=1, strength_stages=('opening',),
                              ablation_stages=('opening',), fixed_depth=1, depth_seconds=10.,
                              wall_seconds=.05, max_plies=2, game_seconds=30., cost_depths=(1,),
                              cost_seconds=5., cost_positions=1, equal_time_budgets=(.05,),
                              teacher_depth=1, teacher_positions=1, tactical_cases=2),
                         **overrides))


def fake_match(spec, task, path, cancelled):
    return play_match(spec, task, path, cancelled, engine_factory=FakeEngine)


def entry(name, *, lower, radius=.1, eligible=True, scheduled=8, opponent=None):
    """A leaderboard row shaped like the harness's, with a direct record."""
    wins = round(lower * scheduled)
    losses = scheduled - wins
    row = dict(name=name, candidate=f'sha-{name}', win_points_lower=lower, eligible=eligible,
               wins=wins, losses=losses, unfinished=0, failures=0, scheduled=scheduled,
               completion_rate=(wins + losses) / scheduled, distinct_trajectories=scheduled,
               paired_interval_95=[max(0., lower - radius), min(1., lower + radius)])
    if opponent is not None:
        half = scheduled // 2
        direct_wins = round(lower * half)
        row['by_opponent'] = {f'sha-{opponent}': dict(
            scheduled=half, wins=direct_wins, losses=half - direct_wins, unfinished=0,
            failures=0, completion_rate=1., win_points_lower=direct_wins / half,
            by_colour={})}
    return row


class PlanTests(unittest.TestCase):
    def design(self, **overrides):
        return small_design(**overrides)

    def plan(self, candidates=None, **overrides):
        return plan(corpus(), candidates=candidates or [CANDIDATE],
                    design=self.design(**overrides), revision='test-revision')

    def test_plan_is_deterministic_and_rebuilds_from_its_own_record(self):
        first, second = self.plan(), self.plan()
        self.assertEqual(first, second)
        validate(first)
        for field, value in (('revision', 'other'), ('sha256', 'x' * 64)):
            broken = dict(first, **{field: value})
            with self.assertRaises(ValueError):
                validate(broken)
        drifted = deepcopy(first)
        drifted['thresholds']['practical_gain'] = .5
        with self.assertRaises(ValueError):
            validate(drifted)
        drifted = deepcopy(first)
        drifted['design']['max_plies'] = 4
        with self.assertRaises(ValueError):
            validate(drifted)
        drifted = deepcopy(first)
        drifted['corpus'] = drifted['corpus'][:-1]
        with self.assertRaises(ValueError):
            validate(drifted)

    def test_only_heldout_positions_are_scheduled_and_lines_are_distinct(self):
        spec = self.plan(starts_per_stage=1, strength_stages=('opening', 'midgame'))
        held = {p['sha256']: p for p in spec['corpus'] if p['pool'] == 'heldout'}
        others = {p['orbit_sha256'] for p in spec['corpus'] if p['pool'] != 'heldout'}
        self.assertTrue(spec['starts'])
        for item in spec['experiments']:
            for position in item['manifest']['positions']:
                self.assertEqual(position['pool'], 'heldout')
                self.assertIn(position['sha256'], held)
                self.assertNotIn(position['orbit_sha256'], others)
        seeds = [held[identity]['seed'] for identity in spec['starts']]
        self.assertEqual(len(seeds), len(set(seeds)))
        # Everything that is not a match reads the search pool instead.
        for position in runner_module.search_states(spec, 2):
            self.assertEqual(position['pool'], 'search')

    def test_only_the_shipped_configuration_is_marked_current(self):
        spec = self.plan()
        self.assertFalse(spec['candidates'][0]['current_default'])
        shipped = plan(corpus(), candidates=[dict(CANDIDATE, genes=dict(DEFAULTS))],
                       design=self.design(), revision='test-revision')
        self.assertTrue(shipped['candidates'][0]['current_default'])

    def test_a_variable_candidate_faces_the_configuration_it_would_displace(self):
        variable = dict(CANDIDATE, name='variable-candidate', variable=True,
                        genes=dict(DEFAULTS, material=7.), baseline='candidate', attribute=False)
        spec = self.plan([CANDIDATE, variable])
        item = next(e for e in spec['experiments']
                    if e['name'] == 'strength-depth-variable-candidate')
        names = {e['name']: e['role'] for e in item['entrants']}
        self.assertEqual(names.get('candidate'), 'incumbent')
        self.assertEqual(names.get('variable-initial-material-baseline'), 'archive')
        self.assertEqual(sorted(spec['cost_targets']), ['candidate', 'prior-incumbent',
                                                        'variable-candidate'])

    def test_entrants_ablations_and_baselines(self):
        spec = self.plan()
        names = sorted(configs(spec))
        self.assertIn('candidate', names)
        self.assertIn('prior-incumbent', names)
        self.assertIn('candidate-restore-advantage', names)
        # One changed gene means exactly one attribution experiment.
        ablations = [e for e in spec['experiments'] if e['name'].startswith('ablation-')]
        self.assertEqual([e['name'] for e in ablations], ['ablation-candidate-advantage'])
        restored = Genome.from_json(json.dumps(
            configs(spec)['candidate-restore-advantage']['genome'])).to_dict()['genes']
        original = Genome.from_json(json.dumps(configs(spec)['candidate']['genome'])).to_dict()['genes']
        self.assertEqual(restored['advantage'], PRIOR_INCUMBENT['advantage'])
        self.assertEqual({k: v for k, v in restored.items() if k != 'advantage'},
                         {k: v for k, v in original.items() if k != 'advantage'})
        # A variable candidate is ablated against the variable baseline.
        variable = dict(CANDIDATE, name='variable-candidate', variable=True,
                        genes=dict(DEFAULTS, material=7.))
        self.assertEqual(ablation_baseline(normalize_candidate(variable)), VARIABLE_BASELINE)
        self.assertEqual([gene for gene, _ in ablation_genes(
            normalize_candidate(variable)['genes'], VARIABLE_BASELINE)], ['material'])

    def test_variable_candidates_never_share_a_schedule_with_flat_ones(self):
        variable = dict(CANDIDATE, name='variable-candidate', variable=True,
                        genes=dict(DEFAULTS, material=7.), baseline='candidate', attribute=False)
        spec = self.plan([CANDIDATE, variable])
        for item in spec['experiments']:
            # References are shared, so the two candidates never meet: no
            # schedule contains two population entrants in different modes.
            modes = {entrant['genome']['version'] for entrant in item['entrants']
                     if entrant['role'] == 'population'}
            self.assertEqual(len(modes), 1, item['name'])
        flat = next(e for e in spec['experiments'] if e['name'] == 'strength-depth')
        self.assertNotIn('variable-candidate', {e['name'] for e in flat['entrants']})

    def test_invalid_designs_thresholds_and_candidates_fail_before_freezing(self):
        for values in ({'practical_gain': 2.}, {'min_completion': -1.},
                       {'max_new_tactical_failures': -1}, {'require_disjoint_intervals': 'yes'}):
            with self.assertRaises(ValueError):
                Thresholds(**values)
        for values in ({'fixed_depth': 0}, {'cost_seconds': 0}, {'cost_depths': ()},
                       {'ablation_stages': ('nonsense',)}, {'tactical_cases': -1}):
            with self.assertRaises(ValueError):
                self.design(**values)
        for row in (dict(CANDIDATE, name=''), dict(CANDIDATE, genes={}),
                    dict(CANDIDATE, genes={'nonsense': 1.}), dict(CANDIDATE, unknown=1)):
            with self.assertRaises(ValueError):
                normalize_candidate(row)
        with self.assertRaises(ValueError):  # A name that shadows a reference.
            self.plan([dict(CANDIDATE, name='prior-incumbent')])
        with self.assertRaises(ValueError):  # A baseline nothing provides.
            self.plan([dict(CANDIDATE, baseline='nobody')])
        with self.assertRaises(ValueError):  # Two entrants cannot share a name.
            self.plan([CANDIDATE, dict(CANDIDATE, genes={'advantage': 41.})])
        with self.assertRaises(ValueError):  # A quota the corpus cannot meet.
            self.plan(starts_per_stage=99)

    def test_detectable_effect_is_published_with_the_plan(self):
        spec = self.plan()
        for item in spec['experiments']:
            self.assertTrue(item['power'])
            identities = {e['name']: e['sha256'] for e in item['entrants']}
            self.assertEqual(set(item['power']) - set(identities), set())
            for name, row in item['power'].items():
                self.assertGreater(row['hoeffding_radius'], 0.)
                self.assertGreaterEqual(row['seed_clusters'], 1)
                self.assertEqual(row['scheduled'],
                                 sum(identities[name] in task['candidates']
                                     for task in item['manifest']['tasks']))


class DecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = plan(corpus(), candidates=[SHIPPED], design=small_design(),
                        revision='test-revision')

    def results(self, *, candidate_lower=.8, baseline_lower=.2, radius=.05, eligible=True,
                tactical_extra=(), proven_extra=(), depth_loss=0, ratio=1.):
        def tactic(failed, proven, strict=None):
            return dict(cases=10, passed=10 - len(failed), failed=len(failed),
                        failed_ids=list(failed), proven_failures=list(proven),
                        strict_failed=len(strict if strict is not None else failed),
                        strict_failed_ids=list(strict if strict is not None else failed),
                        budget_stopped=[], limits=dict(tactics_module.CERTIFIED_LIMITS))

        def profile(depth, labels):
            return dict(equal_time={'by_budget': {'0.05': dict(median_completed_depth=depth,
                                                               min_completed_depth=depth,
                                                               median_work=10)}},
                        static=dict(median_seconds=.001, median_work=10),
                        labels=dict(labels_per_hour=labels, median_seconds_per_label=1.),
                        shipped=dict(limits=dict(max_depth=3, node_limit=200000, time_limit=1.),
                                     median_completed_depth=3, reached_requested_depth=1,
                                     positions=1, searches=[]),
                        ladder=dict(searches=[dict(depth=1, complete=True, work=1000, seconds=.5)]))
        def rows(mode):
            return dict(leaderboards={mode: [
                entry('candidate', lower=candidate_lower, radius=radius, eligible=eligible,
                      opponent='prior-incumbent'),
                entry('prior-incumbent', lower=baseline_lower, radius=radius,
                      opponent='candidate')]})
        depth_board, wall_board = rows('depth'), rows('wall')
        return dict(
            experiments={'strength-depth': dict(report=depth_board, complete=True),
                         'strength-wall': dict(report=wall_board, complete=True)},
            tactics={'candidate': dict(certified=tactic(tactical_extra, proven_extra),
                                       shipped=tactic(tactical_extra, proven_extra)),
                     'prior-incumbent': dict(certified=tactic([], []), shipped=tactic([], []))},
            parity={'candidate': dict(passed=True), 'prior-incumbent': dict(passed=True)},
            cost={'candidate': profile(3 - depth_loss, 100. * ratio),
                  'prior-incumbent': profile(3, 100.)})

    def test_adoption_requires_gain_separation_eligibility_and_safety(self):
        decision = report_module.verdict(self.plan, self.results())
        self.assertEqual(decision['outcomes']['candidate'], 'adopt')
        row = decision['candidates']['candidate']
        self.assertTrue(row['established'] and row['separated'] and not row['blocking'])
        # Overlapping intervals are an observation, not an established gain.
        overlap = report_module.verdict(self.plan, self.results(radius=.5))
        self.assertEqual(overlap['outcomes']['candidate'], 'retain-defaults')
        # Ineligible entrants block any conclusion at all.
        ineligible = report_module.verdict(self.plan, self.results(eligible=False))
        self.assertEqual(ineligible['outcomes']['candidate'], 'inconclusive')
        # A newly failed proof fixture blocks adoption outright.
        unsafe = report_module.verdict(self.plan, self.results(tactical_extra=['t1'],
                                                               proven_extra=['t1']))
        self.assertEqual(unsafe['outcomes']['candidate'], 'inconclusive')
        self.assertTrue(any('proof' in reason for reason in
                            unsafe['candidates']['candidate']['blocking']))
        # Losing depth at equal time blocks adoption even with a strength gain.
        slow = report_module.verdict(self.plan, self.results(depth_loss=2))
        self.assertEqual(slow['outcomes']['candidate'], 'inconclusive')

    def test_an_established_loss_by_the_current_default_recommends_reverting(self):
        decision = report_module.verdict(self.plan, self.results(candidate_lower=.1,
                                                                 baseline_lower=.9))
        row = decision['candidates']['candidate']
        self.assertTrue(row['is_current_default'])
        self.assertTrue(self.plan['candidates'][0]['current_default'])
        self.assertTrue(row['regressed'])
        self.assertLess(row['primary']['head_to_head']['net'], 0.)
        self.assertEqual(row['outcome'], 'revert-recommended')
        # A point margin without a losing direct record is not a regression.
        mixed = self.results(candidate_lower=.1, baseline_lower=.9)
        for board in mixed['experiments'].values():
            for mode in board['report']['leaderboards']:
                for entrant in board['report']['leaderboards'][mode]:
                    for direct in entrant.get('by_opponent', {}).values():
                        direct.update(wins=direct['scheduled'], losses=0)
        self.assertEqual(report_module.verdict(self.plan, mixed)['outcomes']['candidate'],
                         'retain-defaults')

    def test_missing_evidence_is_inconclusive_not_a_pass(self):
        results = self.results()
        results['experiments'] = {}
        decision = report_module.verdict(self.plan, results)
        self.assertEqual(decision['outcomes']['candidate'], 'inconclusive')
        self.assertEqual(decision['candidates']['candidate']['attribution']['status'], 'pending')

    def test_starved_shipped_budget_is_reported_as_a_defect(self):
        results = self.results()
        results['cost']['candidate']['shipped'].update(reached_requested_depth=0,
                                                       median_completed_depth=1)
        rows = report_module.defects(self.plan, results)
        self.assertTrue(any(row['kind'] == 'shipped-budget-starves-search'
                            and row['configuration'] == 'candidate' for row in rows))

    def test_exported_preset_is_loadable_and_marked_with_its_outcome(self):
        results = self.results()
        decision = report_module.verdict(self.plan, results)
        row = configs(self.plan)['candidate']
        record = preset_module.export('candidate', row, plan=self.plan, decision=decision,
                                      profile=results['cost']['candidate'], depth=1)
        self.assertEqual(record['acceptance']['outcome'], 'adopt')
        self.assertEqual(record['measured_limits']['max_depth'], 1)
        self.assertGreater(record['measured_limits']['node_limit'], 1000)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'preset.json'
            path.write_text(json.dumps(record['preset']))
            loaded = SearchConfig.from_file(path)
        self.assertEqual(loaded.advantage_weight, DEFAULTS['advantage'])
        self.assertEqual(loaded.node_limit, record['measured_limits']['node_limit'])


class RunTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name)
        FakeEngine.instances = []
        FakeEngine.action = FakeEngine.cancel = FakeEngine.fail = None
        FakeEngine.stopped = False
        self.plan = plan(corpus(), candidates=[CANDIDATE], design=small_design(),
                         revision='test-revision')

    def tearDown(self):
        self.folder.cleanup()

    def run_matches(self, output='run'):
        return runner_module.run(self.plan, self.path / output, stages=('matches',),
                                 match_player=fake_match)

    def test_matches_run_resume_and_replay(self):
        first = self.run_matches()
        self.assertEqual(first['summary']['status'], 'complete')
        self.assertEqual(first['summary']['total_final'], first['summary']['total_scheduled'])
        played = len(FakeEngine.instances)
        self.assertGreater(played, 0)
        second = self.run_matches()
        self.assertEqual(len(FakeEngine.instances), played, 'Resume replayed a finished game')
        self.assertEqual(second['summary']['total_final'], first['summary']['total_final'])
        audit = runner_module.verify(self.plan, self.path / 'run')
        self.assertEqual(audit['verified_games'], first['summary']['total_scheduled'])
        for name, row in audit['experiments'].items():
            self.assertEqual(row['verified'], row['scheduled'], name)
        # Every candidate is reported on its protocol's own leaderboard.
        decision = json.loads((self.path / 'run' / 'decision.json').read_text())
        self.assertIn('candidate', decision['outcomes'])
        self.assertIn(decision['outcomes']['candidate'], report_module.OUTCOMES)

    def test_a_tampered_record_fails_verification(self):
        self.run_matches()
        name = self.plan['experiments'][0]['name']
        matches = sorted((self.path / 'run' / 'experiments' / name / 'matches').glob('*.json'))
        row = json.loads(matches[0].read_text())
        row['final_state_sha256'] = '0' * 64
        matches[0].write_text(json.dumps(row))
        with self.assertRaises(ValueError):
            runner_module.verify(self.plan, self.path / 'run')

    def test_a_second_plan_cannot_reuse_an_output_directory(self):
        self.run_matches()
        other = plan(corpus(), candidates=[dict(CANDIDATE, genes={'advantage': 41.})],
                     design=Design.from_dict(self.plan['design']), revision='test-revision')
        with self.assertRaises(ValueError):
            runner_module.run(other, self.path / 'run', stages=('matches',),
                              match_player=fake_match)

    def test_unknown_stages_and_worker_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            runner_module.run(self.plan, self.path / 'bad', stages=('nonsense',),
                              match_player=fake_match)
        with self.assertRaises(ValueError):
            runner_module.run(self.plan, self.path / 'bad', workers=99, stages=('matches',),
                              match_player=fake_match)


class TacticalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Capture puzzles the shipped evaluation solves. Wins and eliminations
        # are decided by terminal scores whatever the weights are, so only a
        # fixture the heuristic itself decides can show an evaluation
        # regression at all.
        cls.cases = tuple(case for case in tactics_module.puzzle_cases()
                          if case['category'] == 'safe_capture')[:3]

    def test_game_fixtures_are_certified_and_classified(self):
        rows = tactics_module.game_cases()
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(row['preventable'] for row in rows))
        self.assertTrue(all(row['certificate'] in ('proof', 'bounded_obligation') for row in rows))
        self.assertTrue(any(row['expected'] is None for row in rows),
                        'Original positions accept any move meeting the objective')

    def test_fixtures_carry_their_certificate_class(self):
        for case in tactics_module.puzzle_cases():
            expected = ('proof' if case['category'] in tactics_module.PROVEN_CATEGORIES
                        else 'bounded_obligation')
            self.assertEqual(case['certificate'], expected)
            self.assertTrue(case['preventable'])

    def test_evaluation_reports_passes_failures_and_budget_stops(self):
        result = tactics_module.evaluate(SearchConfig(), cases=self.cases)
        self.assertEqual(result['cases'], len(self.cases))
        self.assertEqual(result['passed'] + result['failed'], result['cases'])
        self.assertFalse(result['mutated_inputs'])
        self.assertEqual(result['limits'], dict(tactics_module.CERTIFIED_LIMITS))
        # A deliberately crippled evaluation must lose fixtures, not pass them.
        crippled = replace(SearchConfig(), count_weight=-100., advantage_weight=-100.,
                           attack_enabled=False, defence_enabled=False)
        broken = tactics_module.evaluate(crippled, cases=self.cases)
        self.assertGreater(broken['failed'], 0)
        comparison = tactics_module.compare(broken, result)
        self.assertEqual(comparison['new_failures'],
                         sorted(set(broken['failed_ids']) - set(result['failed_ids'])))
        self.assertTrue(comparison['new_failures'])

    def test_the_shipped_profile_uses_the_shipped_limits(self):
        self.assertEqual(tactics_module.SHIPPED_LIMITS,
                         dict(max_depth=SearchConfig().max_depth,
                              node_limit=SearchConfig().node_limit,
                              time_limit=SearchConfig().time_limit))


if __name__ == '__main__':
    unittest.main()
