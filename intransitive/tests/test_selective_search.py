"""Guarded selective search: safety properties, exercised cutoffs and oracles."""
from dataclasses import replace
from itertools import product
from math import inf, nextafter
import copy
import unittest
from unittest.mock import patch
import numpy as np

from intransitive.heuristics import AlphaBetaPlayer, SearchConfig, exhaustive_minimax
from intransitive.heuristics.budget import Budget, BudgetExpired
from intransitive.heuristics.position import SearchPosition
from intransitive.heuristics.search import prove
from intransitive.heuristics import selective
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveLogicNumba import Board
from intransitive.IntransitiveDisplay import parse_move
from intransitive.rust_teacher import BINARY, RustTeacher
from intransitive.tests.test_heuristics import position
from intransitive.tests.test_draws import load_history, clock_history

MODES = list(product((False, True), repeat=2))
CONFIG = SearchConfig(max_depth=3, proof_nodes=0, node_limit=10**9, time_limit=60,
                      pvs_enabled=True, aspiration_enabled=True,
                      advantage_weight=25., attack_enabled=False, defence_enabled=False)


def quiet_state():
    return position({'A2':1, 'B2':2, 'A3':3, 'B3':1, 'D4':2,
                     'H7':-1, 'I7':-2, 'H8':-3, 'I8':-1})


def budget():
    return Budget(10**9, 60)


class SelectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        AlphaBetaPlayer(config=CONFIG)._prepare()

    def player(self, **kwargs):
        player = AlphaBetaPlayer(config=replace(CONFIG, **kwargs))
        player._prepare()
        return player

    def test_config_validation_and_scale_policy(self):
        for key, bad in [('nmp_enabled', 1), ('futility_enabled', None),
                         ('nmp_min_depth', 2), ('nmp_reduction', 2),
                         ('futility_max_depth', 3), ('futility_margin', float('nan')),
                         ('futility_margin', .05), ('futility_margin', 17.),
                         ('search_version', 'unknown')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                replace(CONFIG, **{key:bad})
        # The allowance multiplier now reaches below one, so an evolved-scale
        # margin can be shrunk to a value that actually prunes.
        self.assertEqual(replace(CONFIG, futility_margin=1/16).futility_margin, 1/16)
        self.assertFalse(selective.supported(replace(CONFIG, variable_material_enabled=True)))
        self.assertFalse(selective.supported(SearchConfig()))
        self.assertFalse(selective.supported(replace(CONFIG, pressure_enabled=True, pressure_weight=-1)))
        for field in ('count_weight', 'advantage_weight', 'predator_zero_bonus'):
            self.assertFalse(selective.supported(replace(CONFIG, **{field:200})))
        self.assertFalse(selective.supported(replace(CONFIG, pressure_enabled=True, pressure_weight=21)))
        with self.assertRaises(ValueError):
            AlphaBetaPlayer(config=replace(CONFIG, nmp_enabled=True), use_compact=False)._prepare()

    def test_unsupported_scales_refuse_rather_than_silently_disable(self):
        # Issue #66: a technique that is explicitly enabled must never be turned
        # off behind the caller's back by the evaluator-scale interlock.
        for technique in ('nmp_enabled', 'futility_enabled'):
            for field, value in (('advantage_weight', 75.), ('count_weight', 70.),
                                 ('attack_enabled', True), ('variable_material_enabled', True),
                                 ('pressure_weight', 21.)):
                broken = dict({field: value}, **{technique: True})
                if field == 'pressure_weight':
                    broken['pressure_enabled'] = True
                with self.subTest(technique=technique, field=field):
                    with self.assertRaisesRegex(ValueError, 'selective_evaluator_enabled'):
                        replace(CONFIG, **broken)
                    accepted = replace(CONFIG, selective_evaluator_enabled=True, **broken)
                    self.assertTrue(selective.supported(accepted))
        # Unsupported scales alone remain loadable while both techniques are off.
        inert = replace(CONFIG, advantage_weight=75.)
        self.assertFalse(selective.supported(inert))
        self.assertEqual(selective.unsupported_scales(inert), ['advantage_weight=75.0 (expected 25)'])

    def test_activation_reports_unreachable_techniques(self):
        # Every blocker the issue verified by experiment is now visible in the
        # configuration itself, before a single node is searched.
        cfg = replace(CONFIG, selective_evaluator_enabled=True, nmp_enabled=True,
                      futility_enabled=True, lmr_enabled=True, max_depth=2, pvs_enabled=False)
        report = selective.activation(cfg)
        for technique in ('nmp', 'futility'):
            self.assertTrue(any('pvs_enabled' in reason for reason in report[technique]['blockers']))
        self.assertTrue(any('nmp_min_depth' in reason for reason in report['nmp']['blockers']))
        self.assertTrue(any('lmr_min_depth' in reason for reason in report['lmr']['blockers']))
        self.assertEqual(selective.activation(cfg, compact=False)['nmp']['blockers'][0],
                         'requires the compact Python backend')
        self.assertFalse(selective.activation(replace(cfg, lmr_enabled=False))['lmr']['enabled'])
        self.assertEqual(selective.activation(replace(cfg, lmr_enabled=True, max_depth=4,
            compiled_ordering_enabled=False, ordering_enabled=False))['lmr']['blockers'],
            ['requires ordering_enabled or compiled_ordering_enabled'])
        self.assertEqual(len(selective.blocked(cfg)), 3)
        # A configuration where all three can fire reports no blocker at all.
        working = replace(cfg, max_depth=4, pvs_enabled=True)
        self.assertEqual(selective.blocked(working), [])
        result = AlphaBetaPlayer(config=replace(working, proof_nodes=0)).analyze(quiet_state())
        self.assertTrue(result.selective['effective'])
        self.assertEqual(result.selective['unreachable'], {})
        self.assertEqual(result.selective['declared'], ['futility', 'lmr', 'nmp'])
        self.assertIsNone(result.selective['disabled_reason'])
        for technique in result.selective['fired']:
            self.assertGreater(result.selective[selective.TECHNIQUE_COUNTERS[technique]], 0)

    def test_guards_match_their_reference_scans(self):
        """The vectorised guards must answer exactly as the loops they replace."""
        from intransitive.heuristics.selective import distance, quiet_context
        from intransitive.IntransitiveConstants import action_destination
        from intransitive.IntransitiveLogicNumba import raw_movement_mask

        def guarded_reference(position, depth):
            pieces = position.pieces.ravel()
            if position.clock + depth + 1 >= 80 or any(n > 1 for n in position.occurrences.values()):
                return False
            if min(sum(position.counts[:3]), sum(position.counts[3:])) < 4:
                return False
            for square in np.flatnonzero(pieces):
                side = int(pieces[square] < 0)
                if distance(int(square), 80 if side == position.a1 else 0) <= max(3, (depth + 1)//2):
                    return False
            for side in (0, 1):
                actions = np.flatnonzero(raw_movement_mask(position.pieces, side))
                if len(actions) < 8:
                    return False
                for action in actions:
                    x, y = action_destination(int(action))
                    if position.pieces[y, x]:
                        return False
            return True

        def quiet_reference(position, action):
            pieces = position.pieces.ravel()
            source, (x, y) = action // 8, action_destination(action)
            target = y * 9 + x
            if pieces[target]:
                return False
            side = position.side
            goal = 80 if side == position.a1 else 0
            own = [int(s) for s in np.flatnonzero(pieces) if int(pieces[s] < 0) == side]
            if distance(source, goal) <= min(distance(s, goal) for s in own):
                return False
            for square in np.flatnonzero(pieces):
                if (int(pieces[square] < 0) != side
                        and min(distance(int(square), source), distance(int(square), target)) <= 2):
                    return False
            return distance(target, goal) > 3

        from intransitive.tests.test_attribution import random_positions
        states = [quiet_state(), position({'D4':1, 'F6':-2}),
                  position({'A2':1,'B2':2,'A3':3,'B3':1,'D4':2,'E5':1,
                            'H7':-1,'I7':-2,'H8':-3,'I8':-1,'F6':-2,'E4':-3})]
        states += random_positions(games=2, plies=40, seed=101)[:6]
        checked = 0
        for state in states:
            p = SearchPosition(state)
            for depth in (1, 2, 3, 4):
                self.assertEqual(selective.guarded(p, depth, budget()),
                                 guarded_reference(p, depth), (depth,))
            context = quiet_context(p)
            for action in map(int, p.legal()):
                expected = quiet_reference(p, action)
                # Hoisting the shared scan must not change a single answer.
                self.assertEqual(selective.quiet(p, action), expected, action)
                self.assertEqual(selective.quiet(p, action, context), expected, action)
                checked += 1
        self.assertGreater(checked, 250)

    def test_relaxed_probe_guard_only_drops_the_side_to_move_captures(self):
        """The relaxation is one clause, for one technique, in one direction."""
        from intransitive.tests.test_attribution import random_positions
        strict = relaxed = 0
        for state in random_positions(games=4, plies=60, seed=101):
            p = SearchPosition(state)
            a = selective.guarded(p, 3, budget())
            b = selective.guarded(p, 3, budget(), ignore_own_captures=True)
            # Relaxing a refusal can only ever admit more positions.
            self.assertFalse(a and not b)
            strict += a
            relaxed += b
        self.assertGreater(relaxed, strict)
        # The opponent's captures still refuse a probe: this is the threat a
        # pass declines to answer, and it is why the clause is not simply gone.
        p = SearchPosition(quiet_state())
        self.assertTrue(selective.guarded(p, 3, budget(), ignore_own_captures=True))
        with patch('intransitive.heuristics.selective.raw_movement_mask') as mask:
            captures = np.zeros(648, dtype=np.int8)
            captures[parse_move('B3 C4')] = 1
            mask.side_effect = lambda pieces, side: (captures if side != p.side
                                                     else np.ones(648, dtype=np.int8))
            self.assertFalse(selective.guarded(p, 3, budget(), ignore_own_captures=True))
        # The search only ever relaxes it for a probe, and only when asked.
        self.assertFalse(replace(CONFIG, nmp_enabled=True).nmp_relaxed_guard_enabled)

    def test_reductions_are_adaptive_and_keep_a_ply_below_them(self):
        base = replace(CONFIG, lmr_enabled=True, selective_evaluator_enabled=True,
                       nmp_enabled=True, max_depth=12)
        player = AlphaBetaPlayer(config=base)
        # A fixed single ply stays the default, for both techniques.
        self.assertEqual(player._reduction(9, 9, False, 1, 0.), 1)
        self.assertEqual(player._null_reduction(9), 1)
        player.config = replace(base, lmr_depth_divisor=3, lmr_index_divisor=4,
                                nmp_depth_divisor=3)
        self.assertEqual(player._reduction(3, 3, False, 1, 0.), 1)
        self.assertEqual(player._reduction(9, 3, False, 1, 0.), 3)
        self.assertEqual(player._reduction(9, 11, False, 1, 0.), 5)
        self.assertEqual(player._null_reduction(3), 1)
        self.assertEqual(player._null_reduction(9), 3)
        # One ply always survives below the reduction, whatever the divisors say.
        player.config = replace(base, lmr_depth_divisor=1, nmp_depth_divisor=1)
        for depth in range(3, 13):
            self.assertLessEqual(player._reduction(depth, 9, False, 1, 0.), depth - 2)
            self.assertLessEqual(player._null_reduction(depth), depth - 2)
            self.assertGreaterEqual(player._null_reduction(depth), 1)
        # Everything a reduction is refused for stays refused.
        player.config = replace(base, lmr_depth_divisor=4)
        for reason in (dict(depth=2), dict(index=2), dict(capture=True), dict(ply=0),
                       dict(best=inf)):
            arguments = {**dict(depth=9, index=9, capture=False, ply=1, best=0.), **reason}
            self.assertEqual(player._reduction(**arguments), 0, reason)
        for key, bad in (('lmr_depth_divisor', -1), ('lmr_index_divisor', 65),
                         ('nmp_depth_divisor', 33), ('lmr_depth_divisor', 1.5)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                replace(CONFIG, **{key: bad})

    def test_reductions_compose_with_scout_windows(self):
        # Issue #66: a scout search used to win the branch outright, so late
        # quiet moves were reduced only below an already-null window.
        cfg = replace(CONFIG, selective_evaluator_enabled=True, lmr_enabled=True,
                      pvs_enabled=True, aspiration_enabled=False, max_depth=5, proof_nodes=0)
        composed = AlphaBetaPlayer(config=cfg).analyze(quiet_state())
        self.assertGreater(composed.pvs_probes, 0)
        self.assertGreater(composed.selective['lmr_reduced'], 0)
        # Scout searches alone stay exactly as they were: no reduction applies
        # when LMR is off, so PVS remains an exact-value optimization.
        plain = AlphaBetaPlayer(config=replace(cfg, lmr_enabled=False)).analyze(quiet_state())
        self.assertEqual(plain.selective['lmr_reduced'], 0)
        exhaustive = AlphaBetaPlayer(config=replace(cfg, lmr_enabled=False,
                                                    pvs_enabled=False)).analyze(quiet_state())
        self.assertEqual(plain.score, exhaustive.score)

    def test_composed_reductions_cost_no_certified_tactic(self):
        """Reducing the scout is only worth it if the tactics still come back.

        Composing the reduction with the scout search makes it strictly more
        aggressive, so it has to be measured against the same corpus the
        unreduced scout search solves, not merely against a node count.
        """
        from intransitive.tests.test_tactics import CASES
        from intransitive.tests.tactical_oracle import load_case
        base = replace(CONFIG, max_depth=4, proof_nodes=64, pvs_enabled=True,
                       aspiration_enabled=False, selective_evaluator_enabled=True)
        missed, nodes = {}, {}
        for name, options in (('scout', {}), ('reduced', dict(lmr_enabled=True)),
                              ('adaptive', dict(lmr_enabled=True, lmr_depth_divisor=3,
                                                lmr_index_divisor=4))):
            config = replace(base, **options)
            missed[name], nodes[name] = [], 0
            for case in CASES:
                reference, expected = load_case(case)
                result = AlphaBetaPlayer(config=config).analyze(reference.storage())
                nodes[name] += result.nodes
                if result.action != expected:
                    missed[name].append(case['id'])
        # No tactic that the plain scout search finds may be lost to a reduction.
        for name in ('reduced', 'adaptive'):
            self.assertLessEqual(set(missed[name]), set(missed['scout']),
                                 (name, sorted(set(missed[name]) - set(missed['scout']))))
            self.assertLess(nodes[name], nodes['scout'], name)

    def test_verification_reuses_only_its_own_table_namespace(self):
        player = self.player(nmp_enabled=True, max_depth=4)
        p = SearchPosition(quiet_state())
        player._search(p, 3, 0., nextafter(0., inf), 1, budget())
        self.assertGreater(player._selective_stats['verification_searches'], 0)
        namespaces = {key[0].split(b'\0')[0] for key in player.table}
        self.assertIn(b'verified-v1', namespaces)
        self.assertIn(b'selective-v1', namespaces)
        # The context restores every switch it touched, including the namespace.
        self.assertTrue(player.use_table)
        self.assertEqual(player._table_namespace, b'selective-v1\0')
        self.assertFalse(player._selective_disabled)

    def test_pressure_probes_and_evolved_scale_refusal(self):
        from intransitive.teacher_learning import teacher_config
        from intransitive.supervised_minimax import valid_teacher_record
        for nmp, futility in MODES[1:]:
            player = self.player(nmp_enabled=nmp, futility_enabled=futility,
                                 pressure_enabled=True, pressure_weight=10., pressure_cache_entries=16)
            p = SearchPosition(quiet_state())
            alpha = 0. if nmp else 500.
            player._search(p, 3 if nmp else 1, alpha, nextafter(alpha, inf), 1, budget())
            self.assertGreater(player._selective_stats['nmp_cutoffs' if nmp else 'futility_pruned'], 0)
            with self.assertRaisesRegex(ValueError, 'selective_evaluator_enabled'):
                replace(player.config, advantage_weight=75.)
            # The same scales are accepted, and stay effective, once the
            # experimental opt-in is recorded in the search identity.
            evolved = replace(player.config, advantage_weight=75., selective_evaluator_enabled=True,
                              max_depth=4)
            self.assertIn('"selective_evaluator_enabled": true', evolved.identity())
            result = AlphaBetaPlayer(config=evolved).analyze(quiet_state())
            self.assertTrue(result.selective['effective'], result.selective['disabled_reason'])
            with self.assertRaises(ValueError):
                teacher_config({'teacher': {'search': evolved.to_dict()}})
        self.assertFalse(valid_teacher_record(dict(teacher_depth=6, teacher_reason='selective_result')))

    def test_experimental_evaluators_and_margin(self):
        state = quiet_state()
        p = SearchPosition(state)
        for variable in (False, True):
            cfg = SearchConfig(variable_material_enabled=variable,
                selective_evaluator_enabled=True, nmp_enabled=True,
                futility_enabled=True, proof_nodes=0, max_depth=2,
                node_limit=10**9, time_limit=60, pvs_enabled=True)
            self.assertTrue(selective.supported(cfg))
            self.assertGreater(selective.margin(cfg, 2, p), selective.margin(CONFIG, 2, p))
            # A quiet first ply cannot change the piece counts, so material and
            # advantage are charged only from the second ply onwards.
            modules = SearchConfig(selective_evaluator_enabled=True, attack_enabled=False,
                                   defence_enabled=False, overload_enabled=False)
            self.assertEqual(selective.margin(modules, 1, p), 0.)
            self.assertGreater(selective.margin(modules, 2, p), 0.)
            self.assertGreater(selective.margin(cfg, 1, p), 0.)
            self.assertGreater(selective.margin(cfg, 2, p), 2*selective.margin(cfg, 1, p))
            # The validated original allowance is untouched.
            plain = replace(CONFIG, selective_evaluator_enabled=False)
            self.assertEqual(selective.margin(plain, 1), plain.count_weight/2 + plain.advantage_weight)
            signed = replace(cfg, count_weight=-100., advantage_weight=-25., attack_weight=-20., defence_weight=-30.)
            self.assertGreater(selective.margin(signed, 1, p), 0)
            player = AlphaBetaPlayer(config=cfg)
            player._prepare()
            # Non-PV quiet fixtures exercise actual cutoffs with adopted weights.
            for nmp in (False, True):
                player.config = replace(cfg, nmp_enabled=nmp, futility_enabled=not nmp)
                alpha = -9000. if nmp else 9000.
                player._search(p, 3 if nmp else 1, alpha, nextafter(alpha, inf), 1, budget())
                self.assertGreater(player._selective_stats['nmp_cutoffs' if nmp else 'futility_pruned'], 0)
                np.testing.assert_array_equal(p.export(), state)
            if BINARY.exists():
                with RustTeacher() as rust:
                    # Toggling pruning across a reused native search must not leak
                    # selective state between configurations.
                    for pruning in (False, True, False):
                        native = rust.analyze(state, depth=2, seconds=60., proof_nodes=0,
                            variable_material_enabled=variable, selective_evaluator_enabled=True,
                            nmp_enabled=pruning, futility_enabled=pruning, reuse=True)
                        py = AlphaBetaPlayer(config=replace(cfg, nmp_enabled=pruning,
                                                            futility_enabled=pruning)).analyze(state)
                        self.assertEqual(native['selective']['enabled'], pruning)
                        self.assertEqual(native['selective']['effective'], pruning)
                        self.assertAlmostEqual(py.score, native['score'])
                        self.assertEqual(native['completed_depth'], 2)
                    # Both backends refuse experimental scales without the opt-in
                    # rather than accepting the flags and pruning nothing.
                    with self.assertRaises(ValueError):
                        rust.analyze(state, depth=2, seconds=60., proof_nodes=0,
                            variable_material_enabled=variable, selective_evaluator_enabled=False,
                            nmp_enabled=True, futility_enabled=True, reuse=True)
                    with self.assertRaisesRegex(ValueError, 'selective_evaluator_enabled'):
                        replace(cfg, selective_evaluator_enabled=False)
        with self.assertRaises(ValueError):
            replace(CONFIG, selective_evaluator_enabled=1)

    def test_selective_mate_requires_requested_depth(self):
        state = position({'H8':1, 'C3':-2})
        cfg = replace(CONFIG, max_depth=3, nmp_enabled=True, futility_enabled=True, proof_nodes=64)
        result = AlphaBetaPlayer(config=cfg).analyze(state)
        self.assertEqual(result.completed_depth, 3)
        self.assertEqual(result.stop_reason, 'selective_result')
        self.assertEqual(result.explanation['proof']['status'], 'unknown')

    def test_null_roundtrip_clock_repetition_exception_and_ownership(self):
        state = quiet_state()
        for history in ([state[:,:,0]]*3, [state[:,:,0]]*5, clock_history(state[:,:,0],79)):
            state = load_history(Board(), history)
            p = SearchPosition(state)
            key = p.key()
            before = copy.deepcopy(p.__dict__)
            with self.assertRaisesRegex(RuntimeError, 'interrupt'):
                with p.null_turn():
                    self.assertEqual(p.side, 1-before['side'])
                    self.assertEqual(p.a1, before['a1'])
                    self.assertEqual(p.total, before['total'])
                    self.assertEqual(p.history, before['history'])
                    self.assertEqual(p.terminal()[1], 'ongoing')
                    p.push(int(p.legal()[0]))
                    try:
                        self.assertEqual(p.terminal()[1], 'ongoing')
                        raise RuntimeError('interrupt')
                    finally:
                        p.pop()
            self.assertEqual(p.key(), key)
            np.testing.assert_array_equal(p.export(), state)
            self.assertEqual(p.history, before['history'])
            self.assertEqual(p.occurrences, before['occurrences'])
            self.assertEqual(p.counts, before['counts'])
            self.assertEqual(p.stack, [])

    def test_guards_for_sparse_threat_race_capture_and_draw_boundaries(self):
        self.assertTrue(selective.guarded(SearchPosition(quiet_state()),3,budget()))
        cases = [position({'C3':1,'G7':-2}),
                 position({'A1':1,'B1':1,'A2':1,'B2':1,'A3':-1,'B3':-1,'C3':-1,'C2':-1}),
                 position({'A2':1,'B2':2,'A3':3,'H8':1,'H7':-1,'I7':-2,'G8':-3,'I8':-1}),
                 position({'A2':1,'B2':2,'A3':3,'D4':1,'E4':-2,'I7':-2,'H8':-3,'I8':-1})]
        pieces=quiet_state()[:,:,0]
        cases += [load_history(Board(), [pieces]*3),load_history(Board(),clock_history(pieces,78))]
        for state in cases:
            self.assertFalse(selective.guarded(SearchPosition(state),3,budget()))
        p=SearchPosition(quiet_state())
        self.assertFalse(selective.quiet(p,parse_move('D4 E5'))) # fastest runner
        self.assertTrue(selective.quiet(p,parse_move('A2 A1')))

    def test_exercised_nmp_verification_futility_and_restoration(self):
        for nmp,futility in MODES[1:]:
            player=self.player(nmp_enabled=nmp,futility_enabled=futility)
            p=SearchPosition(quiet_state()); before=p.export(); key=p.key()
            alpha=0. if nmp else 500.
            score,line=player._search(p,3 if nmp else 1,alpha,nextafter(alpha,inf),1,budget())
            self.assertTrue(np.isfinite(score)); self.assertTrue(line)
            np.testing.assert_array_equal(p.export(),before)
            self.assertEqual(p.key(),key)
            if nmp:
                self.assertEqual(score, nextafter(alpha, inf))
                self.assertGreater(player._selective_stats['nmp_attempts'],0)
                self.assertGreater(player._selective_stats['verification_searches'],0)
                self.assertGreater(player._selective_stats['nmp_cutoffs'],0)
                self.assertEqual(player._selective_stats['nmp_cutoffs'],player._selective_stats['verification_searches'])
            else:
                self.assertGreater(player._selective_stats['futility_pruned'],0)
            self.assertTrue(player.use_table)
            self.assertFalse(player._null_context)

    def test_verification_failure_and_null_exception_do_not_store_cutoff(self):
        player=self.player(nmp_enabled=True)
        real=player._search
        p=SearchPosition(quiet_state()); before=p.export()
        def probe(state,depth,alpha,beta,ply,b):
            if player._null_context:
                return -100., []
            if player._selective_disabled:
                return -100., []
            return real(state,depth,alpha,beta,ply,b)
        with patch.object(player,'_search',side_effect=probe):
            player._search(p,3,0.,nextafter(0.,inf),1,budget())
        self.assertEqual(player._selective_stats['verification_failures'],1)
        self.assertEqual(player._selective_stats['nmp_cutoffs'],0)
        player.reload()
        for error in (BudgetExpired('time'), BudgetExpired('work'), RuntimeError('fault')):
            def fail(state,depth,alpha,beta,ply,b):
                if player._null_context:
                    raise error
                return real(state,depth,alpha,beta,ply,b)
            with patch.object(player,'_search',side_effect=fail), self.assertRaises(type(error)):
                player._search(p,3,0.,nextafter(0.,inf),1,budget())
            np.testing.assert_array_equal(p.export(),before)
            self.assertTrue(player.use_table)
            self.assertFalse(player._selective_disabled)
            self.assertFalse(player._null_context)
            self.assertFalse(player.table)

    def test_verification_cancellation_and_no_pv_pruning(self):
        player = self.player(nmp_enabled=True, futility_enabled=True)
        p = SearchPosition(quiet_state())
        before = p.export()
        real = player._search
        def interrupt_verification(state, depth, alpha, beta, ply, b):
            if player._null_context:
                return -100., []
            if player._selective_disabled:
                raise BudgetExpired('work')
            return real(state, depth, alpha, beta, ply, b)
        with patch.object(player, '_search', side_effect=interrupt_verification), self.assertRaises(BudgetExpired):
            player._search(p, 3, 0., nextafter(0., inf), 1, budget())
        np.testing.assert_array_equal(p.export(), before)
        self.assertTrue(player.use_table)
        self.assertFalse(player._selective_disabled)
        self.assertFalse(player.table)
        # A full-window node is never itself eligible, even if a TT bound
        # tightens its window to adjacent representable endpoints.
        from intransitive.heuristics.search import Entry, position_key
        player._prepare()
        player.table[(b'selective-v1\0'+position_key(p), 3)] = Entry(3, nextafter(0.,inf), 'upper', -1, ())
        with patch('intransitive.heuristics.selective.guarded', return_value=False) as guard:
            player._search(p, 3, 0., 100., 1, budget())
        # Descendants may be non-PV, but this remaining-depth-three root cannot prune.
        self.assertFalse(any(call.args[1] == 3 for call in guard.call_args_list))

    def test_four_modes_oracle_tt_pvs_pressure_pv_and_fallback(self):
        state=position({'D4':1,'F6':-2})
        for nmp,futility in MODES:
            for pvs in (False,True):
                cfg=replace(CONFIG,nmp_enabled=nmp,futility_enabled=futility,pvs_enabled=pvs,
                            max_depth=2,pressure_enabled=True,pressure_weight=10.)
                player=AlphaBetaPlayer(config=cfg)
                expected,_=exhaustive_minimax(player.game,state,2,cfg,budget())
                for _ in range(2):
                    result=player.analyze(state)
                    self.assertAlmostEqual(result.score,expected)
                    board=state.copy();side=0
                    for action in result.pv:
                        self.assertTrue(player.game.getValidMoves(board,side)[action])
                        board,side=player.game.getNextState(board,side,action)
                    if nmp or futility:
                        self.assertTrue(result.score_bound.startswith('selective_'))
                        self.assertEqual(result.explanation['proof']['status'],'unknown')
                player.config=replace(cfg,nmp_enabled=False,futility_enabled=False)
                self.assertAlmostEqual(player.analyze(state).score,expected)
                player.config=replace(cfg,node_limit=0)
                fallback=player.analyze(state)
                self.assertEqual(fallback.selection_source,'legal_fallback')
                self.assertTrue(player.game.getValidMoves(state,0)[fallback.action])

    def test_proof_integrity_official_win_precedence_and_teacher_rejection(self):
        from intransitive.supervised_minimax import completed_teacher
        state=position({'H8':1,'C3':-2})
        expected=None
        for nmp,futility in MODES:
            cfg=replace(CONFIG,nmp_enabled=nmp,futility_enabled=futility,proof_nodes=64)
            proof=prove(IntransitiveGame(),SearchPosition(state),cfg,budget())
            if expected is None: expected=proof
            self.assertEqual(proof,expected)
            result=AlphaBetaPlayer(config=cfg).analyze(state)
            self.assertEqual(completed_teacher(result,1),not(nmp or futility))
        winning=position({'I9':1,'C3':-2})[:,:,0]
        p=SearchPosition(load_history(Board(),clock_history(winning,80)))
        self.assertEqual(p.terminal(),(0,'corner'))
        with p.null_turn(): self.assertEqual(p.terminal(),(0,'corner'))

    @unittest.skipUnless(BINARY.exists(),'build Rust first')
    def test_native_parity_and_configuration_isolation(self):
        with RustTeacher() as rust:
            for nmp,futility in MODES:
                for state in (position({'D4':1,'F6':-2}),quiet_state()):
                    cfg=replace(CONFIG,nmp_enabled=nmp,futility_enabled=futility,max_depth=2)
                    py=AlphaBetaPlayer(config=cfg).analyze(state)
                    native=rust.analyze(state,depth=2,weight=0.,proof_nodes=0,
                                        nmp_enabled=nmp,futility_enabled=futility,reuse=True,
                                        advantage=25., attack=0., defence=0.)
                    self.assertTrue(native['complete'])
                    self.assertAlmostEqual(py.score,native['score'])
                    self.assertEqual(native['selective']['enabled'],nmp or futility)
            with self.assertRaises(ValueError):rust.analyze(quiet_state(),nmp_enabled=1)
            with self.assertRaises(TypeError):rust.analyze(quiet_state(),unknown_enabled=True)

    def test_all_certified_tactics_in_four_modes(self):
        from intransitive.tests.test_tactics import CASES
        from intransitive.tests.tactical_oracle import load_case
        for nmp, futility in MODES:
            cfg = replace(CONFIG, nmp_enabled=nmp, futility_enabled=futility, proof_nodes=64)
            for case in CASES:
                reference, expected = load_case(case)
                result = AlphaBetaPlayer(config=cfg).analyze(reference.storage())
                self.assertEqual(result.action, expected, (case['id'], nmp, futility))

    def test_human_failures_and_forced_loss_controls(self):
        from intransitive.tests.test_game_blunders import DATA
        from intransitive.tests.reference_rules import Position, position as pieces
        from intransitive.record import load_record
        record=load_record(DATA['record_pgn'])
        baseline = {}
        for nmp,futility in MODES:
            cfg=replace(CONFIG,nmp_enabled=nmp,futility_enabled=futility,proof_nodes=64,
                        pvs_enabled=False,aspiration_enabled=False)
            for case in DATA['original_positions']+DATA['simplified_positions']:
                board=(Position.from_storage(record.states[case['ply']]) if 'ply' in case else
                       Position.fixture(pieces(case['pieces']),player=case['player']))
                result=AlphaBetaPlayer(config=cfg).analyze(board.storage())
                if not (nmp or futility):
                    baseline[case['id']] = result
                # Issue-52's existing tactical assertions remain unchanged.
                # This suite requires no divergence on those regression roots,
                # including their known depth-three failures.
                self.assertEqual(result.action,baseline[case['id']].action,case['id'])
                self.assertEqual(result.score,baseline[case['id']].score,case['id'])
            lost=record.states[77]
            result=AlphaBetaPlayer(config=cfg).analyze(lost)
            self.assertLess(result.score,-90000)
