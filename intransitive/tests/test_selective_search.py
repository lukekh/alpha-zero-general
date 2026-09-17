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
                         ('futility_margin', .9), ('search_version', 'unknown')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                replace(CONFIG, **{key:bad})
        self.assertFalse(selective.supported(replace(CONFIG, variable_material_enabled=True)))
        self.assertFalse(selective.supported(SearchConfig()))
        self.assertFalse(selective.supported(replace(CONFIG, pressure_enabled=True, pressure_weight=-1)))
        for field in ('count_weight', 'advantage_weight', 'predator_zero_bonus'):
            self.assertFalse(selective.supported(replace(CONFIG, **{field:200})))
        self.assertFalse(selective.supported(replace(CONFIG, pressure_enabled=True, pressure_weight=21)))
        with self.assertRaises(ValueError):
            AlphaBetaPlayer(config=replace(CONFIG, nmp_enabled=True), use_compact=False)._prepare()

    def test_pressure_probes_and_evolved_scale_disable(self):
        from intransitive.teacher_learning import teacher_config
        from intransitive.supervised_minimax import valid_teacher_record
        for nmp, futility in MODES[1:]:
            player = self.player(nmp_enabled=nmp, futility_enabled=futility,
                                 pressure_enabled=True, pressure_weight=10., pressure_cache_entries=16)
            p = SearchPosition(quiet_state())
            alpha = 0. if nmp else 500.
            player._search(p, 3 if nmp else 1, alpha, nextafter(alpha, inf), 1, budget())
            self.assertGreater(player._selective_stats['nmp_cutoffs' if nmp else 'futility_pruned'], 0)
            player.config = replace(player.config, advantage_weight=75.)
            result = player.analyze(quiet_state())
            self.assertFalse(result.selective['effective'])
            self.assertEqual(result.selective['nmp_attempts'], 0)
            self.assertEqual(result.selective['futility_pruned'], 0)
            with self.assertRaises(ValueError):
                teacher_config({'teacher': {'search': player.config.to_dict()}})
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
                    for enabled in (False, True, False):
                        native = rust.analyze(state, depth=2, seconds=60., proof_nodes=0,
                            variable_material_enabled=variable, selective_evaluator_enabled=enabled,
                            nmp_enabled=True, futility_enabled=True, reuse=True)
                        py = AlphaBetaPlayer(config=replace(cfg, selective_evaluator_enabled=enabled)).analyze(state)
                        self.assertEqual(native['selective']['effective'], enabled)
                        self.assertAlmostEqual(py.score, native['score'])
                        self.assertEqual(native['completed_depth'], 2)
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
