"""Public/oracle differential play, exact undo, cancellation and search parity."""
from dataclasses import replace
from itertools import product
from math import inf
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

from intransitive.IntransitiveConstants import MAX_TOTAL_PLY, NO_CAPTURE_LIMIT
from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveLogicNumba import validate_state
from intransitive.heuristics import AlphaBetaPlayer, SearchConfig, exhaustive_minimax
from intransitive.heuristics.budget import Budget, BudgetExpired
from intransitive.heuristics.evaluation import Evaluator, terminal_value
from intransitive.heuristics.position import SearchPosition
from intransitive.heuristics.proof import compact_proof, native_compact_proof, warm_compact_proof_kernel
from intransitive.heuristics.search import position_key, prove, prove_reference
from intransitive.record import load_record
from intransitive.tests.reference_rules import Position, position, action_between
from intransitive.tests.test_tactics import CASES
from intransitive.tests.tactical_oracle import load_case


def unlimited():
    return Budget(10**9, 600)


def snapshot(node):
    return (node.export().tobytes(), node.counts, node.history.copy(),
            node.occurrences.copy(), node.key(), len(node.stack))


class CompactSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()
        cls.config = SearchConfig(max_depth=3, proof_nodes=0, node_limit=10**9, time_limit=600)
        AlphaBetaPlayer(config=cls.config)._prepare()

    def assert_position(self, node, reference, state):
        self.assertEqual(node.export().tobytes(), state.tobytes())
        self.assertEqual(node.export().tobytes(), reference.storage().tobytes())
        validate_state(node.export())
        self.assertEqual(set(node.legal()), reference.legal())
        self.assertEqual(set(node.legal()), set(np.flatnonzero(self.game.getValidMoves(state, reference.player))))
        winner, reason = node.terminal()
        expected_reason, rewards = reference.terminal()
        self.assertEqual(reason, expected_reason)
        self.assertEqual(winner, next((side for side in (0, 1) if rewards[side] == 1), -1))
        for side in (0, 1):
            self.assertEqual(terminal_value(self.game, node, side), terminal_value(self.game, state, side))
        self.game.board.copy_state(state, False)
        self.assertEqual(reason, self.game.board.get_terminal_reason())
        self.assertEqual(node.occurrences[node.history[-1]], reference.repetitions())
        self.assertEqual(node.key(), position_key(state))
        self.assertEqual(node.counts, tuple(reference.pieces.count(code) for code in (1, 2, 3, -1, -2, -3)))

    def test_replay_and_random_legal_games_with_sibling_undo(self):
        rng = np.random.default_rng(40)
        record = load_record((Path(__file__).parents[1]/'benchmarks/search_budget/game79.pgn').read_text())
        # The complete recorded public histories, including nonzero move numbers.
        replay = SearchPosition(record.states[0])
        reference = Position.from_storage(record.states[0])
        for index, state in enumerate(record.states):
            self.assert_position(SearchPosition(state), Position.from_storage(state), state)
            self.assert_position(replay, reference, state)
            if index < len(record.actions):
                action = record.actions[index]
                reference, captured = reference.move(action)
                self.assertEqual(bool(replay.push(action)), captured)
        while replay.stack:
            replay.pop()
        self.assertEqual(replay.export().tobytes(), record.states[0].tobytes())
        for game_index in range(5):
            reference = Position.initial()
            node = SearchPosition(reference.storage())
            original = snapshot(node)
            for ply in range(100):
                state = reference.storage()
                self.assert_position(node, reference, state)
                legal = sorted(reference.legal())
                if not legal:
                    break
                before = snapshot(node)
                for action in rng.choice(legal, min(3, len(legal)), replace=False):
                    action = int(action)
                    child, captured = reference.move(action)
                    public, _ = self.game.getNextState(state, reference.player, action)
                    self.assertEqual(bool(node.push(action)), captured)
                    self.assert_position(node, child, public)
                    node.pop()
                    self.assertEqual(snapshot(node), before)
                action = int(rng.choice(legal))
                reference, _ = reference.move(action)
                node.push(action)
            while node.stack:
                node.pop()
            self.assertEqual(snapshot(node), original)

    def test_random_nested_sequences_and_owned_boundary_snapshots(self):
        rng = np.random.default_rng(4001)
        state = self.game.getInitBoard()
        before = state.tobytes()
        # Noncontiguous/read-only valid inputs also get detached on entry.
        state = np.asfortranarray(state)
        state.flags.writeable = False
        node = SearchPosition(state)
        frames = []
        saved = []
        for _ in range(500):
            legal = node.legal()
            if frames and (not len(legal) or len(frames) > 12 or rng.random() < .45):
                expected = frames.pop()
                node.pop()
                self.assertEqual(snapshot(node), expected)
            elif len(legal):
                frames.append(snapshot(node))
                exported = node.export()
                saved.append((exported, exported.tobytes()))
                node.push(int(rng.choice(legal)))
        while frames:
            expected = frames.pop()
            node.pop()
            self.assertEqual(snapshot(node), expected)
        self.assertEqual(state.tobytes(), before)
        self.assertTrue(all(item.tobytes() == data for item, data in saved))
        self.assertFalse(np.shares_memory(state, node.pieces))

    def test_every_capture_type_colour_and_symmetry(self):
        for side, kind, symmetry in product((0, 1), (1, 2, 3), range(12)):
            sign = 1 if side == 0 else -1
            reference = Position.fixture(position({'D4': sign*kind, 'E5': -sign*(kind%3+1),
                                                   'G7': -sign*kind}), player=side).transform(symmetry)
            for board in (reference, reference.relabel()):
                node = SearchPosition(board.storage())
                original = snapshot(node)
                for action in sorted(board.legal()):
                    child, captured = board.move(action)
                    self.assertEqual(bool(node.push(action)), captured)
                    self.assert_position(node, child, child.storage())
                    if captured:
                        self.assertEqual(node.clock, 0)
                        self.assertEqual(len(node.history), 1)
                        self.assertEqual(sum(node.counts), sum(original[1])-1)
                    node.pop()
                    self.assertEqual(snapshot(node), original)

    def test_draw_identity_counts_order_turn_goals_and_move_number(self):
        a = position({'D4': 1, 'F6': -2})
        b = position({'D5': 1, 'F6': -2})
        c = position({'D5': 1, 'F7': -2})
        d = position({'E5': 1, 'F7': -2})
        first = Position.fixture(a, history=(a, b, c, d, a))
        reordered = Position.fixture(a, history=(c, d, a, b, a))
        third = Position.fixture(a, history=(a, b, a, d, a))
        for reference in (first, reordered, third):
            self.assert_position(SearchPosition(reference.storage()), reference, reference.storage())
        key = SearchPosition(first.storage()).key()
        self.assertEqual(key, SearchPosition(reordered.storage()).key())
        self.assertEqual(key, SearchPosition(replace(first, ply=129).storage()).key())
        self.assertNotEqual(key, SearchPosition(third.storage()).key())
        self.assertNotEqual(key, SearchPosition(replace(first, a1_defender=1).storage()).key())
        self.assertNotEqual(key, SearchPosition(Position.fixture(a, player=1).storage()).key())
        self.assertEqual(SearchPosition(first.storage()).terminal()[1], 'ongoing')
        self.assertEqual(SearchPosition(third.storage()).terminal()[1], 'repetition')

    def test_clock_boundary_capture_reset_and_official_win_precedence(self):
        for pieces in ({'H8': 3, 'D4': -1}, {'D4': 1, 'E5': -2}, {'I9': 1, 'D4': -2}):
            current = position(pieces)
            other = position({'C3': 1, 'G7': -3})
            for clock in (29, 30, NO_CAPTURE_LIMIT-1, NO_CAPTURE_LIMIT):
                reference = Position.fixture(current, history=(other,)*clock + (current,))
                for board in (reference, reference.relabel()):
                    node = SearchPosition(board.storage())
                    self.assert_position(node, board, board.storage())
                    before = snapshot(node)
                    for action in sorted(board.legal()):
                        child, _ = board.move(action)
                        node.push(action)
                        self.assert_position(node, child, child.storage())
                        node.pop()
                        self.assertEqual(snapshot(node), before)
        # Initial position is occurrence one, so a legal four-ply loop draws
        # only after its second repetition (the third occurrence).
        reference = Position.fixture(position({'D4': 1, 'G7': -1}))
        node = SearchPosition(reference.storage())
        loop = ('D4 D5', 'G7 G8', 'D5 D4', 'G8 G7') * 2
        for i, move in enumerate(loop):
            action = action_between(*move.split())
            reference, _ = reference.move(action)
            node.push(action)
            self.assert_position(node, reference, reference.storage())
            self.assertEqual(node.terminal()[1], 'repetition' if i == 7 else 'ongoing')

    def test_official_history_rollover_and_total_overflow_are_exact(self):
        game = IntransitiveGame(modelling_draws=False)
        ref = Position.fixture(position({'D4': 1, 'G7': -1}))
        state = replace(ref, ply=126).storage()
        node = SearchPosition(state, modelling_draws=False)
        before = snapshot(node)
        frames = []
        loop = ('D4 D5', 'G7 G8', 'D5 D4', 'G8 G7') * 25
        for move in loop:
            action = action_between(*move.split())
            frames.append(snapshot(node))
            state, _ = game.getNextState(state, node.side, action)
            node.push(action)
            self.assertEqual(node.export().tobytes(), state.tobytes())
        for frame in reversed(frames):
            node.pop()
            self.assertEqual(snapshot(node), frame)
        self.assertEqual(snapshot(node), before)
        node = SearchPosition(replace(ref, ply=MAX_TOTAL_PLY).storage())
        before = snapshot(node)
        with self.assertRaisesRegex(ValueError, 'Total ply overflow'):
            node.push(int(node.legal()[0]))
        self.assertEqual(snapshot(node), before)
        with self.assertRaisesRegex(ValueError, 'Total ply overflow'):
            compact_proof(node, 2, 64, 129)
        self.assertEqual(snapshot(node), before)

    def test_entry_rejects_malformed_states_and_public_moves_still_validate(self):
        state = self.game.getInitBoard()
        for malformed in (state[:, :, :33], state.astype(np.int16), state*0):
            with self.assertRaises(ValueError):
                SearchPosition(malformed)
        node = SearchPosition(state)
        before = snapshot(node)
        with self.assertRaises(ValueError):
            self.game.getNextState(state, 0, -1)
        self.assertEqual(snapshot(node), before)

    def test_fixed_depth_table_reference_and_all_symmetry_perspectives(self):
        board = Position.fixture(position({'D4': 1, 'E5': -2, 'G7': -3}))
        for symmetry in range(12):
            for reference in (board.transform(symmetry), board.transform(symmetry).relabel()):
                state = reference.storage()
                expected, _ = exhaustive_minimax(self.game, state, 2, self.config, unlimited())
                outcomes = []
                for compact, table in product((False, True), repeat=2):
                    player = AlphaBetaPlayer(config=replace(self.config, max_depth=2),
                                             use_compact=compact, use_table=table)
                    result = player.analyze(state)
                    self.assertEqual(result.score, expected)
                    self.assertIn(result.action, reference.legal())
                    outcomes.append((result.score, result.action, result.pv))
                self.assertTrue(all(outcome == outcomes[0] for outcome in outcomes))

    def test_optional_modules_and_material_remain_exact(self):
        state = Position.fixture(position({'D4': 1, 'G7': -2, 'H7': 3})).storage()
        node = SearchPosition(state)
        for flags in product((False, True), repeat=3):
            config = replace(self.config, attack_enabled=flags[0], defence_enabled=flags[1], overload_enabled=flags[2])
            evaluator = Evaluator(self.game, config)
            for side in (0, 1):
                expected = evaluator.score(state, side, unlimited(), proof={'status': 'unknown'})
                actual = evaluator.score(node, side, unlimited(), proof={'status': 'unknown'})
                self.assertEqual(actual.hex(), expected.hex())

    def test_cancellation_errors_cutoffs_restore_position_and_completed_table(self):
        state = Position.fixture(position({'D4': 1, 'E5': -2, 'F6': 3, 'G7': -1})).storage()
        for stop_ply in (1, 2, 3):
            for error in (BudgetExpired('time'), BudgetExpired('work'), RuntimeError('injected')):
                player = AlphaBetaPlayer(config=self.config)
                player._prepare()
                node = SearchPosition(state)
                before = snapshot(node)
                player._search(node, 1, -inf, inf, 0, unlimited())
                completed = player.table.copy()
                search = player._search
                def checked(item, depth, alpha, beta, ply, budget):
                    self.assertEqual(item.counts, tuple(int(np.count_nonzero(item.pieces == c))
                                                       for c in (1, 2, 3, -1, -2, -3)))
                    if ply == stop_ply:
                        raise error
                    return search(item, depth, alpha, beta, ply, budget)
                with patch.object(player, '_search', checked):
                    with self.assertRaises(type(error)):
                        player._search(node, 3, -inf, inf, 0, unlimited())
                self.assertEqual(snapshot(node), before)
                for key, entry in completed.items():
                    self.assertEqual(player.table[key], entry)
                self.assertNotIn((node.key(), 3), player.table)
                player._search(node, 3, -1., 1., 0, unlimited())
                self.assertEqual(snapshot(node), before)
                resumed = player._search(node, 3, -inf, inf, 0, unlimited())
                fresh = AlphaBetaPlayer(config=self.config)
                fresh._prepare()
                self.assertEqual(resumed, fresh._search(SearchPosition(state), 3, -inf, inf, 0, unlimited()))
                self.assertEqual(state.tobytes(), before[0])

    def test_work_limited_analyze_retains_identical_completed_branches(self):
        state = Position.fixture(position({'D4': 1, 'E5': -2, 'G7': -3})).storage()
        for work in (0, 1, 100, 500, 1000, 5000, 20000):
            config = replace(self.config, node_limit=work, proof_nodes=64)
            reference = AlphaBetaPlayer(config=config, use_compact=False).analyze(state)
            compact = AlphaBetaPlayer(config=config).analyze(state)
            for name in ('action', 'score', 'pv', 'work', 'nodes', 'proof_nodes', 'completed_depth',
                         'selected_depth', 'root_moves_completed', 'selection_source', 'score_bound', 'stop_reason'):
                self.assertEqual(getattr(compact, name), getattr(reference, name), (work, name))

    def test_proof_differential_tactics_depths_budgets_and_unwind(self):
        boards = [load_case(case)[0] for case in CASES]
        for board in boards:
            node = SearchPosition(board.storage())
            before = snapshot(node)
            for depth in (1, 2, 3):
                for work in (0, 1, 2, 17, 65, 129, 10000):
                    config = replace(self.config, proof_depth=depth, proof_nodes=64)
                    budget = Budget(work, 600)
                    try:
                        reference = prove_reference(self.game, board.storage(), config, budget)
                        stop = 1 if reference.get('reason') == 'proof budget' else 0
                    except BudgetExpired:
                        stop = 2
                    score, line, counts = compact_proof(node, depth, 64, work)
                    self.assertEqual(tuple(counts), (budget.work, budget.proof_nodes, stop))
                    if not stop and reference['status'] == 'proven':
                        self.assertEqual((score, list(line[line >= 0])), (reference['score'], reference['pv']))
                    self.assertEqual(snapshot(node), before)
        node = SearchPosition(boards[25].storage())
        before = snapshot(node)
        for depth in range(1, 9):
            for work in range(132):
                budget = Budget(work, 600)
                config = replace(self.config, proof_depth=depth, proof_nodes=100000)
                try:
                    prove_reference(self.game, node, config, budget)
                except BudgetExpired:
                    pass
                self.assertEqual(snapshot(node), before)

    def test_native_proof_draw_boundaries_and_sufficient_budget_oracle(self):
        from intransitive.tests.test_compiled_proof import oracle_score
        for pieces in ({'H8': 3, 'D4': -1}, {'D4': 1, 'E5': -2}, {'I9': 1, 'D4': -2}):
            current = position(pieces)
            other = position({'C3': 1, 'G7': -3})
            for length in (1, 3, 5, NO_CAPTURE_LIMIT, NO_CAPTURE_LIMIT+1):
                history = [other]*(length-1) + [current]
                if length in (3, 5):
                    history[0] = current
                if length == 5:
                    history[2] = current
                board = Position.fixture(current, history=history)
                for reference in (board, board.relabel()):
                    node = SearchPosition(reference.storage())
                    before = snapshot(node)
                    for depth in (1, 2):
                        expected = oracle_score(reference, depth)
                        score, _, counts = compact_proof(node, depth, 100000, 1000000)
                        self.assertEqual(counts[2], 0)
                        self.assertEqual(score, expected)
                        self.assertEqual(snapshot(node), before)

    def test_unexpected_analyze_error_retains_previous_result_and_can_resume(self):
        state = Position.fixture(position({'D4': 1, 'E5': -2, 'G7': -3})).storage()
        player = AlphaBetaPlayer(config=replace(self.config, max_depth=2))
        previous = player.analyze(state)
        player.table.clear()
        player._hints.clear()
        search = player._search
        roots = []
        def interrupted(node, depth, alpha, beta, ply, budget):
            if ply == 0:
                roots.append((node, snapshot(node)))
            if ply == 2:
                raise RuntimeError('injected')
            return search(node, depth, alpha, beta, ply, budget)
        with patch.object(player, '_search', interrupted):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                player.analyze(state)
        self.assertIs(player.last_result, previous)
        for node, before in roots:
            self.assertEqual(snapshot(node), before)
        resumed = player.analyze(state)
        self.assertEqual((resumed.score, resumed.action, resumed.pv),
                         (previous.score, previous.action, previous.pv))

    def test_no_public_transitions_or_cold_compilation_inside_compact_search(self):
        state = self.game.getInitBoard()
        player = AlphaBetaPlayer(config=replace(self.config, max_depth=2, proof_nodes=64))
        player._prepare()
        from intransitive.heuristics.position import validate_state as validate
        with (patch.object(player.game, 'getNextState', side_effect=AssertionError('public transition')),
              patch.object(native_compact_proof, 'compile', side_effect=AssertionError('cold proof')),
              patch('intransitive.heuristics.position.validate_state', wraps=validate) as checked):
            result = player.analyze(state)
        self.assertEqual(checked.call_count, 1)
        self.assertEqual(result.completed_depth, 2)
