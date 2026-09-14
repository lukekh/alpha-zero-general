"""Differential proof certificates and deterministic interruption/accounting."""
from functools import lru_cache
from unittest.mock import patch
import unittest
import numpy as np

from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.heuristics import AlphaBetaPlayer, SearchConfig
from intransitive.heuristics.budget import Budget, BudgetExpired
from intransitive.heuristics.evaluation import MATE, MATE_THRESHOLD
from intransitive.heuristics.proof import native_proof, warm_proof_kernel
from intransitive.heuristics.search import prove, prove_reference
from intransitive.tests.reference_rules import Position, position
from intransitive.tests.test_tactics import CASES
from intransitive.tests.tactical_oracle import load_case


def oracle_score(board, depth):
    @lru_cache(None)
    def visit(node, left, ply):
        reason, rewards = node.terminal()
        if reason != 'ongoing':
            if rewards[node.player] == 1:
                return MATE - ply
            if rewards[1 - node.player] == 1:
                return -MATE + ply
            return 0.
        if not left:
            return 0.
        return max(-visit(node.move(a)[0], left - 1, ply + 1) for a in node.legal())
    return visit(board, depth, 0)


class CompiledProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = IntransitiveGame()
        warm_proof_kernel()
        cls.boards = [load_case(case)[0] for case in CASES]
        for pieces in ({'H8': 3, 'I9': -1, 'D4': -2},
                       {'D4': 1, 'E5': -2}, {'H8': 3, 'B2': -1},
                       {'D4': 1}, {'I9': 1, 'D4': -2},
                       {'G7': 3, 'I1': -1}):
            board = Position.fixture(position(pieces))
            cls.boards.extend((board, board.transform(6)))
        quiet = Position.fixture(position({'D4': 1, 'F6': -3}))
        different = position({'C3': 1, 'G7': -3})
        for length in (3, 4, 5, 79, 80, 81):
            # Structurally valid modelling histories, including second/third
            # occurrences and the no-capture boundary.
            history = [different] * (length - 1) + [quiet.pieces]
            if length in (3, 5):
                history[0] = quiet.pieces
            if length == 5:
                history[2] = quiet.pieces
            cls.boards.append(Position.fixture(quiet.pieces, history=history))
        rng = np.random.default_rng(38)
        board = Position.initial()
        for _ in range(30):
            cls.boards.append(board)
            if not board.legal():
                break
            board, _ = board.move(int(rng.choice(sorted(board.legal()))))

    def compare(self, board, depth, limit, work=10**8):
        state = board.storage()
        before = state.tobytes()
        config = SearchConfig(proof_depth=depth, proof_nodes=limit)
        budget = Budget(work, 600)
        try:
            reference = prove_reference(self.game, state, config, budget)
            expected_stop = 1 if reference.get('reason') == 'proof budget' else 0
        except BudgetExpired:
            reference, expected_stop = None, 2
        for specialised in (False, True):
            score, line, counts = native_proof(state, depth, limit, work, specialised)
            self.assertEqual(int(counts[2]), expected_stop)
            self.assertEqual(int(counts[0]), budget.work)
            self.assertEqual(int(counts[1]), budget.proof_nodes)
            if expected_stop == 0:
                if reference['status'] == 'proven':
                    self.assertEqual(score, reference['score'])
                    self.assertEqual(list(line[line >= 0]), reference['pv'])
                else:
                    self.assertLessEqual(abs(score), MATE_THRESHOLD)
        self.assertEqual(state.tobytes(), before)

    def test_differential_outcomes_pvs_and_exact_work(self):
        for board in self.boards:
            for depth in (1, 2, 3):
                with self.subTest(player=board.player, depth=depth, pieces=board.pieces):
                    self.compare(board, depth, 64)

    def test_sufficient_budget_matches_independent_minimax(self):
        for board in self.boards[:112]:
            for depth in (1, 2):
                expected = oracle_score(board, depth)
                score, _, counts = native_proof(board.storage(), depth, 100000, 1000000, True)
                self.assertEqual(counts[2], 0)
                self.assertEqual(score, expected)
        # Three-ply clear runs, including counter-races and colour exchange.
        for board in self.boards[20:40:4]:
            score, _, counts = native_proof(board.storage(), 3, 100000, 1000000, True)
            self.assertEqual(counts[2], 0)
            self.assertEqual(score, oracle_score(board, 3))

    def test_draw_histories_and_official_win_precedence(self):
        for pieces in ({"H8": 3, "I9": -1, "D4": -2},
                       {"D4": 1, "E5": -2}, {"I9": 1, "D4": -2}):
            for length in (79, 80, 81):
                current = position(pieces)
                history = [position({"C3": 1, "G7": -3})] * (length - 1) + [current]
                board = Position.fixture(current, history=history)
                for symmetry in (0, 6):
                    transformed = board.transform(symmetry)
                    self.compare(transformed, 2, 100000)
                    score, _, counts = native_proof(transformed.storage(), 2, 100000, 1000000, True)
                    self.assertEqual(counts[2], 0)
                    self.assertEqual(score, oracle_score(transformed, 2))
        for board in self.boards[112:118]:
            self.compare(board, 2, 100000)
            score, _, counts = native_proof(board.storage(), 2, 100000, 1000000, True)
            self.assertEqual(counts[2], 0)
            self.assertEqual(score, oracle_score(board, 2))

    def test_all_supported_depths_and_larger_budget_fallback(self):
        for board in self.boards[::10]:
            for depth in range(1, 9):
                self.compare(board, depth, 64)
        board = self.boards[25]
        config = SearchConfig(proof_depth=3, proof_nodes=100000)
        with patch('intransitive.heuristics.proof.native_proof', side_effect=AssertionError('unbounded batch')):
            self.assertEqual(prove(self.game, board.storage(), config, Budget(10**7, 600)),
                             prove_reference(self.game, board.storage(), config, Budget(10**7, 600)))

    def test_every_small_work_cap_and_proof_exhaustion(self):
        for board in (self.boards[25], self.boards[100]):
            for work in range(132):
                self.compare(board, 2, 64, work)
            for limit in (1, 2, 3, 16, 63, 64):
                self.compare(board, 3, limit)

    def test_disabled_cold_and_timed_out_proofs_are_sound(self):
        state = self.boards[0].storage()
        for config in (SearchConfig(proof_nodes=0), SearchConfig(proof_depth=0)):
            self.assertEqual(prove(self.game, state, config, Budget(0, 0))['reason'], 'disabled')
        with (patch('intransitive.heuristics.proof.READY', False),
              patch('intransitive.heuristics.proof.native_proof', side_effect=AssertionError('cold compile'))):
            self.assertEqual(prove(self.game, state, SearchConfig(), Budget(1000, 600))['status'], 'proven')
        times = iter((0., 0., 2.))
        budget = Budget(1000, 1., clock=lambda: next(times))
        with self.assertRaisesRegex(BudgetExpired, 'time'):
            prove(self.game, state, SearchConfig(), budget)
        self.assertGreater(budget.proof_nodes, 0)
        self.assertLessEqual(budget.proof_nodes, 64)

    def test_warmed_production_call_cannot_compile_another_signature(self):
        state = self.boards[25].storage()
        with patch.object(native_proof, "compile", side_effect=AssertionError("compile inside deadline")):
            prove(self.game, state, SearchConfig(), Budget(1000, 600))

    def test_work_interruption_retains_identical_completed_root_work(self):
        state = self.boards[25].storage()
        for work in (0, 1, 100, 500, 1000, 5000):
            config = SearchConfig(max_depth=3, node_limit=work, time_limit=600)
            native = AlphaBetaPlayer(config=config).analyze(state)
            with patch('intransitive.heuristics.search.prove', prove_reference):
                reference = AlphaBetaPlayer(config=config).analyze(state)
            for name in ('action', 'score', 'pv', 'work', 'nodes', 'proof_nodes',
                         'completed_depth', 'selected_depth', 'selection_source',
                         'root_moves_completed', 'stop_reason'):
                self.assertEqual(getattr(native, name), getattr(reference, name), name)


if __name__ == '__main__':
    unittest.main()
