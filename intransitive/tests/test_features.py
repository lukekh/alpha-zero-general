"""Compiled features must match their references in value and in charged work.

`evaluation` keeps `attacking_position_reference` and `coverage_reference` as
the readable definitions. The compiled versions in `features` drop work those
cannot — attack walks eight neighbours instead of the shortest-path DAG and the
enemy list, defence shares each runner's route across every defender — so the
two implementations can diverge. These tests are the guard, and they check the
budget as well as the answer: a cheaper kernel that also charged less would
silently deepen every work-limited search.
"""
import unittest

from intransitive.heuristics.config import SearchConfig
from intransitive.heuristics.evaluation import (
    attacking_position, attacking_position_reference, coverage, coverage_reference,
    defensive_position,
)
from intransitive.heuristics.geometry import Geometry
from intransitive.tests.test_attribution import random_positions
from intransitive.tests.test_heuristics import position, unlimited


CONFIG = SearchConfig(attack_enabled=True, defence_enabled=True, proof_depth=0)


def plies(duties):
    """Collapse a coverage map to the earliest ply per defender."""
    return {runner: {defender: min(reply['ply'] for reply in replies)
                     for defender, replies in defenders.items()}
            for runner, defenders in duties.items()}


class CompiledFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.states = random_positions(games=6, plies=40, seed=53)

    def test_attack_matches_its_reference_in_value_and_work(self):
        for state in self.states:
            for side in (0, 1):
                compiled_budget, reference_budget = unlimited(), unlimited()
                compiled = Geometry(state, compiled_budget, routes=True)
                reference = Geometry(state, reference_budget, routes=True)
                with self.subTest(side=side):
                    self.assertEqual(attacking_position(compiled, side, CONFIG),
                                     attacking_position_reference(reference, side, CONFIG))
                    self.assertEqual(compiled_budget.work, reference_budget.work)

    def test_coverage_matches_its_reference_in_value_and_work(self):
        for state in self.states:
            for side in (0, 1):
                compiled_budget, reference_budget = unlimited(), unlimited()
                compiled = Geometry(state, compiled_budget, routes=True)
                reference = Geometry(state, reference_budget, routes=True)
                with self.subTest(side=side):
                    self.assertEqual(plies(coverage(compiled, side)),
                                     plies(coverage_reference(reference, side)))
                    self.assertEqual(compiled_budget.work, reference_budget.work)

    def test_defence_is_unchanged_by_the_compiled_coverage(self):
        for state in self.states:
            for side in (0, 1):
                geometry = Geometry(state, unlimited(), routes=True)
                duties = coverage_reference(Geometry(state, unlimited(), routes=True), side)
                expected = sum(min(1., sum(1. / (1 + min(r['ply'] for r in replies))
                                           for replies in defenders.values()))
                               for defenders in duties.values())
                goal = geometry.goals[1 - side]
                blocker = geometry.by_square.get(goal)
                if blocker is not None and blocker.side == side and geometry.safe(blocker, goal, 2):
                    expected += min(1., sum(abs(p.code) == abs(blocker.code)
                                            for p in geometry.own(1 - side)))
                with self.subTest(side=side):
                    self.assertAlmostEqual(defensive_position(geometry, side, CONFIG),
                                           min(4., expected), places=12)

    def test_the_pair_matrix_is_memoized_per_defending_side(self):
        geometry = Geometry(self.states[-1], unlimited(), routes=True)
        first = geometry.plies(0)
        self.assertIs(first, geometry.plies(0))
        self.assertIsNot(first, geometry.plies(1))

    def test_a_second_pass_over_the_same_side_charges_nothing(self):
        budget = unlimited()
        geometry = Geometry(self.states[-1], budget, routes=True)
        coverage(geometry, 0)
        after_first = budget.work
        coverage(geometry, 0)
        self.assertEqual(budget.work, after_first)

    def test_capture_half_is_one_for_any_safe_adjacent_capture(self):
        # Blue scissors on D4 takes Red paper on E5; nothing answers it.
        state = position({'D4': 2, 'E5': -3, 'A2': 1, 'I8': -1})
        geometry = Geometry(state, unlimited(), routes=True)
        value = attacking_position(geometry, 0, CONFIG)
        reference = attacking_position_reference(
            Geometry(state, unlimited(), routes=True), 0, CONFIG)
        self.assertEqual(value, reference)
        self.assertGreaterEqual(value, 1.)

    def test_a_side_with_no_pieces_scores_zero(self):
        # Runners still appear, with nobody able to answer them, exactly as the
        # reference reports an empty defender map rather than omitting the row.
        state = position({'D4': 2, 'E5': 3})
        geometry = Geometry(state, unlimited(), routes=True)
        self.assertEqual(attacking_position(geometry, 1, CONFIG), 0.)
        self.assertEqual(coverage(geometry, 1),
                         coverage_reference(Geometry(state, unlimited(), routes=True), 1))
        self.assertEqual(defensive_position(geometry, 1, CONFIG), 0.)


if __name__ == '__main__':
    unittest.main()


class RunnerPressureTests(unittest.TestCase):
    """The unproven runner term: an estimate, priced as an ordinary module.

    It reads the same idea the clear-run certificate proves, but cheaply and
    approximately, so these tests pin its shape rather than its truth.
    """

    ON = SearchConfig(attack_enabled=False, defence_enabled=False, proof_depth=0,
                      runner_enabled=True, runner_weight=1.)

    def test_a_clear_corridor_scores_on_distance(self):
        from intransitive.heuristics.evaluation import runner_pressure
        # Blue paper two steps from I9 with nothing of Red's in the box.
        state = position({'G7': 3, 'A5': -1})
        geometry = Geometry(state, unlimited(), routes=True)
        self.assertAlmostEqual(runner_pressure(geometry, 0, self.ON), 1. / 3.)

    def test_one_answering_piece_costs_a_move(self):
        from intransitive.heuristics.evaluation import runner_pressure
        # Red scissors inside the box answers Blue paper, so the runner is
        # priced as though it loses a move going around.
        state = position({'G7': 3, 'H8': -2, 'A5': -1})
        geometry = Geometry(state, unlimited(), routes=True)
        self.assertAlmostEqual(runner_pressure(geometry, 0, self.ON), 1. / 4.)

    def test_two_answering_pieces_score_nothing(self):
        from intransitive.heuristics.evaluation import runner_pressure
        state = position({'G7': 3, 'H8': -2, 'G8': -2, 'A5': -1})
        geometry = Geometry(state, unlimited(), routes=True)
        self.assertEqual(runner_pressure(geometry, 0, self.ON), 0.)

    def test_only_the_nearest_piece_of_each_type_counts(self):
        from intransitive.heuristics.evaluation import runner_pressure
        near = position({'G7': 3, 'A5': -1})
        far = position({'G7': 3, 'B2': 3, 'A5': -1})
        self.assertEqual(
            runner_pressure(Geometry(near, unlimited(), routes=True), 0, self.ON),
            runner_pressure(Geometry(far, unlimited(), routes=True), 0, self.ON))

    def test_a_piece_already_on_the_goal_adds_nothing(self):
        from intransitive.heuristics.evaluation import runner_pressure
        state = position({'I9': 3, 'A5': -1})
        geometry = Geometry(state, unlimited(), routes=True)
        self.assertEqual(runner_pressure(geometry, 0, self.ON), 0.)

    def test_the_module_is_off_by_default(self):
        self.assertFalse(SearchConfig().runner_enabled)
