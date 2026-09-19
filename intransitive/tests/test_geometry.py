"""Shared route maps must answer exactly what the per-piece scans answered.

`Geometry` builds one arrival map per piece code instead of rescanning every
enemy for every piece, square and deadline. The reference implementations below
are the code those maps replaced; the tests assert the two agree over a corpus
of reachable positions, so a future change to the traversal rule that forgets
one of the two paths fails here.
"""
import unittest

import numpy as np

from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.heuristics.budget import Budget
from intransitive.heuristics.flood import (
    nearest_distances, passable_masks, route_distances,
)
from intransitive.heuristics.geometry import (
    NO_PREDATOR, Geometry, arrival, captures, code_slot, distance_map, nearest_distance,
    predator_code,
)
from intransitive.tests.test_attribution import random_positions
from intransitive.tests.test_heuristics import position, unlimited


def reference_safe(geometry, piece, square, ply):
    """The per-enemy scan `Geometry.safe` replaced."""
    return not any(captures(enemy.code, piece.code)
                   and arrival(int(enemy.distances[square]), enemy.side, geometry.turn) <= ply
                   for enemy in geometry.own(1 - piece.side))


def reference_deadline(geometry, runner, square, capture):
    """The inline deadline `Geometry.profile` hoisted out of the defender loop."""
    runner_ply = arrival(int(runner.distances[square]), runner.side, geometry.turn)
    return (arrival(1, runner.side, geometry.turn) - 1 if square == runner.square
            else runner_ply + (1 if capture and square != runner.goal else -1))


class PredatorCodeTests(unittest.TestCase):
    def test_every_code_has_exactly_one_predator(self):
        for code in (1, 2, 3, -1, -2, -3):
            predator = predator_code(code)
            self.assertTrue(captures(predator, code), code)
            others = [other for other in (1, 2, 3, -1, -2, -3)
                      if other != predator and captures(other, code)]
            self.assertEqual(others, [], f'{code} has more than one predator code')

    def test_predator_is_the_opposing_side(self):
        for code in (1, 2, 3, -1, -2, -3):
            self.assertLess(code * predator_code(code), 0)


class NearestDistanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.states = random_positions(games=4, plies=30, seed=23)

    def test_matches_the_minimum_over_single_source_maps(self):
        for state in self.states:
            board = state[:, :, 0]
            squares = [int(s) for s in np.flatnonzero(board.ravel())]
            by_code = {}
            for square in squares:
                by_code.setdefault(int(board.flat[square]), []).append(square)
            for code, sources in by_code.items():
                shared = nearest_distance(board, np.array(sources, dtype=np.int16),
                                          len(sources), code)
                expected = np.minimum.reduce(
                    [distance_map(board, source, code) for source in sources])
                np.testing.assert_array_equal(shared, expected)

    def test_unreachable_squares_stay_at_ninety_nine(self):
        # A lone rock walled in by friendly pieces it can never capture.
        state = position({'A1': 1, 'A2': 1, 'B1': 1, 'B2': 1, 'I9': -1})
        board = state[:, :, 0]
        shared = nearest_distance(board, np.array([0], dtype=np.int16), 1, 1)
        self.assertEqual(int(shared[0]), 0)
        self.assertEqual(int(shared[80]), 99)


class FloodFillTests(unittest.TestCase):
    """The bitboard fill must reproduce the queue search it replaced."""

    @classmethod
    def setUpClass(cls):
        cls.states = random_positions(games=4, plies=30, seed=41)

    def test_flood_matches_the_queue_search_for_every_piece(self):
        for state in self.states:
            board = state[:, :, 0]
            masks = passable_masks(board)
            a1 = int(state[:, :, 82:84].flat[2])
            goals = (80, 0) if a1 == 0 else (0, 80)
            for square in np.flatnonzero(board.ravel()):
                square = int(square)
                code = int(board.flat[square])
                goal = goals[int(code < 0)]
                with self.subTest(square=square, code=code):
                    np.testing.assert_array_equal(
                        route_distances(masks, code_slot(code), square, 81),
                        distance_map(board, square, code))
                    np.testing.assert_array_equal(
                        route_distances(masks, code_slot(code), goal, square),
                        distance_map(board, goal, code, removed=square))

    def test_multi_source_flood_matches_the_queue_search(self):
        for state in self.states:
            board = state[:, :, 0]
            masks = passable_masks(board)
            by_code = {}
            for square in np.flatnonzero(board.ravel()):
                by_code.setdefault(int(board.flat[int(square)]), []).append(int(square))
            for code, squares in by_code.items():
                sources = np.array(squares, dtype=np.int16)
                with self.subTest(code=code):
                    np.testing.assert_array_equal(
                        nearest_distances(masks, code_slot(code), sources, len(sources)),
                        nearest_distance(board, sources, len(sources), code))

    def test_passable_masks_admit_exactly_empty_and_prey_squares(self):
        for state in self.states[:6]:
            board = state[:, :, 0]
            masks = passable_masks(board)
            for code in (1, 2, 3, -1, -2, -3):
                slot = code_slot(code)
                for square in range(81):
                    lane, bit = square // 64, 1 << (square % 64)
                    allowed = bool(int(masks[slot, lane]) & bit)
                    occupant = int(board.flat[square])
                    expected = occupant == 0 or captures(code, occupant)
                    self.assertEqual(allowed, expected, (code, square, occupant))


class SafetyMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.states = random_positions(games=5, plies=30, seed=31)

    def test_safe_matches_the_per_enemy_scan_at_every_square_and_ply(self):
        for state in self.states:
            geometry = Geometry(state, unlimited(), routes=True)
            for piece in geometry.pieces:
                for square in range(0, 81, 5):
                    for ply in (0, 1, 2, 3, 5, 9, 20, 200, 1000):
                        with self.subTest(piece=piece.square, square=square, ply=ply):
                            self.assertEqual(geometry.safe(piece, square, ply),
                                             reference_safe(geometry, piece, square, ply))

    def test_a_piece_with_no_predator_left_is_safe_at_any_ply(self):
        # Blue scissors with every Red rock gone: nothing can ever capture it.
        state = position({'D4': 2, 'E5': -2, 'F6': -3, 'A1': 1})
        geometry = Geometry(state, unlimited(), routes=True)
        scissors = geometry.by_square[3 + 9 * 3]
        self.assertEqual(int(geometry.threat_map(scissors.code)[0]), NO_PREDATOR)
        for ply in (0, 10, 500, NO_PREDATOR - 1):
            self.assertTrue(geometry.safe(scissors, 40, ply))
            self.assertTrue(reference_safe(geometry, scissors, 40, ply))

    def test_maps_are_built_only_for_codes_that_are_asked_about(self):
        geometry = Geometry(random_positions(games=1, plies=6, seed=2)[-1],
                            unlimited(), routes=True)
        self.assertEqual(geometry._threat, {})
        piece = geometry.pieces[0]
        geometry.safe(piece, 40, 3)
        self.assertEqual(list(geometry._threat), [piece.code])

    def test_the_map_is_shared_by_every_piece_of_one_code(self):
        geometry = Geometry(random_positions(games=1, plies=8, seed=4)[-1],
                            unlimited(), routes=True)
        by_code = {}
        for piece in geometry.pieces:
            by_code.setdefault(piece.code, []).append(piece)
        for code, pieces in by_code.items():
            if len(pieces) > 1:
                first = geometry.threat_map(pieces[0].code)
                self.assertIs(first, geometry.threat_map(pieces[1].code))


class RunnerProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.states = random_positions(games=4, plies=30, seed=37)

    def test_profile_reproduces_the_inline_squares_and_deadlines(self):
        for state in self.states:
            geometry = Geometry(state, unlimited(), routes=True)
            for runner in geometry.pieces:
                if runner.distance == 99:
                    continue
                squares, deadlines, plies = geometry.profile(runner)
                self.assertEqual(squares, [runner.square] + geometry.route_squares(runner))
                for square, deadline, ply in zip(squares, deadlines, plies):
                    with self.subTest(runner=runner.square, square=square):
                        self.assertEqual(deadline, reference_deadline(geometry, runner, square, True))
                        self.assertEqual(ply, arrival(int(runner.distances[square]),
                                                      runner.side, geometry.turn))

    def test_profile_is_computed_once_per_runner(self):
        geometry = Geometry(self.states[-1], unlimited(), routes=True)
        runner = next(p for p in geometry.pieces if p.distance < 99)
        self.assertIs(geometry.profile(runner), geometry.profile(runner))


if __name__ == '__main__':
    unittest.main()
