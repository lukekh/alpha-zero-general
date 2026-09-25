"""Proven endgame positions, and the search's duty to find them.

Every case is certified by `tactical_oracle.tactical_proof`, an exhaustive
AND/OR proof over the independent reference rules, so what these positions are
worth is settled without the evaluator, the production search, or any node or
time cutoff. Each test re-derives that proof before asking the engine anything:
a rules change fails on the certification, not on the engine, and the two
failures mean different things.

The suite covers the endgame shapes worth knowing about. A free runner taking
the only fastest route. A capture that is forced because the victim stands on
the single square a step from the corner. Defence by taking the runner with the
type that beats it, and defence by occupying one's own corner with a type the
runner cannot take -- which holds only while a spare piece supplies waiting
moves, since a lone blocker must vacate and lose. Positions where the defender
is the wrong type entirely, so the answer is to ignore it and race. Like against
like, where neither can capture and the tempo alone decides, including in the
middle of the board where guarding the corner looks natural and throws the game.
And transpositions of several of those with pieces added in far regions, proved
to leave the answer untouched.

Each case runs twice, unchanged and with the colours exchanged. The symmetry
group also rotates piece types cyclically, so `endgames.json` records that every
case was certified under all twelve; a matchup proved for one pairing therefore
holds for the other two.
"""

import json
import unittest
from dataclasses import replace
from pathlib import Path

from intransitive.heuristics import AlphaBetaPlayer, SearchConfig
from intransitive.heuristics.kernels import warm_search_kernels
from intransitive.tests.reference_rules import (Position, action_between, map_action,
                                                position)
from intransitive.tests.tactical_oracle import move_name, tactical_proof

DATA = json.loads((Path(__file__).parents[1] / 'heuristics' / 'endgames.json').read_text())
# Depth comes from each case's own proof horizon: asking for a win inside five
# plies while searching three is asking the engine to see what it cannot.
SEARCH = SearchConfig(time_limit=60., node_limit=2_000_000_000)
SYMMETRIES = ((0, 'original_colours'), (6, 'exchanged_colours'))


def solving_moves(node, objective, plies):
    """Every legal move meeting the objective, by proof over the reference rules."""
    good = []
    for action in sorted(node.legal()):
        if objective == 'win':
            ok = tactical_proof(node, node.player, plies, first=action)
        else:
            ok = not tactical_proof(node.move(action)[0], 1 - node.player, plies - 1)
        if ok:
            good.append(action)
    return good


class EndgameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warm_search_kernels()

    def test_fixture_is_well_formed(self):
        self.assertEqual(DATA['schema'], 'intransitive-endgames-1')
        cases = DATA['cases']
        self.assertEqual(len({c['id'] for c in cases}), len(cases))
        for trait in ('simple', 'offence', 'defence', 'race', 'like-vs-like',
                      'wrong-defender', 'centre', 'zugzwang', 'transposition'):
            self.assertTrue(any(trait in c['traits'] for c in cases), trait)
        for case in cases:
            self.assertEqual(case['expected'] is None, case['solutions'] == 0, case['id'])
            self.assertLessEqual(case['solutions'], 1, case['id'])

    def test_transpositions_match_the_case_they_transpose(self):
        """An added-piece variant must keep its base case's answer, not merely have one."""
        by_id = {c['id']: c for c in DATA['cases']}
        variants = [c for c in DATA['cases'] if c['id'].endswith('__far_pieces')]
        self.assertGreaterEqual(len(variants), 5)
        for variant in variants:
            base = by_id[variant['id'].removesuffix('__far_pieces')]
            self.assertEqual(variant['expected'], base['expected'], variant['id'])
            self.assertEqual(variant['objective'], base['objective'], variant['id'])
            self.assertGreater(len(variant['pieces']), len(base['pieces']), variant['id'])
            self.assertEqual({s: p for s, p in variant['pieces'].items()
                              if s in base['pieces']}, base['pieces'], variant['id'])


def endgame_test(case, symmetry):
    def test(self):
        node = Position.fixture(position(case['pieces']), player=case['player'])
        node = node.transform(symmetry)
        reason, _ = node.terminal()
        self.assertEqual(reason, 'ongoing', f"{case['id']} is already over")

        proved = solving_moves(node, case['objective'], case['plies'])
        detail = (f"{case['id']}: {case['why']} Proof gives "
                  f"{[move_name(a) for a in proved]}.")
        self.assertEqual(len(proved), case['solutions'], detail)
        if case['expected'] is None:
            # A proven loss: no move holds, so there is nothing to require of
            # the engine. The value is the standing proof that it is lost.
            return
        want = map_action(action_between(*case['expected'].split('-')), symmetry)
        self.assertEqual([move_name(a) for a in proved], [move_name(want)], detail)

        result = AlphaBetaPlayer(
            config=replace(SEARCH, max_depth=case['plies'])).analyze(node.storage())
        self.assertEqual(result.action, want,
                         f"{detail} Engine played {move_name(result.action)} "
                         f"at depth {result.completed_depth}, score {result.score}.")
    test.__doc__ = f"{case['id']} [{symmetry}]: {case['why']}"
    return test


for _case in DATA['cases']:
    for _symmetry, _suffix in SYMMETRIES:
        setattr(EndgameTests, f"test_{_case['id']}_{_suffix}",
                endgame_test(_case, _symmetry))


if __name__ == '__main__':
    unittest.main()
