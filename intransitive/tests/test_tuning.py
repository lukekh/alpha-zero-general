"""Executable contract for opt-in heuristic candidates (issue #53)."""
from dataclasses import replace
import json
from pathlib import Path
import unittest

import numpy as np

from intransitive.IntransitiveGame import IntransitiveGame
from intransitive.IntransitiveDisplay import parse_move
from intransitive.heuristics import AlphaBetaPlayer, SearchConfig
from intransitive.heuristics.budget import Budget
from intransitive.heuristics.evaluation import Evaluator, HEURISTIC_LIMIT, MATE_THRESHOLD
from intransitive.heuristics.position import SearchPosition
from intransitive.heuristics.pressure import pressure_totals
from intransitive.heuristics.tuning import Genome, GENES, VERSION, DEFAULTS, json_schema, saturation_report
from intransitive.rust_teacher import BINARY, RustTeacher
from intransitive.tests.test_heuristics import position, unlimited

BASE = SearchConfig(max_depth=2, time_limit=60., node_limit=10**9, proof_nodes=0,
                    pressure_cache_entries=8)
FIXTURES = {
    'material': {'D4': 1, 'E4': 1, 'H8': -2},
    'advantage': {'D4': 1, 'E4': 1, 'H7': -2, 'H8': -2},
    'attack': {'G7': 3, 'A9': -1},
    'defence': {'B2': 2, 'D4': -3, 'H8': 1},
    'overload': {'A5': 2, 'C5': -3, 'D1': -3, 'E5': -1},
    'pressure': {'D4': 1, 'E5': -2, 'H8': -3},
}
TERMS = dict(material='piece_count', advantage='piece_advantage', attack='attacking_position',
             defence='defensive_position', overload='overload', pressure='local_pressure')


class GenomeTests(unittest.TestCase):
    def test_roundtrip_normalization_hash_and_checked_in_schema(self):
        a = Genome.from_genes({'pressure': -0.0})
        b = Genome.from_genes()
        self.assertEqual(a, b)
        self.assertEqual(a.config_hash, b.config_hash)
        self.assertEqual(a.to_json(), b.to_json())
        self.assertEqual(Genome.from_json(a.to_json()), a)
        self.assertEqual(b.to_config(), SearchConfig())
        manifest = b.manifest(BASE, implementation_revision='fixture-revision')
        self.assertEqual(manifest, b.manifest(BASE, implementation_revision='fixture-revision'))
        self.assertNotEqual(manifest['manifest_hash'], b.manifest(
            replace(BASE, max_depth=3), implementation_revision='fixture-revision')['manifest_hash'])
        self.assertNotEqual(manifest['manifest_hash'], b.manifest(
            BASE, implementation_revision='another-revision')['manifest_hash'])
        schema = Path(__file__).parents[1] / 'heuristics/module-scales-v2.schema.json'
        self.assertEqual(json.loads(schema.read_text()), json_schema())
        for backend in GENES:
            candidate = Genome.from_genes(backend=backend)
            self.assertEqual(Genome.from_json(candidate.to_json()), candidate)

    def test_reject_unknown_ignored_nonfinite_out_of_range_and_malformed(self):
        for gene in ('count', 'count_weight', 'race_weight', 'pressure_radius', 'predator_zero_bonus',
                     'max_depth', 'attack_enabled', 'unknown'):
            with self.subTest(gene=gene), self.assertRaises(ValueError):
                Genome.from_genes({gene: 1})
        for gene in GENES['python']:
            for value in (float('nan'), float('inf'), -float('inf'), -101, 101, True, '1', None):
                with self.subTest(gene=gene, value=value), self.assertRaises(ValueError):
                    Genome.from_genes({gene: value})
            self.assertNotEqual(Genome.from_genes({gene: 2}).config_hash,
                                Genome.from_genes({gene: 3}).config_hash)
        for text in ('{}', '[]', '{"version": 1}',
                     Genome.from_genes().to_json().replace(VERSION, 'v999'),
                     Genome.from_genes().to_json().replace('"pressure":0.0', '"pressure":0.0,"pressure":1.0')):
            with self.subTest(text=text), self.assertRaises(ValueError):
                Genome.from_json(text)
        for kwargs in ({'race_weight': 1}, {'pressure_radius': 3}, {'predator_zero_bonus': 2},
                       {'attack_enabled': False}, {'count_weight': 50}):
            with self.assertRaises(ValueError):
                Genome.from_genes().to_config(replace(BASE, **kwargs))
        for gene in ('overload',):
            with self.assertRaises(ValueError):
                Genome.from_genes({gene: 0}, backend='rust')
        with self.assertRaises(ValueError):
            Genome.from_genes().native_arguments()
        for backend in (None, [], {}, 'unknown'):
            with self.assertRaises(ValueError):
                Genome.from_genes(backend=backend)
        for genes in (None, [], 1, 'pressure'):
            with self.assertRaises(ValueError):
                Genome.from_json(json.dumps(dict(version=VERSION, backend='rust', genes=genes)))
        native = Genome.from_genes(backend='rust')
        for kwargs in ({'max_depth': 0}, {'max_depth': 33}, {'proof_depth': 3},
                       {'proof_nodes': 65}, {'table_entries': 1000001},
                       {'node_limit': 2**64}, {'time_limit': 1e100}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                native.manifest(replace(BASE, **kwargs), implementation_revision='test')

    def test_each_gene_changes_only_intended_term_and_zero_skips_work(self):
        game = IntransitiveGame()
        for gene, fixture in FIXTURES.items():
            rows = []
            for weight in (0., 1., 2.):
                genome = Genome.from_genes({gene: weight})
                config = genome.to_config(BASE)
                budget = unlimited()
                row = Evaluator(game, config).explain(position(fixture), 0, budget,
                                                     proof={'status': 'unknown'}, diagnostics=False)
                rows.append(row)
                if gene not in ('material', 'advantage'):
                    self.assertEqual(getattr(config, gene + '_enabled'), bool(weight))
                    if weight == 0:
                        self.assertEqual(budget.module_calls[TERMS[gene]], 0)
            term = TERMS[gene]
            self.assertEqual(rows[0]['terms'][term], 0.)
            self.assertNotEqual(rows[1]['terms'][term], 0., gene)
            self.assertAlmostEqual(rows[2]['terms'][term], 2 * rows[1]['terms'][term])
            self.assertAlmostEqual(rows[2]['score'] - rows[1]['score'], rows[1]['terms'][term])
            for other in rows[0]['terms'].keys() - {term}:
                self.assertEqual(rows[0]['terms'][other], rows[2]['terms'][other])
            if gene == 'overload':
                self.assertLess(rows[1]['terms'][term], 0.)

    def test_defaults_scores_choices_and_candidate_cache_isolation(self):
        game = IntransitiveGame()
        state = position(FIXTURES['pressure'])
        old = AlphaBetaPlayer(game, BASE).analyze(state)
        player = AlphaBetaPlayer(game, Genome.from_genes().to_config(BASE))
        default = player.analyze(state)
        self.assertEqual((old.score, old.action, old.completed_depth),
                         (default.score, default.action, default.completed_depth))
        for genes in ({'pressure': 20}, {'advantage': 0}, {'attack': 2},
                      {'defence': 2}, {'overload': 2}, {}):
            player.config = Genome.from_genes(genes).to_config(BASE)
            reused = player.analyze(state)
            fresh = AlphaBetaPlayer(game, player.config).analyze(state)
            self.assertEqual(reused.completed_depth, 2)
            self.assertEqual((reused.score, reused.action), (fresh.score, fresh.action))
            self.assertTrue(game.getValidMoves(state, 0)[reused.action])

    def test_clipping_preflight_and_proof_separation(self):
        entries = {chr(65+x)+str(y+1): 1 for y in range(9) for x in range(9)
                   if (x, y) not in ((0, 0), (8, 8), (0, 8))}
        entries['A9'] = -2
        state = position(entries)
        candidate = Genome.from_genes({'advantage': 100})
        for config in (candidate.to_config(BASE), replace(candidate.to_config(BASE), attack_enabled=True)):
            evaluator = Evaluator(IntransitiveGame(), config)
            budget = unlimited()
            score = evaluator.score(state, 0, budget, proof={'status': 'unknown'})
            row = evaluator.explain(state, 0, unlimited(), proof={'status': 'unknown'}, diagnostics=False)
            self.assertEqual(score, row['score'])
            self.assertEqual(score, HEURISTIC_LIMIT)
            self.assertGreater(row['raw_score'], HEURISTIC_LIMIT)
            self.assertTrue(row['clipped'])
            self.assertEqual(budget.module_calls['heuristic_clipped'], 1)
            self.assertLess(score, MATE_THRESHOLD)
        report = saturation_report(candidate, [state], base=BASE)
        self.assertTrue(report['flagged'])
        self.assertEqual(report['saturation_fraction'], 1.)
        self.assertTrue(candidate.manifest(BASE, implementation_revision='test')['saturation_possible'])
        self.assertTrue(saturation_report(candidate, [], base=BASE)['flagged'])
        win = position({'H8': 3, 'B2': -1})
        proven = Evaluator(IntransitiveGame(), candidate.to_config()).explain(win, 0, unlimited())
        self.assertGreater(proven['score'], MATE_THRESHOLD)
        self.assertNotIn('raw_score', proven)


@unittest.skipUnless(BINARY.exists(), 'Build rust_teacher with cargo build --release first')
class NativeGenomeTests(unittest.TestCase):
    def test_supported_candidates_features_capture_undo_goals_and_history(self):
        game = IntransitiveGame()
        state = position(FIXTURES['pressure'])
        compact = SearchPosition(state)
        compact.push(parse_move('D4 E5'))
        captured = compact.export()
        compact.pop()
        np.testing.assert_array_equal(compact.export(), state)
        from intransitive.tests.test_draws import load_history
        from intransitive.IntransitiveLogicNumba import Board
        repeated = load_history(Board(), [state[:, :, 0]] * 5)
        history = position({'C3': 1, 'G7': -1})
        side = 0
        for move in ('C3 C4', 'G7 G6', 'C4 C3', 'G6 G7', 'C3 C4', 'G7 G6', 'C4 C3'):
            history, side = game.getNextState(history, side, parse_move(move))
        fixtures = [state, captured, compact.export(), history,
                    position(FIXTURES['pressure'], a1=1), repeated,
                    position({'I9': 3, 'B2': -1})]
        with RustTeacher() as rust:
            for weight in (0., 1., 10., 20.):
                genome = Genome.from_genes({'pressure': weight}, backend='rust')
                config = genome.to_config(BASE)
                for fixture in fixtures:
                    side = int(fixture[:, :, 82:84].flat[1])
                    native = rust.inspect(fixture, radius=4, weight=weight)
                    np.testing.assert_array_equal(native['pressure'], pressure_totals(fixture[:, :, 0], 4))
                    expected = Evaluator(game, config).score(fixture, side, unlimited(), proof={'status': 'unknown'})
                    if native['reason'] == 'ongoing':
                        self.assertAlmostEqual(native['score'], expected, places=9)
                    else:
                        # inspect exposes ordinary features; search owns terminal handling.
                        self.assertTrue(game.getGameEnded(fixture, side).any())
                        self.assertFalse(rust.analyze(fixture, **genome.native_arguments(BASE))['complete'])
                    if native['reason'] != 'ongoing':
                        continue
                    # Same process, changing candidates: no stale native TT or eval cache.
                    native = rust.analyze(fixture, **genome.native_arguments(BASE))
                    python = AlphaBetaPlayer(game, config).analyze(fixture)
                    self.assertTrue(native['complete'])
                    self.assertEqual(native['completed_depth'], 2)
                    self.assertAlmostEqual(native['score'], python.score, places=9)
                    self.assertTrue(game.getValidMoves(fixture, side)[native['action']])
                    if native['action'] != python.action:
                        child = SearchPosition(fixture)
                        child.push(native['action'])
                        engine = AlphaBetaPlayer(game, config)
                        engine._prepare()
                        value, _ = engine._search(child, 1, -float('inf'), float('inf'), 1, unlimited())
                        self.assertAlmostEqual(-value, python.score, places=9)


if __name__ == '__main__':
    unittest.main()
