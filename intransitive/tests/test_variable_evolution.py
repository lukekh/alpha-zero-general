"""Variable material stays explicit through breeding, journals and native export."""
from dataclasses import replace
import random
import threading
import unittest
from intransitive.evolution.runner import Search, batch, genome, jobs, validate
from intransitive.evolution.strategy import Settings, initialize, next_population
from intransitive.heuristics.config import SearchConfig
from intransitive.heuristics.tuning import Genome, VARIABLE_VERSION
from intransitive.tests import test_evolution as base_tests
from intransitive.tournament.spec import candidate, effective_config

class VariableGenomeTests(unittest.TestCase):
    def test_roundtrip_identity_native_and_override(self):
        for backend in ('python', 'rust'):
            flat=Genome.from_genes({'material':5},backend=backend)
            var=Genome.from_genes({'material':5},backend=backend,variable_material_enabled=True)
            self.assertEqual(Genome.from_json(var.to_json()),var)
            self.assertEqual(var.to_dict()['version'],VARIABLE_VERSION)
            self.assertNotEqual(flat.config_hash,var.config_hash)
            self.assertTrue(var.to_config().variable_material_enabled)
            self.assertEqual(var.to_config().count_weight,5)
            self.assertGreater(var.manifest(implementation_revision='test')['conservative_absolute_bound'],
                               flat.manifest(implementation_revision='test')['conservative_absolute_bound'])
            if backend=='rust':self.assertTrue(var.native_arguments()['variable_material_enabled'])
            else:
                item=candidate('variable',genome=var.to_dict())
                self.assertTrue(item['evaluation']['variable_material_enabled'])
                with self.assertRaises(ValueError):candidate('bad',genome=var.to_dict(),variable_material_enabled=False)
        with self.assertRaises(ValueError):Genome.from_genes(variable_material_enabled=1)
        with self.assertRaises(ValueError):Settings(variable_material_enabled=1)
        with self.assertRaises(ValueError):Settings(initial_material=101)
        with self.assertRaises(ValueError):Genome.from_genes().to_config(SearchConfig(variable_material_enabled=True))

    def test_breeding_random_and_crossover_preserve_mode(self):
        settings=Settings(variable_material_enabled=True,initial_material=5,population=4,
                          recombination_rate=1,mutation_rate=1)
        for algorithm in ('evolution','random'):
            settings=replace(settings,algorithm=algorithm);rng=random.Random(73)
            pop=initialize(rng,settings,set())
            for _ in range(4):
                self.assertTrue(all(g.variable_material_enabled for g in pop))
                self.assertTrue(all(-100<=v<=100 for g in pop for v in g.values))
                pop=next_population(pop,[g.config_hash for g in pop],rng,settings,set())
            with self.assertRaises(ValueError):next_population(pop,[g.config_hash for g in pop],rng,replace(settings,variable_material_enabled=False),set())

class VariableRunTests(base_tests.EvolutionTests):
    # Exercise all existing resume/cache/budget tests with a variable population
    # and a flat incumbent. Tests using explicit flat genomes remain flat controls.
    def spec(self, **overrides):
        return super().spec(**dict(variable_material_enabled=True,initial_material=5,**overrides))

    def test_canonical_games_and_exports_keep_actual_material_mode(self):
        spec=self.spec();validate(spec)
        self.assertFalse(genome(spec['incumbent']).variable_material_enabled)
        self.assertTrue(genome(spec['archive']).variable_material_enabled)
        search=Search(spec,self.path,threading.Event(),preflight=base_tests.safe_preflight)
        pop=[genome(g) for g in search.state['population']]
        for _,_,canonical,_ in jobs(batch(spec,pop,[],0,'search')):
            for item in canonical['candidates']:
                self.assertEqual(effective_config(item,canonical['protocols'][0]).variable_material_enabled,
                                 genome(item['genome']).variable_material_enabled)
        self.run_search(spec)
        state=self.state()
        self.assertTrue(state['exports'])
        for export in state['exports']:
            self.assertTrue(export['config']['variable_material_enabled'])
            self.assertTrue(genome(export['genome']).variable_material_enabled)
