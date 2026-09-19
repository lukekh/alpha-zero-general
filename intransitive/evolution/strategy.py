"""Seeded, bounded real-valued evolution with an explicit atom at zero."""
from dataclasses import asdict, dataclass
from itertools import combinations
import math

from ..heuristics.tuning import BOUNDS, Genome


@dataclass(frozen=True)
class Settings:
    seed: int = 55
    algorithm: str = 'evolution'
    population: int = 4
    generations: int = 3
    elites: int = 1
    tournament_size: int = 2
    mutation_rate: float = .5
    mutation_sigma: float = .15
    toggle_rate: float = .15
    recombination_rate: float = .5
    random_off_rate: float = .5
    near_default_fraction: float = .5
    hall_size: int = 2
    search_positions: int = 2
    validation_positions: int = 1
    min_completed_depth: int = 1
    resident_engines: int = 4
    match_workers: int = 1  # Concurrent games; needs two resident engines each.
    max_games: int = 200
    max_nodes: int = 100_000_000
    max_seconds: float = 300.
    variable_material_enabled: bool = False
    initial_material: float = 100.

    def __post_init__(self):
        if type(self.variable_material_enabled) is not bool:
            raise ValueError('variable_material_enabled must be boolean')
        Genome.from_genes({'material': self.initial_material})
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if self.algorithm not in ('evolution', 'random'):
            raise ValueError('algorithm must be evolution or random')
        for name in ('population', 'generations', 'elites', 'tournament_size', 'hall_size',
                     'search_positions', 'validation_positions', 'min_completed_depth', 'max_games',
                     'max_nodes', 'match_workers'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.population < 2 or self.elites >= self.population or self.tournament_size > self.population:
            raise ValueError('Need population >= 2, elites < population, tournament_size <= population')
        if type(self.resident_engines) is not int or not 2 <= self.resident_engines <= 24:
            raise ValueError('resident_engines must be in [2, 24]')
        # Each concurrent match holds two engines for its whole game. Covering
        # that up front is what keeps engine acquisition wait-free, so matches
        # cannot deadlock holding one engine each.
        if 2 * self.match_workers > self.resident_engines:
            raise ValueError('resident_engines must cover two per concurrent match')
        # Slack above 2*workers is what lets warm engines stay cached between
        # games; without it every match pays process startup again.
        if self.match_workers > 1 and self.resident_engines == 2 * self.match_workers:
            raise ValueError('resident_engines must exceed two per concurrent match')
        for name in ('mutation_rate', 'toggle_rate', 'recombination_rate',
                     'random_off_rate', 'near_default_fraction'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be finite in [0, 1]')
        for name in ('mutation_sigma', 'max_seconds'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')

    def to_dict(self):
        return asdict(self)


def random_genome(rng, settings):
    return Genome.from_genes({name: 0. if rng.random() < settings.random_off_rate
                              else rng.uniform(low, high) for name, (low, high) in BOUNDS.items()},
                             variable_material_enabled=settings.variable_material_enabled)


def mutate(parent, rng, settings):
    genes = parent.to_dict()['genes'].copy()
    for name, value in genes.items():
        if rng.random() >= settings.mutation_rate:
            continue
        low, high = BOUNDS[name]
        if rng.random() < settings.toggle_rate:
            # Zero is an explicit state, with a uniform reactivation distribution.
            value = rng.uniform(low, high) if value == 0 else 0.
        else:
            # Additive mutation can cross zero in either direction. Sigma is
            # expressed as a fraction of the signed interval width.
            step = rng.gauss(0., 1.) * settings.mutation_sigma * (high-low)
            value = max(low, min(high, value + step))
        genes[name] = value
    return Genome.from_genes(genes, backend=parent.backend,
                             variable_material_enabled=parent.variable_material_enabled)


def initialize(rng, settings, excluded):
    output, seen = [], set(excluded)
    for attempt in range(10000):
        near = len(output) < math.ceil(settings.population * settings.near_default_fraction)
        genome = mutate(Genome.from_genes({'material': settings.initial_material},
                        variable_material_enabled=settings.variable_material_enabled), rng, settings) if near and attempt < 100 else random_genome(rng, settings)
        if genome.config_hash not in seen:
            output.append(genome)
            seen.add(genome.config_hash)
        if len(output) == settings.population:
            return output
    raise ValueError('Initialization distribution cannot produce enough distinct candidates')


def next_population(population, ranked_hashes, rng, settings, excluded):
    if any(g.variable_material_enabled != settings.variable_material_enabled for g in population):
        raise ValueError('Population material mode differs from frozen settings')
    by_id = {g.config_hash: g for g in population}
    rank = {identity: i for i, identity in enumerate(ranked_hashes)}
    output = [by_id[h] for h in ranked_hashes[:settings.elites]] if settings.algorithm == 'evolution' else []
    seen = set(excluded) | {g.config_hash for g in output}

    def parent():
        return min(rng.sample(population, settings.tournament_size), key=lambda g: rank[g.config_hash])

    for attempt in range(10000):
        if settings.algorithm == 'random' or attempt >= 100:
            child = random_genome(rng, settings)
        else:
            child = parent()
            if rng.random() < settings.recombination_rate:
                other = parent()
                child = Genome('python', tuple(a if rng.random() < .5 else b
                                               for a, b in zip(child.values, other.values)), child.variable_material_enabled)
            child = mutate(child, rng, settings)
        if child.config_hash not in seen:
            output.append(child)
            seen.add(child.config_hash)
        if len(output) == settings.population:
            return output
    raise ValueError('Mutation/random distribution cannot produce enough distinct candidates')


def diversity(population):
    widths = [high-low for low, high in BOUNDS.values()]
    distances = [math.sqrt(sum(((x-y)/width)**2 for x, y, width in zip(a.values, b.values, widths)) / len(widths))
                 for a, b in combinations(population, 2)]
    return dict(unique=len({g.config_hash for g in population}),
                mean_normalized_distance=sum(distances) / len(distances) if distances else 0.,
                off_fraction=sum(v == 0 for g in population for v in g.values) / (len(population)*len(widths)))
