"""Opt-in, versioned module-scale contract; no optimizer or default migration."""
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path

from .config import SearchConfig
from .evaluation import HEURISTIC_LIMIT, MATE_THRESHOLD

VERSION = 'intransitive-module-scales-v2'
BACKEND_VERSIONS = {'python': 'python-heuristics-signed-v3', 'rust': 'rust-teacher-routes-v2'}
BOUNDS = {name: (-100., 100.) for name in ('material', 'advantage', 'attack', 'defence', 'overload', 'pressure')}
DEFAULTS = {'material': 100., 'advantage': 23.967050360966205,
            'attack': 25.714516982666414, 'defence': 32.5643023919054, 'overload': 0., 'pressure': 0.}
GENES = {'python': tuple(BOUNDS), 'rust': tuple(g for g in BOUNDS if g != 'overload')}
# These are fixed outside the signed genome, including dormant module weights and pressure geometry.
EVALUATION_FIELDS = tuple(name for name in SearchConfig().to_dict()
                          if name.endswith(('_weight', '_bonus')) or name in (
                              'evaluator_version', 'attack_enabled', 'defence_enabled',
                              'overload_enabled', 'pressure_enabled', 'pressure_radius'))


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return sha256(canonical_json(value).encode('utf-8')).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


@dataclass(frozen=True)
class Genome:
    """Immutable effective coefficients, with signed, independently tunable material and module scales.

    Use from_genes for partial overrides, from_json for strict complete records.
    Zero disables optional modules; enable switches are not independent genes.
    """
    backend: str
    values: tuple

    def __post_init__(self):
        if not isinstance(self.backend, str) or self.backend not in GENES:
            raise ValueError(f'Unsupported backend: {self.backend}')
        if not isinstance(self.values, tuple) or len(self.values) != len(GENES[self.backend]):
            raise ValueError('Expected one value per supported gene')
        normalized = []
        for name, value in zip(GENES[self.backend], self.values):
            low, high = BOUNDS[name]
            if (type(value) not in (int, float) or not low <= value <= high
                    or not math.isfinite(value)):
                raise ValueError(f'{name} must be finite in [{low}, {high}]')
            normalized.append(float(value) if value else 0.)
        object.__setattr__(self, 'values', tuple(normalized))

    @classmethod
    def from_genes(cls, genes=None, *, backend='python'):
        if not isinstance(backend, str) or backend not in GENES:
            raise ValueError(f'Unsupported backend: {backend}')
        if genes is None:
            genes = {}
        if not isinstance(genes, dict):
            raise ValueError('genes must be an object')
        unsupported = set(genes) - set(GENES[backend])
        if unsupported:
            raise ValueError(f'Unsupported or ignored genes for {backend}: {sorted(map(str, unsupported))}')
        return cls(backend, tuple(genes.get(name, DEFAULTS[name]) for name in GENES[backend]))

    @classmethod
    def from_json(cls, text):
        data = json.loads(text, object_pairs_hook=_object)
        if not isinstance(data, dict) or set(data) != {'version', 'backend', 'genes'}:
            raise ValueError('Expected exactly version, backend and genes')
        if data['version'] != VERSION:
            raise ValueError('Unsupported genome version')
        if not isinstance(data['genes'], dict):
            raise ValueError('genes must be an object')
        genome = cls.from_genes(data['genes'], backend=data['backend'])
        if set(data['genes']) != set(GENES[genome.backend]):
            raise ValueError('Serialized genomes must include every supported gene')
        return genome

    def to_dict(self):
        return dict(version=VERSION, backend=self.backend,
                    genes=dict(zip(GENES[self.backend], self.values)))

    def to_json(self):
        return canonical_json(self.to_dict())

    @property
    def config_hash(self):
        """Evaluation identity; use manifest_hash for search limits/provenance too."""
        config = self.to_config()
        return digest(dict(genome=self.to_dict(), backend_version=BACKEND_VERSIONS[self.backend],
                           evaluation={k: getattr(config, k) for k in EVALUATION_FIELDS}))

    def to_config(self, base=None):
        base = base or SearchConfig()
        defaults = SearchConfig()
        changed = [name for name in EVALUATION_FIELDS if getattr(base, name) != getattr(defaults, name)]
        if changed:
            raise ValueError(f'Base must use fixed evaluation defaults: {changed}')
        values = dict(DEFAULTS, **self.to_dict()['genes'])
        updates = {'count_weight': values['material'], 'advantage_weight': values['advantage']}
        for name in ('attack', 'defence', 'overload', 'pressure'):
            updates[name + '_enabled'] = bool(values[name])
            # Keep legacy dormant weights so the default is exactly SearchConfig().
            updates[name + '_weight'] = values[name] or getattr(defaults, name + '_weight')
        return replace(base, **updates)

    def native_arguments(self, base=None):
        """Validated RustTeacher.analyze arguments; fresh search isolates candidates.

        Rust has its own fixed ordering/PVS policy and node accounting. Python
        optimization switches are not a native contract or a parity claim.
        """
        if self.backend != 'rust':
            raise ValueError('Native execution requires a rust genome')
        config = self.to_config(base)
        if not 1 <= config.max_depth <= 32 or config.proof_depth > 2 or config.proof_nodes > 64:
            raise ValueError('Rust requires depth 1..32, proof_depth <= 2 and proof_nodes <= 64')
        if config.table_entries > 1_000_000:
            raise ValueError('Rust requires table_entries <= 1000000')
        # The wire protocol uses unsigned 64-bit integers and millisecond time.
        if config.node_limit >= 2**64 or config.time_limit >= 2**64 / 1000:
            raise ValueError('Native time/work limits exceed the unsigned 64-bit protocol')
        return dict(depth=config.max_depth, seconds=config.time_limit,
                    node_limit=config.node_limit, radius=config.pressure_radius,
                    weight=config.pressure_weight if config.pressure_enabled else 0.,
                    material=config.count_weight, advantage=config.advantage_weight,
                    attack=config.attack_weight if config.attack_enabled else 0.,
                    defence=config.defence_weight if config.defence_enabled else 0.,
                    proof_depth=config.proof_depth, proof_nodes=config.proof_nodes,
                    table_entries=config.table_entries, reuse=False)

    def manifest(self, base=None, *, implementation_revision):
        """Record exact supported dimensions, fixed rules, bounds and run settings."""
        if not isinstance(implementation_revision, str) or not implementation_revision.strip():
            raise ValueError('Record the implementation revision (and dirty-tree digest, if any)')
        config = self.to_config(base)
        # Conservative board-capacity bound, valid even for synthetic fixtures.
        # Each side's feature difference is bounded by the largest side total.
        upper = (81 * abs(config.count_weight) + 81 * 1.75 * abs(config.advantage_weight)
                 + 3 * (abs(config.attack_weight) if config.attack_enabled else 0)
                 + 4 * (abs(config.defence_weight) if config.defence_enabled else 0)
                 + 2 * (abs(config.overload_weight) if config.overload_enabled else 0)
                 + 3280 * (abs(config.pressure_weight) if config.pressure_enabled else 0))
        result = dict(genome=self.to_dict(), config_hash=self.config_hash,
                      backend_version=BACKEND_VERSIONS[self.backend],
                      implementation_revision=implementation_revision,
                      tunable_genes=list(GENES[self.backend]),
                      bounds={k: list(BOUNDS[k]) for k in GENES[self.backend]},
                      normalization={'material_is_tunable': True, 'zero_disables_optional_module': True},
                      search_config=config.to_dict(),
                      cache_policy='Python full config identity; Rust fresh search per candidate',
                      heuristic_limit=HEURISTIC_LIMIT, decisive_threshold=MATE_THRESHOLD,
                      conservative_absolute_bound=upper,
                      saturation_possible=upper >= HEURISTIC_LIMIT,
                      saturation_policy='Report observed raw scores and saturation on representative fixtures before fitness',
                      backend_limitations=('Native overload unsupported; fixed advantage bonuses, ordering/PVS and work accounting'
                                           if self.backend == 'rust' else 'Route and overload modules can dominate runtime'))
        if self.backend == 'rust':
            result['native_arguments'] = self.native_arguments(base)
        result['manifest_hash'] = digest(result)
        return result


def saturation_report(genome, states, *, base=None):
    """Bounded caller-selected fixture preflight; terminal/proof scores excluded.

    Uses the declared proof/search budget for each fixture. Budget exhaustion
    fails preflight rather than silently describing incomplete data as safe.
    """
    from ..IntransitiveGame import IntransitiveGame
    from .budget import Budget
    from .evaluation import Evaluator
    evaluator = Evaluator(IntransitiveGame(), genome.to_config(base))
    rows = []
    for state in states:
        config = evaluator.config
        side = int(state[:, :, 82:84].flat[1])
        row = evaluator.explain(state, side, Budget(config.node_limit, config.time_limit), diagnostics=False)
        if 'raw_score' in row:
            rows.append(row)
    saturated = sum(row['saturated'] for row in rows)
    return dict(config_hash=genome.config_hash, ordinary_positions=len(rows),
                saturated_positions=saturated, clipped_positions=sum(row['clipped'] for row in rows),
                saturation_fraction=saturated / len(rows) if rows else None,
                max_absolute_raw_score=max((abs(row['raw_score']) for row in rows), default=None),
                flagged=not rows or saturated > 0)


def json_schema():
    """Draft 2020-12 schema; runtime validation also rejects non-JSON NaN/Infinity."""
    alternatives = []
    for backend, names in GENES.items():
        alternatives.append(dict(type='object', additionalProperties=False,
            required=['version', 'backend', 'genes'], properties=dict(
                version={'const': VERSION}, backend={'const': backend},
                genes=dict(type='object', additionalProperties=False, required=list(names),
                    properties={name: dict(type='number', minimum=BOUNDS[name][0],
                        maximum=BOUNDS[name][1], default=DEFAULTS[name]) for name in names}))))
    return {'$schema': 'https://json-schema.org/draft/2020-12/schema',
            'title': VERSION, 'oneOf': alternatives}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('genome', type=Path)
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    print(json.dumps(Genome.from_json(args.genome.read_text()).manifest(
        implementation_revision=args.revision), indent=2, allow_nan=False))
