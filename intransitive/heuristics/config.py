"""Serializable experimental weights and hard search/work limits."""
from dataclasses import asdict, dataclass
import json
import math


@dataclass(frozen=True)
class SearchConfig:
    evaluator_version: str = 'intransitive-heuristics-v1'
    count_weight: float = 100.
    race_weight: float = 40.
    advantage_weight: float = 25.
    attack_weight: float = 12.
    defence_weight: float = 10.
    overload_weight: float = 5.
    predator_zero_bonus: float = 1.
    predator_scarcity_bonus: float = .5
    prey_bonus: float = .25
    attack_enabled: bool = False
    defence_enabled: bool = False
    overload_enabled: bool = False
    max_depth: int = 3
    node_limit: int = 200000
    time_limit: float = 1.
    proof_depth: int = 2
    proof_nodes: int = 64
    table_entries: int = 10000

    def __post_init__(self):
        if self.evaluator_version != 'intransitive-heuristics-v1':
            raise ValueError('Unsupported evaluator version')
        for name, value in asdict(self).items():
            if name.endswith('_enabled') and type(value) is not bool:
                raise ValueError(f'{name} must be a boolean')
            if name.endswith(('_weight', '_bonus')) or name == 'time_limit':
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('max_depth', 'node_limit', 'proof_depth', 'proof_nodes', 'table_entries'):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.max_depth > 64 or self.proof_depth > 8:
            raise ValueError('Maximum search depth is 64; maximum proof depth is 8')

    def to_dict(self):
        return asdict(self)

    def identity(self):
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_file(cls, path):
        with open(path) as handle:
            return cls(**json.load(handle))
