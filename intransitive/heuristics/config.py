"""Serializable experimental weights and hard search/work limits."""
from dataclasses import asdict, dataclass
import json
import math


TIME_FIRST_LIMITS = {
    'max_depth': 20,
    'time_limit': 5.,
    # A retained safety ceiling, deliberately far above measured five-second
    # work. Zero keeps its existing legal-fallback meaning.
    'node_limit': 1_000_000_000,
}


@dataclass(frozen=True)
class SearchConfig:
    evaluator_version: str = 'intransitive-heuristics-v2'
    count_weight: float = 100.
    race_weight: float = 0.  # Legacy config field; binary clear-run scoring ignores it.
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
    pressure_enabled: bool = False  # Experimental square-ring RPS pressure.
    pressure_weight: float = 1.
    pressure_radius: int = 4  # 3 = 7x7, 4 = original 9x9; keep inner weights.
    max_depth: int = 3
    node_limit: int = 200000
    time_limit: float = 1.
    proof_depth: int = 2
    proof_nodes: int = 64
    table_entries: int = 10000
    pvs_enabled: bool = False
    aspiration_enabled: bool = False
    aspiration_window: float = 25.
    ordering_enabled: bool = False
    compiled_ordering_enabled: bool = False
    depth_replacement_enabled: bool = False
    pressure_cache_entries: int = 0

    def __post_init__(self):
        if self.evaluator_version == 'intransitive-heuristics-v1':
            # Preserve old preset loading while recording the actual new semantics.
            object.__setattr__(self, 'evaluator_version', 'intransitive-heuristics-v2')
        if self.evaluator_version != 'intransitive-heuristics-v2':
            raise ValueError('Unsupported evaluator version')
        if type(self.pressure_radius) is not int or self.pressure_radius not in (3,4):
            raise ValueError('pressure_radius must be 3 (7x7) or 4 (9x9)')
        for name, value in asdict(self).items():
            if name.endswith('_enabled') and type(value) is not bool:
                raise ValueError(f'{name} must be a boolean')
            if name.endswith(('_weight', '_bonus')) or name == 'time_limit':
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('max_depth', 'node_limit', 'proof_depth', 'proof_nodes', 'table_entries', 'pressure_cache_entries'):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if (isinstance(self.aspiration_window, bool)
                or not isinstance(self.aspiration_window, (int, float))
                or not math.isfinite(self.aspiration_window) or self.aspiration_window <= 0):
            raise ValueError('aspiration_window must be finite and positive')
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
