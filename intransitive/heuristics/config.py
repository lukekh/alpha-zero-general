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
    search_version: str = 'intransitive-selective-v1'
    selective_evaluator_enabled: bool = False  # Experimental margins for evolved/variable material.
    nmp_enabled: bool = False
    nmp_min_depth: int = 3
    nmp_reduction: int = 1
    futility_enabled: bool = False
    futility_max_depth: int = 2
    futility_margin: float = 1.
    quiescence_enabled: bool = False  # Experimental capture-resolving leaf search.
    quiescence_max_plies: int = 8  # Ceiling on the capture chain examined.
    see_ordering_enabled: bool = False  # Cyclic static exchange key in main-search ordering.
    see_quiescence_ordering_enabled: bool = False  # Order quiescence captures by exchange swing.
    see_quiescence_pruning_enabled: bool = False  # Skip quiescence captures the series says lose.
    see_threshold: float = 0.  # Swing below this is skipped; zero keeps even trades.
    compiled_see_enabled: bool = True  # False selects the Python exchange reference.
    delta_pruning_enabled: bool = False  # Skip quiescence captures that cannot reach alpha.
    delta_margin: float = 1.  # Multiplier on the evaluator-unit non-material allowance.
    lmr_enabled: bool = False  # Experimental late move reductions.
    lmr_min_depth: int = 3  # Shallower nodes keep full-depth children.
    lmr_min_index: int = 3  # Moves before this keep full depth.
    lmr_reduction: int = 1  # Plies removed from a reduced child.
    # Issue #68 shallow-depth cutoffs. Each is independently opt-in and off by
    # default; every margin is a multiple of `selective.allowance()`, so a
    # re-tuned genome moves them together instead of stranding a constant.
    # Margin defaults are measured, not inherited: see benchmarks/shallow_pruning.
    # Both cover every observed one-to-three ply swing of the evaluator on the
    # `supported()` scale (1.48 and 0.73 allowances per ply) and on the adopted
    # route genome (0.41 and 0.16). For both, a larger multiple fires less
    # often and is the conservative direction.
    razoring_enabled: bool = False  # Experimental drop to quiescence below alpha.
    razoring_max_depth: int = 2  # Deeper nodes keep their full-width search.
    razoring_margin: float = 1.5  # Allowance multiples per remaining ply.
    reverse_futility_enabled: bool = False  # Experimental static null-move cutoff.
    reverse_futility_max_depth: int = 2
    reverse_futility_margin: float = 1.  # Allowance multiples per remaining ply.
    move_count_pruning_enabled: bool = False  # Experimental late quiet-move skipping.
    move_count_max_depth: int = 3
    move_count_base: int = 12  # Children always searched, before depth squared.
    # Value preserving, so it is not part of the selective (heuristic) family.
    mate_distance_pruning_enabled: bool = False
    race_reduction_enabled: bool = False  # Reduce provably quiet branches on principle.
    counter_move_enabled: bool = False  # Refutation indexed by the opponent's previous move.
    continuation_enabled: bool = False  # History conditioned on the preceding move(s).
    continuation_plies: int = 1  # Plies of preceding context: 1 = opponent's reply, 2 = adds our own.
    history_aging_enabled: bool = False  # Bounded gravity update plus per-iteration decay.
    history_max: int = 16384  # Gravity ceiling; |history| stays below it.
    iir_enabled: bool = False  # Experimental internal iterative reduction/deepening.
    iir_mode: str = 'reduce'  # 'reduce' shortens a moveless node; 'deepen' searches shallow first.
    iir_min_depth: int = 4  # Shallower nodes keep their full depth and order.
    iir_reduction: int = 1  # Plies removed, or removed before the ordering search.
    evaluator_version: str = 'intransitive-heuristics-v2'
    count_weight: float = 100.
    variable_material_enabled: bool = False  # Replace flat piece counts with BASE/REG values.
    variable_material_linear: bool = False  # Drop the square root from own-type scarcity.
    race_weight: float = 0.  # Legacy config field; binary clear-run scoring ignores it.
    advantage_weight: float = 23.967050360966205
    attack_weight: float = 25.714516982666414
    defence_weight: float = 32.5643023919054
    overload_weight: float = 5.
    predator_zero_bonus: float = 1.
    predator_scarcity_bonus: float = .5
    prey_bonus: float = .25
    attack_enabled: bool = True
    defence_enabled: bool = True
    overload_enabled: bool = False
    pressure_enabled: bool = False  # Experimental square-ring RPS pressure.
    pressure_weight: float = 1.
    pressure_radius: int = 4  # 3 = 7x7, 4 = original 9x9; keep inner weights.
    runner_enabled: bool = False  # Experimental unproven runner pressure.
    runner_weight: float = 1.
    max_depth: int = 3
    node_limit: int = 200000
    time_limit: float = 1.
    proof_depth: int = 2
    proof_nodes: int = 64
    certificate_enabled: bool = False  # Experimental forced corner-run certificate.
    certificate_plies: int = 20  # Longest run the certificate will certify.
    certificate_cutoff_enabled: bool = False  # Experimental certificate as an interior bound.
    certificate_cutoff_min_depth: int = 2  # Shallowest node that may pay for a probe.
    certificate_guard_enabled: bool = False  # Certified nodes refuse NMP/futility/LMR.
    table_entries: int = 10000
    mvv_lva_enabled: bool = False
    pvs_enabled: bool = False
    aspiration_enabled: bool = False
    aspiration_window: float = 25.
    ordering_enabled: bool = False
    compiled_ordering_enabled: bool = True
    depth_replacement_enabled: bool = False
    pressure_cache_entries: int = 0

    def __post_init__(self):
        if self.search_version != 'intransitive-selective-v1':
            raise ValueError('Unsupported search version')
        for name, low, high in (('nmp_min_depth', 3, 32), ('nmp_reduction', 1, 8), ('futility_max_depth', 1, 2),
                                ('quiescence_max_plies', 1, 32), ('lmr_min_depth', 2, 32),
                                ('lmr_min_index', 1, 64), ('lmr_reduction', 1, 8),
                                ('razoring_max_depth', 1, 4), ('reverse_futility_max_depth', 1, 6),
                                ('move_count_max_depth', 1, 8), ('move_count_base', 1, 64),
                                ('certificate_cutoff_min_depth', 1, 32),
                                ('continuation_plies', 1, 2), ('history_max', 256, 2**20),
                                ('iir_min_depth', 3, 32), ('iir_reduction', 1, 8)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{name} must be an integer in {low}..{high}')
        if self.nmp_reduction > self.nmp_min_depth - 2:
            raise ValueError('NMP must retain at least one probe ply')
        if self.lmr_reduction > self.lmr_min_depth - 2:
            raise ValueError('LMR must retain at least one ply below the reduction')
        if self.iir_mode not in ('reduce', 'deepen'):
            raise ValueError("iir_mode must be 'reduce' or 'deepen'")
        if self.iir_reduction > self.iir_min_depth - 2:
            raise ValueError('IIR must retain at least one ply below the reduction')
        if (self.counter_move_enabled or self.continuation_enabled) and not self.ordering_enabled:
            # An explicitly enabled technique must never be silently inert (#66).
            raise ValueError('counter_move_enabled/continuation_enabled require ordering_enabled')
        if (type(self.futility_margin) not in (int, float) or not math.isfinite(self.futility_margin)
                or not 1 <= self.futility_margin <= 16):
            raise ValueError('futility_margin must be finite in 1..16')
        # The issue #68 margins admit fractions: the measured safe multiplier
        # on the adopted route genome is below one, because `allowance()` sums
        # weight ceilings rather than the swing those modules actually produce.
        for name in ('razoring_margin', 'reverse_futility_margin'):
            value = getattr(self, name)
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not 0 < value <= 64):
                raise ValueError(f'{name} must be finite in (0, 64]')
        if self.razoring_enabled and not self.quiescence_enabled:
            # Razoring returns a quiescence value. Without quiescence it would
            # return the bare static score, which is a different, untested and
            # far more aggressive technique.
            raise ValueError('razoring_enabled requires quiescence_enabled')
        if (type(self.delta_margin) not in (int, float) or not math.isfinite(self.delta_margin)
                or not 0 <= self.delta_margin <= 16):
            raise ValueError('delta_margin must be finite in 0..16')
        if type(self.see_threshold) not in (int, float) or not math.isfinite(self.see_threshold):
            raise ValueError('see_threshold must be finite')
        if self.variable_material_linear and not self.variable_material_enabled:
            raise ValueError('variable_material_linear requires variable_material_enabled')
        # A flag that silently does nothing is worse than a rejected config:
        # the race test only ever relaxes an LMR condition.
        if self.race_reduction_enabled and not self.lmr_enabled:
            raise ValueError('race_reduction_enabled requires lmr_enabled')
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
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f'{name} must be finite')
                signed = name in ('count_weight', 'advantage_weight', 'attack_weight',
                                  'defence_weight', 'overload_weight', 'pressure_weight',
                                  'runner_weight')
                if not signed and value < 0:
                    raise ValueError(f'{name} must be nonnegative')
        if type(self.certificate_plies) is not int or not 1 <= self.certificate_plies <= 64:
            raise ValueError('certificate_plies must be an integer in 1..64')
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

    def selective_pruning(self):
        """Whether a heuristic cutoff that can change the returned value is on.

        Mate-distance pruning is excluded: it narrows the window to bounds the
        true value already satisfies, so it is value preserving.
        """
        return (self.nmp_enabled or self.futility_enabled or self.razoring_enabled
                or self.reverse_futility_enabled or self.move_count_pruning_enabled)

    def identity(self):
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_file(cls, path):
        with open(path) as handle:
            return cls(**json.load(handle))
