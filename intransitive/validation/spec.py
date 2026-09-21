"""Frozen acceptance plan: entrants, ablations, protocols and predeclared thresholds.

Everything a run may later be judged by is fixed here, before any engine starts:
which configurations are entrants, which held-out starts they play, what counts
as a practical gain or a regression, and which comparison each candidate must
beat. #55 exports are inputs; nothing in this package updates a default, the
generator or the trainer.
"""
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path

from ..heuristics.config import SearchConfig
from ..heuristics.tuning import DEFAULTS, GENES, Genome
from ..tournament.spec import (POOLS, backend_version, candidate, digest, manifest,
                               protocol, runtime_versions)

SCHEMA = 'intransitive-validation-v1'
ROOT = Path(__file__).parents[1]
STAGES = ('opening', 'midgame', 'endgame')
# The defaults in force before 9424a35 adopted the hour-search candidate. The
# adopted weights were never validated outside the tournament that produced
# them, so the configuration a run must be able to recommend returning to is an
# explicit entrant here, not a footnote: 'retain existing defaults' and 'revert
# to the prior incumbent' are both legal outcomes of this protocol.
PRIOR_INCUMBENT = {'material': 100., 'advantage': 25., 'attack': 0., 'defence': 0.,
                   'overload': 0., 'pressure': 0.}
# Reproduces heuristics/configs/pressure-experimental.json at revision 225dc40,
# the same fixed historical opponent the #54 harness and #55 optimizer used.
PRESSURE_ARCHIVE = dict(PRIOR_INCUMBENT, pressure=10.)
# The #55 variable-material runs used the one-twentieth-scale baseline as their
# fixed reference, so a variable candidate's modules are ablated back to that
# and not to a flat configuration it never competed against.
VARIABLE_BASELINE = dict(DEFAULTS, material=5.)


@dataclass(frozen=True)
class Thresholds:
    """Predeclared decision rule. Serialized into the plan before any game runs.

    Margins are differences of the harness's conservative lower win-point bound
    (official wins over scheduled games) between an entrant and its declared
    comparison, measured on the same protocol and the same held-out starts.
    """
    # A gain worth changing a default for, and the mirror-image regression. Both
    # are point-estimate rules; `superiority` additionally demands separation.
    practical_gain: float = .10
    practical_regression: float = .10
    # Adoption also requires the paired interval of the two entrants to be
    # disjoint. With a handful of colour-paired clusters the Hoeffding radius is
    # wide, so this is expected to fail: 'retain existing defaults' is the
    # declared default outcome, and the plan publishes the detectable effect.
    require_disjoint_intervals: bool = True
    # Harness eligibility: every scheduled game attempted, no failures, and this
    # share of games decided by the rules engine.
    min_completion: float = .8
    # Tactical safety. A candidate may not newly fail any independently
    # certified fixture, and may never fail one whose certificate is a proof.
    max_new_tactical_failures: int = 0
    max_proven_win_failures: int = 0
    # Practical cost. At equal move time the candidate's median completed depth
    # may fall at most this far below its comparison's, and a teacher proposal
    # additionally needs this share of the comparison's label throughput.
    max_median_depth_loss: int = 1
    min_label_throughput_ratio: float = .2

    def __post_init__(self):
        for name in ('practical_gain', 'practical_regression', 'min_completion',
                     'min_label_throughput_ratio'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be a fraction in [0, 1]')
        for name in ('max_new_tactical_failures', 'max_proven_win_failures',
                     'max_median_depth_loss'):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if type(self.require_disjoint_intervals) is not bool:
            raise ValueError('require_disjoint_intervals must be a boolean')

    def to_dict(self):
        return dict(sorted(self.__dict__.items()))


@dataclass(frozen=True)
class Design:
    """Run-size knobs. They change what is measured, so they are frozen too."""
    corpus_seed: int = 54
    corpus_lines: int = 24
    # Held-out starts per stage for the strength protocols, and for the cheaper
    # attribution protocol. Starts always come from distinct generated lines.
    starts_per_stage: int = 2
    strength_stages: tuple = STAGES
    ablation_starts_per_stage: int = 1
    ablation_stages: tuple = STAGES
    fixed_depth: int = 3
    depth_seconds: float = 30.
    wall_seconds: float = .25
    # 192 plies is what let the #55 endgame runs finish games officially; a
    # shorter cap would turn playing strength into a completion-rate artefact.
    max_plies: int = 192
    game_seconds: float = 150.
    # Per-move cost ladder; full games at these depths are not affordable and
    # the report says so with the measured per-move cost rather than guessing.
    cost_depths: tuple = (3, 4, 5, 6)
    cost_seconds: float = 120.
    cost_positions: int = 3
    # Equal-time achieved-depth probe, and the teacher's own labelling protocol.
    equal_time_budgets: tuple = (.05, .25, 1., 5.)
    teacher_depth: int = 5
    teacher_positions: int = 3
    # Zero runs every certified tactical fixture. A positive value keeps the
    # first n in fixture order and exists so a smoke can exercise the stage
    # without buying the whole suite; a real run leaves it at zero.
    tactical_cases: int = 0

    def __post_init__(self):
        if type(self.tactical_cases) is not int or self.tactical_cases < 0:
            raise ValueError('tactical_cases must be a nonnegative integer')
        for name in ('corpus_seed', 'corpus_lines', 'starts_per_stage', 'fixed_depth',
                     'ablation_starts_per_stage', 'max_plies', 'cost_positions',
                     'teacher_depth', 'teacher_positions'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('depth_seconds', 'wall_seconds', 'game_seconds', 'cost_seconds'):
            value = getattr(self, name)
            if type(value) not in (int, float) or value <= 0:
                raise ValueError(f'{name} must be positive')
        for name in ('cost_depths', 'equal_time_budgets', 'ablation_stages', 'strength_stages'):
            if not isinstance(getattr(self, name), tuple) or not getattr(self, name):
                raise ValueError(f'{name} must be a nonempty tuple')
        for name in ('ablation_stages', 'strength_stages'):
            if set(getattr(self, name)) - set(STAGES):
                raise ValueError(f'{name} must be drawn from {STAGES}')

    def to_dict(self):
        return {k: list(v) if isinstance(v, tuple) else v for k, v in sorted(self.__dict__.items())}

    @classmethod
    def from_dict(cls, data):
        fields = {f: tuple(v) if isinstance(v, list) else v for f, v in data.items()}
        return cls(**fields)


def implementation():
    """Digest of this package, so a changed decision rule cannot reuse a plan."""
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(Path(__file__).parent.glob('*.py'))})


def genome_of(genes, *, variable=False):
    return Genome.from_genes(genes, variable_material_enabled=variable)


def entrant(name, genes, *, role='population', variable=False):
    """A frozen tournament entrant built from the shared #53 contract."""
    return candidate(name, genome=genome_of(genes, variable=variable).to_dict(), role=role)


def read_export(relative):
    """Load a #55 export, keeping its file identity as provenance.

    Version 1 records predate signed material and fixed it at 100; they are
    re-expressed in the current space with `material: 100` and the original
    version recorded, never silently reinterpreted under the new defaults.
    """
    path = ROOT / relative
    raw = path.read_bytes()
    data = json.loads(raw)
    genes = dict(data['genes'])
    variable = data['version'].startswith('intransitive-variable')
    if data['version'] == 'intransitive-module-scales-v1':
        genes.setdefault('material', 100.)
    if set(genes) != set(GENES[data['backend']]):
        raise ValueError(f'{relative}: unsupported gene set for {data["backend"]}')
    return dict(genes={k: float(v) for k, v in sorted(genes.items())}, variable=variable,
                source=str(relative), source_sha256=hashlib.sha256(raw).hexdigest(),
                source_version=data['version'])


def default_candidates():
    """The #55 exports this repository actually produced, newest evidence first.

    `baseline` names the configuration each candidate would have to displace:
    the adopted default must justify itself against the weights it replaced,
    and anything else must justify itself against the adopted default.
    """
    adopted = read_export('benchmarks/evolution/hour-search-20260917/provisional-genome.json')
    if adopted['genes'] != {k: DEFAULTS[k] for k in sorted(DEFAULTS)}:
        raise ValueError('The hour-search export no longer equals the shipped defaults')
    rows = [dict(adopted, name='adopted-hour-search', baseline='prior-incumbent',
                 note='Adopted as the Python and Rust default in 9424a35 at the user\'s '
                      'explicit request, without held-out acceptance. This run is that acceptance.'),
            dict(read_export('benchmarks/evolution/endgame-search-20260917/provisional-genome.json'),
                 name='endgame-provisional', baseline='adopted-hour-search',
                 note='Provisional endgame winner; tied defaults 3-3 on its own fresh validation.'),
            dict(read_export('benchmarks/evolution/variable-material-20260917/selected-genome.json'),
                 name='variable-material-selected', baseline='adopted-hour-search',
                 note='Variable-material v3 selection; lost 2-5 to flat defaults after consensus.')]
    return rows


CANDIDATE_FIELDS = ('name', 'genes', 'baseline', 'variable', 'attribute', 'current_default',
                    'source', 'source_sha256', 'source_version', 'note')


def normalize_candidate(row):
    """Fill and validate one candidate record, idempotently.

    Provenance is optional only so a hand-written experiment can be run; a
    candidate that came from a #55 export keeps the file identity it came from.
    """
    unknown = sorted(set(row) - set(CANDIDATE_FIELDS))
    if unknown:
        raise ValueError(f'Unknown candidate fields: {unknown}')
    if not isinstance(row.get('name'), str) or not row['name'].strip():
        raise ValueError('Each candidate needs a nonempty name')
    if not isinstance(row.get('genes'), dict) or not row['genes']:
        raise ValueError(f'{row["name"]}: genes must be a nonempty object')
    variable = bool(row.get('variable', False))
    # Store the complete gene set: a partial record would silently inherit the
    # adopted defaults for the rest, and then an ablation would not know which
    # modules this candidate actually moved.
    genes = genome_of(row['genes'], variable=variable).to_dict()['genes']
    result = dict(name=row['name'], genes=dict(sorted(genes.items())),
                  baseline=row.get('baseline', 'prior-incumbent'), variable=variable,
                  source=row.get('source'), source_sha256=row.get('source_sha256'),
                  source_version=row.get('source_version'), note=row.get('note'))
    result['attribute'] = bool(row.get('attribute', result['baseline'] == 'prior-incumbent'))
    # Whether this candidate is the configuration currently shipped is a fact
    # about its genes, not about which comparison it declared. Only the shipped
    # configuration can be recommended for reverting.
    result['current_default'] = bool(not variable and genes == DEFAULTS)
    return result


def ablation_baseline(row):
    return VARIABLE_BASELINE if row.get('variable') else PRIOR_INCUMBENT


def resolve_baseline(row, candidates):
    """The entrant a candidate must displace, as (name, genes, variable).

    A declared comparison that never plays the candidate is not a comparison,
    so this is what the strength schedules are built from.
    """
    if row['baseline'] == 'prior-incumbent':
        return ('prior-incumbent', PRIOR_INCUMBENT, False)
    other = next(c for c in candidates if c['name'] == row['baseline'])
    return (other['name'], other['genes'], other['variable'])


def attributed(row):
    """Attribution is scheduled for the configuration already installed as the
    default; anything else earns an attribution run by first clearing the gain
    threshold, rather than spending games on modules of a candidate no one is
    proposing to adopt."""
    return bool(row.get('attribute', row['baseline'] == 'prior-incumbent'))


def ablation_genes(genes, baseline=PRIOR_INCUMBENT):
    """One ablation per gene the candidate moved, restoring that gene alone.

    Everything else stays exactly as the candidate has it, so a difference is
    attributable to the restored module rather than to a different search.
    """
    return [(gene, dict(genes, **{gene: baseline[gene]}))
            for gene in sorted(genes) if genes[gene] != baseline.get(gene, genes[gene])]


def select_starts(positions, *, pool='heldout', stages=STAGES, per_stage=1):
    """Deterministic held-out starts, at most one per generated line per stage.

    Positions on one line share a split and a trajectory ancestry, so spreading
    the starts over distinct lines is what keeps the colour-paired clusters from
    collapsing into one observation.
    """
    if pool not in POOLS:
        raise ValueError('Unknown pool')
    chosen, used = [], set()
    for stage in stages:
        available = sorted((p for p in positions if p['pool'] == pool and p['stage'] == stage),
                           key=lambda p: p['sha256'])
        picked = []
        for item in available:
            if item['seed'] in used:
                continue
            used.add(item['seed'])
            picked.append(item)
            if len(picked) == per_stage:
                break
        if len(picked) < per_stage:
            raise ValueError(f'Need {per_stage} distinct {pool} lines with a {stage} start; '
                             f'generate a larger frozen corpus')
        chosen.extend(picked)
    return sorted(chosen, key=lambda p: p['sha256'])


def protocols(design):
    """The two match protocols, kept on separate leaderboards by the harness.

    Fixed completed depth isolates the evaluation change; equal move time
    measures the strength that change actually buys once its cost is paid.
    """
    shared = dict(max_plies=design.max_plies, game_seconds=design.game_seconds)
    return dict(depth=protocol('depth', depth=design.fixed_depth, seconds=design.depth_seconds, **shared),
                wall=protocol('wall', seconds=design.wall_seconds, **shared))


def power(spec, gain):
    """What this schedule could detect before it is run.

    The harness's interval is a weighted Hoeffding radius over colour-paired
    outcomes clustered by opening-generation seed. Publishing it with the plan
    keeps a wide interval from being discovered only after a disappointing
    result, and makes 'retain existing defaults' an expected outcome rather
    than a disappointment.
    """
    rows = {}
    for item in spec['candidates']:
        clusters = {}
        for task in spec['tasks']:
            if item['sha256'] in task['candidates']:
                clusters[task['seed']] = clusters.get(task['seed'], 0) + 1
        total = sum(clusters.values())
        if not total:
            continue
        radius = math.sqrt(math.log(40) / 2 * sum((size / total)**2 for size in clusters.values()))
        # Two intervals of this radius are disjoint only when the difference
        # exceeds twice it, so the declared gain implies a cluster count. It is
        # large; publishing it is how a null result stays a null result instead
        # of becoming a claim of equivalence.
        required = math.ceil(math.log(40) / 2 / (gain / 2)**2) if gain > 0 else None
        rows[item['name']] = dict(scheduled=total, seed_clusters=len(clusters),
                                  hoeffding_radius=radius,
                                  separates_declared_gain=radius * 2 < gain,
                                  clusters_for_declared_gain=required)
    return rows


def experiment(name, *, population, references, limits, positions, question, gain=.1):
    """One frozen tournament manifest, with the question it is allowed to answer."""
    spec = manifest(population + references, positions, [limits], pool='heldout',
                    position_limit=len(positions))
    return dict(name=name, question=question, manifest=spec, games=len(spec['tasks']),
                power=power(spec, gain),
                entrants=[dict(name=c['name'], role=c['role'], sha256=c['sha256'],
                               genome=c['genome']) for c in spec['candidates']])


def plan(corpus, *, candidates=None, design=Design(), thresholds=Thresholds(), revision):
    """Freeze every input, entrant, schedule and decision rule for one acceptance run."""
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError('Record the implementation revision (and dirty-tree digest, if any)')
    candidates = [normalize_candidate(row) for row
                  in (candidates if candidates is not None else default_candidates())]
    if not candidates:
        raise ValueError('At least one candidate is required')
    names = {row['name'] for row in candidates}
    if len(names) != len(candidates) or {'prior-incumbent', 'pressure-10-archive'} & names:
        raise ValueError('Candidate names must be unique and must not shadow a reference')
    for row in candidates:
        if row['baseline'] not in names | {'prior-incumbent'}:
            raise ValueError(f'{row["name"]}: baseline must be a reference or another candidate')
    limits = protocols(design)
    references = [entrant('prior-incumbent', PRIOR_INCUMBENT, role='incumbent'),
                  entrant('pressure-10-archive', PRESSURE_ARCHIVE, role='archive')]
    # A variable-material entrant cannot share a schedule with flat entrants and
    # keep 'same evaluator, different scales' true, so it gets its own strength
    # experiment against the same two references.
    flat = [row for row in candidates if not row['variable']]
    variable = [row for row in candidates if row['variable']]
    starts = select_starts(corpus, stages=design.strength_stages,
                           per_stage=design.starts_per_stage)
    ablation_starts = select_starts(corpus, stages=design.ablation_stages,
                                    per_stage=design.ablation_starts_per_stage)
    experiments = []
    for mode, row in sorted(limits.items()):
        if flat:
            experiments.append(experiment(
                f'strength-{mode}', population=[entrant(c['name'], c['genes']) for c in flat],
                references=references, limits=row, positions=starts,
                question=f'Held-out strength of the flat candidates under the {mode} protocol.'))
        for item in variable:
            # A variable entrant needs its own schedule, but it still has to
            # face the configuration it would displace, plus the same fixed
            # variable baseline the #55 runs used as historical opposition.
            base_name, base_genes, base_variable = resolve_baseline(item, candidates)
            experiments.append(experiment(
                f'strength-{mode}-{item["name"]}',
                population=[entrant(item['name'], item['genes'], variable=True)],
                references=[entrant(base_name, base_genes, role='incumbent',
                                    variable=base_variable),
                            entrant('variable-initial-material-baseline', VARIABLE_BASELINE,
                                    role='archive', variable=True)],
                limits=row, positions=starts, gain=thresholds.practical_gain,
                question=f'Held-out strength of {item["name"]} against the configuration it '
                         f'would displace, under the {mode} protocol. Variable material is a '
                         f'fixed evaluator mode, not a scale, so it never shares a schedule '
                         f'with another candidate.'))
    # Attribution runs at fixed depth only: an ablation answers 'which module
    # produced the difference', and mixing in a timing protocol would answer a
    # different question with half the games. Each gene gets its own schedule
    # so the games measure restore-versus-candidate and nothing else; pitting
    # ablations against each other would buy a matrix nobody asked about.
    for row in candidates:
        if not attributed(row):
            continue
        baseline = ablation_baseline(row)
        for gene, genes in ablation_genes(row['genes'], baseline):
            experiments.append(experiment(
                f'ablation-{row["name"]}-{gene}',
                population=[entrant(f'{row["name"]}-restore-{gene}', genes, variable=row['variable'])],
                references=[entrant(row['name'], row['genes'], role='incumbent',
                                    variable=row['variable'])],
                limits=limits['depth'], positions=ablation_starts,
                gain=thresholds.practical_gain,
                question=f'Does {row["name"]} keep its held-out result when {gene} alone is '
                         f'restored to {baseline[gene]}, every other setting fixed?'))
    result = dict(schema=SCHEMA, revision=revision, implementation=implementation(),
                  backend_version=backend_version(), runtime=runtime_versions(),
                  design=design.to_dict(), thresholds=thresholds.to_dict(),
                  candidates=candidates, references=[dict(name=r['name'], role=r['role'],
                                                          sha256=r['sha256'], genome=r['genome'])
                                                     for r in references],
                  corpus=corpus, starts=[p['sha256'] for p in starts],
                  cost_targets=sorted({row['name'] for row in candidates}
                                      | {resolve_baseline(row, candidates)[0] for row in candidates}),
                  ablation_starts=[p['sha256'] for p in ablation_starts],
                  experiments=experiments,
                  policy=dict(
                      pools='Search, validation and held-out pools are split by generated line '
                            'before any selection; symmetry families are deduplicated across the '
                            'whole corpus. Only held-out starts are played here, and they are '
                            'played once: a set reused to choose between candidates stops being '
                            'held out.',
                      comparison='Each candidate declares the configuration it must displace. '
                                 'Every comparison is paired: same starts, both colours, same '
                                 'protocol, same safety limits.',
                      separation='Adoption needs the two paired intervals to be disjoint, which '
                                 'at this schedule size the published detectable effect says is '
                                 'not attainable: this run can confirm the existing defaults, '
                                 'recommend reverting them, or be inconclusive, and it cannot '
                                 'promote anything. A regression is judged on the direct paired '
                                 'head-to-head record and the point margin instead, because a '
                                 'safety finding must not need the same evidence as a promotion.',
                      decision='Predeclared thresholds decide gain, regression and adoption. '
                               '"Retain existing defaults" is a valid outcome and is the default '
                               'one; nothing in this package writes a default, the generator or '
                               'the trainer.',
                      depth='Fixed completed depth isolates the evaluation change. Equal move '
                            'time measures what it buys. The two leaderboards never pool, and a '
                            'deeper candidate is never compared with a shallower baseline.',
                      attribution='Each changed module is restored on its own, against the '
                                  'candidate itself, on held-out starts at fixed depth. '
                                  'Attribution is scheduled for the configuration currently '
                                  'installed as the default; another candidate earns an '
                                  'attribution run by first clearing the gain threshold.',
                      cost='Deeper full games are priced, not assumed: the cost ladder reports '
                           'measured per-move cost and achieved depth at the teacher depth '
                           'instead of extrapolating a result from cheaper games.'))
    result['sha256'] = digest(result)
    return result


def validate(spec):
    """Rebuild the plan from its own record; any drift is a new plan, not a resume."""
    if spec.get('schema') != SCHEMA:
        raise ValueError('Not a validation plan')
    rebuilt = plan(spec['corpus'], candidates=spec['candidates'],
                   design=Design.from_dict(spec['design']),
                   thresholds=Thresholds(**spec['thresholds']), revision=spec['revision'])
    if rebuilt != spec:
        raise ValueError('Changed or stale acceptance plan, decision rule, code or corpus')


def configs(spec):
    """Every distinct entrant configuration in the plan, by name.

    The tactical, parity and cost studies judge configurations, not schedules,
    so they read the same frozen entrants the matches use.
    """
    rows = {}
    for item in spec['experiments']:
        for row in item['entrants']:
            genome = Genome.from_json(json.dumps(row['genome'], allow_nan=False))
            existing = rows.setdefault(row['name'], dict(genome=row['genome'], sha256=row['sha256'],
                                                         config=genome.to_config().to_dict()))
            if existing['sha256'] != row['sha256']:
                raise ValueError(f'{row["name"]} names two different configurations')
    return dict(sorted(rows.items()))


def search_config(row, **overrides):
    return replace(SearchConfig(**row['config']), **overrides)
