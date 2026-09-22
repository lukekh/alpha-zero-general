"""Export an opt-in versioned preset; never a default, a generator or a trainer.

A preset is a file a user chooses to load. It carries the evaluation the plan
froze, a work and time limit measured to actually complete its own depth, and
the acceptance outcome that was reached — including 'retain existing defaults',
so a preset can never be mistaken for an accepted replacement.
"""
import math

from ..heuristics.config import SearchConfig
from ..heuristics.tuning import EVALUATION_FIELDS
from ..tournament.spec import digest

PRESET_VERSION = 'intransitive-heuristic-preset-v1'
# Headroom over the deepest measured search for this evaluation at this depth.
# A preset that can only just finish on the measured positions would silently
# fall back to a shallower search on a harder one.
WORK_MARGIN = 3.
TIME_MARGIN = 4.


def recommended_limits(profile, depth):
    """Work and time limits measured to complete `depth` for this evaluation.

    The shipped defaults pair a depth with a work cap sized for a much cheaper
    evaluator, so an expensive candidate keeps the depth setting and loses the
    depth. A preset that ships its own measured budget cannot do that quietly.
    """
    rows = [row for row in profile['ladder']['searches']
            if row['depth'] == depth and row['complete']]
    if not rows:
        return None
    work = max(row['work'] for row in rows)
    seconds = max(row['seconds'] for row in rows)
    return dict(max_depth=depth,
                node_limit=int(math.ceil(work * WORK_MARGIN / 100_000) * 100_000),
                time_limit=float(max(1., math.ceil(seconds * TIME_MARGIN))),
                measured_worst_work=int(work), measured_worst_seconds=float(seconds),
                positions=len(rows))


def config(row, limits):
    """A plain SearchConfig record, loadable with `SearchConfig.from_file`."""
    fields = {name: value for name, value in row['config'].items()
              if name in EVALUATION_FIELDS or name in ('max_depth', 'node_limit', 'time_limit',
                                                       'proof_depth', 'proof_nodes', 'table_entries')}
    fields.update({k: v for k, v in limits.items() if k in ('max_depth', 'node_limit', 'time_limit')})
    SearchConfig(**fields)  # Reject an unloadable preset where it is written.
    return dict(sorted(fields.items()))


def export(name, row, *, plan, decision, profile, depth=None):
    """The preset file and the record that says what it is and is not."""
    depth = depth or SearchConfig().max_depth
    limits = recommended_limits(profile, depth) if profile else None
    if limits is None:
        limits = dict(max_depth=depth, node_limit=SearchConfig().node_limit,
                      time_limit=SearchConfig().time_limit,
                      measured_worst_work=None, measured_worst_seconds=None, positions=0)
    preset = config(row, limits)
    verdict = decision['candidates'].get(name, {})
    record = dict(preset_version=PRESET_VERSION, name=name, genome=row['genome'],
                  config_hash=row['sha256'], preset=preset, preset_sha256=digest(preset),
                  measured_limits=limits, plan_sha256=plan['sha256'],
                  revision=plan['revision'], backend_version=plan['backend_version'],
                  runtime=plan['runtime'],
                  acceptance=dict(outcome=verdict.get('outcome', 'inconclusive'),
                                  baseline=verdict.get('baseline'),
                                  gain=verdict.get('gain'), separated=verdict.get('separated'),
                                  established=verdict.get('established'),
                                  blocking=verdict.get('blocking', [])),
                  provenance=verdict.get('provenance'),
                  usage='Opt-in only: load with SearchConfig.from_file or intransitive --ab-config. '
                        'Exporting a preset does not adopt it. This file never changes the '
                        'application defaults, the training generator, the teacher or existing '
                        'training labels.',
                  teacher='Using this preset as a training teacher additionally requires the '
                          'label-provenance decision in the acceptance report: new labels must '
                          'record the evaluator/config identity that produced them, and the '
                          'boundary with existing labels must be recorded rather than mixed.')
    record['record_sha256'] = digest({k: v for k, v in record.items() if k != 'record_sha256'})
    return record
