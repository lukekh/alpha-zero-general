"""Correctness checks each candidate must pass before its games mean anything.

A held-out win rate is only evidence if the configuration that produced it is
the configuration that was frozen. These checks re-derive each candidate from
its serialized genome, confirm the search is deterministic and leaves its input
alone, measure evaluator saturation on the search pool, and — where the genome
is one the native backend supports — compare Python and Rust scores directly.
"""
from dataclasses import replace
import json

import numpy as np

from ..heuristics.config import SearchConfig
from ..heuristics.search import AlphaBetaPlayer
from ..heuristics.tuning import GENES, Genome, saturation_report
from ..tournament.spec import unpack

NATIVE_DEPTHS = (1, 2)


def genome_of(row):
    return Genome.from_json(json.dumps(row, allow_nan=False))


def contract(row, *, revision):
    """Round-trip the frozen genome and re-derive its evaluation identity."""
    genome = genome_of(row)
    again = Genome.from_json(genome.to_json())
    config = genome.to_config()
    manifest = genome.manifest(implementation_revision=revision)
    return dict(round_trip=again == genome, genome=genome.to_dict(),
                config_hash=genome.config_hash, manifest_hash=manifest['manifest_hash'],
                conservative_absolute_bound=manifest['conservative_absolute_bound'],
                saturation_possible=manifest['saturation_possible'],
                heuristic_limit=manifest['heuristic_limit'],
                decisive_threshold=manifest['decisive_threshold'],
                evaluation={k: v for k, v in config.to_dict().items()
                            if k.endswith(('_weight', '_enabled', '_bonus'))})


def determinism(config, states, *, limits=None):
    """Two fresh engines on one position must agree, and must not touch it."""
    config = replace(config, **(limits or dict(max_depth=2, time_limit=30., node_limit=10**9)))
    rows = []
    for identity, state in states:
        before = state.copy()
        first = AlphaBetaPlayer(config=config).analyze(state)
        second = AlphaBetaPlayer(config=config).analyze(state)
        rows.append(dict(position=identity, action=int(first.action),
                         repeatable=(first.action == second.action
                                     and first.score == second.score
                                     and first.work == second.work),
                         mutated_input=not np.array_equal(state, before),
                         completed_depth=int(first.completed_depth), work=int(first.work)))
    return dict(positions=rows, repeatable=all(r['repeatable'] for r in rows),
                mutated_inputs=sum(r['mutated_input'] for r in rows))


def saturation(row, positions, *, limits):
    """#53's preflight, on the search pool only: held-out states stay unplayed.

    Saturation is a property of the evaluator's numeric range, not of a
    position's difficulty, so measuring it on the pool the optimizer already
    used costs the held-out set nothing.
    """
    states = [unpack(item['state']) for item in positions]
    return saturation_report(genome_of(row), states, base=limits)


def native_supported(row):
    """Rust implements every gene except overload; a nonzero overload is out."""
    genome = genome_of(row)
    genes = genome.to_dict()['genes']
    missing = sorted(set(GENES['python']) - set(GENES['rust']))
    unsupported = [name for name in missing if genes.get(name)]
    return (not unsupported, unsupported)


def native(row, states, *, depths=NATIVE_DEPTHS, binary=None):
    """Compare Python and Rust on the same scales, or say exactly why not.

    Native and Python order moves differently, so tied actions may differ; the
    scores of a completed exact search may not. Actions are reported, scores
    are the parity claim.
    """
    from ..rust_teacher import BINARY, RustTeacher
    path = binary or BINARY
    supported, unsupported = native_supported(row)
    if not supported:
        return dict(status='unsupported', reason=f'Rust has no {", ".join(unsupported)} gene')
    if not path.exists():
        return dict(status='skipped', reason='Native binary not built (cargo build --release)')
    genome = genome_of(row)
    rust_genome = Genome.from_genes({name: value for name, value
                                     in zip(GENES['python'], genome.values)
                                     if name in GENES['rust']}, backend='rust',
                                    variable_material_enabled=genome.variable_material_enabled)
    teacher = RustTeacher(path)
    rows = []
    try:
        for depth in depths:
            arguments = rust_genome.native_arguments(
                replace(SearchConfig(), max_depth=depth, time_limit=60., node_limit=10**9))
            config = replace(genome.to_config(), max_depth=depth, time_limit=60., node_limit=10**9)
            for identity, state in states:
                result = teacher.analyze(state, **arguments)
                python = AlphaBetaPlayer(config=config).analyze(state)
                rows.append(dict(position=identity, depth=depth,
                                 native_action=int(result['action']), python_action=int(python.action),
                                 native_score=float(result['score']), python_score=float(python.score),
                                 score_difference=abs(float(result['score']) - float(python.score)),
                                 native_complete=bool(result['complete']),
                                 python_complete=not bool(python.stopped)))
    finally:
        teacher.close()
    complete = [r for r in rows if r['native_complete'] and r['python_complete']]
    return dict(status='compared', searches=rows, compared=len(complete),
                max_score_difference=max((r['score_difference'] for r in complete), default=None),
                score_parity=all(r['score_difference'] <= 1e-6 for r in complete),
                action_agreement=(sum(r['native_action'] == r['python_action'] for r in complete)
                                  / len(complete) if complete else None),
                note='Differing tied actions are an ordering difference, not an evaluation '
                     'difference; the score comparison is the parity claim.')


def evaluate(row, *, held_states, search_positions, limits, revision, native_binary=None,
             skip_native=False):
    """Everything a candidate must satisfy before its match record is read."""
    genome = genome_of(row)
    checks = dict(contract=contract(row, revision=revision),
                  determinism=determinism(genome.to_config(), held_states),
                  saturation=saturation(row, search_positions, limits=limits))
    checks['native'] = (dict(status='skipped', reason='Native comparison disabled for this run')
                        if skip_native else native(row, held_states, binary=native_binary))
    # A skipped or unsupported native comparison is not a failure; a native
    # comparison that ran and disagreed is.
    checks['passed'] = bool(checks['contract']['round_trip']
                            and checks['determinism']['repeatable']
                            and not checks['determinism']['mutated_inputs']
                            and not checks['saturation']['flagged']
                            and checks['native'].get('score_parity', True))
    return checks
