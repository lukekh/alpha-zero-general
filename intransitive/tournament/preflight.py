"""Predict a dead protocol before a run, instead of finding it in the report.

`config.activation` answers what the configuration forbids outright. It cannot
answer the other half: a protocol may declare a technique with no blocker at all
and still never fire it, because no search under its time limit ever completes
an iteration deep enough to reach the technique's minimum depth. Issue #66 found
exactly that, and the run report only says so once the games have been played.
This probes one real search per position first.
"""
from ..heuristics.config import activation
from ..record import state_hash

# The root never prunes, so a technique needing `d` plies of remaining depth
# below the root needs a completed iteration of at least `d + 1`. Futility needs
# only a non-root frontier node. Quiescence and MVV-LVA have no depth condition.
def required_depth(config, technique):
    if technique == 'nmp':
        return config.nmp_min_depth + 1
    if technique == 'lmr':
        return config.lmr_min_depth + 1
    if technique == 'futility':
        return 2
    return None


def reachability(config, states, *, game=None, player=None):
    """Search each position once under the protocol's own limits.

    Returns the achieved depths and, for every declared technique with a depth
    condition, whether any position reached it. Reports rather than rejects: a
    technique that fires on some positions and not others is a real protocol,
    just a less informative one than it looks.
    """
    if game is None:
        from ..IntransitiveGame import IntransitiveGame
        game = IntransitiveGame()
    if player is None:
        from ..heuristics.search import AlphaBetaPlayer
        def player(state):
            return AlphaBetaPlayer(game=game, config=config).analyze(state)
    rows = []
    for state in states:
        result = player(state)
        rows.append(dict(sha256=state_hash(state), completed_depth=result.completed_depth,
                         stopped=bool(result.stopped), stop_reason=result.stop_reason,
                         nodes=result.nodes, seconds=result.elapsed))
    depths = [row['completed_depth'] for row in rows]
    deepest = max(depths, default=0)
    preconditions = activation(config)
    techniques, warnings = {}, []
    for name, row in sorted(preconditions.items()):
        if not row['enabled']:
            continue
        needs = required_depth(config, name)
        reached = sum(depth >= needs for depth in depths) if needs else len(rows)
        techniques[name] = dict(needs=needs, positions_reaching=reached, blockers=row['blockers'])
        if row['blockers']:
            warnings.append(f'{name} cannot take effect as configured: ' + '; '.join(row['blockers']))
        elif needs and not reached:
            warnings.append(f'{name} needs a completed depth of {needs} and the deepest probe '
                            f'reached {deepest}: it will not fire under these limits')
        elif needs and reached < len(rows):
            warnings.append(f'{name} needs a completed depth of {needs}, reached on {reached} of '
                            f'{len(rows)} probe positions: it will fire unevenly under these limits')
    return dict(positions=rows, min_completed_depth=min(depths, default=0),
                max_completed_depth=deepest, techniques=techniques, warnings=warnings,
                flagged=bool(warnings))
