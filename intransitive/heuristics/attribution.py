"""Per-square decomposition of a scored board state.

Each module reports an 81-cell array of signed contributions, already
multiplied by its configured coefficient, whose sum is that module's term in
the total score. Positive cells favour the perspective player.

Attribution recomputes the quantities `evaluation` sums rather than borrowing
its running total, so the two paths can disagree. Every report therefore
carries a `residual` per module — the evaluator's own term minus the attributed
sum — and the reconciliation tests fail on any nonzero residual across the
fixture corpus. A visible residual in the UI is a bug indicator, not decoration.

Caps are the reason a module can stop responding to the board: `attacking`
caps broad progress at 2 and captures at 1, `defensive` caps each covered
threat at 1 and the side total at 4, `overload` caps distinct defenders at 2.
Each module reports the raw uncapped feature alongside the value actually used,
so a module pinned at its ceiling is visible instead of silently flat.
"""
from collections import Counter

import numpy as np

from ..IntransitiveConstants import format_coordinate
from .evaluation import (HEURISTIC_LIMIT, MODULES, Evaluator, coverage,
                         overload_conflicts)
from .geometry import Geometry, arrival, captures
from .material import BASE, variable_piece_values
from .pressure import pressure_squares

# Modules whose cost is bounded by the piece index; `overload` replays every
# legal move and builds a child geometry for each, so callers opt into it.
DEFAULT_MODULES = tuple(name for name in MODULES if name != 'overload')


def module_weights(config):
    """Signed coefficients in MODULES order; the evaluator subtracts overload."""
    return dict(zip(MODULES, (config.count_weight, 0., config.advantage_weight,
                              config.attack_weight, config.defence_weight,
                              -config.overload_weight, config.pressure_weight,
                              config.runner_weight)))


def module_active(config):
    """Whether each module contributes to the score at this configuration."""
    return dict(zip(MODULES, (
        True, True, True, config.attack_enabled, config.defence_enabled,
        config.overload_enabled, config.pressure_enabled and bool(config.pressure_weight),
        config.runner_enabled and bool(config.runner_weight))))


class Side:
    """One side's contribution to one module, before the module coefficient."""

    def __init__(self, cells=None, raw=None, used=None, caps=(), detail=None):
        self.cells = dict(cells or {})
        total = sum(self.cells.values())
        self.raw = total if raw is None else raw
        self.used = total if used is None else used
        self.caps = list(caps)
        self.detail = detail or {}

    def rescale(self, cap):
        """Proportionally fit the cells to a capped total and record the cap."""
        total = sum(self.cells.values())
        self.caps.append(dict(cap=cap, raw=total, used=min(cap, total),
                              binding=total > cap))
        if total > cap > 0:
            self.cells = {square: value * cap / total for square, value in self.cells.items()}
        self.raw, self.used = total, min(cap, total)
        return self


def _piece_count(geometry, side, config):
    if not config.variable_material_enabled:
        return Side({piece.square: 1. for piece in geometry.own(side)})
    counts = tuple(sum(p.code == code for p in geometry.pieces) for code in (1, 2, 3, -1, -2, -3))
    own, enemy = (counts[:3], counts[3:]) if side == 0 else (counts[3:], counts[:3])
    values = variable_piece_values(own, enemy, config.variable_material_linear)
    return Side({piece.square: float(values[abs(piece.code) - 1]) / BASE
                 for piece in geometry.own(side)},
                detail={'piece_values': dict(zip(('rock', 'scissors', 'paper'), map(float, values)))})


def _piece_advantage(geometry, side, config):
    enemy = Counter(abs(p.code) for p in geometry.own(1 - side))
    bonus = {kind: (config.predator_zero_bonus * (enemy[(kind + 1) % 3 + 1] == 0)
                    + config.predator_scarcity_bonus / (1 + enemy[(kind + 1) % 3 + 1])
                    + config.prey_bonus * enemy[kind % 3 + 1] / (1 + enemy[kind % 3 + 1]))
             for kind in (1, 2, 3)}
    return Side({piece.square: bonus[abs(piece.code)] for piece in geometry.own(side)},
                detail={'per_piece': {name: bonus[kind] for kind, name
                                      in enumerate(('rock', 'scissors', 'paper'), 1)}})


def _attacking_position(geometry, side, config):
    """Two independently capped halves: broad progress, then one capture."""
    progress, opportunities = Side(), Side()
    deadline = arrival(1, side, geometry.turn) + 1
    for piece in geometry.own(side):
        geometry.budget.charge()
        if any(piece.distances[square] == 1 and geometry.safe(piece, square, deadline)
               for square in geometry.route_squares(piece)):
            progress.cells[piece.square] = 1. / (1 + piece.distance)
        for enemy in geometry.own(1 - side):
            if (captures(piece.code, enemy.code) and piece.distances[enemy.square] == 1
                    and geometry.safe(piece, enemy.square, deadline)):
                opportunities.cells[enemy.square] = 1.
    progress.rescale(2.)
    opportunities.rescale(1.)
    cells = dict(progress.cells)
    for square, value in opportunities.cells.items():
        cells[square] = cells.get(square, 0.) + value
    return Side(cells, raw=progress.raw + opportunities.raw,
                used=progress.used + opportunities.used,
                caps=[dict(row, name=name) for name, part in
                      (('progress', progress), ('captures', opportunities))
                      for row in part.caps])


def _defensive_position(geometry, side, config):
    """Credit each defender's share of a covered threat, then cap the side."""
    result = Side()
    threats = {}
    for runner, defenders in coverage(geometry, side).items():
        shares = {square: 1. / (1 + min(reply['ply'] for reply in replies))
                  for square, replies in defenders.items()}
        total = sum(shares.values())
        scale = min(1., total) / total if total else 0.
        threats[runner] = dict(raw=total, used=min(1., total),
                               defenders=[format_coordinate(s % 9, s // 9) for s in shares])
        for square, share in shares.items():
            result.cells[square] = result.cells.get(square, 0.) + share * scale
    goal = geometry.goals[1 - side]
    blocker = geometry.by_square.get(goal)
    if blocker is not None and blocker.side == side and geometry.safe(blocker, goal, 2):
        hold = min(1., float(sum(abs(p.code) == abs(blocker.code)
                                 for p in geometry.own(1 - side))))
        result.cells[goal] = result.cells.get(goal, 0.) + hold
        result.detail['goal_hold'] = dict(square=format_coordinate(goal % 9, goal // 9), value=hold)
    result.detail['threats'] = {format_coordinate(r % 9, r // 9): row for r, row in threats.items()}
    return result.rescale(4.)


def _overload(geometry, side, config, game):
    """Spread the capped count evenly across the distinct overloaded defenders."""
    conflicts = overload_conflicts(geometry, side, config, game)
    defenders = sorted({pair[2] for pair in conflicts})
    if not defenders:
        return Side()
    used = min(2., float(len(defenders)))
    return Side({square: used / len(defenders) for square in defenders},
                raw=float(len(defenders)), used=used,
                caps=[dict(name='defenders', cap=2., raw=float(len(defenders)),
                           used=used, binding=len(defenders) > 2)],
                detail={'pairs': [[format_coordinate(s % 9, s // 9) for s in pair]
                                  for pair in sorted(conflicts)]})


def _runner_pressure(geometry, side, config):
    """Credit lands on the runner whose corridor was judged clear."""
    from .evaluation import runner_pressure
    goal = geometry.goals[side]
    gx, gy = goal % 9, goal // 9
    closest = {}
    for piece in geometry.own(side):
        kind = abs(piece.code)
        chebyshev = max(abs(piece.square % 9 - gx), abs(piece.square // 9 - gy))
        if kind not in closest or chebyshev < closest[kind][0]:
            closest[kind] = (chebyshev, piece)
    cells, detail = {}, {}
    for kind, (chebyshev, piece) in closest.items():
        if not chebyshev:
            continue
        low_x, high_x = min(piece.square % 9, gx), max(piece.square % 9, gx)
        low_y, high_y = min(piece.square // 9, gy), max(piece.square // 9, gy)
        answering = sum(
            1 for enemy in geometry.own(1 - side)
            if (captures(enemy.code, piece.code) or abs(enemy.code) == abs(piece.code))
            and low_x <= enemy.square % 9 <= high_x and low_y <= enemy.square // 9 <= high_y)
        if answering > 1:
            continue
        cells[piece.square] = 1. / (1. + chebyshev + answering)
        detail[format_coordinate(piece.square % 9, piece.square // 9)] = dict(
            distance=chebyshev, answering=answering)
    value = Side(cells, detail={'runners': detail})
    # The module is the plain sum of its runners, so attribution is exact.
    return value


def _local_pressure(geometry, side, config):
    attack = pressure_squares(geometry.board, config.pressure_radius)
    board = geometry.board.ravel()
    return Side({square: float(attack[square]) for square in np.flatnonzero(attack)
                 if int(board[square] < 0) == side})


def module_sides(geometry, side, config, game, names):
    """Both sides' unweighted contributions for each requested module."""
    result = {}
    for name in names:
        if name == 'clear_run':
            continue
        for label, player in (('own', side), ('opponent', 1 - side)):
            if name == 'piece_count':
                value = _piece_count(geometry, player, config)
            elif name == 'piece_advantage':
                value = _piece_advantage(geometry, player, config)
            elif name == 'attacking_position':
                value = _attacking_position(geometry, player, config)
            elif name == 'defensive_position':
                value = _defensive_position(geometry, player, config)
            elif name == 'overload':
                value = _overload(geometry, player, config, game)
            elif name == 'local_pressure':
                value = _local_pressure(geometry, player, config)
            elif name == 'runner_pressure':
                value = _runner_pressure(geometry, player, config)
            else:
                raise ValueError(f'Unknown module {name}')
            result.setdefault(name, {})[label] = value
    return result


def attribute(game, state, side, budget, config, *, modules=DEFAULT_MODULES, explanation=None):
    """Explain a position square by square from `side`'s perspective.

    `explanation` accepts an already-computed `Evaluator.explain` result so the
    caller pays for one evaluation; otherwise a fresh one is run on `budget`.
    Terminal and proven positions carry no spatial decomposition: their value
    replaces the weighted sum rather than being composed from it.
    """
    names = tuple(dict.fromkeys(modules))
    unknown = set(names) - set(MODULES)
    if unknown:
        raise ValueError(f'Unknown modules: {sorted(unknown)}')
    if side not in (0, 1):
        raise ValueError('Perspective must be player 0 or 1')
    if explanation is None:
        explanation = Evaluator(game, config).explain(state, side, budget, diagnostics=False)
    weights, active = module_weights(config), module_active(config)
    report = dict(perspective=side, score=explanation['score'],
                  raw_score=explanation.get('raw_score', explanation['score']),
                  clipped=bool(explanation.get('clipped')),
                  saturated=bool(explanation.get('saturated')),
                  heuristic_limit=HEURISTIC_LIMIT,
                  terminal=bool(explanation.get('terminal')),
                  proof=explanation.get('proof', {'status': 'unknown'}),
                  terms=dict(explanation.get('terms', {})),
                  weights=weights, active=active, modules={},
                  decomposed=False, total=[0.] * 81)
    if report['terminal'] or report['proof'].get('status') == 'proven':
        report['note'] = ('A terminal or proven result replaces the weighted sum; '
                          'it has no per-square decomposition.')
        report['pv'] = [int(a) for a in report['proof'].get('pv', [])]
        return report
    geometry = Geometry(state, budget, routes=True)
    sides = module_sides(geometry, side, config, game, names)
    total = np.zeros(81)
    for name in names:
        weight = weights[name]
        cells = np.zeros(81)
        if name == 'clear_run':
            report['modules'][name] = dict(
                weight=0., active=True, preview=False,
                term=report['terms'].get('clear_run', 0.),
                attributed=0., residual=0., squares=cells.tolist(), caps=[],
                own=dict(raw=0., used=0., detail={}), opponent=dict(raw=0., used=0., detail={}),
                note='Binary proof or zero; never a weighted positional estimate.')
            continue
        own, opponent = sides[name]['own'], sides[name]['opponent']
        for square, value in own.cells.items():
            cells[square] += weight * value
        for square, value in opponent.cells.items():
            cells[square] -= weight * value
        attributed = float(cells.sum())
        # An inactive module is a preview at its configured coefficient: the
        # evaluator never computed it, so its zero term is not a comparison.
        term = report['terms'].get(name) if active[name] else None
        report['modules'][name] = dict(
            weight=weight, active=active[name], preview=not active[name],
            term=term, attributed=attributed,
            residual=None if term is None else term - attributed,
            squares=cells.tolist(),
            caps=[dict(row, side='own') for row in own.caps]
                 + [dict(row, side='opponent') for row in opponent.caps],
            own=dict(raw=own.raw, used=own.used, detail=own.detail),
            opponent=dict(raw=opponent.raw, used=opponent.used, detail=opponent.detail))
        if active[name]:
            total += cells
    report['total'] = total.tolist()
    report['attributed_total'] = float(total.sum())
    report['decomposed'] = True
    report['residual'] = report['raw_score'] - float(total.sum())
    report['labels'] = [format_coordinate(s % 9, s // 9) for s in range(81)]
    return report
