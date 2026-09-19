"""Separately timed features. Ordinary estimates never enter the mate range."""
from collections import Counter, OrderedDict
from itertools import combinations
from time import perf_counter
import numpy as np
from .geometry import Geometry, arrival, captures
from .material import BASE, MaterialCache, count_pieces, variable_material_total, variable_piece_values

MATE = 100000.
HEURISTIC_LIMIT = 10000.
MATE_THRESHOLD = 90000.
MODULES = ('piece_count', 'clear_run', 'piece_advantage', 'attacking_position',
           'defensive_position', 'overload', 'local_pressure', 'runner_pressure')


def terminal_value(game, state, side, ply=0):
    from .position import SearchPosition
    if isinstance(state, SearchPosition):
        winner, reason = state.terminal()
        if winner >= 0:
            return MATE - ply if winner == side else -MATE + ply
        return None if reason == 'ongoing' else 0.
    result = game.getGameEnded(state, int(state[:, :, 82:84].flat[1]))
    if not result.any():
        return None
    if result[side] == 1:
        return MATE - ply
    if result[1 - side] == 1:
        return -MATE + ply
    return 0.


def piece_count(geometry, side, config):
    """Material from the stacked code array; never materializes Piece objects."""
    if config.variable_material_enabled:
        counts = tuple(int(np.count_nonzero(geometry.codes == code))
                       for code in (1, 2, 3, -1, -2, -3))
        return variable_material_total(counts, side, config.variable_material_linear) / BASE
    return float(np.count_nonzero(geometry.sides == side))


def piece_advantage(geometry, side, config):
    """Favourable type matchups; see `piece_advantage_reference` for the shape."""
    from .features import advantage_value
    return advantage_value(geometry.codes, geometry.sides, side,
                           float(config.predator_zero_bonus),
                           float(config.predator_scarcity_bonus),
                           float(config.prey_bonus))


def piece_advantage_reference(geometry, side, config):
    own = Counter(abs(p.code) for p in geometry.own(side))
    enemy = Counter(abs(p.code) for p in geometry.own(1 - side))
    return sum(count * (config.predator_zero_bonus * (enemy[(kind + 1) % 3 + 1] == 0)
                        + config.predator_scarcity_bonus / (1 + enemy[(kind + 1) % 3 + 1])
                        + config.prey_bonus * enemy[kind % 3 + 1] / (1 + enemy[kind % 3 + 1]))
               for kind, count in own.items())


def race_candidates(geometry, side):
    """Explain possible routes without awarding unproven positional credit."""
    candidates = []
    opponent_arrival = min((arrival(p.distance, p.side, geometry.turn)
                            for p in geometry.own(1 - side) if p.distance < 99), default=198)
    for runner in geometry.own(side):
        geometry.budget.charge()
        if runner.distance == 99:
            candidates.append(dict(runner=runner.square, route=[], arrival=None,
                                   interceptors=[], estimate=0., status='unknown'))
            continue
        interceptors = [dict(defender=d.square, responses=responses)
                        for d in geometry.own(1 - side)
                        if (responses := geometry.intercepts(d, runner))]
        finish = arrival(runner.distance, side, geometry.turn)
        candidates.append(dict(runner=runner.square, route=geometry.route(runner),
                               arrival=finish, opponent_arrival=opponent_arrival,
                               interceptors=interceptors, estimate=0., status='unknown'))
    return candidates


def attacking_position_reference(geometry, side, config):
    """The readable definition. `attacking_position` is its compiled twin."""
    progress, opportunities = [], {}
    for piece in geometry.own(side):
        safe_steps = []
        for square in geometry.route_squares(piece):
            geometry.budget.charge()
            if piece.distances[square] == 1 and geometry.safe(piece, square, arrival(1, side, geometry.turn) + 1):
                safe_steps.append(square)
        progress.append(1. / (1 + piece.distance) if safe_steps else 0.)
        for enemy in geometry.own(1 - side):
            if captures(piece.code, enemy.code) and piece.distances[enemy.square] == 1:
                ply = arrival(1, side, geometry.turn)
                if geometry.safe(piece, enemy.square, ply + 1):
                    opportunities[enemy.square] = 1.
    # Only one move can be made now. Cap broad progress and take one capture.
    return min(2., sum(progress)) + min(1., sum(opportunities.values()))


def attacking_position(geometry, side, config):
    """Safe progress toward the goal, plus one safe capture; see the reference."""
    from .features import NEIGHBOURS, attack_value
    if not len(geometry.squares):
        return 0.
    value, charge = attack_value(
        geometry.sides, geometry.codes, geometry.slots, geometry.distances,
        geometry.goal_distances, geometry.goal_distance, geometry.squares,
        geometry.threat_block(), geometry.board, NEIGHBOURS, side, geometry.turn)
    geometry.budget.charge(int(charge)) if charge else geometry.budget.check()
    return value


def runner_pressure(geometry, side, config):
    """Unproven credit for a runner whose corridor to the goal looks clear.

    A cheap, deliberately approximate reading of the same idea the clear-run
    certificate proves. For the piece of each type standing closest to the goal,
    count the enemies that could answer it — its predator type, or its own type
    blocking the corner — inside the axis-aligned box spanned by the piece and
    the goal. None is a clear corridor and scores on distance; exactly one
    scores as though the runner loses a move going around it; more scores
    nothing.

    This is an estimate and is wrong sometimes: a piece outside the box can step
    into it, any piece can block rather than only a capturing one, and the
    opponent may simply win the race. Those are the gaps that stop the same
    reading being a proof, which is why this is a weighted term and the
    certificate in `clear_run` is separate and strict.
    """
    goal = geometry.goals[side]
    gx, gy = goal % 9, goal // 9
    closest = {}
    for piece in geometry.own(side):
        kind = abs(piece.code)
        chebyshev = max(abs(piece.square % 9 - gx), abs(piece.square // 9 - gy))
        if kind not in closest or chebyshev < closest[kind][0]:
            closest[kind] = (chebyshev, piece)
    total = 0.
    for chebyshev, piece in closest.values():
        if not chebyshev:
            continue
        low_x, high_x = min(piece.square % 9, gx), max(piece.square % 9, gx)
        low_y, high_y = min(piece.square // 9, gy), max(piece.square // 9, gy)
        answering = 0
        for enemy in geometry.own(1 - side):
            if not (captures(enemy.code, piece.code)
                    or abs(enemy.code) == abs(piece.code)):
                continue
            if (low_x <= enemy.square % 9 <= high_x
                    and low_y <= enemy.square // 9 <= high_y):
                answering += 1
                if answering > 1:
                    break
        if answering > 1:
            continue
        # One answering piece is assumed to cost the runner a single detour.
        total += 1. / (1. + chebyshev + answering)
    return total


def coverage_reference(geometry, side):
    """The readable definition. `coverage` is its compiled twin."""
    return {runner.square: {defender.square: responses
                            for defender in geometry.own(side)
                            if (responses := geometry.intercepts(defender, runner))}
            for runner in geometry.own(1 - side) if runner.distance < 99}


def coverage(geometry, side):
    """Which of `side`'s pieces can answer each enemy runner, and how soon.

    Built from the compiled pair matrix. Replies carry the earliest ply, which
    is all every scoring caller reads; `Geometry.intercepts` remains the source
    for the per-square detail the race diagnostics print.
    """
    plies = geometry.plies(side)
    duties = {}
    for runner in geometry.own(1 - side):
        if runner.distance >= 99:
            continue
        row = plies[geometry.at[runner.square]]
        defenders = {}
        for defender in geometry.own(side):
            ply = int(row[geometry.at[defender.square]])
            if ply >= 0:
                defenders[defender.square] = [{'ply': ply}]
        duties[runner.square] = defenders
    return duties


def defensive_position(geometry, side, config):
    """Timely, safe answers to enemy runners; see the reference for the shape."""
    from .features import defence_value
    import numpy as np
    return defence_value(geometry.plies(side), geometry.sides, geometry.codes,
                         geometry.slots, geometry.at, geometry.threat_block(),
                         np.array(geometry.goals, dtype=np.int64), side)


def defensive_position_reference(geometry, side, config):
    duties = coverage(geometry, side)
    value = sum(min(1., sum(1. / (1 + min(r['ply'] for r in replies))
                            for replies in defenders.values()))
                for defenders in duties.values())
    # A safe same-type occupant can close the actual goal completely.
    goal = geometry.goals[1 - side]
    blocker = geometry.by_square.get(goal)
    if blocker is not None and blocker.side == side and geometry.safe(blocker, goal, 2):
        value += min(1., sum(abs(p.code) == abs(blocker.code)
                             for p in geometry.own(1 - side)))
    return min(4., value)


def overload_conflicts(geometry, side, config, game):
    """Bounded opportunity cost, verified over every legal first defence.

    Only analyse the side actually on move (never manufacture a pass). A pair
    needs a unique shared responder and imminent routes. Any legal move that
    wins, draws, removes a threat, or restores coverage disproves that conflict.
    Surviving conflicts are heuristic, not claims of forced loss.

    Returns the surviving (threat, threat, defender) triples; `overload` counts
    their distinct defenders and attribution spreads the term over those squares.
    """
    if side != geometry.turn:
        return set()
    duties = coverage(geometry, side)
    pairs = []
    for first, second in combinations(duties, 2):
        defenders = set(duties[first])
        if len(defenders) != 1 or defenders != set(duties[second]):
            continue
        if max(geometry.by_square[first].distance, geometry.by_square[second].distance) > 4:
            continue
        pairs.append((first, second, next(iter(defenders))))
    if not pairs:
        return set()
    conflicts = set(pairs)
    for action in np.flatnonzero(game.getValidMoves(geometry.state, side)):
        geometry.budget.charge()
        child, turn = game.getNextState(geometry.state, side, int(action))
        terminal = terminal_value(game, child, side)
        if terminal is not None:
            if terminal >= 0:
                return set()
            continue
        after = Geometry(child, geometry.budget)
        after_duties = coverage(after, side)
        for pair in tuple(conflicts):
            first, second, defender = pair
            remaining = [r for r in (first, second) if r in after.by_square
                         and after.by_square[r].side != side
                         and after.by_square[r].distance < 99]
            if len(remaining) < 2:
                # Removing one threat only resolves the pair when the other is covered.
                if not remaining or after_duties.get(remaining[0]):
                    conflicts.discard(pair)
                continue
            responders = [set(after_duties.get(r, {})) for r in remaining]
            if all(responders):
                # Distinct timely responders, or one safe goal blocker, cover both.
                distinct = any(a != b for a in responders[0] for b in responders[1])
                goal = after.goals[1 - side]
                blocker = after.by_square.get(goal)
                held = (blocker is not None and blocker.side == side
                        and after.safe(blocker, goal, max(arrival(after.by_square[r].distance, 1-side, turn) for r in remaining))
                        and all(not captures(after.by_square[r].code, blocker.code) for r in remaining))
                if distinct or held:
                    conflicts.discard(pair)
        if not conflicts:
            return set()
    return conflicts


def overload(geometry, side, config, game):
    """Distinct overloaded defenders, capped; see `overload_conflicts`."""
    return min(2., float(len({pair[2] for pair in
                              overload_conflicts(geometry, side, config, game)})))


class Evaluator:
    def __init__(self, game, config):
        self.game, self.config = game, config
        self.material = MaterialCache(config)
        self.pressure_cache = OrderedDict()

    @staticmethod
    def _clip(total, budget):
        budget.module_calls['ordinary_evaluations'] += 1
        if abs(total) >= HEURISTIC_LIMIT:
            budget.module_calls['heuristic_saturated'] += 1
        if abs(total) > HEURISTIC_LIMIT:
            budget.module_calls['heuristic_clipped'] += 1
        return max(-HEURISTIC_LIMIT, min(HEURISTIC_LIMIT, total))

    def _pressure(self, pieces, budget, count):
        from .pressure import pressure_totals
        start = perf_counter()
        try:
            # Charge the bounded indexing, pair passes and cumulative rings
            # before entering native code. No hidden unbounded search work.
            budget.charge(81 + count*count + self.config.pressure_radius*count + count*(count-1)//2)
            limit = self.config.pressure_cache_entries
            # Only board-local pressure is cached. Terminal/proof/draw results
            # are still checked separately with their full history identities.
            while len(self.pressure_cache) > limit:
                self.pressure_cache.popitem(last=False)
            key = (self.config.pressure_radius,pieces.tobytes()) if limit else None
            if key is not None and key in self.pressure_cache:
                result = self.pressure_cache[key]
                self.pressure_cache.move_to_end(key)
                budget.module_calls['pressure_cache_hit'] += 1
                budget.check()
                return result
            result = pressure_totals(pieces,self.config.pressure_radius)
            budget.check()
            if key is not None:
                if len(self.pressure_cache) >= limit:
                    self.pressure_cache.popitem(last=False)
                self.pressure_cache[key] = result
            return result
        finally:
            budget.module_seconds['local_pressure'] += perf_counter()-start
            budget.module_calls['local_pressure'] += 1

    def score(self, state, side, budget, proof=None, *, counts=None):
        """Search value without allocating explanations or route diagnostics."""
        return self._evaluate(state, side, budget, proof, explain=False, diagnostics=False, counts=counts)

    def explain(self, state, side, budget, proof=None, *, diagnostics=True):
        return self._evaluate(state, side, budget, proof, explain=True, diagnostics=diagnostics)

    def _evaluate(self, state, side, budget, proof, *, explain, diagnostics, counts=None):
        terminal = terminal_value(self.game, state, side)
        if terminal is not None:
            if not explain:
                return terminal
            return dict(score=terminal, terminal=True, proof={'status': 'terminal'},
                        features={}, terms={})
        if proof is None:
            # Import at call time: search uses this evaluator for ordinary leaves.
            # Search callers pass their existing proof so each leaf is proved once.
            from .search import prove
            proof = prove(self.game, state, self.config, budget)
        if proof['status'] == 'proven':
            from .position import SearchPosition
            turn = state.side if isinstance(state, SearchPosition) else int(state[:, :, 82:84].flat[1])
            score = proof['score'] if side == turn else -proof['score']
            winner = side if score > 0 else 1 - side
            budget.check()
            if not explain:
                return score
            return dict(score=score, terminal=False,
                        proof=dict(proof, winner=winner),
                        features={'own': {'clear_run': float(winner == side)},
                                  'opponent': {'clear_run': float(winner != side)}},
                        terms={'clear_run': score}, races={},
                        skipped_modules=[name for name in MODULES if name != 'clear_run'])
        from .position import SearchPosition
        compact = isinstance(state, SearchPosition)
        if compact and counts is None:
            counts = state.counts
        config = self.config
        if not explain and not (config.attack_enabled or config.defence_enabled
                                or config.overload_enabled or config.runner_enabled):
            start = perf_counter()
            try:
                if counts is None:
                    counts = count_pieces(state)
                # Preserve the existing logical work allowance for piece
                # indexing and the two material modules, including cache hits.
                budget.charge(sum(counts) + 2)
                if self.material.config is not config:
                    self.material = MaterialCache(config)
                total = self.material.score(state.material_state if compact else state, side, counts)
                budget.check()
            finally:
                budget.module_seconds['material'] += perf_counter() - start
                budget.module_calls['material'] += 1
            if config.pressure_enabled and config.pressure_weight:
                blue, red = self._pressure(state.pieces if compact else state[:,:,0], budget, sum(counts))
                total += config.pressure_weight * (blue-red if side == 0 else red-blue)
            return self._clip(total, budget)
        if compact:
            state = state.export()
        own, opponent = {}, {}
        details = {}
        config = self.config
        routes = (diagnostics or config.attack_enabled or config.defence_enabled
                  or config.overload_enabled)
        start = perf_counter()
        try:
            geometry = Geometry(state, budget, routes=routes)
        finally:
            module = 'routes' if routes else 'piece_index'
            budget.module_seconds[module] += perf_counter() - start
            budget.module_calls[module] += 1
        functions = (piece_count, None, piece_advantage, attacking_position,
                     defensive_position, None, None, runner_pressure)
        enabled = (True, diagnostics, True, config.attack_enabled, config.defence_enabled, config.overload_enabled,
                   config.pressure_enabled and bool(config.pressure_weight),
                   config.runner_enabled and bool(config.runner_weight))
        # Clear-run scoring is decisive proof or zero, never a weighted estimate.
        weights = (config.count_weight, 0., config.advantage_weight,
                   config.attack_weight, config.defence_weight, -config.overload_weight,
                   config.pressure_weight, config.runner_weight)
        terms = {}
        total = 0.
        for name, function, active, weight in zip(MODULES, functions, enabled, weights):
            values = [0., 0.]
            if name == 'local_pressure' and active:
                totals = self._pressure(geometry.board, budget, len(geometry.squares))
                values = [totals[side], totals[1-side]]
            elif active:
                start = perf_counter()
                try:
                    budget.charge()
                    for i, player in enumerate((side, 1 - side)):
                        if name == 'clear_run':
                            candidates = race_candidates(geometry, player)
                            details['own' if i == 0 else 'opponent'] = candidates
                            values[i] = 0.
                        elif name == 'overload':
                            values[i] = overload(geometry, player, config, self.game)
                        else:
                            values[i] = function(geometry, player, config)
                finally:
                    budget.module_seconds[name] += perf_counter() - start
                    budget.module_calls[name] += 1
            term = weight * (values[0] - values[1])
            total += term
            if explain:
                own[name], opponent[name] = values
                terms[name] = term
        budget.check()
        score = self._clip(total, budget)
        if not explain:
            return score
        result = dict(score=score, raw_score=total, clipped=abs(total) > HEURISTIC_LIMIT,
                    saturated=abs(total) >= HEURISTIC_LIMIT,
                    terminal=False, features={'own': own, 'opponent': opponent}, terms=terms,
                    races=details, proof=proof)
        if config.variable_material_enabled:
            counts = count_pieces(state)
            result['piece_values'] = {
                label: dict(zip(('rock', 'scissors', 'paper'), map(float, variable_piece_values(
                    counts[player * 3:player * 3 + 3], counts[(1-player) * 3:(1-player) * 3 + 3],
                    config.variable_material_linear))))
                for label, player in (('own', side), ('opponent', 1-side))}
            budget.check()
        return result
