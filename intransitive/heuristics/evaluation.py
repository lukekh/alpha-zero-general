"""Six separately timed features. Ordinary estimates never enter the mate range."""
from collections import Counter
from itertools import combinations
from time import perf_counter
import numpy as np
from .geometry import Geometry, arrival, captures

MATE = 100000.
HEURISTIC_LIMIT = 10000.
MATE_THRESHOLD = 90000.
MODULES = ('piece_count', 'clear_run', 'piece_advantage', 'attacking_position',
           'defensive_position', 'overload')


def terminal_value(game, state, side, ply=0):
    result = game.getGameEnded(state, int(state[:, :, 32].flat[1]))
    if not result.any():
        return None
    if result[side] == 1:
        return MATE - ply
    if result[1 - side] == 1:
        return -MATE + ply
    return 0.


def piece_count(geometry, side, config):
    return float(len(geometry.own(side)))


def piece_advantage(geometry, side, config):
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


def attacking_position(geometry, side, config):
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


def coverage(geometry, side):
    return {runner.square: {defender.square: responses
                            for defender in geometry.own(side)
                            if (responses := geometry.intercepts(defender, runner))}
            for runner in geometry.own(1 - side) if runner.distance < 99}


def defensive_position(geometry, side, config):
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


def overload(geometry, side, config, game):
    """Bounded opportunity cost, verified over every legal first defence.

    Only analyse the side actually on move (never manufacture a pass). A pair
    needs a unique shared responder and imminent routes. Any legal move that
    wins, draws, removes a threat, or restores coverage disproves that conflict.
    Surviving conflicts are heuristic, not claims of forced loss.
    """
    if side != geometry.turn:
        return 0.
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
        return 0.
    conflicts = set(pairs)
    for action in np.flatnonzero(game.getValidMoves(geometry.state, side)):
        geometry.budget.charge()
        child, turn = game.getNextState(geometry.state, side, int(action))
        terminal = terminal_value(game, child, side)
        if terminal is not None:
            if terminal >= 0:
                return 0.
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
            return 0.
    return min(2., float(len({pair[2] for pair in conflicts})))


class Evaluator:
    def __init__(self, game, config):
        self.game, self.config = game, config

    def score(self, state, side, budget, proof=None):
        """Search value without allocating explanations or route diagnostics."""
        return self._evaluate(state, side, budget, proof, explain=False, diagnostics=False)

    def explain(self, state, side, budget, proof=None, *, diagnostics=True):
        return self._evaluate(state, side, budget, proof, explain=True, diagnostics=diagnostics)

    def _evaluate(self, state, side, budget, proof, *, explain, diagnostics):
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
            turn = int(state[:, :, 32].flat[1])
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
        own, opponent = {}, {}
        details = {}
        config = self.config
        routes = diagnostics or config.attack_enabled or config.defence_enabled or config.overload_enabled
        start = perf_counter()
        try:
            geometry = Geometry(state, budget, routes=routes)
        finally:
            module = 'routes' if routes else 'piece_index'
            budget.module_seconds[module] += perf_counter() - start
            budget.module_calls[module] += 1
        functions = (piece_count, None, piece_advantage, attacking_position,
                     defensive_position, None)
        enabled = (True, diagnostics, True, config.attack_enabled, config.defence_enabled, config.overload_enabled)
        # Clear-run scoring is decisive proof or zero, never a weighted estimate.
        weights = (config.count_weight, 0., config.advantage_weight,
                   config.attack_weight, config.defence_weight, -config.overload_weight)
        terms = {}
        total = 0.
        for name, function, active, weight in zip(MODULES, functions, enabled, weights):
            values = [0., 0.]
            if active:
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
        score = max(-HEURISTIC_LIMIT, min(HEURISTIC_LIMIT, total))
        if not explain:
            return score
        return dict(score=score,
                    terminal=False, features={'own': own, 'opponent': opponent}, terms=terms,
                    races=details, proof=proof)
