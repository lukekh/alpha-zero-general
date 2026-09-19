"""Bounded adapter for the repository's PUCT MCTS with heuristic leaf values.

Uniform legal priors; tanh(score / value_scale) supplies a two-player value.
No neural model, rollout, noise or tree reuse across moves. Simulation counts
and observed tree depth are recorded separately from Minimax completed depth.
"""
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
from MCTS import MCTS
from .budget import Budget, BudgetExpired
from .evaluation import Evaluator
from .search import AlphaBetaPlayer, SearchResult
from ..IntransitiveGame import IntransitiveGame


def arguments(simulations, cpuct):
    return SimpleNamespace(numMCTSSims=simulations, cpuct=cpuct, fpu=0.,
        prob_fullMCTS=1., ratio_fullMCTS=1, forced_playouts=False, universes=0,
        no_mem_optim=True, temperature=(1., 1., 1.), dirichletAlpha=0.)


class UniformValue:
    def predict(self, state, legal):
        return legal.astype(np.float64) / legal.sum(), np.array([0., 0.])


@lru_cache(maxsize=1)
def warm_mcts():
    game = IntransitiveGame()
    tree = MCTS(game, UniformValue(), arguments(2, 1.))
    for _ in range(2):
        tree.search(game.getInitBoard())


class HeuristicValue:
    def __init__(self, evaluator, budget, scale):
        self.evaluator, self.budget, self.scale = evaluator, budget, scale

    def predict(self, state, legal):
        score = self.evaluator.score(state, 0, self.budget)
        value = float(np.tanh(score / self.scale))
        return legal.astype(np.float64) / legal.sum(), np.array([value, -value])


class BoundedMCTS(MCTS):
    def __init__(self, game, model, args, budget):
        super().__init__(game, model, args)
        self.budget, self.tree_depth, self.max_tree_depth = budget, 0, 0

    def search(self, canonicalBoard, dirichlet_noise=False, forced_playouts=False):
        self.budget.visit()
        self.max_tree_depth = max(self.max_tree_depth, self.tree_depth)
        self.tree_depth += 1
        try:
            return super().search(canonicalBoard, dirichlet_noise, forced_playouts)
        finally:
            self.tree_depth -= 1


class HeuristicMCTSPlayer:
    def __init__(self, *, config, settings):
        self.config, self.settings = config, settings
        self.game = IntransitiveGame()

    def _prepare(self):
        AlphaBetaPlayer(config=self.config)._prepare()
        warm_mcts()

    def analyze(self, state):
        self._prepare()
        side = int(state[:, :, 82:84].flat[1])
        canonical = self.game.getCanonicalForm(state, side)
        budget = Budget(self.config.node_limit, self.config.time_limit)
        target = self.settings['simulations']
        tree = BoundedMCTS(self.game, HeuristicValue(Evaluator(self.game, self.config), budget,
            self.settings['value_scale']), arguments(target, self.settings['cpuct']), budget)
        complete, reason = 0, 'simulation_limit'
        try:
            for _ in range(target):
                tree.search(canonical)
                budget.check()
                complete += 1
        except BudgetExpired as exc:
            reason = exc.reason
        legal = np.flatnonzero(self.game.getValidMoves(canonical, 0))
        node = tree.nodes_data.get(self.game.stringRepresentation(canonical))
        counts = node[5] if node is not None and node[5] is not None else np.zeros(self.game.getActionSize())
        action = int(max(legal, key=lambda a: (counts[a], -a))) if len(legal) else None
        score = float(node[7]) if node is not None and node[7] is not None else None
        return SearchResult(action=action, score=score, completed_depth=0,
            pv=[] if action is None else [action], nodes=budget.nodes, work=budget.work,
            proof_nodes=budget.proof_nodes, elapsed=budget.clock()-budget.start,
            stopped=complete != target, stop_reason=reason,
            selection_source='completed_simulations' if complete == target else 'partial_simulations',
            module_seconds=dict(budget.module_seconds), module_calls=dict(budget.module_calls),
            effective_limits=dict(self.settings, node_limit=self.config.node_limit, time_limit=self.config.time_limit),
            search_kind='mcts', completed_simulations=complete, requested_simulations=target,
            max_tree_depth=tree.max_tree_depth)
