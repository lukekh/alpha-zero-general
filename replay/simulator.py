"""Serve MCTS from a frozen pool instead of the network.

`MCTS` needs no modification to be replayed: it reaches the network only
through `nnet.predict`, so substituting this object re-runs any search
configuration over recorded evaluations.

Replay is exact only while the alternative search stays inside the recorded
state set. A larger simulation budget or a higher `cpuct` will eventually step
outside it, and every such step is counted. A sweep result with a high miss
rate is a partial rerun, not a free off-policy evaluation, so the miss rate is
reported alongside every metric rather than buried.
"""

import numpy as np

from .store import state_key


class ReplayMiss(Exception):
    """The replayed search left the recorded state set."""


class ReplayNet:
    """Frozen-pool stand-in for a network wrapper.

    on_miss:
      'uniform'  count the miss and continue from a uniform prior with value 0
      'strict'   raise, for checking that a configuration stays inside the pool
      'delegate' call a real network, counting the cost that was not free
    """

    def __init__(self, store, game, on_miss='uniform', fallback=None):
        if on_miss not in ('uniform', 'strict', 'delegate'):
            raise ValueError(f'unknown miss policy {on_miss!r}')
        if on_miss == 'delegate' and fallback is None:
            raise ValueError("on_miss='delegate' needs a fallback network")
        self.store = store
        self.game = game
        self.on_miss = on_miss
        self.fallback = fallback
        self._side = {}
        self.reset_counters()

    def reset_counters(self):
        self.hits = 0
        self.misses = 0
        self.distinct_misses = 0

    @property
    def lookups(self):
        return self.hits + self.misses

    @property
    def miss_rate(self):
        return self.misses / self.lookups if self.lookups else 0.0

    def predict(self, board, valid_actions):
        key = state_key(self.game, board)
        found = self.store.get(key, valid_actions)
        if found is not None:
            self.hits += 1
            return found
        self.misses += 1
        if self.on_miss == 'strict':
            raise ReplayMiss('replayed search left the recorded state set')
        cached = self._side.get(key)
        if cached is not None:
            pi, v = cached
            return pi.copy(), v.copy()
        self.distinct_misses += 1
        if self.on_miss == 'delegate':
            pi, v = self.fallback.predict(board, valid_actions)
            pi = np.asarray(pi, dtype=np.float32)
            v = np.asarray(v, dtype=np.float32).reshape(-1)
        else:
            mask = np.asarray(valid_actions).astype(bool)
            pi = np.zeros(self.store.action_size, dtype=np.float32)
            count = int(mask.sum())
            if count:
                pi[mask] = np.float32(1.0 / count)
            v = np.zeros(self.store.num_players, dtype=np.float32)
        self._side[key] = (pi, v)
        return pi.copy(), v.copy()

    def predict_client(self, board, valid_actions, batch_info):
        return self.predict(board, valid_actions)
