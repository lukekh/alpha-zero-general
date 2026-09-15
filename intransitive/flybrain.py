"""The upstream zero-training fly player, using its cached neural responses.

Only the response bank comes from the fly simulator. Our engine owns legal
actions, canonical player labels, history, and game termination.
"""
import json
from pathlib import Path

import numpy as np

UPSTREAM_URL = 'https://github.com/charbelkassab/flybrain-intransitive'
UPSTREAM_COMMIT = 'f22c07ad64a78b9e1423c376dbc289380ef6f6fc'
FEATURE_ORDER = ('goal', 'capture', 'progress', 'danger', 'base_threat')
DEFAULT_BANK = Path(__file__).parent / 'models' / 'flybrain-wired.npz'


class FlybrainPlayer:
    """Sample held-out approach-minus-escape spikes once per legal move.

    Matches experiment.brain_player(bank, 'wired'): repeats 6..11, then
    uniformly random ties within 1e-9 of the maximum. No greedy fallback.
    """
    kind = 'flybrain'
    label = 'Fly connectome · zero training'

    def __init__(self, game, bank_path=None, seed=None):
        self.game = game
        self.bank_path = Path(bank_path) if bank_path is not None else DEFAULT_BANK
        self.rng = np.random.default_rng(seed)
        self.reload()

    def reload(self):
        if not self.bank_path.is_file():
            raise FileNotFoundError(
                f'Flybrain response bank not found: {self.bank_path}. '
                'Run python -m intransitive.flybrain_prepare first.')
        with np.load(self.bank_path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata'].item()))
            patterns = saved['patterns']
            responses = saved['wired']
        expected = np.array(list(np.ndindex(*(2,) * 5)), dtype=np.int8)
        if (metadata.get('format_version') != 1
                or metadata.get('upstream_commit') != UPSTREAM_COMMIT
                or metadata.get('features') != list(FEATURE_ORDER)
                or metadata.get('stim_ms') != 80 or metadata.get('repeats') != 12
                or metadata.get('brain_seed') != 1
                or metadata.get('mode') != 'wired'
                or not np.array_equal(patterns, expected)
                or responses.shape != (32, 12)
                or not np.isfinite(responses).all()):
            raise ValueError('Incompatible Flybrain response bank')
        self.responses = np.array(responses, dtype=np.float64, copy=True)
        self.responses.flags.writeable = False
        self.metadata = metadata

    def play(self, board, nb_moves=0):
        from .IntransitivePlayers import _legal_actions
        from .reference_greedy import features

        actions = _legal_actions(self.game, board)
        values = []
        for action in actions:
            move_features = features(board, int(action))
            index = 0
            for feature in FEATURE_ORDER:
                index = 2 * index + int(move_features[feature])
            repeat = 6 + self.rng.integers(6)
            values.append(self.responses[index, repeat])
        values = np.asarray(values)
        best = np.flatnonzero(values >= values.max() - 1e-9)
        return int(actions[self.rng.choice(best)])

    def choose(self, state, player):
        return self.play(self.game.getCanonicalForm(state, player))
