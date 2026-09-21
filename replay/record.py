"""Record a replay pool: raw network evaluations plus per-move reference decisions.

Games are played with the reference search configuration and Dirichlet noise
off, so the recorded root decision is a clean reference rather than one
particular noise draw. Moves are still *selected* with the self-play temperature
schedule, so the recorded positions keep a realistic self-play distribution.
"""

import hashlib
import json
import pickle
import time
import zlib
from pathlib import Path

import numpy as np

from MCTS import MCTS

from .store import EvalStore, state_key

POSITIONS_SCHEMA = 'az-replay-positions-v1'
POSITIONS_NAME = 'positions.pkl.z'
POSITIONS_MANIFEST = 'positions-manifest.json'


class RecordingNet:
    """Memoising proxy that records every distinct network evaluation.

    Memoising is exact because the network is frozen for the whole recording,
    and it is what makes a fresh tree per move affordable.
    """

    def __init__(self, inner, game, store):
        self.inner = inner
        self.game = game
        self.store = store
        self.calls = 0        # leaf evaluations requested by MCTS
        self.inferences = 0   # forward passes actually performed

    def predict(self, board, valid_actions):
        self.calls += 1
        key = state_key(self.game, board)
        cached = self.store.get(key, valid_actions)
        if cached is not None:
            return cached
        pi, v = self.inner.predict(board, valid_actions)
        self.inferences += 1
        self.store.add(key, pi, v, valid_actions)
        # Return the pool's own round trip, so a sweep at the recording
        # configuration reproduces the recording exactly.
        return self.store.get(key, valid_actions)

    def predict_client(self, board, valid_actions, batch_info):
        # Recording is single-threaded; the batched client path is unused.
        return self.predict(board, valid_actions)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def temperature_schedule(begin, end, half_life):
    """Coach's exponential decay, reproduced so `replay` stays standalone."""
    def temperature(ply):
        return end + (begin - end) * (0.5 ** (ply / half_life))
    return temperature


def sample_action(pi, temperature, rng):
    pi = np.asarray(pi, dtype=np.float64)
    if temperature <= 0.02:
        best = np.flatnonzero(pi == pi.max())
        return int(rng.choice(best))
    weights = pi ** (1.0 / temperature)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return int(np.argmax(pi))
    return int(rng.choice(len(weights), p=weights / total))


def split_of(game_index, split_seed, select_fraction):
    """Leak-free split by game, not by position: plies within a game correlate."""
    digest = hashlib.blake2b(f'{split_seed}:{game_index}'.encode(), digest_size=8).digest()
    return 'select' if int.from_bytes(digest, 'big') / 2 ** 64 < select_fraction else 'confirm'


def record_games(game, net, args, *, games, seed=0, temperature=None, tree_reuse=False,
                 max_plies=400, split_seed=0, select_fraction=0.5, progress=None):
    """Play `games` self-play games, returning (store, positions, summaries)."""
    temperature = temperature or temperature_schedule(1.0, 0.1, 10)
    store = EvalStore(game.getActionSize(), game.num_players)
    recorder = RecordingNet(net, game, store)
    positions, summaries = [], []

    for game_index in range(games):
        rng = np.random.default_rng(np.random.SeedSequence([seed, game_index]))
        split = split_of(game_index, split_seed, select_fraction)
        board = game.getInitBoard()
        current, ply, mcts = 0, 0, None
        while True:
            canonical = game.getCanonicalForm(board, current)
            if mcts is None or not tree_reuse:
                mcts = MCTS(game, recorder, args, dirichlet_noise=False)
            before = recorder.calls
            pi, q, _ = mcts.getActionProb(canonical, temp=1.0, force_full_search=True)
            pi = np.asarray(pi, dtype=np.float64)
            mask = np.asarray(game.getValidMoves(canonical, 0)).astype(bool)
            positions.append(dict(
                game_index=game_index,
                split=split,
                ply=ply,
                state=canonical.copy(),
                key=state_key(game, canonical),
                reference_action=int(np.argmax(pi)),
                # float64: the self-check compares this against a replayed
                # float64 distribution, and a float32 round trip would leave a
                # ~1e-8 residual that looks like a replay error.
                reference_pi=pi[mask].copy(),
                legal_moves=int(mask.sum()),
                reference_q=np.asarray(q, dtype=np.float32),
                reference_evals=recorder.calls - before,
                current_player=current,
            ))
            action = sample_action(pi, temperature(ply), rng)
            board, current = game.getNextState(board, current, action)
            ply += 1
            ended = game.getGameEnded(board, current)
            if ended.any() or ply >= max_plies:
                summaries.append(dict(game_index=game_index, split=split, plies=ply,
                                      terminated=bool(ended.any()),
                                      result=[float(x) for x in ended]))
                break
        if progress is not None:
            progress(game_index + 1, len(positions), len(store), recorder.inferences)

    store.metadata.update(
        games=games, seed=seed, tree_reuse=tree_reuse, max_plies=max_plies,
        positions=len(positions), leaf_calls=recorder.calls,
        inferences=recorder.inferences,
        memoisation_hit_rate=1.0 - recorder.inferences / max(recorder.calls, 1),
        reference_args={k: _plain(v) for k, v in dict(args).items()},
    )
    return store, positions, summaries


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def save_positions(directory, positions, summaries, metadata=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    packed = zlib.compress(pickle.dumps(positions, protocol=pickle.HIGHEST_PROTOCOL), level=6)
    (directory / POSITIONS_NAME).write_bytes(packed)
    counts = {}
    for position in positions:
        counts[position['split']] = counts.get(position['split'], 0) + 1
    manifest = dict(
        schema=POSITIONS_SCHEMA,
        created=time.strftime('%Y-%m-%dT%H:%M:%S'),
        positions=len(positions),
        games=len(summaries),
        positions_by_split=counts,
        games_by_split={s: sum(1 for g in summaries if g['split'] == s) for s in counts},
        compressed_bytes=len(packed),
        sha256=hashlib.sha256(packed).hexdigest(),
        summaries=summaries,
        metadata=dict(metadata or {}),
    )
    (directory / POSITIONS_MANIFEST).write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def load_positions(directory, verify=True):
    directory = Path(directory)
    manifest = json.loads((directory / POSITIONS_MANIFEST).read_text())
    packed = (directory / POSITIONS_NAME).read_bytes()
    if verify and hashlib.sha256(packed).hexdigest() != manifest['sha256']:
        raise ValueError('recorded positions failed their checksum')
    return pickle.loads(zlib.decompress(packed)), manifest
