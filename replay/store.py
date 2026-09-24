"""Frozen network-evaluation pool: the replay simulator's cached outcomes.

Within one training iteration the network is frozen, so `state -> (policy,
value)` is a pure function. Recording it once lets alternative search
configurations be re-run over the same positions without further inference,
which is over 70% of self-play time by this repository's own profiling.

The pool deliberately stores *raw* network output. `MCTS.nodes_data` keeps the
policy after Dirichlet noise and after an in-place renormalisation, so replaying
from it would silently inherit the recording run's noise draw.

A serialized Intransitive state is 6,804 bytes, far too large to key a
million-entry dictionary on, so entries are keyed by a 16-byte BLAKE2b digest of
the same bytes `MCTS` uses for its own node identity.
"""

import hashlib
import json
import pickle
import time
import zlib
from pathlib import Path

import numpy as np

SCHEMA = 'az-replay-pool-v1'
KEY_BYTES = 16
DEFAULT_SHARD_ENTRIES = 50_000
MANIFEST_NAME = 'pool-manifest.json'


def state_key(game, board):
    """Stable 16-byte identity for a canonical board."""
    raw = game.stringRepresentation(board)
    if isinstance(raw, str):
        raw = raw.encode()
    return hashlib.blake2b(raw, digest_size=KEY_BYTES).digest()


class PoolError(Exception):
    """The pool disagrees with the game it is being replayed against."""


class EvalStore:
    """Key -> (policy, value), stored sparsely over the legal-move mask.

    A network that masks invalid actions leaves exact zeros outside the mask, so
    only the legal entries are kept: roughly 40 floats instead of 648 for a
    typical Intransitive position. Any evaluation that does place mass outside
    the mask is kept densely instead and counted, so nothing is silently lost.
    """

    def __init__(self, action_size, num_players, metadata=None, mask_tolerance=0.0):
        self.action_size = int(action_size)
        self.num_players = int(num_players)
        self.metadata = dict(metadata or {})
        self.mask_tolerance = float(mask_tolerance)
        self.sparse = {}
        self.dense = {}
        self.mask_violations = 0

    def __len__(self):
        return len(self.sparse) + len(self.dense)

    def __contains__(self, key):
        return key in self.sparse or key in self.dense

    def add(self, key, pi, v, valid_actions):
        """Store one raw evaluation. Returns False if the key was already held."""
        if key in self:
            return False
        pi = np.asarray(pi, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if pi.size != self.action_size:
            raise PoolError(f'policy has {pi.size} slots, pool expects {self.action_size}')
        mask = np.asarray(valid_actions).astype(bool)
        if np.abs(pi[~mask]).sum() > self.mask_tolerance:
            self.mask_violations += 1
            self.dense[key] = (pi.copy(), v.copy())
        else:
            self.sparse[key] = (pi[mask].copy(), v.copy())
        return True

    def get(self, key, valid_actions):
        """Dense policy and value for `key`, or None on a miss.

        Returns fresh arrays every call: MCTS renormalises the policy in place
        and stores it in its own tree.
        """
        entry = self.sparse.get(key)
        if entry is not None:
            pi_valid, v = entry
            mask = np.asarray(valid_actions).astype(bool)
            if int(mask.sum()) != pi_valid.size:
                raise PoolError('pool entry disagrees with the legal-move count for this '
                                'state: a key collision, or a pool recorded under other rules')
            dense = np.zeros(self.action_size, dtype=np.float32)
            dense[mask] = pi_valid
            return dense, v.copy()
        entry = self.dense.get(key)
        if entry is not None:
            pi, v = entry
            return pi.copy(), v.copy()
        return None

    # ---------------------------------------------------------------- storage

    def _items(self):
        for key, (pi, v) in self.sparse.items():
            yield (key, pi, v, False)
        for key, (pi, v) in self.dense.items():
            yield (key, pi, v, True)

    def save(self, directory, shard_entries=DEFAULT_SHARD_ENTRIES):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        items = list(self._items())
        shards, raw_total, packed_total = [], 0, 0
        for index in range(0, max(len(items), 1), shard_entries):
            chunk = items[index:index + shard_entries]
            raw = pickle.dumps(chunk, protocol=pickle.HIGHEST_PROTOCOL)
            packed = zlib.compress(raw, level=6)
            name = f'shard-{index // shard_entries:05d}.pkl.z'
            (directory / name).write_bytes(packed)
            shards.append(dict(file=name, entries=len(chunk), raw_bytes=len(raw),
                               compressed_bytes=len(packed),
                               sha256=hashlib.sha256(packed).hexdigest()))
            raw_total += len(raw)
            packed_total += len(packed)
        manifest = dict(
            schema=SCHEMA,
            created=time.strftime('%Y-%m-%dT%H:%M:%S'),
            action_size=self.action_size,
            num_players=self.num_players,
            key_bytes=KEY_BYTES,
            mask_tolerance=self.mask_tolerance,
            unique_states=len(self),
            sparse_entries=len(self.sparse),
            dense_entries=len(self.dense),
            mask_violations=self.mask_violations,
            raw_bytes=raw_total,
            compressed_bytes=packed_total,
            shards=shards,
            metadata=self.metadata,
        )
        (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + '\n')
        return manifest

    @classmethod
    def load(cls, directory, verify=True):
        directory = Path(directory)
        manifest = json.loads((directory / MANIFEST_NAME).read_text())
        if manifest.get('schema') != SCHEMA:
            raise PoolError(f"unknown pool schema {manifest.get('schema')!r}")
        store = cls(manifest['action_size'], manifest['num_players'],
                    metadata=manifest.get('metadata'),
                    mask_tolerance=manifest.get('mask_tolerance', 0.0))
        for shard in manifest['shards']:
            packed = (directory / shard['file']).read_bytes()
            if verify and hashlib.sha256(packed).hexdigest() != shard['sha256']:
                raise PoolError(f"shard {shard['file']} failed its checksum")
            for key, pi, v, is_dense in pickle.loads(zlib.decompress(packed)):
                (store.dense if is_dense else store.sparse)[key] = (pi, v)
        store.mask_violations = manifest.get('mask_violations', len(store.dense))
        return store, manifest

    # ------------------------------------------------------------ measurement

    def measure(self):
        """Counts and sizes, for deciding whether a full-scale pool fits in RAM."""
        legal = np.array([pi.size for pi, _ in self.sparse.values()], dtype=np.float64)
        payload = sum(pi.nbytes + v.nbytes for pi, v in self.sparse.values())
        payload += sum(pi.nbytes + v.nbytes for pi, v in self.dense.values())
        # Dict slot plus key bytes plus two ndarray headers, measured empirically
        # at roughly 200 bytes on CPython 3.11; reported so a projection to a
        # full iteration is a division rather than a guess.
        overhead = len(self) * 200
        return dict(
            unique_states=len(self),
            sparse_entries=len(self.sparse),
            dense_entries=len(self.dense),
            mask_violations=self.mask_violations,
            mean_legal_moves=float(legal.mean()) if legal.size else 0.0,
            max_legal_moves=int(legal.max()) if legal.size else 0,
            payload_bytes=int(payload),
            estimated_resident_bytes=int(payload + overhead),
            bytes_per_state=float((payload + overhead) / len(self)) if len(self) else 0.0,
        )
