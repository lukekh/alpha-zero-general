"""Bounded resident candidate processes; search state stays fresh on every move."""
from collections import OrderedDict
import random
import threading

import numpy as np

from ..tournament.runner import EngineProcess, engine_worker, peak_rss_bytes
from ..tournament.spec import digest


class ResettableConnection:
    """Add a between-games seed/handshake command to the existing worker loop."""
    def __init__(self, connection):
        self.connection = connection
        self.ready = None

    def send(self, message):
        if message.get('kind') == 'ready':
            self.ready = message
        self.connection.send(message)

    def recv(self):
        while True:
            message = self.connection.recv()
            if not isinstance(message, dict) or set(message) != {'reset_seed'}:
                return message
            random.seed(message['reset_seed'])
            np.random.seed(message['reset_seed'] % 2**32)
            self.connection.send(dict(self.ready, startup_seconds=0., cpu_seconds=0.,
                                      peak_rss_bytes=peak_rss_bytes(), reused_process=True))

    def close(self):
        self.connection.close()


def reusable_worker(connection, item, limits, seed):
    # The harness still constructs a fresh AlphaBetaPlayer for EVERY move.
    # Only compiled machine code and the immutable candidate survive a game.
    engine_worker(ResettableConnection(connection), item, limits, seed)


class Lease:
    def __init__(self, pool, key, process):
        self.pool, self.key, self.process = pool, key, process
        self.connection = self
        self.broken = False
        self.closed = False

    def send(self, state):
        try:
            self.process.connection.send(state)
        except BaseException:
            self.broken = True
            raise

    def receive(self, deadline, cancelled):
        try:
            return self.process.receive(deadline, cancelled)
        except BaseException:
            self.broken = True
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.pool.release(self.key, self.process, self.broken)


class EnginePool:
    """Bounded resident engines, leased per game, safe for concurrent matches.

    A candidate can be playing several games at once, so a key owns a list of
    idle processes rather than a single one, and two leases on one candidate are
    two separate processes. Engine reuse only ever preserves compiled machine
    code and the immutable candidate; the harness still builds a fresh player
    for every move, so sharing a key across games changes no result.

    `capacity` bounds live processes and `workers` declares how many matches may
    run at once. Requiring `2 * workers <= capacity` is what makes acquire
    wait-free: a match that already holds one engine can always obtain its
    second, so two matches can never deadlock each holding one engine and
    waiting for the other. Eviction only ever takes an idle process.
    """
    def __init__(self, capacity, process_factory=EngineProcess, workers=1):
        if type(capacity) is not int or not 2 <= capacity <= 24:
            raise ValueError('Resident candidate process limit must be in [2, 24]')
        if type(workers) is not int or workers < 1 or 2 * workers > capacity:
            raise ValueError('Resident engines must cover two per concurrent match')
        if capacity == 2 * workers and workers > 1:
            # Every slot would be leased whenever all workers are busy, so no
            # warm engine could ever be cached and each match would pay process
            # startup again. Demand at least one slot of reuse headroom.
            raise ValueError('Resident engines must exceed two per concurrent match')
        self.capacity, self.process_factory, self.workers = capacity, process_factory, workers
        self.idle, self.leased = OrderedDict(), set()
        self.live = 0
        self.lock = threading.Lock()

    @property
    def active(self):
        return len(self.leased)

    def _discard(self, process):
        process.close()
        self.live -= 1

    def _evict(self):
        """Close one idle process, least recently released first."""
        for key, bucket in list(self.idle.items()):
            if bucket:
                self._discard(bucket.pop())
                if not bucket:
                    del self.idle[key]
                return True
        return False

    def drop(self, key):
        with self.lock:
            for process in self.idle.pop(key, []):
                self._discard(process)

    def retain(self, candidates, limits):
        with self.lock:
            if self.leased:
                raise ValueError('Cannot change opponent slate during an active game')
            needed = {digest(dict(candidate=item['sha256'], limits=limits)) for item in candidates}
            for key in [k for k in self.idle if k not in needed]:
                for process in self.idle.pop(key):
                    self._discard(process)

    def acquire(self, item, limits, seed):
        key = digest(dict(candidate=item['sha256'], limits=limits))
        with self.lock:
            process = None
            bucket = self.idle.get(key)
            if bucket:
                process = bucket.pop()
                if not bucket:
                    del self.idle[key]
                try:
                    process.connection.send(dict(reset_seed=seed))
                except (BrokenPipeError, EOFError, OSError):
                    self._discard(process)
                    process = None
            if process is None:
                while self.live >= self.capacity and self._evict():
                    pass
                if self.live >= self.capacity:
                    raise ValueError('All resident engines are in use')
                process = self.process_factory(item, limits, seed, target=reusable_worker)
                self.live += 1
            self.leased.add(process)
            return Lease(self, key, process)

    def release(self, key, process, broken):
        with self.lock:
            self.leased.discard(process)
            if broken:
                self._discard(process)
                return
            self.idle.setdefault(key, []).append(process)
            self.idle.move_to_end(key)

    def close(self):
        with self.lock:
            for key in list(self.idle):
                for process in self.idle.pop(key):
                    process.close()
            for process in list(self.leased):
                process.close()
            self.leased.clear()
            self.live = 0
