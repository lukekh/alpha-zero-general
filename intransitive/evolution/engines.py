"""Bounded resident candidate processes; search state stays fresh on every move."""
from collections import OrderedDict
import random

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
        self.pool.active.remove(self.key)
        if self.broken:
            self.pool.drop(self.key)


class EnginePool:
    """LRU eviction of inactive engines; at most one active game/two leases."""
    def __init__(self, capacity, process_factory=EngineProcess):
        if type(capacity) is not int or not 2 <= capacity <= 8:
            raise ValueError('Resident candidate process limit must be in [2, 8]')
        self.capacity, self.process_factory = capacity, process_factory
        self.processes, self.active = OrderedDict(), set()

    def drop(self, key):
        self.processes.pop(key).close()

    def retain(self, candidates, limits):
        if self.active:
            raise ValueError('Cannot change opponent slate during an active game')
        needed = {digest(dict(candidate=item['sha256'], limits=limits)) for item in candidates}
        for key in list(self.processes):
            if key not in needed:
                self.drop(key)

    def acquire(self, item, limits, seed):
        key = digest(dict(candidate=item['sha256'], limits=limits))
        if key in self.active:
            raise ValueError('Cannot share an engine across concurrent games')
        if key in self.processes:
            process = self.processes[key]
            try:
                process.connection.send(dict(reset_seed=seed))
            except (BrokenPipeError, EOFError, OSError):
                self.drop(key)
        if key not in self.processes:
            if len(self.processes) >= self.capacity:
                inactive = next((k for k in self.processes if k not in self.active), None)
                if inactive is None:
                    raise ValueError('All resident engines are in use')
                self.drop(inactive)
            self.processes[key] = self.process_factory(item, limits, seed, target=reusable_worker)
        self.processes.move_to_end(key)
        self.active.add(key)
        return Lease(self, key, self.processes[key])

    def close(self):
        for key in list(self.processes):
            self.drop(key)
        self.active.clear()
