"""One budget for tree visits, route analysis, move ordering and proof work."""
from collections import defaultdict
from time import perf_counter


class BudgetExpired(Exception):
    """Cooperative search interruption with the limit that was observed first."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


class Budget:
    def __init__(self, nodes, seconds, clock=perf_counter):
        self.clock = clock
        self.start = clock()
        self.seconds = seconds
        self.deadline = self.start + seconds
        self.limit = nodes
        self.work = 0
        self.nodes = 0
        self.proof_nodes = 0
        self.tt_hits = 0
        self.module_seconds = defaultdict(float)
        self.module_calls = defaultdict(int)

    def check(self):
        if self.clock() >= self.deadline:
            # Time deliberately wins a simultaneous time/work observation:
            # every charged operation checks the deadline before the work cap.
            raise BudgetExpired('time')

    def charge(self, amount=1):
        self.check()
        if self.work + amount > self.limit:
            raise BudgetExpired('work')
        self.work += amount

    def visit(self, proof=False):
        self.charge()
        self.nodes += 1
        self.proof_nodes += int(proof)
