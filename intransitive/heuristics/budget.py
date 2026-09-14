"""One budget for tree visits, route analysis, move ordering and proof work."""
from collections import defaultdict
from time import perf_counter


class BudgetExpired(Exception):
    pass


class Budget:
    def __init__(self, nodes, seconds, clock=perf_counter):
        self.clock = clock
        self.start = clock()
        self.deadline = self.start + seconds
        self.limit = nodes
        self.work = 0
        self.nodes = 0
        self.proof_nodes = 0
        self.module_seconds = defaultdict(float)
        self.module_calls = defaultdict(int)

    def check(self):
        if self.clock() >= self.deadline:
            raise BudgetExpired

    def charge(self, amount=1):
        self.check()
        if self.work + amount > self.limit:
            raise BudgetExpired
        self.work += amount

    def visit(self, proof=False):
        self.charge()
        self.nodes += 1
        self.proof_nodes += int(proof)
