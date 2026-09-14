"""Count native allocations separately from uninstrumented speed measurements.

Run with NUMBA_NRT_STATS=1; tracemalloc's retained blocks alone cannot count
transient allocations made inside compiled transitions/proofs.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from numba.core.runtime import rtsys
from intransitive.benchmarks.compact.reproduce import baseline
from intransitive.heuristics import search
from intransitive.heuristics.position import SearchPosition
from intransitive.record import load_record


def measure(call):
    before = rtsys.get_allocation_stats()
    call()
    after = rtsys.get_allocation_stats()
    return {name: end-start for name, start, end in zip(before._fields, before, after)}


def main():
    record = load_record((ROOT/'intransitive/benchmarks/search_budget/game79.pgn').read_text())
    archived, revision = baseline()
    archived.AlphaBetaPlayer()._prepare()
    search.AlphaBetaPlayer()._prepare()
    config = replace(record.config, max_depth=4, node_limit=10**9, time_limit=600)
    rows = []
    for ply in (0, 27, 35, 61):
        state = record.states[ply]
        node = SearchPosition(state)
        game = search.AlphaBetaPlayer().game
        action = int(node.legal()[0])
        def public():
            for _ in range(1000):
                game.getNextState(state, node.side, action)
        def compact():
            for _ in range(1000):
                node.push(action)
                node.pop()
        for kind, call in (('public_make_1000', public), ('compact_make_unmake_1000', compact)):
            rows.append(dict(ply=ply, kind=kind, **measure(call)))
        for name, module in (('baseline', archived), ('compact', search)):
            player = module.AlphaBetaPlayer(config=config)
            rows.append(dict(ply=ply, kind='search_depth4', variant=name,
                             **measure(lambda: player.analyze(state))))
    print(json.dumps(dict(baseline_revision=revision, native_allocation_counts=rows), indent=2))


if __name__ == '__main__':
    main()
