"""Independently verify every measured PV suffix with full-window alpha-beta."""
from pathlib import Path
import json
import sys
from math import inf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from intransitive.heuristics import AlphaBetaPlayer, SearchConfig
from intransitive.heuristics.budget import Budget
from intransitive.heuristics.search import from_table
from intransitive.record import load_record


def main():
    evidence = Path(__file__).with_name('evidence')
    report = json.loads((evidence/'results.json').read_text())
    record = load_record((ROOT/'intransitive/benchmarks/search_budget/game79.pgn').read_text())
    seen = set()
    suffixes = 0
    for row in report['runs']:
        key = row['ply'], row['selected_depth'], row['score'], row['score_bound'], tuple(row['pv'])
        if key in seen or not row['selected_depth']:
            continue
        seen.add(key)
        config = dict(report['configs'][row['variant']+'/'+row['mode']])
        config.update(pvs_enabled=False, aspiration_enabled=False, time_limit=600, node_limit=10**9)
        player = AlphaBetaPlayer(config=SearchConfig(**config))
        player._prepare()
        state = record.states[row['ply']]
        for ply, action in enumerate(row['pv'], 1):
            side = int(state[:, :, 82:84].flat[1])
            assert player.game.getValidMoves(state, side)[action]
            state, _ = player.game.getNextState(state, side, action)
            if ply > row['selected_depth']:
                continue  # The appended bounded-proof certificate has its own horizon.
            value, _ = player._search(state, row['selected_depth']-ply,
                                      -inf, inf, 0, Budget(10**9, 600))
            root_value = from_table(value if ply % 2 == 0 else -value, ply)
            if row['score_bound'] == 'exact':
                assert root_value == row['score'], (key, ply, root_value)
            elif ply == 1:
                # A bound-only PV need not describe optimal deeper replies.
                assert row['score_bound'] == 'upper' and row['score'] >= root_value, (key, root_value)
            suffixes += 1
    print(f'Validated {len(seen)} distinct measured selected lines and {suffixes} main-search suffixes.')


if __name__ == '__main__':
    main()
