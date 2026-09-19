"""Fixed-work native scaling benchmark, with two promoted roots per family.

This does not pause or alter live jobs. Callers must arrange a representative
CPU workload. Native processes, not Python threads, perform the searches.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from statistics import mean
import time

from . import RustTeacher


def family(binary, state_hex, reuse=True):
    rows = []
    with RustTeacher(binary) as client:
        for ply in range(3):
            command = 'search_reuse' if reuse else 'search'
            result = client.request(
                f'{command} 6 300000 1000000000 4 0 2 64 50000 {state_hex}', timeout=310.)
            if result['stop_reason'] == 'terminal':
                break
            if not result['complete']:
                raise RuntimeError(f'Incomplete benchmark search: {result}')
            rows.append(result)
            if ply < 2:
                state_hex = client.request(f'apply {result["action"]} 1 {state_hex}')["state_hex"]
    return rows


def benchmark(binary, positions, output):
    states = [p.read_bytes().hex() for p in sorted(Path(positions).glob('position-*.bin'))]
    if len(states) != 6:
        raise ValueError('Expected six saved opening/midgame/endgame positions')
    report = dict(trials=[], notes=[
        'Identical 12-family workload per trial, each root followed by up to two promoted roots.',
        'Includes native startup/IPC, excludes Python sampling, ablations and corpus writes.',
        'Depth six, material heuristic, 50,000-entry table; fly-brain trainer remains running.'])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    reference = None
    for workers in (4,6,8,8,6,4):
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda s: family(binary,s), states*2))
        seconds = time.perf_counter()-started
        signature = [[(r['action'],r['score'],r['completed_depth']) for r in rows] for rows in results]
        if reference is None:
            reference = signature
        assert signature == reference, 'Parallelism changed search results'
        row = dict(workers=workers, seconds=seconds, labels=sum(map(len,results)),
                   tt_hits=sum(r['tt_hits'] for rows in results for r in rows),
                   promoted_tt_hits=sum(r['tt_hits'] for rows in results for r in rows[1:]))
        report['trials'].append(row)
        output.write_text(json.dumps(report,indent=2))
        print(json.dumps(row),flush=True)
    averages = {n:mean(r['seconds'] for r in report['trials'] if r['workers']==n) for n in (4,6,8)}
    best = min(averages,key=averages.get)
    # Don't consume more CPU for a difference within ordinary run-to-run noise.
    report.update(mean_seconds=averages, speedup={n:averages[4]/v for n,v in averages.items()},
                  selected_workers=best if averages[best] < .95*averages[4] else 4,
                  all_parallel_results_identical=True)
    # A separate cold/reused paired check: same physical child, same depth.
    with RustTeacher(binary) as client:
        root = client.request(f'search_reuse 6 300000 1000000000 4 0 2 64 50000 {states[0]}',310.)
        child = client.request(f'apply {root["action"]} 1 {states[0]}')['state_hex']
        warm = client.request(f'search_reuse 6 300000 1000000000 4 0 2 64 50000 {child}',310.)
        cold = client.request(f'search 6 300000 1000000000 4 0 2 64 50000 {child}',310.)
        assert warm['complete'] and cold['complete'] and abs(warm['score']-cold['score']) < 1e-8
        report['child_reuse'] = dict(reused=warm,fresh=cold)
    output.write_text(json.dumps(report,indent=2))
    return report
