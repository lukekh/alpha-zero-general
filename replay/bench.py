"""Measured inference-batch latency: the real quantity beta2 stands in for.

Dream-RSI needs a parallelism *bonus* because it cannot price a hypothetical
policy's wall clock -- its probes are LLM calls with variable cost. Here the
network's batch-latency curve is a stable, cheap property of the machine, so
the proxy can be replaced by the measurement.

The curve also decides whether the question matters at all. Batch width in this
repository comes from `parallel_inferences` (one in-flight leaf per concurrent
game, see `GenericNNetWrapper.predict_server`), not from any search setting, so
a sweep cannot move it. Making it a search decision means collecting several
leaves per network call inside one search, which costs decision quality. That
is only worth building if this curve shows a throughput win to pay for it.
"""

import statistics
import time

import numpy as np

DEFAULT_SIZES = (1, 2, 4, 8, 16, 32, 64)


def batch_latency(net, boards, masks, sizes=DEFAULT_SIZES, repeats=25, warmup=5):
    """Median latency per ONNX call at each batch size, on real positions."""
    net.switch_target('inference')
    if net.current_mode != 'onnx':
        raise RuntimeError('batch latency needs the ONNX backend')
    session = net.ort_session
    boards = np.asarray(boards, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.bool_)

    rows = []
    for size in sizes:
        index = [i % len(boards) for i in range(size)]
        inputs = {'board': boards[index], 'valid_actions': masks[index]}
        for _ in range(warmup):
            session.run(None, inputs)
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            session.run(None, inputs)
            samples.append(time.perf_counter() - started)
        median = statistics.median(samples)
        rows.append(dict(
            batch=size,
            median_seconds=median,
            min_seconds=min(samples),
            per_position_seconds=median / size,
            positions_per_second=size / median,
        ))
    baseline = rows[0]['per_position_seconds']
    for row in rows:
        row['speedup_vs_batch1'] = baseline / row['per_position_seconds']
    return rows


def wall_clock_per_move(evals, collection, curve):
    """Modelled wall clock for `evals` leaf evaluations at collection size k.

    A search that gathers `collection` leaves per network call needs
    ceil(evals / collection) calls, each costing that batch size's measured
    latency. At collection 1 this is exactly today's sequential search.
    """
    latency = {row['batch']: row['median_seconds'] for row in curve}
    if collection not in latency:
        raise ValueError(f'no measured latency for batch {collection}; '
                         f'measured sizes are {sorted(latency)}')
    calls = -(-int(evals) // int(collection))
    return calls * latency[collection]
