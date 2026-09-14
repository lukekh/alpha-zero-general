"""Measure the fixed v1 baseline: python -m intransitive.benchmark_network."""

import argparse
import json
import platform

import torch
from torch.utils.benchmark import Timer

from .IntransitiveGame import IntransitiveGame
from .IntransitiveNNet import IntransitiveNNet


def benchmark(batch_sizes=(1, 32), repeats=100):
    torch.set_num_threads(1)
    torch.manual_seed(10)
    game = IntransitiveGame()
    net = IntransitiveNNet(game, {'nn_version': 2}).eval()
    state = torch.tensor(game.getInitBoard(), dtype=torch.float32)[None]
    valid = torch.tensor(game.getValidMoves(game.getInitBoard(), 0))[None]
    report = {
        'platform': platform.platform(), 'processor': platform.processor(),
        'python': platform.python_version(), 'torch': torch.__version__,
        'device': 'cpu', 'threads': 1, 'repeats': repeats,
        'parameters': sum(p.numel() for p in net.parameters()),
        'history_encoder_parameters': sum(p.numel() for p in net.history_encoder.parameters()),
        'history_conv_macs_per_state': 81 * 9 * 9 * 8 * 4 * 3 * 3,
        'feature_config': net.feature_config, 'measurements': [],
    }
    with torch.inference_mode():
        for batch_size in batch_sizes:
            boards = state.repeat(batch_size, 1, 1, 1)
            masks = valid.repeat(batch_size, 1)
            current, history = net.extract_features(boards)
            env = dict(net=net, boards=boards, masks=masks, history=history)
            timings = {}
            for name, expression in (
                ('extract_features', 'net.extract_features(boards)'),
                ('encode_history', 'net.encode_history(history)'),
                ('forward', 'net(boards, masks)'),
            ):
                timings[name + '_ms'] = Timer(expression, globals=env, num_threads=1).timeit(repeats).median * 1000
            report['measurements'].append({
                'batch_size': batch_size, **timings,
                'decoded_features_bytes': (current.numel() + history.numel()) * 4,
                'encoded_history_bytes': batch_size * 81 * 4 * 9 * 9 * 4,
            })
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=100)
    options = parser.parse_args()
    print(json.dumps(benchmark(repeats=options.repeats), indent=2))
