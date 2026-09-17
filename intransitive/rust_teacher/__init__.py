"""Experimental native label search; never enabled in live jobs implicitly."""
import json
from pathlib import Path
import selectors
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
BINARY = HERE / 'target/release/intransitive-rust-teacher'


class RustTeacher:
    def __init__(self, binary=BINARY):
        self.binary = Path(binary).resolve()
        self.process = subprocess.Popen([str(self.binary)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1)

    def request(self, command, timeout=10.):
        if self.process.poll() is not None:
            raise RuntimeError('Rust teacher exited')
        self.process.stdin.write(command + '\n')
        self.process.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                self.close()
                raise TimeoutError('Rust teacher response timed out')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Rust teacher closed its output')
        result = json.loads(line)
        if 'error' in result:
            raise ValueError(result['error'])
        return result

    @staticmethod
    def encode(state):
        from ..IntransitiveLogicNumba import validate_state
        validate_state(state)
        return state.tobytes().hex()

    def inspect(self, state, radius=3, weight=10.):
        return self.request(f'inspect {radius} {weight} {self.encode(state)}')

    def apply(self, state, action, modelling=True):
        result = self.request(f'apply {action} {int(modelling)} {self.encode(state)}')
        return np.frombuffer(bytes.fromhex(result['state_hex']), dtype=np.int8).reshape(9, 9, 84).copy()

    def analyze(self, state, *, depth=6, seconds=60., radius=3, weight=10.,
                proof_depth=2, proof_nodes=64, table_entries=50000, node_limit=1_000_000_000, reuse=False,
                nmp_enabled=False, nmp_min_depth=3, nmp_reduction=1,
                futility_enabled=False, futility_max_depth=2, futility_margin=1.):
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError('seconds must be finite and nonnegative')
        from ..heuristics.config import SearchConfig
        settings = dict(nmp_enabled=nmp_enabled, nmp_min_depth=nmp_min_depth,
            nmp_reduction=nmp_reduction, futility_enabled=futility_enabled,
            futility_max_depth=futility_max_depth, futility_margin=futility_margin)
        SearchConfig(**settings)
        options = f'{str(nmp_enabled).lower()} {nmp_min_depth} {nmp_reduction} ' + \
                  f'{str(futility_enabled).lower()} {futility_max_depth} {futility_margin}'
        command = 'search_reuse' if reuse else 'search'
        result = self.request(f'{command} {depth} {int(seconds*1000)} {node_limit} {radius} {weight} '
            f'{proof_depth} {proof_nodes} {table_entries} {options} {self.encode(state)}', timeout=seconds+10.)
        result['search_identity'] = dict(version='intransitive-selective-v1', backend='rust',
            depth=depth, seconds=seconds, node_limit=node_limit, radius=radius, weight=weight,
            proof_depth=proof_depth, proof_nodes=proof_nodes, table_entries=table_entries, **settings)
        result['selective']['depth'] = result['completed_depth']
        result['score_bound'] = ('selective_exact' if nmp_enabled or futility_enabled else 'exact') if result['score'] is not None else None
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2.)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        for pipe in (self.process.stdin, self.process.stdout):
            if not pipe.closed:
                pipe.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
