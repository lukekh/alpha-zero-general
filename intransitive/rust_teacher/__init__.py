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

    @staticmethod
    def material_mode(enabled):
        if type(enabled) is not bool:
            raise ValueError('variable_material_enabled must be a boolean')
        # Preserve compatibility with existing flat-material native binaries.
        return ' 1' if enabled else ''

    def inspect(self, state, radius=4, weight=0., *, material=100., advantage=23.967050360966205,
                attack=25.714516982666414, defence=32.5643023919054, variable_material_enabled=False):
        mode = self.material_mode(variable_material_enabled)
        return self.request(f'inspect {radius} {weight} {material} {advantage} {attack} {defence}{mode} {self.encode(state)}')

    def apply(self, state, action, modelling=True):
        result = self.request(f'apply {action} {int(modelling)} {self.encode(state)}')
        return np.frombuffer(bytes.fromhex(result['state_hex']), dtype=np.int8).reshape(9, 9, 84).copy()

    def analyze(self, state, *, depth=6, seconds=60., radius=4, weight=0.,
                proof_depth=2, proof_nodes=64, table_entries=50000, node_limit=1_000_000_000, reuse=False, material=100., advantage=23.967050360966205,
                attack=25.714516982666414, defence=32.5643023919054, variable_material_enabled=False):
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError('seconds must be finite and nonnegative')
        mode = self.material_mode(variable_material_enabled)
        command = 'search_reuse' if reuse else 'search'
        return self.request(f'{command} {depth} {int(seconds*1000)} {node_limit} {radius} {weight} '
            f'{proof_depth} {proof_nodes} {table_entries} {material} {advantage} {attack} {defence}{mode} {self.encode(state)}', timeout=seconds+10.)

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
