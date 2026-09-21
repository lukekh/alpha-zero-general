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
    def parallel_settings(threads, split_min_depth, split_min_siblings, table_entries,
                          futility_max_depth=2):
        """Validate the branch-parallel interlocks before a search is started.

        Every reason is stated here rather than discovered mid-search. The
        memory rule is a product because each thread owns a private
        transposition table; a shared one would be cheaper and would make the
        search nondeterministic, which #54 and #55 cannot accept.
        """
        for name, value, low, high in (('threads', threads, 1, 64),
                                       ('split_min_depth', split_min_depth, 2, 32),
                                       ('split_min_siblings', split_min_siblings, 2, 64)):
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{name} must be an integer in {low}..{high}')
        if split_min_depth <= futility_max_depth:
            raise ValueError('split_min_depth must exceed futility_max_depth: a split node hands '
                             'its tail to helpers, which do not apply the futility margin')
        if threads * table_entries > 1_000_000:
            raise ValueError('threads * table_entries must be <= 1000000: each thread owns a '
                             'private transposition table, so the ceiling is the product')
        return (threads, split_min_depth, split_min_siblings) != (1, 4, 2)

    @staticmethod
    def material_mode(enabled, linear=False):
        if type(enabled) is not bool or type(linear) is not bool:
            raise ValueError('variable material settings must be booleans')
        if linear and not enabled:
            raise ValueError('variable_material_linear requires variable_material_enabled')
        # Preserve compatibility with existing flat-material native binaries:
        # 0 and 1 mean what they always did, 2 is the new linear mode.
        return (' 2' if linear else ' 1') if enabled else ''

    def inspect(self, state, radius=4, weight=0., *, material=100., advantage=23.967050360966205,
                attack=25.714516982666414, defence=32.5643023919054, variable_material_enabled=False,
                variable_material_linear=False):
        mode = self.material_mode(variable_material_enabled, variable_material_linear)
        return self.request(f'inspect {radius} {weight} {material} {advantage} {attack} {defence}{mode} {self.encode(state)}')

    def apply(self, state, action, modelling=True):
        result = self.request(f'apply {action} {int(modelling)} {self.encode(state)}')
        return np.frombuffer(bytes.fromhex(result['state_hex']), dtype=np.int8).reshape(9, 9, 84).copy()

    def analyze(self, state, *, depth=6, seconds=60., radius=4, weight=0.,
                proof_depth=2, proof_nodes=64, table_entries=50000, node_limit=1_000_000_000, reuse=False,
                material=100., advantage=23.967050360966205,
                attack=25.714516982666414, defence=32.5643023919054, variable_material_enabled=False,
                variable_material_linear=False,
                mvv_lva_enabled=False, selective_evaluator_enabled=False, nmp_enabled=False, nmp_min_depth=3, nmp_reduction=1,
                futility_enabled=False, futility_max_depth=2, futility_margin=1., certificate_enabled=False,
                certificate_cutoff_enabled=False, certificate_cutoff_min_depth=2,
                certificate_guard_enabled=False,
                threads=1, split_min_depth=4, split_min_siblings=2):
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError('seconds must be finite and nonnegative')
        from ..heuristics.config import SearchConfig
        settings = dict(nmp_enabled=nmp_enabled, nmp_min_depth=nmp_min_depth,
            nmp_reduction=nmp_reduction, futility_enabled=futility_enabled,
            futility_max_depth=futility_max_depth, futility_margin=futility_margin)
        runs = dict(certificate_enabled=certificate_enabled,
            certificate_cutoff_enabled=certificate_cutoff_enabled,
            certificate_cutoff_min_depth=certificate_cutoff_min_depth,
            certificate_guard_enabled=certificate_guard_enabled)
        # Validate the request against the scales it will actually search with,
        # so the selective interlock decides the same way in both backends.
        SearchConfig(mvv_lva_enabled=mvv_lva_enabled,
                     selective_evaluator_enabled=selective_evaluator_enabled,
                     count_weight=material, advantage_weight=advantage,
                     attack_weight=attack, attack_enabled=bool(attack),
                     defence_weight=defence, defence_enabled=bool(defence),
                     variable_material_enabled=variable_material_enabled,
                     variable_material_linear=variable_material_linear,
                     pressure_enabled=bool(weight), pressure_weight=weight or 1.,
                     pressure_radius=radius, **runs, **settings)
        parallel = self.parallel_settings(threads, split_min_depth, split_min_siblings,
                                          table_entries, futility_max_depth)
        mode = self.material_mode(variable_material_enabled, variable_material_linear)
        options = f'{str(nmp_enabled).lower()} {nmp_min_depth} {nmp_reduction} ' + \
                  f'{str(futility_enabled).lower()} {futility_max_depth} {futility_margin}'
        # The certificate-as-a-bound group travels whole, so an old binary
        # rejects the request instead of running with its own defaults.
        bound = certificate_cutoff_enabled or certificate_guard_enabled
        # Each optional group is positional, so a later one implies the earlier.
        extended = selective_evaluator_enabled or mvv_lva_enabled or certificate_enabled or bound or parallel
        if extended:
            mode = self.material_mode(variable_material_enabled, variable_material_linear) or ' 0'
            options += ' ' + str(selective_evaluator_enabled).lower()
            if mvv_lva_enabled or certificate_enabled or bound or parallel:
                options += ' ' + str(mvv_lva_enabled).lower()
            if certificate_enabled or bound or parallel:
                options += ' ' + str(certificate_enabled).lower()
            # The parallel group sits behind the certificate-bound group, so
            # asking for threads means sending that group too, at its defaults.
            if bound or parallel:
                options += (f' {str(certificate_cutoff_enabled).lower()}'
                            f' {certificate_cutoff_min_depth}'
                            f' {str(certificate_guard_enabled).lower()}')
            if parallel:
                options += f' {threads} {split_min_depth} {split_min_siblings}'
        # Preserve the existing wire forms when all selective options are defaults.
        options = (' ' + options) if extended or settings != dict(nmp_enabled=False, nmp_min_depth=3,
            nmp_reduction=1, futility_enabled=False, futility_max_depth=2, futility_margin=1.) else ''
        command = 'search_reuse' if reuse else 'search'
        result = self.request(f'{command} {depth} {int(seconds*1000)} {node_limit} {radius} {weight} '
            f'{proof_depth} {proof_nodes} {table_entries} {material} {advantage} {attack} {defence}{mode}{options} {self.encode(state)}', timeout=seconds+10.)
        result['search_identity'] = dict(version='intransitive-selective-v1', backend='rust',
            depth=depth, seconds=seconds, node_limit=node_limit, radius=radius, weight=weight,
            proof_depth=proof_depth, proof_nodes=proof_nodes, table_entries=table_entries,
            material=material, advantage=advantage, attack=attack, defence=defence,
            variable_material_enabled=variable_material_enabled,
            mvv_lva_enabled=mvv_lva_enabled, selective_evaluator_enabled=selective_evaluator_enabled,
            threads=threads, split_min_depth=split_min_depth,
            split_min_siblings=split_min_siblings,
            # Reproducible for a fixed thread count, not across thread counts:
            # helpers hold private tables and a frozen window, so the tree a
            # given count searches is fixed, but a different count searches a
            # different tree. A search that runs out of time is reproducible in
            # neither mode, exactly as the single-threaded search already was.
            deterministic_for_thread_count=True, **runs, **settings)
        if 'selective' in result:
            result['selective']['depth'] = result['completed_depth']
        # A parallel result is exact minimax at its completed depth, but it is
        # not the single-threaded proof that the teacher-label paths accept,
        # so it never borrows the bound name a certificate carries.
        result['score_bound'] = (('selective_exact' if nmp_enabled or futility_enabled else
                                  'parallel_exact' if threads > 1 else 'exact')
                                 if result['score'] is not None else None)
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
