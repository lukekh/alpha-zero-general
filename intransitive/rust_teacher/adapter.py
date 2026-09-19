"""Opt-in native teacher adapter for durable supervised label workers."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from . import RustTeacher
from ..heuristics import SearchConfig
from ..heuristics.search import SearchResult


def validate_backend(config, backend):
    if backend.get('implementation') != 'rust-v1' or backend.get('node_limit_unit') != 'node_visits':
        raise ValueError('Unknown native backend identity or budget units')
    binary = Path(backend['binary'])
    if not binary.is_file() or hashlib.sha256(binary.read_bytes()).hexdigest() != backend['sha256']:
        raise ValueError('Native teacher binary checksum differs')
    defaults = SearchConfig()
    for key in ('evaluator_version','count_weight','advantage_weight','predator_zero_bonus',
                'predator_scarcity_bonus','prey_bonus','attack_enabled','defence_enabled','overload_enabled'):
        if getattr(config,key) != getattr(defaults,key):
            raise ValueError(f'Native teacher does not implement nondefault {key}')
    if config.max_depth > 32 or config.proof_depth > 2 or config.proof_nodes > 64:
        raise ValueError('Native depth/proof limits unsupported')


class RustAlphaBetaPlayer:
    def __init__(self, game, config, backend):
        validate_backend(config, backend)
        self.game, self.config, self.backend = game, config, dict(backend)
        self.table = range(0)
        self.client = None
        self.previous_handler = signal.getsignal(signal.SIGTERM)
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        try:
            self.client = RustTeacher(backend['binary'])
            signal.signal(signal.SIGTERM, self._terminate)
            self._registry(True)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    def _terminate(self, signum, frame):
        if self.client and self.client.process.poll() is None:
            self.client.process.terminate()
        raise KeyboardInterrupt

    def _registry(self, active):
        if not self.backend.get('worker_status_dir'):
            return
        path=Path(self.backend['worker_status_dir'])/f'rust-worker-{os.getpid()}.json'
        pending=path.with_suffix('.pending')
        pending.write_text(json.dumps(dict(parent_pid=os.getpid(),native_pid=self.client.process.pid,
            binary=self.backend['binary'],active=active,updated_epoch=time.time())))
        pending.replace(path)

    def analyze(self, state):
        c = self.config
        r = self.client.analyze(state, depth=c.max_depth, seconds=c.time_limit,
            radius=c.pressure_radius, weight=c.pressure_weight if c.pressure_enabled else 0.,
            proof_depth=c.proof_depth, proof_nodes=c.proof_nodes,
            table_entries=c.table_entries, node_limit=c.node_limit, reuse=True)
        self.table = range(r['table_entries'])
        return SearchResult(action=r['action'], score=r['score'], completed_depth=r['completed_depth'],
            pv=r['pv'], nodes=r['nodes'], work=r['nodes'], proof_nodes=r['proof_nodes'],
            elapsed=r['seconds'], stopped=not r['complete'], stop_reason=r['stop_reason'],
            tt_hits=r['tt_hits'])

    def close(self):
        try:
            if self.client is not None:
                self.client.close()
                self._registry(False)
        finally:
            signal.signal(signal.SIGTERM, self.previous_handler)

    def play(self, state):
        self.last_result = self.analyze(state)
        if self.last_result.action is None:
            raise RuntimeError('Native opponent returned no legal action')
        return self.last_result.action


def native_children(worker_pids, backend):
    """Capture exact descendants before their Python owners are reaped."""
    binary = backend['binary']
    rows = subprocess.check_output(['ps','-axo','pid=,ppid=,command='], text=True)
    found = []
    for line in rows.splitlines():
        columns = line.strip().split(None,2)
        if len(columns)==3 and int(columns[1]) in worker_pids and columns[2]==binary:
            found.append(int(columns[0]))
    if backend.get('worker_status_dir'):
        for pid in worker_pids:
            path=Path(backend['worker_status_dir'])/f'rust-worker-{pid}.json'
            if path.exists():
                record=json.loads(path.read_text())
                if record['parent_pid']==pid and record['binary']==binary and record['active']:
                    found.append(record['native_pid'])
    return sorted(set(found))


def reap_native(pids, backend):
    """Bounded fallback after killed/failed workers; never signal unrelated PIDs."""
    for pid in pids:
        command = subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True).stdout.strip()
        if command != backend['binary']:
            continue
        try:
            os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:
            continue
        deadline=time.monotonic()+1.
        while time.monotonic()<deadline:
            state=subprocess.run(['ps','-p',str(pid),'-o','state='],capture_output=True,text=True).stdout.strip()
            if not state or state.startswith('Z'):
                break
            time.sleep(.02)
        else:
            command=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True).stdout.strip()
            if command==backend['binary']:
                try:
                    os.kill(pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
