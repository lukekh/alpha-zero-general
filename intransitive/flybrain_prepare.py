"""Fetch the pinned upstream model and reproduce its 384 neural simulations.

The heavy build dependencies and connectome stay in an isolated, ignored
checkout. Playing with the resulting bank needs only the normal dependencies.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

from .flybrain import DEFAULT_BANK, FEATURE_ORDER, UPSTREAM_COMMIT, UPSTREAM_URL

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'intransitive/data/flybrain-intransitive'
DATA_URL = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
TABLES = ('body-annotations-male-cns-v1.0-minconf-0.5.feather',
          'body-neurotransmitters-male-cns-v1.0.feather',
          'connectome-weights-male-cns-v1.0-minconf-0.5.feather')
BUILD_DEPENDENCIES = ('numpy==2.4.6', 'scipy==1.17.1', 'pandas==3.0.5', 'pyarrow==25.0.1')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_source(source):
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f'Expected upstream commit {UPSTREAM_COMMIT}; found {commit}')
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', 'HEAD', '--',
                    'brain.py', 'experiment.py', 'rps2.py', 'build_connectome.py'], check=True)


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', UPSTREAM_URL + '.git', str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'checkout', '--detach', UPSTREAM_COMMIT], check=True)
    validate_source(source)
    python = source / '.venv/bin/python'
    if not python.exists():
        subprocess.run([sys.executable, '-m', 'venv', str(source / '.venv')], check=True)
    subprocess.run([str(python), '-m', 'pip', 'install', *BUILD_DEPENDENCIES], check=True)
    (source / 'data').mkdir(exist_ok=True)
    for name in TABLES:
        path = source / 'data' / name
        # HEAD also catches incomplete downloads left by upstream's downloader.
        with urllib.request.urlopen(urllib.request.Request(DATA_URL + name, method='HEAD')) as response:
            size = int(response.headers['Content-Length'])
        if path.exists() and path.stat().st_size == size:
            continue
        print(f'Downloading {name} ({size:,} bytes)', flush=True)
        pending = path.with_suffix('.pending')
        urllib.request.urlretrieve(DATA_URL + name, pending)
        if pending.stat().st_size != size:
            raise ValueError(f'Incomplete download: {name}')
        pending.replace(path)
    subprocess.run([str(python), 'build_connectome.py'], cwd=source, check=True)
    subprocess.run([str(python), '-m', 'intransitive.flybrain_prepare', '--build',
                    '--source', str(source), '--output', str(output)], cwd=ROOT, check=True)


def build(source, output):
    import numpy as np
    source, output = Path(source).resolve(), Path(output).resolve()
    validate_source(source)
    # This runs in the isolated build interpreter, never in a playing process.
    sys.path.insert(0, str(source))
    import brain
    import experiment
    matrix, neurons = brain.load(str(source / 'data') + '/')
    simulator = brain.Brain(brain.stabilise(matrix, neurons), neurons, seed=1)
    if tuple(experiment.FEATURES) != FEATURE_ORDER or experiment.REPEATS != 12 or experiment.STIM_MS != 80:
        raise ValueError('Unexpected upstream experiment settings')
    print(f'Recording 384 responses from {simulator.n:,} neurons', flush=True)
    bank = experiment.record_bank(simulator)
    patterns = np.array(list(bank), dtype=np.int8)
    wired = np.stack([bank[tuple(key)]['wired'] for key in patterns])
    metadata = dict(format_version=1, upstream_url=UPSTREAM_URL, upstream_commit=UPSTREAM_COMMIT,
        features=list(FEATURE_ORDER), stim_ms=80, repeats=12, brain_seed=1, mode='wired',
        neurons=simulator.n, stabilized_connections=int(simulator.W.nnz),
        source_sha256={name: digest(source/name) for name in
                       ('brain.py', 'experiment.py', 'rps2.py', 'build_connectome.py')},
        data_sha256={name: digest(source/'data'/name) for name in TABLES},
        dependencies=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True),
        python=sys.version, connectome_license='CC-BY-4.0; Google Research & HHMI Janelia FlyEM MaleCNS v1.0')
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix('.pending')
    with pending.open('wb') as stream:
        np.savez_compressed(stream, patterns=patterns, wired=wired, metadata=json.dumps(metadata))
    pending.replace(output)
    print(f'Saved {output} (sha256 {digest(output)})', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_BANK)
    parser.add_argument('--build', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    (build if args.build else prepare)(args.source, args.output)
