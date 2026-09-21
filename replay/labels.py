"""Label recorded roots with an Intransitive minimax teacher.

This is the independent reference. Agreement with the recorded search only says
which configuration converges to the recording's own opinion, which is the right
question for a compute-allocation sweep but says nothing about whether that
opinion is any good. The teacher is an outside opinion.

Labels are cheap on purpose. Standard practice for learned-evaluation data is
many shallow fixed-node labels rather than a few deep ones, so the defaults are
a node budget, not the 120-second depth-5 settings of
`intransitive.supervised_minimax`.

Two backends:
  python  `intransitive.heuristics.AlphaBetaPlayer`, always available
  rust    the native teacher, far faster, but the checked-in binary must match
          the Python client's wire format; rebuild it or pass --binary
"""

import contextlib
import hashlib
import json
import pickle
import time
import zlib
from pathlib import Path

LABELS_SCHEMA = 'az-replay-teacher-labels-v1'
LABELS_NAME = 'teacher-labels.pkl.z'
LABELS_MANIFEST = 'teacher-labels-manifest.json'


def _complete(completed_depth, target_depth, stop_reason):
    """The same completion rule `supervised_minimax` uses for teacher labels."""
    return bool(completed_depth is not None
                and (completed_depth >= target_depth or stop_reason == 'proven_result'))


def _binary_identity(path):
    """A stale native binary silently changes labels, so pin which one ran."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return dict(binary=str(path), binary_sha256=digest,
                binary_bytes=path.stat().st_size if path.is_file() else None)


@contextlib.contextmanager
def _rust_engine(depth, seconds, node_limit, binary, **teacher_args):
    from intransitive.rust_teacher import BINARY, RustTeacher

    teacher = RustTeacher(binary) if binary else RustTeacher()
    identity = _binary_identity(binary or BINARY)
    with teacher:
        def analyse(state):
            result = teacher.analyze(state, depth=depth, seconds=seconds,
                                     node_limit=node_limit, **teacher_args)
            return dict(action=result.get('action'), score=result.get('score'),
                        completed_depth=result.get('completed_depth'),
                        stop_reason=result.get('stop_reason'),
                        complete=bool(result.get('complete')))
        yield analyse, identity


@contextlib.contextmanager
def _python_engine(depth, seconds, node_limit, binary=None, **teacher_args):
    from intransitive.heuristics import AlphaBetaPlayer, SearchConfig

    if binary:
        raise ValueError('--binary applies to the rust engine only')
    config = SearchConfig(max_depth=depth, time_limit=seconds,
                          node_limit=node_limit, **teacher_args)
    player = AlphaBetaPlayer(config=config)
    identity = dict(search_version=config.search_version)

    def analyse(state):
        result = player.analyze(state)
        return dict(action=result.action, score=result.score,
                    completed_depth=result.completed_depth,
                    stop_reason=result.stop_reason,
                    complete=_complete(result.completed_depth, depth, result.stop_reason))
    yield analyse, identity


ENGINES = {'python': _python_engine, 'rust': _rust_engine}


def label_positions(positions, *, engine='python', depth=6, seconds=5.0,
                    node_limit=200_000, binary=None, accept_depth=None,
                    progress=None, **teacher_args):
    """Return (labels, records, stats). `labels` maps a pool key to an action.

    `accept_depth` is the floor a partial iterative-deepening result must reach
    to be kept. Left unset it requires the full target depth, matching
    `supervised_minimax`; setting it lower buys many more labels per second at a
    stated, recorded quality floor rather than an unstated one.
    """
    if engine not in ENGINES:
        raise ValueError(f'unknown teacher engine {engine!r}')
    unique = {}
    for position in positions:
        unique.setdefault(position['key'], position)

    labels, records = {}, []
    stats = dict(engine=engine, requested=0, labelled=0, incomplete=0, failed=0,
                 depth_histogram={})
    started = time.perf_counter()
    with ENGINES[engine](depth, seconds, node_limit, binary, **teacher_args) as (analyse, identity):
        stats['engine_identity'] = identity
        for index, (key, position) in enumerate(unique.items()):
            stats['requested'] += 1
            try:
                result = analyse(position['state'])
            except Exception as error:  # a teacher failure is data, not a crash
                stats['failed'] += 1
                records.append(dict(key=key.hex(), error=str(error)))
                continue
            reached = result['completed_depth']
            floor = depth if accept_depth is None else accept_depth
            accepted = result['complete'] or (reached is not None and reached >= floor)
            histogram = stats['depth_histogram']
            histogram[str(reached)] = histogram.get(str(reached), 0) + 1
            if result['action'] is None:
                stats['failed'] += 1
            elif not accepted:
                stats['incomplete'] += 1
            else:
                labels[key] = int(result['action'])
                stats['labelled'] += 1
            records.append(dict(key=key.hex(), **result))
            if progress is not None:
                progress(index + 1, len(unique), stats)
    stats['accept_depth'] = depth if accept_depth is None else accept_depth
    stats['unique_positions'] = len(unique)
    stats['seconds'] = time.perf_counter() - started
    stats['seconds_per_position'] = stats['seconds'] / max(len(unique), 1)
    stats['coverage'] = stats['labelled'] / max(len(unique), 1)
    return labels, records, stats


def save_labels(directory, labels, records, stats, settings):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    packed = zlib.compress(pickle.dumps(labels, protocol=pickle.HIGHEST_PROTOCOL), level=6)
    (directory / LABELS_NAME).write_bytes(packed)
    manifest = dict(schema=LABELS_SCHEMA, created=time.strftime('%Y-%m-%dT%H:%M:%S'),
                    settings=settings, stats=stats,
                    sha256=hashlib.sha256(packed).hexdigest(), records=records)
    (directory / LABELS_MANIFEST).write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def load_labels(directory, verify=True):
    directory = Path(directory)
    manifest = json.loads((directory / LABELS_MANIFEST).read_text())
    packed = (directory / LABELS_NAME).read_bytes()
    if verify and hashlib.sha256(packed).hexdigest() != manifest['sha256']:
        raise ValueError('teacher labels failed their checksum')
    return pickle.loads(zlib.decompress(packed)), manifest
