"""Frozen inputs, exact position identity, and completion-order-free schedules."""
from dataclasses import replace
import base64
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys

import numba
import numpy as np

from ..IntransitiveGame import IntransitiveGame
from ..IntransitiveLogicNumba import deserialize_state
from ..IntransitiveSymmetries import NUM_SYMMETRIES, transform_state
from ..heuristics.config import SearchConfig
from ..heuristics.tuning import EVALUATION_FIELDS, GENES, Genome
from ..record import state_hash

SCHEMA = 'intransitive-tournament-v1'
POOLS = ('search', 'validation', 'heldout')
MODES = ('depth', 'wall', 'mcts')
SCALES = tuple(('count' if name == 'material' else name) + '_weight' for name in GENES['python'])


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def backend_version():
    """Fingerprint evaluator, rules, adapter and search, including local edits."""
    root = Path(__file__).parents[1]
    files = sorted(root.glob('heuristics/*.py')) + sorted(root.glob('Intransitive*.py'))
    files += sorted(Path(__file__).parent.glob('*.py'))
    return digest(dict(files={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in files}, runtime=runtime_versions(), mcts_source=hashlib.sha256((root.parent / 'MCTS.py').read_bytes()).hexdigest()))


def runtime_versions():
    import llvmlite
    return dict(python=sys.version.split()[0], numpy=np.__version__, numba=numba.__version__,
                llvmlite=llvmlite.__version__)


def candidate(name, weights=None, *, role='population', backend='python', genome=None, variable_material_enabled=False):
    """Consume #53's serialized module-scale contract, restricted to Python.

    Material mode is a fixed per-candidate choice; signed scales are tunable.
    Optional zero scales disable their modules. Only Python match engines are
    supported by this harness.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError('Candidate name must be nonempty')
    if role not in ('population', 'incumbent', 'archive') or backend != 'python':
        raise ValueError('Unsupported candidate role/backend; only Python scales are supported')
    if genome is not None:
        if weights is not None:
            raise ValueError('Specify weights or a serialized genome, not both')
        validated = Genome.from_json(json.dumps(genome, allow_nan=False))
        if validated.backend != backend:
            raise ValueError('Genome backend does not match candidate backend')
    else:
        weights = {} if weights is None else weights
        if not isinstance(weights, dict) or set(weights) - set(SCALES):
            raise ValueError(f'Only effective scales {SCALES} are supported')
        validated = Genome.from_genes({('material' if k == 'count_weight' else k.removesuffix('_weight')): v for k, v in weights.items()},
                                      backend=backend)
    genome = validated.to_dict()
    config = replace(validated.to_config(), variable_material_enabled=variable_material_enabled)
    evaluation = {k: v for k, v in config.to_dict().items() if k in EVALUATION_FIELDS}
    identity = dict(backend=backend, backend_version=backend_version(), evaluation=evaluation)
    return dict(name=name, role=role, genome=genome, **identity, sha256=digest(identity))


def effective_config(item, protocol):
    return SearchConfig(**item['evaluation'], **protocol['search'])


def protocol(mode, *, depth=2, seconds=.05, node_limit=10**9, proof_depth=2,
             proof_nodes=64, max_plies=200, game_seconds=120., startup_seconds=120.,
             timeout_grace=5., completion_required=.8, simulations=32, cpuct=1., value_scale=400.):
    if mode not in MODES:
        raise ValueError('Expected depth or wall protocol')
    search = SearchConfig(max_depth=64 if mode == 'wall' else depth,
                          time_limit=seconds, node_limit=node_limit,
                          proof_depth=proof_depth, proof_nodes=proof_nodes)
    if search.max_depth < 1 or search.time_limit <= 0 or search.node_limit < 1:
        raise ValueError('Search limits must be positive')
    if type(max_plies) is not int or max_plies < 1:
        raise ValueError('max_plies must be a positive integer')
    for value in (game_seconds, startup_seconds, timeout_grace):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError('Safety limits must be finite and positive')
    if not 0 <= completion_required <= 1:
        raise ValueError('Invalid completion requirement')
    if type(simulations) is not int or simulations < 2 or simulations > 100000:
        raise ValueError('MCTS simulations must be in [2, 100000]')
    for value in (cpuct, value_scale):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('MCTS cpuct and value_scale must be positive and finite')
    result = dict(mode=mode, search={k: v for k, v in search.to_dict().items()
                                 if k not in EVALUATION_FIELDS},
                max_plies=max_plies, game_seconds=game_seconds,
                startup_seconds=startup_seconds, timeout_grace=timeout_grace,
                completion_required=completion_required,
                cache_policy='fresh search/evaluation caches per move',
                ranking='wins / scheduled games; unresolved worth zero; upper bound includes unresolved; '
                        'eligibility requires completion threshold, all games attempted, no failures')
    if mode == 'mcts':
        result['mcts'] = dict(simulations=simulations, cpuct=cpuct, value_scale=value_scale,
            prior='uniform legal', leaf_value='tanh(heuristic / value_scale)', tree='fresh per move', noise=False)
    return result


def pack(state):
    return base64.b64encode(state.tobytes()).decode('ascii')


def unpack(encoded):
    return deserialize_state(base64.b64decode(encoded, validate=True))


def orbit_hash(state):
    return min(state_hash(transform_state(state, s)) for s in range(NUM_SYMMETRIES))


def replay_opening(actions):
    game = IntransitiveGame(modelling_draws=False)
    state, side = game.getInitBoard(), 0
    for action in actions:
        if type(action) is not int or not 0 <= action < game.getActionSize() or not game.getValidMoves(state, side)[action]:
            raise ValueError('Illegal opening action')
        state, side = game.getNextState(state, side, action)
    return state


def position(actions, *, seed, stage, pool=None):
    state = replay_opening(actions)
    family = orbit_hash(state)
    # All positions along a generated line share one split as well. Orbit
    # validation below additionally prevents symmetry leakage across lines.
    pool = pool or POOLS[int(digest(['split', seed])[:8], 16) % len(POOLS)]
    if pool not in POOLS:
        raise ValueError('Unknown position pool')
    return dict(state=pack(state), sha256=state_hash(state), orbit_sha256=family,
                opening_actions=list(actions), seed=seed, stage=stage, pool=pool)


def generate_positions(seed=54, lines=9, max_plies=240):
    """Bounded capture-biased legal trajectories, with auditable opening lines.

    Stage means opening <= 12 plies, endgame <= 12 remaining pieces, otherwise
    midgame. Never manufacture an endgame by deleting pieces.
    """
    if type(lines) is not int or lines < 1 or type(max_plies) is not int or max_plies < 1:
        raise ValueError('Generation limits must be positive integers')
    output = [position([], seed=seed, stage='official', pool='search')]
    seen = {output[0]['orbit_sha256']}
    game = IntransitiveGame(modelling_draws=False)
    for line in range(lines):
        stream = seed + line + 1
        rng = np.random.default_rng(stream)
        state, side, actions, stages = game.getInitBoard(), 0, [], set()
        for ply in range(max_plies):
            legal = list(map(int, np.flatnonzero(game.getValidMoves(state, side))))
            if not legal:
                break
            from ..IntransitiveConstants import action_destination
            captures = [a for a in legal if state[action_destination(a)[1], action_destination(a)[0], 0]]
            action = int(rng.choice(captures if captures and rng.random() < .9 else legal))
            state, side = game.getNextState(state, side, action)
            actions.append(action)
            if game.getGameEnded(state, side).any():
                break
            stage = ('opening' if ply < 12 else 'endgame' if np.count_nonzero(state[:, :, 0]) <= 12 else 'midgame')
            if (stage == 'opening' and ply < 7) or stage in stages:
                continue
            item = position(actions, seed=stream, stage=stage)
            stages.add(stage)
            if item['orbit_sha256'] not in seen:
                output.append(item)
                seen.add(item['orbit_sha256'])
    return output


def manifest(candidates, positions, protocols, *, pool='search', position_limit=None):
    if pool not in POOLS or not candidates or not protocols:
        raise ValueError('Candidates, protocols and a valid pool are required')
    names, hashes = set(), set()
    for item in candidates:
        rebuilt = candidate(item['name'], genome=item['genome'], role=item['role'], backend=item['backend'],
                            variable_material_enabled=item['evaluation'].get('variable_material_enabled', False))
        if rebuilt != item or item['name'] in names or item['sha256'] in hashes:
            raise ValueError('Invalid, stale or duplicate deterministic candidate')
        names.add(item['name'])
        hashes.add(item['sha256'])
    if not any(c['role'] == 'population' for c in candidates) or len(candidates) < 2:
        raise ValueError('Need a population candidate and a distinct opponent')
    families = set()
    for item in positions:
        state = unpack(item['state'])
        if (state_hash(state) != item['sha256'] or orbit_hash(state) != item['orbit_sha256']
                or replay_opening(item['opening_actions']).tobytes() != state.tobytes()
                or item['pool'] not in POOLS or item['orbit_sha256'] in families):
            raise ValueError('Invalid opening/history/hash or duplicate symmetry family')
        families.add(item['orbit_sha256'])
    selected = sorted((p for p in positions if p['pool'] == pool), key=lambda p: (p['stage'] != 'official', p['sha256']))
    if position_limit is not None:
        if type(position_limit) is not int or position_limit < 1:
            raise ValueError('Position quota must be positive')
        if position_limit > len(selected):
            raise ValueError('Position quota exceeds available distinct positions')
        selected = selected[:position_limit]
    if not selected or len({p['mode'] for p in protocols}) != len(protocols):
        raise ValueError('Empty selected pool or duplicate protocol mode')
    for limits in protocols:
        # Validate all declared search options and safety/ranking policy on
        # load, not just when creating inputs through the convenience helper.
        settings = limits['search']
        rebuilt = protocol(limits['mode'], depth=settings['max_depth'], seconds=settings['time_limit'],
                           node_limit=settings['node_limit'], proof_depth=settings['proof_depth'],
                           proof_nodes=settings['proof_nodes'], max_plies=limits['max_plies'],
                           game_seconds=limits['game_seconds'], startup_seconds=limits['startup_seconds'],
                           timeout_grace=limits['timeout_grace'], completion_required=limits['completion_required'],
                           **({k: limits['mcts'][k] for k in ('simulations', 'cpuct', 'value_scale')} if limits['mode'] == 'mcts' else {}))
        # Other supported common SearchConfig options remain configurable.
        if set(settings) != set(rebuilt['search']):
            raise ValueError('Invalid shared search settings')
        effective_config(candidates[0], limits)
        rebuilt['search'] = settings
        if limits != rebuilt or (limits['mode'] == 'wall' and settings['max_depth'] != 64):
            raise ValueError('Invalid protocol')
    # Freeze the full corpus before selecting a pool; heldout remains sealed.
    result = dict(schema=SCHEMA, runtime=runtime_versions(), candidates=candidates, positions=positions,
                  protocols=protocols, pool=pool, selected_positions=[p['sha256'] for p in selected])
    tasks = []
    for protocol_index, limits in enumerate(protocols):
        for a, b in combinations(sorted(candidates, key=lambda c: c['sha256']), 2):
            if a['role'] != 'population' and b['role'] != 'population':
                continue
            for start in selected:
                pair = dict(protocol=protocol_index, candidates=[a['sha256'], b['sha256']],
                            position=start['sha256'], seed=start['seed'])
                pair_id = digest(dict(pair=pair, limits=limits))
                for colour in (0, 1):
                    colours = pair['candidates'][::1 if colour == 0 else -1]
                    task = dict(**pair, pair_id=pair_id, colours=colours)
                    tasks.append(dict(index=len(tasks), id=digest(task), **task))
    result['tasks'] = tasks
    result['sha256'] = digest(result)
    return result
