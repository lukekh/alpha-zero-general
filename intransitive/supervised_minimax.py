"""Synthetic reachable positions, depth-five teacher labels, symmetry imitation.

No reinforcement-learning replay or heuristic-as-value labels are used.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import time
import traceback
import zlib

os.environ['ORT_DISABLE_TELEMETRY'] = '1'
import onnxruntime as ort
ort.disable_telemetry_events()
import numpy as np
import torch
from .greedy_process import SETTINGS, GameProcesses, cpu_seconds, make_net, seed_all
from .heuristics import AlphaBetaPlayer, SearchConfig
from .heuristics.budget import Budget
from .heuristics.evaluation import Evaluator
from .IntransitiveConstants import action_destination
from .IntransitiveGame import IntransitiveGame
from .IntransitiveDisplay import parse_move
from .IntransitiveLogicNumba import validate_state
from .IntransitiveSymmetries import NUM_SYMMETRIES, transform_action_vector, transform_state

STAGES = ('opening', 'midgame', 'endgame')
TEACHER = dict(max_depth=5, node_limit=1_000_000_000, time_limit=120.)
_GAME_OFFSETS = None


def completed_teacher(result, target_depth):
    return result.completed_depth >= target_depth or result.stop_reason == 'proven_result'


def valid_teacher_record(record):
    # Historical records predate this field and were explicitly generated at 5.
    target = record.get('teacher_target_depth', 5)
    return target >= 5 and (record['teacher_depth'] >= target or record['teacher_reason'] == 'proven_result')


def position_stage(pieces, ply):
    return 'endgame' if pieces<=8 else 'opening' if pieces>=18 and ply<=20 else 'midgame'


def recorded_position(seed, stage, settings):
    global _GAME_OFFSETS
    if _GAME_OFFSETS is None:
        _GAME_OFFSETS=json.loads(Path(settings['recorded_offsets']).read_text())
    rng=np.random.default_rng(seed)
    game=IntransitiveGame(modelling_draws=False)
    model=IntransitiveGame()
    with Path(settings['recorded_games']).open('rb') as stream:
        for _ in range(48):
            offset=int(rng.choice(_GAME_OFFSETS))
            stream.seek(offset)
            raw=json.loads(stream.readline())
            board,side,actions=game.getInitBoard(),0,[]
            chosen=None
            matches=0
            for move in raw.get('moveHistory',[]):
                if game.getGameEnded(board,side).any():
                    break
                try:
                    text=move['move']
                    action=parse_move(text[:2]+' '+text[2:])
                    if not game.getValidMoves(board,side)[action]:
                        break
                    board,side=game.getNextState(board,side,action)
                except (ValueError,KeyError,TypeError):
                    break
                actions.append(action)
                count=int(np.count_nonzero(board[:,:,0]))
                if len(actions)<2 or position_stage(count,len(actions))!=stage or game.getGameEnded(board,side).any():
                    continue
                observed=model.getSearchObservation(game.getCanonicalForm(board,side))
                if model.getGameEnded(observed,0).any():
                    continue
                matches+=1
                if rng.integers(matches)==0:
                    chosen=(observed,dict(seed=seed,stage=stage,ply=len(actions),pieces=count,
                        physical_player=side,actions=list(actions),source='recorded_game',
                        game_id=raw.get('gameID',str(offset)),source_offset=offset))
            if chosen is not None:
                return chosen
    state,provenance=generate_position(seed,stage)
    return state,dict(provenance,recorded_sampling_fallback=True)


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


def save_shard(path, records):
    pending = Path(str(path) + '.pending')
    with pending.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as stream:
            pickle.dump(records, stream, protocol=5)
    pending.replace(path)


def generate_position(seed, stage, max_attempts=24):
    """One independent legal trajectory per example, no arbitrary piece deletion.

    Bias captures and avoid immediate corner-winning moves when alternatives
    exist, to reach sparse positions. This is a synthetic sampling distribution,
    not a claim about normal-play position frequencies.
    """
    if stage not in STAGES:
        raise ValueError('Unknown game stage')
    rng = np.random.default_rng(seed)
    live = IntransitiveGame(modelling_draws=False)
    model = IntransitiveGame()
    for attempt in range(max_attempts):
        board, side, actions = live.getInitBoard(), 0, []
        target_ply = int(rng.integers(2, 17)) if stage == 'opening' else int(rng.integers(20, 61))
        target_pieces = 18 if stage == 'opening' else int(rng.integers(9, 18) if stage=='midgame' else rng.integers(3, 9))
        for ply in range(600):
            if live.getGameEnded(board, side).any():
                break
            count = int(np.count_nonzero(board[:, :, 0]))
            ready = (ply >= target_ply and ((stage=='opening' and count>=18)
                or (stage=='midgame' and 9<=count<=target_pieces)
                or (stage=='endgame' and 2<=count<=target_pieces)))
            if ready:
                canonical = live.getCanonicalForm(board, side)
                observed = model.getSearchObservation(canonical)
                validate_state(observed)
                if not model.getGameEnded(observed, 0).any():
                    return observed, dict(seed=seed, stage=stage, attempt=attempt,
                        ply=ply, pieces=count, physical_player=side, actions=actions,
                        observation_rebased=not np.array_equal(canonical, observed))
            if stage=='opening' and (ply>=16 or count<18):
                break
            if stage=='midgame' and count<9:
                break
            legal = np.flatnonzero(live.getValidMoves(board, side))
            goal = (8, 8) if side==0 else (0, 0)
            moves = [int(a) for a in legal if action_destination(int(a)) != goal]
            if not moves:
                break
            captures = [a for a in moves if board[action_destination(a)[1], action_destination(a)[0], 0] != 0]
            choices = captures if captures and rng.random()<.65 else moves
            action = int(rng.choice(choices))
            actions.append(action)
            board, side = live.getNextState(board, side, action)
    raise RuntimeError(f'Could not sample {stage} position after {max_attempts} trajectories')


def symmetries(state, action, legal):
    game = IntransitiveGame()
    pi = np.zeros(game.getActionSize(), dtype=np.float32)
    pi[action] = 1.
    for symmetry in range(NUM_SYMMETRIES):
        transformed = transform_state(state, symmetry)
        mover = int(transformed[:, :, 82:84].flat[1])
        transformed = game.getCanonicalForm(transformed, mover)
        yield transformed, transform_action_vector(pi, symmetry), transform_action_vector(legal, symmetry)


def orbit_key(state, action, legal):
    # Deliberately coarser than search identity: equal piece arrangements with
    # differing histories must not leak across held-out splits either.
    members = [b[:, :, 0].tobytes() + bytes([int(b[:, :, 82:84].flat[2])])
               for b, _, _ in symmetries(state, action, legal)]
    return hashlib.sha256(min(members)).hexdigest()


def split_for(key):
    bucket = int(key[:12], 16) % 100
    return 'train' if bucket<80 else 'validation' if bucket<90 else 'test'


def policy_examples(record):
    zero = np.zeros(2, dtype=np.float32)
    mask = np.zeros(2, dtype=np.bool_)
    return [zlib.compress(pickle.dumps((b, p.astype(np.float32), zero, v.astype(np.bool_),
            zero, mask, mask), protocol=5), level=1)
            for b, p, v in symmetries(record['state'], record['action'], record['legal'])]


class SymmetryExamples:
    """All twelve variants, materialized only for sampled minibatches."""
    def __init__(self, records):
        self.records = records
        self.sampled_sources = Counter()
        self.sampled_origins = Counter()

    def __len__(self):
        return NUM_SYMMETRIES * len(self.records)

    def __getitem__(self, index):
        record, symmetry = self.records[index // NUM_SYMMETRIES], index % NUM_SYMMETRIES
        self.sampled_sources[record['provenance'].get('source', 'synthetic_root')] += 1
        self.sampled_origins[record.get('ingested_from', 'trainer_or_supplement')] += 1
        game = IntransitiveGame()
        state = transform_state(record['state'], symmetry)
        state = game.getCanonicalForm(state, int(state[:, :, 82:84].flat[1]))
        pi = np.zeros(648, dtype=np.float32)
        pi[record['action']] = 1.
        policy = transform_action_vector(pi, symmetry).astype(np.float32)
        legal = transform_action_vector(record['legal'], symmetry).astype(np.bool_)
        zero = np.zeros(2, dtype=np.float32)
        mask = np.zeros(2, dtype=np.bool_)
        return state, policy, zero, legal, zero, mask, mask


def perturb_position(state, action, rng):
    """Well-formed counterfactuals; not represented as reachable trajectories."""
    pieces = state[:, :, 0].copy()
    source = ((action//8)%9, (action//8)//9)
    protected = {source, action_destination(action), (0,0), (8,8)}
    kind = int(rng.integers(1,4))
    operation = str(rng.choice(('remove_pair', 'add_pair', 'relocate')))
    edits = []
    def squares(code):
        return [(int(x),int(y)) for y,x in np.argwhere(pieces==code) if (int(x),int(y)) not in protected]
    def edit(square, code):
        x,y=square
        edits.append(dict(x=x,y=y,before=int(pieces[y,x]),after=int(code)))
        pieces[y,x]=code
    if operation=='remove_pair':
        for sign in (1,-1):
            choices=squares(sign*kind)
            if not choices:
                return None
            edit(choices[int(rng.integers(len(choices)))], 0)
    elif operation=='add_pair':
        for sign in (1,-1):
            if np.count_nonzero(pieces==sign*kind)>=(4 if kind==3 else 3):
                return None
            choices=squares(0)
            if not choices:
                return None
            edit(choices[int(rng.integers(len(choices)))], sign*kind)
    else:
        code=kind * int(rng.choice((-1,1)))
        origins, destinations=squares(code),squares(0)
        if not origins or not destinations:
            return None
        edit(origins[int(rng.integers(len(origins)))],0)
        edit(destinations[int(rng.integers(len(destinations)))],code)
    if not (np.any(pieces>0) and np.any(pieces<0)):
        return None
    variant=np.zeros_like(state)
    variant[:,:,0]=variant[:,:,1]=pieces
    meta=variant[:,:,82:84]
    meta.flat[:10]=state[:,:,82:84].reshape(-1)[:10]
    meta.flat[3],meta.flat[4],meta.flat[10]=0,1,0
    validate_state(variant)
    return variant, dict(operation=operation,edits=edits,history='fresh one-position counterfactual history')


def verified_ablations(parent, parent_result, config, max_searches=2):
    game=IntransitiveGame()
    evaluator=Evaluator(game,config)
    base_score=evaluator.score(parent['state'],0,Budget(1_000_000,30.),proof=dict(status='unknown'))
    rng=np.random.default_rng(parent['provenance']['seed']+987654)
    attempted=0
    for _ in range(24):
        candidate=perturb_position(parent['state'],parent['action'],rng)
        if candidate is None:
            continue
        state,edit=candidate
        count=int(np.count_nonzero(state[:,:,0]))
        stage=parent['provenance']['stage']
        if not ((stage=='opening' and 18<=count<=20) or (stage=='midgame' and 9<=count<=17)
                or (stage=='endgame' and 2<=count<=8)):
            continue
        if game.getGameEnded(state,0).any():
            continue
        legal=game.getValidMoves(state,0)
        if not legal[parent['action']]:
            continue
        static=evaluator.score(state,0,Budget(1_000_000,30.),proof=dict(status='unknown'))
        if not np.isclose(static,base_score,rtol=0,atol=1e-8):
            continue
        result=AlphaBetaPlayer(game,config).analyze(state)
        attempted+=1
        complete=completed_teacher(result, config.max_depth)
        if complete and result.action==parent['action'] and np.isclose(result.score,parent_result.score,rtol=0,atol=1e-8):
            record=dict(parent,state=state,legal=legal,orbit=orbit_key(state,result.action,legal),
                teacher_depth=result.completed_depth,teacher_reason=result.stop_reason,
                teacher_target_depth=config.max_depth,
                teacher_seconds=result.elapsed,teacher_work=result.work,
                provenance=dict(parent['provenance'],pieces=count,counterfactual=True),
                ablation=dict(edit,parent_orbit=parent['orbit'],static_score=static,
                    parent_static_score=base_score,teacher_score=result.score,parent_teacher_score=parent_result.score))
            return [record],attempted
        if attempted>=max_searches:
            break
    return [],attempted


def promote_tree_roots(teacher, root, result, parent, limit=2):
    """Search physical descendants without recanonicalizing the TT identity.

    Canonicalize only the saved training observations. Internal depth-4/3
    entries remain search hints/bounds, never copied directly into labels.
    """
    game=teacher.game
    records=[]
    side=int(root[:,:,82:84].flat[1])
    actions=list(parent['provenance']['actions'])
    for promoted_ply in range(1,limit+1):
        action=int(result.action)
        root,side=game.getNextState(root,side,action)
        actions.append(action)
        if game.getGameEnded(root,side).any():
            break
        previous_entries=len(teacher.table)
        result=teacher.analyze(root)
        if not completed_teacher(result, teacher.config.max_depth):
            break
        canonical=game.getCanonicalForm(root,side)
        legal=game.getValidMoves(canonical,0)
        assert legal[result.action]
        count=int(np.count_nonzero(root[:,:,0]))
        ply=parent['provenance']['ply']+promoted_ply
        records.append(dict(state=canonical,legal=legal,action=result.action,
            split=parent['split'],family=parent['family'],orbit=orbit_key(canonical,result.action,legal),
            provenance=dict(parent['provenance'],actions=list(actions),ply=ply,pieces=count,
                stage=position_stage(count,ply),physical_player=parent['provenance']['physical_player']^(promoted_ply%2),
                source='promoted_tree_root',parent_orbit=parent['orbit'],promoted_plies=promoted_ply),
            teacher_depth=result.completed_depth,teacher_reason=result.stop_reason,
            teacher_target_depth=teacher.config.max_depth,
            teacher_seconds=result.elapsed,teacher_work=result.work,teacher_score=result.score,
            tt_entries_available=previous_entries,tt_hits=result.tt_hits))
    return records


def label_result_path(settings, seed):
    identity = hashlib.sha256(json.dumps(settings['teacher'],sort_keys=True).encode()).hexdigest()[:16]
    return Path(settings['output'])/'worker-results'/identity/f'seed-{int(seed)}.pkl.gz'


def acknowledge_label_result(settings, seed):
    # The record is already durable in the corpus, and its seed is no longer
    # pending. This removes only the redundant worker-outbox copy.
    if settings.get('persist_label_results'):
        label_result_path(settings,seed).unlink(missing_ok=True)


def label_job(task):
    index, snapshot, settings, seed, side, training = task
    start, cpu = time.perf_counter(), cpu_seconds()
    outbox = label_result_path(settings, seed)
    if settings.get('persist_label_results') and outbox.exists():
        cached = pickle.loads(gzip.decompress(outbox.read_bytes()))
        if cached['row']['seed'] != seed or any(not valid_teacher_record(r) for r in cached['records']):
            raise ValueError('Invalid saved worker result')
        return cached['records'],dict(cached['row'],index=index,pid=os.getpid(),recovered_from_outbox=True)
    stage = STAGES[(seed-settings['dataset_seed']) % 3]
    path = Path(settings['output']) / f'label-worker-{os.getpid()}.json'
    write_json(path, dict(pid=os.getpid(), phase='generating', seed=seed, stage=stage, updated_epoch=time.time()))
    ablation_searches=0
    try:
        if settings.get('recorded_games') and (seed-settings['dataset_seed'])%5==0:
            state,provenance=recorded_position(seed,stage,settings)
        else:
            state, provenance = generate_position(seed, stage)
        game = IntransitiveGame()
        legal = game.getValidMoves(state, 0)
        before = state.tobytes()
        target_depth = settings['teacher']['max_depth']
        write_json(path, dict(pid=os.getpid(), phase=f'depth_{target_depth}_search', seed=seed, stage=stage,
                             teacher=settings['teacher'],
                             pieces=provenance['pieces'], updated_epoch=time.time()))
        teacher=AlphaBetaPlayer(game, SearchConfig(**settings['teacher']))
        result = teacher.analyze(state)
        if not completed_teacher(result, target_depth):
            records, rejected = [], result.stop_reason
        else:
            assert state.tobytes()==before and legal[result.action]
            key = orbit_key(state, result.action, legal)
            family=hashlib.sha256(('recorded-game:'+provenance['game_id']).encode()).hexdigest() if 'game_id' in provenance else key
            records = [dict(state=state, legal=legal, action=result.action,
                split=split_for(family), family=family, orbit=key, provenance=provenance,
                teacher_depth=result.completed_depth, teacher_reason=result.stop_reason,
                teacher_target_depth=target_depth,
                teacher_seconds=result.elapsed, teacher_work=result.work,teacher_score=result.score)]
            write_json(path,dict(pid=os.getpid(),phase='promoting_tree_roots',seed=seed,
                                stage=stage,updated_epoch=time.time()))
            records.extend(promote_tree_roots(teacher,state,result,records[0],limit=settings.get('promoted_roots',2)))
            if settings.get('ablations', True):
                write_json(path,dict(pid=os.getpid(),phase='verifying_ablations',seed=seed,
                                    stage=stage,updated_epoch=time.time()))
                variants,ablation_searches=verified_ablations(records[0],result,SearchConfig(**settings['teacher']))
                records.extend(variants)
            rejected = None
    except RuntimeError as error:
        records, rejected = [], str(error)
    row = dict(index=index, pid=os.getpid(), seed=seed, stage=stage, rejected=rejected,
               ablation_searches=ablation_searches,
               worker_seconds=time.perf_counter()-start, cpu_seconds=cpu_seconds()-cpu)
    if settings.get('persist_label_results'):
        outbox.parent.mkdir(parents=True,exist_ok=True)
        save_shard(outbox,dict(records=records,row=row))
    write_json(path, dict(row, phase='label_completed', updated_epoch=time.time()))
    return records, row


def policy_metrics(net, records):
    if not records:
        return None
    net.switch_target('training')
    net.nnet.eval()
    correct, losses = 0, []
    by_stage = {name: [0,0] for name in STAGES}
    with torch.no_grad():
        for offset in range(0, len(records), 32):
            rows = records[offset:offset+32]
            boards = torch.tensor(np.array([r['state'] for r in rows]), dtype=torch.float32)
            legal = torch.tensor(np.array([r['legal'] for r in rows]), dtype=torch.bool)
            actions = torch.tensor([r['action'] for r in rows], dtype=torch.long)
            log_pi, _ = net.nnet(boards, legal)
            hit = log_pi.argmax(1)==actions
            correct += int(hit.sum())
            losses.extend((-log_pi[torch.arange(len(rows)), actions]).tolist())
            for row, good in zip(rows, hit.tolist()):
                counts = by_stage[row['provenance']['stage']]
                counts[0] += int(good)
                counts[1] += 1
    return dict(positions=len(records), top1_agreement=correct/len(records),
                cross_entropy=float(np.mean(losses)), by_stage=by_stage)


def save_model(net, output, name, metadata):
    pending = '.'+name+'.pending'
    net.save_checkpoint(str(output), pending, dict(supervised_minimax=metadata))
    (output/pending).replace(output/name)


def read_corpus(directory):
    """Load immutable, checksummed shards for a stopped run or supplement."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'dataset-manifest.json').read_text())
    records = []
    for shard in manifest['shards']:
        path = (directory/shard['file']).resolve()
        if not path.is_relative_to(directory/'dataset'):
            raise ValueError('Shard outside dataset directory')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != shard['sha256']:
            raise ValueError('Dataset checksum mismatch')
        rows = pickle.loads(gzip.decompress(raw))
        if len(rows) != shard['positions']:
            raise ValueError('Shard record count mismatch')
        records.extend(rows)
    if len(records) != manifest['unique_positions'] or len({r['orbit'] for r in records}) != len(records):
        raise ValueError('Dataset unique-position count mismatch')
    return records, manifest


def select_supplement(existing, incoming):
    """Preserve held-out families; reject a whole overlapping incoming family."""
    known = {r['orbit'] for r in existing}
    families = {}
    for record in existing:
        family, split = record['family'], record['split']
        if families.setdefault(family, split) != split:
            raise ValueError('Existing family spans held-out splits')
    groups = {}
    for record in incoming:
        if record['split'] not in ('train', 'validation', 'test'):
            raise ValueError('Unknown dataset split')
        if not valid_teacher_record(record):
            raise ValueError('Supplement contains incomplete teacher labels')
        if not record['legal'][record['action']]:
            raise ValueError('Supplement contains illegal target')
        if orbit_key(record['state'], record['action'], record['legal']) != record['orbit']:
            raise ValueError('Supplement orbit does not match state')
        groups.setdefault(record['family'], []).append(record)
    accepted, skipped = [], []
    for family, rows in groups.items():
        splits = {r['split'] for r in rows}
        if len(splits) != 1:
            raise ValueError('Supplement family spans held-out splits')
        split = next(iter(splits))
        if any(r['orbit'] in known for r in rows) or (family in families and families[family] != split):
            skipped.append(family)
            continue
        accepted.extend(rows)
        known.update(r['orbit'] for r in rows)
        families[family] = split
    return accepted, dict(input_positions=len(incoming), accepted_positions=len(accepted),
        by_split=dict(Counter(r['split'] for r in accepted)), skipped_families=skipped)


def read_generator_positions(catalog, existing, limit=1024):
    """Read new extra-generator families without mutating its live catalog.

    No row-ID cursor: primary imports can replace/delete SQLite rows and reuse
    IDs. Orbit membership is authoritative, and echoed primary rows are ignored.
    """
    import sqlite3
    from contextlib import closing
    known = {r['orbit']: r['split'] for r in existing}
    families = {r['family']: r['split'] for r in existing}
    incoming = []
    with closing(sqlite3.connect(f'file:{Path(catalog).resolve()}?mode=ro', uri=True)) as db:
        db.execute('BEGIN')
        groups = db.execute('SELECT family FROM positions WHERE origin="extra" '
                            'GROUP BY family ORDER BY min(id)').fetchall()
        for (family,) in groups:
            rows = db.execute('SELECT orbit,split,record FROM positions '
                              'WHERE origin="extra" AND family=? ORDER BY id', (family,)).fetchall()
            if any((orbit in known and known[orbit] != split) or
                   (family in families and families[family] != split) for orbit, split, _ in rows):
                continue
            fresh = []
            for orbit, split, blob in rows:
                if orbit not in known:
                    record = pickle.loads(zlib.decompress(blob))
                    if (record['orbit'], record['family'], record['split']) != (orbit, family, split):
                        raise ValueError('Catalog index disagrees with stored record')
                    fresh.append(record)
            if not fresh:
                continue
            if incoming and len(incoming)+len(fresh) > limit:
                break
            incoming.extend(fresh)
            if len(incoming) >= limit:
                break
    accepted, report = select_supplement(existing, incoming)
    accepted = [dict(r, ingested_from='generator_catalog') for r in accepted]
    report.update(catalog=str(Path(catalog).resolve()), checked_epoch=time.time())
    return accepted, report


def run(output, seed_checkpoint, deadline, target, workers=4, *, resume=False, supplement=None,
        stream_catalog=None, teacher_config=None):
    output = Path(output).resolve()
    if (output/'status.json').exists() and not resume:
        raise ValueError('Output already started; refusing to overwrite it')
    import fcntl
    lock = (output/'trainer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = json.loads((output/'status.json').read_text()) if resume else {}
    if resume and previous.get('state') == 'running' and previous.get('pid') != os.getpid():
        try:
            os.kill(int(previous['pid']), 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError('Previous trainer still alive; stop it before resuming')
    teacher_config = dict(TEACHER if teacher_config is None else teacher_config)
    SearchConfig(**teacher_config)
    if teacher_config['max_depth'] < 5:
        raise ValueError('Teacher depth must be at least five')
    settings = dict(SETTINGS, dataset_seed=2026091500, output=str(output), teacher=teacher_config,
        recorded_games=str(output/'rps2-games.ndjson'),recorded_offsets=str(output/'recorded-offsets.json'),
        promoted_roots=2,persist_label_results=True)
    net = make_net(settings)
    source = net.load_checkpoint(str(output if resume else seed_checkpoint.parent),
                                 'latest.pt' if resume else seed_checkpoint.name)
    # New policy-only objective: preserve weights, not obsolete RL AdamW moments.
    if not resume:
        net.optimizer = net._pending_optimizer_state = None
        net.optimizer_updates = 0
    net.args['no_compression'] = True
    for parameter in net.nnet.value_head.parameters():
        parameter.requires_grad_(False)
    best = make_net(settings)
    if resume:
        best.load_checkpoint(str(output), 'best.pt')
    else:
        best.load_network(source)
    counts, split_counts, rejected = Counter(), Counter(), Counter()
    records = {'train': [], 'validation': [], 'test': []}
    train_examples, seen, shards = SymmetryExamples(records['train']), set(), []
    quotas={stage:target//3+int(i<target%3) for i,stage in enumerate(STAGES)}
    ablations=0
    sources=Counter()
    teacher_targets=Counter()
    phase, batch, attempts = 'initialization', 0, 0
    started = previous.get('started_epoch', time.time())
    pending_training = 0
    pending_seeds = []
    label_stream = None
    restored = []
    if resume:
        restored, manifest = read_corpus(output)
        shards = manifest['shards']
        batch = max((int(Path(s['file']).stem.split('-')[-1].split('.')[0]) for s in shards), default=0)
        attempts = manifest['next_seed'] - settings['dataset_seed']
        pending_seeds = list(manifest.get('pending_seeds',[]))
        rejected.update(previous.get('rejected', {}))
        trained_shards = source.get('supervised_minimax', {}).get('dataset_shards', 0)
        if trained_shards > len(shards):
            raise ValueError('Checkpoint references unavailable dataset shards')
        trained_positions = sum(s['positions'] for s in shards[:trained_shards])
        pending_training = 12 * sum(r['split'] == 'train' for r in restored[trained_positions:])
        for record in restored:
            seen.add(record['orbit'])
            records[record['split']].append(record)
            counts[record['provenance']['stage']] += 1
            split_counts[record['split']] += 1
            ablations += int('ablation' in record)
            sources['ablation' if 'ablation' in record else record['provenance'].get('source','synthetic_root')] += 1
            teacher_targets[str(record.get('teacher_target_depth',5))] += 1
    supplemental, supplement_report = [], None
    if supplement is not None:
        incoming, _ = read_corpus(supplement)
        supplemental, supplement_report = select_supplement(restored, incoming)
        supplement_report['directory'] = str(Path(supplement).resolve())
        write_json(output/'supplement-import.json', supplement_report)
    streamed_positions = sum(r.get('ingested_from') == 'generator_catalog' for r in restored)
    last_stream_poll, stream_report = 0., None
    pids = []
    def status(state='running', **extra):
        write_json(output/'status.json', dict(state=state, phase=phase, pid=os.getpid(),
            worker_pids=pids, batch=batch, attempted_positions=attempts, unique_positions=len(seen),
            target_positions=target, by_stage=dict(counts), by_split=dict(split_counts),
            augmented_examples=12*len(seen), training_examples=len(train_examples),
            verified_ablations=ablations,
            sources=dict(sources),
            generation_teacher=teacher_config, teacher_targets=dict(teacher_targets),
            pending_seeds=pending_seeds, save_mode='per_worker_completion',
            supplement=supplement_report,
            stream_catalog=str(stream_catalog) if stream_catalog else None,
            streamed_positions=streamed_positions, last_stream_import=stream_report,
            optimizer_updates=net.optimizer_updates, rejected=dict(rejected),
            started_epoch=started, deadline_epoch=deadline, updated_epoch=time.time(), **extra))
    def persist_manifest():
        write_json(output/'dataset-manifest.json', dict(schema='synthetic-minimax-policy-v1',
            shards=shards,unique_positions=len(seen),by_stage=dict(counts),by_split=dict(split_counts),
            symmetry_variants=NUM_SYMMETRIES,augmented_examples=12*len(seen),verified_ablations=ablations,
            sources=dict(sources),streamed_positions=streamed_positions,teacher=teacher_config,
            teacher_targets=dict(teacher_targets),teacher_label_scope='Per-record target depth; historical untagged records are depth 5',
            value_targets=False,next_seed=settings['dataset_seed']+attempts,pending_seeds=pending_seeds,
            save_mode='per_worker_completion'))
    def stop(*_):
        raise KeyboardInterrupt
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
        signal.signal(sig, stop)
    if deadline<=time.time():
        raise ValueError('Deadline has already passed')
    signal.setitimer(signal.ITIMER_REAL, deadline-time.time())
    write_json(output/'settings.json', dict(settings, target_positions=target, workers=workers,
        stage_quotas=quotas, symmetries=NUM_SYMMETRIES,
        splits='80/10/10 by parent symmetry-orbit hash; all ablations inherit the parent split',
        ablations='paired additions/removals or relocations; re-search requires unchanged static score and full requested-depth action/score',
        augmentation_storage='all twelve variants indexed lazily; no 1.2-million-example RAM expansion',
        tree_labels='up to two principal-variation descendants extended to the full requested depth using the same transposition table',
        recorded_sampling='20% of root requests use validated game prefixes; fallback to synthetic if no stage match',
        supervision='one-hot minimax policy only; value and Q masked; value-head weights frozen',
        sampling='independent randomized legal trajectories; capture-biased, avoids immediate corner wins',
        source_checkpoint=str(seed_checkpoint), source_optimizer_updates=source.get('optimizer_updates'),
        optimizer_reset_reason=None if resume else 'new supervised objective; no RL replay or AdamW moments reused',
        resumed=resume, supplement=supplement_report,
        stream_catalog=str(stream_catalog) if stream_catalog else None,
        stream_poll_seconds=60, stream_batch_positions=1024,
        deadline_epoch=deadline))
    if not resume:
        save_model(net, output, 'seed.pt', dict(phase='before_supervised_training'))
        save_model(net, output, 'best.pt', dict(phase='before_supervised_training'))
    status()
    end_state = 'finished'
    try:
        with GameProcesses(workers, snapshot=output/'seed.pt', settings=settings, _task=label_job) as pool:
            pids = pool.pids
            while (len(seen)<target or supplemental or pending_training) and time.time()<deadline:
                batch += 1
                phase = 'generating_and_labelling'
                status()
                if stream_catalog and not supplemental and time.time()-last_stream_poll >= 60:
                    phase = 'reading_generator_catalog'
                    status()
                    existing = [r for split in records.values() for r in split]
                    supplemental, stream_report = read_generator_positions(stream_catalog, existing)
                    last_stream_poll = time.time()
                if supplemental or pending_training:
                    phase = 'importing_supplement'
                    status()
                    labelled = [([r], dict(rejected=None)) for r in supplemental]
                    supplemental = []
                    timing = dict(seconds=0.)
                else:
                    if label_stream is None:
                        if not pending_seeds:
                            n = 4*workers
                            pending_seeds = list(range(settings['dataset_seed']+attempts, settings['dataset_seed']+attempts+n))
                            attempts += n
                            persist_manifest()  # reserve seeds durably before dispatch
                        label_stream = pool.iter_results(output/'seed.pt',pending_seeds,True,deadline=deadline)
                    result,timing = next(label_stream)
                    labelled = [result]
                accepted, new_training = [], pending_training
                pending_training = 0
                completed_seeds = []
                for data, row in labelled:
                    if row.get('seed') in pending_seeds:
                        pending_seeds.remove(row['seed'])
                        completed_seeds.append(row['seed'])
                    if row['rejected']:
                        rejected[row['rejected']] += 1
                    if data and data[0]['orbit'] in seen:
                        rejected['duplicate_parent_orbit']+=1
                        continue
                    for record in data:
                        stage = record['provenance']['stage']
                        if record['orbit'] in seen:
                            rejected['duplicate_orbit'] += 1
                            continue
                        if counts[stage]>=quotas[stage]:
                            continue
                        seen.add(record['orbit'])
                        counts[stage] += 1
                        split = record['split']
                        split_counts[split] += 1
                        records[split].append(record)
                        accepted.append(record)
                        ablations+=int('ablation' in record)
                        sources['ablation' if 'ablation' in record else record['provenance'].get('source','synthetic_root')]+=1
                        teacher_targets[str(record.get('teacher_target_depth',5))] += 1
                        streamed_positions += int(record.get('ingested_from') == 'generator_catalog')
                        if split=='train':
                            new_training += NUM_SYMMETRIES
                if accepted:
                    shard = output/'dataset'/f'shard-{batch:05d}.pkl.gz'
                    save_shard(shard, accepted)
                    shards.append(dict(file=str(shard.relative_to(output)), positions=len(accepted),
                        sha256=hashlib.sha256(shard.read_bytes()).hexdigest()))
                persist_manifest()
                for seed in completed_seeds:
                    acknowledge_label_result(settings,seed)
                if label_stream is not None and not pending_seeds:
                    assert next(label_stream,None) is None
                    label_stream = None
                if len(train_examples)>=64 and new_training and records['validation']:
                    phase = 'supervised_optimization'
                    status()
                    before = policy_metrics(net, records['validation'])
                    incumbent = policy_metrics(best, records['validation'])
                    seed_all(settings['dataset_seed']+batch)
                    net.args['batches_per_epoch'] = max(1, min(100, 3*new_training//64))
                    train_examples.sampled_sources.clear()
                    train_examples.sampled_origins.clear()
                    net.train(train_examples)
                    assert all(torch.isfinite(p).all() for p in net.nnet.parameters())
                    after = policy_metrics(net, records['validation'])
                    promoted = after['cross_entropy']<incumbent['cross_entropy']
                    metadata = dict(batch=batch, positions=len(seen), dataset_shards=len(shards),
                        policy_only=True, value_head_frozen=True, validation=after,
                        original_checkpoint=str(seed_checkpoint),
                        sampled_training_sources=dict(train_examples.sampled_sources))
                    metadata['sampled_training_origins'] = dict(train_examples.sampled_origins)
                    save_model(net, output, 'latest.pt', metadata)
                    if promoted:
                        save_model(net, output, 'best.pt', metadata)
                        best.load_checkpoint(str(output), 'best.pt')
                    row = dict(batch=batch, unique_positions=len(seen), optimizer_updates=net.optimizer_updates,
                        validation_before=before, validation_after=after, incumbent_validation=incumbent,
                        promoted=promoted, label_seconds=timing['seconds'],
                        sampled_training_sources=dict(train_examples.sampled_sources))
                    row['sampled_training_origins'] = dict(train_examples.sampled_origins)
                    with (output/'training.jsonl').open('a') as stream:
                        stream.write(json.dumps(row)+'\n')
                    print(json.dumps(row), flush=True)
                phase = 'batch_complete'
                status()
    except (KeyboardInterrupt, TimeoutError):
        end_state = 'finished' if time.time()>=deadline else 'stopped'
    except BaseException:
        status('failed', error=traceback.format_exc())
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    # The test split is not consulted during checkpoint selection.
    phase = 'final_heldout_test'
    status()
    write_json(output/'test-report.json', dict(best_policy=policy_metrics(best, records['test']),
        positions=len(seen), target_reached=len(seen)>=target, reason=end_state))
    phase = 'complete'
    status(end_state)
    lock.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--positions', type=int, default=30000)
    parser.add_argument('--workers', type=int, choices=(1,2,4), default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--supplement', type=Path)
    parser.add_argument('--stream-catalog', type=Path)
    parser.add_argument('--teacher-depth', type=int, default=5)
    parser.add_argument('--teacher-seconds', type=float)
    args = parser.parse_args()
    if args.positions<3:
        parser.error('--positions must be at least three')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'dataset').mkdir(exist_ok=True)
    run(args.output, args.checkpoint.resolve(), args.deadline, args.positions, args.workers,
        resume=args.resume, supplement=args.supplement, stream_catalog=args.stream_catalog,
        teacher_config=dict(TEACHER,max_depth=args.teacher_depth,
            time_limit=args.teacher_seconds if args.teacher_seconds is not None else (120. if args.teacher_depth<=5 else 1800.)))
