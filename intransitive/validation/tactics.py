"""Per-candidate tactical safety, certified by the independent rules.

Every fixture is certified before any candidate searches it, using the
immutable reference rules rather than the evaluator under test, so a fixture
that has become ambiguous fails loudly instead of quietly excusing a candidate.
Each result carries what its certificate actually proves: a forced win is a
proof, a bounded material or survival obligation is not, and a position where
no legal move avoids the loss is scored as already lost rather than as a
defensive mistake.
"""
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np

from ..heuristics.config import SearchConfig
from ..heuristics.search import AlphaBetaPlayer
from ..record import parse_record_move
from ..tests import test_game_blunders as blunders
from ..tests.reference_rules import Position, map_action, position
from ..tests.tactical_oracle import certify, load_case, move_name

# The committed suites' settings: high enough that machine speed cannot turn a
# move-quality test into a timeout, and identical for every candidate.
CERTIFIED_LIMITS = dict(max_depth=3, node_limit=2_000_000, time_limit=30.)
# What the shipped bot actually gets. An evaluation that cannot finish its own
# default depth inside its own default work budget is a practical fact about
# the candidate, not about this suite, so both profiles are measured.
SHIPPED_LIMITS = {name: getattr(SearchConfig(), name)
                  for name in ('max_depth', 'node_limit', 'time_limit')}
PROVEN_CATEGORIES = ('immediate_goal', 'clear_run')
PROVEN_OBJECTIVES = ('immediate_goal', 'win_goal_run')


@lru_cache(maxsize=1)
def puzzle_cases():
    """The 100 certified puzzles, each with its independently unique answer."""
    rows = []
    fixtures = json.loads((Path(__file__).parents[1] / 'heuristics' / 'tactics.json').read_text())
    for case in fixtures['positions']:
        certify(case)
        board, expected = load_case(case)
        rows.append(dict(id=case['id'], family='puzzle', category=case['category'],
                         certificate='proof' if case['category'] in PROVEN_CATEGORIES
                                     else 'bounded_obligation',
                         preventable=True, expected=[expected], board=board,
                         reason=case['reason']))
    return tuple(rows)


@lru_cache(maxsize=1)
def game_cases():
    """The 12 harder regressions from the analysed game, with the same treatment.

    An original position accepts any move meeting its tactical objective; the
    simplified layouts are certified unique. A position where the objective is
    unreachable from every legal move is recorded as already lost: turning a
    forced loss into a failed-defence test is exactly the mistake #52 warns of.
    """
    rows, states = [], replay_states()
    for case in blunders.DATA['original_positions']:
        board = states[case['ply']]
        if blunders.state_hash(board.storage()) != case['state_sha256']:
            raise ValueError(f'{case["id"]}: replayed state does not match the fixture hash')
        controls = [parse_record_move(move) for move in case['good_moves']]
        if not all(blunders.meets_objective(board, move, case) for move in controls):
            raise ValueError(f'{case["id"]}: positive control no longer meets the objective')
        if blunders.meets_objective(board, parse_record_move(case['bad_move']), case):
            raise ValueError(f'{case["id"]}: negative control now meets the objective')
        rows.append(dict(id=case['id'], family='game', category=case['objective'],
                         certificate='proof' if case['objective'] in PROVEN_OBJECTIVES
                                     else 'bounded_obligation',
                         preventable=True, expected=None, case=case, board=board,
                         reason=case['reason']))
    for case in blunders.DATA['simplified_positions']:
        for symmetry, suffix in ((0, 'original_colours'), (6, 'exchanged_colours')):
            board = Position.fixture(position(case['pieces']), player=case['player']).transform(symmetry)
            expected = map_action(parse_record_move(case['expected']), symmetry)
            good = {a for a in board.legal() if blunders.meets_objective(board, a, case)}
            if good != {expected}:
                raise ValueError(f'{case["id"]}: fixture no longer has a unique answer')
            rows.append(dict(id=f'{case["id"]}_{suffix}', family='game', category=case['objective'],
                             certificate='proof' if case['objective'] in PROVEN_OBJECTIVES
                                         else 'bounded_obligation',
                             preventable=True, expected=sorted(good), case=case, board=board,
                             reason=case['reason']))
    return tuple(rows)


@lru_cache(maxsize=1)
def replay_states():
    """Replay the recorded game through the independent rules, as the suite does."""
    record = blunders.load_record(blunders.DATA['record_pgn'])
    board = Position.initial()
    states = [board]
    np.testing.assert_array_equal(board.storage(), record.states[0])
    for action, stored in zip(record.actions, record.states[1:]):
        board, _ = board.move(action)
        np.testing.assert_array_equal(board.storage(), stored)
        states.append(board)
    return states


def judge(case, action):
    """Did this move satisfy the fixture's independently certified obligation?"""
    if case['expected'] is not None:
        return action in case['expected']
    return blunders.meets_objective(case['board'], action, case['case'])


def run_case(case, config):
    """One fresh engine per fixture: no table hint may cross a position."""
    state = case['board'].storage()
    before = state.copy()
    player = AlphaBetaPlayer(config=config)
    begin = time.perf_counter()
    result = player.analyze(state)
    latency = time.perf_counter() - begin
    mutated = not np.array_equal(state, before)
    return dict(id=case['id'], family=case['family'], category=case['category'],
                certificate=case['certificate'], preventable=case['preventable'],
                expected=[move_name(a) for a in case['expected']] if case['expected'] else None,
                chosen=move_name(result.action), passed=bool(judge(case, result.action)),
                stopped=bool(result.stopped), completed_depth=int(result.completed_depth),
                selected_depth=int(result.selected_depth), work=int(result.work),
                stop_reason=result.stop_reason, score=float(result.score),
                latency_seconds=latency, mutated_input=mutated,
                pv=[move_name(a) for a in result.pv])


def evaluate(config, *, limits=None, cases=None):
    """Score one candidate configuration over every fixture at fixed search limits.

    The evaluation fields come from the candidate; depth, work and time come
    from the profile, identically for every candidate, so a difference is a
    difference in evaluation and not in resource.
    """
    limits = dict(CERTIFIED_LIMITS if limits is None else limits)
    config = replace(SearchConfig(**config) if isinstance(config, dict) else config, **limits)
    cases = tuple(cases) if cases is not None else puzzle_cases() + game_cases()
    rows = [run_case(case, config) for case in cases]
    for row in rows:
        # The committed suites demand a completed search as well as the right
        # move. Both readings are kept: 'passed' judges the evaluation, and
        # 'strict' additionally judges whether this candidate could afford the
        # search inside the profile's budget.
        row['strict_passed'] = row['passed'] and not row['stopped']
    failures = [row for row in rows if not row['passed']]
    strict = [row for row in rows if not row['strict_passed']]
    return dict(limits=limits, cases=len(rows), passed=len(rows) - len(failures),
                failed=len(failures), strict_passed=len(rows) - len(strict),
                strict_failed=len(strict),
                strict_failed_ids=sorted(row['id'] for row in strict),
                failed_ids=sorted(row['id'] for row in failures),
                proven_failures=sorted(row['id'] for row in failures
                                       if row['certificate'] == 'proof'),
                obligation_failures=sorted(row['id'] for row in failures
                                           if row['certificate'] != 'proof'),
                already_lost=sorted(row['id'] for row in rows if not row['preventable']),
                budget_stopped=sorted(row['id'] for row in rows if row['stopped']),
                mutated_inputs=sorted(row['id'] for row in rows if row['mutated_input']),
                by_category={name: dict(
                    cases=sum(r['category'] == name for r in rows),
                    failed=sum(r['category'] == name and not r['passed'] for r in rows))
                    for name in sorted({r['category'] for r in rows})},
                completed_depths={str(d): sum(r['completed_depth'] == d for r in rows)
                                  for d in sorted({r['completed_depth'] for r in rows})},
                total_work=sum(r['work'] for r in rows),
                total_seconds=sum(r['latency_seconds'] for r in rows), results=rows,
                interpretation='A proof failure is a missed forced win or a preventable forced '
                               'loss. An obligation failure is a bounded material or survival '
                               'mistake, not a game-theoretic verdict. Fixtures where no legal '
                               'move meets the objective are listed as already lost and are '
                               'never counted against a candidate.')


def compare(candidate, baseline):
    """Regressions are fixtures the baseline solves and the candidate does not."""
    new = sorted(set(candidate['failed_ids']) - set(baseline['failed_ids']))
    fixed = sorted(set(baseline['failed_ids']) - set(candidate['failed_ids']))
    strict_new = sorted(set(candidate['strict_failed_ids']) - set(baseline['strict_failed_ids']))
    return dict(new_failures=new, fixed=fixed, new_strict_failures=strict_new,
                strict_failed=candidate['strict_failed'],
                baseline_strict_failed=baseline['strict_failed'],
                new_proven_failures=sorted(set(candidate['proven_failures'])
                                           - set(baseline['proven_failures'])),
                proven_failures=list(candidate['proven_failures']),
                budget_stopped=len(candidate['budget_stopped']),
                passed=candidate['passed'], failed=candidate['failed'],
                baseline_passed=baseline['passed'], baseline_failed=baseline['failed'])
