"""Certify small tactics using the independent Python rules, never the evaluator.

Winning certificates prove a unique fastest win within the stated horizon.
Defence certificates prove the only move avoiding loss on the next reply.
Capture certificates prove a unique, immediately bankable material gain. They
do not claim to solve the subsequent game.
"""

from functools import lru_cache

from .reference_rules import Position, action_between, destination, position


def tactical_proof(board, side, plies, *, material_target=None, first=None):
    """Independent bounded AND/OR proof, optionally fixing the first move.

    A True result proves a win or the specified material threshold against all
    replies. False means the objective cannot be forced within this horizon;
    it says nothing about the eventual game result. Ordering and memoization
    save work but never discard a legal defence. There is no heuristic score,
    production search call, time cutoff, or node cutoff here.
    """
    @lru_cache(maxsize=32768)
    def visit(node, remaining):
        reason, rewards = node.terminal()
        if reason != 'ongoing':
            return rewards[side] == 1
        if remaining == 0:
            return (material_target is not None and
                    material(node, side) >= material_target)
        goal = 80 if node.player == node.a1_defender else 0
        actions = sorted(node.legal(), key=lambda a: (
            destination(a) != goal,
            not bool(node.pieces[destination(a)]),
            max(abs(destination(a) % 9 - goal % 9),
                abs(destination(a) // 9 - goal // 9)), a))
        outcomes = (visit(node.move(a)[0], remaining - 1) for a in actions)
        return any(outcomes) if node.player == side else all(outcomes)

    try:
        if first is not None:
            if first not in board.legal():
                return False
            return visit(board.move(first)[0], plies - 1)
        return visit(board, plies)
    finally:
        visit.cache_clear()


def load_case(case):
    board = Position.fixture(position(case['pieces']), player=case['player'])
    source, target = case['expected'].split('-')
    return board, action_between(source, target)


def move_name(action):
    def square_name(index):
        return chr(ord('A') + index % 9) + str(index // 9 + 1)
    return square_name(action // 8) + '-' + square_name(destination(action))


def forced_win(board, side, remaining):
    """AND/OR terminal-only proof; False at a horizon means unproved, not drawn."""
    reason, rewards = board.terminal()
    if reason != 'ongoing':
        return rewards[side] == 1
    if remaining == 0:
        return False
    outcomes = (forced_win(board.move(a)[0], side, remaining - 1)
                for a in sorted(board.legal()))
    return any(outcomes) if board.player == side else all(outcomes)


def material(board, side):
    return sum(1 if (p < 0) == bool(side) else -1
               for p in board.pieces if p)


def certify(case):
    """Raise on ambiguous/incorrect fixtures; return the independently best set."""
    board, expected = load_case(case)
    assert board.terminal()[0] == 'ongoing', 'Fixture is already terminal'
    actions = sorted(board.legal())
    assert len(actions) >= 2, 'A forced legal move is not a decision puzzle'
    assert expected in actions, 'Expected move is illegal'
    children = {a: board.move(a)[0] for a in actions}
    side = board.player
    category = case['category']

    if category in ('immediate_goal', 'clear_run'):
        plies = 1 if category == 'immediate_goal' else 3
        best = {a for a, child in children.items()
                if forced_win(child, side, plies - 1)}
        if category == 'immediate_goal':
            assert children[expected].terminal()[0] == 'corner'
        else:
            assert not any(forced_win(child, side, 0)
                           for child in children.values()), 'Win is immediate'
            # The advertised clear run ends at the corner, not by elimination.
            for reply in children[expected].legal():
                response = children[expected].move(reply)[0]
                assert response.terminal()[0] == 'ongoing'
                assert any(response.move(a)[0].terminal() ==
                           ('corner', (1., -1.) if side == 0 else (-1., 1.))
                           for a in response.legal()), 'Run can be intercepted'
    elif category in ('goal_defence', 'save_piece'):
        best = set()
        for action, child in children.items():
            assert child.terminal()[0] == 'ongoing', 'Use a win category instead'
            replies = (child.move(a)[0] for a in child.legal())
            losing_replies = [p for p in replies if p.terminal()[1][side] == -1]
            if not losing_replies:
                best.add(action)
            elif category == 'goal_defence':
                assert any(p.terminal()[0] == 'corner' for p in losing_replies)
            else:
                assert any(not any(p and (p < 0) == bool(side)
                                   for p in response.pieces)
                           for response in losing_replies), 'Not a piece loss'
        if category == 'save_piece':
            assert board.pieces[destination(expected)] == 0, 'Expected an escape'
            assert sum(bool(p) and (p < 0) == bool(side)
                       for p in board.pieces) == 1
    elif category == 'safe_capture':
        values = {}
        for action, child in children.items():
            assert child.terminal()[0] == 'ongoing', 'Use a win category instead'
            replies = [child.move(a)[0] for a in child.legal()]
            assert all(p.terminal()[1][side] != 1 for p in replies), \
                'Use a win category instead'
            # Poisoned captures may lose the last friendly piece. Losing the
            # game must rank below every material balance (at most 81 pieces).
            values[action] = min(-1000 if p.terminal()[1][side] == -1
                                 else material(p, side) for p in replies)
        best_value = max(values.values())
        best = {a for a, value in values.items() if value == best_value}
        assert best_value == material(board, side) + 1, 'No guaranteed free piece'
        assert board.pieces[destination(expected)] != 0, 'Expected a capture'
        assert not forced_win(board, side, 3), 'Use a win category instead'
    else:
        raise AssertionError(f'Unknown category: {category}')

    assert best == {expected}, (
        f"{case['id']}: expected {case['expected']}; certificate gives "
        f"{', '.join(map(move_name, sorted(best))) or 'no move'}")
    return best
