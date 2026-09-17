"""Compiled move-ranking/sorting loop; never removes a legal candidate."""
from functools import lru_cache
import numpy as np
from numba import njit
from .kernels import winning_actions
from ..IntransitiveConstants import DIRECTIONS


@njit(cache=True)
def exposed(pieces, square, code, removed):
    x,y = square%9,square//9
    for dy in range(-1,2):
        for dx in range(-1,2):
            nx,ny = x+dx,y+dy
            if not (0 <= nx < 9 and 0 <= ny < 9) or ny*9+nx == removed:
                continue
            enemy = int(pieces[ny,nx])
            if enemy*code < 0 and abs(enemy)%3+1 == abs(code):
                return True
    return False


@njit(cache=True)
def ordered_actions(pieces, actions, side, goal, preferred, prior, killers, history, enhanced, capture_values=None):
    """Exact old rank when enhanced=False and capture_values=None; otherwise add safe capture/escape,
    killer and history priorities. The final action key makes ties deterministic.
    """
    wins = winning_actions(pieces,actions,side,goal)
    extra = 2 if capture_values is not None else 0
    ranks = np.zeros((len(actions),11+extra),dtype=np.float64)
    own_goal = 80-goal
    for i in range(len(actions)):
        action = int(actions[i])
        square = action//8
        dx,dy = DIRECTIONS[action%8]
        x,y = square%9+dx,square//9+dy
        destination = 9*y+x
        occupant = int(pieces[y,x])
        threat = occupant*(1 if side == 0 else -1) < 0 and max(
            abs(x-own_goal%9),abs(y-own_goal//9)) <= 1
        ranks[i,0] = wins[i]
        ranks[i,1] = action == preferred
        ranks[i,2] = prior[action]
        ranks[i,3] = threat or destination == own_goal
        ranks[i,6] = occupant != 0
        ranks[i,9+extra] = -max(abs(x-goal%9),abs(y-goal//9))
        ranks[i,10+extra] = -action
        if capture_values is not None and occupant != 0:
            mover = int(pieces[square//9,square%9])
            ranks[i,7] = capture_values[1,abs(occupant)-1]
            ranks[i,8] = -capture_values[0,abs(mover)-1]
        if enhanced:
            mover = int(pieces[square//9,square%9])
            unsafe = exposed(pieces,destination,mover,destination)
            ranks[i,4] = occupant != 0 and not unsafe
            ranks[i,5] = exposed(pieces,square,mover,-1) and not unsafe
            ranks[i,7+extra] = 2 if action == killers[0] else 1 if action == killers[1] else 0
            ranks[i,8+extra] = history[action]
    # Stable descending insertion sort over a bounded list of legal moves.
    order = np.arange(len(actions))
    for i in range(1,len(actions)):
        item = order[i]
        j = i-1
        while j >= 0:
            greater = False
            for field in range(11+extra):
                if ranks[item,field] != ranks[order[j],field]:
                    greater = ranks[item,field] > ranks[order[j],field]
                    break
            if not greater:
                break
            order[j+1] = order[j]
            j -= 1
        order[j+1] = item
    return actions[order]


def material_order_values(counts, side, variable):
    """Pre-move [attacker army, victim army] per-type values; never weighted twice."""
    from .material import BASE, variable_piece_values
    if not variable:
        return np.full((2,3), BASE, dtype=np.float64)
    own = counts[side*3:side*3+3]
    enemy = counts[(1-side)*3:(1-side)*3+3]
    return np.asarray((variable_piece_values(own, enemy), variable_piece_values(enemy, own)), dtype=np.float64)


@lru_cache(maxsize=1)
def warm_ordering():
    for board in (np.zeros((9,9),dtype=np.int8),np.zeros((9,9,84),dtype=np.int8)[:,:,0]):
        for values in (None, np.ones((2,3), dtype=np.float64)):
            ordered_actions(board,np.empty(0,dtype=np.int64),0,80,-1,
                            np.full(648,-np.inf),np.full(2,-1,dtype=np.int64),np.zeros(648,dtype=np.int64),False,values)
