"""Experimental, blocker-blind 7x7/9x9 RPS pressure with cumulative defence.

Each opposing predator/prey pair contributes once: an attack credit to the
predator's side, equivalently an equal threat penalty to the victim's side.
Subtracting the two attack totals avoids counting that relationship twice.
Off-board padding is empty; distance is Chebyshev (square rings).
"""
from functools import lru_cache
import numpy as np
from numba import njit


DISTANCES = np.array([[max(abs(a//9-b//9), abs(a%9-b%9))
                       for b in range(81)] for a in range(81)], dtype=np.int8)
DISCOUNTS = np.array([2.**-n for n in range(81)], dtype=np.float64)
WEIGHTS = np.array([0., 2., 1., .5, .25], dtype=np.float64)
for _table in (DISTANCES, DISCOUNTS, WEIGHTS):
    _table.flags.writeable = False


@njit(cache=True)
def pressure_totals(pieces, radius=4):
    """Return positive attack totals (Blue, Red), respecting victim defence.

    Two bounded occupied-piece pair passes; no paths, child boards, tensor
    convolutions, or public-state exports. Friendly defenders are unconditional.
    """
    if radius != 3 and radius != 4:
        raise ValueError('Pressure radius must be three or four')
    squares = np.empty(81, dtype=np.int16)
    codes = np.empty(81, dtype=np.int8)
    count = 0
    for y in range(9):
        for x in range(9):
            code = pieces[y,x]
            if code:
                squares[count], codes[count] = y*9+x, code
                count += 1
    defenders = np.zeros((count,radius+1), dtype=np.int16)
    for victim in range(count):
        # The victim's prey type captures the victim's predator.
        defender_kind = abs(int(codes[victim])) % 3 + 1
        for friend in range(count):
            if codes[victim]*int(codes[friend]) <= 0 or abs(int(codes[friend])) != defender_kind:
                continue
            distance = DISTANCES[squares[victim],squares[friend]]
            if 1 <= distance <= radius:
                defenders[victim,distance] += 1
        for ring in range(2,radius+1):
            defenders[victim,ring] += defenders[victim,ring-1]
    blue, red = 0., 0.
    for first in range(count):
        for second in range(first+1,count):
            a, b = int(codes[first]), int(codes[second])
            if a*b >= 0 or abs(a) == abs(b):
                continue
            distance = DISTANCES[squares[first],squares[second]]
            if not 1 <= distance <= radius:
                continue
            attacker, victim = (first,second) if abs(a)%3+1 == abs(b) else (second,first)
            value = WEIGHTS[distance]*DISCOUNTS[defenders[victim,distance]]
            if codes[attacker] > 0:
                blue += value
            else:
                red += value
    return blue, red


@lru_cache(maxsize=1)
def warm_pressure_kernel():
    # Warm default and explicit-radius calls for both board layouts.
    for pieces in (np.zeros((9,9),dtype=np.int8),np.zeros((9,9,84),dtype=np.int8)[:,:,0]):
        pressure_totals(pieces)
        pressure_totals(pieces,3)
        pressure_totals(pieces,4)
