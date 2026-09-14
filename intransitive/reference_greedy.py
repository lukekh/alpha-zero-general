"""Move evaluation from charbelkassab/flybrain-intransitive's rps2.py.

Source: https://github.com/charbelkassab/flybrain-intransitive/blob/main/rps2.py
Only the heuristic is ported; legal moves and termination use our game engine.
"""

from .IntransitiveConstants import (
    METADATA_PLANE, META_A1_DEFENDER, action_destination, decode_action,
)
from .IntransitiveLogicNumba import raw_movement_mask


FEATURES = ('goal', 'capture', 'progress', 'danger', 'base_threat')
REFERENCE_VALUE = dict(goal=1000., base_threat=-500., capture=10., danger=-8., progress=2.)


def features(state, action):
    """Return five binary features for a legal action, without mutating state.

    Coordinates stay physical, including when player labels are canonicalized.
    Threats use piece-only legality, just as in the reference: hypothetical
    repetition/no-capture cutoffs do not suppress the base-threat feature.
    """
    x, y, _ = decode_action(action)
    nx, ny = action_destination(action)
    pieces = state[:, :, 0]
    piece = int(pieces[y, x])
    player = 0 if piece > 0 else 1
    defender = int(state[:, :, METADATA_PLANE:].flat[META_A1_DEFENDER])
    target = 8 if player == defender else 0
    own_base = 8 - target
    goal = nx == target and ny == target
    after = pieces.copy()
    after[y, x] = 0
    after[ny, nx] = piece

    danger = False
    base_threat = False
    if not goal:
        neighbours = after[max(0, ny-1):min(9, ny+2), max(0, nx-1):min(9, nx+2)]
        # R=1, S=2, P=3: the previous type in the cycle captures this type.
        predator = (abs(piece) + 1) % 3 + 1
        enemy_predator = -predator if piece > 0 else predator
        danger = bool((neighbours == enemy_predator).any())
        replies = raw_movement_mask(after, 1 - player)
        base_threat = any(action_destination(int(reply)) == (own_base, own_base)
                          for reply in replies.nonzero()[0])

    return dict(
        goal=float(goal), capture=float(pieces[ny, nx] != 0),
        progress=float(max(abs(target-nx), abs(target-ny))
                       < max(abs(target-x), abs(target-y))),
        danger=float(danger), base_threat=float(base_threat),
    )


def reference_value(move_features):
    return sum(REFERENCE_VALUE[key] * move_features[key] for key in FEATURES)
