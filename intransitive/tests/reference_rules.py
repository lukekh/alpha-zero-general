"""Small immutable Python oracle derived from the confirmed rules.

No production imports: tuples, explicit capture pairs, coordinate arithmetic,
and board-plus-turn history implement play independently of the Numba engine.
NumPy is used only at the version-2 storage boundary. Fixtures may supply
structurally valid histories that are not reachable; generated games never do.
"""

from dataclasses import dataclass, replace

import numpy as np


STEPS = ((0, 1), (1, 1), (1, 0), (1, -1),
         (0, -1), (-1, -1), (-1, 0), (-1, 1))
CAPTURES = {(1, 2), (2, 3), (3, 1)}


def square(name):
    return 9 * (int(name[1]) - 1) + ord(name[0]) - ord('A')


def position(placements):
    pieces = [0] * 81
    for name, piece in placements.items():
        pieces[square(name)] = piece
    return tuple(pieces)


def action_between(source, destination):
    source, destination = square(source), square(destination)
    dy = destination // 9 - source // 9
    dx = destination % 9 - source % 9
    return 8 * source + STEPS.index((dx, dy))


def destination(action):
    source, direction = divmod(action, 8)
    dx, dy = STEPS[direction]
    x, y = source % 9 + dx, source // 9 + dy
    return 9 * y + x if 0 <= x < 9 and 0 <= y < 9 else None


def mobility(pieces, player):
    moves = set()
    for source, piece in enumerate(pieces):
        if piece == 0 or (piece < 0) != bool(player):
            continue
        for direction in range(8):
            action = 8 * source + direction
            target = destination(action)
            if target is None:
                continue
            defender = pieces[target]
            if defender == 0 or (piece * defender < 0 and
                                 (abs(piece), abs(defender)) in CAPTURES):
                moves.add(action)
    return frozenset(moves)


@dataclass(frozen=True)
class Position:
    pieces: tuple
    player: int
    a1_defender: int
    history: tuple  # (piece tuple, next player), since the latest capture
    clock: int = 0
    ply: int = 0

    @classmethod
    def fixture(cls, pieces, player=0, a1_defender=0, history=None, ply=None):
        history = (pieces,) if history is None else tuple(history)
        turns = tuple((p, (player - len(history) + 1 + i) % 2)
                      for i, p in enumerate(history))
        clock = len(history) - 1
        return cls(pieces, player, a1_defender, turns, clock,
                   clock if ply is None else ply)

    @classmethod
    def initial(cls):
        placements = {}
        for piece, blue, red in (
            (1, 'B4 C3 D2', 'F8 G7 H6'),
            (2, 'C5 D4 E3', 'E7 F6 G5'),
            (3, 'B5 C4 D3 E2', 'E8 F7 G6 H5'),
        ):
            placements.update((s, piece) for s in blue.split())
            placements.update((s, -piece) for s in red.split())
        return cls.fixture(position(placements))

    @classmethod
    def from_storage(cls, state):
        meta = state[:, :, 82:84].ravel()
        pieces = tuple(int(p) for p in state[:, :, 0].flat)
        history = tuple((tuple(int(p) for p in state[:, :, i + 1].flat),
                         int(meta[10 + i])) for i in range(int(meta[4])))
        return cls(pieces, int(meta[1]), int(meta[2]), history, int(meta[3]),
                   sum(int(meta[5 + i]) * 128**i for i in range(5)))

    def storage(self):
        state = np.zeros((9, 9, 84), dtype=np.int8)
        state[:, :, 0] = np.asarray(self.pieces).reshape(9, 9)
        meta = np.zeros(162, dtype=np.int8)
        meta[:5] = (2, self.player, self.a1_defender, self.clock, len(self.history))
        for i in range(5):
            meta[5 + i] = (self.ply // 128**i) % 128
        for i, (pieces, player) in enumerate(self.history):
            state[:, :, i + 1] = np.asarray(pieces).reshape(9, 9)
            meta[10 + i] = player
        state[:, :, 82:84] = meta.reshape(9, 9, 2)
        return state

    def repetitions(self):
        return self.history.count((self.pieces, self.player))

    def terminal(self):
        winners = []
        for target, defender in ((0, self.a1_defender), (80, 1 - self.a1_defender)):
            piece = self.pieces[target]
            if piece and int(piece < 0) != defender:
                winners.append(int(piece < 0))
        if len(winners) > 1:
            raise ValueError('Simultaneous corner winners')
        if winners:
            winner, reason = winners[0], 'corner'
        elif not mobility(self.pieces, self.player):
            winner, reason = 1 - self.player, 'stalemate'
        else:
            reason = ('repetition' if self.repetitions() >= 3 else
                      'no-capture limit' if self.clock >= 80 else 'ongoing')
            return reason, (0.0, 0.0) if reason == 'ongoing' else (1e-4, 1e-4)
        return reason, (1.0, -1.0) if winner == 0 else (-1.0, 1.0)

    def legal(self, player=None):
        if self.terminal()[0] != 'ongoing':
            return frozenset()
        return mobility(self.pieces, self.player if player is None else player)

    def move(self, action):
        if action not in self.legal():
            raise ValueError('Illegal reference action')
        target = destination(action)
        captured = self.pieces[target] != 0
        pieces = list(self.pieces)
        pieces[target], pieces[action // 8] = pieces[action // 8], 0
        pieces = tuple(pieces)
        player = 1 - self.player
        history = () if captured else self.history
        return Position(pieces, player, self.a1_defender,
                        history + ((pieces, player),),
                        0 if captured else self.clock + 1, self.ply + 1), captured

    def relabel(self):
        return replace(self, pieces=tuple(-p for p in self.pieces),
                       player=1 - self.player, a1_defender=1 - self.a1_defender,
                       history=tuple((tuple(-p for p in ps), 1 - p)
                                     for ps, p in self.history))

    def transform(self, symmetry):
        cycles, diagonal, exchange = symmetry % 3, symmetry // 3 % 2, symmetry // 6
        type_maps = ((0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2))

        def transform_pieces(pieces):
            out = [0] * 81
            for source, piece in enumerate(pieces):
                x, y = map_coordinate(source % 9, source // 9, diagonal, exchange)
                sign = -1 if (piece < 0) != bool(exchange) else 1
                out[9 * y + x] = sign * type_maps[cycles][abs(piece)]
            return tuple(out)

        # Derive ownership from the old corner that maps onto A1.
        old_defender = 1 - self.a1_defender if exchange else self.a1_defender
        return replace(self, pieces=transform_pieces(self.pieces),
                       player=self.player ^ exchange,
                       a1_defender=old_defender ^ exchange,
                       history=tuple((transform_pieces(ps), p ^ exchange)
                                     for ps, p in self.history))


def map_coordinate(x, y, diagonal, exchange):
    if diagonal:
        x, y = y, x
    if exchange:
        x, y = 8 - y, 8 - x
    return x, y


def map_action(action, symmetry):
    source, direction = divmod(action, 8)
    x, y = source % 9, source // 9
    dx, dy = STEPS[direction]
    diagonal, exchange = symmetry // 3 % 2, symmetry // 6
    tx, ty = map_coordinate(x, y, diagonal, exchange)
    nx, ny = map_coordinate(x + dx, y + dy, diagonal, exchange)
    return 8 * (9 * ty + tx) + STEPS.index((nx - tx, ny - ty))
