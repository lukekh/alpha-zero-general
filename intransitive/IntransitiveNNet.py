"""Version 2 policy/value baseline; feature semantics live in this module."""

import torch
from torch import nn
from torch.nn import functional as F

from .IntransitiveConstants import (
    ACTION_SIZE, BOARD_SIZE, CURRENT_PLANE, HISTORY_CAPACITY, HISTORY_START,
    MAX_TOTAL_PLY, METADATA_PLANE, META_A1_DEFENDER, META_HISTORY_LENGTH,
    META_HISTORY_PLAYERS, META_NEXT_PLAYER, META_NO_CAPTURE, META_TOTAL_PLY,
    NO_CAPTURE_LIMIT, NUMBER_PLAYERS, STATE_SHAPE, STATE_VERSION, TOTAL_PLY_DIGITS,
)

NETWORK_VERSION = 2
PIECE_CODES = (1, 2, 3, -1, -2, -3)
FEATURE_CONFIG = {
    'state_version': STATE_VERSION,
    'network_version': NETWORK_VERSION,
    'piece_order': ('own rock', 'own scissors', 'own paper',
                    'opponent rock', 'opponent scissors', 'opponent paper'),
    'current_order': ('six piece planes', 'own defended corner',
                      'opponent defended corner', 'noncapture / 80',
                      'current occurrences / 3', 'log1p(total ply) / log1p(max ply)'),
    'history_order': 'oldest first, including current position; 81 slots',
    'history_slot_order': ('six piece planes', 'valid', 'side to move (0 own, 1 opponent)'),
    'history_channels': 4,
    'trunk_channels': 64,
    'residual_blocks': 4,
    'action_order': '8 * (9*y + x) + direction; N, NE, E, SE, S, SW, W, NW',
    'value_order': ('canonical player 0', 'canonical player 1'),
    'total_ply_max': MAX_TOTAL_PLY,
}


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.layers(x))


class IntransitiveNNet(nn.Module):
    def __init__(self, game, args):
        super().__init__()
        if (tuple(game.getBoardSize()) != STATE_SHAPE
                or game.getActionSize() != ACTION_SIZE
                or game.num_players != NUMBER_PLAYERS):
            raise ValueError('Intransitive network requires state v2, 648 actions and two players')
        self.version = args['nn_version']
        self.board_size = STATE_SHAPE
        self.action_size = ACTION_SIZE
        self.feature_config = dict(FEATURE_CONFIG)
        # The wrapper reconstructs a validated network from checkpoint metadata.
        if self.version == -1:
            return
        if self.version != NETWORK_VERSION:
            raise ValueError(f'Unsupported Intransitive network version {self.version}')

        self.register_buffer('piece_codes', torch.tensor(PIECE_CODES).reshape(1, 1, 6, 1, 1))
        self.register_buffer('history_slots', torch.arange(HISTORY_CAPACITY).reshape(1, -1))
        corners = torch.zeros(1, 2, BOARD_SIZE, BOARD_SIZE)
        corners[0, 0, 0, 0], corners[0, 1, -1, -1] = 1, 1
        self.register_buffer('corners', corners)
        self.register_buffer('ply_weights', torch.tensor(
            [128.0**i for i in range(TOTAL_PLY_DIGITS)]))
        self.register_buffer('ply_scale', torch.log1p(torch.tensor(float(MAX_TOTAL_PLY))))
        self.history_encoder = nn.Sequential(nn.Conv2d(8, 4, 3, padding=1), nn.ReLU())
        self.projection = nn.Sequential(
            nn.Conv2d(11 + HISTORY_CAPACITY * 4, 64, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.trunk = nn.Sequential(*(ResidualBlock(64) for _ in range(4)))
        self.policy_head = nn.Conv2d(64, 8, 1)
        self.value_head = nn.Sequential(
            nn.Conv2d(64, 4, 1), nn.ReLU(), nn.Flatten(1),
            nn.Linear(4 * BOARD_SIZE * BOARD_SIZE, 64), nn.ReLU(), nn.Linear(64, 2),
        )

    def extract_features(self, state):
        """Decode raw (B,9,9,84) state into current and masked history planes.

        Inputs must obey state v2 and already use the canonical player frame.
        Storage validation belongs to the game boundary, outside the export graph.
        Reserved bytes and the version tag are never spatial gameplay features.
        """
        if not torch.jit.is_tracing() and (state.ndim != 4 or tuple(state.shape[1:]) != STATE_SHAPE):
            raise ValueError('Intransitive network requires state v2 batches shaped (batch, 9, 9, 84); '
                             'migrate version-1 replay data before training')
        meta = state[:, :, :, METADATA_PLANE:].reshape(-1, 2 * BOARD_SIZE * BOARD_SIZE)
        pieces = state[:, :, :, CURRENT_PLANE]
        history = state[:, :, :, HISTORY_START:HISTORY_START + HISTORY_CAPACITY].permute(0, 3, 1, 2)
        valid = self.history_slots < meta[:, META_HISTORY_LENGTH:META_HISTORY_LENGTH + 1]
        mask = valid[:, :, None, None, None].to(state.dtype)
        turns = meta[:, META_HISTORY_PLAYERS:META_HISTORY_PLAYERS + HISTORY_CAPACITY]
        categorical = (history[:, :, None] == self.piece_codes).to(state.dtype)
        history_features = torch.cat((
            categorical,
            torch.ones_like(history[:, :, None]),
            turns[:, :, None, None, None].expand(-1, -1, 1, BOARD_SIZE, BOARD_SIZE),
        ), dim=2) * mask

        matches = (history == pieces[:, None]).flatten(2).all(dim=2)
        matches = matches & (turns == meta[:, META_NEXT_PLAYER:META_NEXT_PLAYER + 1]) & valid
        repetitions = matches.to(state.dtype).sum(dim=1, keepdim=True) / 3.0
        clock = meta[:, META_NO_CAPTURE:META_NO_CAPTURE + 1] / NO_CAPTURE_LIMIT
        ply = (meta[:, META_TOTAL_PLY:META_TOTAL_PLY + TOTAL_PLY_DIGITS]
               * self.ply_weights).sum(dim=1, keepdim=True)
        elapsed = torch.log1p(ply) / self.ply_scale
        scalars = torch.cat((clock, repetitions, elapsed), dim=1)
        defender = meta[:, META_A1_DEFENDER:META_A1_DEFENDER + 1, None, None]
        own_goal = (1 - defender) * self.corners[:, :1] + defender * self.corners[:, 1:]
        opponent_goal = defender * self.corners[:, :1] + (1 - defender) * self.corners[:, 1:]
        current = torch.cat((
            (pieces[:, None] == self.piece_codes[:, 0]).to(state.dtype),
            own_goal, opponent_goal,
            scalars[:, :, None, None].expand(-1, -1, BOARD_SIZE, BOARD_SIZE),
        ), dim=1)
        return current, history_features

    def encode_history(self, history_features):
        encoded = self.history_encoder(history_features.reshape(-1, 8, BOARD_SIZE, BOARD_SIZE))
        encoded = encoded.reshape(-1, HISTORY_CAPACITY, 4, BOARD_SIZE, BOARD_SIZE)
        # Mask after convolution too: bias and neighbouring valid flags must not
        # turn a padded slot into a learned feature. Concatenation retains order.
        encoded = encoded * history_features[:, :, 6:7]
        return encoded.flatten(1, 2)

    @staticmethod
    def square_major_logits(direction_logits):
        return direction_logits.permute(0, 2, 3, 1).reshape(-1, ACTION_SIZE)

    def forward(self, state, valid_actions):
        current, history = self.extract_features(state)
        x = self.projection(torch.cat((current, self.encode_history(history)), dim=1))
        x = self.trunk(x)
        logits = self.square_major_logits(self.policy_head(x))
        # Finite log probabilities keep KL(0 || illegal) and its gradients finite.
        pi = F.log_softmax(logits.masked_fill(~valid_actions, -1e8), dim=1)
        # A terminal all-false mask yields zero probability for every action.
        pi = pi.masked_fill(~valid_actions, -1e8)
        return pi, torch.tanh(self.value_head(x))
