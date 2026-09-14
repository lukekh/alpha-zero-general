"""Shared training, checkpoint, and ONNX wrapper for Intransitive."""

import torch

from GenericNNetWrapper import GenericNNetWrapper
from .IntransitiveNNet import FEATURE_CONFIG, NETWORK_VERSION, IntransitiveNNet


CHECKPOINT_FORMAT_VERSION = 1


class NNetWrapper(GenericNNetWrapper):
    def init_nnet(self, game, nn_args):
        self.game = game
        self.nnet = IntransitiveNNet(game, nn_args)

    def checkpoint_format(self):
        persistent = bool(self.args.get('persist_optimizer', False))
        return dict(format_version=CHECKPOINT_FORMAT_VERSION,
                    game='intransitive', board_size=tuple(self.board_size),
                    action_size=self.action_size, num_players=self.num_players,
                    optimizer_state='persistent' if persistent else 'recreated',
                    scheduler_state='constant learning rate' if persistent else 'recreated')

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar',
                        additional_keys=None):
        keys = dict(additional_keys or {})
        if self.nnet.version != NETWORK_VERSION:
            raise ValueError('Load an Intransitive network before saving a checkpoint')
        keys['intransitive_config'] = dict(self.nnet.feature_config)
        keys['intransitive_checkpoint'] = self.checkpoint_format()
        keys['nn_args'] = dict(self.args, nn_version=self.nnet.version)
        keys['nn_version'] = self.nnet.version
        if self.args.get('persist_optimizer', False):
            state = (self.optimizer.state_dict() if self.optimizer is not None
                     else self._pending_optimizer_state)
            if state is not None:
                keys['optim_state'] = state
            keys['optimizer_updates'] = self.optimizer_updates
        super().save_checkpoint(folder, filename, keys)

    def load_network(self, checkpoint, strict=False):
        """Validate semantics and weights before replacing the running network.

        Also accept the pre-envelope checkpoints from issue #10, which already
        stored the complete feature configuration. Unknown formats require an
        explicit migration; partial weight loading cannot establish compatibility.
        Caller training settings remain in effect. Optimizer state is serialized
        only for checkpoints that explicitly opt into persistent training.
        """
        if not isinstance(checkpoint, dict):
            raise ValueError('Intransitive checkpoint must be a dictionary')
        if 'intransitive_checkpoint' in checkpoint:
            actual = checkpoint['intransitive_checkpoint']
            if not isinstance(actual, dict):
                raise ValueError('Invalid Intransitive checkpoint format metadata')
            for key, expected in self.checkpoint_format().items():
                if (self.args.get('nn_version') == -1 and key == 'optimizer_state'
                        and actual.get(key) in ('recreated', 'persistent')):
                    continue
                if (self.args.get('nn_version') == -1 and key == 'scheduler_state'
                        and actual.get(key) in ('recreated', 'constant learning rate')):
                    continue
                if actual.get(key) != expected:
                    raise ValueError(f'Incompatible Intransitive checkpoint {key}: '
                                     f'expected {expected!r}, got {actual.get(key)!r}')
        config = checkpoint.get('intransitive_config')
        if not isinstance(config, dict):
            raise ValueError('Missing Intransitive feature configuration')
        for key, expected in FEATURE_CONFIG.items():
            if config.get(key) != expected:
                raise ValueError(f'Incompatible Intransitive feature {key}: '
                                 f'expected {expected!r}, got {config.get(key)!r}')
        if checkpoint.get('nn_version', NETWORK_VERSION) != NETWORK_VERSION:
            raise ValueError('Incompatible Intransitive nn_version')
        candidate = IntransitiveNNet(self.game, dict(self.args, nn_version=NETWORK_VERSION))
        state = checkpoint.get('state_dict')
        if not isinstance(state, dict):
            raise ValueError('Missing Intransitive state_dict')
        # These buffers define feature meaning, rather than learned parameters.
        for key in ('piece_codes', 'history_slots', 'corners', 'ply_weights', 'ply_scale'):
            saved = state.get(key)
            expected = getattr(candidate, key)
            if (not isinstance(saved, torch.Tensor) or saved.dtype != expected.dtype
                    or not torch.equal(saved.cpu(), expected)):
                raise ValueError(f'Incompatible Intransitive feature buffer {key}')
        persistent = bool(self.args.get('persist_optimizer', False))
        optim_state = checkpoint.get('optim_state')
        inference_only = self.args.get('nn_version') == -1
        if optim_state is not None and not persistent and not inference_only:
            raise ValueError('Persistent optimizer checkpoint requires persist_optimizer=True')
        if persistent and checkpoint.get('optimizer_updates', 0) and optim_state is None:
            raise ValueError('Persistent optimizer checkpoint is missing optim_state')
        if persistent and optim_state is not None:
            try:
                probe = torch.optim.AdamW(candidate.parameters(), lr=self.args['learn_rate'])
                probe.load_state_dict(optim_state)
            except Exception as error:
                raise ValueError(f'Invalid persistent optimizer state: {error}') from error
        candidate.load_state_dict(state, strict=True)
        self.nnet = candidate
        self.optimizer = None
        self._pending_optimizer_state = optim_state if persistent else None
        self.optimizer_updates = int(checkpoint.get('optimizer_updates', 0)) if persistent else 0
        self.requestKnowledgeTransfer = False
        self.ort_session = None
        self.current_mode = 'cpu'
