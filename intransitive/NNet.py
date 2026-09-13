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
        return dict(format_version=CHECKPOINT_FORMAT_VERSION,
                    game='intransitive', board_size=tuple(self.board_size),
                    action_size=self.action_size, num_players=self.num_players,
                    optimizer_state='recreated', scheduler_state='recreated')

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar',
                        additional_keys=None):
        keys = dict(additional_keys or {})
        if self.nnet.version != NETWORK_VERSION:
            raise ValueError('Load an Intransitive network before saving a checkpoint')
        keys['intransitive_config'] = dict(self.nnet.feature_config)
        keys['intransitive_checkpoint'] = self.checkpoint_format()
        keys['nn_args'] = dict(self.args, nn_version=self.nnet.version)
        keys['nn_version'] = self.nnet.version
        super().save_checkpoint(folder, filename, keys)

    def load_network(self, checkpoint, strict=False):
        """Validate semantics and weights before replacing the running network.

        Also accept the pre-envelope checkpoints from issue #10, which already
        stored the complete feature configuration. Unknown formats require an
        explicit migration; partial weight loading cannot establish compatibility.
        Caller training settings remain in effect (no optimizer is serialized).
        """
        if not isinstance(checkpoint, dict):
            raise ValueError('Intransitive checkpoint must be a dictionary')
        if 'intransitive_checkpoint' in checkpoint:
            actual = checkpoint['intransitive_checkpoint']
            if not isinstance(actual, dict):
                raise ValueError('Invalid Intransitive checkpoint format metadata')
            for key, expected in self.checkpoint_format().items():
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
        candidate.load_state_dict(state, strict=True)
        self.nnet = candidate
        self.requestKnowledgeTransfer = False
        self.ort_session = None
        self.current_mode = 'cpu'
