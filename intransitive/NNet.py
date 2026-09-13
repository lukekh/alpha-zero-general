"""Shared training, checkpoint, and ONNX wrapper for Intransitive."""

from GenericNNetWrapper import GenericNNetWrapper
from .IntransitiveNNet import IntransitiveNNet


class NNetWrapper(GenericNNetWrapper):
    def init_nnet(self, game, nn_args):
        self.nnet = IntransitiveNNet(game, nn_args)

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar',
                        additional_keys=None):
        keys = dict(additional_keys or {})
        keys['intransitive_config'] = dict(self.nnet.feature_config)
        super().save_checkpoint(folder, filename, keys)
