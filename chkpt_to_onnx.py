#!/usr/bin/env python3

import torch
import time
import sys
import argparse


def load_checkpoint(filepath):
	checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
	if 'intransitive_config' in checkpoint or 'intransitive_checkpoint' in checkpoint:
		from intransitive.IntransitiveGame import IntransitiveGame
		from intransitive.NNet import NNetWrapper
		wrapper = NNetWrapper(IntransitiveGame(), {'nn_version': -1})
		wrapper.load_network(checkpoint)
		nnet = wrapper.nnet
	else:
		nnet = checkpoint['full_model']
		nnet.load_state_dict(checkpoint['state_dict'])
	nn_shape = f'{nnet.nb_vect}x{nnet.vect_dim}' if 'vect_dim' in nnet.__dict__ else f'{nnet.board_size}'
	print(f'NN version: {nnet.version}, network i/o shape: {nn_shape} -> {nnet.action_size}, total nb of nnet params: {sum(p.numel() for p in nnet.parameters())}')
	return nnet

def export_onnx(nnet, output_filepath):
	if 'vect_dim' in nnet.__dict__:
		dummy_board         = torch.randn(1, nnet.nb_vect, nnet.vect_dim, dtype=torch.float32)
	else:
		dummy_board         = torch.randn(nnet.board_size, dtype=torch.float32).unsqueeze(0)
	dummy_valid_actions = torch.BoolTensor(torch.randn(1, nnet.action_size)>0.5)
	nnet.to('cpu')
	nnet.eval()

	torch.onnx.export(
		nnet,
		(dummy_board, dummy_valid_actions),
		output_filepath,
		opset_version=16,
		dynamo=False,
		input_names = ['board', 'valid_actions'],
		output_names = ['pi', 'v'],
		dynamic_axes={
			'board'        : {0: 'batch_size'},
			'valid_actions': {0: 'batch_size'},
			'pi'           : {0: 'batch_size'},
			'v'            : {0: 'batch_size'},
		}
	)

def main():
	parser = argparse.ArgumentParser(description='converter')  
	parser.add_argument('--input' , '-i' , action='store', default=None                 , type=str  , help='Input file')
	parser.add_argument('--output', '-o' , action='store', default='exported_model.onnx', type=str  , help='Output file')
	args = parser.parse_args()

	nnet = load_checkpoint(args.input)
	export_onnx(nnet, args.output)


if __name__ == '__main__':
	main()
