import os
import sys
import time
import pickle
import zlib
import tempfile

os.environ["OMP_NUM_THREADS"] = "1" # PyTorch more efficient this way

import numpy as np
from tqdm import tqdm
from time import sleep

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim
import torch.onnx
import onnxruntime as ort
import onnx
torch.set_num_threads(1) # PyTorch more efficient this way

class GenericNNetWrapper(NeuralNet):
	def __init__(self, game, nn_args):
		self.args = nn_args
		# The established training loop recreates AdamW and OneCycleLR for every
		# call.  Experiments that explicitly opt into continuous learning keep the
		# optimizer (and its moments) here and serialize it through the game wrapper.
		# A loaded persistent checkpoint defers optimizer construction until train()
		# so inference-only consumers do not allocate optimizer state.
		self.optimizer = None
		self._pending_optimizer_state = None
		self.optimizer_updates = 0
		self.device = {
			'training' : 'cpu', #'cuda' if torch.cuda.is_available() else 'cpu',
			'inference': 'onnx',
			'just_loaded': 'cpu',
		}
		self.current_mode = 'cpu'
		self.init_nnet(game, nn_args)
		self.ort_session = None

		self.board_size = game.getBoardSize()
		self.action_size = game.getActionSize()
		self.num_players = game.num_players
		self.requestKnowledgeTransfer = False

	def init_nnet(self, game, nn_args):
		pass

	def train(self, examples, validation_set=None, save_folder=None, every=0):
		"""
		examples: legacy ``(board, pi, v, legal, q)`` records or extended
			``(board, pi, v, legal, q, value_mask, q_mask)`` records.  Masks
			are per-player booleans.  They let policy-only/teacher examples omit
			unknown value or search-Q targets without fabricating labels.
		"""
		self.switch_target('training')
		persistent = bool(self.args.get('persist_optimizer', False))
		if persistent:
			if self.optimizer is None:
				self.optimizer = optim.AdamW(self.nnet.parameters(), lr=self.args['learn_rate'])
				if self._pending_optimizer_state is not None:
					self.optimizer.load_state_dict(self._pending_optimizer_state)
					self._pending_optimizer_state = None
			optimizer = self.optimizer
			# A persistent OneCycle schedule cannot be extended safely after its
			# predeclared horizon.  Continuous experiments therefore use constant
			# learning rate and preserve AdamW state across bounded phases.
			for group in optimizer.param_groups:
				group['lr'] = self.args['learn_rate']
		else:
			optimizer = optim.AdamW(self.nnet.parameters(), lr=self.args['learn_rate'])
		batch_count = self.args.get('batches_per_epoch', int(len(examples) / self.args['batch_size']))
		if isinstance(batch_count, bool) or not isinstance(batch_count, int) or batch_count < 1:
			raise ValueError('batches_per_epoch must be a positive integer')
		if len(examples) < self.args['batch_size']:
			raise ValueError('Training needs at least one full batch of examples')
		scheduler = (None if persistent else optim.lr_scheduler.OneCycleLR(
			optimizer, max_lr=self.args['learn_rate'], steps_per_epoch=batch_count,
			epochs=self.args['epochs']))

		t = tqdm(total=self.args['epochs'] * batch_count, desc='Train ep0', colour='blue', ncols=120, mininterval=0.5, disable=None)
		for epoch in range(self.args['epochs']):
			t.set_description(f'Train ep{epoch + 1}')
			self.nnet.train()
			pi_losses, v_losses = AverageMeter(), AverageMeter()
	
			for i_batch in range(batch_count):
				sample_ids = np.random.choice(len(examples), size=self.args['batch_size'], replace=False)
				picked = self.pick_examples(examples, sample_ids)
				boards, pis, vs, valid_actions, qs = picked[:5]
				value_masks = q_masks = None
				if len(picked) == 7:
					value_masks = torch.BoolTensor(np.array(picked[5]).astype(np.bool_))
					q_masks = torch.BoolTensor(np.array(picked[6]).astype(np.bool_))
				boards = torch.FloatTensor(np.array(boards).astype(np.float32))
				valid_actions = torch.BoolTensor(np.array(valid_actions).astype(np.bool_))
				target_pis = torch.FloatTensor(np.array(pis).astype(np.float32))
				target_vs = torch.FloatTensor(np.array(vs).astype(np.float32))
				target_qs = torch.FloatTensor(np.array(qs).astype(np.float32))

				# predict
				optimizer.zero_grad(set_to_none=True)
				out_pi, out_v = self.nnet(boards, valid_actions)
				l_pi = self.loss_pi(target_pis, out_pi)
				l_v = (self.loss_v(target_vs, target_qs, out_v)
						if value_masks is None else
						self.loss_v(target_vs, target_qs, out_v, value_masks, q_masks))
				total_loss = l_pi + 0.25*l_v # Weight 0.25 * value

				# record loss
				pi_losses.update(l_pi.item(), boards.size(0))
				v_losses.update(l_v.item(), boards.size(0))
				t.set_postfix(PI=pi_losses, V=v_losses, refresh=False)

				# compute gradient and do SGD step
				total_loss.backward()
				optimizer.step()
				self.optimizer_updates += 1
				if scheduler is not None:
					scheduler.step()

				t.update()

				if validation_set and ((i_batch + batch_count*epoch) % every == 0):
					print(self.evaluate(validation_set))
					self.nnet.train()
					if (i_batch > 0) and save_folder:
						self.save_checkpoint(save_folder, filename=f'intermediary_{i_batch}.pt')

		t.close()
		
	def predict(self, board, valid_actions):
		"""
		board: np array with board
		"""
		# timing

		# preparing input
		self.switch_target('inference')

		if self.current_mode == 'onnx':
			ort_outs = self.ort_session.run(None, {
				'board': np.expand_dims(board.astype(np.float32), 0),
				'valid_actions': np.expand_dims(np.array(valid_actions).astype(np.bool_), 0),
			})
			pi, v = np.exp(ort_outs[0][0]), ort_outs[1][0]
			return pi, v

		else:
			board = torch.FloatTensor(board.astype(np.float32)).unsqueeze(0)
			valid_actions = torch.BoolTensor(np.array(valid_actions).astype(np.bool_)).unsqueeze(0)
			if self.current_mode == 'cuda':
				board, valid_actions = board.contiguous().cuda(), valid_actions.contiguous().cuda()
			self.nnet.eval()
			with torch.no_grad():
				pi, v = self.nnet(board, valid_actions)
			pi, v = torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]
			return pi, v

	def predict_client(self, board, valid_actions, batch_info):
		if self.current_mode != 'onnx':
			raise Exception('Batch prediction only in ONNX mode')
		i_thread, i_result, shared_memory, locks = batch_info

		# Store inputs in shared memory
		shared_memory[i_thread] = (
			np.expand_dims(board.astype(np.float32), 0),
			np.expand_dims(np.array(valid_actions).astype(np.bool_), 0),
		)
		# Unblock next thread (= next MCTS or server), and wait for our turn
		locks[i_thread+1].release()
		locks[i_thread].acquire()

		# Retrieve results in shared memory
		ort_outs = shared_memory[i_result]
		pi, v = np.exp(ort_outs[0]), ort_outs[1]

		return pi, v

	def predict_server(self, nb_threads, shared_memory, locks):
		self.switch_target('inference')
		locks[0].release()

		while shared_memory[-1] <= 1:
			locks[-1].acquire() # Wait for all inputs

			# Finished (or zero-quota) workers have no input. Never pad with
			# another game's history or evaluate stale slots during shutdown.
			active = [i for i in range(nb_threads) if shared_memory[i] is not None]
			if active and shared_memory[-1] != 2:
				ort_outs = self.ort_session.run(None, {
					'board': np.concatenate([shared_memory[i][0] for i in active]),
					'valid_actions': np.concatenate([shared_memory[i][1] for i in active]),
				})
				for row, i in enumerate(active):
					shared_memory[i+nb_threads] = (ort_outs[0][row], ort_outs[1][row])

			locks[0].release() # Unblock 1st thread

	def evaluate(self, validation_set):
		# print()
		# print(f'LR = {scheduler.get_last_lr()[0]:.1e}', end=' ')

		# Evaluation
		self.nnet.eval()
		with torch.no_grad():
			picked_examples = [pickle.loads(zlib.decompress(e)) if isinstance(e, bytes) else e
							   for e in validation_set]
			picked = self._split_fields(picked_examples)
			boards, pis, vs, valid_actions, qs = picked[:5]
			value_masks = q_masks = None
			if len(picked) == 7:
				value_masks = torch.BoolTensor(np.array(picked[5]).astype(np.bool_))
				q_masks = torch.BoolTensor(np.array(picked[6]).astype(np.bool_))
			boards = torch.FloatTensor(np.array(boards).astype(np.float32))
			valid_actions = torch.BoolTensor(np.array(valid_actions).astype(np.bool_))
			target_pis = torch.FloatTensor(np.array(pis).astype(np.float32))
			target_vs = torch.FloatTensor(np.array(vs).astype(np.float32))
			target_qs = torch.FloatTensor(np.array(qs).astype(np.float32))

			# compute output
			out_pi, out_v = self.nnet(boards, valid_actions)
			value_loss = (self.loss_v(target_vs, target_qs, out_v)
						  if value_masks is None else
						  self.loss_v(target_vs, target_qs, out_v, value_masks, q_masks))
			total_loss = self.loss_pi(target_pis, out_pi) + value_loss
			return total_loss.item()

	def loss_pi(self, targets, outputs):
		loss_ = torch.nn.KLDivLoss(reduction="batchmean")
		return loss_(outputs, targets)

		# loss_ = torch.nn.CrossEntropyLoss()
		# return loss_(outputs, targets)

		# return -torch.sum(torch.log(targets) * torch.exp(outputs)) / targets.size()[0]

	def loss_v(self, targets_V, targets_Q, outputs, value_mask=None, q_mask=None):
		if value_mask is None and q_mask is None:
			targets = (targets_V + self.args['q_weight'] * targets_Q) / (1+self.args['q_weight'])
			return torch.sum((targets - outputs) ** 2) / (targets_V.size()[0] * targets_V.size()[-1]) # Normalize by batch size * nb of players
		if value_mask is None or q_mask is None:
			raise ValueError('value_mask and q_mask must be supplied together')
		if value_mask.shape != targets_V.shape or q_mask.shape != targets_Q.shape:
			raise ValueError('Target masks must match the per-player target shape')
		value_weight = value_mask.to(outputs.dtype)
		q_weight = q_mask.to(outputs.dtype) * self.args['q_weight']
		weight = value_weight + q_weight
		active = weight > 0
		if not torch.any(active):
			# Retain a differentiable zero: policy-only batches must not update the
			# value head, but total_loss.backward() still has a single code path.
			return outputs.sum() * 0
		denominator = torch.where(active, weight, torch.ones_like(weight))
		targets = (targets_V * value_weight + targets_Q * q_weight) / denominator
		return torch.sum(((targets - outputs) ** 2)[active]) / active.sum()

	def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar', additional_keys={}):
		filepath = os.path.join(folder, filename)
		if not os.path.exists(folder):
			# print("Checkpoint Directory does not exist! Making directory {}".format(folder))
			os.mkdir(folder)
		# else:
		#     print("Checkpoint Directory exists! ")

		data = {
			'state_dict': self.nnet.state_dict(),
			'full_model': self.nnet,
		}
		data.update(additional_keys)
		torch.save(data, filepath)

	def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
		# https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
		filepath = os.path.join(folder, filename)
		if not os.path.exists(filepath):
			raise FileNotFoundError("No model in path {}".format(filepath))
		try:
			checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
			self.load_network(checkpoint, strict=(self.args['nn_version']>0))
		except Exception as error:
			raise ValueError("Cannot load checkpoint {}: {}".format(filepath, error)) from error
		self.switch_target('just_loaded')
		return checkpoint
			
	def load_network(self, checkpoint, strict=False):
		def load_not_strict(network_state_to_load, target_network):
			target_state = target_network.state_dict()
			for name, params in network_state_to_load.items():
				if name in target_state:
					target_params = target_state[name]
					if target_params.shape == params.shape:
						params.copy_(target_params)
						# print(f'no problem to copy {name}')
					elif target_params.dim() == params.dim():
						if len(target_params.shape) == 1:
							min_size = min(target_params.shape[0], params.shape[0])
							target_params[:min_size] = params[:min_size]
						elif len(target_params.shape) == 2:
							min_size_0, min_size_1 = min(target_params.shape[0], params.shape[0]), min(target_params.shape[1], params.shape[1])
							target_params[:min_size_0, :min_size_1] = params[:min_size_0, :min_size_1]
						elif len(target_params.shape) == 3:
							min_size_0, min_size_1, min_size_2 = min(target_params.shape[0], params.shape[0]), min(target_params.shape[1], params.shape[1]), min(target_params.shape[2], params.shape[2])
							target_params[:min_size_0, :min_size_1, :min_size_2] = params[:min_size_0, :min_size_1, :min_size_2]
						elif len(target_params.shape) == 4:
							min_size_0, min_size_1, min_size_2, min_size_3 = min(target_params.shape[0], params.shape[0]), min(target_params.shape[1], params.shape[1]), min(target_params.shape[2], params.shape[2]), min(target_params.shape[3], params.shape[3])
							target_params[:min_size_0, :min_size_1, :min_size_2, :min_size_3] = params[:min_size_0, :min_size_1, :min_size_2, :min_size_3]
						else:
							raise Exception('Unsupported number of dimensions')

						print(f'{name}: load {params.shape}  target {target_params.shape}, used {(min_size_0)}')
					else:
						print(f'{name}: couldnt match loaded {params.shape}  and target {target_params.shape}, using standard initialization')
						
				# else:
				# 	print(f'hasnt loaded layer {name} because not in target')

		if strict and (checkpoint['full_model'].version != self.args['nn_version']):
			print('Checkpoint includes NN version', checkpoint['full_model'].version, ', but you ask version', self.args['nn_version'], ' so not loading it and initiate knowledge transfer')
			self.requestKnowledgeTransfer = True
			return

		try:
			self.nnet.load_state_dict(checkpoint['state_dict'])
			self.nnet.version = checkpoint['full_model'].version
		except:
			if strict:
				print('Cant load NN ', checkpoint['full_model'].version, 'in checkpoint, so initiate knowledge transfer')
				self.requestKnowledgeTransfer = True
			else:
				if self.nnet.version > 0:
					try:
						load_not_strict(checkpoint['state_dict'], self.nnet)
						print('Could load state dict but NOT STRICT, saved archi-version was', checkpoint['full_model'].version)
					except:
						self.nnet = checkpoint['full_model']
						print('Had to load full model AS IS, saved archi-version was', checkpoint['full_model'].version, 'and WONT BE UPDATED')
						if input("Continue? [y|n]") != "y":
							sys.exit()
				else:
					self.nnet = checkpoint['full_model']


	def switch_target(self, mode):
		target_device = self.device[mode]
		if target_device == self.current_mode:
			return

		if target_device == 'cpu':
			self.nnet.cpu()
			torch.cuda.empty_cache()
			self.ort_session = None # Make ONNX export invalid
		elif target_device == 'onnx':
			self.nnet.cpu()
			self.export_and_load_onnx()
		elif target_device == 'cuda':
			self.nnet.cuda()
			self.ort_session = None # Make ONNX export invalid
		elif target_device == 'just_loaded':
			self.ort_session = None # Make ONNX export invalid
		
		self.current_mode = target_device

	def export_and_load_onnx(self):
		dummy_board         = torch.randn(self.board_size, dtype=torch.float32).unsqueeze(0)
		dummy_valid_actions = torch.BoolTensor(torch.randn(self.action_size)>0.5).unsqueeze(0)
		self.nnet.to('cpu')
		self.nnet.eval()

		with tempfile.TemporaryDirectory(prefix='azg-onnx-') as export_dir:
			temporary_file = os.path.join(export_dir, 'model.onnx')
			torch.onnx.export(
				self.nnet,
				(dummy_board, dummy_valid_actions),
				temporary_file,
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
			if ort.__version__ >= '1.17.0':
				# Convert ONNX file to most recent opset version
				model_with_old_opset = onnx.load(temporary_file)
				model_with_new_opset = onnx.version_converter.convert_version(model_with_old_opset, 21)
				onnx.save(model_with_new_opset, temporary_file)

			opts = ort.SessionOptions()
			opts.intra_op_num_threads = opts.inter_op_num_threads = 1
			opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
			self.ort_session = ort.InferenceSession(temporary_file, sess_options=opts, providers=['CPUExecutionProvider'])
			os.remove(temporary_file)

	def pick_examples(self, examples, sample_ids):
		if self.args['no_compression']:
			picked_examples = [examples[i] for i in sample_ids]
		else: 
			picked_examples = [pickle.loads(zlib.decompress(examples[i])) for i in sample_ids]
		return self._split_fields(picked_examples)

	def _split_fields(self, examples):
		"""Validate one homogeneous legacy or masked-target batch."""
		lengths = {len(example) for example in examples}
		if not lengths or not lengths <= {5, 7}:
			raise ValueError('Training examples must contain 5 or 7 fields')
		if lengths == {5, 7}:
			normalized = []
			for example in examples:
				if len(example) == 5:
					players = np.asarray(example[2]).shape
					example = tuple(example) + (np.ones(players, dtype=np.bool_),
												 np.ones(players, dtype=np.bool_))
				normalized.append(example)
			examples = normalized
		return list(zip(*examples))
	
	def reshape_boards(self, numpy_boards):
		# Some game needs to reshape boards before being an input of NNet
		return numpy_boards

	def number_params(self):
		total_params = sum(p.numel() for p in self.nnet.parameters())
		trainable_params = sum(p.numel() for p in self.nnet.parameters() if p.requires_grad)
		return total_params, trainable_params

if __name__ == "__main__":
	import argparse
	import os.path
	import time
	from GameSwitcher import import_game

	parser = argparse.ArgumentParser(description='NNet loader')
	parser.add_argument('game'               , action='store', default='splendor', help='The name of the game to play')
	parser.add_argument('--input'      , '-i', action='store', default=None , help='Input NN to load')
	parser.add_argument('--output'     , '-o', action='store', default=None , help='Prefix for output NN')
	parser.add_argument('--training'   , '-T', action='store', default=None , help='')
	parser.add_argument('--test'       , '-t', action='store', default=None , help='')

	parser.add_argument('--learn-rate' , '-l' , action='store', default=0.0003, type=float, help='')
	parser.add_argument('--dropout'    , '-d' , action='store', default=0.3   , type=float, help='')
	parser.add_argument('--epochs'     , '-p' , action='store', default=2    , type=int  , help='')
	parser.add_argument('--batch-size' , '-b' , action='store', default=32   , type=int  , help='')
	parser.add_argument('--nb-samples' , '-N' , action='store', default=9999 , type=int  , help='How many samples (in thousands)')
	parser.add_argument('--nn-version' , '-V' , action='store', default=-1   , type=int  , help='Which architecture to choose')
	parser.add_argument('--q-weight'   , '-q' , action='store', default=0.5  , type=float, help='Weight for mixing Q into value loss')
	args = parser.parse_args()
	Game, NNet, players, NUMBER_PLAYERS = import_game(args.game)

	output = (args.output if args.output else 'output_') + str(int(time.time()))[-6:]

	g = Game()
	nn_args = dict(
		lr=args.learn_rate,
		dropout=args.dropout,
		epochs=args.epochs,
		batch_size=args.batch_size,
		nn_version=args.nn_version,
		learn_rate=args.learn_rate,
		no_compression=False,
		q_weight=args.q_weight,
	)
	nnet = NNet(g, nn_args)
	if args.input:
		nnet.load_checkpoint(os.path.dirname(args.input), os.path.basename(args.input))
	elif args.nn_version == -1:
		raise Exception("You have to specify at least a NN file to load or a NN version")

	from fvcore.nn import FlopCountAnalysis
	dummy_board         = torch.randn(g.getBoardSize(), dtype=torch.float32).unsqueeze(0)
	dummy_valid_actions = torch.BoolTensor(torch.randn(1, g.getActionSize())>0.5)
	nnet.nnet.eval()
	flops = FlopCountAnalysis(nnet.nnet, (dummy_board, dummy_valid_actions))
	flops.unsupported_ops_warnings(False)
	# flops.uncalled_modules_warnings(False)
	print(f'V{nnet.nnet.version} -> {flops.total()/1000000:.1f} MFlops, nb params {nnet.number_params()[0]:.2e}')
	# print(flops.by_module().most_common(15))
	# print({ k:v//1000 for k,v in flops.by_module().items() if k.count('.') <= 2 })
	# breakpoint()

	if not args.training:
		if args.input:
			checkpoint = torch.load(args.input, map_location='cpu', weights_only=False)
			for k in sorted(checkpoint.keys()):
				if k not in ['state_dict', 'full_model', 'optim_state']:
					print(f'  {k}: {checkpoint[k]}')
			print(f'Board shape: {list(dummy_board.shape)}, valids shape: {list(dummy_valid_actions.shape)}')
		exit()
	with open(args.training, "rb") as f:
		examples = pickle.load(f)
	trainExamples = []
	for e in examples:
		trainExamples.extend(e)
	if args.test is None:
		splitNumber = len(trainExamples) // 10
		testExamples, trainExamples = trainExamples[-splitNumber:], trainExamples[:-splitNumber]
	else:
		with open(args.test, "rb") as f:
			examples = pickle.load(f)
		testExamples = []
		for e in examples:
			testExamples.extend(e)
	trainExamples = trainExamples[-args.nb_samples*1000:]
	print(f'Number of samples: training {len(trainExamples)}, testing {len(testExamples)}; number of epochs {args.epochs}')

	# print({ k:v//1000 for k,v in flops.by_module().items() if k.count('.') <= 1 })
	# breakpoint()
	
	# trainExamples_small = trainExamples[::30]
	# testExamples_small = testExamples[::30]
	# nnet.args['learn_rate'], nnet.args['lr'], nnet.args['batch_size'] = 3e-2, 3e-2, 32
	# nnet.optimizer = None
	# save_every = 1e5 // nnet.args['batch_size']
	# nnet.train(trainExamples_small, testExamples_small, '', save_every)

	# nnet.args['learn_rate'], nnet.args['lr'], nnet.args['batch_size'] = 3e-4, 3e-4, 512
	# nnet.optimizer = None
	save_every = (1e5 // nnet.args['batch_size']) - 1
	nnet.train(trainExamples, testExamples, output, save_every)

	nnet.save_checkpoint(output, filename='last.pt')
