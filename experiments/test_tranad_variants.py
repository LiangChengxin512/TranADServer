import argparse
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.device import get_best_device, get_dtype_for_device


def import_experiment_utils(args):
	original_argv = sys.argv[:]
	sys.argv = [original_argv[0], '--dataset', args.dataset, '--model', args.model]
	try:
		from src.experiment_utils import (
			build_model,
			canonical_model_name,
			checkpoint_path,
			evaluate_merlin,
			evaluate_experiment_model,
			load_checkpoint,
			load_processed_dataset,
			prepare_experiment_data,
			write_json,
		)
	finally:
		sys.argv = original_argv
	return {
		'build_model': build_model,
		'canonical_model_name': canonical_model_name,
		'checkpoint_path': checkpoint_path,
		'evaluate_merlin': evaluate_merlin,
		'evaluate_experiment_model': evaluate_experiment_model,
		'load_checkpoint': load_checkpoint,
		'load_processed_dataset': load_processed_dataset,
		'prepare_experiment_data': prepare_experiment_data,
		'write_json': write_json,
	}


def parse_score_topk(value):
	value = str(value).strip()
	if value.lower() == 'auto':
		return 'auto'
	try:
		topk = int(value)
	except ValueError as exc:
		raise argparse.ArgumentTypeError('--score-topk must be a positive integer or auto') from exc
	if topk < 1:
		raise argparse.ArgumentTypeError('--score-topk must be a positive integer or auto')
	return topk


def parse_args():
	parser = argparse.ArgumentParser(description='Evaluate TranAD variants and repository baselines.')
	parser.add_argument('--dataset', type=str, default='synthetic')
	parser.add_argument('--model', type=str, default='TranAD')
	parser.add_argument('--batch-size', type=int, default=128)
	parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
	parser.add_argument('--checkpoint-dir', type=str, default='experiment_checkpoints')
	parser.add_argument('--checkpoint', type=str, default=None)
	parser.add_argument('--output-dir', type=str, default='experiment_logs')
	parser.add_argument('--less', action='store_true')
	parser.add_argument('--score-agg', type=str, default='mean', choices=['mean', 'max', 'p95', 'topk'])
	parser.add_argument('--score-topk', type=parse_score_topk, default=3)
	parser.add_argument('--seed', type=int, default=None)
	return parser.parse_args()


def main():
	args = parse_args()
	if args.seed is not None:
		random.seed(args.seed)
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(args.seed)
	utils = import_experiment_utils(args)
	args.model = utils['canonical_model_name'](args.model)
	train_np, test_np, labels = utils['load_processed_dataset'](args.dataset, less=args.less)
	out_folder = os.path.join(args.output_dir, f'{args.model}_{args.dataset}')
	os.makedirs(out_folder, exist_ok=True)
	if args.model == 'MERLIN':
		metrics, per_dim = utils['evaluate_merlin'](test_np, labels)
		utils['write_json'](os.path.join(out_folder, 'test_metrics.json'), {
			'checkpoint': None,
			'device': 'cpu',
			'dtype': 'numpy',
			**metrics,
		})
		per_dim.to_csv(os.path.join(out_folder, 'test_per_dim.csv'), index=False)
		print(metrics)
		return
	device = get_best_device(args.device, args.model)
	dtype = get_dtype_for_device(device)

	model = utils['build_model'](args.model, labels.shape[1], device, dtype, batch_size=args.batch_size)
	ckpt = args.checkpoint or utils['checkpoint_path'](args.checkpoint_dir, args.model, args.dataset, 'best.ckpt')
	if not os.path.exists(ckpt):
		ckpt = utils['checkpoint_path'](args.checkpoint_dir, args.model, args.dataset, 'latest.ckpt')
	utils['load_checkpoint'](ckpt, model, device=device, dtype=dtype)

	train_data, test_data, _, _ = utils['prepare_experiment_data'](model, train_np, test_np, device, dtype)
	metrics, per_dim, _, _ = utils['evaluate_experiment_model'](
		model, train_data, test_data, labels, args.batch_size,
		show_progress=True, score_agg=args.score_agg, topk=args.score_topk
	)
	utils['write_json'](os.path.join(out_folder, 'test_metrics.json'), {
		'checkpoint': ckpt,
		'device': str(device),
		'dtype': str(dtype),
		**metrics,
	})
	per_dim.to_csv(os.path.join(out_folder, 'test_per_dim.csv'), index=False)
	print(metrics)


if __name__ == '__main__':
	main()
