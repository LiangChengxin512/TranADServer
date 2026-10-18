import argparse
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
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
			evaluate_experiment_model,
			load_checkpoint,
			load_processed_dataset,
			prepare_experiment_data,
			save_checkpoint,
			train_experiment_epoch,
			write_json,
		)
	finally:
		sys.argv = original_argv
	return {
		'build_model': build_model,
		'canonical_model_name': canonical_model_name,
		'checkpoint_path': checkpoint_path,
		'evaluate_experiment_model': evaluate_experiment_model,
		'load_checkpoint': load_checkpoint,
		'load_processed_dataset': load_processed_dataset,
		'prepare_experiment_data': prepare_experiment_data,
		'save_checkpoint': save_checkpoint,
		'train_experiment_epoch': train_experiment_epoch,
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
	parser = argparse.ArgumentParser(description='Train TranAD variants and trainable repository baselines with epoch-level metrics.')
	parser.add_argument('--dataset', type=str, default='synthetic')
	parser.add_argument('--model', type=str, default='TranAD')
	parser.add_argument('--epochs', type=int, default=5)
	parser.add_argument('--batch-size', type=int, default=128)
	parser.add_argument('--learning-rate', type=float, default=None)
	parser.add_argument('--weight-decay', type=float, default=1e-5)
	parser.add_argument('--lr-step', type=int, default=5)
	parser.add_argument('--lr-gamma', type=float, default=0.9)
	parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
	parser.add_argument('--less', action='store_true')
	parser.add_argument('--resume', action='store_true')
	parser.add_argument('--checkpoint-dir', type=str, default='experiment_checkpoints')
	parser.add_argument('--log-dir', type=str, default='experiment_logs')
	parser.add_argument('--eval-every', type=int, default=1)
	parser.add_argument('--aux-weight', type=float, default=1.0)
	parser.add_argument('--assoc-weight', type=float, default=0.01)
	parser.add_argument('--memory-weight', type=float, default=0.001)
	parser.add_argument('--score-agg', type=str, default='mean', choices=['mean', 'max', 'p95', 'topk'])
	parser.add_argument('--score-topk', type=parse_score_topk, default=3)
	parser.add_argument('--seed', type=int, default=1)
	return parser.parse_args()


def main():
	args = parse_args()
	utils = import_experiment_utils(args)
	args.model = utils['canonical_model_name'](args.model)
	if args.model == 'MERLIN':
		raise SystemExit('MERLIN is parameter-free in this repository. Use experiments/test_tranad_variants.py --model MERLIN.')
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)

	device = get_best_device(args.device, args.model)
	dtype = get_dtype_for_device(device)
	train_np, test_np, labels = utils['load_processed_dataset'](args.dataset, less=args.less)

	model = utils['build_model'](
		args.model,
		labels.shape[1],
		device,
		dtype,
		batch_size=args.batch_size,
		lr_override=args.learning_rate,
		seed=args.seed,
	)
	train_data, test_data, _, _ = utils['prepare_experiment_data'](model, train_np, test_np, device, dtype)

	optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.lr_gamma)

	config = vars(args).copy()
	config.update({'device': str(device), 'dtype': str(dtype)})
	latest_path = utils['checkpoint_path'](args.checkpoint_dir, args.model, args.dataset, 'latest.ckpt')
	best_path = utils['checkpoint_path'](args.checkpoint_dir, args.model, args.dataset, 'best.ckpt')
	log_folder = os.path.join(args.log_dir, f'{args.model}_{args.dataset}')
	os.makedirs(log_folder, exist_ok=True)
	log_path = os.path.join(log_folder, 'metrics.csv')
	config_path = os.path.join(log_folder, 'config.json')
	utils['write_json'](config_path, config)

	start_epoch = 0
	best_f1 = -1.0
	if args.resume and os.path.exists(latest_path):
		checkpoint = utils['load_checkpoint'](latest_path, model, optimizer, scheduler, device=device, dtype=dtype)
		start_epoch = checkpoint.get('epoch', -1) + 1
		best_f1 = checkpoint.get('metrics', {}).get('f1', -1.0)

	aux_weights = {
		'default': args.aux_weight,
		'association': args.assoc_weight,
		'memory_entropy': args.memory_weight,
	}
	rows = []
	if os.path.exists(log_path) and args.resume:
		rows = pd.read_csv(log_path).to_dict('records')

	for epoch in range(start_epoch, start_epoch + args.epochs):
		train_metrics = utils['train_experiment_epoch'](
			model, train_data, optimizer, scheduler, epoch, args.batch_size, aux_weights
		)
		row = {'epoch': epoch, **train_metrics}
		if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
			eval_metrics, per_dim, _, _ = utils['evaluate_experiment_model'](
				model, train_data, test_data, labels, args.batch_size,
				show_progress=True, score_agg=args.score_agg, topk=args.score_topk
			)
			row.update(eval_metrics)
			per_dim.to_csv(os.path.join(log_folder, f'per_dim_epoch_{epoch}.csv'), index=False)
		rows.append(row)
		pd.DataFrame(rows).to_csv(log_path, index=False)
		utils['save_checkpoint'](latest_path, model, optimizer, scheduler, epoch, row, config)
		if row.get('f1', -1.0) > best_f1:
			best_f1 = row['f1']
			utils['save_checkpoint'](best_path, model, optimizer, scheduler, epoch, row, config)
		print(row)


if __name__ == '__main__':
	main()
