#!/usr/bin/env python3
"""Train and/or evaluate one progressive TranAD M0--M4 experiment."""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.device import get_best_device, get_dtype_for_device


DATASET_PREFIX = {
	'SMD': 'machine-1-1_',
	'SMAP': 'P-1_',
	'MSL': 'C-1_',
	'UCR': '136_',
	'NAB': 'ec2_request_latency_system_failure_',
}
DATASETS = ('synthetic', 'SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'MSDS', 'UCR', 'MBA', 'NAB')
VARIANTS = ('M0', 'M1', 'M2', 'M3', 'M4')


def import_experiment_utils(dataset, model_name):
	original_argv = sys.argv[:]
	sys.argv = [original_argv[0], '--dataset', dataset, '--model', model_name]
	try:
		from src.experiment_utils import (
			build_model,
			evaluate_experiment_model,
			load_checkpoint,
			load_processed_dataset,
			prepare_experiment_data,
			save_checkpoint,
			train_experiment_epoch,
			validation_reconstruction_loss,
			write_json,
		)
	finally:
		sys.argv = original_argv
	return {
		'build_model': build_model,
		'evaluate_experiment_model': evaluate_experiment_model,
		'load_checkpoint': load_checkpoint,
		'load_processed_dataset': load_processed_dataset,
		'prepare_experiment_data': prepare_experiment_data,
		'save_checkpoint': save_checkpoint,
		'train_experiment_epoch': train_experiment_epoch,
		'validation_reconstruction_loss': validation_reconstruction_loss,
		'write_json': write_json,
	}


def processed_files(dataset):
	folder = PROJECT_ROOT / 'processed' / dataset
	prefix = DATASET_PREFIX.get(dataset, '')
	return [folder / f'{prefix}{split}.npy' for split in ('train', 'test', 'labels')]


def warn_dataset_missing(dataset):
	message = f'Warning: Dataset [{dataset}] not found, skipping...'
	print(f'\033[93m{message}\033[0m', file=sys.stderr)


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--mode', choices=('train', 'test', 'train-test'), default='train-test')
	parser.add_argument('--variant', choices=VARIANTS, required=True)
	parser.add_argument('--dataset', choices=DATASETS, required=True)
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--epochs', type=int, default=10)
	parser.add_argument('--batch-size', type=int, default=128)
	parser.add_argument('--validation-fraction', type=float, default=0.1)
	parser.add_argument('--patience', type=int, default=3)
	parser.add_argument('--min-delta', type=float, default=1e-4)
	parser.add_argument('--learning-rate', type=float, default=None)
	parser.add_argument('--weight-decay', type=float, default=1e-5)
	parser.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
	parser.add_argument('--checkpoint', type=str, default=None,
		help='Checkpoint path; defaults to checkpoints/{dataset}/{variant}/seed_{seed}/model_best.pt.')
	parser.add_argument('--log-dir', type=str, default=None)
	parser.add_argument('--metrics-csv', type=str, default='experiments_results/ablation_raw_metrics.csv')
	parser.add_argument('--score-agg', choices=('mean', 'adaptive_topk'), default=None)
	parser.add_argument('--less', action='store_true', help='Train using 20%% of the available training series.')
	return parser.parse_args()


def seed_everything(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def append_metrics(path, row):
	path = Path(path)
	if not path.is_absolute():
		path = PROJECT_ROOT / path
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('a', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(handle, fieldnames=list(row))
		if path.stat().st_size == 0:
			writer.writeheader()
		writer.writerow(row)


def main():
	args = parse_args()
	os.chdir(PROJECT_ROOT)
	files = processed_files(args.dataset)
	if not all(path.is_file() for path in files):
		warn_dataset_missing(args.dataset)
		return 0
	if args.mode in ('train', 'train-test') and args.epochs < 1:
		raise SystemExit('--epochs must be at least 1 when training.')
	if args.batch_size < 1:
		raise SystemExit('--batch-size must be at least 1.')
	if not 0.0 < args.validation_fraction < 1.0:
		raise SystemExit('--validation-fraction must be between 0 and 1.')
	if args.patience < 1:
		raise SystemExit('--patience must be at least 1.')
	if args.min_delta < 0:
		raise SystemExit('--min-delta must be non-negative.')

	seed_everything(args.seed)
	model_name = f'TranAD_{args.variant}'
	utils = import_experiment_utils(args.dataset, model_name)
	device = get_best_device(args.device, model_name)
	dtype = get_dtype_for_device(device)
	train_np, test_np, labels = utils['load_processed_dataset'](args.dataset, less=args.less)
	validation_size = max(1, int(np.ceil(train_np.shape[0] * args.validation_fraction)))
	if validation_size >= train_np.shape[0]:
		raise ValueError('Training data must contain at least two points for a train/validation split.')
	train_split = train_np[:-validation_size]
	validation_split = train_np[-validation_size:]
	model = utils['build_model'](
		model_name,
		labels.shape[1],
		device,
		dtype,
		batch_size=args.batch_size,
		lr_override=args.learning_rate,
		seed=args.seed,
	)
	train_data, validation_data, _, _ = utils['prepare_experiment_data'](
		model, train_split, validation_split, device, dtype
	)
	calibration_data = torch.cat((train_data, validation_data), dim=0)
	_, test_data, _, _ = utils['prepare_experiment_data'](
		model, train_split, test_np, device, dtype
	)
	checkpoint_path = Path(args.checkpoint) if args.checkpoint else (
		PROJECT_ROOT / 'checkpoints' / args.dataset / args.variant / f'seed_{args.seed}' / 'model_best.pt'
	)
	if not checkpoint_path.is_absolute():
		checkpoint_path = PROJECT_ROOT / checkpoint_path
	log_dir = Path(args.log_dir) if args.log_dir else (
		PROJECT_ROOT / 'experiment_logs' / args.dataset / args.variant / f'seed_{args.seed}'
	)
	if not log_dir.is_absolute():
		log_dir = PROJECT_ROOT / log_dir
	log_dir.mkdir(parents=True, exist_ok=True)
	score_agg = args.score_agg or ('adaptive_topk' if args.variant == 'M4' else 'mean')
	config = {
		'variant': args.variant,
		'dataset': args.dataset,
		'seed': args.seed,
		'epochs': args.epochs,
		'validation_fraction': args.validation_fraction,
		'patience': args.patience,
		'min_delta': args.min_delta,
		'batch_size': args.batch_size,
		'device': str(device),
		'dtype': str(dtype),
		'score_agg': score_agg,
		'checkpoint': str(checkpoint_path),
	}

	if args.mode in ('train', 'train-test'):
		optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=args.weight_decay)
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
		history = []
		aux_weights = {'default': 1.0, 'association': 0.01, 'memory_entropy': 0.001}
		best_validation_loss = float('inf')
		early_stopping_reference = float('inf')
		best_epoch = None
		epochs_without_improvement = 0
		for epoch in range(args.epochs):
			metrics = utils['train_experiment_epoch'](
				model, train_data, optimizer, scheduler, epoch, args.batch_size, aux_weights
			)
			validation_loss = utils['validation_reconstruction_loss'](
				model, validation_data, args.batch_size, epoch
			)
			row = {
				'epoch': epoch + 1,
				**metrics,
				'validation_reconstruction_loss': validation_loss,
			}
			history.append(row)
			print(row)
			if validation_loss < best_validation_loss:
				best_validation_loss = validation_loss
				best_epoch = epoch
				utils['save_checkpoint'](
					str(checkpoint_path),
					model,
					optimizer,
					scheduler,
					epoch,
					{'validation_reconstruction_loss': validation_loss},
					config,
				)
			if validation_loss < early_stopping_reference - args.min_delta:
				early_stopping_reference = validation_loss
				epochs_without_improvement = 0
			else:
				epochs_without_improvement += 1
			if epochs_without_improvement >= args.patience:
				print(
					f'Early stopping after epoch {epoch + 1}: validation loss did not improve '
					f'by at least {args.min_delta:g} for {args.patience} consecutive epochs.'
				)
				break
		pd.DataFrame(history).to_csv(log_dir / 'train_metrics.csv', index=False)
		if best_epoch is None:
			raise RuntimeError('Training finished without a finite validation checkpoint.')
		print(f'Saved best checkpoint from epoch {best_epoch + 1}: {checkpoint_path}')
	elif not checkpoint_path.is_file():
		raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

	if args.mode == 'train':
		return 0
	checkpoint_info = utils['load_checkpoint'](str(checkpoint_path), model, device=device, dtype=dtype)
	best_epoch = int(checkpoint_info.get('epoch', -1)) + 1
	best_validation_loss = checkpoint_info.get('metrics', {}).get('validation_reconstruction_loss')

	metrics, per_dim, _, _ = utils['evaluate_experiment_model'](
		model,
		calibration_data,
		test_data,
		labels,
		args.batch_size,
		show_progress=True,
		score_agg=score_agg,
		topk=3,
	)
	metrics.update({
		'variant': args.variant,
		'dataset': args.dataset,
		'seed': args.seed,
		'best_epoch': best_epoch,
		'best_validation_reconstruction_loss': best_validation_loss,
	})
	utils['write_json'](str(log_dir / 'test_metrics.json'), {
		'checkpoint': str(checkpoint_path),
		'device': str(device),
		'dtype': str(dtype),
		**metrics,
	})
	per_dim.to_csv(log_dir / 'test_per_dim.csv', index=False)
	row = {
		'seed': args.seed,
		'dataset': args.dataset,
		'model': args.variant,
		'precision': metrics['precision'],
		'recall': metrics['recall'],
		'f1': metrics['f1'],
		'roc_auc': metrics['roc_auc'],
		'best_epoch': best_epoch,
		'best_validation_reconstruction_loss': best_validation_loss,
		'score_agg': metrics['score_agg'],
		'selected_topk': metrics.get('selected_topk'),
		'threshold': metrics.get('threshold'),
		'device': str(device),
		'checkpoint': str(checkpoint_path),
	}
	append_metrics(args.metrics_csv, row)
	print(metrics)
	print(f'Appended metrics: {args.metrics_csv}')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
