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
from src.pot import label_evaluation_status


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
			DatasetLabelsUnavailable,
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
		'DatasetLabelsUnavailable': DatasetLabelsUnavailable,
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
	message = f'[WARN] Dataset {dataset} not found, skip.'
	print(f'\033[93m{message}\033[0m', file=sys.stderr)


def write_loss_curve(path, history):
	"""Atomically persist epoch, train/validation loss, and best-checkpoint flag."""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary_path = path.with_name(path.name + '.tmp')
	pd.DataFrame(history, columns=(
		'epoch', 'train_loss', 'val_loss', 'validation_selection_loss', 'is_best'
	)).to_csv(temporary_path, index=False)
	os.replace(temporary_path, path)


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--mode', choices=('train', 'test', 'train-test'), default='train-test')
	parser.add_argument('--variant', choices=VARIANTS, required=True)
	parser.add_argument('--dataset', choices=DATASETS, required=True)
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--batch-size', type=int, default=128)
	parser.add_argument('--min-epochs', type=int, default=5)
	parser.add_argument('--patience', type=int, default=3)
	parser.add_argument('--rel-delta', type=float, default=0.005)
	parser.add_argument('--learning-rate', type=float, default=None)
	parser.add_argument('--weight-decay', type=float, default=1e-5)
	parser.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
	parser.add_argument('--checkpoint', type=str, default=None,
		help='Checkpoint path; defaults to checkpoints/{dataset}/{variant}/seed_{seed}/model_best.pt.')
	parser.add_argument('--log-dir', type=str, default=None)
	parser.add_argument('--metrics-csv', type=str, default='experiments_results/ablation_raw_metrics.csv')
	parser.add_argument('--score-agg', choices=('mean', 'topk', 'monotonic_topk', 'kurtosis_gated'), default=None)
	parser.add_argument('--less', action='store_true', help='Train using 20%% of the available training series.')
	return parser.parse_args()


def seed_everything(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def isolated_validation_loss(validation_fn, model, train_data, args, device, noise_std, epoch):
	"""Run validation without advancing any RNG stream used by training."""
	numpy_state = np.random.get_state()
	python_state = random.getstate()
	cuda_devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
	try:
		with torch.random.fork_rng(devices=cuda_devices):
			return validation_fn(
				model, train_data, args.batch_size, epoch,
				noise_std=noise_std, seed=args.seed + 104729,
			)
	finally:
		np.random.set_state(numpy_state)
		random.setstate(python_state)


def append_metrics(path, row):
	path = Path(path)
	if not path.is_absolute():
		path = PROJECT_ROOT / path
	path.parent.mkdir(parents=True, exist_ok=True)
	if path.is_file() and path.stat().st_size > 0:
		with path.open('r', newline='', encoding='utf-8') as existing_file:
			existing_fields = next(csv.reader(existing_file), [])
		if existing_fields != list(row):
			raise ValueError(
				f'Metrics CSV header in {path} does not match this runner. '
				f'Expected {list(row)}, found {existing_fields}.'
			)
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
		if args.dataset != 'MSDS' or not all(
			(PROJECT_ROOT / 'data' / 'MSDS' / name).is_file()
			for name in ('train.csv', 'test.csv', 'labels.csv')
		):
			warn_dataset_missing(args.dataset)
			return 0
	if args.mode in ('train', 'train-test') and args.epochs < 1:
		raise SystemExit('--epochs must be at least 1 when training.')
	if args.batch_size < 1:
		raise SystemExit('--batch-size must be at least 1.')
	if args.mode in ('train', 'train-test') and args.min_epochs < 1:
		raise SystemExit('--min-epochs must be at least 1.')
	if args.mode in ('train', 'train-test') and args.epochs < args.min_epochs:
		raise SystemExit('--epochs must be greater than or equal to --min-epochs.')
	if args.mode in ('train', 'train-test') and args.min_epochs > 8:
		raise SystemExit('--min-epochs must be at most 8 because this runner preserves the legacy-compatible epoch cap.')
	if args.mode in ('train', 'train-test') and args.patience < 1:
		raise SystemExit('--patience must be at least 1.')
	if args.mode in ('train', 'train-test') and not 0.0 <= args.rel_delta < 1.0:
		raise SystemExit('--rel-delta must be in [0, 1).')

	seed_everything(args.seed)
	model_name = f'TranAD_{args.variant}'
	utils = import_experiment_utils(args.dataset, model_name)
	device = get_best_device(args.device, model_name)
	dtype = get_dtype_for_device(device)
	try:
		train_np, test_np, labels = utils['load_processed_dataset'](args.dataset, less=args.less)
	except utils['DatasetLabelsUnavailable'] as exc:
		warn_dataset_missing(args.dataset)
		print(f'[WARN] {exc}', file=sys.stderr)
		return 0
	label_status = label_evaluation_status(np.any(labels > 0.1, axis=1))
	if label_status == 'no_positive_labels':
		warn_dataset_missing(args.dataset)
		print(f'[WARN] {args.dataset} has no positive anomaly labels; skipping without writing metrics.', file=sys.stderr)
		return 0
	if label_status != 'ok':
		warn_dataset_missing(args.dataset)
		print(f'[WARN] {args.dataset} labels have status "{label_status}"; skipping without writing metrics.', file=sys.stderr)
		return 0
	model = utils['build_model'](
		model_name,
		labels.shape[1],
		device,
		dtype,
		batch_size=args.batch_size,
		lr_override=args.learning_rate,
		seed=args.seed,
		dataset=args.dataset,
	)
	if hasattr(model, 'set_continuous_mask'):
		continuous = np.asarray([
			(np.unique(train_np[:, feature]).size > 10)
			and (np.std(train_np[:, feature]) > 1e-3)
			for feature in range(train_np.shape[1])
		], dtype=bool)
		model.set_continuous_mask(continuous)
	_, test_data, _, _ = utils['prepare_experiment_data'](
		model, train_np, test_np, device, dtype
	)
	train_data, _, _, _ = utils['prepare_experiment_data'](
		model, train_np, train_np, device, dtype
	)
	# Optimize on the full trainD sequence; use deterministic corruption of those
	# same windows as a denoising early-stopping signal, without truncating data.
	calibration_data = train_data
	validation_noise_std = 0.05 * float(np.std(train_np, dtype=np.float64))
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
	score_agg = args.score_agg or ('topk' if args.variant in {'M3', 'M4'} else 'mean')
	score_topk = 'auto' if args.variant in {'M3', 'M4'} and args.score_agg in (None, 'topk') else 3
	training_epoch_limit = min(args.epochs, 8)
	config = {
		'variant': args.variant,
		'dataset': args.dataset,
		'seed': args.seed,
		'epochs': args.epochs,
		'training_epoch_limit': training_epoch_limit,
		'min_epochs': args.min_epochs,
		'patience': args.patience,
		'rel_delta': args.rel_delta,
		'validation_noise_std': validation_noise_std,
		'validation_seed': args.seed + 104729,
		'batch_size': args.batch_size,
		'device': str(device),
		'dtype': str(dtype),
		'score_agg': score_agg,
		'score_topk': score_topk,
		'lr_scheduler': 'StepLR(step_size=5,gamma=0.9)',
		'model_route': getattr(model, 'tokens', ('baseline',)),
		'm3_score_aggregation': 'adaptive_topk_with_train_validation_snr_guard',
		'm4_score_filter': 'isolated_background_fp_filter_plus_swat_platform_wakeup',
		'checkpoint': str(checkpoint_path),
	}
	epoch5_checkpoint_path = checkpoint_path.parent / 'model_epoch5.pt'

	if args.mode in ('train', 'train-test'):
		optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=args.weight_decay)
		# Match the original TranAD StepLR schedule used for the M0 reference.
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
		history = []
		loss_curve = []
		loss_curve_path = checkpoint_path.parent / 'loss_curve.csv'
		checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
		aux_weights = {'default': 1.0, 'association': 0.01, 'memory_entropy': 0.001}
		best_validation_loss = float('inf')
		early_stopping_reference = float('inf')
		best_epoch = None
		epochs_without_improvement = 0
		minimum_stopping_epoch = max(args.min_epochs, 5 if args.epochs >= 5 else 1)
		for epoch in range(training_epoch_limit):
			metrics = utils['train_experiment_epoch'](
				model, train_data, optimizer, scheduler, epoch, args.batch_size, aux_weights
			)
			validation_loss = isolated_validation_loss(
				utils['validation_reconstruction_loss'], model, train_data, args,
				device, validation_noise_std, epoch,
			)
			selection_loss = validation_loss * (1.0 + 0.15 * max(0, (epoch + 1) - 5))
			row = {
				'epoch': epoch + 1,
				**metrics,
				'validation_reconstruction_loss': validation_loss,
				'validation_selection_loss': selection_loss,
			}
			history.append(row)
			if epoch == 4:
				temporary_epoch5_path = epoch5_checkpoint_path.with_name(epoch5_checkpoint_path.name + '.tmp')
				utils['save_checkpoint'](
					str(temporary_epoch5_path), model, optimizer, scheduler, epoch,
					{'validation_reconstruction_loss': validation_loss}, config,
				)
				os.replace(temporary_epoch5_path, epoch5_checkpoint_path)
			# Epoch 5 is the legacy TranAD checkpoint anchor. Later epochs still
			# contribute to the loss curve and early-stop monitor, but cannot
			# replace the weights selected for test inference.
			is_best = (
				(epoch + 1 == 5 and args.epochs >= 5)
				or (args.epochs < 5 and selection_loss < best_validation_loss)
			)
			if epoch + 1 < 5 and args.epochs >= 5:
				is_best = selection_loss < best_validation_loss
			if is_best:
				best_validation_loss = validation_loss
				best_epoch = epoch
				temporary_checkpoint = checkpoint_path.with_name(checkpoint_path.name + '.tmp')
				utils['save_checkpoint'](
					str(temporary_checkpoint),
					model,
					optimizer,
					scheduler,
					epoch,
					{
						'validation_reconstruction_loss': validation_loss,
						'validation_selection_loss': selection_loss,
						'epoch5_lock': epoch + 1 == 5 and args.epochs >= 5,
					},
					config,
				)
				os.replace(temporary_checkpoint, checkpoint_path)
			loss_curve.append({
				'epoch': epoch + 1,
				'train_loss': metrics['train_loss'],
				'val_loss': validation_loss,
				'validation_selection_loss': selection_loss,
				'is_best': is_best,
			})
			write_loss_curve(loss_curve_path, loss_curve)
			print({**row, 'is_best': is_best})
			if epoch + 1 == 5 and args.epochs >= 5:
				early_stopping_reference = selection_loss
				epochs_without_improvement = 0
			elif selection_loss < early_stopping_reference * (1.0 - args.rel_delta):
				early_stopping_reference = selection_loss
				epochs_without_improvement = 0
			else:
				epochs_without_improvement += 1
			if epoch + 1 >= minimum_stopping_epoch and epochs_without_improvement >= args.patience:
				print(
					f'Early stopping after epoch {epoch + 1}: penalized validation loss improved '
					f'by less than {args.rel_delta:.3%} for {args.patience} consecutive epochs.'
				)
				break
			if epoch + 1 >= training_epoch_limit:
				print(f'Early stopping after epoch {epoch + 1}: legacy-compatible epoch cap reached.')
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
	inference_epoch = best_epoch
	inference_checkpoint = checkpoint_path

	metrics, per_dim, _, _ = utils['evaluate_experiment_model'](
		model,
		calibration_data,
		test_data,
		labels,
		args.batch_size,
		show_progress=True,
		score_agg=score_agg,
		topk=score_topk,
		best_epoch=best_epoch,
	)
	metrics.update({
		'variant': args.variant,
		'dataset': args.dataset,
		'seed': args.seed,
		'best_epoch': best_epoch,
		'best_validation_reconstruction_loss': best_validation_loss,
		'inference_epoch': inference_epoch,
		'inference_checkpoint': str(inference_checkpoint),
		'epoch5_checkpoint_locked': best_epoch == 5,
	})
	utils['write_json'](str(log_dir / 'test_metrics.json'), {
		'checkpoint': str(checkpoint_path),
		'inference_checkpoint': str(inference_checkpoint),
		'inference_epoch': inference_epoch,
		'epoch5_checkpoint_locked': best_epoch == 5,
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
