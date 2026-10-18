"""Benchmark batch inference latency and throughput for trained anomaly detectors.

The timer covers model forward computation only: data-window construction,
checkpoint loading, and metric post-processing are deliberately excluded.
Most repository baselines only implement single-window ``forward`` methods;
for those models a batch is evaluated as the same sequential per-window loop
used by the existing experiment inference code.  TranAD and TranAD_E use their
native vectorized batch forward pass.
"""

import argparse
import contextlib
import csv
import io
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.device import get_best_device, get_dtype_for_device
from src.experiment_utils import (
	build_model,
	canonical_model_name,
	load_checkpoint,
	load_processed_dataset,
	prepare_experiment_data,
)
from src.merlin import merlin


MODELS = ('MERLIN', 'DAGMM', 'MSCRED', 'MAD_GAN', 'USAD', 'MTAD_GAT', 'CAE_M', 'GDN', 'TranAD', 'TranAD_E')
DATASETS = ('synthetic', 'MBA', 'SMAP')


def parse_args():
	parser = argparse.ArgumentParser(description='Benchmark batch latency and throughput from trained checkpoints.')
	parser.add_argument('--checkpoint-dir', default='experiment_checkpoints/multiseed_adaptive_k/seed_5')
	parser.add_argument('--output-dir', default='experiment_tables/inference_benchmarks/seed_5')
	parser.add_argument('--models', nargs='+', default=list(MODELS), choices=MODELS)
	parser.add_argument('--datasets', nargs='+', default=list(DATASETS), choices=DATASETS)
	parser.add_argument('--batch-size', type=int, default=128)
	parser.add_argument('--warmup-batches', type=int, default=5)
	parser.add_argument('--timed-batches', type=int, default=10)
	parser.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
	parser.add_argument('--merlin-min-length', type=int, default=60)
	parser.add_argument('--merlin-max-length', type=int, default=62)
	return parser.parse_args()


def synchronize(device):
	if device.type == 'cuda':
		torch.cuda.synchronize(device)
	elif device.type == 'mps':
		torch.mps.synchronize()


def neural_batch_forward(model, model_name, batch):
	"""Execute the repository's deployed forward semantics for one batch."""
	if model_name in {'TranAD', 'TranAD_E'}:
		local_bs = batch.shape[0]
		window = batch.permute(1, 0, 2)
		elem = window[-1].view(1, local_bs, window.shape[-1])
		return model(window, elem)

	# These baselines define forward() for one window only.  Keeping this loop
	# measures their actual repository inference path while reporting per-batch time.
	outputs = []
	for window in batch:
		if model_name == 'MTAD_GAT':
			outputs.append(model(window, None))
		else:
			outputs.append(model(window))
	return outputs


def merlin_batch_forward(batch, min_length, max_length):
	"""MERLIN is parameter-free and sequence-level, not a neural mini-batch model."""
	with contextlib.redirect_stdout(io.StringIO()):
		return merlin(batch, min_length, max_length)


def benchmark_neural(model, model_name, data, device, batch_size, warmup_batches, timed_batches):
	n_complete_batches = data.shape[0] // batch_size
	required = warmup_batches + timed_batches
	if n_complete_batches < required:
		raise ValueError(
			f'Need {required} complete batches of size {batch_size}, but only {n_complete_batches} are available.'
		)
	model.eval()
	with torch.inference_mode():
		for index in range(warmup_batches):
			start = index * batch_size
			neural_batch_forward(model, model_name, data[start:start + batch_size])

		latencies = []
		for index in range(warmup_batches, required):
			start = index * batch_size
			batch = data[start:start + batch_size]
			synchronize(device)
			begin = time.perf_counter()
			neural_batch_forward(model, model_name, batch)
			synchronize(device)
			latencies.append(time.perf_counter() - begin)
	return np.asarray(latencies, dtype=np.float64)


def benchmark_merlin(test_np, batch_size, warmup_batches, timed_batches, min_length, max_length):
	if batch_size < max_length:
		raise ValueError('--batch-size must be at least --merlin-max-length for MERLIN.')
	n_complete_batches = len(test_np) // batch_size
	required = warmup_batches + timed_batches
	if n_complete_batches < required:
		raise ValueError(
			f'Need {required} complete batches of size {batch_size}, but only {n_complete_batches} are available.'
		)
	for index in range(warmup_batches):
		start = index * batch_size
		merlin_batch_forward(test_np[start:start + batch_size], min_length, max_length)
	latencies = []
	for index in range(warmup_batches, required):
		start = index * batch_size
		begin = time.perf_counter()
		merlin_batch_forward(test_np[start:start + batch_size], min_length, max_length)
		latencies.append(time.perf_counter() - begin)
	return np.asarray(latencies, dtype=np.float64)


def summarize(model, dataset, checkpoint, latencies, batch_size, device, inference_mode):
	throughputs = batch_size / latencies
	return {
		'model': model,
		'dataset': dataset,
		'checkpoint': checkpoint or '',
		'device': str(device),
		'inference_mode': inference_mode,
		'batch_size': batch_size,
		'timed_batches': int(len(latencies)),
		'batch_latency_ms_mean': float(np.mean(latencies) * 1000.0),
		'batch_latency_ms_std': float(np.std(latencies) * 1000.0),
		'throughput_samples_per_s_mean': float(np.mean(throughputs)),
		'throughput_samples_per_s_std': float(np.std(throughputs)),
		'status': 'ok',
		'error': '',
	}


def error_row(model, dataset, checkpoint, batch_size, error):
	return {
		'model': model, 'dataset': dataset, 'checkpoint': checkpoint or '', 'device': '',
		'inference_mode': '', 'batch_size': batch_size, 'timed_batches': 0,
		'batch_latency_ms_mean': '', 'batch_latency_ms_std': '',
		'throughput_samples_per_s_mean': '', 'throughput_samples_per_s_std': '',
		'status': 'error', 'error': str(error),
	}


def write_results(rows, args):
	os.makedirs(args.output_dir, exist_ok=True)
	csv_path = os.path.join(args.output_dir, 'batch_inference_benchmark.csv')
	json_path = os.path.join(args.output_dir, 'batch_inference_benchmark.json')
	fields = list(rows[0].keys())
	with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(handle, fieldnames=fields)
		writer.writeheader()
		writer.writerows(rows)
	with open(json_path, 'w', encoding='utf-8') as handle:
		json.dump({
			'configuration': vars(args),
			'timing_scope': 'forward computation only; checkpoint loading and data preparation excluded',
			'merlin_note': 'MERLIN is parameter-free and sequence-level. Its rows time one batch-sized sequence and throughput is time-series points/s.',
			'results': rows,
		}, handle, indent=2)
	return csv_path, json_path


def main():
	args = parse_args()
	if args.batch_size < 1 or args.warmup_batches < 0 or args.timed_batches < 1:
		raise ValueError('batch size must be positive, warmup batches non-negative, and timed batches positive.')
	rows = []
	for dataset in args.datasets:
		train_np, test_np, labels = load_processed_dataset(dataset)
		for requested_model in args.models:
			model_name = canonical_model_name(requested_model)
			checkpoint = '' if model_name == 'MERLIN' else os.path.join(
				args.checkpoint_dir, f'{model_name}_{dataset}', 'best.ckpt'
			)
			try:
				if model_name == 'MERLIN':
					latencies = benchmark_merlin(
						test_np, args.batch_size, args.warmup_batches, args.timed_batches,
						args.merlin_min_length, args.merlin_max_length,
					)
					row = summarize(model_name, dataset, checkpoint, latencies, args.batch_size, 'cpu', 'sequence_per_batch')
				else:
					if not os.path.isfile(checkpoint):
						raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
					device = get_best_device(args.device, model_name)
					dtype = get_dtype_for_device(device)
					model = build_model(model_name, labels.shape[1], device, dtype, batch_size=args.batch_size)
					load_checkpoint(checkpoint, model, device=device, dtype=dtype)
					_, test_data, _, _ = prepare_experiment_data(model, train_np, test_np, device, dtype)
					latencies = benchmark_neural(
						model, model_name, test_data, device, args.batch_size,
						args.warmup_batches, args.timed_batches,
					)
					mode = 'native_vectorized_batch' if model_name in {'TranAD', 'TranAD_E'} else 'sequential_windows_in_batch'
					row = summarize(model_name, dataset, checkpoint, latencies, args.batch_size, device, mode)
				rows.append(row)
				print(f"{dataset:9s} {model_name:10s} {row['batch_latency_ms_mean']:.2f} ms/batch  "
					f"{row['throughput_samples_per_s_mean']:.2f} samples/s")
			except Exception as exc:
				rows.append(error_row(model_name, dataset, checkpoint, args.batch_size, exc))
				print(f'{dataset:9s} {model_name:10s} ERROR: {exc}', file=sys.stderr)

	csv_path, json_path = write_results(rows, args)
	print(f'CSV: {csv_path}')
	print(f'JSON: {json_path}')
	if any(row['status'] != 'ok' for row in rows):
		raise SystemExit('One or more benchmarks failed; see the output files for details.')


if __name__ == '__main__':
	main()
