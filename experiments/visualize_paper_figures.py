import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from src.device import get_best_device, get_dtype_for_device


def import_experiment_utils(args):
	original_argv = sys.argv[:]
	sys.argv = [original_argv[0], '--dataset', args.dataset, '--model', args.model]
	try:
		from src.experiment_utils import (
			build_model,
			checkpoint_path,
			evaluate_loss_arrays,
			is_tranad_model,
			load_checkpoint,
			load_processed_dataset,
			predict_experiment_model,
			prepare_experiment_data,
			write_json,
		)
	finally:
		sys.argv = original_argv
	return {
		'build_model': build_model,
		'checkpoint_path': checkpoint_path,
		'evaluate_loss_arrays': evaluate_loss_arrays,
		'is_tranad_model': is_tranad_model,
		'load_checkpoint': load_checkpoint,
		'load_processed_dataset': load_processed_dataset,
		'predict_experiment_model': predict_experiment_model,
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
	parser = argparse.ArgumentParser(
		description='Render paper-style anomaly prediction and focus/attention figures from a checkpoint.'
	)
	parser.add_argument('--dataset', type=str, default='synthetic')
	parser.add_argument('--model', type=str, default='TranAD')
	parser.add_argument('--batch-size', type=int, default=128)
	parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
	parser.add_argument('--checkpoint-dir', type=str, default='experiment_checkpoints')
	parser.add_argument('--checkpoint', type=str, default=None)
	parser.add_argument('--output-dir', type=str, default='experiment_visualizations')
	parser.add_argument('--less', action='store_true')
	parser.add_argument('--score-agg', type=str, default='mean', choices=['mean', 'max', 'p95', 'topk'])
	parser.add_argument('--score-topk', type=parse_score_topk, default=3)
	parser.add_argument('--start', type=int, default=None)
	parser.add_argument('--length', type=int, default=400)
	parser.add_argument('--max-dims', type=int, default=4)
	parser.add_argument('--seed', type=int, default=1)
	return parser.parse_args()


def binary_regions(mask):
	mask = np.asarray(mask).astype(bool)
	if mask.size == 0:
		return []
	edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
	starts = np.where(edges == 1)[0]
	ends = np.where(edges == -1)[0]
	return list(zip(starts, ends))


def shade_regions(ax, x, mask, color, alpha, label=None, edge_only=False):
	for index, (start, end) in enumerate(binary_regions(mask)):
		left = x[start]
		right = x[end - 1] + 1
		ax.axvspan(
			left,
			right,
			facecolor='none' if edge_only else color,
			edgecolor=color if edge_only else 'none',
			alpha=alpha,
			linewidth=1.0,
			label=label if index == 0 else None,
		)


def choose_region(labels, scores, requested_start, length):
	n_points = len(scores)
	length = n_points if length <= 0 else min(length, n_points)
	if requested_start is not None:
		return max(0, min(requested_start, n_points - length)), length
	anomalies = np.flatnonzero(labels)
	if anomalies.size:
		anchor = int(anomalies[0])
	elif n_points:
		anchor = int(np.nanargmax(scores))
	else:
		anchor = 0
	start = anchor - length // 4
	return max(0, min(start, n_points - length)), length


def choose_dims(labels, losses, start, length, max_dims):
	end = start + length
	region_labels = np.asarray(labels[start:end])
	region_losses = np.asarray(losses[start:end])
	label_counts = np.sum(region_labels, axis=0)
	loss_scores = np.mean(region_losses, axis=0)
	scale = float(np.max(loss_scores)) + 1.0 if loss_scores.size else 1.0
	rank = loss_scores + scale * (label_counts > 0)
	count = max(1, min(max_dims, losses.shape[1]))
	return np.argsort(rank)[::-1][:count]


def normalized(values):
	values = np.nan_to_num(np.asarray(values, dtype=np.float64))
	if values.size == 0:
		return values
	low = np.min(values)
	high = np.max(values)
	if high - low < 1e-12:
		return np.zeros_like(values)
	return (values - low) / (high - low)


def render_anomaly_prediction(path, dataset, model_name, raw_test, predictions, losses, labels, final_pred, scores, threshold, start, length, dims):
	end = start + length
	x = np.arange(start, end)
	labels_final = np.sum(labels[start:end], axis=1) >= 1
	pred_final = np.asarray(final_pred[start:end]).astype(bool)
	n_rows = len(dims) + 2
	fig, axes = plt.subplots(n_rows, 1, figsize=(13, 2.0 * n_rows), sharex=True)
	for row, dim in enumerate(dims):
		ax = axes[row]
		ax.plot(x, raw_test[start:end, dim], color='#1f2937', linewidth=1.1, label=f'Input feature {dim}')
		ax.plot(
			x,
			predictions[start:end, dim],
			color='#2563eb',
			linewidth=0.9,
			alpha=0.9,
			label='Reconstruction',
		)
		shade_regions(ax, x, pred_final, '#dc2626', 0.20, label='Predicted anomaly')
		shade_regions(ax, x, labels_final, '#f59e0b', 0.85, label='Ground truth', edge_only=True)
		ax.set_ylabel(f'Feature {dim}')
		ax.grid(axis='y', color='#d1d5db', linewidth=0.5, alpha=0.7)
		if row == 0:
			ax.legend(loc='upper right', ncol=2, fontsize=8)
	score_ax = axes[-2]
	score_ax.plot(x, scores[start:end], color='#111827', linewidth=1.1, label='Aggregated anomaly score')
	if np.isfinite(threshold):
		score_ax.axhline(threshold, color='#dc2626', linestyle='--', linewidth=1.0, label='POT threshold')
	shade_regions(score_ax, x, pred_final, '#dc2626', 0.20)
	shade_regions(score_ax, x, labels_final, '#f59e0b', 0.85, edge_only=True)
	score_ax.set_ylabel('Score')
	score_ax.legend(loc='upper right', fontsize=8)
	score_ax.grid(axis='y', color='#d1d5db', linewidth=0.5, alpha=0.7)
	strip_ax = axes[-1]
	strip = np.vstack([labels_final.astype(int), pred_final.astype(int)])
	strip_ax.imshow(
		strip,
		aspect='auto',
		interpolation='nearest',
		extent=(start, end, 0, 2),
		cmap=ListedColormap(['#f3f4f6', '#dc2626']),
		vmin=0,
		vmax=1,
	)
	strip_ax.set_yticks([1.5, 0.5])
	strip_ax.set_yticklabels(['Ground truth', 'Prediction'])
	strip_ax.set_xlabel('Test time step')
	strip_ax.set_title('Binary anomaly regions', loc='left', fontsize=10)
	fig.suptitle(f'Anomaly prediction visualization: {model_name} on {dataset}', fontsize=14)
	fig.tight_layout()
	fig.savefig(path, dpi=220, bbox_inches='tight')
	plt.close(fig)


def capture_attention_matrix(model, test_data, sample_index, is_tranad):
	for module in model.modules():
		if hasattr(module, 'capture_attention'):
			module.capture_attention = True
			if hasattr(module, 'last_attn_weights'):
				module.last_attn_weights = None
	if not is_tranad:
		return None
	window = test_data[sample_index:sample_index + 1]
	window = window.permute(1, 0, 2)
	elem = window[-1, :, :].view(1, 1, window.shape[-1])
	with torch.no_grad():
		model(window, elem)
	candidates = []
	for module in model.modules():
		weights = getattr(module, 'last_attn_weights', None)
		if weights is None:
			continue
		matrix = weights.detach().cpu().numpy()
		while matrix.ndim > 2:
			matrix = matrix.mean(axis=0)
		if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] and matrix.shape[0] > 1:
			candidates.append(matrix)
	if not candidates:
		return None
	return max(candidates, key=lambda matrix: matrix.shape[0])


def render_focus_attention(path, dataset, model_name, attention, losses, labels, final_pred, scores, start, length, dims):
	end = start + length
	x = np.arange(start, end)
	label_slice = np.sum(labels[start:end], axis=1) >= 1
	pred_slice = np.asarray(final_pred[start:end]).astype(bool)
	feature_focus = normalized(np.mean(losses[start:end], axis=0))
	top_features = np.argsort(feature_focus)[::-1][:min(20, losses.shape[1])]
	time_focus = normalized(scores[start:end])
	fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={'width_ratios': [1.0, 1.0, 1.45]})
	if attention is not None:
		image = axes[0].imshow(attention, aspect='auto', origin='lower', cmap='magma')
		axes[0].set_title('Temporal self-attention')
		axes[0].set_xlabel('Key time in window')
		axes[0].set_ylabel('Query time in window')
	else:
		proxy = losses[start:end, dims].T
		image = axes[0].imshow(proxy, aspect='auto', origin='lower', cmap='magma')
		axes[0].set_title('Loss contribution heatmap')
		axes[0].set_xlabel('Selected time range')
		axes[0].set_ylabel('Feature')
		axes[0].set_yticks(np.arange(len(dims)))
		axes[0].set_yticklabels([str(dim) for dim in dims])
	fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
	axes[1].barh(np.arange(len(top_features)), feature_focus[top_features], color='#0f766e')
	axes[1].invert_yaxis()
	axes[1].set_yticks(np.arange(len(top_features)))
	axes[1].set_yticklabels([str(feature) for feature in top_features], fontsize=8)
	axes[1].set_xlim(0, 1.0)
	axes[1].set_xlabel('Normalized focus')
	axes[1].set_ylabel('Feature')
	axes[1].set_title('Focus over features')
	axes[1].grid(axis='x', color='#d1d5db', linewidth=0.5, alpha=0.7)
	time_ax = axes[2]
	time_ax.plot(x, time_focus, color='#7c3aed', linewidth=1.2)
	shade_regions(time_ax, x, pred_slice, '#dc2626', 0.20, label='Predicted anomaly')
	shade_regions(time_ax, x, label_slice, '#f59e0b', 0.85, label='Ground truth', edge_only=True)
	time_ax.set_ylim(-0.02, 1.05)
	time_ax.set_xlabel('Test time step')
	time_ax.set_ylabel('Normalized focus')
	time_ax.set_title('Focus over time')
	time_ax.grid(axis='y', color='#d1d5db', linewidth=0.5, alpha=0.7)
	time_ax.legend(loc='upper right', fontsize=8)
	fig.suptitle(f'Focus and attention visualization: {model_name} on {dataset}', fontsize=14)
	fig.tight_layout()
	fig.savefig(path, dpi=220, bbox_inches='tight')
	plt.close(fig)


def main():
	args = parse_args()
	utils = import_experiment_utils(args)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	device = get_best_device(args.device, args.model)
	dtype = get_dtype_for_device(device)
	train_np, test_np, labels = utils['load_processed_dataset'](args.dataset, less=args.less)
	model = utils['build_model'](args.model, labels.shape[1], device, dtype, batch_size=args.batch_size)
	checkpoint = args.checkpoint or utils['checkpoint_path'](
		args.checkpoint_dir, args.model, args.dataset, 'best.ckpt'
	)
	if not os.path.exists(checkpoint):
		checkpoint = utils['checkpoint_path'](args.checkpoint_dir, args.model, args.dataset, 'latest.ckpt')
	if not os.path.exists(checkpoint):
		raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
	utils['load_checkpoint'](checkpoint, model, device=device, dtype=dtype)
	train_data, test_data, _, _ = utils['prepare_experiment_data'](model, train_np, test_np, device, dtype)
	train_loss, _ = utils['predict_experiment_model'](model, train_data, args.batch_size)
	test_loss, y_pred = utils['predict_experiment_model'](model, test_data, args.batch_size)
	metrics, _, final_pred, _, test_scores = utils['evaluate_loss_arrays'](
		train_loss,
		test_loss,
		labels,
		show_progress=True,
		score_agg=args.score_agg,
		topk=args.score_topk,
	)
	labels_final = np.sum(labels, axis=1) >= 1
	start, length = choose_region(labels_final, test_scores, args.start, args.length)
	dims = choose_dims(labels, test_loss, start, length, args.max_dims)
	focus_index = start + int(np.argmax(test_scores[start:start + length]))
	attention = capture_attention_matrix(
		model, test_data, focus_index, utils['is_tranad_model'](model)
	)
	out_folder = os.path.join(args.output_dir, f'{args.model}_{args.dataset}')
	os.makedirs(out_folder, exist_ok=True)
	figure2_path = os.path.join(out_folder, 'figure2_anomaly_prediction.png')
	figure3_path = os.path.join(out_folder, 'figure3_focus_attention.png')
	render_anomaly_prediction(
		figure2_path,
		args.dataset,
		args.model,
		test_np,
		y_pred,
		test_loss,
		labels,
		final_pred,
		test_scores,
		metrics['threshold'],
		start,
		length,
		dims,
	)
	render_focus_attention(
		figure3_path,
		args.dataset,
		args.model,
		attention,
		test_loss,
		labels,
		final_pred,
		test_scores,
		start,
		length,
		dims,
	)
	utils['write_json'](os.path.join(out_folder, 'visualization_metrics.json'), {
		'model': args.model,
		'dataset': args.dataset,
		'checkpoint': checkpoint,
		'device': str(device),
		'dtype': str(dtype),
		'start': start,
		'length': length,
		'dimensions': [int(dim) for dim in dims],
		'focus_index': focus_index,
		'attention_available': attention is not None,
		**metrics,
	})
	print({'figure2': figure2_path, 'figure3': figure3_path, **metrics})


if __name__ == '__main__':
	main()
