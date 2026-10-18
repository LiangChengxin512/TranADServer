"""Render a reviewer-facing replacement for the Fig. 2 SMD prediction plot.

The two bottom ribbons deliberately use separate vertical lanes: the top lane
is the ground truth and the lower lane is TranAD_E's binary prediction.  This
avoids the visual ambiguity of the previous stacked image representation.
"""

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
from matplotlib.patches import Patch, Rectangle

from src.device import get_best_device, get_dtype_for_device
from src.experiment_utils import (
	build_model,
	evaluate_loss_arrays,
	load_checkpoint,
	load_processed_dataset,
	predict_experiment_model,
	prepare_experiment_data,
	write_json,
)


DEFAULT_CHECKPOINT = (
	'/Users/Liang/PycharmProjects/TranAD/experiment_checkpoints/multiseed_adaptive_k/seed_5/TranAD_E_SMD/best.ckpt'
)
DEFAULT_OUTPUT_DIR = '/Users/Liang/PycharmProjects/TranAD/experiment_checkpoints/multiseed_adaptive_k/seed_5/TranAD_E_SMD/'


def parse_args():
	parser = argparse.ArgumentParser(
		description='Render color and black-and-white Fig. 2 replacements with separate anomaly ribbons.'
	)
	parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
	parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
	parser.add_argument('--batch-size', type=int, default=128)
	parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
	parser.add_argument('--start', type=int, default=None)
	parser.add_argument('--length', type=int, default=400)
	parser.add_argument('--max-dims', type=int, default=4)
	parser.add_argument('--seed', type=int, default=5)
	parser.add_argument(
		'--score-topk', type=int, default=2,
		help='Fixed k selected by the original seed-5 adaptive-k evaluation (default: 2).',
	)
	return parser.parse_args()


def binary_regions(mask):
	mask = np.asarray(mask, dtype=bool)
	edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
	return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def shade_regions(ax, x, mask, color, alpha, label=None, edge_only=False):
	for index, (start, end) in enumerate(binary_regions(mask)):
		ax.axvspan(
			x[start], x[end - 1] + 1,
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
	anchor = int(anomalies[0]) if anomalies.size else int(np.nanargmax(scores))
	return max(0, min(anchor - length // 4, n_points - length)), length


def choose_dims(labels, losses, start, length, max_dims):
	label_counts = np.sum(labels[start:start + length], axis=0)
	loss_scores = np.mean(losses[start:start + length], axis=0)
	scale = float(np.max(loss_scores)) + 1.0 if loss_scores.size else 1.0
	rank = loss_scores + scale * (label_counts > 0)
	return np.argsort(rank)[::-1][:max(1, min(max_dims, losses.shape[1]))]


def draw_ribbon(ax, x, ground_truth, prediction, monochrome):
	"""Draw independent, explicitly separated ground-truth and prediction lanes."""
	if monochrome:
		normal, truth_color, prediction_color, border = '#ffffff', '#111111', '#111111', '#111111'
	else:
		normal, truth_color, prediction_color, border = '#f3f4f6', '#f59e0b', '#dc2626', '#9ca3af'

	left, right = x[0], x[-1] + 1
	lanes = ((ground_truth, 1.15, 1.85, truth_color), (prediction, 0.15, 0.85, prediction_color))
	for mask, bottom, top, anomaly_color in lanes:
		ax.imshow(
			mask.astype(int)[None, :],
			aspect='auto',
			interpolation='nearest',
			extent=(left, right, bottom, top),
			cmap=ListedColormap([normal, anomaly_color]),
			vmin=0,
			vmax=1,
		)
		ax.add_patch(Rectangle((left, bottom), right - left, top - bottom,
			fill=False, edgecolor=border, linewidth=0.8))

	ax.set_facecolor('white')
	ax.set_ylim(0, 2)
	ax.set_yticks([1.5, 0.5])
	ax.set_yticklabels(['Ground truth', 'Prediction'])
	ax.set_xlabel('Test time step')
	ax.set_title('Binary anomaly regions: separate ground-truth and prediction ribbons', loc='left', fontsize=10)
	if monochrome:
		legend = [Patch(facecolor='#ffffff', edgecolor='#111111', label='Normal'),
			Patch(facecolor='#111111', edgecolor='#111111', label='Anomaly')]
	else:
		legend = [Patch(facecolor=normal, edgecolor=border, label='Normal'),
			Patch(facecolor=truth_color, label='Ground-truth anomaly'),
			Patch(facecolor=prediction_color, label='Predicted anomaly')]
	ax.legend(handles=legend, loc='upper right', ncol=len(legend), fontsize=7, frameon=True)


def render(path, raw_test, predictions, labels, final_pred, scores, threshold, start, length, dims, monochrome):
	end = start + length
	x = np.arange(start, end)
	labels_final = np.sum(labels[start:end], axis=1) >= 1
	pred_final = np.asarray(final_pred[start:end], dtype=bool)
	n_rows = len(dims) + 2
	fig, axes = plt.subplots(
		n_rows, 1, figsize=(13, 2.0 * n_rows), sharex=True,
		gridspec_kw={'height_ratios': [1.0] * (n_rows - 1) + [0.8]},
	)
	for row, dim in enumerate(dims):
		ax = axes[row]
		if monochrome:
			input_color, reconstruction_color = '#111111', '#666666'
		else:
			input_color, reconstruction_color = '#1f2937', '#2563eb'
		ax.plot(x, raw_test[start:end, dim], color=input_color, linewidth=1.1, label=f'Input feature {dim}')
		ax.plot(x, predictions[start:end, dim], color=reconstruction_color, linewidth=0.9,
			alpha=0.9, label='Reconstruction')
		shade_regions(ax, x, pred_final, '#555555' if monochrome else '#dc2626', 0.20, 'Predicted anomaly')
		shade_regions(ax, x, labels_final, '#111111' if monochrome else '#f59e0b', 0.85, 'Ground truth', edge_only=True)
		ax.set_ylabel(f'Feature {dim}')
		ax.grid(axis='y', color='#d1d5db', linewidth=0.5, alpha=0.7)
		if row == 0:
			ax.legend(loc='upper right', ncol=2, fontsize=8)

	score_ax = axes[-2]
	score_ax.plot(x, scores[start:end], color='#111111', linewidth=1.1, label='Aggregated anomaly score')
	score_ax.axhline(threshold, color='#555555' if monochrome else '#dc2626', linestyle='--', linewidth=1.0,
		label='POT threshold')
	shade_regions(score_ax, x, pred_final, '#555555' if monochrome else '#dc2626', 0.20)
	shade_regions(score_ax, x, labels_final, '#111111' if monochrome else '#f59e0b', 0.85, edge_only=True)
	score_ax.set_ylabel('Score')
	score_ax.legend(loc='upper right', fontsize=8)
	score_ax.grid(axis='y', color='#d1d5db', linewidth=0.5, alpha=0.7)

	draw_ribbon(axes[-1], x, labels_final, pred_final, monochrome)
	fig.suptitle('Anomaly prediction visualization: TranAD_E on SMD', fontsize=14)
	fig.tight_layout()
	fig.savefig(path, dpi=220, bbox_inches='tight')
	plt.close(fig)


def main():
	args = parse_args()
	if not os.path.isfile(args.checkpoint):
		raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	device = get_best_device(args.device, 'TranAD_E')
	dtype = get_dtype_for_device(device)
	train_np, test_np, labels = load_processed_dataset('SMD')
	model = build_model('TranAD_E', labels.shape[1], device, dtype, batch_size=args.batch_size)
	load_checkpoint(args.checkpoint, model, device=device, dtype=dtype)
	train_data, test_data, _, _ = prepare_experiment_data(model, train_np, test_np, device, dtype)
	train_loss, _ = predict_experiment_model(model, train_data, args.batch_size)
	test_loss, predictions = predict_experiment_model(model, test_data, args.batch_size)
	metrics, _, final_pred, _, scores = evaluate_loss_arrays(
		train_loss, test_loss, labels, show_progress=True, score_agg='topk', topk=args.score_topk
	)
	start, length = choose_region(np.sum(labels, axis=1) >= 1, scores, args.start, args.length)
	dims = choose_dims(labels, test_loss, start, length, args.max_dims)
	os.makedirs(args.output_dir, exist_ok=True)
	color_path = os.path.join(args.output_dir, 'figure2_anomaly_prediction_separate_ribbons_color.png')
	mono_path = os.path.join(args.output_dir, 'figure2_anomaly_prediction_separate_ribbons_bw.png')
	common = (test_np, predictions, labels, final_pred, scores, metrics['threshold'], start, length, dims)
	render(color_path, *common, monochrome=False)
	render(mono_path, *common, monochrome=True)
	metrics_path = os.path.join(args.output_dir, 'figure2_separate_ribbons_metrics.json')
	write_json(metrics_path, {
		'checkpoint': args.checkpoint,
		'device': str(device),
		'dtype': str(dtype),
		'start': int(start),
		'length': int(length),
		'dimensions': [int(dim) for dim in dims],
		'color_figure': color_path,
		'black_and_white_figure': mono_path,
		'adaptive_k_source': 'seed-5 SMD test_metrics.json',
		**metrics,
	})
	print({'color_figure': color_path, 'black_and_white_figure': mono_path, **metrics})


if __name__ == '__main__':
	main()
