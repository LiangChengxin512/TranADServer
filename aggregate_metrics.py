#!/usr/bin/env python3
"""Summarize multi-seed M0--M4 metrics as paper-ready Mean ± Std cells."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PREFIX = {
	'SMD': 'machine-1-1_',
	'SMAP': 'P-1_',
	'MSL': 'C-1_',
	'UCR': '136_',
	'NAB': 'ec2_request_latency_system_failure_',
}
METRICS = {
	'Precision': 'precision',
	'Recall': 'recall',
	'F1': 'f1',
	'ROC-AUC': 'roc_auc',
}


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--input', default='experiments_results/ablation_raw_metrics.csv')
	parser.add_argument('--output', default='experiments_results/ablation_summary_mean_std.csv')
	parser.add_argument('--digits', type=int, default=4)
	return parser.parse_args()


def resolve(path):
	path = Path(path)
	return path if path.is_absolute() else PROJECT_ROOT / path


def dataset_has_usable_labels(dataset):
	"""Require complete processed inputs and at least one positive label."""
	dataset = str(dataset)
	prefix = DATASET_PREFIX.get(dataset, '')
	folder = PROJECT_ROOT / 'processed' / dataset
	paths = [folder / f'{prefix}{split}.npy' for split in ('train', 'test', 'labels')]
	if not all(path.is_file() for path in paths):
		return False
	try:
		train = np.load(paths[0], mmap_mode='r')
		test = np.load(paths[1], mmap_mode='r')
		labels = np.load(paths[2], mmap_mode='r')
		if labels.ndim == 1:
			labels = labels[:, None]
		if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1] or labels.shape != test.shape:
			return False
		labels = np.asarray(labels)
		return bool(np.any(np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0) > 0.1))
	except (OSError, ValueError, TypeError):
		return False


def main():
	args = parse_args()
	input_path = resolve(args.input)
	output_path = resolve(args.output)
	if not input_path.is_file():
		raise SystemExit(f'Raw metrics file not found: {input_path}')
	rows = pd.read_csv(input_path)
	required = {'dataset', 'model', *METRICS.values()}
	missing = required.difference(rows.columns)
	if missing:
		raise SystemExit(f'Metrics CSV is missing columns: {", ".join(sorted(missing))}')
	for metric in METRICS.values():
		rows[metric] = pd.to_numeric(rows[metric], errors='coerce')
	finite_metrics = np.isfinite(rows[list(METRICS.values())].to_numpy(dtype=np.float64)).all(axis=1)
	rows = rows.loc[finite_metrics].copy()
	if 'dataset' in rows.columns:
		usable = {
			dataset: dataset_has_usable_labels(dataset)
			for dataset in rows['dataset'].dropna().astype(str).unique()
		}
		rows = rows.loc[rows['dataset'].astype(str).map(usable).fillna(False)].copy()
	if rows.empty:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		pd.DataFrame(columns=['dataset', 'model', 'seed_count', *METRICS]).to_csv(output_path, index=False)
		print(f'Wrote summary for 0 dataset/model groups: {output_path}')
		return
	if 'seed' in rows.columns:
		# Re-running a seed replaces its prior contribution in the summary.
		rows = rows.drop_duplicates(subset=['dataset', 'model', 'seed'], keep='last')
	grouped = rows.groupby(['dataset', 'model'], sort=True, dropna=False)
	if 'seed' in rows.columns:
		summary = grouped.agg(seed_count=('seed', 'count'))
	else:
		summary = grouped.size().to_frame('seed_count')
	for output_name, metric in METRICS.items():
		means = grouped[metric].mean()
		stds = grouped[metric].std(ddof=1).fillna(0.0)
		summary[output_name] = [f'{mean:.{args.digits}f} ± {std:.{args.digits}f}' for mean, std in zip(means, stds)]
	summary = summary.reset_index()[['dataset', 'model', 'seed_count', *METRICS]]
	output_path.parent.mkdir(parents=True, exist_ok=True)
	summary.to_csv(output_path, index=False)
	print(f'Wrote summary for {len(summary)} dataset/model groups: {output_path}')


if __name__ == '__main__':
	main()
