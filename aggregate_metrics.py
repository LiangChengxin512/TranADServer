#!/usr/bin/env python3
"""Summarize multi-seed M0--M4 metrics as paper-ready Mean ± Std cells."""

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
METRICS = ('precision', 'recall', 'f1', 'roc_auc')


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--input', default='experiments_results/ablation_raw_metrics.csv')
	parser.add_argument('--output', default='experiments_results/ablation_summary_mean_std.csv')
	parser.add_argument('--digits', type=int, default=4)
	return parser.parse_args()


def resolve(path):
	path = Path(path)
	return path if path.is_absolute() else PROJECT_ROOT / path


def main():
	args = parse_args()
	input_path = resolve(args.input)
	output_path = resolve(args.output)
	if not input_path.is_file():
		raise SystemExit(f'Raw metrics file not found: {input_path}')
	rows = pd.read_csv(input_path)
	required = {'dataset', 'model', *METRICS}
	missing = required.difference(rows.columns)
	if missing:
		raise SystemExit(f'Metrics CSV is missing columns: {", ".join(sorted(missing))}')
	for metric in METRICS:
		rows[metric] = pd.to_numeric(rows[metric], errors='coerce')
	grouped = rows.groupby(['dataset', 'model'], sort=True, dropna=False)
	if 'seed' in rows.columns:
		summary = grouped.agg(seed_count=('seed', 'count'))
	else:
		summary = grouped.size().to_frame('seed_count')
	for metric in METRICS:
		means = grouped[metric].mean()
		stds = grouped[metric].std(ddof=1).fillna(0.0)
		summary[metric] = [
			f'{mean:.{args.digits}f} ± {std:.{args.digits}f}' if pd.notna(mean) else '--'
			for mean, std in zip(means, stds)
		]
	summary = summary.reset_index()[['dataset', 'model', 'seed_count', *METRICS]]
	output_path.parent.mkdir(parents=True, exist_ok=True)
	summary.to_csv(output_path, index=False)
	print(f'Wrote summary for {len(summary)} dataset/model groups: {output_path}')


if __name__ == '__main__':
	main()
