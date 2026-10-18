import argparse
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DATASETS = ['NAB', 'MBA', 'SWaT', 'synthetic', 'UCR', 'MSL', 'SMAP', 'SMD']
DEFAULT_SEEDS = ['1', '2', '3', '4', '5']
METRICS = [
	('precision', 'Precision'),
	('recall', 'Recall'),
	('f1', 'F1'),
	('roc_auc', 'AUC'),
]
VARIANTS = [
	{
		'variant': 'TranAD',
		'model': 'TranAD',
		'source': 'existing',
		'subdir': None,
		'score_setting': 'mean',
	},
	{
		'variant': 'TranAD+adaptive-topk',
		'model': 'TranAD',
		'source': 'ablation',
		'subdir': 'tranad_adaptive_topk',
		'score_setting': 'topk@auto',
	},
	{
		'variant': 'TranAD+learnable-ensemble',
		'model': 'TranAD_E',
		'source': 'ablation',
		'subdir': 'tranad_e_mean',
		'score_setting': 'mean',
	},
	{
		'variant': 'TranAD_E+adaptive-topk',
		'model': 'TranAD_E',
		'source': 'existing',
		'subdir': None,
		'score_setting': 'topk@auto',
	},
]


def parse_args():
	parser = argparse.ArgumentParser(
		description='Collect adaptive-k ablation metrics and render mean±std tables.'
	)
	parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
	parser.add_argument('--seeds', nargs='+', default=DEFAULT_SEEDS)
	parser.add_argument('--existing-log-dir', type=str, default='experiment_logs/multiseed_adaptive_k')
	parser.add_argument('--ablation-log-dir', type=str, default='experiment_logs/adaptive_k_ablation')
	parser.add_argument('--output-dir', type=str, default='experiment_tables/adaptive_k_ablation')
	parser.add_argument('--digits', type=int, default=4)
	parser.add_argument('--strict', action='store_true')
	return parser.parse_args()


def metric_path(args, variant, seed, dataset):
	if variant['source'] == 'existing':
		return os.path.join(
			args.existing_log_dir,
			f'seed_{seed}',
			f'{variant["model"]}_{dataset}',
			'test_metrics.json',
		)
	return os.path.join(
		args.ablation_log_dir,
		f'seed_{seed}',
		variant['subdir'],
		f'{variant["model"]}_{dataset}',
		'test_metrics.json',
	)


def read_json(path):
	if not os.path.exists(path):
		return None
	with open(path, 'r', encoding='utf-8') as f:
		return json.load(f)


def load_rows(args):
	rows = []
	missing = []
	for seed in args.seeds:
		for dataset in args.datasets:
			for variant in VARIANTS:
				path = metric_path(args, variant, seed, dataset)
				payload = read_json(path)
				if payload is None:
					missing.append(path)
				row = {
					'seed': seed,
					'dataset': dataset,
					'variant': variant['variant'],
					'model': variant['model'],
					'score_setting': variant['score_setting'],
					'experiment_source': variant['source'],
					'path': path,
					'status': 'ok' if payload is not None else 'missing',
					'score_agg': payload.get('score_agg') if payload is not None else None,
					'score_topk': payload.get('score_topk') if payload is not None else None,
					'selected_topk': payload.get('selected_topk') if payload is not None else None,
					'checkpoint': payload.get('checkpoint') if payload is not None else None,
				}
				for metric, _ in METRICS:
					row[metric] = payload.get(metric, np.nan) if payload is not None else np.nan
				rows.append(row)
	if missing and args.strict:
		raise FileNotFoundError(f'Missing {len(missing)} metric files. First missing file: {missing[0]}')
	return pd.DataFrame(rows), missing


def summarize(rows, datasets, digits):
	summary_rows = []
	variants = [variant['variant'] for variant in VARIANTS]
	for dataset in datasets:
		for variant in variants:
			subset = rows[(rows['dataset'] == dataset) & (rows['variant'] == variant)]
			for metric, _ in METRICS:
				values = pd.to_numeric(subset[metric], errors='coerce').dropna().to_numpy(dtype=float)
				if values.size:
					mean = float(np.mean(values))
					std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
					formatted = f'{mean:.{digits}f}±{std:.{digits}f}'
				else:
					mean = np.nan
					std = np.nan
					formatted = '--'
				summary_rows.append({
					'dataset': dataset,
					'variant': variant,
					'metric': metric,
					'n': int(values.size),
					'mean': mean,
					'std': std,
					'formatted': formatted,
				})
	return pd.DataFrame(summary_rows)


def metric_frames(summary, datasets, metric):
	variants = [variant['variant'] for variant in VARIANTS]
	subset = summary[summary['metric'] == metric]
	text = subset.pivot(index='dataset', columns='variant', values='formatted')
	mean = subset.pivot(index='dataset', columns='variant', values='mean')
	text = text.reindex(index=datasets, columns=variants)
	mean = mean.reindex(index=datasets, columns=variants)
	return text, mean


def cell_colors(mean_frame):
	colors = []
	for _, row in mean_frame.iterrows():
		valid = row.dropna()
		low = float(valid.min()) if not valid.empty else 0.0
		high = float(valid.max()) if not valid.empty else 0.0
		row_colors = []
		for value in row:
			if pd.isna(value):
				row_colors.append('#f3f4f6')
			elif high - low < 1e-12:
				row_colors.append('#dbeafe')
			else:
				alpha = (float(value) - low) / (high - low)
				row_colors.append(plt.cm.YlGn(0.25 + 0.60 * alpha))
		colors.append(row_colors)
	return colors


def render_table(text_frame, mean_frame, metric_label, path):
	text = text_frame.fillna('--').astype(str).to_numpy().tolist()
	fig_width = max(9.0, 1.25 * (text_frame.shape[1] + 2))
	fig_height = max(3.6, 0.48 * (text_frame.shape[0] + 3))
	fig, ax = plt.subplots(figsize=(fig_width, fig_height))
	ax.axis('off')
	table = ax.table(
		cellText=text,
		cellColours=cell_colors(mean_frame),
		colLabels=list(text_frame.columns),
		rowLabels=list(text_frame.index),
		cellLoc='center',
		rowLoc='center',
		loc='center',
	)
	table.auto_set_font_size(False)
	table.set_fontsize(7.0)
	table.scale(1.0, 1.36)
	for (row, col), cell in table.get_celld().items():
		cell.set_edgecolor('#9ca3af')
		cell.set_linewidth(0.5)
		if row == 0 or col == -1:
			cell.set_facecolor('#e5e7eb')
			cell.set_text_props(weight='bold', color='#111827')
	ax.set_title(f'Adaptive top-k ablation: {metric_label} mean±std', fontsize=13, pad=16)
	fig.savefig(path, dpi=240, bbox_inches='tight')
	plt.close(fig)


def write_latex_tables(frames, path):
	with open(path, 'w', encoding='utf-8') as f:
		for metric_key, metric_label in METRICS:
			f.write(f'% {metric_label} mean±std\n')
			f.write(frames[metric_key].fillna('--').to_latex(escape=False))
			f.write('\n')


def main():
	args = parse_args()
	os.makedirs(args.output_dir, exist_ok=True)
	rows, missing = load_rows(args)
	summary = summarize(rows, args.datasets, args.digits)

	raw_path = os.path.join(args.output_dir, 'adaptive_k_ablation_metrics_long.csv')
	summary_path = os.path.join(args.output_dir, 'adaptive_k_ablation_summary_long.csv')
	rows.to_csv(raw_path, index=False)
	summary.to_csv(summary_path, index=False)
	created = [raw_path, summary_path]

	frames = {}
	for metric_key, metric_label in METRICS:
		text_frame, mean_frame = metric_frames(summary, args.datasets, metric_key)
		frames[metric_key] = text_frame
		csv_path = os.path.join(args.output_dir, f'adaptive_k_ablation_{metric_key}_mean_std.csv')
		png_path = os.path.join(args.output_dir, f'adaptive_k_ablation_{metric_key}_mean_std.png')
		text_frame.to_csv(csv_path)
		render_table(text_frame, mean_frame, metric_label, png_path)
		created.extend([csv_path, png_path])

	latex_path = os.path.join(args.output_dir, 'adaptive_k_ablation_tables.tex')
	write_latex_tables(frames, latex_path)
	created.append(latex_path)

	print({
		'created': created,
		'missing_metric_files': len(missing),
		'first_missing_file': missing[0] if missing else None,
	})


if __name__ == '__main__':
	main()
