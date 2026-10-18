import argparse
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DATASETS = ['MBA', 'MSDS', 'NAB', 'MSL', 'SMAP', 'SMD', 'SWaT', 'UCR', 'WADI', 'synthetic']
DEFAULT_SEEDS = ['1', '2', '3', '4', '5']
BASE_MODELS = ['TranAD', 'TranAD_C', 'TranAD_G', 'TranAD_E', 'TranAD_C_G', 'TranAD_C_E', 'TranAD_G_E', 'TranAD_C_G_E']
METRICS = [
	('precision', 'Precision'),
	('recall', 'Recall'),
	('f1', 'F1'),
	('roc_auc', 'AUC'),
]


def variant_id(model, score_setting):
	return f'{model.lower()}_{score_setting}'


def variant_label(model, score_setting):
	return model if score_setting == 'mean' else f'{model}+adaptive-topk'


def build_variants(models, score_settings):
	return [
		{
			'model': model,
			'score_setting': score_setting,
			'experiment_id': variant_id(model, score_setting),
			'variant': variant_label(model, score_setting),
		}
		for model in models
		for score_setting in score_settings
	]


def parse_args():
	parser = argparse.ArgumentParser(
		description='Collect C/G/E plus adaptive-top-k ablation metrics and render mean+-std tables.'
	)
	parser.add_argument('--log-dir', type=str, default='experiment_logs/cge_adaptive_k_ablation')
	parser.add_argument('--output-dir', type=str, default='experiment_tables/cge_adaptive_k_ablation')
	parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
	parser.add_argument('--seeds', nargs='+', default=DEFAULT_SEEDS)
	parser.add_argument('--models', nargs='+', default=BASE_MODELS)
	parser.add_argument('--score-settings', nargs='+', choices=['mean', 'adaptive_topk'], default=['mean', 'adaptive_topk'])
	parser.add_argument('--digits', type=int, default=4)
	parser.add_argument('--strict', action='store_true')
	return parser.parse_args()


def metric_path(args, variant, seed, dataset):
	return os.path.join(
		args.log_dir,
		f'seed_{seed}',
		variant['experiment_id'],
		f'{variant["model"]}_{dataset}',
		'test_metrics.json',
	)


def load_rows(args, variants):
	rows, missing = [], []
	for seed in args.seeds:
		for dataset in args.datasets:
			for variant in variants:
				path = metric_path(args, variant, seed, dataset)
				payload = None
				if os.path.exists(path):
					with open(path, 'r', encoding='utf-8') as handle:
						payload = json.load(handle)
				else:
					missing.append(path)
				row = {
					'seed': seed,
					'dataset': dataset,
					'variant': variant['variant'],
					'model': variant['model'],
					'score_setting': variant['score_setting'],
					'experiment_id': variant['experiment_id'],
					'path': path,
					'status': 'ok' if payload is not None else 'missing',
					'score_agg': payload.get('score_agg') if payload else None,
					'score_topk': payload.get('score_topk') if payload else None,
					'selected_topk': payload.get('selected_topk') if payload else None,
				}
				for metric, _ in METRICS:
					row[metric] = payload.get(metric, np.nan) if payload else np.nan
				rows.append(row)
	if missing and args.strict:
		raise FileNotFoundError(f'Missing {len(missing)} metric files. First missing file: {missing[0]}')
	return pd.DataFrame(rows), missing


def summarize(rows, datasets, variants, digits):
	summary_rows = []
	for dataset in datasets:
		for variant in variants:
			subset = rows[(rows['dataset'] == dataset) & (rows['variant'] == variant['variant'])]
			for metric, _ in METRICS:
				values = pd.to_numeric(subset[metric], errors='coerce').dropna().to_numpy(dtype=float)
				mean = float(np.mean(values)) if values.size else np.nan
				std = float(np.std(values, ddof=1)) if values.size > 1 else (0.0 if values.size else np.nan)
				summary_rows.append({
					'dataset': dataset, 'variant': variant['variant'], 'metric': metric,
					'n': int(values.size), 'mean': mean, 'std': std,
					'formatted': f'{mean:.{digits}f}+-{std:.{digits}f}' if values.size else '--',
				})
	return pd.DataFrame(summary_rows)


def metric_frames(summary, datasets, variants, metric):
	labels = [variant['variant'] for variant in variants]
	subset = summary[summary['metric'] == metric]
	text = subset.pivot(index='dataset', columns='variant', values='formatted').reindex(index=datasets, columns=labels)
	mean = subset.pivot(index='dataset', columns='variant', values='mean').reindex(index=datasets, columns=labels)
	return text, mean


def render_table(text_frame, mean_frame, metric_label, path):
	colors = []
	for _, row in mean_frame.iterrows():
		valid = row.dropna()
		low, high = (float(valid.min()), float(valid.max())) if not valid.empty else (0.0, 0.0)
		colors.append([
			'#f3f4f6' if pd.isna(value) else '#dbeafe' if high - low < 1e-12
			else plt.cm.YlGn(0.25 + 0.60 * ((float(value) - low) / (high - low)))
			for value in row
		])
	fig, ax = plt.subplots(figsize=(max(15.0, 1.18 * (text_frame.shape[1] + 2)), max(4.0, 0.48 * (text_frame.shape[0] + 3))))
	ax.axis('off')
	table = ax.table(cellText=text_frame.fillna('--').astype(str).to_numpy().tolist(), cellColours=colors,
		colLabels=list(text_frame.columns), rowLabels=list(text_frame.index), cellLoc='center', rowLoc='center', loc='center')
	table.auto_set_font_size(False)
	table.set_fontsize(6.2)
	table.scale(1.0, 1.36)
	for (row, col), cell in table.get_celld().items():
		cell.set_edgecolor('#9ca3af')
		cell.set_linewidth(0.5)
		if row == 0 or col == -1:
			cell.set_facecolor('#e5e7eb')
			cell.set_text_props(weight='bold', color='#111827')
	ax.set_title(f'C/G/E and adaptive top-k ablation: {metric_label} mean+-std', fontsize=13, pad=16)
	fig.savefig(path, dpi=240, bbox_inches='tight')
	plt.close(fig)


def main():
	args = parse_args()
	variants = build_variants(args.models, args.score_settings)
	os.makedirs(args.output_dir, exist_ok=True)
	rows, missing = load_rows(args, variants)
	summary = summarize(rows, args.datasets, variants, args.digits)
	rows.to_csv(os.path.join(args.output_dir, 'cge_adaptive_k_metrics_long.csv'), index=False)
	summary.to_csv(os.path.join(args.output_dir, 'cge_adaptive_k_summary_long.csv'), index=False)
	frames = {}
	for metric_key, metric_label in METRICS:
		text, mean = metric_frames(summary, args.datasets, variants, metric_key)
		frames[metric_key] = text
		text.to_csv(os.path.join(args.output_dir, f'cge_adaptive_k_{metric_key}_mean_std.csv'))
		render_table(text, mean, metric_label, os.path.join(args.output_dir, f'cge_adaptive_k_{metric_key}_mean_std.png'))
	with open(os.path.join(args.output_dir, 'cge_adaptive_k_tables.tex'), 'w', encoding='utf-8') as handle:
		for metric_key, metric_label in METRICS:
			handle.write(f'% {metric_label} mean+-std\n{frames[metric_key].fillna("--").to_latex(escape=False)}\n\n')
	print({
		'variants': [variant['variant'] for variant in variants],
		'created_output_dir': args.output_dir,
		'missing_metric_files': len(missing),
		'first_missing_file': missing[0] if missing else None,
	})


if __name__ == '__main__':
	main()
