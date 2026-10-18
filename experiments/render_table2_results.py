import argparse
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DATASETS = ['NAB', 'UCR', 'MBA', 'SMAP', 'MSL', 'SWaT', 'WADI', 'SMD', 'MSDS']
DEFAULT_MODELS = [
	'DAGMM',
	'LSTM_AD',
	'MSCRED',
	'MAD_GAN',
	'OmniAnomaly',
	'MTAD_GAT',
	'USAD',
	'GDN',
	'CAE_M',
	'MERLIN',
	'TranAD',
]
METRICS = [
	('f1', 'F1'),
	('precision', 'Precision'),
	('recall', 'Recall'),
	('roc_auc', 'AUC'),
]
MODEL_LABELS = {
	'LSTM_AD': 'LSTM-NDT',
	'MAD_GAN': 'MAD-GAN',
	'MTAD_GAT': 'MTAD-GAT',
	'CAE_M': 'CAE-M',
}


def parse_args():
	parser = argparse.ArgumentParser(description='Render Table 2-style metric tables from experiment test logs.')
	parser.add_argument('--log-dir', type=str, default='experiment_logs')
	parser.add_argument('--output-dir', type=str, default='experiment_tables/table2')
	parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
	parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS)
	parser.add_argument('--digits', type=int, default=4)
	parser.add_argument('--strict', action='store_true')
	return parser.parse_args()


def metric_path(log_dir, model, dataset):
	return os.path.join(log_dir, f'{model}_{dataset}', 'test_metrics.json')


def load_rows(args):
	rows = []
	missing = []
	for dataset in args.datasets:
		for model in args.models:
			path = metric_path(args.log_dir, model, dataset)
			row = {'dataset': dataset, 'model': model, 'path': path}
			if os.path.exists(path):
				with open(path, 'r', encoding='utf-8') as f:
					payload = json.load(f)
				for key, _ in METRICS:
					row[key] = payload.get(key, np.nan)
			else:
				missing.append(path)
				for key, _ in METRICS:
					row[key] = np.nan
			rows.append(row)
	if missing and args.strict:
		raise FileNotFoundError(f'Missing {len(missing)} test metric files. First missing file: {missing[0]}')
	return pd.DataFrame(rows), missing


def metric_frame(rows, datasets, models, metric):
	frame = rows.pivot(index='dataset', columns='model', values=metric)
	return frame.reindex(index=datasets, columns=models)


def display_models(models):
	return [MODEL_LABELS.get(model, model) for model in models]


def format_cell(value, digits):
	return '--' if pd.isna(value) else f'{float(value):.{digits}f}'


def cell_colors(frame):
	colors = []
	for _, row in frame.iterrows():
		valid = row.dropna()
		row_colors = []
		low = float(valid.min()) if not valid.empty else 0.0
		high = float(valid.max()) if not valid.empty else 0.0
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


def render_table(frame, metric_label, path, digits):
	text = [[format_cell(value, digits) for value in row] for row in frame.to_numpy()]
	fig_width = max(8.0, 1.02 * (frame.shape[1] + 2))
	fig_height = max(3.4, 0.44 * (frame.shape[0] + 3))
	fig, ax = plt.subplots(figsize=(fig_width, fig_height))
	ax.axis('off')
	table = ax.table(
		cellText=text,
		cellColours=cell_colors(frame),
		colLabels=display_models(list(frame.columns)),
		rowLabels=list(frame.index),
		cellLoc='center',
		rowLoc='center',
		loc='center',
	)
	table.auto_set_font_size(False)
	table.set_fontsize(7.5)
	table.scale(1.0, 1.32)
	for (row, col), cell in table.get_celld().items():
		cell.set_edgecolor('#9ca3af')
		cell.set_linewidth(0.5)
		if row == 0 or col == -1:
			cell.set_facecolor('#e5e7eb')
			cell.set_text_props(weight='bold', color='#111827')
	ax.set_title(f'Table 2-style comparison: {metric_label}', fontsize=13, pad=16)
	fig.savefig(path, dpi=240, bbox_inches='tight')
	plt.close(fig)


def write_latex_tables(frames, path, digits):
	with open(path, 'w', encoding='utf-8') as f:
		for metric_key, metric_label in METRICS:
			frame = frames[metric_key].copy()
			frame.columns = display_models(list(frame.columns))
			f.write(f'% {metric_label}\n')
			f.write(frame.to_latex(float_format=lambda value: f'{value:.{digits}f}', na_rep='--'))
			f.write('\n')


def main():
	args = parse_args()
	os.makedirs(args.output_dir, exist_ok=True)
	rows, missing = load_rows(args)
	long_path = os.path.join(args.output_dir, 'table2_metrics_long.csv')
	rows.to_csv(long_path, index=False)
	frames = {}
	created = [long_path]
	for metric_key, metric_label in METRICS:
		frame = metric_frame(rows, args.datasets, args.models, metric_key)
		frames[metric_key] = frame
		csv_path = os.path.join(args.output_dir, f'table2_{metric_key}.csv')
		png_path = os.path.join(args.output_dir, f'table2_{metric_key}.png')
		frame.to_csv(csv_path)
		render_table(frame, metric_label, png_path, args.digits)
		created.extend([csv_path, png_path])
	latex_path = os.path.join(args.output_dir, 'table2_tables.tex')
	write_latex_tables(frames, latex_path, args.digits)
	created.append(latex_path)
	print({
		'created': created,
		'missing_metric_files': len(missing),
		'first_missing_file': missing[0] if missing else None,
	})


if __name__ == '__main__':
	main()
