import argparse
import json
import os

import pandas as pd


DEFAULT_DATASETS = ['NAB', 'UCR', 'MBA', 'SMD', 'SMAP', 'MSL', 'SWaT', 'synthetic']
METRICS = ['precision', 'recall', 'f1', 'roc_auc']


def parse_args():
	parser = argparse.ArgumentParser(description='Collect TranAD/TranAD_E ablation metrics into CSV files.')
	parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
	parser.add_argument('--existing-log-dir', type=str, default='experiment_logs')
	parser.add_argument('--ablation-log-dir', type=str, default='experiment_logs/ablation_tranad_e_topk')
	parser.add_argument('--output-dir', type=str, default='experiment_tables/tranad_e_ablation')
	return parser.parse_args()


def metric_path(root, model, dataset):
	return os.path.join(root, f'{model}_{dataset}', 'test_metrics.json')


def read_metrics(path):
	if not os.path.exists(path):
		return None
	with open(path, 'r', encoding='utf-8') as f:
		return json.load(f)


def make_rows(args):
	variants = [
		{
			'variant': 'TranAD',
			'model': 'TranAD',
			'log_root': args.existing_log_dir,
			'score_setting': 'mean',
			'experiment_source': 'existing_baseline',
		},
		{
			'variant': 'TranAD+topk',
			'model': 'TranAD',
			'log_root': os.path.join(args.ablation_log_dir, 'tranad_topk'),
			'score_setting': 'topk@3',
			'experiment_source': 'new_ablation',
		},
		{
			'variant': 'TranAD_E',
			'model': 'TranAD_E',
			'log_root': os.path.join(args.ablation_log_dir, 'tranad_e_mean'),
			'score_setting': 'mean',
			'experiment_source': 'new_ablation',
		},
		{
			'variant': 'TranAD_E+topk',
			'model': 'TranAD_E',
			'log_root': args.existing_log_dir,
			'score_setting': 'topk@3',
			'experiment_source': 'existing_final',
		},
	]
	rows = []
	for dataset in args.datasets:
		for variant in variants:
			path = metric_path(variant['log_root'], variant['model'], dataset)
			payload = read_metrics(path)
			row = {
				'dataset': dataset,
				'variant': variant['variant'],
				'model': variant['model'],
				'score_setting': variant['score_setting'],
				'experiment_source': variant['experiment_source'],
				'metrics_path': path,
				'status': 'ok' if payload is not None else 'missing',
			}
			for metric in METRICS:
				row[metric] = payload.get(metric) if payload is not None else None
			row['threshold'] = payload.get('threshold') if payload is not None else None
			row['checkpoint'] = payload.get('checkpoint') if payload is not None else None
			row['device'] = payload.get('device') if payload is not None else None
			row['dtype'] = payload.get('dtype') if payload is not None else None
			row['recorded_score_agg'] = payload.get('score_agg') if payload is not None else None
			rows.append(row)
	return rows


def write_wide_tables(frame, output_dir):
	for metric in METRICS:
		wide = frame.pivot(index='dataset', columns='variant', values=metric)
		wide.to_csv(os.path.join(output_dir, f'ablation_{metric}.csv'))


def main():
	args = parse_args()
	os.makedirs(args.output_dir, exist_ok=True)
	frame = pd.DataFrame(make_rows(args))
	long_path = os.path.join(args.output_dir, 'ablation_metrics_long.csv')
	frame.to_csv(long_path, index=False)
	write_wide_tables(frame, args.output_dir)
	summary = {
		'rows': int(len(frame)),
		'missing_rows': int((frame['status'] == 'missing').sum()),
		'output_dir': args.output_dir,
		'long_csv': long_path,
	}
	print(summary)


if __name__ == '__main__':
	main()
