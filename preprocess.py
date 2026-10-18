import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from src.folderconstants import *
from shutil import copyfile

datasets = ['synthetic', 'SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'MSDS', 'UCR', 'MBA', 'NAB']
wadi_drop = ['2_LS_001_AL', '2_LS_002_AL','2_P_001_STATUS','2_P_002_STATUS']


class MSDSLabelsUnavailable(ValueError):
	"""Raised when the local MSDS source has no usable anomaly annotations."""


def _is_index_or_time_column(series, name):
	name = str(name).strip().lower()
	if not name or name.startswith('unnamed:') or name in {
		'index', 'timestamp', 'time', 'date', 'datetime', 'row', 'row_id', 'id', 'step', 'sequence'
	}:
		return True
	values = series.dropna()
	if values.empty:
		return True
	if pd.api.types.is_numeric_dtype(values):
		return False
	return pd.to_datetime(values, errors='coerce').notna().mean() > 0.9


def msds_has_positive_labels(labels_path=None):
	"""Return whether the raw MSDS label CSV contains any positive annotation."""
	labels_path = labels_path or os.path.join(data_folder, 'MSDS', 'labels.csv')
	if not os.path.isfile(labels_path):
		return False
	frame = pd.read_csv(labels_path)
	if frame.empty:
		return False
	columns = [column for column in frame.columns if not _is_index_or_time_column(frame[column], column)]
	if not columns:
		return False
	values = frame[columns].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
	return bool(
		np.isfinite(values).any()
		and np.any(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0) > 0.1)
	)


def _read_msds_labels(labels_path, n_rows, n_features):
	frame = pd.read_csv(labels_path)
	columns = [column for column in frame.columns if not _is_index_or_time_column(frame[column], column)]
	if not columns:
		raise MSDSLabelsUnavailable(f'MSDS labels contain no annotation columns: {labels_path}')
	values = frame[columns].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
	if not np.any(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0) > 0.1):
		raise MSDSLabelsUnavailable(
			f'MSDS source labels contain no positive anomaly annotations: {labels_path}'
		)
	return align_msds_label_array(values, n_rows, n_features)


def align_msds_label_array(values, n_rows, n_features):
	values = np.asarray(values, dtype=np.float64)
	if values.ndim == 1:
		values = values[:, None]
	if values.ndim != 2:
		raise ValueError(f'MSDS labels must be a 1D or 2D array, got {values.shape}.')
	values = (np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0) > 0.1).astype(np.float64)
	# Keep per-channel labels when they match signal dimensions. Otherwise,
	# treat them as global event labels and broadcast their logical OR.
	if values.shape[1] == 1:
		values = np.repeat(values, n_features, axis=1)
	elif values.shape[1] != n_features:
		values = np.repeat(np.any(values > 0.1, axis=1, keepdims=True), n_features, axis=1).astype(np.float64)

	# Max-pool source intervals so short anomalies survive downsampling.
	if values.shape[0] != n_rows:
		if n_rows <= 0 or values.shape[0] < n_rows:
			raise ValueError(
				f'MSDS labels have {values.shape[0]} rows but test data has {n_rows}; '
				'cannot align labels safely.'
			)
		boundaries = np.floor(np.linspace(0, values.shape[0], n_rows + 1)).astype(np.int64)
		aligned = np.zeros((n_rows, n_features), dtype=np.float64)
		for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
			aligned[index] = np.max(values[start:max(start + 1, end)], axis=0)
		values = aligned
	return values

def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float64,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape

def load_and_save2(category, filename, dataset, dataset_folder, shape):
	temp = np.zeros(shape)
	with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
		ls = f.readlines()
	for line in ls:
		pos, values = line.split(':')[0], line.split(':')[1].split(',')
		start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i)-1 for i in values]
		temp[start-1:end-1, indx] = 1
	print(dataset, category, filename, temp.shape)
	np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)

def normalize(a):
	a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
	return (a / 2 + 0.5)

def normalize2(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = min(a), max(a)
	return (a - min_a) / (max_a - min_a), min_a, max_a

def normalize3(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = np.min(a, axis = 0), np.max(a, axis = 0)
	return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a

def convertNumpy(df):
	x = df[df.columns[3:]].values[::10, :]
	return (x - x.min(0)) / (x.ptp(0) + 1e-4)

def load_data(dataset):
	folder = os.path.join(output_folder, dataset)
	os.makedirs(folder, exist_ok=True)
	if dataset == 'synthetic':
		train_file = os.path.join(data_folder, dataset, 'synthetic_data_with_anomaly-s-1.csv')
		test_labels = os.path.join(data_folder, dataset, 'test_anomaly.csv')
		dat = pd.read_csv(train_file, header=None)
		split = 10000
		train = normalize(dat.values[:, :split].reshape(split, -1))
		test = normalize(dat.values[:, split:].reshape(split, -1))
		lab = pd.read_csv(test_labels, header=None)
		lab[0] -= split
		labels = np.zeros(test.shape)
		for i in range(lab.shape[0]):
			point = lab.values[i][0]
			labels[point-30:point+30, lab.values[i][1:]] = 1
		test += labels * np.random.normal(0.75, 0.1, test.shape)
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset == 'SMD':
		dataset_folder = 'data/SMD'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
				s = load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
				load_and_save2('labels', filename, filename.strip('.txt'), dataset_folder, s)
	elif dataset == 'UCR':
		dataset_folder = 'data/UCR'
		file_list = os.listdir(dataset_folder)
		for filename in file_list:
			if not filename.endswith('.txt'): continue
			vals = filename.split('.')[0].split('_')
			dnum, vals = int(vals[0]), vals[-3:]
			vals = [int(i) for i in vals]
			temp = np.genfromtxt(os.path.join(dataset_folder, filename),
								dtype=np.float64,
								delimiter=',')
			min_temp, max_temp = np.min(temp), np.max(temp)
			temp = (temp - min_temp) / (max_temp - min_temp)
			train, test = temp[:vals[0]], temp[vals[0]:]
			labels = np.zeros_like(test)
			labels[vals[1]-vals[0]:vals[2]-vals[0]] = 1
			train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
			for file in ['train', 'test', 'labels']:
				np.save(os.path.join(folder, f'{dnum}_{file}.npy'), eval(file))
	elif dataset == 'NAB':
		dataset_folder = 'data/NAB'
		file_list = os.listdir(dataset_folder)
		with open(dataset_folder + '/labels.json') as f:
			labeldict = json.load(f)
		for filename in file_list:
			if not filename.endswith('.csv'): continue
			df = pd.read_csv(dataset_folder+'/'+filename)
			vals = df.values[:,1]
			labels = np.zeros_like(vals, dtype=np.float64)
			for timestamp in labeldict['realKnownCause/'+filename]:
				tstamp = timestamp.replace('.000000', '')
				index = np.where(((df['timestamp'] == tstamp).values + 0) == 1)[0][0]
				labels[index-4:index+4] = 1
			min_temp, max_temp = np.min(vals), np.max(vals)
			vals = (vals - min_temp) / (max_temp - min_temp)
			train, test = vals.astype(float), vals.astype(float)
			train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
			fn = filename.replace('.csv', '')
			for file in ['train', 'test', 'labels']:
				np.save(os.path.join(folder, f'{fn}_{file}.npy'), eval(file))
	elif dataset == 'MSDS':
		dataset_folder = 'data/MSDS'
		train_frame = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
		test_frame = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))
		train_columns = [
			column for column in train_frame.columns
			if not _is_index_or_time_column(train_frame[column], column)
		]
		test_columns = [
			column for column in test_frame.columns
			if not _is_index_or_time_column(test_frame[column], column)
		]
		if train_columns != test_columns or not train_columns:
			raise ValueError(
				'MSDS train and test CSVs must contain the same non-index feature columns.'
			)
		train_source = train_frame[train_columns].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float64)
		test_source = test_frame[test_columns].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float64)
		train_source, test_source = train_source[::5], test_source[::5]
		labels = _read_msds_labels(
			os.path.join(dataset_folder, 'labels.csv'), len(test_source), test_source.shape[1]
		)
		_, min_a, max_a = normalize3(np.concatenate((train_source, test_source), axis=0))
		train, _, _ = normalize3(train_source, min_a, max_a)
		test, _, _ = normalize3(test_source, min_a, max_a)
		# Write all three arrays through temporary files only after labels have
		# passed validation; stale all-zero caches are never rewritten as truth.
		for name, values in (('train', train), ('test', test), ('labels', labels)):
			path = os.path.join(folder, f'{name}.npy')
			temporary = path + '.tmp.npy'
			np.save(temporary, np.asarray(values, dtype=np.float64))
			os.replace(temporary, path)
	elif dataset == 'SWaT':
		dataset_folder = 'data/SWaT'
		file = os.path.join(dataset_folder, 'series.json')
		df_train = pd.read_json(file, lines=True)[['val']][3000:6000]
		df_test  = pd.read_json(file, lines=True)[['val']][7000:12000]
		train, min_a, max_a = normalize2(df_train.values)
		test, _, _ = normalize2(df_test.values, min_a, max_a)
		labels = pd.read_json(file, lines=True)[['noti']][7000:12000] + 0
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset in ['SMAP', 'MSL']:
		dataset_folder = 'data/SMAP_MSL'
		file = os.path.join(dataset_folder, 'labeled_anomalies.csv')
		values = pd.read_csv(file)
		values = values[values['spacecraft'] == dataset]
		filenames = values['chan_id'].values.tolist()
		for fn in filenames:
			train = np.load(f'{dataset_folder}/train/{fn}.npy')
			test = np.load(f'{dataset_folder}/test/{fn}.npy')
			train, min_a, max_a = normalize3(train)
			test, _, _ = normalize3(test, min_a, max_a)
			np.save(f'{folder}/{fn}_train.npy', train)
			np.save(f'{folder}/{fn}_test.npy', test)
			labels = np.zeros(test.shape)
			indices = values[values['chan_id'] == fn]['anomaly_sequences'].values[0]
			indices = indices.replace(']', '').replace('[', '').split(', ')
			indices = [int(i) for i in indices]
			for i in range(0, len(indices), 2):
				labels[indices[i]:indices[i+1], :] = 1
			np.save(f'{folder}/{fn}_labels.npy', labels)
	elif dataset == 'WADI':
		dataset_folder = 'data/WADI'
		ls = pd.read_csv(os.path.join(dataset_folder, 'WADI_attacklabels.csv'))
		train = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days.csv'), skiprows=1000, nrows=2e5)
		test = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdata.csv'))
		train.dropna(how='all', inplace=True); test.dropna(how='all', inplace=True)
		train.fillna(0, inplace=True); test.fillna(0, inplace=True)
		test['Time'] = test['Time'].astype(str)
		test['Time'] = pd.to_datetime(test['Date'] + ' ' + test['Time'])
		labels = test.copy(deep = True)
		for i in test.columns.tolist()[3:]: labels[i] = 0
		for i in ['Start Time', 'End Time']: 
			ls[i] = ls[i].astype(str)
			ls[i] = pd.to_datetime(ls['Date'] + ' ' + ls[i])
		for index, row in ls.iterrows():
			to_match = row['Affected'].split(', ')
			matched = []
			for i in test.columns.tolist()[3:]:
				for tm in to_match:
					if tm in i: 
						matched.append(i); break			
			st, et = str(row['Start Time']), str(row['End Time'])
			labels.loc[(labels['Time'] >= st) & (labels['Time'] <= et), matched] = 1
		train, test, labels = convertNumpy(train), convertNumpy(test), convertNumpy(labels)
		print(train.shape, test.shape, labels.shape)
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset == 'MBA':
		dataset_folder = 'data/MBA'
		ls = pd.read_excel(os.path.join(dataset_folder, 'labels.xlsx'))
		train = pd.read_excel(os.path.join(dataset_folder, 'train.xlsx'))
		test = pd.read_excel(os.path.join(dataset_folder, 'test.xlsx'))
		train, test = train.values[1:,1:].astype(float), test.values[1:,1:].astype(float)
		train, min_a, max_a = normalize3(train)
		test, _, _ = normalize3(test, min_a, max_a)
		ls = ls.values[:,1].astype(int)
		labels = np.zeros_like(test)
		for i in range(-20, 20):
			labels[ls + i, :] = 1
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	else:
		raise Exception(f'Not Implemented. Check one of {datasets}')

if __name__ == '__main__':
	commands = sys.argv[1:]
	load = []
	if len(commands) > 0:
		for d in commands:
			load_data(d)
	else:
		print("Usage: python preprocess.py <datasets>")
		print(f"where <datasets> is space separated list of {datasets}")
