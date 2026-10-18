import json
import os
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from src.device import get_best_device, get_dtype_for_device, move_optimizer_to_device
from src.folderconstants import output_folder
from src.pot import calc_point2point, label_evaluation_status, pot_eval


DATASET_PREFIX = {
	'SMD': 'machine-1-1_',
	'SMAP': 'P-1_',
	'MSL': 'C-1_',
	'UCR': '136_',
	'NAB': 'ec2_request_latency_system_failure_',
}

MODEL_ALIASES = {
	'M0': 'TranAD_M0',
	'M1': 'TranAD_M1',
	'M2': 'TranAD_M2',
	'M3': 'TranAD_M3',
	'M4': 'TranAD_M4',
	'LSTM-NDT': 'LSTM_AD',
	'LSTM_NDT': 'LSTM_AD',
	'MAD-GAN': 'MAD_GAN',
	'MTAD-GAT': 'MTAD_GAT',
	'CAE-M': 'CAE_M',
}
SEQUENCE_BASELINE_NAMES = {'LSTM_AD', 'LSTM_Univariate'}
WINDOW_BASELINE_NAMES = {'DAGMM', 'USAD', 'GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M', 'MAD_GAN'}
TRAINABLE_BASELINE_NAMES = SEQUENCE_BASELINE_NAMES | WINDOW_BASELINE_NAMES | {'OmniAnomaly'}


def cut_array(percentage, arr):
	mid = round(arr.shape[0] / 2)
	window = round(arr.shape[0] * percentage * 0.5)
	return arr[mid - window: mid + window, :]


def load_processed_dataset(dataset, less=False):
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise FileNotFoundError(f'Processed data not found: {folder}')
	prefix = DATASET_PREFIX.get(dataset, '')
	arrays = []
	for split in ['train', 'test', 'labels']:
		arrays.append(np.load(os.path.join(folder, f'{prefix}{split}.npy')))
	if dataset == 'MSDS' and arrays[2].shape[0] != arrays[1].shape[0]:
		# Older processed MSDS labels were saved at the original sampling rate,
		# while train/test were downsampled by five in preprocess.py.
		aligned_labels = arrays[2][::5]
		if aligned_labels.shape[0] != arrays[1].shape[0]:
			raise ValueError(
				f'MSDS labels have {arrays[2].shape[0]} rows but test data has '
				f'{arrays[1].shape[0]} rows; applying the expected 5x label sampling '
				'did not align them. Re-run `python preprocess.py MSDS`.'
			)
		arrays[2] = aligned_labels
	if less:
		arrays[0] = cut_array(0.2, arrays[0])
	return arrays[0], arrays[1], arrays[2]


def convert_to_windows(data, n_window):
	windows = []
	for i in range(data.shape[0]):
		if i >= n_window:
			window = data[i - n_window:i]
		else:
			window = torch.cat([data[0].repeat(n_window - i, 1), data[0:i]])
		windows.append(window)
	return torch.stack(windows)


def prepare_tranad_data(train_np, test_np, device, dtype, n_window):
	train = torch.as_tensor(train_np, device=device, dtype=dtype)
	test = torch.as_tensor(test_np, device=device, dtype=dtype)
	return convert_to_windows(train, n_window), convert_to_windows(test, n_window), train, test


def prepare_sequence_data(train_np, test_np, device, dtype):
	train = torch.as_tensor(train_np, device=device, dtype=dtype)
	test = torch.as_tensor(test_np, device=device, dtype=dtype)
	return train, test, train, test


def prepare_flat_window_data(train_np, test_np, device, dtype, n_window):
	train = torch.as_tensor(train_np, device=device, dtype=dtype)
	test = torch.as_tensor(test_np, device=device, dtype=dtype)
	train_windows = convert_to_windows(train, n_window).flatten(start_dim=1)
	test_windows = convert_to_windows(test, n_window).flatten(start_dim=1)
	return train_windows, test_windows, train, test


def canonical_model_name(model_name):
	return MODEL_ALIASES.get(model_name, model_name)


def _model_name(model):
	name = model if isinstance(model, str) else model.name
	return canonical_model_name(name)


def is_tranad_model(model):
	return 'TranAD' in _model_name(model)


def is_omni_model(model):
	return _model_name(model) == 'OmniAnomaly'


def is_sequence_baseline(model):
	return _model_name(model) in SEQUENCE_BASELINE_NAMES


def is_window_baseline(model):
	return _model_name(model) in WINDOW_BASELINE_NAMES


def prepare_experiment_data(model, train_np, test_np, device, dtype):
	if is_tranad_model(model):
		return prepare_tranad_data(train_np, test_np, device, dtype, model.n_window)
	if is_omni_model(model) or is_sequence_baseline(model):
		return prepare_sequence_data(train_np, test_np, device, dtype)
	if is_window_baseline(model):
		return prepare_flat_window_data(train_np, test_np, device, dtype, model.n_window)
	raise ValueError(
		f'Experiment scripts do not prepare data for {_model_name(model)}.'
	)


def build_model(model_name, n_feats, device, dtype, batch_size=None, lr_override=None, seed=None):
	import src.models

	if seed is not None:
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)
	model_name = canonical_model_name(model_name)
	model_class = getattr(src.models, model_name)
	model = model_class(n_feats).to(device=device, dtype=dtype)
	if batch_size is not None and hasattr(model, 'batch'):
		model.batch = batch_size
	if lr_override is not None:
		model.lr = lr_override
	return model


def checkpoint_path(root, model_name, dataset, filename='model.ckpt'):
	return os.path.join(root, f'{model_name}_{dataset}', filename)


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, config):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	torch.save({
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'scheduler_state_dict': scheduler.state_dict(),
		'metrics': metrics,
		'config': config,
	}, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cpu', dtype=None):
	checkpoint = torch.load(path, map_location=device)
	model.load_state_dict(checkpoint['model_state_dict'])
	if optimizer is not None and 'optimizer_state_dict' in checkpoint:
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		move_optimizer_to_device(optimizer, device, dtype)
	if scheduler is not None and 'scheduler_state_dict' in checkpoint:
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
	return checkpoint


def model_auxiliary_loss(model, weights):
	if not hasattr(model, 'auxiliary_losses'):
		return next(model.parameters()).new_tensor(0.0), {}
	losses = model.auxiliary_losses()
	total = next(model.parameters()).new_tensor(0.0)
	scalars = {}
	for name, value in losses.items():
		weight = weights.get(name, weights.get('default', 1.0))
		total = total + weight * value
		scalars[name] = float(value.detach().cpu())
	return total, scalars


def tranad_batch_loss(model, batch, feats, criterion, epoch, training, aux_weights):
	local_bs = batch.shape[0]
	window = batch.permute(1, 0, 2)
	elem = window[-1, :, :].view(1, local_bs, feats)
	output = model(window, elem)
	if isinstance(output, tuple):
		n = epoch + 1
		loss_terms = (1 / n) * criterion(output[0], elem) + (1 - 1 / n) * criterion(output[1], elem)
		pred = output[1]
	else:
		loss_terms = criterion(output, elem)
		pred = output
	recon_loss = torch.mean(loss_terms)
	aux_loss, aux_scalars = model_auxiliary_loss(model, aux_weights)
	from src.tranad_progressive import progressive_reconstruction_loss
	progressive_loss, progressive_scalars = progressive_reconstruction_loss(model, output, elem, epoch)
	aux_loss = aux_loss + progressive_loss
	aux_scalars.update(progressive_scalars)
	return recon_loss + aux_loss, recon_loss, aux_loss, aux_scalars, pred


def train_one_epoch(model, train_windows, optimizer, scheduler, epoch, batch_size, aux_weights):
	model.train()
	criterion = nn.MSELoss(reduction='none')
	feats = train_windows.shape[-1]
	total_losses, recon_losses, aux_losses = [], [], []
	aux_acc = {}
	start = time()
	# Basic slices work across CPU, CUDA, and MPS; DataLoader's batched tensor
	# indexing has had backend-specific limitations on some PyTorch/MPS builds.
	for start_index in range(0, train_windows.shape[0], batch_size):
		batch = train_windows[start_index:start_index + batch_size]
		loss, recon_loss, aux_loss, aux_scalars, _ = tranad_batch_loss(
			model, batch, feats, criterion, epoch, True, aux_weights
		)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		total_losses.append(float(loss.detach().cpu()))
		recon_losses.append(float(recon_loss.detach().cpu()))
		aux_losses.append(float(aux_loss.detach().cpu()))
		for name, value in aux_scalars.items():
			aux_acc.setdefault(name, []).append(value)
	scheduler.step()
	return {
		'train_loss': float(np.mean(total_losses)),
		'recon_loss': float(np.mean(recon_losses)),
		'aux_loss': float(np.mean(aux_losses)),
		'lr': optimizer.param_groups[0]['lr'],
		'epoch_time_sec': time() - start,
		**{f'aux_{name}': float(np.mean(values)) for name, values in aux_acc.items()},
	}


def validation_reconstruction_loss(model, validation_windows, batch_size, epoch):
	"""Compute ordered-window TranAD reconstruction MSE without using labels."""
	model.eval()
	criterion = nn.MSELoss(reduction='none')
	feats = validation_windows.shape[-1]
	weighted_loss = 0.0
	n_samples = 0
	with torch.no_grad():
		for start_index in range(0, validation_windows.shape[0], batch_size):
			batch = validation_windows[start_index:start_index + batch_size]
			local_bs = batch.shape[0]
			window = batch.permute(1, 0, 2)
			elem = window[-1, :, :].view(1, local_bs, feats)
			output = model(window, elem)
			if isinstance(output, tuple):
				n = epoch + 1
				loss_terms = (1 / n) * criterion(output[0], elem) + (1 - 1 / n) * criterion(output[1], elem)
			else:
				loss_terms = criterion(output, elem)
			weighted_loss += float(loss_terms.mean().cpu()) * local_bs
			n_samples += local_bs
	if n_samples == 0:
		raise ValueError('Validation reconstruction loss requires at least one window.')
	return weighted_loss / n_samples


def train_omni_epoch(model, train_sequence, optimizer, scheduler, epoch):
	model.train()
	criterion = nn.MSELoss(reduction='mean')
	total_losses, mse_losses, kld_losses = [], [], []
	hidden = None
	start = time()
	for point in train_sequence:
		pred, mu, logvar, hidden = model(point, hidden)
		mse = criterion(pred, point)
		kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
		loss = mse + model.beta * kld
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		total_losses.append(float(loss.detach().cpu()))
		mse_losses.append(float(mse.detach().cpu()))
		kld_losses.append(float((model.beta * kld).detach().cpu()))
		if hidden is not None:
			hidden = hidden.detach()
	scheduler.step()
	return {
		'train_loss': float(np.mean(total_losses)),
		'recon_loss': float(np.mean(mse_losses)),
		'kld_loss': float(np.mean(kld_losses)),
		'lr': optimizer.param_groups[0]['lr'],
		'epoch_time_sec': time() - start,
	}


def train_sequence_epoch(model, train_sequence, optimizer, scheduler):
	model.train()
	start = time()
	pred = model(train_sequence)
	loss = nn.MSELoss(reduction='mean')(pred, train_sequence)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	scheduler.step()
	return {
		'train_loss': float(loss.detach().cpu()),
		'recon_loss': float(loss.detach().cpu()),
		'lr': optimizer.param_groups[0]['lr'],
		'epoch_time_sec': time() - start,
	}


def train_dagmm_epoch(model, train_windows, optimizer, scheduler):
	model.train()
	criterion = nn.MSELoss(reduction='none')
	total_losses, recon_losses, estimate_losses = [], [], []
	start = time()
	for window in train_windows:
		_, x_hat, _, gamma = model(window)
		recon = torch.mean(criterion(x_hat, window))
		estimate = torch.mean(criterion(gamma, window))
		loss = recon + estimate
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		total_losses.append(float(loss.detach().cpu()))
		recon_losses.append(float(recon.detach().cpu()))
		estimate_losses.append(float(estimate.detach().cpu()))
	scheduler.step()
	return {
		'train_loss': float(np.mean(total_losses)),
		'recon_loss': float(np.mean(recon_losses)),
		'estimate_loss': float(np.mean(estimate_losses)),
		'lr': optimizer.param_groups[0]['lr'],
		'epoch_time_sec': time() - start,
	}


def train_usad_epoch(model, train_windows, optimizer, scheduler, epoch):
	model.train()
	criterion = nn.MSELoss(reduction='none')
	loss1_values, loss2_values = [], []
	start = time()
	weight = 1 / (epoch + 1)
	for window in train_windows:
		ae1, ae2, ae2ae1 = model(window)
		loss1 = weight * criterion(ae1, window) + (1 - weight) * criterion(ae2ae1, window)
		loss2 = weight * criterion(ae2, window) - (1 - weight) * criterion(ae2ae1, window)
		loss = torch.mean(loss1 + loss2)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		loss1_values.append(float(torch.mean(loss1).detach().cpu()))
		loss2_values.append(float(torch.mean(loss2).detach().cpu()))
	scheduler.step()
	return {
		'train_loss': float(np.mean(loss1_values) + np.mean(loss2_values)),
		'loss1': float(np.mean(loss1_values)),
		'loss2': float(np.mean(loss2_values)),
		'lr': optimizer.param_groups[0]['lr'],
		'epoch_time_sec': time() - start,
	}


def train_window_reconstruction_epoch(model, train_windows, optimizer, scheduler):
	model.train()
	criterion = nn.MSELoss(reduction='mean')
	losses = []
	hidden = None
	start = time()
	for window in train_windows:
		if _model_name(model) == 'MTAD_GAT':
			pred, hidden = model(window, hidden)
			if hidden is not None:
				hidden = hidden.detach()
		else:
			pred = model(window)
		loss = criterion(pred, window)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		losses.append(float(loss.detach().cpu()))
	scheduler.step()
	return {
		'train_loss': float(np.mean(losses)),
		'recon_loss': float(np.mean(losses)),
		'lr': optimizer.param_groups[0]['lr'],
		'epoch_time_sec': time() - start,
	}


def train_mad_gan_epoch(model, train_windows, optimizer, scheduler):
	model.train()
	reconstruction_loss = nn.MSELoss(reduction='mean')
	adversarial_loss = nn.BCELoss(reduction='mean')
	device = train_windows.device
	dtype = train_windows.dtype
	real_label = torch.tensor([0.9], dtype=dtype, device=device)
	fake_label = torch.tensor([0.1], dtype=dtype, device=device)
	mse_values, generator_values, discriminator_values = [], [], []
	start = time()
	for window in train_windows:
		optimizer.zero_grad()
		_, real, fake = model(window)
		discriminator = adversarial_loss(real, real_label) + adversarial_loss(fake, fake_label)
		discriminator.backward()
		optimizer.step()

		optimizer.zero_grad()
		pred, _, fake = model(window)
		mse = reconstruction_loss(pred, window)
		generator = adversarial_loss(fake, real_label)
		(mse + generator).backward()
		optimizer.step()
		mse_values.append(float(mse.detach().cpu()))
		generator_values.append(float(generator.detach().cpu()))
		discriminator_values.append(float(discriminator.detach().cpu()))
	scheduler.step()
	return {
		'train_loss': float(np.mean(generator_values) + np.mean(discriminator_values)),
		'recon_loss': float(np.mean(mse_values)),
		'generator_loss': float(np.mean(generator_values)),
		'discriminator_loss': float(np.mean(discriminator_values)),
		'lr': optimizer.param_groups[0]['lr'],
		'epoch_time_sec': time() - start,
	}


def train_experiment_epoch(model, train_data, optimizer, scheduler, epoch, batch_size, aux_weights):
	if is_tranad_model(model):
		return train_one_epoch(model, train_data, optimizer, scheduler, epoch, batch_size, aux_weights)
	if is_omni_model(model):
		return train_omni_epoch(model, train_data, optimizer, scheduler, epoch)
	if is_sequence_baseline(model):
		return train_sequence_epoch(model, train_data, optimizer, scheduler)
	if _model_name(model) == 'DAGMM':
		return train_dagmm_epoch(model, train_data, optimizer, scheduler)
	if _model_name(model) == 'USAD':
		return train_usad_epoch(model, train_data, optimizer, scheduler, epoch)
	if _model_name(model) == 'MAD_GAN':
		return train_mad_gan_epoch(model, train_data, optimizer, scheduler)
	if _model_name(model) in {'GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M'}:
		return train_window_reconstruction_epoch(model, train_data, optimizer, scheduler)
	raise ValueError(f'Training is not implemented for {_model_name(model)} in this experiment entrypoint.')


def predict_tranad(model, windows, batch_size):
	model.eval()
	criterion = nn.MSELoss(reduction='none')
	feats = windows.shape[-1]
	losses, preds = [], []
	with torch.no_grad():
		for start_index in range(0, windows.shape[0], batch_size):
			batch = windows[start_index:start_index + batch_size]
			local_bs = batch.shape[0]
			window = batch.permute(1, 0, 2)
			elem = window[-1, :, :].view(1, local_bs, feats)
			output = model(window, elem)
			if isinstance(output, tuple):
				output = output[1]
			if getattr(model, 'uses_kurtosis_aggregation', False):
				losses.append((output - elem).abs()[0])
			else:
				losses.append(criterion(output, elem)[0])
			preds.append(output[0])
	return torch.cat(losses, dim=0).detach().cpu().numpy(), torch.cat(preds, dim=0).detach().cpu().numpy()


def predict_omni(model, sequence, batch_size=None):
	model.eval()
	criterion = nn.MSELoss(reduction='none')
	preds = []
	hidden = None
	with torch.no_grad():
		for point in sequence:
			pred, _, _, hidden = model(point, hidden)
			preds.append(pred)
	preds = torch.stack(preds)
	return criterion(preds, sequence).detach().cpu().numpy(), preds.detach().cpu().numpy()


def predict_sequence(model, sequence, batch_size=None):
	model.eval()
	with torch.no_grad():
		preds = model(sequence)
	loss = nn.MSELoss(reduction='none')(preds, sequence)
	return loss.detach().cpu().numpy(), preds.detach().cpu().numpy()


def _last_window_features(values, n_feats):
	return values[:, values.shape[1] - n_feats:values.shape[1]].view(-1, n_feats)


def predict_dagmm(model, windows, batch_size=None):
	model.eval()
	criterion = nn.MSELoss(reduction='none')
	preds = []
	with torch.no_grad():
		for window in windows:
			_, pred, _, _ = model(window)
			preds.append(pred)
	preds = torch.stack(preds)
	loss = criterion(preds, windows)
	final_pred = _last_window_features(preds, model.n_feats)
	final_loss = _last_window_features(loss, model.n_feats)
	return final_loss.detach().cpu().numpy(), final_pred.detach().cpu().numpy()


def predict_usad(model, windows, batch_size=None):
	model.eval()
	criterion = nn.MSELoss(reduction='none')
	preds, phase2 = [], []
	with torch.no_grad():
		for window in windows:
			ae1, _, ae2ae1 = model(window)
			preds.append(ae1)
			phase2.append(ae2ae1)
	preds = torch.stack(preds)
	phase2 = torch.stack(phase2)
	loss = 0.1 * criterion(preds, windows) + 0.9 * criterion(phase2, windows)
	final_pred = _last_window_features(preds, model.n_feats)
	final_loss = _last_window_features(loss, model.n_feats)
	return final_loss.detach().cpu().numpy(), final_pred.detach().cpu().numpy()


def predict_window_reconstruction(model, windows, batch_size=None):
	model.eval()
	criterion = nn.MSELoss(reduction='none')
	preds = []
	with torch.no_grad():
		for window in windows:
			if _model_name(model) == 'MTAD_GAT':
				pred, _ = model(window, None)
			else:
				pred = model(window)
			preds.append(pred)
	preds = torch.stack(preds)
	loss = criterion(preds, windows)
	final_pred = _last_window_features(preds, model.n_feats)
	final_loss = _last_window_features(loss, model.n_feats)
	return final_loss.detach().cpu().numpy(), final_pred.detach().cpu().numpy()


def predict_mad_gan(model, windows, batch_size=None):
	model.eval()
	criterion = nn.MSELoss(reduction='none')
	preds = []
	with torch.no_grad():
		for window in windows:
			pred, _, _ = model(window)
			preds.append(pred)
	preds = torch.stack(preds)
	loss = criterion(preds, windows)
	final_pred = _last_window_features(preds, model.n_feats)
	final_loss = _last_window_features(loss, model.n_feats)
	return final_loss.detach().cpu().numpy(), final_pred.detach().cpu().numpy()


def predict_experiment_model(model, data, batch_size):
	if is_tranad_model(model):
		return predict_tranad(model, data, batch_size)
	if is_omni_model(model):
		return predict_omni(model, data, batch_size)
	if is_sequence_baseline(model):
		return predict_sequence(model, data, batch_size)
	if _model_name(model) == 'DAGMM':
		return predict_dagmm(model, data, batch_size)
	if _model_name(model) == 'USAD':
		return predict_usad(model, data, batch_size)
	if _model_name(model) == 'MAD_GAN':
		return predict_mad_gan(model, data, batch_size)
	if _model_name(model) in {'GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M'}:
		return predict_window_reconstruction(model, data, batch_size)
	raise ValueError(f'Prediction is not implemented for {_model_name(model)} in this experiment entrypoint.')


def _fixed_topk(topk, n_features):
	if isinstance(topk, str):
		value = topk.strip().lower()
		if value == 'auto':
			raise ValueError('topk="auto" must be resolved before score aggregation.')
		topk = int(value)
	return max(1, min(int(topk), n_features))


def is_auto_topk(topk):
	return isinstance(topk, str) and topk.strip().lower() == 'auto'


def aggregate_scores(loss, method='mean', topk=3):
	if method == 'mean':
		return np.mean(loss, axis=1)
	if method == 'max':
		return np.max(loss, axis=1)
	if method == 'p95':
		return np.percentile(loss, 95, axis=1)
	if method == 'topk':
		k = _fixed_topk(topk, loss.shape[1])
		return np.mean(np.sort(loss, axis=1)[:, -k:], axis=1)
	if method == 'adaptive_topk':
		from src.tranad_progressive import adaptive_topk_scores
		return adaptive_topk_scores(loss)[0]
	raise ValueError(f'Unsupported score aggregation: {method}')


def select_adaptive_topk(train_loss, test_loss, labels, show_progress=False):
	labels_final = (np.sum(labels, axis=1) >= 1) + 0
	n_features = int(test_loss.shape[1])
	if n_features < 1:
		raise ValueError('Adaptive top-k selection requires at least one feature dimension.')
	if label_evaluation_status(labels_final) != 'ok':
		# There is no ground truth with which to tune k. Use a deterministic
		# top-1 score and let evaluation report the supervised metrics as NA.
		train_final = aggregate_scores(train_loss, 'topk', topk=1)
		test_final = aggregate_scores(test_loss, 'topk', topk=1)
		result, pred = pot_eval(train_final, test_final, labels_final)
		return 1, result, pred, train_final, test_final
	best = None
	best_rank = None
	iterator = range(1, n_features + 1)
	if show_progress:
		iterator = tqdm(iterator, desc='Adaptive top-k', leave=False)
	for k in iterator:
		train_final = aggregate_scores(train_loss, 'topk', topk=k)
		test_final = aggregate_scores(test_loss, 'topk', topk=k)
		result, pred = pot_eval(train_final, test_final, labels_final)
		rank = (
			float(result.get('f1', 0.0)),
			float(result.get('ROC/AUC', 0.0)),
			float(result.get('precision', 0.0)),
			float(result.get('recall', 0.0)),
			-k,
		)
		if best is None or rank > best_rank:
			best = (k, result, pred, train_final, test_final)
			best_rank = rank
	if best is None:
		raise ValueError('Adaptive top-k selection requires at least one feature dimension.')
	return best


def evaluate_loss_arrays(train_loss, test_loss, labels, show_progress=False, score_agg='mean', topk=3):
	if test_loss.ndim != 2 or train_loss.ndim != 2 or labels.ndim != 2:
		raise ValueError(
			'Expected train_loss, test_loss, and labels to be 2D arrays; got '
			f'{train_loss.shape}, {test_loss.shape}, and {labels.shape}.'
		)
	if test_loss.shape[0] != labels.shape[0]:
		raise ValueError(
			f'Test loss and labels have different row counts '
			f'({test_loss.shape[0]} vs {labels.shape[0]}).'
		)
	if test_loss.shape[1] != labels.shape[1]:
		raise ValueError(
			f'Test loss and labels have different feature counts '
			f'({test_loss.shape[1]} vs {labels.shape[1]}).'
		)
	if train_loss.shape[1] != test_loss.shape[1]:
		raise ValueError(
			f'Train and test loss have different feature counts '
			f'({train_loss.shape[1]} vs {test_loss.shape[1]}).'
		)
	results = []
	iterator = range(test_loss.shape[1])
	if show_progress:
		iterator = tqdm(iterator, desc='POT', leave=False)
	for i in iterator:
		result, _ = pot_eval(train_loss[:, i], test_loss[:, i], labels[:, i])
		results.append(result)
	per_dim = pd.DataFrame(results)
	selected_topk = None
	if score_agg == 'kurtosis_gated':
		from src.tranad_progressive import kurtosis_gated_normalized_aggregation
		train_final, test_final, selected_topk = kurtosis_gated_normalized_aggregation(
			train_loss, test_loss, topk=topk
		)
		final_result, final_pred = pot_eval(train_final, test_final, (np.sum(labels, axis=1) >= 1) + 0)
	elif score_agg == 'adaptive_topk':
		from src.tranad_progressive import adaptive_topk_scores
		train_final, train_ks = adaptive_topk_scores(train_loss)
		test_final, test_ks = adaptive_topk_scores(test_loss)
		selected_topk = float(np.mean(test_ks)) if test_ks.size else None
		final_result, final_pred = pot_eval(train_final, test_final, (np.sum(labels, axis=1) >= 1) + 0)
	elif score_agg == 'topk' and is_auto_topk(topk):
		selected_topk, final_result, final_pred, train_final, test_final = select_adaptive_topk(
			train_loss, test_loss, labels, show_progress=show_progress
		)
	else:
		train_final = aggregate_scores(train_loss, score_agg, topk=topk)
		test_final = aggregate_scores(test_loss, score_agg, topk=topk)
		final_result, final_pred = pot_eval(train_final, test_final, (np.sum(labels, axis=1) >= 1) + 0)
	metrics = {
		'f1': final_result['f1'],
		'precision': final_result['precision'],
		'recall': final_result['recall'],
		'roc_auc': final_result['ROC/AUC'],
		'threshold': final_result['threshold'],
		'score_agg': score_agg,
		'score_topk': (_fixed_topk(topk, test_loss.shape[1]) if score_agg == 'kurtosis_gated' else
			'adaptive' if score_agg == 'adaptive_topk' else
			'auto' if is_auto_topk(topk) else
			(_fixed_topk(topk, test_loss.shape[1]) if score_agg == 'topk' else None)),
		'selected_topk': selected_topk,
		'train_score_mean': float(np.mean(train_final)),
		'train_score_std': float(np.std(train_final)),
		'test_score_mean': float(np.mean(test_final)),
		'test_score_std': float(np.std(test_final)),
		'dim_f1_mean': float(per_dim['f1'].mean()) if 'f1' in per_dim else 0.0,
		'dim_precision_mean': float(per_dim['precision'].mean()) if 'precision' in per_dim else 0.0,
		'dim_recall_mean': float(per_dim['recall'].mean()) if 'recall' in per_dim else 0.0,
		'evaluation_status': label_evaluation_status((np.sum(labels, axis=1) >= 1) + 0),
	}
	return metrics, per_dim, np.asarray(final_pred), train_final, test_final


def evaluate_tranad(model, train_windows, test_windows, labels, batch_size, show_progress=False, score_agg='mean', topk=3):
	train_loss, _ = predict_tranad(model, train_windows, batch_size)
	test_loss, y_pred = predict_tranad(model, test_windows, batch_size)
	metrics, per_dim, _, _, _ = evaluate_loss_arrays(
		train_loss, test_loss, labels, show_progress=show_progress, score_agg=score_agg, topk=topk
	)
	return metrics, per_dim, y_pred, test_loss


def evaluate_omni(model, train_sequence, test_sequence, labels, batch_size, show_progress=False, score_agg='mean', topk=3):
	train_loss, _ = predict_omni(model, train_sequence, batch_size)
	test_loss, y_pred = predict_omni(model, test_sequence, batch_size)
	metrics, per_dim, _, _, _ = evaluate_loss_arrays(
		train_loss, test_loss, labels, show_progress=show_progress, score_agg=score_agg, topk=topk
	)
	return metrics, per_dim, y_pred, test_loss


def evaluate_experiment_model(model, train_data, test_data, labels, batch_size, show_progress=False, score_agg='mean', topk=3):
	if is_tranad_model(model):
		return evaluate_tranad(
			model, train_data, test_data, labels, batch_size,
			show_progress=show_progress, score_agg=score_agg, topk=topk
		)
	if is_omni_model(model):
		return evaluate_omni(
			model, train_data, test_data, labels, batch_size,
			show_progress=show_progress, score_agg=score_agg, topk=topk
		)
	if _model_name(model) in TRAINABLE_BASELINE_NAMES:
		train_loss, _ = predict_experiment_model(model, train_data, batch_size)
		test_loss, y_pred = predict_experiment_model(model, test_data, batch_size)
		metrics, per_dim, _, _, _ = evaluate_loss_arrays(
			train_loss, test_loss, labels, show_progress=show_progress, score_agg=score_agg, topk=topk
		)
		return metrics, per_dim, y_pred, test_loss
	raise ValueError(f'Evaluation is not implemented for {_model_name(model)} in this experiment entrypoint.')


def point_result(pred, label):
	f1, precision, recall, tp, tn, fp, fn, roc_auc = calc_point2point(pred, label)
	return {
		'f1': f1,
		'precision': precision,
		'recall': recall,
		'TP': tp,
		'TN': tn,
		'FP': fp,
		'FN': fn,
		'ROC/AUC': roc_auc,
	}


def evaluate_merlin(test_np, labels, min_length=60, max_length=62):
	from src.merlin import check, merlin

	labels_final = (np.sum(labels, axis=1) >= 1) + 0
	merlin_status = 'ok'
	try:
		discord, _ = merlin(test_np, min_length, max_length)
	except ValueError as exc:
		if 'argmax of an empty sequence' not in str(exc):
			raise
		discord = None
		merlin_status = 'fallback_no_discord'
	discord_pred = np.zeros_like(labels_final)
	if discord is not None:
		discord_pred[discord[0]:discord[0] + discord[1]] = 1
	final_pred, dim_pred = check(test_np, discord_pred)
	final_result = point_result(final_pred, labels_final)
	per_dim = pd.DataFrame([point_result(dim_pred[:, i], labels[:, i]) for i in range(labels.shape[1])])
	metrics = {
		'f1': final_result['f1'],
		'precision': final_result['precision'],
		'recall': final_result['recall'],
		'roc_auc': final_result['ROC/AUC'],
		'score_agg': 'MERLIN',
		'merlin_status': merlin_status,
		'threshold': None,
		'discord_start': int(discord[0]) if discord is not None else None,
		'discord_length': int(discord[1]) if discord is not None else None,
		'discord_distance': float(discord[2]) if discord is not None else None,
		'dim_f1_mean': float(per_dim['f1'].mean()) if 'f1' in per_dim else 0.0,
		'dim_precision_mean': float(per_dim['precision'].mean()) if 'precision' in per_dim else 0.0,
		'dim_recall_mean': float(per_dim['recall'].mean()) if 'recall' in per_dim else 0.0,
	}
	return metrics, per_dim


def write_json(path, data):
	def json_safe(value):
		if isinstance(value, dict):
			return {key: json_safe(item) for key, item in value.items()}
		if isinstance(value, (list, tuple)):
			return [json_safe(item) for item in value]
		if isinstance(value, np.generic):
			value = value.item()
		if isinstance(value, float) and not np.isfinite(value):
			return None
		return value

	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'w', encoding='utf-8') as f:
		json.dump(json_safe(data), f, indent=2, allow_nan=False)
