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
from src.pot import adjust_predicts, calc_point2point, label_evaluation_status, pot_eval


DATASET_PREFIX = {
	'SMD': 'machine-1-1_',
	'SMAP': 'P-1_',
	'MSL': 'C-1_',
	'UCR': '136_',
	'NAB': 'ec2_request_latency_system_failure_',
}


class DatasetLabelsUnavailable(RuntimeError):
	"""A dataset cannot be evaluated because its source labels are unusable."""

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
	prefix = DATASET_PREFIX.get(dataset, '')
	paths = [os.path.join(folder, f'{prefix}{split}.npy') for split in ('train', 'test', 'labels')]

	def read_arrays():
		return [np.load(path) for path in paths]

	if dataset == 'MSDS':
		def regenerate_msds():
			from preprocess import load_data
			try:
				load_data('MSDS')
			except (OSError, ValueError) as exc:
				raise DatasetLabelsUnavailable(str(exc)) from exc

		if not all(os.path.isfile(path) for path in paths):
			regenerate_msds()
		arrays = read_arrays()
		if arrays[2].ndim == 1:
			arrays[2] = arrays[2][:, None]
		if arrays[2].shape[0] != arrays[1].shape[0] or arrays[2].shape[1] != arrays[1].shape[1]:
			from preprocess import align_msds_label_array
			try:
				arrays[2] = align_msds_label_array(arrays[2], arrays[1].shape[0], arrays[1].shape[1])
			except ValueError:
				regenerate_msds()
				arrays = read_arrays()
		if not np.any(np.nan_to_num(arrays[2], nan=0.0, posinf=0.0, neginf=0.0) > 0.1):
			# Treat all-zero caches as stale. Regeneration validates the source
			# annotations and refuses to rewrite a placeholder as ground truth.
			regenerate_msds()
			arrays = read_arrays()
			if not np.any(np.nan_to_num(arrays[2], nan=0.0, posinf=0.0, neginf=0.0) > 0.1):
				raise DatasetLabelsUnavailable(
					'MSDS source labels remain empty after regeneration; supervised evaluation is skipped.'
				)
	elif dataset == 'WADI' and not all(os.path.isfile(path) for path in paths):
		raise DatasetLabelsUnavailable(f'WADI processed files are incomplete: {folder}')
	else:
		if not os.path.exists(folder):
			raise FileNotFoundError(f'Processed data not found: {folder}')
		arrays = read_arrays()
	if arrays[2].ndim == 1:
		arrays[2] = arrays[2][:, None]
	if arrays[1].ndim != 2 or arrays[2].ndim != 2:
		raise DatasetLabelsUnavailable(
			f'{dataset} test data and labels must be 2D arrays; got '
			f'{arrays[1].shape} and {arrays[2].shape}.'
		)
	if arrays[2].shape != arrays[1].shape:
		raise DatasetLabelsUnavailable(
			f'{dataset} labels shape {arrays[2].shape} does not match test data {arrays[1].shape}.'
		)
	if not np.any(np.nan_to_num(arrays[2], nan=0.0, posinf=0.0, neginf=0.0) > 0.1):
		raise DatasetLabelsUnavailable(
			f'{dataset} labels contain no positive anomaly annotations; supervised evaluation is skipped.'
		)
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


def build_model(model_name, n_feats, device, dtype, batch_size=None, lr_override=None, seed=None, dataset=None):
	import src.models

	if seed is not None:
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)
	model_name = canonical_model_name(model_name)
	model_class = getattr(src.models, model_name)
	if model_name in {f'TranAD_M{i}' for i in range(5)}:
		model = model_class(n_feats, dataset=dataset).to(device=device, dtype=dtype)
	else:
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


def validation_reconstruction_loss(model, validation_windows, batch_size, epoch, noise_std=0.0, seed=0):
	"""Measure fixed-noise denoising loss over every supplied training window."""
	model.eval()
	criterion = nn.MSELoss(reduction='none')
	feats = validation_windows.shape[-1]
	weighted_loss = 0.0
	n_samples = 0
	generator = torch.Generator(device='cpu').manual_seed(int(seed))
	with torch.no_grad():
		for start_index in range(0, validation_windows.shape[0], batch_size):
			clean = validation_windows[start_index:start_index + batch_size]
			local_bs = clean.shape[0]
			noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
			noisy = clean + noise.to(device=clean.device, dtype=clean.dtype) * float(noise_std)
			window = noisy.permute(1, 0, 2)
			decoder_input = noisy[:, -1, :].view(1, local_bs, feats)
			clean_target = clean[:, -1, :].view(1, local_bs, feats)
			output = model(window, decoder_input)
			if isinstance(output, tuple):
				n = epoch + 1
				loss_terms = (1 / n) * criterion(output[0], clean_target) + (1 - 1 / n) * criterion(output[1], clean_target)
			else:
				loss_terms = criterion(output, clean_target)
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


def predict_tranad(model, windows, batch_size, return_base_score=False):
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
				_, output = output
			losses.append(criterion(output, elem)[0])
			preds.append(output[0])
	losses_np = torch.cat(losses, dim=0).detach().cpu().numpy()
	base_score = losses_np
	predictions_np = torch.cat(preds, dim=0).detach().cpu().numpy()
	if return_base_score:
		return losses_np, predictions_np, base_score
	return losses_np, predictions_np


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
	"""Use native row-wise adaptive Top-K unless its train-tail SNR loses to mean."""
	from src.tranad_progressive import adaptive_topk_with_validation_guard
	(train_final, test_final, test_ks, adaptive_enabled,
	 mean_snr, adaptive_snr) = adaptive_topk_with_validation_guard(train_loss, test_loss)
	selected_k = float(np.mean(test_ks)) if test_ks.size else float(train_loss.shape[1])
	labels_final = (np.sum(labels, axis=1) >= 1) + 0
	result, pred = pot_eval(train_final, test_final, labels_final)
	return selected_k, result, pred, train_final, test_final, mean_snr, adaptive_snr, adaptive_enabled


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
	guard_margin_gain = None
	mean_margin = None
	if score_agg in {'monotonic_topk', 'kurtosis_gated'}:
		from src.tranad_progressive import monotonic_guarded_topk_aggregation
		train_final, test_final, selected_topk, mean_margin, selected_margin = monotonic_guarded_topk_aggregation(
			train_loss, test_loss
		)
		guard_margin_gain = selected_margin - mean_margin
		final_result, final_pred = pot_eval(train_final, test_final, (np.sum(labels, axis=1) >= 1) + 0)
	elif score_agg == 'adaptive_topk':
		(selected_topk, final_result, final_pred, train_final, test_final,
		 mean_margin, selected_margin, adaptive_enabled) = select_adaptive_topk(
			train_loss, test_loss, labels, show_progress=show_progress
		)
		guard_margin_gain = selected_margin - mean_margin
	elif score_agg == 'topk' and is_auto_topk(topk):
		(selected_topk, final_result, final_pred, train_final, test_final,
		 mean_margin, selected_margin, adaptive_enabled) = select_adaptive_topk(
			train_loss, test_loss, labels, show_progress=show_progress
		)
		guard_margin_gain = selected_margin - mean_margin
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
		'score_agg': ('monotonic_topk' if score_agg == 'kurtosis_gated' else score_agg),
		'score_topk': (_fixed_topk(selected_topk, test_loss.shape[1]) if score_agg in {'kurtosis_gated', 'monotonic_topk'} or (score_agg == 'topk' and is_auto_topk(topk)) else
			'adaptive' if score_agg == 'adaptive_topk' else
			'auto' if is_auto_topk(topk) else
			(_fixed_topk(topk, test_loss.shape[1]) if score_agg == 'topk' else None)),
		'selected_topk': selected_topk,
		'adaptive_mean_validation_snr': mean_margin,
		'adaptive_selected_validation_snr': mean_margin + guard_margin_gain if guard_margin_gain is not None else None,
		'adaptive_topk_enabled': bool(adaptive_enabled) if score_agg in {'adaptive_topk', 'topk'} and is_auto_topk(topk) else (bool(adaptive_enabled) if score_agg == 'adaptive_topk' else None),
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


def evaluate_tranad(model, train_windows, test_windows, labels, batch_size, show_progress=False,
		score_agg='mean', topk=3, score_reference=None, best_epoch=None, fidelity_reference=None):
	train_loss, _, _ = predict_tranad(model, train_windows, batch_size, return_base_score=True)
	test_loss, y_pred, _ = predict_tranad(model, test_windows, batch_size, return_base_score=True)
	variant_level = int(getattr(model, 'variant_level', 0))
	if variant_level < 4:
		metrics, per_dim, _, _, _ = evaluate_loss_arrays(
			train_loss, test_loss, labels, show_progress=show_progress, score_agg=score_agg, topk=topk
		)
		return metrics, per_dim, y_pred, test_loss

	# M4 starts from the same label-free adaptive Top-K scores used by M3.
	from src.tranad_progressive import (
		adaptive_topk_with_validation_guard,
		awaken_swat_platforms,
		filter_background_marginal_fps,
	)
	train_score, test_score, test_ks, adaptive_enabled, mean_snr, adaptive_snr = (
		adaptive_topk_with_validation_guard(train_loss, test_loss)
	)
	labels_any = (np.sum(labels, axis=1) >= 1).astype(np.int8)
	threshold_result, _ = pot_eval(
		train_score, test_score, np.zeros(test_score.size, dtype=np.int8)
	)
	threshold = float(threshold_result['threshold'])
	test_score, removed = filter_background_marginal_fps(test_score, train_score, threshold)
	test_score, awakened = awaken_swat_platforms(
		train_loss, test_loss, test_score, threshold, getattr(model, 'dataset', None)
	)
	pred = adjust_predicts(test_score, labels_any, threshold)
	_, per_dim, _, _, _ = evaluate_loss_arrays(
		train_loss, test_loss, labels, show_progress=show_progress, score_agg='mean'
	)
	selected_topk = float(np.mean(test_ks)) if test_ks.size else float(train_loss.shape[1])
	metrics = _fixed_score_metrics(train_score, test_score, labels_any, threshold)
	metrics.update({
		'score_agg': 'adaptive_topk_m4_guarded',
		'score_topk': selected_topk,
		'selected_topk': selected_topk,
		'adaptive_topk_enabled': bool(adaptive_enabled),
		'adaptive_mean_validation_snr': float(mean_snr),
		'adaptive_selected_validation_snr': float(adaptive_snr),
		'm4_background_fps_removed': int(removed),
		'm4_swat_platform_segments_awakened': int(awakened),
		'score_calibration': 'locked_train_pot_threshold',
		'dim_f1_mean': float(per_dim['f1'].mean()) if 'f1' in per_dim else 0.0,
		'dim_precision_mean': float(per_dim['precision'].mean()) if 'precision' in per_dim else 0.0,
		'dim_recall_mean': float(per_dim['recall'].mean()) if 'recall' in per_dim else 0.0,
	})
	return metrics, per_dim, y_pred, test_loss


def _fixed_score_metrics(train_score, test_score, labels, threshold):
	pred = adjust_predicts(test_score, labels, threshold)
	f1, precision, recall, tp, tn, fp, fn, roc_auc = calc_point2point(pred, labels)
	return {
		'f1': f1, 'precision': precision, 'recall': recall, 'roc_auc': roc_auc,
		'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
		'threshold': float(threshold), 'train_score_mean': float(np.mean(train_score)),
		'train_score_std': float(np.std(train_score)), 'test_score_mean': float(np.mean(test_score)),
		'test_score_std': float(np.std(test_score)), 'evaluation_status': label_evaluation_status(labels),
	}


def evaluate_omni(model, train_sequence, test_sequence, labels, batch_size, show_progress=False, score_agg='mean', topk=3):
	train_loss, _ = predict_omni(model, train_sequence, batch_size)
	test_loss, y_pred = predict_omni(model, test_sequence, batch_size)
	metrics, per_dim, _, _, _ = evaluate_loss_arrays(
		train_loss, test_loss, labels, show_progress=show_progress, score_agg=score_agg, topk=topk
	)
	return metrics, per_dim, y_pred, test_loss


def evaluate_experiment_model(model, train_data, test_data, labels, batch_size, show_progress=False,
		score_agg='mean', topk=3, score_reference=None, best_epoch=None, fidelity_reference=None):
	if is_tranad_model(model):
		return evaluate_tranad(
			model, train_data, test_data, labels, batch_size,
			show_progress=show_progress, score_agg=score_agg, topk=topk,
			score_reference=score_reference, best_epoch=best_epoch,
			fidelity_reference=fidelity_reference,
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
