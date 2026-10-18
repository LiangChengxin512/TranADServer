"""Progressive M0--M4 TranAD ablation extensions.

The original :class:`TranAD` implementation remains the M0 reference. New
embedding, graph, loss, and score components live here and are only attached to
the generated ablation classes.
"""

import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ABLATION_VARIANTS = {f'M{i}' for i in range(5)}
ACTIVE_VARIANCE_EPS = 1e-4
SCORE_EPS = 1e-8


class MultiScaleCausalEmbedding(nn.Module):
	"""Parallel causal depthwise-separable convolutions with LayerScale fusion."""

	def __init__(self, channels, initial_scale):
		super().__init__()
		self.depthwise = nn.ModuleList([
			nn.Conv1d(channels, channels, kernel_size=k, groups=channels, bias=False)
			for k in (1, 3, 5)
		])
		self.pointwise = nn.ModuleList([
			nn.Conv1d(channels, channels, kernel_size=1)
			for _ in (1, 3, 5)
		])
		self.output_projection = nn.Conv1d(3 * channels, channels, kernel_size=1)
		self.layer_scale = nn.Parameter(torch.full((channels,), float(initial_scale)))

	def forward(self, x):
		# TranAD uses (time, batch, channels); Conv1d uses (batch, channels, time).
		conv_input = x.permute(1, 2, 0)
		features = []
		for kernel, depthwise, pointwise in zip((1, 3, 5), self.depthwise, self.pointwise):
			# Padding only on the left ensures position t cannot see any sample > t.
			causal_input = F.pad(conv_input, (kernel - 1, 0)) if kernel > 1 else conv_input
			features.append(pointwise(F.gelu(depthwise(causal_input))))
		local = self.output_projection(torch.cat(features, dim=1)).permute(2, 0, 1)
		scale = self.layer_scale.view(1, 1, -1)
		return x + scale * local


class NormalPrototypeMemory(nn.Module):
	"""Attention-addressed normal prototypes for encoder hidden states."""

	def __init__(self, hidden_size, prototypes=16, initial_scale=0.1):
		super().__init__()
		self.hidden_size = hidden_size
		self.prototypes = nn.Parameter(torch.randn(prototypes, hidden_size) * 0.02)
		self.alpha = nn.Parameter(torch.tensor(float(initial_scale)))

	def forward(self, hidden):
		address = torch.softmax(
			torch.matmul(hidden, self.prototypes.transpose(0, 1)) / math.sqrt(self.hidden_size),
			dim=-1,
		)
		memory = torch.matmul(address, self.prototypes)
		return hidden + self.alpha * memory


class AdaptiveGatedGraphAttention(nn.Module):
	"""Variance-masked feature graph with cosine soft-threshold adjacency."""

	def __init__(self, n_features, hidden=None, variance_epsilon=ACTIVE_VARIANCE_EPS):
		super().__init__()
		self.n_features = n_features
		self.variance_epsilon = variance_epsilon
		hidden = hidden or max(4, min(32, n_features * 2))
		self.node_embedding = nn.Parameter(torch.randn(n_features, hidden) * 0.02)
		self.signal_projection = nn.Linear(1, hidden, bias=False)
		self.value = nn.Linear(1, hidden, bias=False)
		self.output = nn.Linear(hidden, 1, bias=False)
		initial_threshold = torch.tensor(0.3)
		self.soft_threshold_logit = nn.Parameter(torch.logit(initial_threshold))
		gate_hidden = max(4, min(32, n_features))
		self.gate = nn.Sequential(
			nn.Linear(n_features, gate_hidden),
			nn.GELU(),
			nn.Linear(gate_hidden, 1),
		)
		nn.init.zeros_(self.gate[-1].weight)
		nn.init.constant_(self.gate[-1].bias, -2.0)

	def forward(self, x):
		if self.n_features == 1:
			return x
		time, batch, features = x.shape
		# A channel that is effectively constant over this batch never sends or
		# receives graph messages. It follows the residual identity path below.
		active = x.var(dim=0, unbiased=False) >= self.variance_epsilon  # [batch, features]
		nodes = x.reshape(time * batch, features, 1)
		node_ids = self.node_embedding.unsqueeze(0)
		embedded = node_ids + self.signal_projection(nodes)
		embedded = F.normalize(embedded, p=2, dim=-1, eps=SCORE_EPS)
		cosine = torch.bmm(embedded, embedded.transpose(1, 2))
		threshold = torch.sigmoid(self.soft_threshold_logit)
		adjacency = F.relu(cosine - threshold)
		active_flat = active.unsqueeze(0).expand(time, -1, -1).reshape(time * batch, features)
		pair_mask = active_flat.unsqueeze(2) & active_flat.unsqueeze(1)
		adjacency = adjacency * pair_mask.to(dtype=adjacency.dtype)

		# Keep a self-loop for empty rows; inactive channels are still bypassed.
		empty_rows = adjacency.sum(dim=-1, keepdim=True) <= SCORE_EPS
		identity = torch.eye(features, device=x.device, dtype=x.dtype).unsqueeze(0)
		adjacency = adjacency + empty_rows.to(dtype=x.dtype) * identity
		adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(SCORE_EPS)
		messages = torch.bmm(adjacency, self.value(nodes))
		graph_features = self.output(messages).reshape(time, batch, features)
		gamma = torch.sigmoid(self.gate(x.mean(dim=0))).unsqueeze(0)
		updated = (1.0 - gamma) * x + gamma * graph_features
		return torch.where(active.unsqueeze(0), updated, x)


def _rfft_pair(target, prediction):
	"""Compute a matched FFT pair, falling back to CPU when MPS lacks rFFT."""
	device = target.device
	try:
		return torch.fft.rfft(target, dim=0), torch.fft.rfft(prediction, dim=0), device
	except (RuntimeError, NotImplementedError):
		if device.type != 'mps':
			raise
		target_cpu, prediction_cpu = target.to('cpu'), prediction.to('cpu')
		return torch.fft.rfft(target_cpu, dim=0), torch.fft.rfft(prediction_cpu, dim=0), device


def _spectral_flatness(target):
	"""Measure broadband content in values and first differences per channel.

	The maximum keeps an isolated step edge broadband even though the raw step
	can have low spectral flatness, while a periodic waveform stays narrowband
	in both representations. Constant channels are excluded from the average.
	"""
	if target.shape[0] < 8:
		return target.new_tensor(1.0)
	device = target.device

	def per_channel_flatness(signal):
		centered = signal - signal.mean(dim=0, keepdim=True)
		try:
			power = torch.fft.rfft(centered, dim=0).abs().square()
		except (RuntimeError, NotImplementedError):
			if device.type != 'mps':
				raise
			power = torch.fft.rfft(centered.to('cpu'), dim=0).abs().square()
		# DC level and offset are not periodic structure.
		if power.shape[0] > 1:
			power = power[1:]
		mean_power = power.mean(dim=0)
		log_geometric_mean = torch.log(power.clamp_min(SCORE_EPS)).mean(dim=0)
		return (torch.exp(log_geometric_mean) / mean_power.clamp_min(SCORE_EPS)).clamp(0.0, 1.0).to(device=device)

	first_difference = target[1:] - target[:-1]
	value_flatness = per_channel_flatness(target)
	difference_flatness = per_channel_flatness(first_difference)
	active = target.var(dim=0, unbiased=False) > SCORE_EPS
	if not bool(active.any().item()):
		return target.new_tensor(1.0)
	flatness = torch.maximum(value_flatness, difference_flatness).to(device=device)
	return flatness[active].mean().clamp(0.0, 1.0)


def _amplitude_spectrum_loss(target, prediction):
	"""Mean-channel L1 rFFT amplitude difference, normalized by time length."""
	if target.shape[0] < 2:
		return prediction.sum() * 0.0
	target_spectrum, prediction_spectrum, device = _rfft_pair(target, prediction)
	target_amplitude = target_spectrum.abs()
	prediction_amplitude = prediction_spectrum.abs()
	loss = (target_amplitude - prediction_amplitude).abs().sum(dim=0).mean() / target.shape[0]
	return loss.to(device=device)


def progressive_reconstruction_loss(model, outputs, target, epoch):
	"""Keep TranAD's two-stage objective unchanged for every ablation level.

	M3 and M4 add causal residual scoring after reconstruction. They do not add
	new training penalties, which keeps M0's baseline objective intact.
	"""
	zero = target.new_tensor(0.0)
	return zero, {'time_loss': 0.0, 'freq_loss': 0.0, 'freq_weight': 0.0, 'trend_loss': 0.0}


def adaptive_topk_scores(residuals):
	"""Legacy adaptive top-k helper retained for earlier experiment scripts."""
	values = residuals.detach().cpu().numpy() if torch.is_tensor(residuals) else residuals
	values = np.asarray(values)
	if values.ndim != 2:
		raise ValueError('Adaptive top-k aggregation expects a [time, channels] residual matrix.')
	if values.shape[1] == 0:
		raise ValueError('Adaptive top-k aggregation needs at least one channel.')
	if values.shape[1] == 1:
		return values[:, 0], np.ones(values.shape[0], dtype=int)
	mean = np.mean(values, axis=1)
	std = np.std(values, axis=1)
	coefficient = std / (np.abs(mean) + 1e-12)
	ks = np.clip(np.ceil(values.shape[1] / (1.0 + coefficient)).astype(int), 1, values.shape[1])
	sorted_values = np.sort(values, axis=1)[:, ::-1]
	mask = np.arange(values.shape[1])[None, :] < ks[:, None]
	scores = (sorted_values * mask).sum(axis=1) / ks
	return scores, ks


def causal_persistence_filter(residuals, beta=0.35, median_window=5, mad_scale=3.0):
	"""EMA smooth residuals and gate isolated spikes against a causal median."""
	values = np.asarray(residuals, dtype=np.float64)
	if values.ndim != 2:
		raise ValueError('Residual filtering expects a [time, channels] matrix.')
	if not 0.0 <= beta < 1.0:
		raise ValueError('EMA beta must be in [0, 1).')
	if median_window < 1:
		raise ValueError('The causal median window must be positive.')
	if values.shape[0] == 0:
		return values.copy()
	ema = np.empty_like(values)
	filtered = np.empty_like(values)
	ema[0] = values[0]
	filtered[0] = values[0]
	for index in range(1, values.shape[0]):
		ema[index] = (1.0 - beta) * values[index] + beta * ema[index - 1]
		start = max(0, index - median_window + 1)
		local = values[start:index + 1]
		median = np.median(local, axis=0)
		mad = np.median(np.abs(local - median[None, :]), axis=0) * 1.4826
		departure = np.abs(values[index] - median)
		gate = np.clip((departure - mad_scale * mad) / np.maximum(departure, SCORE_EPS), 0.0, 1.0)
		filtered[index] = (1.0 - gate) * ema[index] + gate * median
	return filtered


def causal_horizon_shift_scores(windows, train_mean, train_std, decay=0.995, active_epsilon=1e-4):
	"""Score local/slow-horizon shifts against the normal training baseline."""
	values = windows.detach().cpu().numpy() if torch.is_tensor(windows) else np.asarray(windows)
	values = np.asarray(values, dtype=np.float64)
	if values.ndim != 3:
		raise ValueError('Horizon shift scoring expects [time, window, channels] data.')
	train_mean = np.asarray(train_mean, dtype=np.float64).reshape(-1)
	train_std = np.asarray(train_std, dtype=np.float64).reshape(-1)
	if values.shape[2] != train_mean.size or train_mean.size != train_std.size:
		raise ValueError('Training statistics do not match the window feature count.')
	local_mean = values.mean(axis=1)
	slow_baseline = np.empty_like(local_mean)
	baseline = train_mean.copy()
	for index, current in enumerate(local_mean):
		baseline = decay * baseline + (1.0 - decay) * current
		slow_baseline[index] = baseline
	deviation = np.maximum(np.abs(local_mean - train_mean), np.abs(local_mean - slow_baseline))
	active = train_std > active_epsilon
	excess = np.maximum(deviation - 2.0 * train_std, 0.0)
	shift = excess ** 2
	shift[:, ~active] = 0.0
	return shift


def iqr_calibration_floor(train_residuals, reference_values, noise_scale=0.05):
	"""Protect POT calibration from collapsed clean-train residuals."""
	residuals = np.asarray(train_residuals, dtype=np.float64)
	reference = np.asarray(reference_values, dtype=np.float64)
	if residuals.ndim != 2 or reference.ndim != 2:
		raise ValueError('IQR calibration expects [time, channels] matrices.')
	if residuals.shape[1] != reference.shape[1]:
		raise ValueError('Calibration residuals and reference data must have the same channel count.')
	if residuals.shape[0] == 0:
		return residuals.copy(), np.zeros(residuals.shape[1], dtype=np.float64)
	q25, q75 = np.quantile(residuals, (0.25, 0.75), axis=0)
	# Use Tukey's lower fence as a robust lower bound. Adding a positive IQR
	# fraction here would shift the whole POT reference distribution and can
	# change a previously calibrated M0 threshold even when there is no collapse.
	iqr = np.maximum(q75 - q25, 0.0)
	robust_floor = q25 - 1.5 * iqr
	noise_floor = (noise_scale * np.std(reference, axis=0)) ** 2
	finite_scale = np.maximum(np.max(np.abs(residuals), axis=0), 1.0)
	floor = np.maximum.reduce((robust_floor, noise_floor, np.finfo(np.float64).eps * finite_scale))
	return np.maximum(residuals, floor[None, :]), floor


def _topk_channel_mean(residuals, k):
	values = np.asarray(residuals, dtype=np.float64)
	features = values.shape[1]
	k = min(max(int(k), 1), features)
	if k == features:
		return np.mean(values, axis=1)
	selected = np.partition(values, features - k, axis=1)[:, features - k:]
	return np.mean(selected, axis=1)


def monotonic_guarded_topk_aggregation(train_residuals, test_residuals, quantile=0.99):
	"""Choose sparse raw-residual Top-K only when its POT tail margin beats mean.

	The gate is selected without test labels. It protects the M3 mean score as
	the fallback, while making no claim that an unlabeled margin guarantees a
	classification-metric improvement.
	"""
	train = np.asarray(train_residuals, dtype=np.float64)
	test = np.asarray(test_residuals, dtype=np.float64)
	if train.ndim != 2 or test.ndim != 2:
		raise ValueError('Top-k aggregation expects [time, channels] residual matrices.')
	if train.shape[1] != test.shape[1] or train.shape[1] == 0:
		raise ValueError('Train and test residuals must have the same non-zero feature count.')
	if train.shape[0] == 0 or test.shape[0] == 0:
		raise ValueError('Top-k aggregation requires non-empty train and test residuals.')
	if not np.isfinite(train).all() or not np.isfinite(test).all():
		raise ValueError('Top-k aggregation requires finite residuals.')
	features = train.shape[1]
	from src.pot import pot_eval

	def candidate(k):
		train_score = _topk_channel_mean(train, k)
		test_score = _topk_channel_mean(test, k)
		threshold_result, _ = pot_eval(train_score, test_score, np.zeros(test_score.shape[0], dtype=np.int8))
		threshold = float(threshold_result['threshold'])
		tail = float(np.quantile(test_score, quantile))
		spread = float(np.quantile(train_score, 0.75) - np.quantile(train_score, 0.25))
		margin = (tail - threshold) / max(spread, SCORE_EPS)
		return train_score, test_score, threshold, margin

	baseline = candidate(features)
	selected_k = features
	selected = baseline
	for k in range(1, features):
		current = candidate(k)
		if current[3] > selected[3] + 1e-12:
			selected_k = k
			selected = current
	return selected[0], selected[1], selected_k, baseline[3], selected[3]


def kurtosis_gated_normalized_aggregation(train_residuals, test_residuals, topk=3, epsilon=SCORE_EPS):
	"""Backward-compatible alias for the unnormalized guarded Top-K scorer."""
	train_score, test_score, selected_k, _, _ = monotonic_guarded_topk_aggregation(
		train_residuals, test_residuals
	)
	return train_score, test_score, selected_k


def make_tranad_ablation_class(variant):
	if variant not in ABLATION_VARIANTS:
		raise AttributeError(f'Unknown TranAD ablation variant: {variant}')
	# Delayed import avoids a cycle while src.models is initializing.
	from src.models import TranAD

	class TranADProgressive(TranAD):
		def __init__(self, feats):
			super().__init__(feats)
			self.name = f'TranAD_{variant}'
			self.variant = variant
			self.variant_level = int(variant[1])
			conv_scale = 0.02 if feats == 1 else 0.1
			self.causal_embedding = MultiScaleCausalEmbedding(2 * feats, conv_scale) if self.variant_level >= 1 else None
			self.graph_attention = AdaptiveGatedGraphAttention(feats) if self.variant_level >= 2 else None
			self.prototype_memory = NormalPrototypeMemory(2 * feats) if self.variant_level >= 2 else None
			self.uses_monotonic_topk = self.variant_level >= 4

		def encode(self, src, c, tgt):
			if self.variant_level == 0:
				return super().encode(src, c, tgt)
			if self.graph_attention is not None:
				src = self.graph_attention(src)
				c = self.graph_attention(c)
			src = torch.cat((src, c), dim=2)
			if self.causal_embedding is not None:
				src = self.causal_embedding(src)
			src = src * math.sqrt(self.n_feats)
			src = self.pos_encoder(src)
			memory = self.transformer_encoder(src)
			if self.prototype_memory is not None:
				memory = self.prototype_memory(memory)
			tgt = tgt.repeat(1, 1, 2)
			return tgt, memory

	TranADProgressive.__name__ = f'TranAD_{variant}'
	TranADProgressive.__qualname__ = TranADProgressive.__name__
	return TranADProgressive


def is_tranad_ablation_name(name):
	return re.fullmatch(r'TranAD_M[0-4]', str(name)) is not None
