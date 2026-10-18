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
ACTIVE_VARIANCE_EPS = 1e-5
SCORE_EPS = 1e-8


class MultiScaleCausalEmbedding(nn.Module):
	"""Parallel causal depthwise-separable convolutions with LayerScale fusion."""

	def __init__(self, channels):
		super().__init__()
		self.input_projection = nn.Linear(channels, channels)
		with torch.no_grad():
			self.input_projection.weight.copy_(torch.eye(channels))
			self.input_projection.bias.zero_()
		self.depthwise = nn.ModuleList([
			nn.Conv1d(channels, channels, kernel_size=k, groups=channels, bias=False)
			for k in (1, 3, 5)
		])
		self.pointwise = nn.ModuleList([
			nn.Conv1d(channels, channels, kernel_size=1)
			for _ in (1, 3, 5)
		])
		self.output_projection = nn.Conv1d(3 * channels, channels, kernel_size=1)
		self.layer_scale = nn.Parameter(torch.full((channels,), 1e-4))
		self.layer_norm = nn.LayerNorm(channels)

	def forward(self, x):
		# TranAD uses (time, batch, channels); Conv1d uses (batch, channels, time).
		base = self.input_projection(x)
		conv_input = x.permute(1, 2, 0)
		features = []
		for kernel, depthwise, pointwise in zip((1, 3, 5), self.depthwise, self.pointwise):
			# Padding only on the left ensures position t cannot see any sample > t.
			causal_input = F.pad(conv_input, (kernel - 1, 0)) if kernel > 1 else conv_input
			features.append(pointwise(F.gelu(depthwise(causal_input))))
		local = self.output_projection(torch.cat(features, dim=1)).permute(2, 0, 1)
		scale = self.layer_scale.view(1, 1, -1)
		return self.layer_norm(base + scale * local)


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
	"""Return the M3/M4 Huber, gated spectral, and trend training penalties."""
	if getattr(model, 'variant_level', 0) < 3 or not isinstance(outputs, tuple):
		zero = target.new_tensor(0.0)
		return zero, {
			'time_loss': 0.0,
			'freq_loss': 0.0,
			'freq_weight': 0.0,
			'trend_loss': 0.0,
		}
	ordered_target = target.squeeze(0)
	if ordered_target.shape[0] < 1:
		zero = target.new_tensor(0.0)
		return zero, {'time_loss': 0.0, 'freq_loss': 0.0, 'freq_weight': 0.0, 'trend_loss': 0.0}

	phase_weight = 1.0 / max(1, epoch + 1)
	weights = (phase_weight, 1.0 - phase_weight)
	time_loss = target.new_tensor(0.0)
	freq_loss = target.new_tensor(0.0)
	trend_loss = target.new_tensor(0.0)
	for weight, prediction in zip(weights, outputs):
		ordered_prediction = prediction.squeeze(0)
		time_loss = time_loss + weight * F.smooth_l1_loss(ordered_prediction, ordered_target)
		freq_loss = freq_loss + weight * _amplitude_spectrum_loss(ordered_target, ordered_prediction)
		if ordered_target.shape[0] > 1:
			target_delta = ordered_target[1:] - ordered_target[:-1]
			prediction_delta = ordered_prediction[1:] - ordered_prediction[:-1]
			trend_loss = trend_loss + weight * F.l1_loss(prediction_delta, target_delta)

	flatness = _spectral_flatness(ordered_target).to(device=target.device, dtype=target.dtype)
	alpha_0 = getattr(model, 'freq_loss_weight', 0.1)
	flatness_threshold = getattr(model, 'spectral_flatness_threshold', 0.3)
	flatness_steepness = getattr(model, 'spectral_flatness_steepness', 10.0)
	alpha_freq = alpha_0 * torch.sigmoid(flatness_steepness * (flatness_threshold - flatness))
	beta = getattr(model, 'trend_loss_weight', 0.1)
	extra_loss = time_loss + alpha_freq * freq_loss + beta * trend_loss
	return extra_loss, {
		'time_loss': float(time_loss.detach().cpu()),
		'freq_loss': float(freq_loss.detach().cpu()),
		'freq_weight': float(alpha_freq.detach().cpu()),
		'trend_loss': float(trend_loss.detach().cpu()),
	}


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


def kurtosis_gated_normalized_aggregation(train_residuals, test_residuals, topk=3, epsilon=SCORE_EPS):
	"""Standardize residual channels on train data and blend mean/top-k by kurtosis."""
	train = np.asarray(train_residuals, dtype=np.float64)
	test = np.asarray(test_residuals, dtype=np.float64)
	if train.ndim != 2 or test.ndim != 2:
		raise ValueError('Kurtosis-gated aggregation expects [time, channels] matrices.')
	if train.shape[1] != test.shape[1]:
		raise ValueError('Train and test residuals must have the same channel count.')
	if train.shape[1] == 0:
		raise ValueError('Kurtosis-gated aggregation requires at least one channel.')
	if train.shape[0] == 0 or test.shape[0] == 0:
		raise ValueError('Kurtosis-gated aggregation requires non-empty train and test residuals.')

	mean = np.mean(train, axis=0, dtype=np.float64)
	std = np.std(train, axis=0, dtype=np.float64)
	denominator = std + epsilon
	train_z = (train - mean) / denominator
	test_z = (test - mean) / denominator
	k = min(max(int(topk), 1), train.shape[1])

	def aggregate(z):
		row_mean = np.mean(z, axis=1)
		centered = z - row_mean[:, None]
		variance = np.mean(centered ** 2, axis=1)
		fourth_moment = np.mean(centered ** 4, axis=1)
		kurtosis = np.zeros_like(variance)
		valid = variance > epsilon
		kurtosis[valid] = fourth_moment[valid] / (variance[valid] ** 2) - 3.0
		# Center the sigmoid at zero excess kurtosis: plain sigmoid(0)=0.5 would
		# retain a half-strength top-k term on Gaussian background residuals.
		sigmoid = 1.0 / (1.0 + np.exp(-np.clip(kurtosis, -60.0, 60.0)))
		gate = np.maximum(2.0 * sigmoid - 1.0, 0.0)
		largest = np.partition(z, z.shape[1] - k, axis=1)[:, -k:]
		topk_mean = np.mean(largest, axis=1)
		score = (1.0 - gate) * row_mean + gate * topk_mean
		return score

	return aggregate(train_z), aggregate(test_z), k


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
			self.freq_loss_weight = 0.1
			self.spectral_flatness_threshold = 0.3
			self.spectral_flatness_steepness = 10.0
			self.trend_loss_weight = 0.1
			self.causal_embedding = MultiScaleCausalEmbedding(2 * feats) if self.variant_level >= 1 else None
			self.graph_attention = AdaptiveGatedGraphAttention(feats) if self.variant_level >= 2 else None
			self.uses_kurtosis_aggregation = self.variant_level >= 4

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
			tgt = tgt.repeat(1, 1, 2)
			return tgt, memory

	TranADProgressive.__name__ = f'TranAD_{variant}'
	TranADProgressive.__qualname__ = TranADProgressive.__name__
	return TranADProgressive


def is_tranad_ablation_name(name):
	return re.fullmatch(r'TranAD_M[0-4]', str(name)) is not None
