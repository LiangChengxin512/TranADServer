"""Progressive M0--M4 TranAD ablation extensions.

The original :class:`TranAD` implementation remains the M0 reference.  The
modules in this file are only attached to the generated ablation classes, so
existing TranAD checkpoints and entrypoints keep their original architecture.
"""

import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ABLATION_VARIANTS = {f'M{i}' for i in range(5)}


class MultiScaleCausalEmbedding(nn.Module):
	"""Identity-residual embedding with normalized, strictly causal conv features."""

	def __init__(self, channels):
		super().__init__()
		self.projection = nn.Linear(channels, channels)
		with torch.no_grad():
			self.projection.weight.copy_(torch.eye(channels))
			self.projection.bias.zero_()
		self.depthwise = nn.ModuleList([
			nn.Conv1d(channels, channels, kernel_size=k, groups=channels, bias=False)
			for k in (1, 3, 5)
		])
		self.pointwise = nn.ModuleList([
			nn.Conv1d(channels, channels, kernel_size=1)
			for _ in (1, 3, 5)
		])
		self.fusion = nn.Conv1d(3 * channels, channels, kernel_size=1)
		self.layer_scale = nn.Parameter(torch.full((channels,), 1e-4))
		self.norm = nn.LayerNorm(channels)

	def forward(self, x):
		# TranAD uses (time, batch, channels); Conv1d uses (batch, channels, time).
		base = self.projection(x)
		conv_input = x.permute(1, 2, 0)
		features = []
		for kernel, depthwise, pointwise in zip((1, 3, 5), self.depthwise, self.pointwise):
			# Left-only padding prevents a position from seeing future samples.
			causal_input = F.pad(conv_input, (kernel - 1, 0)) if kernel > 1 else conv_input
			features.append(pointwise(F.gelu(depthwise(causal_input))))
		local = self.fusion(torch.cat(features, dim=1)).permute(2, 0, 1)
		# Normalize the local branch before LayerScale.  This keeps the baseline
		# projection intact at initialization, including for one-feature inputs.
		return base + self.layer_scale.view(1, 1, -1) * self.norm(local)


class AdaptiveGatedGraphAttention(nn.Module):
	"""Feature-node attention with an explicit univariate identity short-circuit."""

	def __init__(self, n_features, hidden=None):
		super().__init__()
		self.n_features = n_features
		hidden = hidden or max(4, min(32, n_features * 2))
		self.node_embedding = nn.Parameter(torch.randn(n_features, hidden) * 0.02)
		self.signal_projection = nn.Linear(1, hidden, bias=False)
		self.query = nn.Linear(hidden, hidden, bias=False)
		self.key = nn.Linear(hidden, hidden, bias=False)
		self.value = nn.Linear(1, hidden, bias=False)
		self.output = nn.Linear(hidden, 1, bias=False)
		self.adjacency_logits = nn.Parameter(torch.zeros(n_features, n_features))
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
			# A graph over one node is degenerate; preserve the M1 representation.
			return x
		time, batch, features = x.shape
		nodes = x.reshape(time * batch, features, 1)
		global_pool = x.mean(dim=0).mean(dim=0)
		gamma = torch.sigmoid(self.gate(global_pool)).view(1, 1, 1)
		node_ids = self.node_embedding.unsqueeze(0)
		embedded = node_ids + self.signal_projection(nodes)
		query = self.query(embedded)
		key = self.key(embedded)
		attention_logits = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(query.shape[-1])
		attention_logits = attention_logits + self.adjacency_logits.unsqueeze(0)
		attention = torch.softmax(attention_logits, dim=-1)
		messages = torch.matmul(attention, self.value(nodes))
		graph_features = self.output(messages).reshape(time, batch, features)
		return (1.0 - gamma) * x + gamma * graph_features


def fft_amplitude_loss(target, prediction):
	"""Compare amplitude spectra across consecutive points in a training batch.

	Sliding-window batches preserve time order in this repository.  The leading
	(singleton) TranAD output axis is removed, then rFFT runs along that ordered
	batch/time axis.  MPS implementations without FFT support transparently use
	CPU; the device copy keeps autograd connected to the model output.
	"""
	target = target.squeeze(0)
	prediction = prediction.squeeze(0)
	if target.shape[0] < 2:
		return prediction.sum() * 0.0
	device = target.device
	if device.type == 'mps':
		target_for_fft = target.to('cpu')
		prediction_for_fft = prediction.to('cpu')
	else:
		target_for_fft = target
		prediction_for_fft = prediction
	try:
		target_spectrum = torch.fft.rfft(target_for_fft, dim=0).abs()
		prediction_spectrum = torch.fft.rfft(prediction_for_fft, dim=0).abs()
	except (RuntimeError, NotImplementedError):
		target_spectrum = torch.fft.rfft(target.to('cpu'), dim=0).abs()
		prediction_spectrum = torch.fft.rfft(prediction.to('cpu'), dim=0).abs()
	loss = (target_spectrum - prediction_spectrum).abs().sum(dim=0) / target.shape[0]
	return loss.mean().to(device=device)


def focal_mse_loss(target, prediction):
	error = (target - prediction).abs()
	return ((1.0 + torch.tanh(error)) * error.square()).mean()


def progressive_reconstruction_loss(model, outputs, target, epoch):
	"""Return M3/M4 FFT and focal terms for TranAD's two reconstruction phases."""
	if getattr(model, 'variant_level', 0) < 3 or not isinstance(outputs, tuple):
		zero = target.new_tensor(0.0)
		return zero, {'fft_loss': 0.0, 'focal_loss': 0.0}
	phase_weight = 1.0 / max(1, epoch + 1)
	weights = (phase_weight, 1.0 - phase_weight)
	fft = sum(weight * fft_amplitude_loss(target, pred) for weight, pred in zip(weights, outputs))
	focal = sum(weight * focal_mse_loss(target, pred) for weight, pred in zip(weights, outputs))
	alpha = getattr(model, 'fft_loss_weight', 0.1)
	beta = getattr(model, 'focal_loss_weight', 0.1)
	return alpha * fft + beta * focal, {
		'fft_loss': float(fft.detach().cpu()),
		'focal_loss': float(focal.detach().cpu()),
	}


def adaptive_topk_scores(residuals):
	"""Average a variance-selected number of largest channel residuals per step.

	The channel count is determined from each row's residual coefficient of
	variation and never consults anomaly labels.  Higher cross-channel spread
	selects fewer channels, while uniform residuals select all channels.
	"""
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
			self.fft_loss_weight = 0.1
			self.focal_loss_weight = 0.1
			self.causal_embedding = MultiScaleCausalEmbedding(2 * feats) if self.variant_level >= 1 else None
			self.graph_attention = AdaptiveGatedGraphAttention(feats) if self.variant_level >= 2 else None
			self.uses_adaptive_topk = self.variant_level >= 4

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
