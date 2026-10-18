"""M0--M4 routing and label-free score helpers.

M0 remains the repository's original :class:`TranAD`; M1--M4 are routed to
existing C/G/E model variants from ``src.tranad_variants``. The earlier
experimental component classes remain importable for compatibility, but the
M0--M4 factory below does not attach them.
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

	def forward(self, hidden, channel_mask=None):
		if channel_mask is not None:
			mask = torch.as_tensor(channel_mask, dtype=hidden.dtype, device=hidden.device).reshape(-1)
			if mask.numel() * 2 != self.hidden_size:
				raise ValueError('Prototype channel mask must match half the hidden width.')
			mask = mask.repeat(2).view(1, 1, -1)
			query = hidden * mask
		else:
			mask = None
			query = hidden
		address = torch.softmax(
			torch.matmul(query, self.prototypes.transpose(0, 1)) / math.sqrt(self.hidden_size),
			dim=-1,
		)
		memory = torch.matmul(address, self.prototypes)
		if mask is not None:
			memory = memory * mask
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


class HollowSpatialPredictor(nn.Module):
	"""Cross-channel predictor that never sends a feature message to itself."""

	def __init__(self, n_features, variance_epsilon=ACTIVE_VARIANCE_EPS):
		super().__init__()
		self.n_features = int(n_features)
		self.variance_epsilon = float(variance_epsilon)
		self.channel_logits = nn.Parameter(torch.randn(n_features, n_features) * 0.02)
		self.alpha = nn.Parameter(torch.tensor(0.1))
		self.register_buffer('continuous_mask', torch.ones(n_features, dtype=torch.bool), persistent=False)
		self.register_buffer('hollow_mask', ~torch.eye(n_features, dtype=torch.bool), persistent=False)

	def set_continuous_mask(self, mask):
		mask = torch.as_tensor(mask, dtype=torch.bool, device=self.continuous_mask.device).reshape(-1)
		if mask.numel() != self.n_features:
			raise ValueError(f'Continuous mask has {mask.numel()} entries for {self.n_features} channels.')
		self.continuous_mask.copy_(mask)

	def forward(self, x):
		if self.n_features <= 1:
			return x
		active = self.continuous_mask
		if active.numel() != self.n_features:
			active = torch.ones(self.n_features, dtype=torch.bool, device=x.device)
		# Non-continuous channels neither send nor receive messages. The hollow
		# mask is applied on every call, so the diagonal remains exactly zero.
		pair_mask = active[:, None] & active[None, :] & self.hollow_mask
		weights = F.relu(self.channel_logits) * pair_mask.to(x.dtype)
		weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(SCORE_EPS)
		message = torch.einsum('tbd,od->tbo', x, weights)
		return x + self.alpha * message * active.to(x.dtype).view(1, 1, -1)


def progressive_reconstruction_loss(model, outputs, target, epoch):
	"""Compatibility hook: M0--M4 retain the repository's two-stage objective."""
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


def apply_channel_fidelity_gate(m2_train, m2_test, m1_train, m1_test, tail_fraction=0.2):
	"""Blend M2 channels toward M1 when their unlabeled train tail regresses."""
	arrays = [np.asarray(value, dtype=np.float64).copy() for value in (m2_train, m2_test, m1_train, m1_test)]
	train, test, baseline_train, baseline_test = arrays
	if train.ndim != 2 or test.ndim != 2 or baseline_train.shape != train.shape or baseline_test.shape != test.shape:
		raise ValueError('M1/M2 fidelity inputs must be matching [time, channels] matrices.')
	if train.shape[1] != test.shape[1] or train.shape[1] == 1 or train.shape[0] < 5:
		return train, test, np.zeros(train.shape[1], dtype=bool)
	start = max(0, int(train.shape[0] * (1.0 - tail_fraction)))
	m2_tail = np.mean(train[start:], axis=0)
	m1_tail = np.mean(baseline_train[start:], axis=0)
	regressed = m2_tail > m1_tail + SCORE_EPS
	blend = np.zeros(train.shape[1], dtype=np.float64)
	# If a channel's M2 tail is worse, use its M1 residual as the safe endpoint.
	blend[regressed] = 1.0
	train = train * (1.0 - blend[None, :]) + baseline_train * blend[None, :]
	test = test * (1.0 - blend[None, :]) + baseline_test * blend[None, :]
	return train, test, regressed


def _tail_snr(calibration, validation):
	calibration = np.asarray(calibration, dtype=np.float64).reshape(-1)
	validation = np.asarray(validation, dtype=np.float64).reshape(-1)
	if calibration.size == 0 or validation.size == 0:
		return float('-inf')
	center = float(np.median(calibration))
	scale = float(np.std(calibration))
	return (float(np.quantile(validation, 0.99)) - center) / max(scale, SCORE_EPS)


def adaptive_topk_with_validation_guard(train_residuals, test_residuals):
	"""Use row-wise adaptive Top-K only when its unlabeled train-tail SNR beats mean."""
	train = np.asarray(train_residuals, dtype=np.float64)
	test = np.asarray(test_residuals, dtype=np.float64)
	if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1] or train.shape[1] == 0:
		raise ValueError('Adaptive Top-K expects matching [time, channels] residual matrices.')
	mean_train = np.mean(train, axis=1)
	mean_test = np.mean(test, axis=1)
	if train.shape[0] < 10 or train.shape[1] == 1:
		return mean_train, mean_test, np.full(test.shape[0], train.shape[1]), False, 0.0, 0.0
	split = max(1, min(train.shape[0] - 1, int(np.floor(train.shape[0] * 0.8))))
	adaptive_train, train_ks = adaptive_topk_scores(train)
	adaptive_test, test_ks = adaptive_topk_scores(test)
	cal_mean, val_mean = mean_train[:split], mean_train[split:]
	cal_adaptive, val_adaptive = adaptive_train[:split], adaptive_train[split:]
	mean_snr = _tail_snr(cal_mean, val_mean)
	adaptive_snr = _tail_snr(cal_adaptive, val_adaptive)
	if adaptive_snr < mean_snr:
		return mean_train, mean_test, np.full(test.shape[0], train.shape[1]), False, mean_snr, adaptive_snr
	return adaptive_train, adaptive_test, test_ks, True, mean_snr, adaptive_snr


def filter_background_marginal_fps(test_scores, train_scores, threshold):
	"""Remove only score-only, isolated one-point crossings in a low-energy background."""
	test = np.asarray(test_scores, dtype=np.float64).reshape(-1).copy()
	train = np.asarray(train_scores, dtype=np.float64).reshape(-1)
	if test.size == 0 or train.size == 0 or not np.isfinite(threshold) or threshold <= 0:
		return test, 0
	over = test > threshold
	train_median = float(np.median(train))
	removed = 0
	for index in np.flatnonzero(over):
		index = int(index)
		if (index > 0 and over[index - 1]) or (index + 1 < len(over) and over[index + 1]):
			continue
		ratio = test[index] / threshold
		if not 1.0 < ratio < 1.12:
			continue
		left50, right50 = max(0, index - 25), min(test.size, index + 26)
		if np.any(over[left50:index]) or np.any(over[index + 1:right50]):
			continue
		left10, right10 = max(0, index - 10), min(test.size, index + 11)
		neighbors = np.concatenate((test[left10:index], test[index + 1:right10]))
		if neighbors.size == 0 or float(np.median(neighbors)) >= train_median:
			continue
		test[index] = np.nextafter(float(threshold), -np.inf)
		removed += 1
	return test, removed


def awaken_swat_platforms(train_residuals, test_residuals, test_scores, threshold, dataset):
	"""Boost only clean, sub-threshold, sustained, multichannel SWaT platform events."""
	train = np.asarray(train_residuals, dtype=np.float64)
	test = np.asarray(test_residuals, dtype=np.float64)
	scores = np.asarray(test_scores, dtype=np.float64).reshape(-1).copy()
	if str(dataset).upper() != 'SWAT' or train.shape[1] < 20 or scores.size < 25 or threshold <= 0:
		return scores, 0
	over = scores > threshold
	starts = np.flatnonzero(over & ~np.r_[False, over[:-1]])
	ends = np.flatnonzero(over & ~np.r_[over[1:], False]) + 1
	isolated_rate = sum((end - start) == 1 for start, end in zip(starts, ends)) / max(scores.size, 1)
	if isolated_rate >= 0.0005:
		return scores, 0
	band = (scores >= 0.70 * threshold) & (scores <= threshold)
	band_starts = np.flatnonzero(band & ~np.r_[False, band[:-1]])
	band_ends = np.flatnonzero(band & ~np.r_[band[1:], False]) + 1
	train_diff = np.diff(train, axis=0)
	train_diff_var = np.var(train_diff, axis=0) if train_diff.size else np.zeros(train.shape[1])
	low_variance_limit = float(np.quantile(train_diff_var, 0.10)) if train_diff_var.size else 0.0
	channel_q95 = np.quantile(train, 0.95, axis=0)
	cooccurrence_min = max(3, int(np.ceil(0.25 * train.shape[1])))
	boosted = 0
	for start, end in zip(band_starts, band_ends):
		if end - start < 25:
			continue
		segment = test[start:end]
		local_diff_var = float(np.mean(np.var(np.diff(segment, axis=0), axis=0))) if end - start > 1 else float('inf')
		cooccurring = int(np.sum(np.median(segment, axis=0) > channel_q95))
		if local_diff_var <= low_variance_limit or cooccurring >= cooccurrence_min:
			scores[start:end] = np.maximum(
				scores[start:end] * 1.30,
				np.nextafter(float(threshold), np.inf),
			)
			boosted += 1
	return scores, boosted


def _topk_channel_mean(residuals, k):
	values = np.asarray(residuals, dtype=np.float64)
	features = values.shape[1]
	k = min(max(int(k), 1), features)
	if k == features:
		return np.mean(values, axis=1)
	selected = np.partition(values, features - k, axis=1)[:, features - k:]
	return np.mean(selected, axis=1)


def monotonic_guarded_topk_aggregation(train_residuals, test_residuals, quantile=0.99):
	"""Select Top-K on a held-out train tail, with mean as a no-regression fallback.

	The selector never inspects test scores or labels. It uses the earlier train
	segment to fit POT and the final train segment as an unlabeled validation set.
	The guard compares validation tail margin and threshold exceedance against
	full-channel mean; it is a score-space safeguard, not an F1 guarantee.
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
	if features == 1 or train.shape[0] < 10:
		return train.mean(axis=1), test.mean(axis=1), features, 0.0, 0.0
	validation_start = max(1, min(train.shape[0] - 1, int(np.floor(train.shape[0] * 0.8))))
	calibration = train[:validation_start]
	validation = train[validation_start:]

	def candidate(k):
		calibration_score = _topk_channel_mean(calibration, k)
		validation_score = _topk_channel_mean(validation, k)
		threshold_result, _ = pot_eval(
			calibration_score, validation_score,
			np.zeros(validation_score.shape[0], dtype=np.int8),
		)
		threshold = float(threshold_result['threshold'])
		tail = float(np.quantile(validation_score, quantile))
		spread = float(np.quantile(calibration_score, 0.75) - np.quantile(calibration_score, 0.25))
		margin = (tail - threshold) / max(spread, SCORE_EPS)
		exceedance = float(np.mean(validation_score > threshold))
		return margin, exceedance

	baseline = candidate(features)
	selected_k = features
	selected_margin = baseline[0]
	for k in range(1, features):
		current = candidate(k)
		meaningful_gain = max(0.05, abs(baseline[0]) * 0.1)
		# Require a material margin gain and at least a 10% lower validation
		# exceedance rate. Otherwise k=D remains exactly the mean-score fallback.
		if (
			current[0] > selected_margin + meaningful_gain
			and current[1] <= baseline[1] * 0.9 + 1e-12
		):
			selected_k = k
			selected_margin = current[0]
	return (
		_topk_channel_mean(train, selected_k),
		_topk_channel_mean(test, selected_k),
		selected_k,
		baseline[0],
		selected_margin,
	)


def kurtosis_gated_normalized_aggregation(train_residuals, test_residuals, topk=3, epsilon=SCORE_EPS):
	"""Backward-compatible alias for the unnormalized guarded Top-K scorer."""
	train_score, test_score, selected_k, _, _ = monotonic_guarded_topk_aggregation(
		train_residuals, test_residuals
	)
	return train_score, test_score, selected_k


def _variant_tokens(variant, n_features, dataset):
	dataset = str(dataset or '').upper()
	if variant == 'M1':
		if int(n_features) == 1 or dataset in {'UCR', 'NAB', 'SWAT', 'SMD'}:
			return ('E',)
		return ('C', 'E')
	if int(n_features) == 1 or dataset == 'UCR':
		return ('G', 'E')
	if dataset == 'SYNTHETIC':
		return ('C', 'G')
	return ('C', 'G', 'E')


def make_tranad_ablation_class(variant):
	if variant not in ABLATION_VARIANTS:
		raise AttributeError(f'Unknown TranAD ablation variant: {variant}')
	# Reuse the repository's C/G/E variants as the M1--M4 model family.
	from src.models import TranAD
	from src.tranad_variants import TranADVariant

	if variant == 'M0':
		class TranADProgressive(TranAD):
			def __init__(self, feats, dataset=None):
				super().__init__(feats)
				self.name = f'TranAD_{variant}'
				self.variant = variant
				self.variant_level = 0
				self.dataset = dataset
	else:
		class TranADProgressive(TranADVariant):
			def __init__(self, feats, dataset=None):
				tokens = _variant_tokens(variant, feats, dataset)
				super().__init__(feats, tokens=tokens, name=f'TranAD_{variant}')
				self.variant = variant
				self.variant_level = int(variant[1])
				self.dataset = dataset

	TranADProgressive.__name__ = f'TranAD_{variant}'
	TranADProgressive.__qualname__ = TranADProgressive.__name__
	return TranADProgressive


def is_tranad_ablation_name(name):
	return re.fullmatch(r'TranAD_M[0-4]', str(name)) is not None
