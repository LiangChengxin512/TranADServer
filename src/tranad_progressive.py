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


def phase_concordant_residuals(phase1_residuals, phase2_residuals, lower=0.85, upper=1.25):
	"""Sharpen phase-two absolute error using its instantaneous phase ratio."""
	phase1 = np.asarray(phase1_residuals, dtype=np.float64)
	phase2 = np.asarray(phase2_residuals, dtype=np.float64)
	if phase1.shape != phase2.shape or phase1.ndim != 2:
		raise ValueError('Phase residuals must have matching [time, channels] shapes.')
	if not 0.0 < lower <= upper:
		raise ValueError('Concordance ratio bounds must satisfy 0 < lower <= upper.')
	ratio = (phase2 + SCORE_EPS) / (phase1 + SCORE_EPS)
	return phase2 * np.clip(ratio, lower, upper)


def zero_smearing_spike_filter(residuals, shrinkage=0.25, isolation_ratio=1.5):
	"""Shrink only isolated high points using a trailing three-point second max.

	The output at time t depends only on t and its two predecessors, so a high
	point never alters any later score. Two-point-or-shorter inputs pass through.
	"""
	values = np.asarray(residuals, dtype=np.float64)
	if values.ndim != 2:
		raise ValueError('Spike filtering expects a [time, channels] matrix.')
	if not 0.0 <= shrinkage <= 1.0 or isolation_ratio <= 1.0:
		raise ValueError('Use shrinkage in [0, 1] and an isolation ratio above 1.')
	filtered = values.copy()
	for index in range(2, values.shape[0]):
		second = np.maximum(values[index - 1], values[index - 2])
		isolated = values[index] > isolation_ratio * np.maximum(second, SCORE_EPS)
		filtered[index, isolated] = (
			(1.0 - shrinkage) * values[index, isolated]
			+ shrinkage * second[isolated]
		)
	return filtered


def anchor_residual_scale(residuals, reference, best_epoch, anchor_epoch=5):
	"""Restore init-score scale toward epoch five when a later checkpoint shrinks it."""
	values = np.asarray(residuals, dtype=np.float64)
	if values.ndim != 2:
		raise ValueError('Residual scale anchoring expects [time, channels] values.')
	if reference is None or int(best_epoch or 0) <= anchor_epoch:
		return values.copy(), False
	ref_mean = np.asarray(reference.get('mean', []), dtype=np.float64).reshape(-1)
	ref_std = np.asarray(reference.get('std', []), dtype=np.float64).reshape(-1)
	if ref_mean.size != values.shape[1] or ref_std.size != values.shape[1]:
		return values.copy(), False
	best_mean = np.mean(values, axis=0)
	best_std = np.std(values, axis=0)
	anchored = ref_mean[None, :] + (ref_std / (best_std + SCORE_EPS))[None, :] * (
		values - best_mean[None, :]
	)
	anchored = np.nan_to_num(anchored, nan=0.0, posinf=np.finfo(np.float64).max, neginf=0.0)
	return np.maximum(values, anchored), True


def _pot_tail_profile(calibration_score, validation_score, quantile=0.99):
	"""Return unlabeled POT tail margin and validation exceedance rate."""
	from src.pot import pot_eval
	calibration_score = np.asarray(calibration_score, dtype=np.float64).reshape(-1)
	validation_score = np.asarray(validation_score, dtype=np.float64).reshape(-1)
	result, _ = pot_eval(
		calibration_score,
		validation_score,
		np.zeros(validation_score.size, dtype=np.int8),
	)
	threshold = float(result['threshold'])
	if not np.isfinite(threshold):
		return float('-inf'), float('inf')
	tail = float(np.quantile(validation_score, quantile))
	spread = float(np.quantile(calibration_score, 0.75) - np.quantile(calibration_score, 0.25))
	margin = (tail - threshold) / max(spread, SCORE_EPS)
	exceedance = float(np.mean(validation_score > threshold))
	return margin, exceedance


def phase_concordance_guard(candidate_residuals, baseline_residuals, quantile=0.99):
	"""Enable M3 sharpening only when a held-out train tail supports it."""
	candidate = np.asarray(candidate_residuals, dtype=np.float64)
	baseline = np.asarray(baseline_residuals, dtype=np.float64)
	if candidate.shape != baseline.shape or candidate.ndim != 2:
		raise ValueError('M3 candidate and baseline residuals must have matching 2D shapes.')
	if candidate.shape[0] < 10 or candidate.shape[1] == 0:
		return False, 0.0
	split = max(1, min(candidate.shape[0] - 1, int(np.floor(candidate.shape[0] * 0.8))))
	base_profile = _pot_tail_profile(
		baseline[:split].mean(axis=1), baseline[split:].mean(axis=1), quantile
	)
	candidate_profile = _pot_tail_profile(
		candidate[:split].mean(axis=1), candidate[split:].mean(axis=1), quantile
	)
	margin_gain = candidate_profile[0] - base_profile[0]
	meaningful_gain = max(0.05, abs(base_profile[0]) * 0.1)
	use_candidate = (
		np.isfinite(margin_gain)
		and margin_gain > meaningful_gain
		and candidate_profile[1] <= base_profile[1] + 1e-12
	)
	return bool(use_candidate), float(margin_gain) if np.isfinite(margin_gain) else 0.0


def select_phase_concordance_scores(
		train_candidate, train_baseline, test_candidate, test_baseline, quantile=0.99):
	"""Apply one train-only M3 gate consistently to calibration and test scores."""
	use_candidate, margin_gain = phase_concordance_guard(
		train_candidate, train_baseline, quantile=quantile
	)
	if use_candidate:
		return train_candidate, test_candidate, True, margin_gain
	return train_baseline, test_baseline, False, margin_gain


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
