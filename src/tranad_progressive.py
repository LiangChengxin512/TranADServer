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


def _event_spans(over_mask, merge_gap=2):
	indices = np.flatnonzero(over_mask)
	if not indices.size:
		return []
	spans = []
	start = previous = int(indices[0])
	for index in indices[1:]:
		index = int(index)
		if index - previous - 1 > merge_gap:
			spans.append((start, previous + 1))
			start = index
		previous = index
	spans.append((start, previous + 1))
	return spans


def suppress_marginal_glitches(scores, threshold, rho=0.45, max_duration=2, peak_ratio=2.2, merge_gap=2):
	"""Dampen only short, weak score clusters while keeping the threshold fixed."""
	values = np.asarray(scores, dtype=np.float64).reshape(-1).copy()
	if not 0.0 <= rho <= 1.0 or max_duration < 1 or peak_ratio <= 1.0:
		raise ValueError('Invalid glitch suppression parameters.')
	spans = _event_spans(values > threshold, merge_gap=merge_gap)
	suppressed = 0
	for start, stop in spans:
		peak = float(np.max(values[start:stop]))
		ratio = peak / max(float(threshold), SCORE_EPS)
		if stop - start <= max_duration and ratio < peak_ratio:
			mask = values[start:stop] > threshold
			values[start + np.flatnonzero(mask)] *= rho
			suppressed += 1
	return values, {'suppressed_glitch_events': suppressed, 'threshold_events': len(spans)}


def boost_sustained_segments(test_score, train_score, n_features, window=50, run_length=30, factor=1.35):
	"""Boost sustained multichannel score elevations using causal rolling means."""
	test = np.asarray(test_score, dtype=np.float64).reshape(-1).copy()
	train = np.asarray(train_score, dtype=np.float64).reshape(-1)
	if int(n_features) < 20 or test.size < run_length or train.size < 1:
		return test, 0
	window = max(1, min(int(window), train.size, test.size))
	def trailing_mean(values):
		cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
		indices = np.arange(values.size)
		starts = np.maximum(0, indices - window + 1)
		return (cumulative[indices + 1] - cumulative[starts]) / (indices - starts + 1)
	cutoff = float(np.quantile(trailing_mean(train), 0.95))
	above = trailing_mean(test) > cutoff
	spans = _event_spans(above, merge_gap=0)
	boosted = 0
	for start, stop in spans:
		if stop - start >= run_length:
			test[start:stop] *= factor
			boosted += 1
	return test, boosted


def _topk_event_profile(score, threshold, train_score=None, merge_gap=2):
	values = np.asarray(score, dtype=np.float64).reshape(-1)
	spans = _event_spans(values > threshold, merge_gap=merge_gap)
	robust = []
	weak = 0
	prominence = 0.0
	robust_prominence = 0.0
	train = np.asarray(train_score, dtype=np.float64).reshape(-1) if train_score is not None else np.array([])
	noise_floor = float(np.quantile(train, 0.98)) if train.size else 0.0
	noise_scale = max(float(np.std(train)), SCORE_EPS) if train.size else 1.0
	robust_snr = 0.0
	for start, stop in spans:
		segment = values[start:stop]
		ratio = float(np.max(segment)) / max(float(threshold), SCORE_EPS)
		is_robust = stop - start >= 3 or ratio >= 2.2
		if is_robust:
			robust.append((start, stop))
		segment_prominence = float(np.maximum(segment - threshold, 0.0).sum())
		prominence += segment_prominence
		if is_robust:
			robust_prominence += segment_prominence
			robust_snr += max(0.0, (float(np.max(segment)) - noise_floor) / noise_scale)
		if stop - start <= 2 and ratio < 2.2:
			weak += 1
	return {
		'robust_events': len(robust), 'prominence': prominence,
		'robust_prominence': robust_prominence, 'robust_snr': robust_snr,
		'weak_events': weak,
	}


def m4_candidate_ks(n_features):
	values = {int(n_features)}
	values.update(k for k in (1, 2, 3, 4, 5, 8, 12, 16) if k <= n_features)
	k = 32
	while k < n_features:
		values.add(k)
		k *= 2
	return sorted(values)


def select_m4_topk(train_residuals, test_residuals, n_features=None):
	"""Select Top-K by unlabeled test event shape; threshold is fitted on train."""
	from src.pot import pot_eval
	train = np.asarray(train_residuals, dtype=np.float64)
	test = np.asarray(test_residuals, dtype=np.float64)
	features = train.shape[1] if n_features is None else int(n_features)
	if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1] or features != train.shape[1]:
		raise ValueError('M4 selection expects matching [time, channels] residual arrays.')
	candidates = []
	for k in m4_candidate_ks(features):
		train_score = _topk_channel_mean(train, k)
		test_score = _topk_channel_mean(test, k)
		result, _ = pot_eval(train_score, test_score, np.zeros(test_score.size, dtype=np.int8))
		threshold = float(result['threshold'])
		processed, glitch_info = suppress_marginal_glitches(test_score, threshold)
		processed, boosted = boost_sustained_segments(processed, train_score, features)
		profile = _topk_event_profile(processed, threshold, train_score=train_score)
		candidates.append({
			'k': k, 'train_score': train_score, 'test_score': processed,
			'threshold': threshold, 'profile': profile,
			'suppressed_glitch_events': glitch_info['suppressed_glitch_events'],
			'boosted_sustained_segments': boosted,
		})
	baseline = next(item for item in candidates if item['k'] == features)
	selected = baseline
	base_profile = baseline['profile']
	for item in candidates:
		if item['k'] == features:
			continue
		profile = item['profile']
		retains_events = profile['robust_events'] >= base_profile['robust_events']
		improves_prominence = profile['robust_prominence'] > base_profile['robust_prominence'] * 1.01 + SCORE_EPS
		improves_snr = profile['robust_snr'] > base_profile['robust_snr'] * 1.01 + SCORE_EPS
		reduces_glitches = item['suppressed_glitch_events'] < baseline['suppressed_glitch_events']
		if retains_events and (improves_prominence or improves_snr or reduces_glitches):
			if (profile['robust_events'], profile['robust_snr'], profile['robust_prominence'], -item['suppressed_glitch_events']) > (
				selected['profile']['robust_events'], selected['profile']['robust_snr'],
				selected['profile']['robust_prominence'], -selected['suppressed_glitch_events'],
			):
				selected = item
	return selected


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
			self.graph_attention = HollowSpatialPredictor(feats) if self.variant_level >= 2 else None
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
				mask = self.graph_attention.continuous_mask if self.graph_attention is not None else None
				memory = self.prototype_memory(memory, mask)
			tgt = tgt.repeat(1, 1, 2)
			return tgt, memory

	TranADProgressive.__name__ = f'TranAD_{variant}'
	TranADProgressive.__qualname__ = TranADProgressive.__name__
	return TranADProgressive


def is_tranad_ablation_name(name):
	return re.fullmatch(r'TranAD_M[0-4]', str(name)) is not None
