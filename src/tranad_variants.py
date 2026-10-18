import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder
from torch.nn import TransformerEncoder

from src.constants import lr
from src.dlutils import PositionalEncoding, TransformerDecoderLayer, TransformerEncoderLayer


VARIANT_TOKENS = {'R', 'V', 'A', 'M', 'C', 'G', 'E'}


def _choose_heads(embed_dim, preferred):
	for heads in range(min(embed_dim, preferred), 0, -1):
		if embed_dim % heads == 0:
			return heads
	return 1


class ReversibleInstanceNorm(nn.Module):
	def __init__(self, n_feats, eps=1e-5, affine=True):
		super().__init__()
		self.eps = eps
		self.affine = affine
		if affine:
			self.weight = nn.Parameter(torch.ones(1, 1, n_feats))
			self.bias = nn.Parameter(torch.zeros(1, 1, n_feats))
		self._mean = None
		self._std = None

	def norm(self, x):
		self._mean = x.mean(dim=0, keepdim=True).detach()
		self._std = torch.sqrt(x.var(dim=0, keepdim=True, unbiased=False) + self.eps).detach()
		return self.normalize(x)

	def normalize(self, x):
		x = (x - self._mean) / self._std
		if self.affine:
			x = x * self.weight + self.bias
		return x

	def denorm(self, x):
		if self._mean is None or self._std is None:
			return x
		if self.affine:
			x = (x - self.bias) / (self.weight + self.eps)
		return x * self._std + self._mean


class VariableTemporalMixer(nn.Module):
	def __init__(self, n_feats, n_window, dropout=0.1):
		super().__init__()
		time_heads = _choose_heads(n_feats, n_feats)
		variable_heads = _choose_heads(n_window, 4)
		self.temporal_attn = nn.MultiheadAttention(n_feats, time_heads, dropout=dropout)
		self.variable_attn = nn.MultiheadAttention(n_window, variable_heads, dropout=dropout)
		self.temporal_norm = nn.LayerNorm(n_feats)
		self.variable_norm = nn.LayerNorm(n_window)
		self.gate = nn.Parameter(torch.tensor(0.0))
		self.dropout = nn.Dropout(dropout)

	def forward(self, src):
		time_out, _ = self.temporal_attn(src, src, src)
		time_out = self.temporal_norm(src + self.dropout(time_out))

		var_in = src.permute(2, 1, 0)
		var_out, _ = self.variable_attn(var_in, var_in, var_in)
		var_out = self.variable_norm(var_in + self.dropout(var_out)).permute(2, 1, 0)

		gate = torch.sigmoid(self.gate)
		return gate * time_out + (1.0 - gate) * var_out


class TemporalConvAdapter(nn.Module):
	def __init__(self, n_feats, dropout=0.1):
		super().__init__()
		self.depthwise = nn.Conv1d(n_feats, n_feats, kernel_size=3, padding=1, groups=n_feats)
		self.pointwise = nn.Conv1d(n_feats, n_feats, kernel_size=1)
		self.norm = nn.LayerNorm(n_feats)
		self.dropout = nn.Dropout(dropout)
		self.gate = nn.Parameter(torch.tensor(-2.0))

	def forward(self, src):
		x = src.permute(1, 2, 0)
		x = self.pointwise(F.gelu(self.depthwise(x))).permute(2, 0, 1)
		return self.norm(src + torch.sigmoid(self.gate) * self.dropout(x))


class FeatureGate(nn.Module):
	def __init__(self, n_feats):
		super().__init__()
		hidden = max(4, n_feats // 2)
		self.net = nn.Sequential(
			nn.Linear(2 * n_feats, hidden),
			nn.GELU(),
			nn.Linear(hidden, n_feats),
		)
		self.strength = nn.Parameter(torch.tensor(-2.0))

	def forward(self, src):
		x = src.permute(1, 0, 2)
		stats = torch.cat([x.mean(dim=1), x.std(dim=1, unbiased=False)], dim=-1)
		gate = torch.sigmoid(self.net(stats)).unsqueeze(0)
		return src * (1.0 + torch.sigmoid(self.strength) * (gate - 0.5))


class PhaseOutputBlend(nn.Module):
	def __init__(self):
		super().__init__()
		self.logit = nn.Parameter(torch.tensor(2.0))

	def forward(self, x1, x2):
		weight = torch.sigmoid(self.logit)
		return (1.0 - weight) * x1 + weight * x2


class AssociationDiscrepancy(nn.Module):
	def __init__(self, n_feats, n_window, eps=1e-6):
		super().__init__()
		self.query = nn.Linear(n_feats, n_feats)
		self.key = nn.Linear(n_feats, n_feats)
		self.sigma = nn.Linear(n_feats, 1)
		self.eps = eps
		dist = torch.arange(n_window, dtype=torch.float32)
		dist = torch.abs(dist[:, None] - dist[None, :])
		self.register_buffer('distances', dist)

	def forward(self, src):
		x = src.permute(1, 0, 2)
		q = self.query(x)
		k = self.key(x)
		series = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / math.sqrt(x.shape[-1]), dim=-1)

		sigma = F.softplus(self.sigma(x)) + self.eps
		distances = self.distances.to(device=x.device, dtype=x.dtype)
		prior = torch.exp(-(distances.unsqueeze(0) ** 2) / (2.0 * sigma.pow(2)))
		prior = prior / (prior.sum(dim=-1, keepdim=True) + self.eps)

		series = series.clamp_min(self.eps)
		prior = prior.clamp_min(self.eps)
		kl_sp = (series * (series.log() - prior.log())).sum(dim=-1)
		kl_ps = (prior * (prior.log() - series.log())).sum(dim=-1)
		return 0.5 * (kl_sp + kl_ps).mean()


class SparseMemoryRefiner(nn.Module):
	def __init__(self, n_feats, n_memory=32, shrink=0.0025):
		super().__init__()
		self.n_memory = n_memory
		self.shrink = shrink
		self.memory = nn.Parameter(F.normalize(torch.randn(n_memory, n_feats), dim=-1))
		self.refine = nn.Sequential(
			nn.Linear(2 * n_feats, n_feats),
			nn.Sigmoid(),
		)
		self.gate = nn.Parameter(torch.tensor(0.0))

	def forward(self, pred):
		query = pred[0]
		memory = F.normalize(self.memory, dim=-1)
		attn = torch.softmax(torch.matmul(query, memory.t()) / math.sqrt(query.shape[-1]), dim=-1)
		if self.shrink > 0:
			attn = F.relu(attn - self.shrink) * attn / (torch.abs(attn - self.shrink) + 1e-12)
			attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-12)
		read = torch.matmul(attn, memory)
		refined = self.refine(torch.cat([query, read], dim=-1)).unsqueeze(0)
		gate = torch.sigmoid(self.gate)
		entropy = -(attn.clamp_min(1e-12) * attn.clamp_min(1e-12).log()).sum(dim=-1).mean()
		return (1.0 - gate) * pred + gate * refined, entropy


class TranADVariant(nn.Module):
	def __init__(self, feats, tokens, name):
		super().__init__()
		self.name = name
		self.tokens = tuple(tokens)
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.n_window = 10
		self.n = self.n_feats * self.n_window
		self._aux_losses = {}

		self.revin = ReversibleInstanceNorm(feats) if 'R' in self.tokens else None
		self.conv_adapter = TemporalConvAdapter(feats) if 'C' in self.tokens else None
		self.feature_gate = FeatureGate(feats) if 'G' in self.tokens else None
		self.mixer = VariableTemporalMixer(feats, self.n_window) if 'V' in self.tokens else None
		self.association = AssociationDiscrepancy(feats, self.n_window) if 'A' in self.tokens else None
		self.memory = SparseMemoryRefiner(feats) if 'M' in self.tokens else None
		self.phase_blend = PhaseOutputBlend() if 'E' in self.tokens else None

		self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
		decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

	def _reset_aux(self):
		self._aux_losses = {}

	def auxiliary_losses(self):
		return self._aux_losses

	def auxiliary_loss(self):
		if not self._aux_losses:
			return next(self.parameters()).new_tensor(0.0)
		return sum(self._aux_losses.values())

	def encode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		tgt = tgt.repeat(1, 1, 2)
		return tgt, memory

	def forward(self, src, tgt):
		self._reset_aux()
		if self.revin is not None:
			src = self.revin.norm(src)
			tgt = self.revin.normalize(tgt)

		if self.conv_adapter is not None:
			src = self.conv_adapter(src)

		if self.feature_gate is not None:
			src = self.feature_gate(src)

		if self.mixer is not None:
			src = self.mixer(src)

		if self.association is not None:
			self._aux_losses['association'] = self.association(src)

		c = torch.zeros_like(src)
		x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
		c = (x1 - src) ** 2
		x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))

		if self.phase_blend is not None:
			x2 = self.phase_blend(x1, x2)

		if self.memory is not None:
			x2, entropy = self.memory(x2)
			self._aux_losses['memory_entropy'] = entropy

		if self.revin is not None:
			x1 = self.revin.denorm(x1)
			x2 = self.revin.denorm(x2)

		return x1, x2


def is_tranad_variant_name(name):
	if not name.startswith('TranAD_'):
		return False
	tokens = name.split('_')[1:]
	return bool(tokens) and all(token in VARIANT_TOKENS for token in tokens)


def make_tranad_variant_class(name):
	tokens = tuple(name.split('_')[1:])
	if not is_tranad_variant_name(name):
		raise AttributeError(name)

	class _Variant(TranADVariant):
		def __init__(self, feats):
			super().__init__(feats, tokens=tokens, name=name)

	_Variant.__name__ = name
	_Variant.__qualname__ = name
	return _Variant
