from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
from tensordict import TensorDict


# Discrete categorical action space — per-instrument logits over [min_trade, ..., max_trade]
# This avoids the saturation/bistability issues that arise with tanh-Gaussian over integer actions.


@dataclass(frozen=True)
class StructuredActionSpace:
    """Per-instrument allowed trade deltas as an explicit (sorted, deduped) list. Earlier
    versions used (min, max) integer ranges; this is now stored as `trade_deltas` with the
    integer range expanded out — non-uniform spacings are allowed (e.g. fine 1-step bins
    near zero plus coarse 5-step bins at the extremes).

    Construction accepts either:
      - `trade_deltas`: sequence of per-instrument iterables of allowed integer deltas, or
      - `min_trade_delta` + `max_trade_delta`: legacy integer ranges that get expanded to
        unit-step lists for backward compatibility.

    `min_trade_delta` and `max_trade_delta` remain available as derived properties for
    callers that only need the bounds.
    """

    instrument_order: tuple
    trade_deltas: tuple  # tuple[tuple[int, ...]] — sorted, deduped, contains 0

    def __init__(self, instrument_order, trade_deltas=None,
                 min_trade_delta=None, max_trade_delta=None):
        object.__setattr__(self, "instrument_order", tuple(instrument_order))
        if trade_deltas is not None:
            normalized = tuple(
                tuple(sorted(set(int(d) for d in deltas))) for deltas in trade_deltas)
        else:
            assert min_trade_delta is not None and max_trade_delta is not None, (
                "StructuredActionSpace requires either trade_deltas or "
                "(min_trade_delta + max_trade_delta)")
            import warnings
            warnings.warn(
                "StructuredActionSpace(min_trade_delta=, max_trade_delta=) is deprecated; "
                "pass `trade_deltas=` (per-instrument list of allowed integer deltas) "
                "instead. The min/max form expands to a uniform unit-step range and is "
                "scheduled for removal. Reachable today only when loading a pre-schema-v? "
                "policy artifact.",
                DeprecationWarning, stacklevel=2,
            )
            normalized = tuple(
                tuple(range(int(lo), int(hi) + 1))
                for lo, hi in zip(min_trade_delta, max_trade_delta))
        # Validate: each instrument's list must be non-empty and contain 0 (no-trade bin
        # is always required so the position-feasibility mask never fully masks a row).
        for name, deltas in zip(self.instrument_order, normalized):
            if not deltas:
                raise ValueError(f"trade_deltas for instrument {name!r} is empty")
            if 0 not in deltas:
                raise ValueError(
                    f"trade_deltas for instrument {name!r} must include 0; got {deltas}")
        object.__setattr__(self, "trade_deltas", normalized)

    @property
    def dimension(self):
        return len(self.instrument_order)

    @property
    def min_trade_delta(self):
        return tuple(min(d) for d in self.trade_deltas)

    @property
    def max_trade_delta(self):
        return tuple(max(d) for d in self.trade_deltas)

    def to_artifact_payload(self):
        return {
            "instrument_order": list(self.instrument_order),
            "trade_deltas": [list(d) for d in self.trade_deltas],
        }


def _vocab_size(registry, name):
    return max(int(len(registry.get(name, {}))), 1)


def _sanitize_features(features):
    sanitized = torch.nan_to_num(features.to(dtype=torch.float32), nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    return (torch.sign(sanitized) * torch.log1p(sanitized.abs())).clamp(min=-16.0, max=16.0)


class _EntityEncoder(nn.Module):
    def __init__(self, feature_dim, id_vocabs, emb_dim, token_dim, type_id):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(int(v), emb_dim) for v in id_vocabs])
        input_dim = int(feature_dim) + len(id_vocabs) * emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.register_buffer('type_id', torch.tensor(int(type_id), dtype=torch.long), persistent=False)

    def forward(self, features, ids):
        features = _sanitize_features(features)
        emb_parts = [emb(ids[..., i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([features] + emb_parts, dim=-1) if emb_parts else features
        return self.mlp(x)


class _GlobalEncoder(nn.Module):
    def __init__(self, feature_dim, token_dim, type_id):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(int(feature_dim), token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.register_buffer('type_id', torch.tensor(int(type_id), dtype=torch.long), persistent=False)

    def forward(self, features):
        return self.mlp(_sanitize_features(features)).unsqueeze(1)


class _TemporalEncoder(nn.Module):
    """1D-conv summary of a (B, K, n_channels) trajectory window. Output is a single token of
    `token_dim` per scenario, fed alongside the existing entity tokens into the transformer.
    The conv learns its own temporal weighting — strictly more flexible than the hand-picked
    LOOKBACK_WINDOWS and able to capture nonlinear patterns (acceleration, regime change)."""

    def __init__(self, n_channels, window, token_dim, type_id, *, hidden=32, kernel=5):
        super().__init__()
        if int(n_channels) <= 0 or int(window) <= 0:
            self.conv = None
            self.proj = nn.Linear(1, token_dim)  # unused fallback
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(int(n_channels), hidden, kernel_size=kernel),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, kernel_size=kernel),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.proj = nn.Linear(hidden, token_dim)
        self.register_buffer('type_id', torch.tensor(int(type_id), dtype=torch.long), persistent=False)

    def forward(self, temporal_window):
        # temporal_window shape: (B, K, n_channels) — sanitize to log-domain to handle the
        # mtm channel's $-scale alongside the log-spot channel's O(1) scale.
        if self.conv is None or temporal_window.shape[-1] == 0 or temporal_window.shape[-2] == 0:
            B = temporal_window.shape[0]
            zeros = torch.zeros((B, 1, self.proj.out_features),
                                dtype=temporal_window.dtype, device=temporal_window.device)
            return zeros
        x = _sanitize_features(temporal_window)
        # (B, K, C) -> (B, C, K) for conv1d
        x = x.permute(0, 2, 1).contiguous()
        x = self.conv(x).squeeze(-1)  # (B, hidden)
        return self.proj(x).unsqueeze(1)  # (B, 1, token_dim)


class _EntityTransformer(nn.Module):
    def __init__(self, entity_layout, token_dim, emb_dim, n_heads, n_layers):
        super().__init__()
        registry = entity_layout['registry']
        leg_vocabs = [_vocab_size(registry, name) for name in entity_layout['legs']['id_names']]
        instr_vocabs = [_vocab_size(registry, name) for name in entity_layout['instruments']['id_names']]
        cash_vocabs = [_vocab_size(registry, name) for name in entity_layout['cash_accounts']['id_names']]

        self.leg_encoder = _EntityEncoder(entity_layout['legs']['feature_dim'], leg_vocabs, emb_dim, token_dim, type_id=0)
        self.instr_encoder = _EntityEncoder(entity_layout['instruments']['feature_dim'], instr_vocabs, emb_dim, token_dim, type_id=1)
        self.cash_encoder = _EntityEncoder(entity_layout['cash_accounts']['feature_dim'], cash_vocabs, emb_dim, token_dim, type_id=2)
        self.global_encoder = _GlobalEncoder(entity_layout['globals']['feature_dim'], token_dim, type_id=3)
        temporal_meta = entity_layout.get('temporal', {})
        self.temporal_encoder = _TemporalEncoder(
            n_channels=int(temporal_meta.get('n_channels', 0)),
            window=int(temporal_meta.get('window', 0)),
            token_dim=token_dim, type_id=4,
        )
        # 5 token types: legs / instruments / cash / globals / temporal.
        self.type_embedding = nn.Embedding(5, token_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=n_heads, dim_feedforward=token_dim * 2,
            dropout=0.0, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.token_dim = token_dim

    def _add_type(self, tokens, type_id):
        return tokens + self.type_embedding(type_id).reshape(1, 1, -1)

    def forward(self, entity_state):
        legs = self._add_type(self.leg_encoder(entity_state['legs'], entity_state['legs_ids']), self.leg_encoder.type_id)
        instr = self._add_type(self.instr_encoder(entity_state['instruments'], entity_state['instruments_ids']), self.instr_encoder.type_id)
        cash = self._add_type(self.cash_encoder(entity_state['cash_accounts'], entity_state['cash_accounts_ids']), self.cash_encoder.type_id)
        glb = self._add_type(self.global_encoder(entity_state['globals']), self.global_encoder.type_id)
        temporal_window = entity_state.get('temporal', None)
        if temporal_window is not None and temporal_window.shape[-1] > 0:
            temporal = self._add_type(self.temporal_encoder(temporal_window), self.temporal_encoder.type_id)
        else:
            temporal = None

        L, I, C = legs.shape[1], instr.shape[1], cash.shape[1]
        token_list = [legs, instr, cash, glb]
        if temporal is not None:
            token_list.append(temporal)
        tokens = torch.cat(token_list, dim=1)
        batch_size = tokens.shape[0]
        global_mask = torch.ones((batch_size, glb.shape[1]), dtype=torch.bool, device=tokens.device)
        mask_list = [
            entity_state['legs_mask'].to(torch.bool),
            entity_state['instruments_mask'].to(torch.bool),
            entity_state['cash_accounts_mask'].to(torch.bool),
            global_mask,
        ]
        if temporal is not None:
            mask_list.append(torch.ones((batch_size, temporal.shape[1]), dtype=torch.bool, device=tokens.device))
        masks = torch.cat(mask_list, dim=1)
        context = self.transformer(tokens, src_key_padding_mask=~masks)
        T = temporal.shape[1] if temporal is not None else 0
        return {
            'legs': context[:, :L, :],
            'instruments': context[:, L:L + I, :],
            'cash_accounts': context[:, L + I:L + I + C, :],
            'globals': context[:, L + I + C:L + I + C + glb.shape[1], :],
            'temporal': context[:, L + I + C + glb.shape[1]:, :] if T > 0 else None,
            'mask': masks,
        }


def _masked_mean(tokens, mask):
    if tokens.shape[1] == 0:
        return tokens.new_zeros(tokens.shape[0], tokens.shape[2])
    weight = mask.to(tokens.dtype).unsqueeze(-1)
    return (tokens * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


class _PrivilegedEncoder(nn.Module):
    """Encodes privileged simulator state (latent OU factors, regime params, etc.) into a fixed-dim
    vector that the asymmetric critic concatenates with the actor's pooled context. Used only by the
    value head — actor never sees these inputs."""

    def __init__(self, factor_dims, hidden_dim):
        super().__init__()
        # Sort keys for deterministic input order across rollout/update calls.
        self.factor_names = tuple(sorted(factor_dims.keys()))
        total_input = int(sum(int(factor_dims[k]) for k in self.factor_names))
        self.input_dim = total_input
        if total_input == 0:
            self.mlp = None
            self.output_dim = 0
        else:
            self.mlp = nn.Sequential(
                nn.Linear(total_input, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.output_dim = int(hidden_dim)

    def forward(self, privileged_state):
        if self.mlp is None or not privileged_state:
            return None
        sanitized = []
        for name in self.factor_names:
            t = privileged_state[name].to(dtype=torch.float32)
            t = torch.nan_to_num(t, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
            t = (torch.sign(t) * torch.log1p(t.abs())).clamp(min=-16.0, max=16.0)
            sanitized.append(t)
        x = torch.cat(sanitized, dim=-1)
        return self.mlp(x)


class StructuredRebalancePolicy(nn.Module):
    def __init__(
        self,
        *,
        action_space,
        entity_layout,
        privileged_layout=None,
        position_limits=None,
        token_dim=64,
        emb_dim=8,
        n_heads=4,
        n_layers=2,
        device="cpu",
    ):
        super().__init__()
        self.action_space = action_space
        self.entity_layout = entity_layout
        self.privileged_layout = dict(privileged_layout or {})
        self.position_limits = dict(position_limits or {})
        self.token_dim = int(token_dim)
        self.emb_dim = int(emb_dim)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.device = torch.device(device)

        # Per-instrument categorical: each instrument has its own (possibly non-uniform)
        # list of allowed trade deltas — bin count = len of that list.
        self._action_bins = tuple(len(d) for d in action_space.trade_deltas)
        self._max_bins = max(self._action_bins) if self._action_bins else 1

        self.backbone = _EntityTransformer(entity_layout, self.token_dim, self.emb_dim, self.n_heads, self.n_layers).to(self.device)
        self.policy_head = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self._max_bins),
        ).to(self.device)
        # Zero-init final logits so the initial categorical distribution is uniform across bins.
        # Prior +0.5 bias on no-trade caused argmax to lock at no-trade for weak-gradient runs
        # even when stochastic training looked fine.
        with torch.no_grad():
            self.policy_head[-1].weight.zero_()
            self.policy_head[-1].bias.zero_()
        # Asymmetric critic: backbone output (pooled, 4*token_dim) + privileged-encoder output.
        # Privileged factors are simulator-only (latent OU state, regime params, etc.) — actor never
        # consumes them. Value-loss gradients flow back through the shared backbone, teaching the
        # actor's encoder to extract regime-relevant info from observable history.
        self.privileged_encoder = _PrivilegedEncoder(self.privileged_layout, self.token_dim).to(self.device)
        self._privileged_dim = int(self.privileged_encoder.output_dim)
        # _ctx_pool concatenates legs + instruments + cash + globals = 4 token vectors, plus
        # the temporal token when present (n_channels > 0).
        temporal_present = int(entity_layout.get('temporal', {}).get('n_channels', 0)) > 0
        self._ctx_pool_dim = self.token_dim * (5 if temporal_present else 4)
        # Wider 2-hidden-layer value head. Earlier single-hidden head at width=token_dim
        # (=64) had insufficient capacity to fit terminal-rewards spanning 6+ orders of
        # magnitude, leaving val_loss plateaued. Start at 2×token_dim (=128) — minimal
        # capacity bump that tests the binary "is V capacity-bound?" hypothesis. If it
        # still plateaus, capacity isn't the issue. If it converges, can scale to 4× later.
        v_hidden = 2 * self.token_dim
        self.value_head = nn.Sequential(
            nn.Linear(self._ctx_pool_dim + self._privileged_dim, v_hidden),
            nn.GELU(),
            nn.Linear(v_hidden, v_hidden),
            nn.GELU(),
            nn.Linear(v_hidden, 1),
        ).to(self.device)

        # Per-instrument allowed-deltas tensor padded to (n_instruments, max_bins) with the
        # last valid value (so bin lookups for invalid bin indices return a known sentinel
        # — they're masked out via _logit_mask before sampling, so the value is irrelevant
        # for outputs but must be finite for arithmetic).
        deltas_padded = []
        for deltas in action_space.trade_deltas:
            row = list(deltas) + [deltas[-1]] * (self._max_bins - len(deltas))
            deltas_padded.append(row)
        self.register_buffer(
            "_trade_deltas",
            torch.tensor(deltas_padded, dtype=torch.int64, device=self.device),  # (I, max_bins)
            persistent=False,
        )
        # Bounds kept for diagnostic-friendly summaries (not used in mask/lookup arithmetic).
        self.register_buffer(
            "_min_trade_delta",
            torch.tensor(action_space.min_trade_delta, dtype=torch.int64, device=self.device).reshape(1, -1),
            persistent=False,
        )
        self.register_buffer(
            "_max_trade_delta",
            torch.tensor(action_space.max_trade_delta, dtype=torch.int64, device=self.device).reshape(1, -1),
            persistent=False,
        )
        # mask out logits beyond each instrument's bin count (when bins differ across instruments)
        bins_tensor = torch.tensor(self._action_bins, dtype=torch.int64, device=self.device)
        bin_indices = torch.arange(self._max_bins, device=self.device).unsqueeze(0)
        invalid = bin_indices >= bins_tensor.unsqueeze(1)
        self.register_buffer("_logit_mask", invalid, persistent=False)
        # Per-instrument hard position limits for feasibility masking at sample/evaluate time.
        # Without this, the policy proposes bins that get silently clipped by the evaluator and
        # the PPO ratio is biased: log_prob is over the unconstrained distribution but the
        # observed reward is from the clipped action.
        BIG = 10**9
        instrument_order = action_space.instrument_order
        pl = self.position_limits or {}
        min_pos = tuple(int(pl.get(name, {}).get('min_position', -BIG)) for name in instrument_order)
        max_pos = tuple(int(pl.get(name, {}).get('max_position', +BIG)) for name in instrument_order)
        self.register_buffer(
            "_min_position",
            torch.tensor(min_pos, dtype=torch.int64, device=self.device).reshape(1, -1),
            persistent=False,
        )
        self.register_buffer(
            "_max_position",
            torch.tensor(max_pos, dtype=torch.int64, device=self.device).reshape(1, -1),
            persistent=False,
        )
        # Structural invariant: the no-trade bin (delta=0) must be representable AND must keep the
        # position inside [min_position, max_position]. The env enforces position-in-bounds via
        # `_enforce_position_limits`, so delta=0 always preserves the invariant — i.e. the no-trade
        # bin is always feasible. Without this, a fully-masked logit row produces NaN through
        # log_softmax (not -inf) and silently corrupts the PPO loss.
        for i, name in enumerate(instrument_order):
            deltas = action_space.trade_deltas[i]
            if 0 not in deltas:
                raise ValueError(
                    f"Action_Space for instrument {name!r} must include the no-trade delta 0; "
                    f"got Trade_Deltas={list(deltas)}. Without this the policy can encounter "
                    "fully-masked logit rows (no feasible bin), producing NaN in log_softmax."
                )
            if min_pos[i] > max_pos[i]:
                raise ValueError(
                    f"Position_Limits for instrument {name!r} are empty: "
                    f"Min_Position={min_pos[i]} > Max_Position={max_pos[i]}."
                )

    def _entity_state_to_device(self, entity_state):
        return entity_state.to(self.device)

    def _ctx_pool(self, ctx, *, detach=False):
        L = ctx['legs'].shape[1]
        I = ctx['instruments'].shape[1]
        C = ctx['cash_accounts'].shape[1]
        mask = ctx['mask']
        legs = ctx['legs'].detach() if detach else ctx['legs']
        instr = ctx['instruments'].detach() if detach else ctx['instruments']
        cash = ctx['cash_accounts'].detach() if detach else ctx['cash_accounts']
        glb = ctx['globals'].detach() if detach else ctx['globals']
        legs_pool = _masked_mean(legs, mask[:, :L])
        instr_pool = _masked_mean(instr, mask[:, L:L + I])
        cash_pool = _masked_mean(cash, mask[:, L + I:L + I + C])
        global_pool = glb.squeeze(1)
        parts = [legs_pool, instr_pool, cash_pool, global_pool]
        # Temporal token is optional (zero-channel falls back to None) — concat its post-
        # transformer representation when present so the policy/value heads can attend to it.
        temporal_ctx = ctx.get('temporal')
        if temporal_ctx is not None and temporal_ctx.shape[1] > 0:
            t = temporal_ctx.detach() if detach else temporal_ctx
            parts.append(t.squeeze(1))
        return torch.cat(parts, dim=-1)

    def _feasible_mask(self, positions):
        """Returns a (B, I, max_bins) bool tensor with True for bins whose resulting position
        would violate `position_limits`. With the explicit-deltas action space, the trade
        delta of bin k for instrument i is `_trade_deltas[i, k]`, so the resulting position
        is `positions[b,i] + _trade_deltas[i, k]`. Mask any bin whose result falls outside
        [min_position[i], max_position[i]]."""
        positions_t = positions.to(dtype=torch.int64, device=self.device)
        # Broadcast: positions (B, I, 1) + _trade_deltas (1, I, max_bins) → (B, I, max_bins)
        resulting = positions_t.unsqueeze(-1) + self._trade_deltas.unsqueeze(0)
        below = resulting < self._min_position.unsqueeze(-1)
        above = resulting > self._max_position.unsqueeze(-1)
        return below | above

    def _policy_outputs(self, entity_state, positions=None):
        es = self._entity_state_to_device(entity_state)
        ctx = self.backbone(es)
        # logits: [B, I, max_bins]; mask invalid bins to -inf so they get probability 0
        logits = self.policy_head(ctx['instruments'])
        logits = logits.masked_fill(self._logit_mask.unsqueeze(0), float('-inf'))
        if positions is not None:
            logits = logits.masked_fill(self._feasible_mask(positions), float('-inf'))
        return logits, ctx

    def _value(self, ctx, privileged_state=None):
        # Discrete action distribution is non-saturating, so value loss can flow through the
        # backbone safely. The privileged encoder consumes simulator-only state and contributes a
        # separate embedding concatenated with the pooled actor context.
        pooled = self._ctx_pool(ctx, detach=False)
        priv_emb = self.privileged_encoder(privileged_state) if privileged_state else None
        if priv_emb is None:
            if self._privileged_dim > 0:
                # Layout configured but no factors supplied at this step — pad with zeros so the
                # value-head input dim is consistent.
                priv_emb = torch.zeros((pooled.shape[0], self._privileged_dim), dtype=pooled.dtype, device=pooled.device)
                value_input = torch.cat([pooled, priv_emb], dim=-1)
            else:
                value_input = pooled
        else:
            value_input = torch.cat([pooled, priv_emb], dim=-1)
        return self.value_head(value_input).squeeze(-1)

    def _bins_to_trade_deltas(self, action_bins):
        # action_bins: int64 [B, I] in [0, n_bins_i). trade_delta = lookup in _trade_deltas[i].
        # _trade_deltas is (I, max_bins); use gather along last axis with bin indices.
        # Expand _trade_deltas to (1, I, max_bins) and gather (B, I, 1) → (B, I).
        deltas = self._trade_deltas.unsqueeze(0).expand(action_bins.shape[0], -1, -1)
        return deltas.gather(-1, action_bins.unsqueeze(-1)).squeeze(-1)

    def forward(self, entity_state, privileged_state=None, positions=None):
        logits, ctx = self._policy_outputs(entity_state, positions=positions)
        return TensorDict({'logits': logits, 'value': self._value(ctx, privileged_state)}, batch_size=[logits.shape[0]])

    def sample(self, entity_state, *, deterministic=False, privileged_state=None, positions=None):
        logits, ctx = self._policy_outputs(entity_state, positions=positions)
        log_probs_per_instr = torch.log_softmax(logits, dim=-1)
        if deterministic:
            action_bins = log_probs_per_instr.argmax(dim=-1)
        else:
            # logits-based Categorical avoids re-softmax inside the constructor and is more
            # numerically robust than passing exp'd probs (which re-renormalizes and is more
            # NaN-fragile under masked rows).
            action_bins = torch.distributions.Categorical(logits=logits).sample()
        action_bin_log_probs = log_probs_per_instr.gather(-1, action_bins.unsqueeze(-1)).squeeze(-1)
        log_prob = action_bin_log_probs.sum(-1)
        value = self._value(ctx, privileged_state)
        return TensorDict({
            'logits': logits,
            'action_bins': action_bins,
            'log_prob': log_prob,
            'value': value,
        }, batch_size=[logits.shape[0]])

    def evaluate_action(self, entity_state, action_bins, privileged_state=None, positions=None):
        logits, ctx = self._policy_outputs(entity_state, positions=positions)
        log_probs_per_instr = torch.log_softmax(logits, dim=-1)
        action_bin_log_probs = log_probs_per_instr.gather(-1, action_bins.unsqueeze(-1)).squeeze(-1)
        log_prob = action_bin_log_probs.sum(-1)
        # entropy = -Σ p log p with the convention p log p = 0 when p = 0. With feasibility-masked
        # logits, masked bins have log_p = -inf and p = 0; computing `p * log_p` would produce
        # 0 × -inf = NaN that ALSO propagates NaN gradients. Replace -inf log_probs with 0 before
        # the multiplication — for masked bins the 0 prob factor zeroes out the contribution
        # cleanly in both forward and backward.
        probs = log_probs_per_instr.exp()
        safe_log_probs = log_probs_per_instr.masked_fill(log_probs_per_instr == float('-inf'), 0.0)
        entropy_per_instr = -(probs * safe_log_probs).sum(-1)
        entropy = entropy_per_instr.sum(-1)
        value = self._value(ctx, privileged_state)
        return log_prob, entropy, value, logits

    def map_actions(self, output):
        action_bins = output['action_bins'].to(dtype=torch.int64, device=self.device)
        ordered_trade_deltas = self._bins_to_trade_deltas(action_bins)
        result = {
            'action_bins': action_bins,
            'ordered_trade_deltas': ordered_trade_deltas,
            'trade_deltas': {name: ordered_trade_deltas[:, i] for i, name in enumerate(self.action_space.instrument_order)},
        }
        for key in ('logits', 'log_prob', 'value'):
            if key in output.keys():
                result[key] = output[key]
        return result

    def to_artifact(self):
        state = self.state_dict()
        return {
            "artifact_version": 7,  # bumped for the wider value head (3 Linear/2 hidden vs 2/1)
            "policy_object_type": "StructuredRebalancePolicy",
            "entity_layout": self.entity_layout,
            "privileged_layout": dict(self.privileged_layout),
            "position_limits": dict(self.position_limits),
            "action_space": self.action_space.to_artifact_payload(),
            "model": {
                "object": "EntityTransformer",
                "token_dim": self.token_dim,
                "emb_dim": self.emb_dim,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers,
            },
            "state_dict": {
                key: {
                    "dtype": str(value.detach().cpu().dtype).replace("torch.", ""),
                    "shape": list(value.detach().cpu().shape),
                    "data": value.detach().cpu().tolist(),
                }
                for key, value in state.items()
            },
        }


def save_policy_artifact(artifact, file_path):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, default=str), encoding="utf-8")
    return path


def load_policy_artifact(file_path, *, device="cpu"):
    path = Path(file_path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    model = dict(artifact["model"])
    action_space_payload = dict(artifact["action_space"])
    # Backward compat: older artifacts stored min/max only; newer artifacts store an explicit
    # `trade_deltas` list. Honor whichever is present.
    if "trade_deltas" in action_space_payload:
        action_space = StructuredActionSpace(
            instrument_order=tuple(action_space_payload["instrument_order"]),
            trade_deltas=tuple(tuple(int(d) for d in deltas)
                               for deltas in action_space_payload["trade_deltas"]),
        )
    else:
        action_space = StructuredActionSpace(
            instrument_order=tuple(action_space_payload["instrument_order"]),
            min_trade_delta=tuple(int(value) for value in action_space_payload["min_trade_delta"]),
            max_trade_delta=tuple(int(value) for value in action_space_payload["max_trade_delta"]),
        )
    policy = StructuredRebalancePolicy(
        action_space=action_space,
        entity_layout=artifact["entity_layout"],
        privileged_layout=dict(artifact.get("privileged_layout") or {}),
        position_limits=dict(artifact.get("position_limits") or {}),
        token_dim=int(model.get("token_dim", 64)),
        emb_dim=int(model.get("emb_dim", 8)),
        n_heads=int(model.get("n_heads", 4)),
        n_layers=int(model.get("n_layers", 2)),
        device=device,
    )
    state_dict = {
        str(key): torch.tensor(
            value["data"],
            dtype=getattr(torch, str(value["dtype"])),
            device=policy.device,
        ).reshape(tuple(int(dim) for dim in value["shape"]))
        for key, value in artifact["state_dict"].items()
    }
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


if __name__ == '__main__':
    pass
