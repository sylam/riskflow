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
    instrument_order: tuple
    min_trade_delta: tuple
    max_trade_delta: tuple

    def __init__(self, instrument_order, min_trade_delta, max_trade_delta):
        object.__setattr__(self, "instrument_order", tuple(instrument_order))
        object.__setattr__(self, "min_trade_delta", tuple(min_trade_delta))
        object.__setattr__(self, "max_trade_delta", tuple(max_trade_delta))

    @property
    def dimension(self):
        return len(self.instrument_order)

    def to_artifact_payload(self):
        return {
            "instrument_order": list(self.instrument_order),
            "min_trade_delta": list(self.min_trade_delta),
            "max_trade_delta": list(self.max_trade_delta),
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
        self.type_embedding = nn.Embedding(4, token_dim)

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

        L, I, C = legs.shape[1], instr.shape[1], cash.shape[1]
        tokens = torch.cat([legs, instr, cash, glb], dim=1)
        batch_size = tokens.shape[0]
        global_mask = torch.ones((batch_size, glb.shape[1]), dtype=torch.bool, device=tokens.device)
        masks = torch.cat([
            entity_state['legs_mask'].to(torch.bool),
            entity_state['instruments_mask'].to(torch.bool),
            entity_state['cash_accounts_mask'].to(torch.bool),
            global_mask,
        ], dim=1)
        context = self.transformer(tokens, src_key_padding_mask=~masks)
        return {
            'legs': context[:, :L, :],
            'instruments': context[:, L:L + I, :],
            'cash_accounts': context[:, L + I:L + I + C, :],
            'globals': context[:, L + I + C:, :],
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

        # Per-instrument categorical: each instrument has (max - min + 1) discrete trade actions.
        self._action_bins = tuple(int(hi) - int(lo) + 1 for lo, hi in zip(action_space.min_trade_delta, action_space.max_trade_delta))
        self._max_bins = max(self._action_bins) if self._action_bins else 1

        self.backbone = _EntityTransformer(entity_layout, self.token_dim, self.emb_dim, self.n_heads, self.n_layers).to(self.device)
        self.policy_head = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self._max_bins),
        ).to(self.device)
        # uniform-init logits (zero weights, zero bias). prior init had a +0.5 bias on the no-trade
        # bin which biased deterministic argmax toward no-trade and trapped weak-gradient runs into
        # a no-trade attractor at evaluation time even when training rollouts were exploring.
        with torch.no_grad():
            self.policy_head[-1].weight.zero_()
            self.policy_head[-1].bias.zero_()
        # Asymmetric critic: backbone output (pooled, 4*token_dim) + privileged-encoder output.
        # Privileged factors are simulator-only (latent OU state, regime params, etc.) — actor never
        # consumes them. Value-loss gradients flow back through the shared backbone, teaching the
        # actor's encoder to extract regime-relevant info from observable history.
        self.privileged_encoder = _PrivilegedEncoder(self.privileged_layout, self.token_dim).to(self.device)
        self._privileged_dim = int(self.privileged_encoder.output_dim)
        self.value_head = nn.Sequential(
            nn.Linear(self.token_dim * 4 + self._privileged_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, 1),
        ).to(self.device)

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
        self.register_buffer(
            "_action_feature_scale",
            torch.maximum(self._max_trade_delta.abs(), self._min_trade_delta.abs()).clamp_min(1).to(dtype=torch.float32),
            persistent=False,
        )
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
        return torch.cat([legs_pool, instr_pool, cash_pool, global_pool], dim=-1)

    def _feasible_mask(self, positions):
        """Returns a (B, I, max_bins) bool tensor with True for bins whose resulting position
        would violate `position_limits`. trade_delta of bin k = k + min_trade_delta[i], so the
        resulting position is positions[b,i] + min_trade_delta[i] + k. Mask any bin whose result
        falls outside [min_position[i], max_position[i]]."""
        positions_t = positions.to(dtype=torch.int64, device=self.device)
        bin_indices = torch.arange(self._max_bins, device=self.device, dtype=torch.int64)
        # Broadcast: positions (B, I, 1) + min_trade_delta (1, I, 1) + bin_indices (max_bins,) → (B, I, max_bins)
        resulting = positions_t.unsqueeze(-1) + self._min_trade_delta.unsqueeze(-1) + bin_indices.reshape(1, 1, -1)
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
        # action_bins: int64 [B, I] in [0, n_bins_i). trade_delta = bin + min_trade.
        return action_bins + self._min_trade_delta

    def _trade_deltas_to_bins(self, trade_deltas):
        return trade_deltas - self._min_trade_delta

    def forward(self, entity_state, privileged_state=None, positions=None):
        logits, ctx = self._policy_outputs(entity_state, positions=positions)
        return TensorDict({'logits': logits, 'value': self._value(ctx, privileged_state)}, batch_size=[logits.shape[0]])

    def sample(self, entity_state, *, deterministic=False, privileged_state=None, positions=None):
        logits, ctx = self._policy_outputs(entity_state, positions=positions)
        log_probs_per_instr = torch.log_softmax(logits, dim=-1)
        if deterministic:
            action_bins = log_probs_per_instr.argmax(dim=-1)
        else:
            probs = log_probs_per_instr.exp()
            action_bins = torch.distributions.Categorical(probs=probs).sample()
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

    def action_features(self, output):
        ordered = torch.as_tensor(output['ordered_trade_deltas'], dtype=torch.float32, device=self.device)
        return ordered / self._action_feature_scale

    def to_artifact(self):
        state = self.state_dict()
        return {
            "artifact_version": 6,
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
    policy = StructuredRebalancePolicy(
        action_space=StructuredActionSpace(
            instrument_order=tuple(action_space_payload["instrument_order"]),
            min_trade_delta=tuple(int(value) for value in action_space_payload["min_trade_delta"]),
            max_trade_delta=tuple(int(value) for value in action_space_payload["max_trade_delta"]),
        ),
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
