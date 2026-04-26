from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.modules import MLP


@dataclass(frozen=True)
class StructuredActionSpace:
    instrument_order: tuple[str, ...]
    min_trade_delta: tuple[int, ...]
    max_trade_delta: tuple[int, ...]

    def __init__(self, instrument_order, min_trade_delta, max_trade_delta):
        object.__setattr__(self, "instrument_order", tuple(name for name in instrument_order))
        object.__setattr__(self, "min_trade_delta", tuple(value for value in min_trade_delta))
        object.__setattr__(self, "max_trade_delta", tuple(value for value in max_trade_delta))
        if not self.instrument_order:
            raise ValueError("StructuredActionSpace.instrument_order must be non-empty")
        if not (
            len(self.instrument_order) == len(self.min_trade_delta) == len(self.max_trade_delta)
        ):
            raise ValueError("StructuredActionSpace entries must have matching lengths")
        for instrument_name, min_delta, max_delta in zip(
            self.instrument_order,
            self.min_trade_delta,
            self.max_trade_delta,
        ):
            if min_delta > max_delta:
                raise ValueError(
                    f"StructuredActionSpace has min_trade_delta > max_trade_delta for {instrument_name}"
                )

    @property
    def dimension(self) -> int:
        return len(self.instrument_order)

    def to_artifact_payload(self) -> Dict[str, Any]:
        return {
            "instrument_order": list(self.instrument_order),
            "min_trade_delta": list(self.min_trade_delta),
            "max_trade_delta": list(self.max_trade_delta),
        }


@dataclass(frozen=True)
class TorchRLStateFeatureExtractor:
    feature_dim: int

    def __init__(self, *, feature_dim):
        normalized_feature_dim = int(feature_dim)
        object.__setattr__(self, "feature_dim", normalized_feature_dim)
        if normalized_feature_dim <= 0:
            raise ValueError("TorchRLStateFeatureExtractor.feature_dim must be positive")

    def __call__(self, feature_batch) -> torch.Tensor:
        tensor = torch.as_tensor(feature_batch, dtype=torch.float32)
        if tensor.ndim != 2:
            raise ValueError("policy_features must have shape [B, F]")
        if int(tensor.shape[1]) != self.feature_dim:
            raise ValueError("policy_features width does not match canonical feature_dim")
        return tensor

    def to_artifact_payload(self) -> Dict[str, Any]:
        return {
            "object": "TorchRLStateFeatureExtractor",
            "contract": "canonical_policy_features",
            "feature_dim": self.feature_dim,
        }


class _StructuredPolicyHead(nn.Module):
    def __init__(self, *, input_dim, action_dim, hidden_layers, activation, device):
        super().__init__()
        try:
            activation_class = getattr(torch.nn.modules.activation, activation)
        except AttributeError as exc:
            raise ValueError(f"Unsupported activation: {activation}") from exc
        if any(int(width) <= 0 for width in hidden_layers):
            raise ValueError("hidden_layers must contain only positive integers")
        self.network = MLP(
            in_features=input_dim,
            out_features=int(action_dim),
            num_cells=[width for width in hidden_layers],
            activation_class=activation_class,
            device=device,
        )

    def forward(self, features):
        raw_output = self.network(features)
        rebalance_vector = torch.tanh(raw_output)
        return rebalance_vector


class StructuredRebalancePolicy(nn.Module):
    def __init__(
        self,
        *,
        action_space,
        feature_extractor,
        hidden_layers=(64, 64),
        activation="ReLU",
        device="cpu",
    ):
        super().__init__()
        self.action_space = action_space
        self.feature_extractor = feature_extractor
        self.hidden_layers = tuple(int(value) for value in hidden_layers)
        self.activation = str(activation)
        self.device = torch.device(device)
        input_dim = getattr(feature_extractor, "feature_dim", None)
        self.actor = _StructuredPolicyHead(
            input_dim=input_dim,
            action_dim=action_space.dimension,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            device=self.device,
        ).to(self.device)
        self.register_buffer(
            "_min_trade_delta",
            torch.tensor(action_space.min_trade_delta, dtype=torch.float32, device=self.device).reshape(1, -1),
            persistent=False,
        )
        self.register_buffer(
            "_max_trade_delta",
            torch.tensor(action_space.max_trade_delta, dtype=torch.float32, device=self.device).reshape(1, -1),
            persistent=False,
        )
        self.register_buffer(
            "_min_trade_delta_abs",
            self._min_trade_delta.abs(),
            persistent=False,
        )
        self.register_buffer(
            "_action_feature_scale",
            torch.maximum(self._max_trade_delta.abs(), self._min_trade_delta_abs).clamp_min(1.0),
            persistent=False,
        )

    def extract_features(self, feature_batch) -> torch.Tensor:
        return self.feature_extractor(feature_batch).to(device=self.device)

    def forward(self, feature_batch):
        features = self.extract_features(feature_batch)
        rebalance_vector = self.actor(features)
        return TensorDict(
            {
                "features": features,
                "rebalance_vector": rebalance_vector,
            },
            batch_size=[int(features.shape[0])],
        )

    def sample(self, feature_batch, *, epsilon=0.0, greedy=False):
        output = self(feature_batch)
        if greedy or float(epsilon) <= 0.0:
            return output
        batch_size = int(output.batch_size[0])
        random_mask = torch.rand(batch_size, device=self.device) < float(epsilon)
        if not bool(random_mask.any().item()):
            return output
        output = output.clone(False)
        random_vector = 2.0 * torch.rand(
            (batch_size, self.action_space.dimension),
            dtype=torch.float32,
            device=self.device,
        ) - 1.0
        output["rebalance_vector"] = torch.where(
            random_mask.unsqueeze(1),
            random_vector,
            output["rebalance_vector"],
        )
        return output

    def _continuous_trade_deltas(self, rebalance_vector: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(rebalance_vector, min=-1.0, max=1.0)
        positive_trade = clipped.clamp(min=0.0) * self._max_trade_delta
        negative_trade = clipped.clamp(max=0.0) * self._min_trade_delta_abs
        return torch.where(
            clipped > 0.0,
            positive_trade,
            torch.where(clipped < 0.0, negative_trade, torch.zeros_like(clipped)),
        )

    def _normalize_trade_deltas(self, trade_deltas: torch.Tensor) -> torch.Tensor:
        return trade_deltas / self._action_feature_scale

    def map_actions(self, output) -> Dict[str, Any]:
        rebalance_vector = torch.as_tensor(output["rebalance_vector"], dtype=torch.float32, device=self.device)
        if rebalance_vector.ndim != 2:
            raise ValueError("rebalance_vector must have shape [B, A]")
        if int(rebalance_vector.shape[1]) != self.action_space.dimension:
            raise ValueError("rebalance_vector action dimension does not match action_space")
        mapped = self._continuous_trade_deltas(rebalance_vector)
        rounded = torch.sign(mapped) * torch.floor(mapped.abs() + 0.5)
        bounded = torch.max(torch.min(rounded, self._max_trade_delta), self._min_trade_delta).to(torch.int64)
        ordered_trade_deltas = bounded
        return {
            "rebalance_vector": rebalance_vector,
            "ordered_trade_deltas": ordered_trade_deltas,
            "trade_deltas": {
                instrument_name: ordered_trade_deltas[:, index]
                for index, instrument_name in enumerate(self.action_space.instrument_order)
            },
        }

    def action_features(self, output) -> torch.Tensor:
        if "ordered_trade_deltas" in output:
            ordered_trade_deltas = torch.as_tensor(output["ordered_trade_deltas"], dtype=torch.float32, device=self.device)
            return self._normalize_trade_deltas(ordered_trade_deltas)
        rebalance_vector = torch.as_tensor(output["rebalance_vector"], dtype=torch.float32, device=self.device)
        if rebalance_vector.ndim != 2:
            raise ValueError("rebalance_vector must have shape [B, A]")
        continuous_trade_deltas = self._continuous_trade_deltas(rebalance_vector)
        return self._normalize_trade_deltas(continuous_trade_deltas)

    def to_artifact(self) -> Dict[str, Any]:
        if not hasattr(self.feature_extractor, "to_artifact_payload"):
            raise TypeError("StructuredRebalancePolicy artifact export requires a serializable feature extractor")
        state = self.state_dict()
        if not state:
            raise ValueError("StructuredRebalancePolicy cannot be serialized before its model is initialized")
        first_tensor = next(iter(state.values()))
        if isinstance(first_tensor, nn.parameter.UninitializedParameter):
            raise ValueError("StructuredRebalancePolicy cannot be serialized before its lazy layers are initialized")
        feature_dim = getattr(self.feature_extractor, "feature_dim", None)
        if feature_dim is None:
            raise ValueError("StructuredRebalancePolicy artifact could not infer feature dimension")
        return {
            "artifact_version": 2,
            "policy_object_type": "StructuredRebalancePolicy",
            "feature_extractor": self.feature_extractor.to_artifact_payload(),
            "action_space": self.action_space.to_artifact_payload(),
            "model": {
                "object": "MLP",
                "hidden_layers": list(self.hidden_layers),
                "activation": self.activation,
                "output_heads": {
                    "Trade_Deltas": {
                        "object": "TanhDeltaHead",
                    },
                },
                "feature_dim": feature_dim,
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


def save_policy_artifact(artifact: Mapping[str, Any], file_path) -> Path:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2), encoding="utf-8")
    return path


def load_policy_artifact(file_path, *, device: str = "cpu") -> StructuredRebalancePolicy:
    path = Path(file_path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if str(artifact.get("policy_object_type")) != "StructuredRebalancePolicy":
        raise ValueError("Unsupported policy artifact type")
    model_payload = dict(artifact.get("model", {}))
    action_space_payload = dict(artifact.get("action_space", {}))
    feature_extractor_payload = dict(artifact.get("feature_extractor", {}))
    feature_dim = int(feature_extractor_payload.get("feature_dim", model_payload.get("feature_dim")))
    policy = StructuredRebalancePolicy(
        action_space=StructuredActionSpace(
            instrument_order=tuple(action_space_payload.get("instrument_order", ())),
            min_trade_delta=tuple(int(value) for value in action_space_payload.get("min_trade_delta", ())),
            max_trade_delta=tuple(int(value) for value in action_space_payload.get("max_trade_delta", ())),
        ),
        feature_extractor=TorchRLStateFeatureExtractor(feature_dim=feature_dim),
        hidden_layers=tuple(int(value) for value in model_payload.get("hidden_layers", (64, 64))),
        activation=str(model_payload.get("activation", "ReLU")),
        device=device,
    )
    _ = policy(torch.zeros((1, feature_dim), dtype=torch.float32, device=policy.device))
    serialized_state = dict(artifact.get("state_dict", {}))
    state_dict = {
        str(key): torch.tensor(
            value.get("data", []),
            dtype=getattr(torch, str(value.get("dtype", "float32"))),
            device=policy.device,
        ).reshape(tuple(int(dim) for dim in value.get("shape", ())))
        for key, value in serialized_state.items()
    }
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy
