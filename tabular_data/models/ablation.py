"""Model ablation helpers.

This module provides:
1) Quantum ablation: replace the quantum circuit block with a frozen 1-hidden-layer MLP.
2) Classical ablation: freeze classical parameters while keeping quantum parameters trainable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


ABLATION_QUANTUM = "quantum"
ABLATION_CLASSICAL = "classical"
ABLATION_SUFFIX_TO_TYPE = {
    "_abla_q": ABLATION_QUANTUM,
    "_abla_c": ABLATION_CLASSICAL,
}


@dataclass
class AblationResult:
    model: Any
    applied: bool
    skipped: bool
    reason: str | None = None


@dataclass
class AblationSpec:
    requested_model: str
    base_model: str
    ablation_type: str | None
    suffix: str | None


class _FrozenOneHiddenLayerMLP(nn.Module):
    """Frozen one-hidden-layer MLP used to replace quantum blocks."""

    def __init__(self, in_features: int, out_features: int, hidden_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, out_features),
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.net(x)


class _SoftmaxScale(nn.Module):
    """Apply softmax over feature axis, then scale by a constant."""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, x):
        return torch.softmax(x, dim=-1) * self.scale


def _int_size(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, (tuple, list)):
        if not value:
            return 0
        prod = 1
        for dim in value:
            prod *= int(dim)
        return prod
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _normalization_from_measurement(
    measurement_strategy: str | None,
    n_photons: int | None,
) -> tuple[nn.Module | None, str | None]:
    if measurement_strategy is None:
        return nn.Identity(), None

    if measurement_strategy == "probs":
        return _SoftmaxScale(scale=1.0), None

    if measurement_strategy == "mode_expectations":
        if n_photons is None or int(n_photons) <= 0:
            return None, "Quantum ablation with mode_expectations requires positive n_photons."
        return _SoftmaxScale(scale=float(n_photons)), None

    return None, f"Unsupported measurement_strategy `{measurement_strategy}` for quantum ablation."


def _make_frozen_replacement_from_quantum_layer(
    quantum_layer: Any,
    measurement_strategy: str | None = None,
    n_photons: int | None = None,
) -> tuple[nn.Module | None, str | None]:
    in_features = _int_size(getattr(quantum_layer, "input_size", 0))
    out_features = _int_size(getattr(quantum_layer, "output_size", 0))
    if in_features <= 0 or out_features <= 0:
        return None, "Could not infer quantum block input/output size."
    hidden = max(1, in_features)
    frozen_mlp = _FrozenOneHiddenLayerMLP(
        in_features=in_features,
        out_features=out_features,
        hidden_features=hidden,
    )
    normalizer, norm_reason = _normalization_from_measurement(
        measurement_strategy=measurement_strategy,
        n_photons=n_photons,
    )
    if normalizer is None:
        return None, norm_reason
    if isinstance(normalizer, nn.Identity):
        return frozen_mlp, None
    return nn.Sequential(frozen_mlp, normalizer), None


def _replace_quantum_block_torch(
    model_name: str,
    model: Any,
    measurement_strategy: str | None = None,
    n_photons: int | None = None,
) -> AblationResult:
    if model_name in {
        "dressed_quantum_circuit",
        "dressed_quantum_circuit_reservoir",
    }:
        if not hasattr(model, "dqc") or not isinstance(model.dqc, nn.Sequential):
            return AblationResult(
                model=model, applied=False, skipped=True, reason="Missing expected `dqc` block."
            )
        quantum_layer = model.dqc[0]
        replacement, reason = _make_frozen_replacement_from_quantum_layer(
            quantum_layer=quantum_layer,
            measurement_strategy=measurement_strategy,
            n_photons=n_photons,
        )
        if replacement is None:
            return AblationResult(
                model=model,
                applied=False,
                skipped=True,
                reason=reason,
            )
        model.dqc[0] = replacement
        return AblationResult(model=model, applied=True, skipped=False)

    if model_name in {"multiple_paths_model", "multiple_paths_model_reservoir"}:
        if not hasattr(model, "pqc"):
            return AblationResult(
                model=model, applied=False, skipped=True, reason="Missing expected `pqc` block."
            )
        quantum_layer = model.pqc
        replacement, reason = _make_frozen_replacement_from_quantum_layer(
            quantum_layer=quantum_layer,
            measurement_strategy=measurement_strategy,
            n_photons=n_photons,
        )
        if replacement is None:
            return AblationResult(
                model=model,
                applied=False,
                skipped=True,
                reason=reason,
            )
        model.pqc = replacement
        return AblationResult(model=model, applied=True, skipped=False)

    return AblationResult(
        model=model,
        applied=False,
        skipped=True,
        reason=f"Quantum ablation not implemented for torch model `{model_name}`.",
    )


def _replace_quantum_block_sklearn_kernel(
    model_name: str,
    model: Any,
    measurement_strategy: str | None = None,
    n_photons: int | None = None,
) -> AblationResult:
    if model_name != "q_rks":
        return AblationResult(
            model=model,
            applied=False,
            skipped=True,
            reason=f"Quantum ablation not implemented for sklearn_kernel model `{model_name}`.",
        )
    if not hasattr(model, "pqc") or not isinstance(model.pqc, nn.Sequential):
        return AblationResult(
            model=model,
            applied=False,
            skipped=True,
            reason="Missing expected `pqc` sequential block.",
        )
    quantum_layer = model.pqc[0]
    replacement, reason = _make_frozen_replacement_from_quantum_layer(
        quantum_layer=quantum_layer,
        measurement_strategy=measurement_strategy,
        n_photons=n_photons,
    )
    if replacement is None:
        return AblationResult(
            model=model,
            applied=False,
            skipped=True,
            reason=reason,
        )
    model.pqc[0] = replacement
    return AblationResult(model=model, applied=True, skipped=False)


def _apply_quantum_ablation(
    model_dict: dict[str, Any],
    measurement_strategy: str | None = None,
    n_photons: int | None = None,
) -> AblationResult:
    model_type = model_dict.get("type")
    model_name = model_dict.get("name", "")
    model = model_dict.get("model")

    if model is None:
        return AblationResult(
            model=None,
            applied=False,
            skipped=True,
            reason="Missing `model` entry in model_dict.",
        )

    if model_type == "torch":
        return _replace_quantum_block_torch(
            model_name=model_name,
            model=model,
            measurement_strategy=measurement_strategy,
            n_photons=n_photons,
        )

    if model_type == "sklearn_kernel":
        return _replace_quantum_block_sklearn_kernel(
            model_name=model_name,
            model=model,
            measurement_strategy=measurement_strategy,
            n_photons=n_photons,
        )

    return AblationResult(
        model=model,
        applied=False,
        skipped=True,
        reason=f"Quantum ablation unsupported for model type `{model_type}`.",
    )


def _apply_classical_ablation(model_dict: dict[str, Any]) -> AblationResult:
    model_type = model_dict.get("type")
    model_name = model_dict.get("name", "")
    model = model_dict.get("model")

    if model is None:
        return AblationResult(
            model=None,
            applied=False,
            skipped=True,
            reason="Missing `model` entry in model_dict.",
        )

    if model_name.endswith("_reservoir"):
        return AblationResult(
            model=model,
            applied=False,
            skipped=True,
            reason="Classical ablation skipped for reservoir models.",
        )

    if model_type != "torch":
        return AblationResult(
            model=model,
            applied=False,
            skipped=True,
            reason=f"Classical ablation unsupported for model type `{model_type}`.",
        )

    if model_name == "dressed_quantum_circuit":
        quantum_prefix = "dqc.0"
    elif model_name == "multiple_paths_model":
        quantum_prefix = "pqc"
    else:
        return AblationResult(
            model=model,
            applied=False,
            skipped=True,
            reason=f"Classical ablation unsupported for torch model `{model_name}`.",
        )

    trainable_quantum = 0
    frozen_classical = 0
    for param_name, param in model.named_parameters():
        if param_name.startswith(quantum_prefix):
            if param.requires_grad:
                trainable_quantum += param.numel()
        else:
            if param.requires_grad:
                param.requires_grad = False
                frozen_classical += param.numel()

    if trainable_quantum == 0:
        return AblationResult(
            model=model,
            applied=False,
            skipped=True,
            reason="No trainable quantum parameters found (likely reservoir or unsupported configuration).",
        )

    if frozen_classical == 0:
        return AblationResult(
            model=model,
            applied=False,
            skipped=True,
            reason="No trainable classical parameters found to freeze.",
        )

    return AblationResult(model=model, applied=True, skipped=False)


def ablate_model(
    model_dict: dict[str, Any],
    ablation_type: str,
    measurement_strategy: str | None = None,
    n_photons: int | None = None,
) -> AblationResult:
    """
    Apply ablation to a model dictionary in-place.

    Args:
        model_dict: Dict with keys at least `type`, `name`, `model`.
        ablation_type: One of `quantum` or `classical`.

    Returns:
        AblationResult with `applied/skipped` and an optional `reason`.
    """
    if ablation_type == ABLATION_QUANTUM:
        result = _apply_quantum_ablation(
            model_dict=model_dict,
            measurement_strategy=measurement_strategy,
            n_photons=n_photons,
        )
    elif ablation_type == ABLATION_CLASSICAL:
        result = _apply_classical_ablation(model_dict)
    else:
        return AblationResult(
            model=model_dict.get("model"),
            applied=False,
            skipped=True,
            reason=f"Unknown ablation type `{ablation_type}`. Use `quantum` or `classical`.",
        )

    model_dict["model"] = result.model
    return result


def parse_ablation_model_name(model_name: str) -> AblationSpec:
    for suffix, ablation_type in ABLATION_SUFFIX_TO_TYPE.items():
        if model_name.endswith(suffix):
            return AblationSpec(
                requested_model=model_name,
                base_model=model_name[: -len(suffix)],
                ablation_type=ablation_type,
                suffix=suffix,
            )
    return AblationSpec(
        requested_model=model_name,
        base_model=model_name,
        ablation_type=None,
        suffix=None,
    )


def can_apply_ablation(
    model_type: str,
    model_name: str,
    ablation_type: str,
) -> tuple[bool, str | None]:
    if ablation_type == ABLATION_QUANTUM:
        allowed = {
            ("torch", "dressed_quantum_circuit"),
            ("torch", "dressed_quantum_circuit_reservoir"),
            ("torch", "multiple_paths_model"),
            ("torch", "multiple_paths_model_reservoir"),
            ("sklearn_kernel", "q_rks"),
        }
        if (model_type, model_name) in allowed:
            return True, None
        return (
            False,
            f"Quantum ablation unsupported for model `{model_name}` with type `{model_type}`.",
        )

    if ablation_type == ABLATION_CLASSICAL:
        if model_name.endswith("_reservoir"):
            return False, "Classical ablation is not defined for reservoir models."
        allowed = {
            ("torch", "dressed_quantum_circuit"),
            ("torch", "multiple_paths_model"),
        }
        if (model_type, model_name) in allowed:
            return True, None
        return (
            False,
            f"Classical ablation unsupported for model `{model_name}` with type `{model_type}`.",
        )

    return False, f"Unknown ablation type `{ablation_type}`."


def apply_ablation_if_requested(
    model_type: str,
    model_name: str,
    model_obj: Any,
    training_params: dict[str, Any] | None,
    measurement_strategy: str | None = None,
    n_photons: int | None = None,
) -> AblationResult:
    params = training_params or {}
    ablation_type = params.get("ablation_type")
    if not ablation_type:
        return AblationResult(model=model_obj, applied=False, skipped=False)

    model_dict = {"type": model_type, "name": model_name, "model": model_obj}
    return ablate_model(
        model_dict,
        ablation_type,
        measurement_strategy=measurement_strategy,
        n_photons=n_photons,
    )
