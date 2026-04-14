#!/usr/bin/env python3
"""Analyze model and quantum-circuit outputs across hidden_manifold_d_6.

This script compares output behavior across data dimensionality `d` for all
registered models/backends, including ablation variants where supported.

Metrics per configuration:
- output size (number of output channels)
- sum(output) for the first sample
- mean and variance of sum(output) across an evaluation subset

For photonic models, measurement settings are configurable and default to:
- (mode_expectations, none)
- (probs, none)  # displayed as `probabilities` in reports

The implementation is intentionally open to adding new groupings later.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import resource
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Import project modules by adding /tabular_data to sys.path.
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABULAR_ROOT = PROJECT_ROOT / "tabular_data"
if str(TABULAR_ROOT) not in sys.path:
    sys.path.insert(0, str(TABULAR_ROOT))

from datasets.fetch_data import fetch_data  # noqa: E402
from models.ablation import (  # noqa: E402
    ABLATION_CLASSICAL,
    ABLATION_QUANTUM,
    ablate_model,
    can_apply_ablation,
)
from models.fetch_model import fetch_model  # noqa: E402
from models.photonic_models.q_rks import get_random_w_b, get_x_r_i_s  # noqa: E402
from registry import CLASSICAL_MODELS, GATE_MODELS, PHOTONIC_MODELS  # noqa: E402
from utils.photonic_dims import get_photonic_mn  # noqa: E402

import torch  # noqa: E402


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DEFAULT_PHOTONIC_MEASUREMENT_CONFIGS = [
    {"measurement_strategy": "mode_expectations", "grouping": "none"},
    {"measurement_strategy": "probs", "grouping": "none"},
]


@dataclass(frozen=True)
class ModelSpec:
    backend: str
    model: str


@dataclass
class EvalContext:
    x_eval_torch: torch.Tensor
    x_eval_np: np.ndarray
    input_size: int


@dataclass
class OutputAccumulator:
    output_size: int | None = None
    first_sample_sum: float | None = None
    n_samples: int = 0
    mean_sum: float = 0.0
    m2_sum: float = 0.0

    def update(self, values: np.ndarray | None) -> None:
        if values is None:
            return

        arr = np.asarray(values)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)

        if self.output_size is None:
            self.output_size = int(arr.shape[1])

        sums = arr.sum(axis=1).astype(float)
        if sums.size == 0:
            return

        if self.first_sample_sum is None:
            self.first_sample_sum = float(sums[0])

        for value in sums:
            self.n_samples += 1
            delta = value - self.mean_sum
            self.mean_sum += delta / self.n_samples
            delta2 = value - self.mean_sum
            self.m2_sum += delta * delta2

    def as_dict(self) -> dict[str, Any]:
        if self.n_samples == 0:
            return {
                "output_size": None,
                "sum_output_first_sample": None,
                "sum_output_mean": None,
                "sum_output_variance": None,
            }
        return {
            "output_size": self.output_size,
            "sum_output_first_sample": self.first_sample_sum,
            "sum_output_mean": float(self.mean_sum),
            "sum_output_variance": float(self.m2_sum / self.n_samples),
        }


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def _progress(msg: str) -> None:
    print(f"[analyze_model_outputs] {msg}", flush=True)


def _rss_mb() -> float:
    """Best-effort process RSS in MB (platform-dependent ru_maxrss units)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux commonly reports KB.
    if rss > 10_000_000:
        return float(rss) / (1024.0 * 1024.0)
    return float(rss) / 1024.0


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_num_neurons(value: Any) -> list[int]:
    """Pick a deterministic architecture when HP files contain multiple candidates."""
    if value is None:
        return []
    if isinstance(value, list) and value and isinstance(value[0], list):
        return [int(v) for v in value[0]]
    if isinstance(value, list):
        return [int(v) for v in value]
    return []


def _single_run_model_hps(
    backend: str, model: str, random_state: int
) -> dict[str, Any]:
    if backend == "photonic":
        hps = _load_json(
            TABULAR_ROOT / "hyperparameters/single_run/photonic_model_hps.json"
        )
        out = copy.deepcopy(hps[model]["default"])
        if model in {
            "dressed_quantum_circuit",
            "dressed_quantum_circuit_reservoir",
            "multiple_paths_model",
            "multiple_paths_model_reservoir",
        }:
            out["type"] = "torch"
        elif model == "data_reuploading":
            out["type"] = "reuploading"
        elif model == "q_rks":
            out["type"] = "sklearn_kernel"
        else:
            out["type"] = "sklearn_q_kernel"
        if "numNeurons" in out:
            out["numNeurons"] = _normalize_num_neurons(out["numNeurons"])
        return out

    if backend == "gate":
        hps = _load_json(
            TABULAR_ROOT / "hyperparameters/single_run/gate_model_hps.json"
        )
        out = copy.deepcopy(hps[model]["default"])
        out["random_state"] = random_state
        if "numNeurons" in out:
            out["numNeurons"] = _normalize_num_neurons(out["numNeurons"])
        if model in {
            "dressed_quantum_circuit",
            "dressed_quantum_circuit_reservoir",
            "multiple_paths_model",
            "multiple_paths_model_reservoir",
            "data_reuploading",
        }:
            out["type"] = "jax_sklearn_gate"
        elif model == "q_rks":
            out["type"] = "gate_rks"
        else:
            out["type"] = "sklearn_gate"
        return out

    if backend == "classical":
        hps = _load_json(
            TABULAR_ROOT / "hyperparameters/single_run/classical_model_hps.json"
        )
        out = copy.deepcopy(hps[model]["default"])
        out["random_state"] = random_state
        if model == "mlp":
            out["numNeurons"] = _normalize_num_neurons(out.get("numNeurons", []))
            out["type"] = "torch"
        elif model == "rks":
            out["type"] = "sklearn_kernel"
        else:
            out["type"] = "sklearn"
        return out

    raise ValueError(f"Unsupported backend: {backend}")


def _dataset_hps_for_analysis() -> dict[str, Any]:
    hps = _load_json(TABULAR_ROOT / "hyperparameters/single_run/dataset_hps.json")
    base = copy.deepcopy(hps["hidden_manifold"])
    base["labels_treatment"] = "0_1"
    base["num_train"] = None
    base["num_test"] = None
    return base


def _load_eval_context(
    d: int, manifold_dim: int, sample_limit: int, random_state: int
) -> EvalContext:
    dataset_name = f"hidden_manifold_{d}_{manifold_dim}"
    dataset_hps = _dataset_hps_for_analysis()
    _, _, x_train, _, _, _ = fetch_data(
        dataset_name, random_state=random_state, **dataset_hps
    )

    if sample_limit > 0 and x_train.shape[0] > sample_limit:
        x_eval_torch = x_train[:sample_limit]
    else:
        x_eval_torch = x_train

    x_eval_np = x_eval_torch.detach().cpu().numpy()
    return EvalContext(
        x_eval_torch=x_eval_torch,
        x_eval_np=x_eval_np,
        input_size=int(x_eval_torch.shape[1]),
    )


def _torch_linear_layers(model: torch.nn.Module) -> list[int]:
    sizes = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            sizes.append(int(module.out_features))
    return sizes


def _gate_linear_layers(model: Any) -> list[str]:
    layers: list[str] = []
    params = getattr(model, "params_", None)
    if not isinstance(params, dict):
        return layers

    if "input_weights" in params:
        shape = tuple(np.asarray(params["input_weights"]).shape)
        layers.append(f"input_weights:{shape}")

    output_weights = params.get("output_weights")
    if output_weights is not None:
        if isinstance(output_weights, tuple):
            for idx, w in enumerate(output_weights):
                shape = tuple(np.asarray(w).shape)
                layers.append(f"output_weights_{idx}:{shape}")
        else:
            shape = tuple(np.asarray(output_weights).shape)
            layers.append(f"output_weights:{shape}")

    return layers


def _to_positive_int(value: Any) -> int | None:
    try:
        v = int(value)
    except Exception:
        return None
    return v if v > 0 else None


def _extract_mn_from_input_state(input_state: Any) -> tuple[int | None, int | None]:
    """Infer (m, n) from a Fock input-state vector when available."""
    if input_state is None:
        return None, None
    try:
        arr = np.asarray(input_state)
    except Exception:
        return None, None
    if arr.ndim != 1 or arr.size == 0:
        return None, None
    return _to_positive_int(arr.size), _to_positive_int(arr.sum())


def _infer_photonic_mn(
    model: Any, model_name: str, input_size: int
) -> tuple[int | None, int | None]:
    """Resolve (m, n) from instantiated photonic model, fallback to runtime sizing rule."""
    # Merlin reuploading implementation is fixed at m=2, n=1.
    if model_name == "data_reuploading":
        return 2, 1

    # Prefer explicit attributes if a model defines them.
    for obj in (
        model,
        getattr(model, "model", None),
        getattr(model, "quantum_model", None),
    ):
        if obj is None:
            continue
        m_val = _to_positive_int(getattr(obj, "m", None))
        n_val = _to_positive_int(getattr(obj, "n", None))
        if m_val is not None and n_val is not None:
            return m_val, n_val

    # Torch photonic models expose a first quantum layer in dqc/pqc.
    q_layer = None
    if hasattr(model, "dqc"):
        q_layer = model.dqc
    elif hasattr(model, "pqc"):
        q_layer = model.pqc

    if q_layer is not None:
        m_val, n_val = _extract_mn_from_input_state(
            getattr(q_layer, "input_state", None)
        )
        if m_val is not None and n_val is not None:
            return m_val, n_val

    # q_kernel_method wraps the quantum object in quantum_kernel/feature_map.
    for state_obj in (
        getattr(model, "quantum_kernel", None),
        getattr(getattr(model, "quantum_kernel", None), "feature_map", None),
    ):
        if state_obj is None:
            continue
        m_val, n_val = _extract_mn_from_input_state(
            getattr(state_obj, "input_state", None)
        )
        if m_val is not None and n_val is not None:
            return m_val, n_val

    # Final fallback: use the same sizing rule as model instantiation.
    return get_photonic_mn(int(input_size))


def _extract_architecture(
    backend: str, model_name: str, model: Any, model_hps: dict[str, Any], d: int
) -> dict[str, Any]:
    n_qubits = None
    n_modes = None
    n_photons = None

    if backend == "photonic":
        if model_name == "data_reuploading":
            n_modes, n_photons = 2, 1
        else:
            n_modes, n_photons = get_photonic_mn(int(d), mode="feature_plus_one")

    if backend == "gate":
        if hasattr(model, "n_qubits_") and model.n_qubits_ is not None:
            n_qubits = int(model.n_qubits_)
        elif model_name == "data_reuploading":
            n_qubits = int(math.ceil(d / 3))
        elif model_name in {
            "dressed_quantum_circuit",
            "dressed_quantum_circuit_reservoir",
            "multiple_paths_model",
            "multiple_paths_model_reservoir",
            "q_kernel_method_reservoir",
        }:
            n_qubits = int(d)

    if backend == "photonic" and isinstance(model, torch.nn.Module):
        linear_layers = _torch_linear_layers(model)
    elif backend == "classical" and isinstance(model, torch.nn.Module):
        linear_layers = _torch_linear_layers(model)
    elif backend == "gate":
        linear_layers = _gate_linear_layers(model)
    else:
        linear_layers = []

    return {
        "n_qubits": n_qubits,
        "n_modes": n_modes,
        "n_photons": n_photons,
        "linear_layers": json.dumps(linear_layers),
    }


def _instantiate_model(
    spec: ModelSpec,
    model_hps: dict[str, Any],
    input_size: int,
    output_size: int,
) -> Any:
    params = copy.deepcopy(model_hps)

    if spec.backend == "photonic" and spec.model != "data_reuploading":
        params["m"], params["n"] = get_photonic_mn(
            int(input_size), mode="feature_plus_one"
        )

    return fetch_model(
        spec.model,
        spec.backend,
        input_size,
        output_size,
        **params,
    )


def _prepare_gate_model(model_name: str, model: Any, x_np: np.ndarray) -> np.ndarray:
    model.initialize(x_np.shape[1], classes=np.array([-1, 1]))
    if hasattr(model, "transform"):
        return np.asarray(model.transform(x_np))
    return x_np


def _run_model_outputs(
    spec: ModelSpec,
    model_name: str,
    model: Any,
    x_torch: torch.Tensor,
    x_np: np.ndarray,
    batch_size: int,
) -> tuple[OutputAccumulator, OutputAccumulator, str | None]:
    """Return (model_accumulator, circuit_accumulator, note)."""
    note = None
    model_acc = OutputAccumulator()
    circuit_acc = OutputAccumulator()

    # Photonic data_reuploading (Merlin) is not a plain torch Module model object,
    # but exposes a torch quantum submodule that can be evaluated without fitting.
    if spec.backend == "photonic" and model_name == "data_reuploading":
        n = x_np.shape[0]
        bs = max(1, int(batch_size))
        q_model = getattr(model, "quantum_model", None)
        if q_model is None:
            return (
                model_acc,
                circuit_acc,
                "Photonic data_reuploading missing `quantum_model`; skipped.",
            )

        device = getattr(model, "device", torch.device("cpu"))
        if hasattr(q_model, "eval"):
            q_model.eval()

        with torch.no_grad():
            for start in range(0, n, bs):
                end = min(start + bs, n)
                x_chunk_t = torch.as_tensor(
                    x_np[start:end], dtype=torch.float32, device=device
                )
                encoded = x_chunk_t * float(getattr(model, "alpha", 1.0))

                if hasattr(q_model, "layer"):
                    probs_t = q_model.layer(encoded)
                    p10_t = probs_t[..., 1].reshape(-1, 1)
                else:
                    # Fallback: quantum_model forward returns p10 only.
                    p10_t = q_model(x_chunk_t).reshape(-1, 1)
                    probs_t = torch.cat((1.0 - p10_t, p10_t), dim=1)

                model_acc.update(p10_t.detach().cpu().numpy())
                circuit_acc.update(probs_t.detach().cpu().numpy())

        return (
            model_acc,
            circuit_acc,
            "Photonic data_reuploading model output reports p10 quantum feature; circuit output reports [p01, p10].",
        )

    # Torch models (photonic DQC/MPM and classical MLP)
    if isinstance(model, torch.nn.Module):
        model.eval()
        n = x_torch.shape[0]
        bs = max(1, int(batch_size))

        with torch.no_grad():
            if spec.backend == "photonic" and model_name in {
                "dressed_quantum_circuit",
                "dressed_quantum_circuit_reservoir",
            }:
                for start in range(0, n, bs):
                    end = min(start + bs, n)
                    scaled = model.scaling(x_torch[start:end])
                    circ_t = model.dqc[0](scaled)
                    out_t = model.dqc[1](circ_t)
                    circuit_acc.update(circ_t.detach().cpu().numpy())
                    model_acc.update(out_t.detach().cpu().numpy())
            elif spec.backend == "photonic" and model_name in {
                "multiple_paths_model",
                "multiple_paths_model_reservoir",
            }:
                for start in range(0, n, bs):
                    end = min(start + bs, n)
                    x_chunk = x_torch[start:end]
                    scaled = model.scaling(x_chunk)
                    circ_t = model.pqc(scaled)
                    post_t = (
                        model.post_circuit(circ_t)
                        if model.post_circuit is not None
                        else circ_t
                    )
                    mlp_in = torch.cat((x_chunk, post_t), dim=1)
                    out_t = model.mlp(mlp_in)
                    circuit_acc.update(circ_t.detach().cpu().numpy())
                    model_acc.update(out_t.detach().cpu().numpy())
            elif spec.backend == "photonic" and model_name == "q_rks":
                # QRKS has no direct final classifier output without fitting SVC.
                # Use quantum-feature outputs as the analyzable model proxy.
                w, b = get_random_w_b(model.R, model.input_size)
                for start in range(0, n, bs):
                    end = min(start + bs, n)
                    x_chunk_np = x_np[start:end]
                    x_r = get_x_r_i_s(x_chunk_np, w, b, model.R, model.gamma)
                    q_in = torch.tensor(x_r, dtype=torch.float32).view(
                        len(x_chunk_np) * model.R, -1
                    )
                    q_in = q_in * model.scaling
                    q_raw_t = model.pqc[0](q_in)
                    q_feat_t = model.pqc[1](q_raw_t)
                    q_raw_np = q_raw_t.detach().cpu().numpy()
                    q_feat_np = q_feat_t.detach().cpu().numpy()
                    q_dim = q_raw_np.shape[1] if q_raw_np.ndim == 2 else 1
                    circuit_acc.update(
                        q_raw_np.reshape(len(x_chunk_np), model.R * q_dim)
                    )
                    model_acc.update(q_feat_np.reshape(len(x_chunk_np), model.R))
                note = "q_rks model output reports quantum features (R channels) before SVC."
            else:
                for start in range(0, n, bs):
                    end = min(start + bs, n)
                    out_t = model(x_torch[start:end])
                    model_acc.update(out_t.detach().cpu().numpy())

        return model_acc, circuit_acc, note

    # Gate models
    if spec.backend == "gate":
        if model_name == "q_kernel_method_reservoir":
            return (
                model_acc,
                circuit_acc,
                "Gate IQP kernel has no direct per-sample circuit feature vector in this analysis.",
            )

        x_trans = _prepare_gate_model(model_name, model, x_np)

        if model_name in {
            "dressed_quantum_circuit",
            "dressed_quantum_circuit_reservoir",
        }:
            logits = np.asarray(model.chunked_forward(model.params_, x_trans))
            circuits = []
            for i in range(x_trans.shape[0]):
                enc = np.asarray(model.input_transform(model.params_, x_trans[i]))
                circuits.append(np.asarray(model.circuit(model.params_, enc)))
            model_acc.update(logits)
            circuit_acc.update(np.asarray(circuits))
            return model_acc, circuit_acc, note

        if model_name in {"multiple_paths_model", "multiple_paths_model_reservoir"}:
            logits = np.asarray(model.chunked_forward(model.params_, x_trans))
            circuits = [
                np.asarray(model.circuit(model.params_, x_trans[i]))
                for i in range(x_trans.shape[0])
            ]
            model_acc.update(logits)
            circuit_acc.update(np.asarray(circuits))
            return model_acc, circuit_acc, note

        if model_name == "data_reuploading":
            expvals = np.asarray(model.chunked_forward(model.params_, x_trans))
            obs = int(model.observable_weight)
            reduced = np.mean(expvals[:, :obs], axis=1)
            probs = np.c_[(1.0 - reduced) / 2.0, (1.0 + reduced) / 2.0]
            circuits = [
                np.asarray(model.circuit(model.params_, x_trans[i]))
                for i in range(x_trans.shape[0])
            ]
            model_acc.update(probs)
            circuit_acc.update(np.asarray(circuits))
            return model_acc, circuit_acc, note

        if model_name == "q_rks":
            features = np.asarray(model.transform(x_np))
            n = x_np.shape[0]
            n_eps = int(model.n_episodes)
            n_q = int(model.n_qubits_)
            input_features = np.zeros([n_eps, n, n_q])
            for e in range(n_eps):
                stacked_beta = np.stack([model.params_["betas"][e] for _ in range(n)])
                input_features[e] = (
                    model.params_["omegas"][e] @ model.scaler.transform(x_np).T
                    + stacked_beta.T
                ).T * model.scaling
            flat_in = np.reshape(input_features, (n * n_eps, -1))
            raw = np.asarray(model.forward(flat_in)).reshape(n, n_eps * n_q)
            model_acc.update(features)
            circuit_acc.update(raw)
            return (
                model_acc,
                circuit_acc,
                "Gate q_rks model output reports transformed feature vector before logistic head fit.",
            )

    # Classical non-torch models (e.g., rbf_svc, rks)
    if spec.backend == "classical":
        if model_name == "rks":
            k_train, _ = model.get_kernels(x_np, x_np)
            model_acc.update(k_train)
            return (
                model_acc,
                circuit_acc,
                "Classical rks output reports precomputed kernel matrix rows.",
            )
        return (
            model_acc,
            circuit_acc,
            "Model output requires fitting an sklearn classifier; skipped in this forward-only analysis.",
        )

    return (
        model_acc,
        circuit_acc,
        "No output extractor implemented for this model/backend.",
    )


def _analyze_one_configuration(
    d: int,
    spec: ModelSpec,
    model_hps: dict[str, Any],
    eval_ctx: EvalContext,
    measurement_strategy: str,
    grouping: str,
    ablation_label: str,
    ablation_type: str | None,
    batch_size: int,
    debug_memory: bool = False,
    force_gc_per_config: bool = False,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "d": d,
        "dataset": f"hidden_manifold_{d}_6",
        "backend": spec.backend,
        "model": spec.model,
        "ablation": ablation_label,
        "measurement_strategy": "probabilities"
        if measurement_strategy == "probs"
        else measurement_strategy,
        "grouping": grouping,
        "status": "ok",
        "note": None,
        "ablation_applied": None,
        "ablation_skipped": None,
        "ablation_reason": None,
    }

    hps = copy.deepcopy(model_hps)
    # Keep extension open for future grouping configs.
    hps["measurement_strategy"] = measurement_strategy
    hps["grouping"] = grouping
    try:
        model = _instantiate_model(spec, hps, eval_ctx.input_size, output_size=2)
    except Exception as exc:
        row.update({"status": "init_failed", "note": str(exc)})
        return row

    if ablation_type is not None:
        model_type = hps.get("type", "")
        ablation_n_photons = None
        if spec.backend == "photonic":
            if spec.model == "data_reuploading":
                ablation_n_photons = 1
            else:
                _, ablation_n_photons = get_photonic_mn(
                    int(eval_ctx.input_size), mode="feature_plus_one"
                )
        ablation_result = ablate_model(
            {"type": model_type, "name": spec.model, "model": model},
            ablation_type,
            measurement_strategy=measurement_strategy,
            n_photons=ablation_n_photons,
        )
        model = ablation_result.model
        row["ablation_applied"] = bool(ablation_result.applied)
        row["ablation_skipped"] = bool(ablation_result.skipped)
        row["ablation_reason"] = ablation_result.reason
    else:
        row["ablation_applied"] = False
        row["ablation_skipped"] = False

    architecture = _extract_architecture(
        backend=spec.backend,
        model_name=spec.model,
        model=model,
        model_hps=hps,
        d=d,
    )
    row.update(architecture)

    try:
        if debug_memory:
            _progress(
                "memory before outputs "
                f"(d={d}, backend={spec.backend}, model={spec.model}, "
                f"ablation={ablation_label}, measurement={row['measurement_strategy']}): "
                f"rss_mb={_rss_mb():.2f}"
            )
        _progress(
            "running outputs "
            f"(d={d}, backend={spec.backend}, model={spec.model}, "
            f"ablation={ablation_label}, measurement={row['measurement_strategy']}, batch_size={batch_size})"
        )
        model_acc, circuit_acc, note = _run_model_outputs(
            spec=spec,
            model_name=spec.model,
            model=model,
            x_torch=eval_ctx.x_eval_torch,
            x_np=eval_ctx.x_eval_np,
            batch_size=batch_size,
        )
        row["note"] = note
        row.update({f"model_{k}": v for k, v in model_acc.as_dict().items()})
        row.update({f"circuit_{k}": v for k, v in circuit_acc.as_dict().items()})
    except Exception as exc:
        tb = traceback.format_exc()
        _progress(
            "eval failed "
            f"(d={d}, backend={spec.backend}, model={spec.model}, "
            f"ablation={ablation_label}, measurement={row['measurement_strategy']}): {exc}\n{tb}"
        )
        row.update({"status": "eval_failed", "note": f"{exc}\n{tb}"})
    finally:
        # Release per-config references aggressively when debugging memory growth.
        if "model" in locals():
            del model
        if force_gc_per_config:
            gc.collect()
        if debug_memory:
            _progress(
                "memory after outputs "
                f"(d={d}, backend={spec.backend}, model={spec.model}, "
                f"ablation={ablation_label}, measurement={row['measurement_strategy']}): "
                f"rss_mb={_rss_mb():.2f}"
            )

    return row


def _build_model_specs(include_classical: bool) -> list[ModelSpec]:
    specs = [ModelSpec("photonic", m) for m in PHOTONIC_MODELS]
    # Gate backend does not support plain q_kernel_method name.
    specs.extend(ModelSpec("gate", m) for m in GATE_MODELS if m != "q_kernel_method")
    if include_classical:
        specs.extend(ModelSpec("classical", m) for m in CLASSICAL_MODELS)
    return specs


def _measurement_configs_for_model(
    spec: ModelSpec, args: argparse.Namespace
) -> list[dict[str, str]]:
    if spec.backend != "photonic":
        return [{"measurement_strategy": "none", "grouping": "none"}]
    if args.photonic_measurement == "default":
        return DEFAULT_PHOTONIC_MEASUREMENT_CONFIGS
    if args.photonic_measurement == "mode_only":
        return [{"measurement_strategy": "mode_expectations", "grouping": "none"}]
    return [{"measurement_strategy": "probs", "grouping": "none"}]


def _compatible_ablations(
    model_type: str, model_name: str
) -> list[tuple[str, str | None]]:
    options: list[tuple[str, str | None]] = [("base", None)]
    for label, ablation_type in [
        ("abla_q", ABLATION_QUANTUM),
        ("abla_c", ABLATION_CLASSICAL),
    ]:
        compatible, _ = can_apply_ablation(
            model_type=model_type, model_name=model_name, ablation_type=ablation_type
        )
        if compatible:
            options.append((label, ablation_type))
    return options


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze model/circuit outputs across hidden_manifold_d_6."
    )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_out_dir = (
        PROJECT_ROOT / f"tabular_data/results/analyze_model_outputs/{timestamp}"
    )
    parser.add_argument("--d-min", type=int, default=2)
    parser.add_argument("--d-max", type=int, default=20)
    parser.add_argument("--manifold-dim", type=int, default=6)
    parser.add_argument(
        "--max-samples",
        "--sample-limit",
        dest="max_samples",
        type=int,
        default=25,
        help="Max training samples per d for output stats (default: 25).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for iterative forward passes to reduce peak memory usage.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--photonic-measurement",
        type=str,
        default="default",
        choices=["default", "mode_only", "probs_only"],
        help="default = (mode_expectations,none) and (probs,none).",
    )
    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument(
        "--debug-memory",
        action="store_true",
        help="Print process RSS (MB) before/after each configuration.",
    )
    parser.add_argument(
        "--force-gc-per-config",
        action="store_true",
        help="Run gc.collect() after every configuration.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=default_out_dir / "model_output_analysis.csv",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=default_out_dir / "model_output_analysis.md",
    )
    return parser.parse_args()


def _select_d_values(d_min: int, d_max: int) -> list[int]:
    """Select at most three representative d values: min, midpoint, max.

    Midpoint uses integer floor division and duplicate values are removed while
    preserving order, so edge cases like d_max == d_min + 1 collapse safely.
    """
    if d_min > d_max:
        raise ValueError(f"d_min must be <= d_max, got d_min={d_min}, d_max={d_max}")

    midpoint = (d_min + d_max) // 2
    values = [d_min, midpoint, d_max]

    selected: list[int] = []
    for d in values:
        if d not in selected:
            selected.append(d)
    return selected


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _render_markdown(df: pd.DataFrame, out_md: Path) -> None:
    cols = [
        "d",
        "backend",
        "model",
        "ablation",
        "measurement_strategy",
        "grouping",
        "n_qubits",
        "n_modes",
        "n_photons",
        "linear_layers",
        "model_output_size",
        "model_sum_output_mean",
        "model_sum_output_variance",
        "circuit_output_size",
        "circuit_sum_output_mean",
        "circuit_sum_output_variance",
        "ablation_applied",
        "ablation_reason",
        "status",
        "note",
    ]
    present = [c for c in cols if c in df.columns]
    column_descriptions = {
        "d": "Input dimension.",
        "backend": "Model backend family (photonic/gate/classical).",
        "model": "Model name.",
        "ablation": "Ablation label (`base`, `abla_q`, `abla_c`).",
        "measurement_strategy": "Photonic measurement mode; `none` for non-photonic models.",
        "grouping": "Output grouping strategy.",
        "n_qubits": "Gate-model qubit count (when applicable).",
        "n_modes": "Photonic mode count (when applicable).",
        "n_photons": "Photonic photon count (when applicable).",
        "linear_layers": "Serialized list of linear-layer shapes.",
        "model_output_size": "Flattened size of model output vector per sample.",
        "model_sum_output_mean": "Mean of per-sample model-output sums.",
        "model_sum_output_variance": "Variance of per-sample output sums.",
        "circuit_output_size": "Flattened size of circuit-level output vector per sample.",
        "circuit_sum_output_mean": "Mean of per-sample circuit-output sums.",
        "circuit_sum_output_variance": "Variance of per-sample circuit-output sums.",
        "ablation_applied": "Whether ablation was applied.",
        "ablation_reason": "Why ablation was skipped/limited (if any).",
        "status": "Execution status (`ok`, `init_failed`, `eval_failed`).",
        "note": "Additional model-specific context.",
    }

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Model Output Analysis\n\n")
        f.write("Dataset family: `hidden_manifold_d_6`\n\n")
        f.write("## Column Glossary\n\n")
        for col in present:
            desc = column_descriptions.get(col, "No description available.")
            f.write(f"- `{col}`: {desc}\n")
        f.write("\n")

        f.write("## Results By d and Model\n\n")
        if len(df) == 0:
            f.write("_No rows to display._\n")
            return

        table_df = df[present].copy()
        ablation_order = {"base": 0, "abla_q": 1, "abla_c": 2}

        for d_value in sorted(table_df["d"].dropna().unique().tolist()):
            d_df = table_df.loc[table_df["d"] == d_value].copy()
            f.write(f"### d = {int(d_value)}\n\n")

            model_keys = ["backend", "model", "measurement_strategy", "grouping"]
            ordered_groups = list(
                d_df[model_keys]
                .drop_duplicates()
                .sort_values(model_keys, kind="stable")
                .itertuples(index=False, name=None)
            )

            for idx, (backend, model, measurement_strategy, grouping) in enumerate(
                ordered_groups, start=1
            ):
                model_df = d_df.loc[
                    (d_df["backend"] == backend)
                    & (d_df["model"] == model)
                    & (d_df["measurement_strategy"] == measurement_strategy)
                    & (d_df["grouping"] == grouping)
                ].copy()

                model_df["_ablation_order"] = model_df["ablation"].map(
                    lambda v: ablation_order.get(v, 99)
                )
                model_df = model_df.sort_values(
                    ["_ablation_order", "ablation"], kind="stable"
                ).drop(columns=["_ablation_order"])

                f.write(
                    f"#### `{backend}/{model}` | measurement=`{measurement_strategy}` | grouping=`{grouping}`\n\n"
                )
                f.write(
                    f"*Table {idx}: ablation comparison (`base`, `abla_q`, `abla_c` when available).*"
                    "\n\n"
                )
                f.write(model_df.to_markdown(index=False))
                f.write("\n\n---\n\n")


def main() -> None:
    args = parse_args()

    _progress("starting analysis")
    if args.debug_memory:
        _progress(f"initial rss_mb={_rss_mb():.2f}")
    _progress(
        f"args: d_min={args.d_min}, d_max={args.d_max}, manifold_dim={args.manifold_dim}, "
        f"max_samples={args.max_samples}, batch_size={args.batch_size}, random_state={args.random_state}"
    )
    d_values = _select_d_values(args.d_min, args.d_max)
    specs = _build_model_specs(include_classical=not args.skip_classical)
    _progress(f"selected d values: {d_values}")
    _progress(f"total model specs: {len(specs)}")

    rows: list[dict[str, Any]] = []

    for d in d_values:
        _progress(f"loading evaluation context for d={d}")
        eval_ctx = _load_eval_context(
            d=d,
            manifold_dim=args.manifold_dim,
            sample_limit=args.max_samples,
            random_state=args.random_state,
        )
        _progress(
            f"loaded eval context for d={d}: samples={eval_ctx.x_eval_np.shape[0]}, input_size={eval_ctx.input_size}"
        )

        for spec in specs:
            _progress(f"preparing spec backend={spec.backend} model={spec.model}")
            base_hps = _single_run_model_hps(
                spec.backend, spec.model, args.random_state
            )
            ablations = _compatible_ablations(base_hps.get("type", ""), spec.model)
            measurement_configs = _measurement_configs_for_model(spec, args)

            for meas_cfg in measurement_configs:
                for ablation_label, ablation_type in ablations:
                    row = _analyze_one_configuration(
                        d=d,
                        spec=spec,
                        model_hps=base_hps,
                        eval_ctx=eval_ctx,
                        measurement_strategy=meas_cfg["measurement_strategy"],
                        grouping=meas_cfg["grouping"],
                        ablation_label=ablation_label,
                        ablation_type=ablation_type,
                        batch_size=args.batch_size,
                        debug_memory=args.debug_memory,
                        force_gc_per_config=args.force_gc_per_config,
                    )
                    rows.append(row)
            _progress(f"finished spec backend={spec.backend} model={spec.model}")

        _progress(f"completed d={d}")

    _progress("building dataframe")
    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["d", "backend", "model", "measurement_strategy", "ablation"],
        kind="stable",
    ).reset_index(drop=True)

    _progress(f"saving outputs to csv={args.out_csv} md={args.out_md}")
    _ensure_parent(args.out_csv)
    _ensure_parent(args.out_md)
    df.to_csv(args.out_csv, index=False)
    _render_markdown(df, args.out_md)

    print(f"Saved CSV: {args.out_csv}")
    print(f"Saved Markdown report: {args.out_md}")
    print(f"Rows: {len(df)}")
    print(f"Analyzed d values: {d_values}")


if __name__ == "__main__":
    main()
