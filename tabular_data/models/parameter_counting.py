import logging

import numpy as np


def _param_size(value) -> int:
    if value is None:
        return 0
    if isinstance(value, dict):
        return sum(_param_size(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_param_size(v) for v in value)
    if hasattr(value, "numel"):
        try:
            return int(value.numel())
        except TypeError:
            pass
    size_attr = getattr(value, "size", None)
    if isinstance(size_attr, (int, np.integer)):
        return int(size_attr)
    try:
        return int(np.asarray(value).size)
    except Exception:
        return 0


def _count_torch_trainable_params(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _count_torch_quantum_params(model_name: str, module) -> int:
    if model_name in {"dressed_quantum_circuit", "dressed_quantum_circuit_reservoir"}:
        return sum(
            p.numel()
            for name, p in module.named_parameters()
            if p.requires_grad and name.startswith("dqc.0")
        )
    if model_name in {"multiple_paths_model", "multiple_paths_model_reservoir"}:
        return sum(
            p.numel()
            for name, p in module.named_parameters()
            if p.requires_grad and name.startswith("pqc")
        )
    return 0


def _count_gate_jax_param_split(params: dict | None) -> tuple[int, int]:
    if not params:
        return 0, 0
    quantum_keys = {"circuit_weights", "thetas", "omegas", "betas"}
    classical_keys = {"input_weights", "output_weights", "weights", "alphas"}
    quantum = 0
    classical = 0
    for key, value in params.items():
        size = _param_size(value)
        if key in quantum_keys:
            quantum += size
        elif key in classical_keys:
            classical += size
        else:
            classical += size
    return quantum, classical


def _count_gate_rks_trained_params(params: dict | None) -> int:
    if not params:
        return 0
    # For gate q_rks, only the linear model coefficients are trained.
    return _param_size(params.get("weights"))


def count_parameters(model_dict, hp_opt=False):
    """Return model complexity metrics using trainable parameters only.

    Args:
        model_dict: Dict containing at least:
            - ``type``: model family/type identifier
            - ``model``: fitted model object
            - ``name``: base model name (optional, used for splits)
        hp_opt: When True, unwrap sklearn search wrappers before counting.

    Returns:
        Tuple ``(num_params, num_quantum_params, num_classical_params, num_support_vectors)``.
    """
    # If the model comes from HP optimization, we usually take the model out of its SK wrapper.
    # For multiclass gate wrappers using one-vs-rest, keep the wrapper to aggregate counts.
    if hp_opt:
        wrapped_model = model_dict["model"]
        unwrapped_model = (
            wrapped_model
            if getattr(wrapped_model, "ovr_models_", None)
            else wrapped_model.model
        )
        return count_parameters(
            {
                "type": model_dict["type"],
                "name": model_dict.get("name"),
                "model": unwrapped_model,
            }
        )

    model_type = model_dict["type"]
    model = model_dict["model"]
    model_name = model_dict.get("name")
    num_params = 0
    num_quantum_params = 0
    num_classical_params = 0
    num_support_vectors = 0

    if model_type == "torch":
        num_params = _count_torch_trainable_params(model)
        num_quantum_params = _count_torch_quantum_params(model_name, model)
        num_classical_params = max(0, num_params - num_quantum_params)
    elif model_type == "reuploading":
        num_quantum_params = _count_torch_trainable_params(model.quantum_model)
        num_classical_params = 0
        num_params = num_quantum_params
    elif model_type == "sklearn_q_kernel":
        optimizable_model = model.quantum_kernel
        # Count only trainable quantum-kernel parameters.
        num_quantum_params = _count_torch_trainable_params(optimizable_model)
        num_classical_params = 0
        num_params = num_quantum_params
        num_support_vectors = len(model.model.support_)
    elif model_type == "sklearn_kernel":
        if hasattr(model, "pqc"):
            num_params = _count_torch_trainable_params(model.pqc)
            num_quantum_params = sum(
                p.numel()
                for name, p in model.pqc.named_parameters()
                if p.requires_grad and name.startswith("0.")
            )
            num_classical_params = max(0, num_params - num_quantum_params)
        else:
            num_params = 0
            num_quantum_params = 0
            num_classical_params = 0
        num_support_vectors = len(model.model.support_)
    elif model_type == "sklearn":
        num_params = 0
        num_quantum_params = 0
        num_classical_params = 0
        num_support_vectors = len(model.model.support_)
    elif model_type == "jax_sklearn_gate":
        if getattr(model, "ovr_models_", None):
            num_quantum_params = 0
            num_classical_params = 0
            for _, binary_model in model.ovr_models_:
                q_count, c_count = _count_gate_jax_param_split(
                    getattr(binary_model, "params_", None)
                )
                num_quantum_params += q_count
                num_classical_params += c_count
        else:
            num_quantum_params, num_classical_params = _count_gate_jax_param_split(
                getattr(model, "params_", None)
            )
        num_params = num_quantum_params + num_classical_params
    elif model_type == "gate_rks":
        if getattr(model, "ovr_models_", None):
            num_classical_params = 0
            for _, binary_model in model.ovr_models_:
                num_classical_params += _count_gate_rks_trained_params(
                    getattr(binary_model, "params_", None)
                )
        else:
            num_classical_params = _count_gate_rks_trained_params(
                getattr(model, "params_", None)
            )
        num_quantum_params = 0
        num_params = num_quantum_params + num_classical_params
    elif model_type == "sklearn_gate":
        num_params = 0
        num_quantum_params = 0
        num_classical_params = 0
        if getattr(model, "ovr_models_", None):
            num_support_vectors = sum(
                len(binary_model.svm.support_) for _, binary_model in model.ovr_models_
            )
        else:
            num_support_vectors = len(model.svm.support_)
    else:
        raise NotImplementedError(f"Unknown model type: {model_type}")
    logging.warning(f"Number of parameters: {num_params}")
    logging.warning(f"Number of quantum parameters: {num_quantum_params}")
    logging.warning(f"Number of classical parameters: {num_classical_params}")
    logging.warning(f"Number of support vectors: {num_support_vectors}")
    return num_params, num_quantum_params, num_classical_params, num_support_vectors
