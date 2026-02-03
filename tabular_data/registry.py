DATASET_BASE_NAMES = [
    "downscaled_mnist_pca",
    "hidden_manifold",
    "two_curves",
]

PHOTONIC_MODELS = [
    "dressed_quantum_circuit",
    "dressed_quantum_circuit_reservoir",
    "multiple_paths_model",
    "multiple_paths_model_reservoir",
    "data_reuploading",
    "q_kernel_method",
    "q_kernel_method_reservoir",
    "q_rks",
]

GATE_MODELS = [
    "dressed_quantum_circuit",
    "dressed_quantum_circuit_reservoir",
    "multiple_paths_model",
    "multiple_paths_model_reservoir",
    "data_reuploading",
    "q_kernel_method",
    "q_kernel_method_reservoir",
    "q_rks",
]

CLASSICAL_MODELS = [
    "mlp",
    "rbf_svc",
    "rks",
]

RUN_TYPES = [
    "single",
    "hyperparam_search",
]

BACKENDS = [
    "photonic",
    "gate",
    "classical",
]

HP_PROFILES = [
    "minimal",
    "full",
]


def get_dataset_base_name(dataset: str) -> str:
    return next(
        name for name in DATASET_BASE_NAMES if dataset.startswith(name)
    )


def list_models() -> list[str]:
    return sorted(set(PHOTONIC_MODELS + GATE_MODELS + CLASSICAL_MODELS))
