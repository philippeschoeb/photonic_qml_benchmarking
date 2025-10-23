import json

hps = {
    "torch": {
        "criterion": "CrossEntropyLoss",
        "output_size": 2,
        "optimizer": "Adam",
        "scheduler": "None",
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "momentum": 0.9,
        "weight_decay": 0.0,
    },
    "reuploading": {
        "track_history": True,
        "lr": 1e-3,
        "batch_size": 32,
        "patience": 50,
        "tau": 1.0,
        "convergence_tolerance": 1e-6,
        "output_size": 2,
    },
    "sklearn_q_kernel": {"optimizer": "Adam", "lr": 1e-3, "output_size": 2},
    "sklearn_kernel": {"output_size": 2},
    "sklearn": {"output_size": 2},
    "jax_sklearn_gate": {"output_size": 2},
    "sklearn_gate": {"output_size": 2},
}

with open("../single_run/training_hps.json", "w") as fp:
    json.dump(hps, fp, indent=4)
