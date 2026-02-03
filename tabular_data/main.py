"""
Main run file for tabular data benchmarking.
"""

import argparse
from run_scripts.run import run_single, run_search
from helper import architecture_help
from registry import CLASSICAL_MODELS, HP_PROFILES, RUN_TYPES, get_dataset_base_name, list_models
import wandb
import sys


def main(
    dataset,
    model,
    architecture,
    backend,
    run_type,
    random_state,
    use_wandb,
    hp_profile,
):
    # Setup Wandb
    if use_wandb:
        wandb.init(
            project="photonic-qml-benchmarking-tabular",
            name=f"{model}&{dataset}&{run_type}",
            tags=[model, dataset, run_type, architecture, backend],
        )
        wandb.run.summary["model"] = model
        wandb.run.summary["dataset"] = dataset
        wandb.run.summary["run_type"] = run_type
        wandb.run.summary["architecture"] = architecture
        wandb.run.summary["backend"] = backend
        wandb.run.summary["random_state"] = random_state

    if run_type == "single":
        run_single(dataset, model, architecture, backend, random_state, use_wandb)
    elif run_type == "hyperparam_search":
        run_search(
            dataset, model, architecture, backend, random_state, use_wandb, hp_profile
        )
    else:
        raise ValueError(f"Invalid run type: {run_type}")

    # Close Wandb
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmarking with different datasets and models"
    )

    # Required arguments
    parser.add_argument(
        "--dataset", type=str, required=False, help="Name of the dataset to use"
    )

    parser.add_argument(
        "--model", type=str, required=False, help="Name of the model to use"
    )

    parser.add_argument(
        "--architecture",
        type=str,
        required=False,
        help="Name of the architecture to use",
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=["photonic", "gate"],
        default=None,
        help="Choose the QML backend: 'photonic' for photonic-based, 'gate' for gate-based",
    )

    parser.add_argument(
        "--run_type",
        type=str,
        choices=RUN_TYPES,
        required=False,
        help="Type of run: single run or hyperparameter search",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        required=False,
        default=None,
        help="Random state to use",
    )

    parser.add_argument(
        "--no-wandb", action="store_false", dest="wandb", help="Disable wandb"
    )
    parser.set_defaults(wandb=True)  # default is True
    parser.add_argument(
        "--hp_profile",
        type=str,
        choices=HP_PROFILES,
        default=None,
        help="Hyperparameter search profile (used only for run_type=hyperparam_search)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List supported datasets/models/run types and exit",
    )

    # Parse and handle args
    args = parser.parse_args()
    dataset = args.dataset
    model = args.model
    run_type = args.run_type
    architecture = args.architecture
    backend = args.backend
    hp_profile = args.hp_profile

    if args.list:
        print("Datasets:")
        print("  - downscaled_mnist_pca_{d}")
        print("  - hidden_manifold_{d}_{m}")
        print("  - two_curves_{d}_{D}")
        print("Models:")
        for name in list_models():
            print(f"  - {name}")
        print("Run types:")
        for name in RUN_TYPES:
            print(f"  - {name}")
        print("Hyperparam profiles:")
        for name in HP_PROFILES:
            print(f"  - {name}")
        sys.exit(0)

    if dataset is None:
        parser.error("--dataset is required unless --list is set")
    if model is None:
        parser.error("--model is required unless --list is set")
    if run_type is None:
        parser.error("--run_type is required unless --list is set")

    if model == "data_reuploading_reservoir":
        raise ValueError(
            "data_reuploading_reservoir is not supported in this benchmarking study. Please select a different model."
        )

    use_wandb = args.wandb
    if backend is None:
        backend = "photonic"
    if hp_profile is None:
        hp_profile = "minimal"
    # Validate dataset base name early for clearer errors
    try:
        get_dataset_base_name(dataset)
    except StopIteration as exc:
        raise ValueError(f"Unknown dataset name: {dataset}") from exc

    # If classical model, change backend to classical
    if model in CLASSICAL_MODELS:
        backend = "classical"
    print(f"Backend type: {backend}")

    # If the user wants help with different architecture configurations
    if architecture == "help":
        print(architecture_help(model, backend))
        sys.exit(0)

    if architecture is None:
        architecture = "default"

    random_state = args.random_state
    if random_state is None:
        random_state = 42
    main(
        dataset,
        model,
        architecture,
        backend,
        run_type,
        random_state,
        use_wandb,
        hp_profile,
    )
