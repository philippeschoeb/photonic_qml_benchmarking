# ruff: noqa: F401

import logging
from datetime import datetime
import os
import random
from typing import Optional
import numpy as np
import torch
import joblib
import json
import wandb
from time import time

from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, GridSearchCV
from skopt import BayesSearchCV

from run_scripts.single_run import get_hyperparams as single_hps
from run_scripts.hyperparam_search_run import get_hyperparams_halving_grid
from run_scripts.hyperparam_search_run import get_hyperparams_bayes
from run_scripts.hyperparam_search_run import serialize_param_grid
from run_scripts.run_output_utils import (
    _count_search_configurations,
    save_hp_search_summary,
    save_long_training_event,
)
from datasets.fetch_data import fetch_data, fetch_sk_data
from models.fetch_model import fetch_model, fetch_sk_model
from models.ablation import (
    can_apply_ablation,
    parse_ablation_model_name,
)
from models.parameter_counting import count_parameters
from training.distribute_training import distribute_training
from utils.save_metrics import (
    save_train_losses_accs,
    save_train_loss_final_accs,
    save_final_accs,
    save_hyperparams,
    save_sk_train_losses_accs,
    save_search_hyperparams,
)
from utils.photonic_dims import get_photonic_mn


def run_single(
    dataset,
    model,
    architecture,
    backend,
    random_state,
    use_wandb,
    big_script_name=None,
    max_train_time_seconds=None,
):
    ablation_spec = parse_ablation_model_name(model)
    base_model = ablation_spec.base_model
    ablation_type = ablation_spec.ablation_type

    # Setup logging and return save directory
    save_dir = set_up_logging(
        dataset=dataset,
        model=model,
        backend=backend,
        run_type="single",
        big_script_name=big_script_name,
    )
    # Setup random state across torch, numpy and random
    random_state = set_up_random_state(random_state)

    # Log 'single run' beginning
    separator = "##" * 40
    logging.warning(
        f"SINGLE RUN {separator}\nDataset: {dataset}\nModel: {model}\nArchitecture: {architecture}\nBackend: {backend}\nRunType: single\nRandomState: {random_state}\n"
    )
    if max_train_time_seconds is not None:
        logging.warning(
            "Single-run max training time budget: %.1f seconds",
            float(max_train_time_seconds),
        )

    # Fetch hyperparameters for single run
    hyperparams = single_hps(dataset, base_model, architecture, backend, random_state)
    dataset_hps = hyperparams["dataset"]
    model_hps = hyperparams["model"]
    training_hps = hyperparams["training"]
    if max_train_time_seconds is not None:
        training_hps["max_train_time_seconds"] = float(max_train_time_seconds)
    if ablation_type is not None:
        can_apply, skip_reason = can_apply_ablation(
            model_type=model_hps["type"],
            model_name=base_model,
            ablation_type=ablation_type,
        )
        if not can_apply:
            logging.warning("Skipping run `%s`: %s", model, skip_reason)
            return
        training_hps["ablation_type"] = ablation_type
    # Fetch data
    train_loader, test_loader, x_train, x_test, y_train, y_test = fetch_data(
        dataset, random_state, **dataset_hps
    )
    # Fill default counts if None so wandb shows actual sizes
    if dataset_hps.get("num_train") is None:
        dataset_hps["num_train"] = len(x_train)
    if dataset_hps.get("num_test") is None:
        dataset_hps["num_test"] = len(x_test)
    # Setup hyperparameters in wandb
    if use_wandb:
        wandb.run.config.update(hyperparams)
    # Get input size
    input_size = x_train.shape[1]
    # Get model type
    model_type = model_hps["type"]

    # Align photonic modes/photons with feature count.
    if backend == "photonic" and base_model != "data_reuploading":
        model_hps["m"], model_hps["n"] = get_photonic_mn(input_size)

    # Fetch model
    model = fetch_model(
        base_model, backend, input_size, training_hps["output_size"], **model_hps
    )
    # Define model dictionary
    model_dict = {"type": model_hps["type"], "name": model_hps["name"], "model": model}

    # Training
    logging.warning("Starting training")
    start = time()
    results_dict = distribute_training(
        model_dict,
        train_loader,
        test_loader,
        x_train,
        x_test,
        y_train,
        y_test,
        **training_hps,
    )
    end = time()
    training_time = end - start
    logging.warning(f"Training completed in {training_time} seconds")
    if results_dict.get("timed_out"):
        logging.warning(
            "Training run was cut short due to max_train_time_seconds=%s.",
            results_dict.get("max_train_time_seconds", max_train_time_seconds),
        )

    # Number of parameters after training (for 'sklearn_gate' models)
    (
        num_params,
        num_quantum_params,
        num_classical_params,
        num_support_vectors,
    ) = count_parameters(model_dict)

    # Save metrics based on model type
    if model_type == "torch":
        save_train_losses_accs(
            results_dict["train_losses"],
            results_dict["test_losses"],
            results_dict["train_accs"],
            results_dict["test_accs"],
            save_dir,
            use_wandb,
        )
    elif model_type == "reuploading":
        save_train_loss_final_accs(
            results_dict["train_losses"],
            results_dict["final_train_acc"],
            results_dict["final_test_acc"],
            save_dir,
            use_wandb,
        )
    elif model_type == "sklearn_q_kernel":
        save_train_loss_final_accs(
            results_dict["train_losses"],
            results_dict["final_train_acc"],
            results_dict["final_test_acc"],
            save_dir,
            use_wandb,
        )
    elif model_type == "sklearn_kernel":
        save_final_accs(
            results_dict["final_train_acc"],
            results_dict["final_test_acc"],
            save_dir,
            use_wandb,
        )
    elif model_type == "sklearn":
        save_final_accs(
            results_dict["final_train_acc"],
            results_dict["final_test_acc"],
            save_dir,
            use_wandb,
        )
    elif model_type == "jax_sklearn_gate":
        loss_history = results_dict["model"].loss_history_
        save_train_loss_final_accs(
            loss_history,
            results_dict["final_train_acc"],
            results_dict["final_test_acc"],
            save_dir,
            use_wandb,
        )
    elif model_type == "sklearn_gate" or model_type == "gate_rks":
        save_final_accs(
            results_dict["final_train_acc"],
            results_dict["final_test_acc"],
            save_dir,
            use_wandb,
        )
    else:
        raise NotImplementedError(f"Unknown model type: {model_type}")
    logging.warning(f"Metrics saved at {save_dir}\n\n")

    # Save hyperparameters, number of parameters / support vectors and training time
    model_hps["num_params"] = num_params
    model_hps["num_quantum_params"] = num_quantum_params
    model_hps["num_classical_params"] = num_classical_params
    model_hps["num_support_vectors"] = num_support_vectors
    training_hps["training_time"] = training_time
    save_hyperparams(
        {"dataset": dataset_hps, "model": model_hps, "training": training_hps}, save_dir
    )
    if results_dict.get("timed_out"):
        save_long_training_event(
            save_dir=save_dir,
            dataset=dataset,
            big_script_name=big_script_name,
            event={
                "dataset": dataset,
                "model": model,
                "backend": backend,
                "run_type": "single",
                "status": "cut_short",
                "event_type": "max_train_time_reached",
                "source": results_dict.get("timeout_stage", "training_loop"),
                "reason": "Reached max_train_time_seconds during training.",
                "max_train_time_seconds": results_dict.get(
                    "max_train_time_seconds", max_train_time_seconds
                ),
                "run_dir": save_dir,
                "hyperparameters": {
                    "dataset": dataset_hps,
                    "model": model_hps,
                    "training": training_hps,
                },
            },
        )
    # Also in wandb
    if use_wandb:
        wandb.log(
            {
                "number_of_params": num_params,
                "number_of_quantum_params": num_quantum_params,
                "number_of_classical_params": num_classical_params,
                "num_support_vectors": num_support_vectors,
                "training_time": training_time,
            }
        )
    return


def run_search(
    dataset,
    model,
    architecture,
    backend,
    random_state,
    use_wandb,
    hp_profile,
    big_script_name=None,
    max_train_time_seconds=None,
):
    ablation_spec = parse_ablation_model_name(model)
    base_model = ablation_spec.base_model
    ablation_type = ablation_spec.ablation_type

    # Setup logging and return save directory
    save_dir = set_up_logging(
        dataset=dataset,
        model=model,
        backend=backend,
        run_type="hyperparam_search",
        big_script_name=big_script_name,
    )
    # Setup random state across torch, numpy and random
    random_state = set_up_random_state(random_state)

    # Log 'hyperparameter search' beginning
    separator = "##" * 40
    logging.warning(
        f"HYPERPARAM SEARCH {separator}\nDataset: {dataset}\nModel: {model}\nArchitecture: {architecture}\nBackend: {backend}\nRunType: hyperparam_search\nRandomState: {random_state}"
    )

    # Load model_search_assignment
    with open(
        "hyperparameters/hyperparam_search/model_search_assignment.json", "r"
    ) as f:
        model_search_assignment = json.load(f)
    halving_grid_list = model_search_assignment["halving_grid"]
    bayes_list = model_search_assignment["bayes"]
    grid_list = model_search_assignment["grid"]

    # Fetch hyperparameter grid for hyperparameter search
    force_full_grid = hp_profile == "minimal"
    if force_full_grid:
        param_grid, dataset_hps, model_hps, training_hps = get_hyperparams_halving_grid(
            dataset, base_model, architecture, backend, random_state, hp_profile
        )
        search_type = "grid_search"
    elif base_model in halving_grid_list:
        param_grid, dataset_hps, model_hps, training_hps = get_hyperparams_halving_grid(
            dataset, base_model, architecture, backend, random_state, hp_profile
        )
        search_type = "halving_grid_search"
    elif base_model in bayes_list:
        param_grid, dataset_hps, model_hps, training_hps = get_hyperparams_bayes(
            dataset, base_model, architecture, backend, random_state, hp_profile
        )
        search_type = "bayes_search"
    elif base_model in grid_list:
        param_grid, dataset_hps, model_hps, training_hps = get_hyperparams_halving_grid(
            dataset, base_model, architecture, backend, random_state, hp_profile
        )
        search_type = "grid_search"
    else:
        raise NotImplementedError(f"Unknown model name for hp search: {base_model}")
    logging.warning(f"Hyperparameters search type: {search_type}")
    logging.warning(f"Hyperparameters profile: {hp_profile}")
    if max_train_time_seconds is not None:
        logging.warning(
            "Per-fit max training time budget requested: %.1f seconds",
            float(max_train_time_seconds),
        )
    number_of_configs = _count_search_configurations(param_grid)
    logging.warning(
        f"Number of hyperparameter configurations considered: {number_of_configs}"
    )

    param_grid_serializable = serialize_param_grid(param_grid)
    if isinstance(param_grid_serializable, list):
        param_grid_serializable = {"grids": param_grid_serializable}
    param_grid_serializable["number_of_configs"] = number_of_configs
    if max_train_time_seconds is not None:
        budget = float(max_train_time_seconds)
        training_hps["max_train_time_seconds"] = [budget]
        timeout_meta = {
            "dataset": dataset,
            "model": model,
            "backend": backend,
            "run_type": "hyperparam_search",
            "search_type": search_type,
            "hp_profile": hp_profile,
        }
        timeout_meta_json = json.dumps(timeout_meta, sort_keys=True)
        if isinstance(param_grid, list):
            for grid in param_grid:
                grid["training_params__max_train_time_seconds"] = [budget]
                grid["training_params__timeout_events_metadata"] = [timeout_meta_json]
                if big_script_name:
                    repo_root = os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    )
                    dataset_slug = dataset.replace("/", "_")
                    timeout_events_path = os.path.join(
                        repo_root,
                        "tabular_data",
                        "results",
                        big_script_name,
                        f"long_training_{dataset_slug}.jsonl",
                    )
                else:
                    timeout_events_path = os.path.join(
                        save_dir, "long_training_events.jsonl"
                    )
                grid["training_params__timeout_events_path"] = [timeout_events_path]
        else:
            param_grid["training_params__max_train_time_seconds"] = [budget]
            param_grid["training_params__timeout_events_metadata"] = [timeout_meta_json]
            if big_script_name:
                repo_root = os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                dataset_slug = dataset.replace("/", "_")
                timeout_events_path = os.path.join(
                    repo_root,
                    "tabular_data",
                    "results",
                    big_script_name,
                    f"long_training_{dataset_slug}.jsonl",
                )
            else:
                timeout_events_path = os.path.join(save_dir, "long_training_events.jsonl")
            param_grid["training_params__timeout_events_path"] = [timeout_events_path]

    # Get model type
    model_type = model_hps["type"][0]
    if ablation_type is not None:
        can_apply, skip_reason = can_apply_ablation(
            model_type=model_type,
            model_name=base_model,
            ablation_type=ablation_type,
        )
        if not can_apply:
            logging.warning(
                "Skipping run `%s`: %s",
                model,
                skip_reason,
            )
            return
        if isinstance(param_grid, list):
            for grid in param_grid:
                grid["training_params__ablation_type"] = [ablation_type]
        else:
            param_grid["training_params__ablation_type"] = [ablation_type]
        if "grids" in param_grid_serializable:
            for grid in param_grid_serializable["grids"]:
                grid["training_params__ablation_type"] = [ablation_type]
        else:
            param_grid_serializable["training_params__ablation_type"] = [ablation_type]

    # Fetch data
    x_train, x_test, y_train, y_test = fetch_sk_data(dataset, **dataset_hps)
    # Fill default counts if None so wandb shows actual sizes
    if dataset_hps.get("num_train") in (None, [None]):
        dataset_hps["num_train"] = [len(x_train)]
    if dataset_hps.get("num_test") in (None, [None]):
        dataset_hps["num_test"] = [len(x_test)]
    # Save hyperparameter grid to Wandb
    if use_wandb:
        wandb.run.config.update(param_grid_serializable)
        wandb.config.update({"hp_search_type": search_type})
        wandb.config.update({"hp_profile": hp_profile})
        wandb.run.summary["hp_search_type"] = search_type
        wandb.run.summary["hp_profile"] = hp_profile

    # Fetch model
    sk_model = fetch_sk_model(base_model, backend)

    # Check device / override parallelism if requested
    # For safety and avoid crashes, n_jobs=1
    n_jobs = 1

    # Merlin-based photonic kernels do not play nicely with multiprocessing
    if (backend == "photonic" and n_jobs != 1) and (
        base_model == "q_rks" or base_model == "q_kernel_method"
    ):
        logging.warning(
            "Forcing n_jobs=1 for photonic q_rks to avoid worker crashes during hyperparameter search."
        )
        n_jobs = 1

    # Check model
    if force_full_grid:
        # GridSearchCV (forced for minimal profile)
        search = GridSearchCV(
            sk_model, param_grid=param_grid, cv=3, n_jobs=n_jobs, verbose=2
        )
    elif base_model in halving_grid_list:
        # HalvingGridSearchCV
        search = HalvingGridSearchCV(
            sk_model,
            param_grid=param_grid,
            cv=3,
            n_jobs=n_jobs,
            verbose=2,
            min_resources=20,
        )
    elif base_model in bayes_list:
        # BayesSearchCV
        search = BayesSearchCV(
            sk_model,
            param_grid,
            scoring="accuracy",
            cv=3,
            n_jobs=n_jobs,
            verbose=2,
            n_iter=100,
        )
    elif base_model in grid_list:
        # GridSearchCV
        search = GridSearchCV(
            sk_model, param_grid=param_grid, cv=3, n_jobs=n_jobs, verbose=2
        )
    else:
        raise NotImplementedError(f"Unknown model name for hp search: {base_model}")

    # Hyperparameter search
    logging.warning("HPs search started")
    start = time()
    search.fit(x_train, y_train)
    end = time()
    hp_search_time = end - start
    logging.warning(
        f"HPs search completed in {hp_search_time} seconds, best test accuracy reached: {search.best_score_:.4f}"
    )

    # Suggestion based on optimization time
    if hp_search_time > 3600:
        suggestion = "Optimization is too long. Reduce search space or search strategy complexity."
    elif hp_search_time < 120:
        suggestion = "Optimization is too short. You should increase search strategy complexity if possible."
    else:
        suggestion = "HP search duration is as expected."

    logging.warning(f"Suggestion: {suggestion}")
    if use_wandb:
        wandb.log({"suggestion": suggestion})

    # Get best model and final accuracies
    best_model = search.best_estimator_
    eval_start = time()
    y_pred_train = best_model.predict(x_train)
    final_train_acc = accuracy_score(y_train, y_pred_train)
    y_pred_test = best_model.predict(x_test)
    final_test_acc = accuracy_score(y_test, y_pred_test)
    eval_time = time() - eval_start
    refit_time = float(getattr(search, "refit_time_", 0.0))
    optimal_model_train_eval_time = refit_time + eval_time
    logging.warning(
        f"Final train accuracy: {final_train_acc:.4f} | Final test accuracy: {final_test_acc:.4f}"
    )
    logging.warning(
        f"Optimal model train+eval time: {optimal_model_train_eval_time:.4f} seconds"
    )

    model_dict = {"type": model_type, "name": base_model, "model": best_model}

    # Number of parameters after training (for 'sklearn_gate' models)
    (
        num_params,
        num_quantum_params,
        num_classical_params,
        num_support_vectors,
    ) = count_parameters(model_dict, True)

    # Save number of parameters / support vectors and training time
    param_grid_serializable["num_params"] = num_params
    param_grid_serializable["num_quantum_params"] = num_quantum_params
    param_grid_serializable["num_classical_params"] = num_classical_params
    param_grid_serializable["num_support_vectors"] = num_support_vectors
    param_grid_serializable["training_time"] = hp_search_time
    param_grid_serializable["hp_search_time_seconds"] = hp_search_time
    param_grid_serializable["optimal_model_train_eval_time_seconds"] = (
        optimal_model_train_eval_time
    )
    param_grid_serializable["final_train_acc"] = final_train_acc
    param_grid_serializable["final_test_acc"] = final_test_acc
    # Also in wandb
    if use_wandb:
        wandb.log(
            {
                "number_of_params": num_params,
                "number_of_quantum_params": num_quantum_params,
                "number_of_classical_params": num_classical_params,
                "num_support_vectors": num_support_vectors,
                "training_time": hp_search_time,
                "hp_search_time_seconds": hp_search_time,
                "optimal_model_train_eval_time_seconds": optimal_model_train_eval_time,
            }
        )

    # Save metrics based on model type
    if model_type == "torch":
        save_sk_train_losses_accs(
            best_model.train_losses,
            best_model.train_accuracies,
            final_test_acc,
            save_dir,
            use_wandb,
        )
    elif model_type == "reuploading":
        save_final_accs(final_train_acc, final_test_acc, save_dir, use_wandb)
    elif model_type == "sklearn":
        save_final_accs(final_train_acc, final_test_acc, save_dir, use_wandb)
    elif model_type == "sklearn_kernel":
        save_final_accs(final_train_acc, final_test_acc, save_dir, use_wandb)
    elif model_type == "jax_sklearn_gate":
        save_final_accs(final_train_acc, final_test_acc, save_dir, use_wandb)
    elif model_type == "sklearn_q_kernel":
        save_final_accs(final_train_acc, final_test_acc, save_dir, use_wandb)
    elif model_type == "sklearn_gate":
        save_final_accs(final_train_acc, final_test_acc, save_dir, use_wandb)
    elif model_type == "gate_rks":
        save_final_accs(final_train_acc, final_test_acc, save_dir, use_wandb)
    else:
        raise NotImplementedError(f"Unknown model type: {model_type}")
    logging.warning(f"Metrics saved at {save_dir}\n\n")

    # Save optimal hyperparameters
    best_params = search.best_params_
    save_search_hyperparams(param_grid_serializable, best_params, save_dir, use_wandb)
    if use_wandb:
        wandb.log({"number_of_configs": number_of_configs})
    save_hp_search_summary(
        save_dir=save_dir,
        dataset=dataset,
        big_script_name=big_script_name,
        summary={
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": dataset,
            "model": model,
            "ablation_type": ablation_type,
            "base_model": base_model,
            "backend": backend,
            "search_type": search_type,
            "hp_profile": hp_profile,
            "best_params": best_params,
            "final_train_acc": final_train_acc,
            "final_test_acc": final_test_acc,
            "num_params": num_params,
            "num_quantum_params": num_quantum_params,
            "num_classical_params": num_classical_params,
            "num_support_vectors": num_support_vectors,
            "number_of_configs": number_of_configs,
            "hp_search_time_seconds": hp_search_time,
            "optimal_model_train_eval_time_seconds": optimal_model_train_eval_time,
            "run_dir": save_dir,
        },
    )
    # Save them to Wandb too
    if use_wandb:
        wandb.run.config.update({"best_hps": best_params})
        # Save all other results too
        hp_artifact = wandb.Artifact("hyperparam_search_results", type="dataset")
        with open(save_dir + "/cv_results.json", "w") as f:
            json.dump(search.cv_results_, f, default=str)
        hp_artifact.add_file(save_dir + "/cv_results.json")
        wandb.log_artifact(hp_artifact)
        # Skip saving best model, for now (because it is problematic for some models)
        # joblib.dump(best_model, save_dir + "/best_model.pkl")
        # artifact = wandb.Artifact("best_model.pkl", type="model")
        # artifact.add_file(save_dir + "/best_model.pkl")
        # wandb.log_artifact(artifact)
    return


def set_up_logging(
    dataset: str,
    model: str,
    backend: str,
    run_type: str,
    big_script_name: Optional[str] = None,
):
    # Create folder path
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if run_type == "single":
        run_type_label = "single_run"
    elif run_type == "hyperparam_search":
        run_type_label = "hp_search"
    else:
        raise ValueError(f"Unknown run_type: {run_type}")
    model_backend_dataset = f"{model}_{backend}_{dataset}"
    model_backend = f"{model}_{backend}"
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    path_parts = [repo_root, "tabular_data", "results"]
    if big_script_name:
        path_parts.append(big_script_name)
        # If a grouped run path is provided (e.g. script_name/timestamp),
        # store model outputs directly beneath it.
        if "/" in big_script_name:
            path_parts.append(model_backend)
        else:
            run_folder = f"{run_type_label}_{timestamp}"
            path_parts.extend([run_folder, model_backend_dataset])
    else:
        run_folder = f"{run_type_label}_{timestamp}"
        path_parts.extend([run_folder, model_backend_dataset])
    log_dir = os.path.join(*path_parts)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "logs.txt")

    # --- Clear existing handlers ---
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Suppress JAX debug messages
    jax_logger = logging.getLogger("jax")
    jax_logger.setLevel(logging.WARNING)
    jax_logger.propagate = False

    # --- File handler (INFO and above) ---
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )

    # --- Console handler (WARNING and above) ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    # --- Attach handlers ---
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Set overall level to lowest you want to capture (DEBUG in file)
    root_logger.setLevel(logging.DEBUG)

    logging.warning(f"Logging set up. Logs will be saved to {log_file}\n")
    return log_dir


def set_up_random_state(random_state: int):
    if random_state is None:
        return

    # Python
    random.seed(random_state)

    # NumPy
    np.random.seed(random_state)

    # PyTorch
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)

    # Ensures reproducibility in cudnn backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return random_state
