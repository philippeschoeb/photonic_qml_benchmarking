import logging
from datetime import datetime
import os
import random
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
from datasets.fetch_data import fetch_data, fetch_sk_data
from models.fetch_model import fetch_model, fetch_sk_model
from training.distribute_training import distribute_training
from results.save_metrics import (
    save_train_losses_accs,
    save_train_loss_final_accs,
    save_final_accs,
    save_hyperparams,
    save_sk_train_losses_accs,
    save_sk_final_test_acc,
    save_search_hyperparams
)


def run_single(dataset, model, architecture, backend, random_state, use_wandb):
    # Setup logging and return save directory
    save_dir = set_up_logging(dataset, model, architecture, backend)
    # Setup random state across torch, numpy and random
    random_state = set_up_random_state(random_state)

    # Log 'single run' beginning
    separator = '##' * 40
    logging.warning(f'SINGLE RUN {separator}\nDataset: {dataset}\nModel: {model}\nArchitecture: {architecture}\nBackend: {backend}\nRunType: single\nRandomState: {random_state}\n')

    # Fetch hyperparameters for single run
    hyperparams = single_hps(dataset, model, architecture, backend, random_state)
    dataset_hps = hyperparams['dataset']
    model_hps = hyperparams['model']
    training_hps = hyperparams['training']
    # Setup hyperparameters in wandb
    if use_wandb:
        wandb.run.config.update(hyperparams)

    # Fetch data
    train_loader, test_loader, x_train, x_test, y_train, y_test = fetch_data(dataset, random_state, **dataset_hps)
    # Get input size
    input_size = x_train.shape[1]
    # Get model type
    model_type = model_hps['type']

    # Fetch model
    model = fetch_model(model, backend, input_size, training_hps['output_size'], **model_hps)
    # Define model dictionary
    model_dict = {'type': model_hps['type'], 'name': model_hps['name'], 'model': model}

    # Training
    logging.warning('Starting training')
    start = time()
    results_dict = distribute_training(model_dict, train_loader, test_loader, x_train, x_test, y_train, y_test, **training_hps)
    end = time()
    training_time = end - start
    logging.warning(f'Training completed in {training_time} seconds')

    # Number of parameters after training (for 'sklearn_gate' models)
    num_params, num_support_vectors = count_parameters(model_dict)

    # Save metrics based on model type
    if model_type == 'torch':
        save_train_losses_accs(results_dict['train_losses'], results_dict['test_losses'], results_dict['train_accs'], results_dict['test_accs'], save_dir, use_wandb)
    elif model_type == 'reuploading':
        save_train_loss_final_accs(results_dict['train_losses'], results_dict['final_train_acc'], results_dict['final_test_acc'], save_dir, use_wandb)
    elif model_type == 'sklearn_q_kernel':
        save_train_loss_final_accs(results_dict['train_losses'], results_dict['final_train_acc'], results_dict['final_test_acc'], save_dir, use_wandb)
    elif model_type == 'sklearn_kernel':
        save_final_accs(results_dict['final_train_acc'], results_dict['final_test_acc'], save_dir, use_wandb)
    elif model_type == 'sklearn':
        save_final_accs(results_dict['final_train_acc'], results_dict['final_test_acc'], save_dir, use_wandb)
    elif model_type == 'jax_sklearn_gate':
        loss_history = results_dict['model'].loss_history_
        save_train_loss_final_accs(loss_history, results_dict['final_train_acc'], results_dict['final_test_acc'], save_dir, use_wandb)
    elif model_type == 'sklearn_gate' or model_type == 'gate_rks':
        save_final_accs(results_dict['final_train_acc'], results_dict['final_test_acc'], save_dir, use_wandb)
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')
    logging.warning(f'Metrics saved at {save_dir}\n\n')

    # Save hyperparameters, number of parameters / support vectors and training time
    model_hps['num_params'] = num_params
    model_hps['num_support_vectors'] = num_support_vectors
    training_hps['training_time'] = training_time
    save_hyperparams({'dataset': dataset_hps, 'model': model_hps, 'training': training_hps}, save_dir)
    # Also in wandb
    if use_wandb:
        wandb.log({'number_of_params': num_params, 'num_support_vectors': num_support_vectors, 'training_time': training_time})
    return


def run_search(dataset, model, architecture, backend, random_state, use_wandb):
    # Setup logging and return save directory
    save_dir = set_up_logging(dataset, model, architecture, backend)
    # Setup random state across torch, numpy and random
    random_state = set_up_random_state(random_state)

    # Log 'hyperparameter search' beginning
    separator = '##' * 40
    logging.warning(f'HYPERPARAM SEARCH {separator}\nDataset: {dataset}\nModel: {model}\nArchitecture: {architecture}\nBackend: {backend}\nRunType: hyperparam_search\nRandomState: {random_state}')

    # Load model_search_assignment
    with open("hyperparameters/hyperparam_search/model_search_assignment.json", "r") as f:
        model_search_assignment = json.load(f)
    halving_grid_list = model_search_assignment['halving_grid']
    bayes_list = model_search_assignment['bayes']
    grid_list = model_search_assignment['grid']

    # Fetch hyperparameter grid for hyperparameter search
    if model in halving_grid_list:
        param_grid, dataset_hps, model_hps, training_hps = get_hyperparams_halving_grid(dataset, model, architecture, backend, random_state)
        search_type = 'halving_grid_search'
    elif model in bayes_list:
        param_grid, dataset_hps, model_hps, training_hps = get_hyperparams_bayes(dataset, model, architecture, backend, random_state)
        search_type = 'bayes_search'
    elif model in grid_list:
        param_grid, dataset_hps, model_hps, training_hps = get_hyperparams_halving_grid(dataset, model, architecture, backend, random_state)
        search_type = 'grid_search'
    else:
        raise NotImplementedError(f'Unknown model name for hp search: {model}')
    logging.warning(f'Hyperparameters search type: {search_type}')

    # Save hyperparameter grid to Wandb
    if use_wandb:
        wandb.run.config.update(param_grid)
        wandb.config.update({"hp_search_type": search_type})
        wandb.run.summary['hp_search_type'] = search_type

    # Get device
    device = training_hps['device'][0]
    # Get model type
    model_type = model_hps['type'][0]

    # Fetch data
    x_train, x_test, y_train, y_test = fetch_sk_data(dataset, **dataset_hps)

    # Fetch model
    sk_model = fetch_sk_model(model, backend)

    # Check device / override parallelism if requested
    if 'n_jobs' in training_hps:
        n_jobs = training_hps['n_jobs'][0]
    elif device == torch.device('cpu'):
        n_jobs = -1
    else:
        n_jobs = 1

    # Merlin-based photonic kernels do not play nicely with multiprocessing
    if (backend == 'photonic' and n_jobs != 1) and (model == 'q_rks' or model == 'q_kernel_method'):
        logging.warning('Forcing n_jobs=1 for photonic q_rks to avoid worker crashes during hyperparameter search.')
        n_jobs = 1

    # Check model
    if model in halving_grid_list:
        # HalvingGridSearchCV
        search = HalvingGridSearchCV(sk_model, param_grid=param_grid, cv=3, n_jobs=n_jobs, verbose=2, min_resources=20)
    elif model in bayes_list:
        # BayesSearchCV
        search = BayesSearchCV(sk_model, param_grid, scoring='accuracy', cv=3, n_jobs=n_jobs, verbose=2, n_iter=100)
    elif model in grid_list:
        # GridSearchCV
        search = GridSearchCV(sk_model, param_grid=param_grid, cv=3, n_jobs=n_jobs, verbose=2)
    else:
        raise NotImplementedError(f'Unknown model name for hp search: {model}')

    # Hyperparameter search
    logging.warning(f'HPs search started')
    search.fit(x_train, y_train)
    logging.warning(f'HPs search completed, best test accuracy reached: {search.best_score_:.4f}')
    # Get best model and final accuracies
    best_model = search.best_estimator_
    y_pred_train = best_model.predict(x_train)
    final_train_acc = accuracy_score(y_train, y_pred_train)
    y_pred_test = best_model.predict(x_test)
    final_test_acc = accuracy_score(y_test, y_pred_test)
    logging.warning(f'Final train accuracy: {final_train_acc:.4f} | Final test accuracy: {final_test_acc:.4f}')

    # Save metrics based on model type
    if model_type == 'torch':
        save_sk_train_losses_accs(best_model.train_losses, best_model.train_accuracies, final_test_acc, save_dir, use_wandb)
    elif model_type == 'sklearn':
        save_final_accs(final_train_acc, final_test_acc, save_dir, use_wandb)
    elif model_type == 'sklearn_kernel':
        save_final_accs(final_train_acc, final_test_acc, save_dir, use_wandb)
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')
    logging.warning(f'Metrics saved at {save_dir}\n\n')

    # Save optimal hyperparameters
    best_params = search.best_params_
    save_search_hyperparams(param_grid, best_params, save_dir)
    # Save them to Wandb too
    if use_wandb:
        wandb.run.config.update({"best_hps": best_params})
        # Save all other results too
        hp_artifact = wandb.Artifact('hyperparam_search_results', type='dataset')
        with open('cv_results.json', 'w') as f:
            json.dump(search.cv_results_, f, default=str)
        hp_artifact.add_file('cv_results.json')
        wandb.log_artifact(hp_artifact)
        # Save best model
        joblib.dump(best_model, 'best_model.pkl')
        artifact = wandb.Artifact('best_model.pkl', type='model')
        artifact.add_file('best_model.pkl')
        wandb.log_artifact(artifact)
    return


def set_up_logging(dataset: str, model: str, architecture: str, backend: str):
    # Create folder path
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = os.path.join(repo_root, "tabular_data", "results", dataset, model + '_' + backend, architecture, timestamp)
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
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

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


def count_parameters(model_dict):
    model_type = model_dict['type']
    model = model_dict['model']
    num_params = 0
    num_support_vectors = 0

    if model_type == 'torch':
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif model_type == 'reuploading':
        num_params = sum(p.numel() for p in model.quantum_model.parameters() if p.requires_grad)
    elif model_type == 'sklearn_q_kernel':
        optimizable_model = model.quantum_kernel
        num_params = sum(p.numel() for p in optimizable_model.parameters())
        num_support_vectors = len(model.model.support_)
    elif model_type == 'sklearn_kernel':
        num_support_vectors = len(model.model.support_)
    elif model_type == 'sklearn':
        num_support_vectors = len(model.model.support_)
    elif model_type == 'jax_sklearn_gate':
        num_params = sum(p.size if not isinstance(p, tuple) else sum(e.size for e in p) for p in model.params_.values())
    elif model_type == 'gate_rks':
        num_params = sum(p.size if not isinstance(p, tuple) else sum(e.size for e in p) for p in model.params_.values())
    elif model_type == 'sklearn_gate':
        num_support_vectors = len(model.svm.support_)
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')
    logging.warning(f'Number of parameters: {num_params}')
    logging.warning(f'Number of support vectors: {num_support_vectors}')
    return num_params, num_support_vectors
