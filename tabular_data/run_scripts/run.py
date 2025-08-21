import logging
from datetime import datetime
import os
import random
import numpy as np
import torch
from sklearn.utils import check_random_state
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_scripts.single_run import get_hyperparams as single_hps
from run_scripts.hyperparam_search_run import get_hyperparams as search_hps
from datasets.fetch_data import fetch_data
from models.fetch_model import fetch_model
from training.distribute_training import distribute_training
from results.save_metrics import save_train_losses_accs, save_train_loss_final_accs, save_final_accs, save_hyperparams

def run_single(dataset, model, architecture, backend, random_state):
    save_dir = set_up_logging(dataset, model, architecture, backend)
    random_state, rng = set_up_random_state(random_state)
    separator = '#' * 40
    logging.warning(f'SINGLE RUN {separator}\nDataset: {dataset}\nModel: {model}\nArchitecture: {architecture}\nBackend: {backend}\nRunType: single\nRandomState: {random_state}\n')
    hyperparams = single_hps(dataset, model, architecture, backend, rng)

    dataset_hps = hyperparams['dataset']
    train_loader, test_loader, x_train, x_test, y_train, y_test = fetch_data(dataset, rng, **dataset_hps)
    input_size = x_train.shape[1]

    model_hps = hyperparams['model']
    training_hps = hyperparams['training']
    model = fetch_model(model, backend, input_size, training_hps['output_size'], **model_hps)
    model_dict = {'type': model_hps['type'], 'name': model_hps['name'], 'model': model}

    # Model characteristics
    num_params = count_parameters(model_dict)

    # Training
    logging.warning('Starting training')
    results_dict = distribute_training(model_dict, train_loader, test_loader, x_train, x_test, y_train, y_test, **training_hps)
    logging.warning('Training completed')

    # Handle training results
    model_type = results_dict['type']
    model_name = results_dict['name']

    if model_type == 'torch':
        save_train_losses_accs(results_dict['train_losses'], results_dict['test_losses'], results_dict['train_accs'], results_dict['test_accs'], save_dir)
    elif model_type == 'reuploading':
        save_train_loss_final_accs(results_dict['train_losses'], results_dict['final_train_acc'], results_dict['final_test_acc'], save_dir)
    elif model_type == 'sklearn_q_kernel':
        save_train_loss_final_accs(results_dict['train_losses'], results_dict['final_train_acc'], results_dict['final_test_acc'], save_dir)
    elif model_type == 'sklearn_kernel':
        save_final_accs(results_dict['final_train_acc'], results_dict['final_test_acc'], save_dir)
    elif model_type == 'sklearn':
        save_final_accs(results_dict['final_train_acc'], results_dict['final_test_acc'], save_dir)
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')
    logging.warning(f'Metrics saved at {save_dir}\n\n')

    # Save hyperparameters too
    model_hps['num_params'] = num_params
    save_hyperparams({'dataset': dataset_hps, 'model': model_hps, 'training': training_hps}, save_dir)


def run_search(dataset, model, architecture, backend, random_state):
    save_dir = set_up_logging(dataset, model, architecture, backend)
    random_state, rng = set_up_random_state(random_state)
    separator = '#' * 40
    logging.info(f'HYPERPARAM SEARCH {separator}\nDataset: {dataset}\nModel: {model}\nArchitecture: {architecture}\nBackend: {backend}\nRunType: hyperparam_search\nRandomState: {random_state}')
    hyperparams = search_hps(dataset, model, architecture, backend)


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

    # Sklearn
    rng = check_random_state(random_state)

    return random_state, rng


def count_parameters(model_dict):
    model_type = model_dict['type']
    model = model_dict['model']
    if model_type == 'torch':
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif model_type == 'reuploading':
        num_params = sum(p.numel() for p in model.quantum_model.parameters() if p.requires_grad)
    elif model_type == 'sklearn_q_kernel':
        optimizable_model = model.quantum_kernel
        num_params = sum(p.numel() for p in optimizable_model.parameters())
    elif model_type == 'sklearn_kernel':
        #TODO
        num_params = 0
    elif model_type == 'sklearn':
        #TODO
        num_params = 0
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')
    logging.warning(f'Number of parameters: {num_params}')
    return num_params

