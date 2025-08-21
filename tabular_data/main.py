"""
Main run file for tabular data benchmarking.
"""

import argparse
from run_scripts.run import run_single, run_search
import sys
import os
# Current file is tabular_data/main.py
tabular_data_folder = os.path.dirname(os.path.abspath(__file__))  # .../tabular_data
repo_root = os.path.dirname(tabular_data_folder)                 # parent of tabular_data

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)  # add repo root to path

def main(dataset, model, architecture, backend, run_type, random_state):
    if run_type == 'single':
        run_single(dataset, model, architecture, backend, random_state)
    elif run_type == 'hyperparam_search':
        run_search(dataset, model, architecture, backend, random_state)
    else:
        raise ValueError(f'Invalid run type: {run_type}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run benchmarking with different datasets and models")

    # Required arguments
    parser.add_argument("--dataset", type=str, required=True,
                        help="Name of the dataset to use")

    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model to use")

    parser.add_argument('--architecture', type=str, required=False, help="Name of the architecture to use")

    parser.add_argument("--backend", type=str, choices=["photonic", "gate"], default="photonic",
                        help="Choose the QML backend: 'photonic' for photonic-based, 'gate' for gate-based")

    parser.add_argument("--run_type", type=str, choices=["single", "hyperparam_search"], required=True,
                        help="Type of run: single run or hyperparameter search")

    parser.add_argument('--random_state', type=int, required=False, default=None, help="Random state to use")

    # Parse and handle args
    args = parser.parse_args()
    dataset = args.dataset
    model = args.model
    architecture = args.architecture
    if architecture is None:
        architecture = 'default'
    backend = args.backend
    # If classical model, change backend to classical
    if model in ['mlp', 'rbf_svc', 'rks']:
        backend = 'classical'
    run_type = args.run_type
    random_state = args.random_state
    if random_state is None:
        random_state = 42
    main(dataset, model, architecture, backend, run_type, random_state)