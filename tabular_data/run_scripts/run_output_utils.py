import csv
import json
import os
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import ParameterGrid
from utils.long_training_events import append_long_training_event_with_csv


def _serialize_for_json(obj):
    if isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.device):
        return str(obj)
    return obj


def _count_search_configurations(param_grid):
    if isinstance(param_grid, dict):
        return len(ParameterGrid(param_grid))
    if isinstance(param_grid, list):
        return sum(len(ParameterGrid(grid)) for grid in param_grid)
    raise TypeError(f"Unsupported param_grid type: {type(param_grid)!r}")


def _summary_csv_row(summary: dict) -> dict:
    return {
        "timestamp": summary.get("timestamp"),
        "dataset": summary.get("dataset"),
        "model": summary.get("model"),
        "backend": summary.get("backend"),
        "search_type": summary.get("search_type"),
        "hp_profile": summary.get("hp_profile"),
        "number_of_configs": summary.get("number_of_configs"),
        "final_train_acc": summary.get("final_train_acc"),
        "final_test_acc": summary.get("final_test_acc"),
        "num_params": summary.get("num_params"),
        "num_quantum_params": summary.get("num_quantum_params"),
        "num_classical_params": summary.get("num_classical_params"),
        "num_support_vectors": summary.get("num_support_vectors"),
        "hp_search_time_seconds": summary.get("hp_search_time_seconds"),
        "optimal_model_train_eval_time_seconds": summary.get(
            "optimal_model_train_eval_time_seconds"
        ),
        "best_params": json.dumps(summary.get("best_params", {}), default=str),
        "run_dir": summary.get("run_dir"),
    }


def _rewrite_summary_csv_from_jsonl(jsonl_path: str, csv_path: str) -> None:
    rows = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(_summary_csv_row(json.loads(line)))

    if not rows:
        return

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_hp_search_summary(
    save_dir: str,
    summary: dict,
    dataset: str,
    big_script_name: Optional[str] = None,
):
    """Persist per-run summary and append to dataset-level aggregate files."""
    os.makedirs(save_dir, exist_ok=True)
    summary = _serialize_for_json(summary)

    per_run_path = os.path.join(save_dir, "hp_search_summary.json")
    with open(per_run_path, "w") as f:
        json.dump(summary, f, indent=4, default=str)

    if not big_script_name:
        return

    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    summary_dir = os.path.join(repo_root, "tabular_data", "results", big_script_name)
    os.makedirs(summary_dir, exist_ok=True)

    dataset_slug = dataset.replace("/", "_")
    jsonl_path = os.path.join(summary_dir, f"hp_search_{dataset_slug}.jsonl")
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(summary, default=str) + "\n")

    csv_path = os.path.join(summary_dir, f"hp_search_{dataset_slug}.csv")
    _rewrite_summary_csv_from_jsonl(jsonl_path, csv_path)

    configs_csv_path = os.path.join(
        summary_dir, f"hp_search_{dataset_slug}_number_of_configs.csv"
    )
    configs_csv_row = {
        "timestamp": summary.get("timestamp"),
        "dataset": summary.get("dataset"),
        "model": summary.get("model"),
        "backend": summary.get("backend"),
        "number_of_configs": summary.get("number_of_configs"),
        "run_dir": summary.get("run_dir"),
    }
    write_configs_header = not os.path.exists(configs_csv_path)
    with open(configs_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(configs_csv_row.keys()))
        if write_configs_header:
            writer.writeheader()
        writer.writerow(configs_csv_row)


def save_long_training_event(
    save_dir: str,
    event: dict,
    dataset: str,
    big_script_name: Optional[str] = None,
):
    """Persist timeout/skip events locally and optionally into grouped aggregates."""
    os.makedirs(save_dir, exist_ok=True)

    per_run_jsonl = os.path.join(save_dir, "long_training_events.jsonl")
    per_run_csv = os.path.join(save_dir, "long_training_events.csv")
    append_long_training_event_with_csv(per_run_jsonl, per_run_csv, event)

    if not big_script_name:
        return

    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    summary_dir = os.path.join(repo_root, "tabular_data", "results", big_script_name)
    os.makedirs(summary_dir, exist_ok=True)

    dataset_slug = dataset.replace("/", "_")
    agg_jsonl = os.path.join(summary_dir, f"long_training_{dataset_slug}.jsonl")
    agg_csv = os.path.join(summary_dir, f"long_training_{dataset_slug}.csv")
    append_long_training_event_with_csv(agg_jsonl, agg_csv, event)
