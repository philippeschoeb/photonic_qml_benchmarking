import csv
import json
import os
from datetime import datetime

import numpy as np
import torch


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


def _event_csv_row(event: dict) -> dict:
    return {
        "timestamp": event.get("timestamp"),
        "dataset": event.get("dataset"),
        "model": event.get("model"),
        "backend": event.get("backend"),
        "run_type": event.get("run_type"),
        "status": event.get("status"),
        "event_type": event.get("event_type"),
        "source": event.get("source"),
        "reason": event.get("reason"),
        "max_train_time_seconds": event.get("max_train_time_seconds"),
        "run_dir": event.get("run_dir"),
        "hyperparameters": json.dumps(event.get("hyperparameters", {}), default=str),
    }


def _rewrite_csv_from_jsonl(jsonl_path: str, csv_path: str) -> None:
    rows = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(_event_csv_row(json.loads(line)))

    if not rows:
        return

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_long_training_event(path: str, event: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = _serialize_for_json(dict(event))
    payload.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    with open(path, "a") as f:
        f.write(json.dumps(payload, default=str) + "\n")
    if path.endswith(".jsonl"):
        csv_path = f"{path[:-6]}.csv"
        _rewrite_csv_from_jsonl(path, csv_path)


def append_long_training_event_with_csv(jsonl_path: str, csv_path: str, event: dict) -> None:
    append_long_training_event(jsonl_path, event)
    if not jsonl_path.endswith(".jsonl") or f"{jsonl_path[:-6]}.csv" != csv_path:
        _rewrite_csv_from_jsonl(jsonl_path, csv_path)
