#!/usr/bin/env python3
"""Summarize long-training events (cut short / skipped) from JSONL reports."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
    return events


def _resolve_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob("long_training*.jsonl"))
    return []


def _stable_hp_signature(hps: Any) -> str:
    if not hps:
        return "{}"
    try:
        return json.dumps(hps, sort_keys=True, separators=(",", ":"), default=str)
    except TypeError:
        return json.dumps(str(hps))


def _print_summary(events: list[dict[str, Any]], top_k: int) -> None:
    print(f"Loaded events: {len(events)}")
    if not events:
        return

    status_counts = Counter(e.get("status", "unknown") for e in events)
    event_type_counts = Counter(e.get("event_type", "unknown") for e in events)
    print("Status counts:")
    for key, value in status_counts.most_common():
        print(f"  - {key}: {value}")
    print("Event type counts:")
    for key, value in event_type_counts.most_common():
        print(f"  - {key}: {value}")

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        dataset = str(event.get("dataset", "unknown"))
        model = str(event.get("model", event.get("model_name", "unknown")))
        backend = str(event.get("backend", "unknown"))
        grouped[(model, backend, dataset)].append(event)

    print("\nPer model/backend/dataset:")
    for (model, backend, dataset), rows in sorted(
        grouped.items(), key=lambda kv: len(kv[1]), reverse=True
    ):
        hp_counter: Counter[str] = Counter(
            _stable_hp_signature(r.get("hyperparameters")) for r in rows
        )
        print(f"- model={model}, backend={backend}, dataset={dataset}: {len(rows)} event(s)")
        for hp_sig, count in hp_counter.most_common(top_k):
            print(f"    top_hp_count={count} hp={hp_sig}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize long-training events from a JSONL file or a directory "
            "containing long_training*.jsonl files."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a long_training JSONL file or a results directory.",
    )
    parser.add_argument(
        "--top_k_hyperparams",
        type=int,
        default=3,
        help="How many top hyperparameter signatures to print per group.",
    )
    args = parser.parse_args()

    files = _resolve_inputs(args.input)
    if not files:
        print(f"No long-training JSONL files found at: {args.input}", file=sys.stderr)
        return 1

    all_events: list[dict[str, Any]] = []
    for file_path in files:
        all_events.extend(_load_events(file_path))

    _print_summary(all_events, top_k=max(1, args.top_k_hyperparams))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
