"""Visualize minimal hyperparameter-search summaries."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "matplotlib is required for visu_hp_minimal.py. Install it with: pip install matplotlib"
    ) from exc


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class SummaryRow:
    dataset: str
    model: str
    backend: str
    model_key: str
    final_train_acc: float
    final_test_acc: float
    num_params: int
    num_support_vectors: int
    hp_search_time_seconds: float
    optimal_model_train_eval_time_seconds: float


@dataclass
class ConfigCountRow:
    dataset: str
    model: str
    backend: str
    model_key: str
    number_of_configs: int


MODEL_FAMILIES = {
    "mlp_style": {
        "title": "MLP-Style Models",
        "models": {
            "dressed_quantum_circuit",
            "dressed_quantum_circuit_reservoir",
            "multiple_paths_model",
            "multiple_paths_model_reservoir",
            "data_reuploading",
            "mlp",
        },
    },
    "kernel": {
        "title": "Kernel Models",
        "models": {
            "q_kernel_method",
            "q_kernel_method_reservoir",
            "rbf_svc",
        },
    },
    "rks": {
        "title": "RKS Models",
        "models": {
            "q_rks",
            "rks",
        },
    },
}


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    alt = ROOT_DIR / path_str
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Path not found: {path_str}")


def _discover_csvs(inputs: Iterable[str]) -> list[Path]:
    csvs: list[Path] = []
    for p in inputs:
        resolved = _resolve_path(p)
        if resolved.is_file():
            if (
                resolved.suffix.lower() == ".csv"
                and not resolved.name.endswith("_number_of_configs.csv")
            ):
                csvs.append(resolved)
            continue
        csvs.extend(
            sorted(
                c
                for c in resolved.glob("hp_search_*.csv")
                if not c.name.endswith("_number_of_configs.csv")
            )
        )
    # Preserve caller-provided --file_path order (first occurrence wins),
    # while still deduplicating paths that resolve to the same file.
    unique: list[Path] = []
    seen: set[Path] = set()
    for c in csvs:
        resolved_csv = c.resolve()
        if resolved_csv in seen:
            continue
        seen.add(resolved_csv)
        unique.append(resolved_csv)
    if not unique:
        raise FileNotFoundError(
            "No hp_search_*.csv files found in provided --file_path inputs."
        )
    return unique


def _discover_config_count_csvs(inputs: Iterable[str]) -> list[Path]:
    csvs: list[Path] = []
    for p in inputs:
        resolved = _resolve_path(p)
        if resolved.is_file():
            if (
                resolved.suffix.lower() == ".csv"
                and resolved.name.endswith("_number_of_configs.csv")
            ):
                csvs.append(resolved)
            continue
        csvs.extend(sorted(resolved.glob("hp_search_*_number_of_configs.csv")))

    unique: list[Path] = []
    seen: set[Path] = set()
    for c in csvs:
        resolved_csv = c.resolve()
        if resolved_csv in seen:
            continue
        seen.add(resolved_csv)
        unique.append(resolved_csv)
    return unique


def _to_float(row: dict[str, str], key: str) -> float:
    value = (row.get(key) or "").strip()
    return float(value) if value else 0.0


def _to_int(row: dict[str, str], key: str) -> int:
    value = (row.get(key) or "").strip()
    return int(float(value)) if value else 0


def _load_rows(csv_files: list[Path]) -> list[SummaryRow]:
    rows: list[SummaryRow] = []
    for file in csv_files:
        with file.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                dataset = (r.get("dataset") or "").strip()
                model = (r.get("model") or "").strip()
                backend = (r.get("backend") or "").strip()
                if not dataset or not model:
                    continue
                model_key = f"{model} ({backend})" if backend else model
                rows.append(
                    SummaryRow(
                        dataset=dataset,
                        model=model,
                        backend=backend,
                        model_key=model_key,
                        final_train_acc=_to_float(r, "final_train_acc"),
                        final_test_acc=_to_float(r, "final_test_acc"),
                        num_params=_to_int(r, "num_params"),
                        num_support_vectors=_to_int(r, "num_support_vectors"),
                        hp_search_time_seconds=_to_float(r, "hp_search_time_seconds"),
                        optimal_model_train_eval_time_seconds=_to_float(
                            r, "optimal_model_train_eval_time_seconds"
                        ),
                    )
                )
    if not rows:
        raise ValueError("No valid rows found in discovered CSV files.")
    return rows


def _load_config_count_rows(csv_files: list[Path]) -> list[ConfigCountRow]:
    rows: list[ConfigCountRow] = []
    for file in csv_files:
        with file.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                dataset = (r.get("dataset") or "").strip()
                model = (r.get("model") or "").strip()
                backend = (r.get("backend") or "").strip()
                if not dataset or not model:
                    continue
                model_key = f"{model} ({backend})" if backend else model
                rows.append(
                    ConfigCountRow(
                        dataset=dataset,
                        model=model,
                        backend=backend,
                        model_key=model_key,
                        number_of_configs=_to_int(r, "number_of_configs"),
                    )
                )
    return rows


def _build_index(rows: list[SummaryRow]) -> tuple[list[str], list[str], dict[tuple[str, str], SummaryRow]]:
    model_keys = list(dict.fromkeys(r.model_key for r in rows))
    datasets = list(dict.fromkeys(r.dataset for r in rows))
    lookup = {(r.model_key, r.dataset): r for r in rows}
    return model_keys, datasets, lookup


def _hatches(n: int) -> list[str]:
    base = ["", "//", "\\\\", "xx", "..", "++", "oo", "**"]
    return [base[i % len(base)] for i in range(n)]


def _backend_family(model_key: str) -> str:
    key = model_key.lower()
    if "photonic" in key:
        return "photonic"
    if "gate" in key:
        return "gate"
    return "classical"


def _adjust_color(hex_color: str, factor: float) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(hex_color)
    if factor >= 1.0:
        # Lighten towards white
        t = min(factor - 1.0, 1.0)
        return (
            r + (1.0 - r) * t,
            g + (1.0 - g) * t,
            b + (1.0 - b) * t,
        )
    # Darken towards black
    return (r * factor, g * factor, b * factor)


def _is_dark_rgb(rgb: tuple[float, float, float], threshold: float = 0.45) -> bool:
    # Relative luminance (sRGB); lower values are visually darker.
    r, g, b = rgb
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luminance < threshold


def _normalize_baseline_accuracy(score: float) -> float:
    value = float(score)
    if value > 1.0 and value <= 100.0:
        return value / 100.0
    return value


def _load_baselines() -> dict[str, dict[str, float]]:
    baseline_file = ROOT_DIR / "results" / "datasets" / "baselines.json"
    if not baseline_file.exists():
        return {}
    try:
        raw = json.loads(baseline_file.read_text())
    except json.JSONDecodeError:
        return {}
    baselines: dict[str, dict[str, float]] = {}
    for dataset, model_scores in raw.items():
        if not isinstance(model_scores, dict):
            continue
        cleaned: dict[str, float] = {}
        for model_name, score in model_scores.items():
            try:
                cleaned[str(model_name)] = float(score)
            except (TypeError, ValueError):
                continue
        if cleaned:
            baselines[str(dataset)] = cleaned
    return baselines


def _parse_dataset_identifier(dataset: str) -> tuple[str, dict[str, int]] | None:
    if dataset.startswith("hidden_manifold_"):
        parts = dataset.split("_")
        if len(parts) != 4:
            return None
        return "hidden_manifold", {"d": int(parts[2]), "m": int(parts[3])}
    if dataset.startswith("two_curves_"):
        parts = dataset.split("_")
        if len(parts) != 4:
            return None
        return "two_curves", {"d": int(parts[2]), "D": int(parts[3])}
    if dataset.startswith("downscaled_mnist_pca_"):
        parts = dataset.split("_")
        if len(parts) < 4:
            return None
        return "downscaled_mnist_pca", {"d": int(parts[3])}
    if dataset.startswith("spiral_"):
        parts = dataset.split("_")
        if len(parts) != 2:
            return None
        return "spiral", {"d": int(parts[1])}
    return None


def _infer_single_varying_dataset_variable(datasets: list[str]) -> tuple[str, list[tuple[str, int]]] | None:
    unique_datasets = list(dict.fromkeys(datasets))
    if len(unique_datasets) < 2:
        return None

    parsed = []
    for dataset in unique_datasets:
        info = _parse_dataset_identifier(dataset)
        if info is None:
            return None
        parsed.append((dataset, info[0], info[1]))

    base_names = {base for _, base, _ in parsed}
    if len(base_names) != 1:
        return None

    variable_names = list(parsed[0][2].keys())
    varying = []
    for var_name in variable_names:
        values = {params[var_name] for _, _, params in parsed}
        if len(values) > 1:
            varying.append(var_name)

    if len(varying) != 1:
        return None

    variable_name = varying[0]
    dataset_points = sorted(
        ((dataset, params[variable_name]) for dataset, _, params in parsed),
        key=lambda item: item[1],
    )
    return variable_name, dataset_points


def _family_model_keys(rows: list[SummaryRow], family_name: str) -> list[str]:
    models = MODEL_FAMILIES[family_name]["models"]
    return [
        model_key
        for model_key in dict.fromkeys(r.model_key for r in rows)
        if any(r.model_key == model_key and r.model in models for r in rows)
    ]


def _variable_axis_label(variable_name: str) -> str:
    if variable_name == "d":
        return "Data Dimensionality d"
    if variable_name in {"m", "D"}:
        return f"Data Complexity {variable_name}"
    return variable_name


def _plot_multi_path_point_accuracy_by_family(
    family_name: str,
    family_model_keys: list[str],
    dataset_points: list[tuple[str, int]],
    lookup: dict[tuple[str, str], SummaryRow],
    baselines_by_dataset: dict[str, dict[str, float]],
    variable_name: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = plt.get_cmap("tab10")
    x_values = [value for _, value in dataset_points]

    for idx, model_key in enumerate(family_model_keys):
        xs = []
        ys = []
        for dataset, x_value in dataset_points:
            row = lookup.get((model_key, dataset))
            if row is None:
                continue
            xs.append(x_value)
            ys.append(float(row.final_test_acc))
        if not xs:
            continue
        color = palette(idx % 10)
        ax.plot(xs, ys, color=color, linewidth=1.4, alpha=0.85, label=model_key)
        ax.scatter(xs, ys, color=color, edgecolors="black", linewidths=0.7, s=58, zorder=3)

    baseline_colors = ["#9467bd", "#8c564b", "#17becf"]
    baseline_names = sorted(
        {
            baseline_name
            for dataset, _ in dataset_points
            for baseline_name in baselines_by_dataset.get(dataset, {}).keys()
        }
    )
    for idx, baseline_name in enumerate(baseline_names):
        xs = []
        ys = []
        for dataset, x_value in dataset_points:
            score = baselines_by_dataset.get(dataset, {}).get(baseline_name)
            if score is None:
                continue
            xs.append(x_value)
            ys.append(_normalize_baseline_accuracy(score))
        if not xs:
            continue
        color = baseline_colors[idx % len(baseline_colors)]
        ax.plot(
            xs,
            ys,
            color=color,
            linestyle=":",
            linewidth=1.8,
            marker="s",
            markersize=5,
            label=f"{baseline_name} baseline",
        )

    ax.set_title(f"{MODEL_FAMILIES[family_name]['title']}: Test Accuracy")
    ax.set_xlabel(_variable_axis_label(variable_name))
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, 1)
    ax.set_xticks(x_values)
    ax.grid(axis="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_multi_path_point_dual_metric_by_family(
    family_name: str,
    family_model_keys: list[str],
    dataset_points: list[tuple[str, int]],
    lookup: dict[tuple[str, str], SummaryRow],
    variable_name: str,
    out_path: Path,
    metric_a_name: str,
    metric_b_name: str,
    metric_a_getter,
    metric_b_getter,
    ylabel_a: str,
    ylabel_b: str,
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    palette = plt.get_cmap("tab10")
    x_values = [value for _, value in dataset_points]

    for idx, model_key in enumerate(family_model_keys):
        xs = []
        vals_a = []
        vals_b = []
        for dataset, x_value in dataset_points:
            row = lookup.get((model_key, dataset))
            if row is None:
                continue
            xs.append(x_value)
            vals_a.append(float(metric_a_getter(row)))
            vals_b.append(float(metric_b_getter(row)))
        if not xs:
            continue
        color = palette(idx % 10)
        ax1.plot(xs, vals_a, color=color, linewidth=1.4, alpha=0.85, label=model_key)
        ax1.scatter(xs, vals_a, color=color, edgecolors="black", linewidths=0.7, s=54, zorder=3)
        ax2.plot(xs, vals_b, color=color, linewidth=1.4, alpha=0.85, label=model_key)
        ax2.scatter(xs, vals_b, color=color, edgecolors="black", linewidths=0.7, s=54, zorder=3)

    ax1.set_title(f"{MODEL_FAMILIES[family_name]['title']}: {metric_a_name}")
    ax1.set_ylabel(ylabel_a)
    ax1.grid(axis="both", alpha=0.25)
    ax1.legend(loc="best", fontsize=8)

    ax2.set_title(f"{MODEL_FAMILIES[family_name]['title']}: {metric_b_name}")
    ax2.set_xlabel(_variable_axis_label(variable_name))
    ax2.set_ylabel(ylabel_b)
    ax2.set_xticks(x_values)
    ax2.grid(axis="both", alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_multi_path_point_family_figures(
    rows: list[SummaryRow],
    lookup: dict[tuple[str, str], SummaryRow],
    baselines_by_dataset: dict[str, dict[str, float]],
    variable_name: str,
    dataset_points: list[tuple[str, int]],
    output_dir: Path,
    point_mode: bool = False,
) -> list[Path]:
    saved_paths: list[Path] = []
    for family_name in MODEL_FAMILIES:
        family_model_keys = _family_model_keys(rows, family_name)
        if not family_model_keys:
            continue

        accuracy_path = _with_optional_point_suffix(
            output_dir / f"hp_minimal_accuracy_{family_name}.png",
            point_mode,
        )
        _plot_multi_path_point_accuracy_by_family(
            family_name=family_name,
            family_model_keys=family_model_keys,
            dataset_points=dataset_points,
            lookup=lookup,
            baselines_by_dataset=baselines_by_dataset,
            variable_name=variable_name,
            out_path=accuracy_path,
        )
        saved_paths.append(accuracy_path)

        params_path = _with_optional_point_suffix(
            output_dir / f"hp_minimal_params_vectors_{family_name}.png",
            point_mode,
        )
        _plot_multi_path_point_dual_metric_by_family(
            family_name=family_name,
            family_model_keys=family_model_keys,
            dataset_points=dataset_points,
            lookup=lookup,
            variable_name=variable_name,
            out_path=params_path,
            metric_a_name="Parameters",
            metric_b_name="Support Vectors",
            metric_a_getter=lambda r: r.num_params,
            metric_b_getter=lambda r: r.num_support_vectors,
            ylabel_a="Parameters",
            ylabel_b="Support Vectors",
        )
        saved_paths.append(params_path)

        times_path = _with_optional_point_suffix(
            output_dir / f"hp_minimal_times_{family_name}.png",
            point_mode,
        )
        _plot_multi_path_point_dual_metric_by_family(
            family_name=family_name,
            family_model_keys=family_model_keys,
            dataset_points=dataset_points,
            lookup=lookup,
            variable_name=variable_name,
            out_path=times_path,
            metric_a_name="HP Search Time",
            metric_b_name="Training+Eval Time",
            metric_a_getter=lambda r: r.hp_search_time_seconds,
            metric_b_getter=lambda r: r.optimal_model_train_eval_time_seconds,
            ylabel_a="Time (s)",
            ylabel_b="Time (s)",
        )
        saved_paths.append(times_path)

    return saved_paths


def _plot_grouped(
    model_keys: list[str],
    datasets: list[str],
    lookup: dict[tuple[str, str], SummaryRow],
    metric_a_name: str,
    metric_b_name: str,
    metric_a_getter,
    metric_b_getter,
    color_a: str,
    color_b: str,
    ylabel: str,
    title: str,
    out_path: Path,
    annotate_counts: bool = False,
    annotate_metric_b_values: bool = False,
    baselines: dict[str, float] | None = None,
    legend_outside_right: bool = False,
    include_backend_legend: bool = False,
    group_fill_ratio: float = 0.8,
    max_bar_width: float = 0.20,
    point_mode: bool = False,
) -> None:
    x = np.arange(len(model_keys))
    n_sets = max(1, len(datasets))
    bars_per_model = 2 * n_sets
    width = min(group_fill_ratio / bars_per_model, max_bar_width)
    start = -group_fill_ratio / 2 + width / 2
    hatch_patterns = _hatches(n_sets)

    fig, ax = plt.subplots(figsize=(max(12, len(model_keys) * 1.3), 7))
    backend_color_factor = {
        "photonic": 0.65,
        "gate": 1.40,
        "classical": 1.00,
    }

    for ds_idx, dataset in enumerate(datasets):
        offsets = [
            start + width * (2 * ds_idx),
            start + width * (2 * ds_idx + 1),
        ]
        vals_a = []
        vals_b = []
        for model_key in model_keys:
            row = lookup.get((model_key, dataset))
            if row is None:
                vals_a.append(0.0)
                vals_b.append(0.0)
            else:
                vals_a.append(metric_a_getter(row))
                vals_b.append(metric_b_getter(row))

        colors_a = []
        colors_b = []
        for model_key in model_keys:
            backend = _backend_family(model_key)
            factor = backend_color_factor[backend]
            colors_a.append(_adjust_color(color_a, factor))
            colors_b.append(_adjust_color(color_b, factor))

        if point_mode:
            bars_a = []
            bars_b = []
            for x_pos, y_val, face_color in zip(x + offsets[0], vals_a, colors_a):
                ax.scatter(
                    x_pos,
                    y_val,
                    s=64,
                    color=face_color,
                    edgecolors="black",
                    linewidths=0.8,
                    marker="o",
                    zorder=3,
                )
            for x_pos, y_val, face_color in zip(x + offsets[1], vals_b, colors_b):
                ax.scatter(
                    x_pos,
                    y_val,
                    s=64,
                    color=face_color,
                    edgecolors="black",
                    linewidths=0.8,
                    marker="D",
                    zorder=3,
                )
        else:
            bars_a = ax.bar(
                x + offsets[0],
                vals_a,
                width=width,
                color=colors_a,
                edgecolor="black",
                linewidth=0.6,
                hatch=hatch_patterns[ds_idx],
            )
            bars_b = ax.bar(
                x + offsets[1],
                vals_b,
                width=width,
                color=colors_b,
                edgecolor="black",
                linewidth=0.6,
                hatch=hatch_patterns[ds_idx],
            )

        if annotate_metric_b_values and not point_mode:
            thin_bar = width < 0.06
            if point_mode:
                point_iter = zip(x + offsets[1], vals_b, colors_b)
            else:
                point_iter = (
                    (b.get_x() + b.get_width() / 2, b.get_height(), b.get_facecolor()[:3])
                    for b in bars_b
                )
            for x_pos, h, face_rgb in point_iter:
                if h <= 0:
                    continue
                if point_mode:
                    y = h + 0.02
                    va = "bottom"
                elif thin_bar:
                    y = h + 0.01
                    va = "bottom"
                else:
                    y = h + 0.015 if h < 0.12 else h - 0.04
                    va = "bottom" if h < 0.12 else "top"
                use_white_text = _is_dark_rgb(face_rgb)
                text_color = "white" if use_white_text else "black"
                ax.text(
                    x_pos,
                    y,
                    f"{h:.2f}",
                    ha="center",
                    va=va,
                    fontsize=8 if point_mode else 6,
                    fontweight="bold",
                    color=text_color,
                    rotation=0 if point_mode else (90 if thin_bar else 0),
                    bbox=(
                        dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.7, linewidth=0)
                        if (point_mode or thin_bar) and not use_white_text
                        else None
                    ),
                )

        if annotate_counts and not point_mode:
            ymax = max(1.0, max(vals_a + vals_b))
            if point_mode:
                point_groups = [
                    list(zip(x + offsets[0], vals_a, colors_a)),
                    list(zip(x + offsets[1], vals_b, colors_b)),
                ]
            else:
                point_groups = [
                    [
                        (b.get_x() + b.get_width() / 2, b.get_height(), b.get_facecolor()[:3])
                        for b in bars_a
                    ],
                    [
                        (b.get_x() + b.get_width() / 2, b.get_height(), b.get_facecolor()[:3])
                        for b in bars_b
                    ],
                ]
            for points in point_groups:
                for x_pos, h, face_rgb in points:
                    if h <= 0:
                        continue
                    text_inside_bar = (not point_mode) and h >= 0.08 * ymax
                    y = h - 0.04 * ymax if text_inside_bar else h + 0.01 * ymax
                    va = "top" if text_inside_bar else "bottom"
                    text_color = "white" if (text_inside_bar and _is_dark_rgb(face_rgb)) else "black"
                    ax.text(
                        x_pos,
                        y,
                        f"{int(round(h))}",
                        ha="center",
                        va=va,
                        fontsize=9 if point_mode else 8,
                        fontweight="bold",
                        color=text_color,
                    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(model_keys, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if point_mode:
        metric_legend = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=color_a, markeredgecolor="black", markersize=8, label=metric_a_name),
            Line2D([0], [0], marker="D", color="none", markerfacecolor=color_b, markeredgecolor="black", markersize=8, label=metric_b_name),
        ]
    else:
        metric_legend = [
            Patch(facecolor=color_a, edgecolor="black", label=metric_a_name),
            Patch(facecolor=color_b, edgecolor="black", label=metric_b_name),
        ]
    dataset_legend = [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=hatch_patterns[i],
            label=datasets[i],
        )
        for i in range(len(datasets))
    ]
    backend_legend = [
        Patch(facecolor="#5f5f5f", edgecolor="black", alpha=1.00, label="photonic"),
        Patch(facecolor="#ececec", edgecolor="black", alpha=1.00, label="gate"),
        Patch(facecolor="#9e9e9e", edgecolor="black", alpha=1.00, label="classical"),
    ]
    baseline_handles = []
    if baselines:
        baseline_colors = ["#9467bd", "#8c564b", "#17becf"]
        for idx, (name, score) in enumerate(baselines.items()):
            color = baseline_colors[idx % len(baseline_colors)]
            line = ax.axhline(
                y=_normalize_baseline_accuracy(score),
                color=color,
                linestyle=":",
                linewidth=1.6,
                label=f"{name} baseline",
            )
            baseline_handles.append(line)

    if legend_outside_right:
        first = ax.legend(
            handles=metric_legend + baseline_handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 0.42, 0.26, 0.44),
            borderaxespad=0.0,
            title="Metric",
            ncol=2,
            fontsize=9,
            title_fontsize=10,
        )
    else:
        first = ax.legend(handles=metric_legend + baseline_handles, loc="upper left", title="Metric", ncol=2)
    ax.add_artist(first)
    if include_backend_legend and len(datasets) > 1:
        secondary_handles = dataset_legend + backend_legend
        secondary_title = "Dataset / Backend"
    elif include_backend_legend:
        secondary_handles = backend_legend
        secondary_title = "Backend"
    elif len(datasets) > 1:
        secondary_handles = dataset_legend
        secondary_title = "Dataset"
    else:
        secondary_handles = []
        secondary_title = ""

    if secondary_handles:
        if legend_outside_right:
            ax.legend(
                handles=secondary_handles,
                loc="upper left",
                bbox_to_anchor=(1.01, 0.00, 0.26, 0.38),
                borderaxespad=0.0,
                title=secondary_title,
                ncol=2,
                fontsize=9,
                title_fontsize=10,
            )
        else:
            ax.legend(handles=secondary_handles, loc="upper right", title=secondary_title, ncol=2)

    if legend_outside_right:
        fig.tight_layout(rect=[0, 0, 0.80, 1])
    else:
        fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_single_metric_grouped(
    model_keys: list[str],
    datasets: list[str],
    lookup: dict[tuple[str, str], int],
    metric_name: str,
    color: str,
    ylabel: str,
    title: str,
    out_path: Path,
    include_backend_legend: bool = False,
    group_fill_ratio: float = 0.8,
    max_bar_width: float = 0.35,
    point_mode: bool = False,
) -> None:
    x = np.arange(len(model_keys))
    n_sets = max(1, len(datasets))
    width = min(group_fill_ratio / n_sets, max_bar_width)
    start = -group_fill_ratio / 2 + width / 2
    hatch_patterns = _hatches(n_sets)

    fig, ax = plt.subplots(figsize=(max(12, len(model_keys) * 1.3), 7))
    backend_color_factor = {
        "photonic": 0.65,
        "gate": 1.40,
        "classical": 1.00,
    }

    for ds_idx, dataset in enumerate(datasets):
        offset = start + width * ds_idx
        values = []
        colors = []
        for model_key in model_keys:
            values.append(float(lookup.get((model_key, dataset), 0)))
            backend = _backend_family(model_key)
            factor = backend_color_factor[backend]
            colors.append(_adjust_color(color, factor))

        if point_mode:
            bars = []
            for x_pos, y_val, face_color in zip(x + offset, values, colors):
                ax.scatter(
                    x_pos,
                    y_val,
                    s=72,
                    color=face_color,
                    edgecolors="black",
                    linewidths=0.8,
                    marker="o",
                    zorder=3,
                )
        else:
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                color=colors,
                edgecolor="black",
                linewidth=0.6,
                hatch=hatch_patterns[ds_idx],
            )

        ymax = max(1.0, max(values) if values else 1.0)
        if not point_mode:
            point_iter = (
                (b.get_x() + b.get_width() / 2, b.get_height(), b.get_facecolor()[:3])
                for b in bars
            )
            for x_pos, h, face_rgb in point_iter:
                if h <= 0:
                    continue
                text_inside_bar = h >= 0.08 * ymax
                y = h - 0.05 * ymax if text_inside_bar else h + 0.02 * ymax
                va = "top" if text_inside_bar else "bottom"
                text_color = "white" if (text_inside_bar and _is_dark_rgb(face_rgb)) else "black"
                ax.text(
                    x_pos,
                    y,
                    f"{int(round(h))}",
                    ha="center",
                    va=va,
                    fontsize=11,
                    fontweight="bold",
                    color=text_color,
                    bbox=(
                        dict(
                            boxstyle="round,pad=0.18",
                            facecolor=(0, 0, 0, 0.22) if text_inside_bar and text_color == "white" else (1, 1, 1, 0.75),
                            linewidth=0,
                        )
                        if not text_inside_bar or text_color != "white"
                        else None
                    ),
                )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(model_keys, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if point_mode:
        metric_legend = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="black", markersize=8, label=metric_name)
        ]
    else:
        metric_legend = [Patch(facecolor=color, edgecolor="black", label=metric_name)]
    dataset_legend = [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=hatch_patterns[i],
            label=datasets[i],
        )
        for i in range(len(datasets))
    ]
    backend_legend = [
        Patch(facecolor="#5f5f5f", edgecolor="black", alpha=1.00, label="photonic"),
        Patch(facecolor="#ececec", edgecolor="black", alpha=1.00, label="gate"),
        Patch(facecolor="#9e9e9e", edgecolor="black", alpha=1.00, label="classical"),
    ]

    first = ax.legend(handles=metric_legend, loc="upper left", title="Metric")
    ax.add_artist(first)
    if include_backend_legend and len(datasets) > 1:
        secondary_handles = dataset_legend + backend_legend
        secondary_title = "Dataset / Backend"
    elif include_backend_legend:
        secondary_handles = backend_legend
        secondary_title = "Backend"
    elif len(datasets) > 1:
        secondary_handles = dataset_legend
        secondary_title = "Dataset"
    else:
        secondary_handles = []
        secondary_title = ""

    if secondary_handles:
        ax.legend(
            handles=secondary_handles,
            loc="upper right",
            title=secondary_title,
            ncol=2,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _default_output_dir(csv_files: list[Path], num_input_paths: int) -> Path:
    if num_input_paths > 1:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        return (
            ROOT_DIR
            / "results"
            / "run_all_hyperparam_search_minimal"
            / "combined_figs"
            / timestamp
        )
    if len(csv_files) == 1:
        return csv_files[0].parent / "figures_hp_minimal"
    common = csv_files[0].parent
    for p in csv_files[1:]:
        common = Path(*common.parts[: len(Path(common).parts)])
        while str(common) != "/" and common not in p.parents and common != p.parent:
            common = common.parent
    if common.exists():
        return common / "figures_hp_minimal"
    return ROOT_DIR / "results" / "figures_hp_minimal"


def _sanitize_filename(name: str) -> str:
    cleaned = []
    for ch in name.lower():
        if ch.isalnum():
            cleaned.append(ch)
        else:
            cleaned.append("_")
    text = "".join(cleaned).strip("_")
    while "__" in text:
        text = text.replace("__", "_")
    return text or "model"


def _with_optional_point_suffix(path: Path, point_mode: bool) -> Path:
    if not point_mode:
        return path
    return path.with_name(f"{path.stem}_point{path.suffix}")


def _plot_each_model_figures(
    model_keys: list[str],
    datasets: list[str],
    lookup: dict[tuple[str, str], SummaryRow],
    baselines_by_dataset: dict[str, dict[str, float]],
    output_dir: Path,
    title_dataset_suffix: str = "",
    point_mode: bool = False,
) -> list[Path]:
    each_model_dir = output_dir / "each_model"
    each_model_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    x = np.arange(len(datasets))
    width = 0.175
    x_limits = (-0.5, len(datasets) - 0.5)

    for model_key in model_keys:
        acc_train = []
        acc_test = []
        counts_params = []
        counts_vectors = []
        time_search = []
        time_train_eval = []

        for dataset in datasets:
            row = lookup.get((model_key, dataset))
            if row is None:
                acc_train.append(0.0)
                acc_test.append(0.0)
                counts_params.append(0.0)
                counts_vectors.append(0.0)
                time_search.append(0.0)
                time_train_eval.append(0.0)
            else:
                acc_train.append(float(row.final_train_acc))
                acc_test.append(float(row.final_test_acc))
                counts_params.append(float(row.num_params))
                counts_vectors.append(float(row.num_support_vectors))
                time_search.append(float(row.hp_search_time_seconds))
                time_train_eval.append(float(row.optimal_model_train_eval_time_seconds))

        fig = plt.figure(figsize=(13, 9))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.40, wspace=0.25)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        if point_mode:
            bars_train = []
            bars_test = []
            ax1.plot(x, acc_train, color="#1f77b4", linewidth=1.2, alpha=0.55, zorder=2)
            ax1.plot(x, acc_test, color="#d62728", linewidth=1.2, alpha=0.55, zorder=2)
            ax1.scatter(x, acc_train, color="#1f77b4", edgecolors="black", linewidths=0.7, s=56, marker="o", label="Train Accuracy", zorder=3)
            ax1.scatter(x, acc_test, color="#d62728", edgecolors="black", linewidths=0.7, s=56, marker="D", label="Test Accuracy", zorder=3)
        else:
            bars_train = ax1.bar(x - width / 2, acc_train, width=width, color="#1f77b4", edgecolor="black", linewidth=0.6, label="Train Accuracy")
            bars_test = ax1.bar(x + width / 2, acc_test, width=width, color="#d62728", edgecolor="black", linewidth=0.6, label="Test Accuracy")
        baseline_colors = ["#9467bd", "#8c564b", "#17becf"]
        baseline_handles = []
        baseline_names = sorted(
            {
                baseline_name
                for dataset in datasets
                for baseline_name in baselines_by_dataset.get(dataset, {}).keys()
            }
        )
        for idx, baseline_name in enumerate(baseline_names):
            baseline_vals = [
                _normalize_baseline_accuracy(baselines_by_dataset.get(dataset, {}).get(baseline_name))
                if baselines_by_dataset.get(dataset, {}).get(baseline_name) is not None
                else np.nan
                for dataset in datasets
            ]
            if np.isnan(baseline_vals).all():
                continue
            color = baseline_colors[idx % len(baseline_colors)]
            if len(datasets) == 1:
                line = ax1.axhline(
                    y=baseline_vals[0],
                    linestyle=":",
                    linewidth=1.6,
                    color=color,
                    label=f"{baseline_name} baseline",
                    zorder=4,
                )
            else:
                (line,) = ax1.plot(
                    x,
                    baseline_vals,
                    linestyle=":",
                    linewidth=1.6,
                    color=color,
                    label=f"{baseline_name} baseline",
                    zorder=4,
                )
            baseline_handles.append(line)
        ax1.set_title("Final Train/Test Accuracy")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(0, 1)
        ax1.set_xticks(x)
        ax1.set_xticklabels(datasets, rotation=25, ha="right")
        ax1.set_xlim(*x_limits)
        ax1.grid(axis="y", alpha=0.25)
        if not point_mode:
            for bars in (bars_train, bars_test):
                for b in bars:
                    h = b.get_height()
                    if h <= 0:
                        continue
                    face_rgb = b.get_facecolor()[:3]
                    use_white_text = _is_dark_rgb(face_rgb)
                    ax1.text(
                        b.get_x() + b.get_width() / 2,
                        h + 0.015 if h < 0.12 else h - 0.04,
                        f"{h:.2f}",
                        ha="center",
                        va="bottom" if h < 0.12 else "top",
                        fontsize=10,
                        fontweight="bold",
                        color="white" if use_white_text else "black",
                        bbox=dict(
                            boxstyle="round,pad=0.10",
                            facecolor=(0, 0, 0, 0.22) if use_white_text else (1, 1, 1, 0.60),
                            linewidth=0,
                        ),
                    )
        handles, labels = ax1.get_legend_handles_labels()
        metric_handles = []
        metric_labels = []
        for h, lbl in zip(handles, labels):
            if lbl in {"Train Accuracy", "Test Accuracy"}:
                metric_handles.append(h)
                metric_labels.append(lbl)
        ordered_handles = metric_handles + baseline_handles
        ordered_labels = metric_labels + [h.get_label() for h in baseline_handles]
        ax1.legend(ordered_handles, ordered_labels, loc="upper right")

        if point_mode:
            ax2.plot(x, counts_params, color="#2ca02c", linewidth=1.2, alpha=0.55, zorder=2)
            ax2.plot(x, counts_vectors, color="#ff7f0e", linewidth=1.2, alpha=0.55, zorder=2)
            ax2.scatter(x, counts_params, color="#2ca02c", edgecolors="black", linewidths=0.7, s=56, marker="o", label="Parameters", zorder=3)
            ax2.scatter(x, counts_vectors, color="#ff7f0e", edgecolors="black", linewidths=0.7, s=56, marker="D", label="Support Vectors", zorder=3)
        else:
            ax2.bar(x - width / 2, counts_params, width=width, color="#2ca02c", edgecolor="black", linewidth=0.6, label="Parameters")
            ax2.bar(x + width / 2, counts_vectors, width=width, color="#ff7f0e", edgecolor="black", linewidth=0.6, label="Support Vectors")
        ax2.set_title("Parameters / Support Vectors")
        ax2.set_ylabel("Count")
        ax2.set_xticks(x)
        ax2.set_xticklabels(datasets, rotation=25, ha="right")
        ax2.set_xlim(*x_limits)
        ax2.grid(axis="y", alpha=0.25)
        ax2.legend(loc="upper right")

        if point_mode:
            ax3.plot(x, time_search, color="#d62728", linewidth=1.2, alpha=0.55, zorder=2)
            ax3.plot(x, time_train_eval, color="#2ca02c", linewidth=1.2, alpha=0.55, zorder=2)
            ax3.scatter(x, time_search, color="#d62728", edgecolors="black", linewidths=0.7, s=56, marker="o", label="HP Search Time", zorder=3)
            ax3.scatter(x, time_train_eval, color="#2ca02c", edgecolors="black", linewidths=0.7, s=56, marker="D", label="Training+Eval Time", zorder=3)
        else:
            ax3.bar(x - width / 2, time_search, width=width, color="#d62728", edgecolor="black", linewidth=0.6, label="HP Search Time")
            ax3.bar(x + width / 2, time_train_eval, width=width, color="#2ca02c", edgecolor="black", linewidth=0.6, label="Training+Eval Time")
        ax3.set_title("Search Time / Training+Eval Time")
        ax3.set_ylabel("Time (s)")
        ax3.set_xticks(x)
        ax3.set_xticklabels(datasets, rotation=25, ha="right")
        ax3.set_xlim(*x_limits)
        ax3.grid(axis="y", alpha=0.25)
        ax3.legend(loc="upper right")

        fig.suptitle(f"{model_key}{title_dataset_suffix}", fontsize=15)
        fig.subplots_adjust(top=0.90, bottom=0.10, left=0.07, right=0.98, hspace=0.42, wspace=0.28)

        out_path = _with_optional_point_suffix(
            each_model_dir / f"{_sanitize_filename(model_key)}.png",
            point_mode,
        )
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize minimal HP search summaries from one or more result folders."
    )
    parser.add_argument(
        "--file_path",
        nargs="+",
        required=True,
        help="One or more result folders (or CSV files). Example: results/hm_2_6_2026_02_24_13_30_19",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Optional output directory for figures. "
            "Default: inferred from input. With multiple --file_path values, "
            "defaults to results/run_all_hyperparam_search_minimal/combined_figs/{date_time}."
        ),
    )
    parser.add_argument(
        "--point",
        action="store_true",
        help="Render all generated figures with points instead of bars.",
    )
    args = parser.parse_args()

    csv_files = _discover_csvs(args.file_path)
    config_count_csv_files = _discover_config_count_csvs(args.file_path)
    rows = _load_rows(csv_files)
    model_keys, datasets, lookup = _build_index(rows)
    config_count_lookup: dict[tuple[str, str], int] = {}
    if config_count_csv_files:
        config_count_rows = _load_config_count_rows(config_count_csv_files)
        config_count_lookup = {
            (row.model_key, row.dataset): row.number_of_configs
            for row in config_count_rows
        }
    baselines_by_dataset = _load_baselines()
    include_baselines = len(args.file_path) == 1
    single_dataset = len(datasets) == 1
    dataset_title_suffix = f" ({datasets[0]})" if single_dataset else ""
    each_model_title_dataset_suffix = ""
    if len(args.file_path) == 1:
        each_model_title_dataset_suffix = f" on {', '.join(datasets)}"
    fig1_baselines = None
    if include_baselines and single_dataset:
        fig1_baselines = baselines_by_dataset.get(datasets[0])

    output_dir = (
        _resolve_path(args.output_dir)
        if args.output_dir is not None
        else _default_output_dir(csv_files, len(args.file_path))
    )

    multi_path_point_mode = len(args.file_path) > 1 and args.point
    if multi_path_point_mode:
        inferred = _infer_single_varying_dataset_variable(datasets)
        if inferred is None:
            print(
                "WARNING: could not infer a single varying dataset variable across the provided paths; "
                "falling back to the default model-axis point plots."
            )
        else:
            variable_name, dataset_points = inferred
            generated_paths = _plot_multi_path_point_family_figures(
                rows=rows,
                lookup=lookup,
                baselines_by_dataset=baselines_by_dataset,
                variable_name=variable_name,
                dataset_points=dataset_points,
                output_dir=output_dir,
                point_mode=args.point,
            )
            for path in generated_paths:
                print(path)
            return

    fig1 = _with_optional_point_suffix(output_dir / "hp_minimal_accuracy.png", args.point)
    _plot_grouped(
        model_keys=model_keys,
        datasets=datasets,
        lookup=lookup,
        metric_a_name="Train Accuracy",
        metric_b_name="Test Accuracy",
        metric_a_getter=lambda r: r.final_train_acc,
        metric_b_getter=lambda r: r.final_test_acc,
        color_a="#1f77b4",
        color_b="#d62728",
        ylabel="Accuracy",
        title=f"Final Train/Test Accuracy After Minimal HP Search{dataset_title_suffix}",
        out_path=fig1,
        annotate_counts=False,
        annotate_metric_b_values=True,
        baselines=fig1_baselines,
        legend_outside_right=True,
        include_backend_legend=True,
        group_fill_ratio=0.9,
        max_bar_width=0.30,
        point_mode=args.point,
    )

    fig2 = _with_optional_point_suffix(
        output_dir / "hp_minimal_params_vectors.png",
        args.point,
    )
    _plot_grouped(
        model_keys=model_keys,
        datasets=datasets,
        lookup=lookup,
        metric_a_name="Parameters",
        metric_b_name="Support Vectors",
        metric_a_getter=lambda r: float(r.num_params),
        metric_b_getter=lambda r: float(r.num_support_vectors),
        color_a="#2ca02c",
        color_b="#ff7f0e",
        ylabel="Count",
        title=f"Number of Parameters / Support Vectors{dataset_title_suffix}",
        out_path=fig2,
        annotate_counts=True,
        include_backend_legend=True,
        group_fill_ratio=0.95,
        max_bar_width=0.35,
        point_mode=args.point,
    )

    fig3 = _with_optional_point_suffix(output_dir / "hp_minimal_times.png", args.point)
    _plot_grouped(
        model_keys=model_keys,
        datasets=datasets,
        lookup=lookup,
        metric_a_name="HP Search Time",
        metric_b_name="Training+Eval Time",
        metric_a_getter=lambda r: r.hp_search_time_seconds,
        metric_b_getter=lambda r: r.optimal_model_train_eval_time_seconds,
        color_a="#d62728",
        color_b="#2ca02c",
        ylabel="Time (s)",
        title=f"HP Search Time and Best-Model Training+Eval Time{dataset_title_suffix}",
        out_path=fig3,
        annotate_counts=False,
        include_backend_legend=True,
        point_mode=args.point,
    )

    fig4 = None
    if config_count_lookup and len(args.file_path) == 1:
        fig4 = _with_optional_point_suffix(
            output_dir / "hp_minimal_number_of_configs.png",
            args.point,
        )
        _plot_single_metric_grouped(
            model_keys=model_keys,
            datasets=datasets,
            lookup=config_count_lookup,
            metric_name="Number of Configs",
            color="#9467bd",
            ylabel="Count",
            title=f"Number of Hyperparameter Configurations{dataset_title_suffix}",
            out_path=fig4,
            include_backend_legend=True,
            group_fill_ratio=0.95,
            max_bar_width=0.50,
            point_mode=args.point,
        )

    each_model_figures = _plot_each_model_figures(
        model_keys=model_keys,
        datasets=datasets,
        lookup=lookup,
        baselines_by_dataset=baselines_by_dataset if include_baselines else {},
        output_dir=output_dir,
        title_dataset_suffix=each_model_title_dataset_suffix,
        point_mode=args.point,
    )

    print(fig1)
    print(fig2)
    print(fig3)
    if fig4 is not None:
        print(fig4)
    print(output_dir / "each_model")
    print(f"generated_each_model_figures={len(each_model_figures)}")


if __name__ == "__main__":
    main()
