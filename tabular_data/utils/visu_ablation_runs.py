"""Visualize ablation runs aggregated across versions (mean/std)."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "matplotlib is required for visu_ablation_runs.py. Install it with: pip install matplotlib"
    ) from exc

from visu_hp_minimal import (
    MODEL_FAMILIES,
    ROOT_DIR,
    SummaryRow,
    _family_model_keys,
    _infer_single_varying_dataset_variable,
    _is_ablation_model_name,
    _load_baselines,
    _load_rows,
    _multi_path_point_dataset_title,
    _normalize_baseline_accuracy,
    _sanitize_filename,
    _split_ablation_variant,
    _variable_axis_label,
)


@dataclass
class SummaryRowStd:
    dataset: str
    model: str
    backend: str
    model_key: str
    final_train_acc: float
    final_test_acc: float
    num_params: float
    num_quantum_params: float
    num_classical_params: float
    num_support_vectors: float
    hp_search_time_seconds: float
    optimal_model_train_eval_time_seconds: float
    final_train_acc_std: float
    final_test_acc_std: float
    num_params_std: float
    num_quantum_params_std: float
    num_classical_params_std: float
    num_support_vectors_std: float
    hp_search_time_seconds_std: float
    optimal_model_train_eval_time_seconds_std: float
    n_runs: int


def _discover_run_dirs(
    runs_root: Path,
    dataset_prefix: str,
    settings: list[str],
    versions: list[str],
) -> list[Path]:
    run_dirs: list[Path] = []
    for setting in settings:
        for version in versions:
            pattern = f"{dataset_prefix}_{setting}_v{version}_*"
            matches = sorted(runs_root.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"No run folder found for pattern: {runs_root / pattern}")
            run_dirs.append(matches[-1])
    return run_dirs


def _discover_summary_csvs(run_dirs: list[Path]) -> list[Path]:
    csvs: list[Path] = []
    for run_dir in run_dirs:
        files = sorted(
            c
            for c in run_dir.glob("hp_search_*.csv")
            if not c.name.endswith("_number_of_configs.csv")
        )
        if len(files) != 1:
            raise ValueError(
                f"Expected exactly one summary CSV in {run_dir}, found {len(files)}."
            )
        csvs.append(files[0].resolve())
    return csvs


def _discover_config_csvs(run_dirs: list[Path]) -> list[Path]:
    csvs: list[Path] = []
    for run_dir in run_dirs:
        files = sorted(run_dir.glob("hp_search_*_number_of_configs.csv"))
        if len(files) != 1:
            continue
        csvs.append(files[0].resolve())
    return csvs


def _aggregate_rows(rows: list[SummaryRow]) -> list[SummaryRowStd]:
    grouped: dict[tuple[str, str, str], list[SummaryRow]] = {}
    for row in rows:
        grouped.setdefault((row.dataset, row.model, row.backend), []).append(row)

    out: list[SummaryRowStd] = []
    for (_, _, _), group in grouped.items():
        first = group[0]

        def stats(getter: Callable[[SummaryRow], float]) -> tuple[float, float]:
            vals = np.asarray([float(getter(r)) for r in group], dtype=float)
            return float(np.mean(vals)), float(np.std(vals, ddof=0))

        train_m, train_s = stats(lambda r: r.final_train_acc)
        test_m, test_s = stats(lambda r: r.final_test_acc)
        params_m, params_s = stats(lambda r: r.num_params)
        qparams_m, qparams_s = stats(lambda r: r.num_quantum_params)
        cparams_m, cparams_s = stats(lambda r: r.num_classical_params)
        sv_m, sv_s = stats(lambda r: r.num_support_vectors)
        hp_t_m, hp_t_s = stats(lambda r: r.hp_search_time_seconds)
        te_t_m, te_t_s = stats(lambda r: r.optimal_model_train_eval_time_seconds)

        out.append(
            SummaryRowStd(
                dataset=first.dataset,
                model=first.model,
                backend=first.backend,
                model_key=first.model_key,
                final_train_acc=train_m,
                final_test_acc=test_m,
                num_params=params_m,
                num_quantum_params=qparams_m,
                num_classical_params=cparams_m,
                num_support_vectors=sv_m,
                hp_search_time_seconds=hp_t_m,
                optimal_model_train_eval_time_seconds=te_t_m,
                final_train_acc_std=train_s,
                final_test_acc_std=test_s,
                num_params_std=params_s,
                num_quantum_params_std=qparams_s,
                num_classical_params_std=cparams_s,
                num_support_vectors_std=sv_s,
                hp_search_time_seconds_std=hp_t_s,
                optimal_model_train_eval_time_seconds_std=te_t_s,
                n_runs=len(group),
            )
        )
    return out


def _save_aggregate_csv(rows: list[SummaryRowStd], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "model",
        "backend",
        "n_runs",
        "final_train_acc_mean",
        "final_train_acc_std",
        "final_test_acc_mean",
        "final_test_acc_std",
        "num_params_mean",
        "num_params_std",
        "num_quantum_params_mean",
        "num_quantum_params_std",
        "num_classical_params_mean",
        "num_classical_params_std",
        "num_support_vectors_mean",
        "num_support_vectors_std",
        "hp_search_time_seconds_mean",
        "hp_search_time_seconds_std",
        "optimal_model_train_eval_time_seconds_mean",
        "optimal_model_train_eval_time_seconds_std",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "dataset": r.dataset,
                    "model": r.model,
                    "backend": r.backend,
                    "n_runs": r.n_runs,
                    "final_train_acc_mean": r.final_train_acc,
                    "final_train_acc_std": r.final_train_acc_std,
                    "final_test_acc_mean": r.final_test_acc,
                    "final_test_acc_std": r.final_test_acc_std,
                    "num_params_mean": r.num_params,
                    "num_params_std": r.num_params_std,
                    "num_quantum_params_mean": r.num_quantum_params,
                    "num_quantum_params_std": r.num_quantum_params_std,
                    "num_classical_params_mean": r.num_classical_params,
                    "num_classical_params_std": r.num_classical_params_std,
                    "num_support_vectors_mean": r.num_support_vectors,
                    "num_support_vectors_std": r.num_support_vectors_std,
                    "hp_search_time_seconds_mean": r.hp_search_time_seconds,
                    "hp_search_time_seconds_std": r.hp_search_time_seconds_std,
                    "optimal_model_train_eval_time_seconds_mean": r.optimal_model_train_eval_time_seconds,
                    "optimal_model_train_eval_time_seconds_std": r.optimal_model_train_eval_time_seconds_std,
                }
            )


def _plot_family_accuracy_with_std(
    family_name: str,
    family_model_keys: list[str],
    dataset_points: list[tuple[str, int]],
    lookup: dict[tuple[str, str], SummaryRowStd],
    baselines_by_dataset: dict[str, dict[str, float]],
    variable_name: str,
    out_path: Path,
    title_dataset_suffix: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = plt.get_cmap("tab10")
    x_values = [value for _, value in dataset_points]

    for idx, model_key in enumerate(family_model_keys):
        xs = []
        ys = []
        yerr = []
        for dataset, x_value in dataset_points:
            row = lookup.get((model_key, dataset))
            if row is None:
                continue
            xs.append(x_value)
            ys.append(float(row.final_test_acc))
            yerr.append(float(row.final_test_acc_std))
        if not xs:
            continue
        color = palette(idx % 10)
        ax.plot(xs, ys, color=color, linewidth=1.4, alpha=0.85, label=model_key)
        ax.scatter(xs, ys, color=color, edgecolors="black", linewidths=0.7, s=58, zorder=3)
        ax.errorbar(xs, ys, yerr=yerr, fmt="none", ecolor=color, elinewidth=1.1, capsize=3, zorder=4)

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

    ax.set_title(f"{MODEL_FAMILIES[family_name]['title']}: Test Accuracy (mean +/- std){title_dataset_suffix}")
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


def _plot_family_dual_metric_with_std(
    family_name: str,
    family_model_keys: list[str],
    dataset_points: list[tuple[str, int]],
    lookup: dict[tuple[str, str], SummaryRowStd],
    variable_name: str,
    out_path: Path,
    metric_a_name: str,
    metric_b_name: str,
    metric_a_getter,
    metric_b_getter,
    metric_a_std_getter,
    metric_b_std_getter,
    ylabel_a: str,
    ylabel_b: str,
    title_dataset_suffix: str = "",
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    palette = plt.get_cmap("tab10")
    x_values = [value for _, value in dataset_points]

    for idx, model_key in enumerate(family_model_keys):
        xs = []
        vals_a = []
        vals_b = []
        err_a = []
        err_b = []
        for dataset, x_value in dataset_points:
            row = lookup.get((model_key, dataset))
            if row is None:
                continue
            xs.append(x_value)
            vals_a.append(float(metric_a_getter(row)))
            vals_b.append(float(metric_b_getter(row)))
            err_a.append(float(metric_a_std_getter(row)))
            err_b.append(float(metric_b_std_getter(row)))
        if not xs:
            continue
        color = palette(idx % 10)
        ax1.plot(xs, vals_a, color=color, linewidth=1.4, alpha=0.85, label=model_key)
        ax1.scatter(xs, vals_a, color=color, edgecolors="black", linewidths=0.7, s=54, zorder=3)
        ax1.errorbar(xs, vals_a, yerr=err_a, fmt="none", ecolor=color, elinewidth=1.1, capsize=3, zorder=4)
        ax2.plot(xs, vals_b, color=color, linewidth=1.4, alpha=0.85, label=model_key)
        ax2.scatter(xs, vals_b, color=color, edgecolors="black", linewidths=0.7, s=54, zorder=3)
        ax2.errorbar(xs, vals_b, yerr=err_b, fmt="none", ecolor=color, elinewidth=1.1, capsize=3, zorder=4)

    ax1.set_title(f"{MODEL_FAMILIES[family_name]['title']}: {metric_a_name} (mean +/- std){title_dataset_suffix}")
    ax1.set_ylabel(ylabel_a)
    ax1.grid(axis="both", alpha=0.25)
    ax1.legend(loc="best", fontsize=8)

    ax2.set_title(f"{MODEL_FAMILIES[family_name]['title']}: {metric_b_name} (mean +/- std){title_dataset_suffix}")
    ax2.set_xlabel(_variable_axis_label(variable_name))
    ax2.set_ylabel(ylabel_b)
    ax2.set_xticks(x_values)
    ax2.grid(axis="both", alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_family_figures_with_std(
    rows: list[SummaryRowStd],
    lookup: dict[tuple[str, str], SummaryRowStd],
    baselines_by_dataset: dict[str, dict[str, float]],
    variable_name: str,
    dataset_points: list[tuple[str, int]],
    output_dir: Path,
    title_dataset_suffix: str = "",
) -> list[Path]:
    saved_paths: list[Path] = []
    for family_name in MODEL_FAMILIES:
        family_model_keys = _family_model_keys(rows, family_name)
        if not family_model_keys:
            continue

        accuracy_path = output_dir / f"hp_ablation_accuracy_{family_name}.png"
        _plot_family_accuracy_with_std(
            family_name=family_name,
            family_model_keys=family_model_keys,
            dataset_points=dataset_points,
            lookup=lookup,
            baselines_by_dataset=baselines_by_dataset,
            variable_name=variable_name,
            out_path=accuracy_path,
            title_dataset_suffix=title_dataset_suffix,
        )
        saved_paths.append(accuracy_path)

        params_path = output_dir / f"hp_ablation_params_vectors_{family_name}.png"
        _plot_family_dual_metric_with_std(
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
            metric_a_std_getter=lambda r: r.num_params_std,
            metric_b_std_getter=lambda r: r.num_support_vectors_std,
            ylabel_a="Parameters",
            ylabel_b="Support Vectors",
            title_dataset_suffix=title_dataset_suffix,
        )
        saved_paths.append(params_path)

        times_path = output_dir / f"hp_ablation_times_{family_name}.png"
        _plot_family_dual_metric_with_std(
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
            metric_a_std_getter=lambda r: r.hp_search_time_seconds_std,
            metric_b_std_getter=lambda r: r.optimal_model_train_eval_time_seconds_std,
            ylabel_a="Time (s)",
            ylabel_b="Time (s)",
            title_dataset_suffix=title_dataset_suffix,
        )
        saved_paths.append(times_path)
    return saved_paths


def _plot_each_model_figures_with_std(
    rows: list[SummaryRowStd],
    datasets: list[str],
    lookup: dict[tuple[str, str], SummaryRowStd],
    output_dir: Path,
    title_dataset_suffix: str = "",
    point_mode: bool = False,
    variable_name: str | None = None,
    dataset_points: list[tuple[str, int]] | None = None,
) -> list[Path]:
    each_model_dir = output_dir / "each_model"
    each_model_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    if point_mode and dataset_points:
        ordered_datasets = [dataset for dataset, _ in dataset_points]
        x_values = np.asarray([value for _, value in dataset_points], dtype=float)
    else:
        ordered_datasets = list(datasets)
        x_values = np.arange(len(ordered_datasets), dtype=float)
    variant_order = ["orig", "q", "c"]
    variant_labels = {"orig": "Original", "q": "Q Ablation", "c": "C Ablation"}
    variant_hatches = {"orig": "", "q": "//", "c": "xx"}

    grouped_model_keys: dict[str, dict[str, str]] = {}
    for row in rows:
        base_model, variant = _split_ablation_variant(row.model)
        base_model_key = f"{base_model} ({row.backend})" if row.backend else base_model
        grouped_model_keys.setdefault(base_model_key, {})
        grouped_model_keys[base_model_key][variant] = row.model_key

    for base_model_key, variant_to_model_key in grouped_model_keys.items():
        ordered_variants = [v for v in variant_order if v in variant_to_model_key]
        if not ordered_variants:
            continue
        has_real_ablation = any(v in variant_to_model_key for v in ("q", "c"))

        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        ax1, ax2, ax3 = axes
        width = min(0.18, 0.88 / (2 * max(1, len(ordered_variants))))
        start = -0.44 + width / 2
        variant_colors = {
            "orig": "#1f77b4",
            "q": "#ff7f0e",
            "c": "#2ca02c",
        }
        all_acc_lows: list[float] = []
        all_acc_highs: list[float] = []
        all_param_lows: list[float] = []
        all_param_highs: list[float] = []

        for idx, variant in enumerate(ordered_variants):
            model_key = variant_to_model_key[variant]
            label_suffix = variant_labels.get(variant, variant)
            hatch = variant_hatches.get(variant, "..")
            vals_train = []
            vals_train_std = []
            vals_test = []
            vals_test_std = []
            vals_params = []
            vals_params_std = []
            vals_time = []
            vals_time_std = []
            for dataset in ordered_datasets:
                row = lookup.get((model_key, dataset))
                if row is None:
                    vals_train.append(0.0)
                    vals_train_std.append(0.0)
                    vals_test.append(0.0)
                    vals_test_std.append(0.0)
                    vals_params.append(0.0)
                    vals_params_std.append(0.0)
                    vals_time.append(0.0)
                    vals_time_std.append(0.0)
                else:
                    vals_train.append(float(row.final_train_acc))
                    vals_train_std.append(float(row.final_train_acc_std))
                    vals_test.append(float(row.final_test_acc))
                    vals_test_std.append(float(row.final_test_acc_std))
                    vals_params.append(float(row.num_params))
                    vals_params_std.append(float(row.num_params_std))
                    vals_time.append(float(row.hp_search_time_seconds))
                    vals_time_std.append(float(row.hp_search_time_seconds_std))

            if has_real_ablation:
                all_acc_lows.extend([max(0.0, v - s) for v, s in zip(vals_test, vals_test_std)])
                all_acc_highs.extend([min(1.0, v + s) for v, s in zip(vals_test, vals_test_std)])
            else:
                all_acc_lows.extend(
                    [max(0.0, v - s) for v, s in zip(vals_train, vals_train_std)]
                    + [max(0.0, v - s) for v, s in zip(vals_test, vals_test_std)]
                )
                all_acc_highs.extend(
                    [min(1.0, v + s) for v, s in zip(vals_train, vals_train_std)]
                    + [min(1.0, v + s) for v, s in zip(vals_test, vals_test_std)]
                )
            all_param_lows.extend([max(0.0, v - s) for v, s in zip(vals_params, vals_params_std)])
            all_param_highs.extend([v + s for v, s in zip(vals_params, vals_params_std)])

            off_train = start + width * (2 * idx)
            off_test = start + width * (2 * idx + 1)

            if point_mode:
                color = variant_colors.get(variant, "#7f7f7f")
                # Same x-axis positions for all ablations; distinguish by color.
                if not has_real_ablation:
                    ax1.plot(x_values, vals_train, color=color, linewidth=1.2, alpha=0.70, linestyle="--", zorder=2)
                ax1.plot(x_values, vals_test, color=color, linewidth=1.4, alpha=0.90, linestyle="-", zorder=2)
                if not has_real_ablation:
                    ax1.scatter(
                        x_values,
                        vals_train,
                        color="white",
                        edgecolors=color,
                        linewidths=1.2,
                        s=50,
                        marker="o",
                        label=f"{label_suffix} Train",
                        zorder=3,
                    )
                ax1.scatter(
                    x_values,
                    vals_test,
                    color=color,
                    edgecolors="black",
                    linewidths=0.7,
                    s=52,
                    marker="D",
                    label=f"{label_suffix} Test",
                    zorder=3,
                )
                if not has_real_ablation:
                    ax1.errorbar(x_values, vals_train, yerr=vals_train_std, fmt="none", ecolor=color, elinewidth=1.0, capsize=2, zorder=4)
                ax1.errorbar(x_values, vals_test, yerr=vals_test_std, fmt="none", ecolor=color, elinewidth=1.0, capsize=2, zorder=4)

                ax2.plot(x_values, vals_params, color=color, linewidth=1.4, alpha=0.85, zorder=2)
                ax2.scatter(x_values, vals_params, color=color, edgecolors="black", linewidths=0.7, s=52, marker="o", label=f"{label_suffix}", zorder=3)
                ax2.errorbar(x_values, vals_params, yerr=vals_params_std, fmt="none", ecolor=color, elinewidth=1.0, capsize=2, zorder=4)

                ax3.plot(x_values, vals_time, color=color, linewidth=1.4, alpha=0.85, zorder=2)
                ax3.scatter(x_values, vals_time, color=color, edgecolors="black", linewidths=0.7, s=52, marker="o", label=f"{label_suffix}", zorder=3)
                ax3.errorbar(x_values, vals_time, yerr=vals_time_std, fmt="none", ecolor=color, elinewidth=1.0, capsize=2, zorder=4)
            else:
                if not has_real_ablation:
                    ax1.bar(x_values + off_train, vals_train, width=width, color="#1f77b4", edgecolor="black", linewidth=0.6, hatch=hatch, label=f"Train ({label_suffix})")
                ax1.bar(x_values + off_test, vals_test, width=width, color="#d62728", edgecolor="black", linewidth=0.6, hatch=hatch, label=f"Test ({label_suffix})")
                if not has_real_ablation:
                    ax1.errorbar(x_values + off_train, vals_train, yerr=vals_train_std, fmt="none", ecolor="black", elinewidth=1.0, capsize=2, zorder=4)
                ax1.errorbar(x_values + off_test, vals_test, yerr=vals_test_std, fmt="none", ecolor="black", elinewidth=1.0, capsize=2, zorder=4)

                ax2.bar(x_values + off_train, vals_params, width=width, color="#2ca02c", edgecolor="black", linewidth=0.6, hatch=hatch, label=f"Params ({label_suffix})")
                ax2.errorbar(x_values + off_train, vals_params, yerr=vals_params_std, fmt="none", ecolor="black", elinewidth=1.0, capsize=2, zorder=4)

                ax3.bar(x_values + off_train, vals_time, width=width, color="#d62728", edgecolor="black", linewidth=0.6, hatch=hatch, label=f"HP Search Time ({label_suffix})")
                ax3.errorbar(x_values + off_train, vals_time, yerr=vals_time_std, fmt="none", ecolor="black", elinewidth=1.0, capsize=2, zorder=4)

            # Display mean values directly on train/test bars only.
            if not point_mode:
                test_positions = x_values + off_test
                if not has_real_ablation:
                    train_positions = x_values + off_train
                    for x_pos, val in zip(train_positions, vals_train):
                        ax1.text(
                            x_pos,
                            min(0.995, val + 0.025),
                            f"{val:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            bbox=dict(boxstyle="round,pad=0.08", facecolor="white", alpha=0.70, linewidth=0),
                        )
                for x_pos, val in zip(test_positions, vals_test):
                    ax1.text(
                        x_pos,
                        min(0.995, val + 0.025),
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.08", facecolor="white", alpha=0.70, linewidth=0),
                    )

        if has_real_ablation:
            ax1.set_title("Final Test Accuracy")
            ax1.set_ylabel("Test Accuracy")
        else:
            ax1.set_title("Final Train/Test Accuracy")
            ax1.set_ylabel("Accuracy")
        if point_mode and all_acc_lows and all_acc_highs:
            lo = min(all_acc_lows)
            hi = max(all_acc_highs)
            span = max(hi - lo, 0.03)
            pad = max(0.01, 0.12 * span)
            lo = max(0.0, lo - pad)
            hi = min(1.0, hi + pad)
            if hi - lo < 0.05:
                center = 0.5 * (hi + lo)
                lo = max(0.0, center - 0.025)
                hi = min(1.0, center + 0.025)
            ax1.set_ylim(lo, hi)
        else:
            ax1.set_ylim(0, 1)
        ax1.grid(axis="y", alpha=0.25)
        ax1.legend(loc="upper right")

        ax2.set_title("Parameters")
        if point_mode and all_param_lows and all_param_highs:
            p_lo = min(all_param_lows)
            p_hi = max(all_param_highs)
            p_span = max(p_hi - p_lo, 1.0)
            p_pad = max(1.0, 0.10 * p_span)
            ax2.set_ylim(max(0.0, p_lo - p_pad), p_hi + p_pad)
            from matplotlib.ticker import MaxNLocator

            ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.set_ylabel("Count")
        ax2.grid(axis="y", alpha=0.25)
        ax2.legend(loc="upper right")

        ax3.set_title("HP Search Time")
        ax3.set_ylabel("Time (s)")
        ax3.grid(axis="y", alpha=0.25)
        ax3.legend(loc="upper right")
        ax3.set_xticks(x_values)
        if point_mode and variable_name is not None and dataset_points:
            ax3.set_xticklabels([str(v) for _, v in dataset_points], rotation=0, ha="center")
            ax3.set_xlabel(_variable_axis_label(variable_name))
        else:
            ax3.set_xticklabels(ordered_datasets, rotation=25, ha="right")

        point_dataset_suffix = ""
        if point_mode and dataset_points:
            dataset_title = _multi_path_point_dataset_title([dataset for dataset, _ in dataset_points])
            if dataset_title:
                point_dataset_suffix = f" on {dataset_title}"
        fig.suptitle(f"{base_model_key}{point_dataset_suffix}{title_dataset_suffix}", fontsize=15)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        suffix = "_point" if point_mode else ""
        out_path = each_model_dir / f"{_sanitize_filename(base_model_key)}{suffix}.png"
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def _default_output_dir(runs_root: Path, dataset_prefix: str, settings: list[str]) -> Path:
    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    joined = "_".join(settings)
    return runs_root / "combined_figs_ablation" / f"{dataset_prefix}_{joined}_{stamp}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize ablation HP-search runs aggregated across versions (mean/std)."
    )
    parser.add_argument(
        "--runs_root",
        type=str,
        default=str(ROOT_DIR / "results" / "run_all_hp_search_minimal_ablation"),
        help="Directory containing folders like hm_10_2_v1_...",
    )
    parser.add_argument(
        "--dataset_prefix",
        type=str,
        default="hm",
        help="Dataset prefix in folder names (e.g., hm or tc).",
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        default=["10_2", "10_10"],
        help="Setting tokens used in folder names (e.g., 10_2 10_10).",
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["1", "2", "3"],
        help="Run versions to aggregate (e.g., 1 2 3).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory for figures and aggregate CSV.",
    )
    parser.add_argument(
        "--point",
        action="store_true",
        help="Render generated figures with points instead of bars.",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    run_dirs = _discover_run_dirs(
        runs_root=runs_root,
        dataset_prefix=args.dataset_prefix,
        settings=args.settings,
        versions=args.versions,
    )
    csv_files = _discover_summary_csvs(run_dirs)
    rows_all = _load_rows(csv_files)

    rows_agg = _aggregate_rows(rows_all)
    rows_global_agg = [r for r in rows_agg if not _is_ablation_model_name(r.model)]
    if not rows_global_agg:
        rows_global_agg = rows_agg

    datasets = list(dict.fromkeys(r.dataset for r in rows_global_agg))
    lookup = {(r.model_key, r.dataset): r for r in rows_global_agg}
    each_model_datasets = list(dict.fromkeys(r.dataset for r in rows_agg))
    each_model_lookup = {(r.model_key, r.dataset): r for r in rows_agg}

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else _default_output_dir(runs_root, args.dataset_prefix, args.settings)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_csv = output_dir / "hp_ablation_aggregate_mean_std.csv"
    _save_aggregate_csv(rows_agg, aggregate_csv)

    inferred = _infer_single_varying_dataset_variable(datasets)
    if inferred is None:
        raise ValueError(
            "Could not infer a single varying dataset variable from aggregated ablation datasets. "
            "Expected comparable datasets such as hidden_manifold_10_2 and hidden_manifold_10_10."
        )
    variable_name, dataset_points = inferred
    dataset_title = _multi_path_point_dataset_title([dataset for dataset, _ in dataset_points])
    title_dataset_suffix = f" on {dataset_title}" if dataset_title else ""
    family_figures = _plot_family_figures_with_std(
        rows=rows_global_agg,
        lookup=lookup,
        baselines_by_dataset=_load_baselines(),
        variable_name=variable_name,
        dataset_points=dataset_points,
        output_dir=output_dir,
        title_dataset_suffix=title_dataset_suffix,
    )

    each_model_figures = _plot_each_model_figures_with_std(
        rows=rows_agg,
        datasets=each_model_datasets,
        lookup=each_model_lookup,
        output_dir=output_dir,
        title_dataset_suffix="",
        point_mode=args.point,
        variable_name=variable_name,
        dataset_points=dataset_points,
    )

    print("Selected run dirs:")
    for d in run_dirs:
        print(d)
    print(aggregate_csv)
    for fig_path in family_figures:
        print(fig_path)
    print(output_dir / "each_model")
    print(f"generated_each_model_figures={len(each_model_figures)}")


if __name__ == "__main__":
    main()
