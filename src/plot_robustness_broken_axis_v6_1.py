#!/usr/bin/env python3
r"""Draw the revised manuscript Fig. 4 with two broken-axis panels.

The script reads the frozen joint-test robustness summary only.  It does not
run inference, retrain a model, resample data, or select checkpoints.  Each
perturbation family is represented by one semantic panel: a low-error window
and an elevated-error window share the same x axis and are joined by explicit
axis-break marks.  Thus the RFCFormer results remain in the figure while the
six lower-error trajectories remain readable at the final double-column size.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from release_paths import DATA_ROOT, MODELS_ROOT, RECORDS_ROOT, OUTPUT_ROOT, SOURCE_ROOT
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "robustness-broken-axis-6.1.0"
DEFAULT_CSV = (RECORDS_ROOT / 'inverse_robustness_joint_test' / 'robustness_summary.csv')
DEFAULT_OUTPUT_DIR = (OUTPUT_ROOT / 'paper_figures' / 'robustness_v6_1')

MODEL_ORDER = [
    "physics_rescnn",
    "fixed_radar_rescnn",
    "cnn",
    "transformer",
    "supervised_rescnn",
    "bilstm",
    "rfcformer_adapted",
]
MODEL_LABELS = {
    "physics_rescnn": "Physics-ResCNN",
    "fixed_radar_rescnn": "Fixed-radar ResCNN",
    "cnn": "CNN",
    "transformer": "Transformer",
    "supervised_rescnn": "Supervised ResCNN",
    "bilstm": "BiLSTM",
    "rfcformer_adapted": "RFCFormer-adapted-v2",
}
# Restrained, print-friendly palette.  Colour is paired with marker shape so
# the curves remain distinguishable after grayscale reproduction.
MODEL_COLORS = {
    "physics_rescnn": "#3B7FB6",
    "fixed_radar_rescnn": "#7D91A7",
    "cnn": "#6B9A83",
    "transformer": "#9B87B4",
    "supervised_rescnn": "#C1A05B",
    "bilstm": "#6E6E6E",
    "rfcformer_adapted": "#C96E44",
}
MODEL_MARKERS = {
    "physics_rescnn": "o",
    "fixed_radar_rescnn": "s",
    "cnn": "^",
    "transformer": "D",
    "supervised_rescnn": "P",
    "bilstm": "X",
    "rfcformer_adapted": "v",
}


def _configure_matplotlib():
    cache = Path(tempfile.gettempdir()) / "matplotlib-robustness-broken-axis-v6-1"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return plt


def _series(summary: pd.DataFrame, model_id: str, family: str) -> tuple[np.ndarray, np.ndarray]:
    clean = summary.loc[(summary["model_id"] == model_id) & (summary["condition"] == "clean")]
    levels = summary.loc[(summary["model_id"] == model_id) & (summary["family"] == family)].sort_values("level_db")
    if len(clean) != 1 or len(levels) != 3:
        raise ValueError(f"Expected one clean and three {family} rows for {model_id}.")
    return (
        np.concatenate(([0.0], levels["level_db"].to_numpy(dtype=float))),
        np.concatenate((clean["rmse_m"].to_numpy(dtype=float), levels["rmse_m"].to_numpy(dtype=float))),
    )


def _draw_break(axis_upper, axis_lower, side: str) -> None:
    """Draw compact diagonal marks that identify the omitted y interval."""
    d = 0.013
    style = dict(color="#202020", clip_on=False, linewidth=0.8)
    if side == "left":
        axis_upper.plot((-d, +d), (-d, +d), transform=axis_upper.transAxes, **style)
        axis_lower.plot((-d, +d), (1 - d, 1 + d), transform=axis_lower.transAxes, **style)
    else:
        axis_upper.plot((1 - d, 1 + d), (-d, +d), transform=axis_upper.transAxes, **style)
        axis_lower.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=axis_lower.transAxes, **style)


def _style_pair(axis_upper, axis_lower, title: str, upper_limits: tuple[float, float], lower_limits: tuple[float, float]) -> None:
    for axis in (axis_upper, axis_lower):
        axis.set_xlim(-0.05, 2.05)
        axis.set_xticks([0.0, 0.5, 1.0, 2.0])
        axis.grid(color="#D7D7D7", alpha=0.62, linewidth=0.45)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=9)
    axis_upper.set_ylim(*upper_limits)
    axis_lower.set_ylim(*lower_limits)
    axis_upper.spines.bottom.set_visible(False)
    axis_lower.spines.top.set_visible(False)
    axis_upper.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    axis_lower.set_xlabel("Input perturbation RMSE (dB)")
    axis_upper.set_title(title, loc="left", fontweight="bold", pad=5)
    _draw_break(axis_upper, axis_lower, "left")
    _draw_break(axis_upper, axis_lower, "right")


def _plot_pair(axis_upper, axis_lower, summary: pd.DataFrame, family: str) -> tuple[list[Any], list[str]]:
    handles: list[Any] = []
    labels: list[str] = []
    for model_id in MODEL_ORDER:
        x, y = _series(summary, model_id, family)
        primary = model_id == "physics_rescnn"
        line_args = dict(
            color=MODEL_COLORS[model_id],
            marker=MODEL_MARKERS[model_id],
            markersize=4.5,
            markeredgewidth=0.35,
            linewidth=1.8 if primary else 1.25,
            zorder=4 if primary else 3,
            label=MODEL_LABELS[model_id],
        )
        # Plot to both coordinate windows; clipping leaves each valid portion in
        # place while the break marks make the omitted vertical interval explicit.
        upper_line, = axis_upper.plot(x, y, **line_args)
        axis_lower.plot(x, y, **line_args)
        handles.append(upper_line)
        labels.append(MODEL_LABELS[model_id])
    return handles, labels


def make_figure(summary: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    plt = _configure_matplotlib()
    figure = plt.figure(figsize=(7.16, 4.80))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.05, 1.45], hspace=0.06, wspace=0.27)
    white_upper = figure.add_subplot(grid[0, 0])
    correlated_upper = figure.add_subplot(grid[0, 1])
    white_lower = figure.add_subplot(grid[1, 0], sharex=white_upper)
    correlated_lower = figure.add_subplot(grid[1, 1], sharex=correlated_upper)
    figure.subplots_adjust(left=0.082, right=0.99, top=0.91, bottom=0.24)

    _style_pair(white_upper, white_lower, "(a) White perturbation", (2.5, 17.5), (0.45, 2.10))
    _style_pair(correlated_upper, correlated_lower, "(b) Range-correlated perturbation", (2.20, 7.0), (0.45, 2.05))
    handles, labels = _plot_pair(white_upper, white_lower, summary, "white_noise_db")
    _plot_pair(correlated_upper, correlated_lower, summary, "correlated_noise_db")
    figure.text(0.018, 0.57, "EDH RMSE (m)", rotation=90, va="center", ha="center", fontsize=9)
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=4, frameon=False, fontsize=9, handlelength=2.0, columnspacing=1.12, handletextpad=0.45)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "robustness_noise_curves_broken_axis.pdf"
    png_path = output_dir / "robustness_noise_curves_broken_axis.png"
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(png_path, dpi=360, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return {
        "figure_layout": "two perturbation panels, each with an explicit broken y axis",
        "models_in_each_panel": MODEL_ORDER,
        "white_y_windows_m": [[0.45, 2.10], [2.5, 17.5]],
        "correlated_y_windows_m": [[0.45, 2.05], [2.20, 7.0]],
        "minimum_display_font_pt": 9,
        "output_pdf": str(pdf_path),
        "output_png": str(png_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = pd.read_csv(args.summary_csv, encoding="utf-8-sig")
    required = {"model_id", "condition", "family", "level_db", "rmse_m"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Robustness summary is missing columns: {missing}")
    result = {
        "script_version": SCRIPT_VERSION,
        "inference_rerun": False,
        "model_or_checkpoint_selection": False,
        "input_summary": str(args.summary_csv),
        **make_figure(summary, args.output_dir),
    }
    (args.output_dir / "robustness_broken_axis_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
