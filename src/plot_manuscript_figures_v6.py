#!/usr/bin/env python3
r"""Regenerate manuscript Figs. 2, 3, 4, and 6 from frozen data only.

This publication-layout script performs no inference, retraining, model
selection, or data resampling.  It reads the frozen map geometry, frozen joint-
test robustness summary, and frozen identifiability summary, then applies a
consistent 9-pt minimum typography policy at each figure's final print width.

For robustness, all seven models, including RFCFormer-adapted-v2, remain in
the full-scale panels.  Companion expanded panels transparently omit that one
outlying curve only to resolve the comparison among the remaining models.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from release_paths import DATA_ROOT, MODELS_ROOT, RECORDS_ROOT, OUTPUT_ROOT, SOURCE_ROOT
from typing import Any, Mapping

import numpy as np
import pandas as pd


SCRIPT_VERSION = "manuscript-figures-6.0.0"
PROGRAM_DIR = Path(__file__).resolve().parent
DEFAULT_TGRS_DIR = (DATA_ROOT / "geometry")
DEFAULT_ROBUSTNESS_CSV = (RECORDS_ROOT / 'inverse_robustness_joint_test' / 'robustness_summary.csv')
DEFAULT_IDENTIFIABILITY_CSV = (RECORDS_ROOT / 'inverse_identifiability_analysis' / 'sample_identifiability_metrics.csv')
DEFAULT_OUTPUT_DIR = (OUTPUT_ROOT / 'paper_figures' / 'manuscript_figures_v6')

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
MODEL_COLORS = {
    "physics_rescnn": "#0072B2",
    "fixed_radar_rescnn": "#7F7F7F",
    "cnn": "#4D4D4D",
    "transformer": "#8C8C8C",
    "supervised_rescnn": "#9A9A9A",
    "bilstm": "#595959",
    "rfcformer_adapted": "#D55E00",
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


def _as_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _as_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_as_json(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_as_json(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _configure_matplotlib():
    cache = Path(tempfile.gettempdir()) / "matplotlib-manuscript-figures-v6"
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
        "axes.titlesize": 9.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.45,
    })
    return plt


def _save(figure, output_dir: Path, name: str) -> None:
    figure.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(output_dir / f"{name}.png", dpi=600, bbox_inches="tight")


def _load_map_assets(tgrs_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    geometry_path = tgrs_dir / "path_geometry_100x101.npz"
    mask_path = tgrs_dir / "east_china_sea_ocean_mask_globe_1km.npz"
    if not geometry_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(f"Missing frozen map assets: {geometry_path} or {mask_path}")
    with np.load(geometry_path, allow_pickle=False) as geometry:
        latitude = np.asarray(geometry["path_latitude"], dtype=np.float64)
        longitude = np.asarray(geometry["path_longitude"], dtype=np.float64)
    with np.load(mask_path, allow_pickle=False) as mask:
        mask_latitude = np.asarray(mask["latitude"], dtype=np.float64)
        mask_longitude = np.asarray(mask["longitude"], dtype=np.float64)
        ocean = np.asarray(mask["ocean_mask"], dtype=bool)
    if latitude.shape != longitude.shape or latitude.shape[1] != 101:
        raise ValueError("Unexpected frozen path-geometry shape.")
    if ocean.shape != (len(mask_latitude), len(mask_longitude)):
        raise ValueError("Frozen ocean-mask dimensions do not match its axes.")
    return latitude, longitude, mask_latitude, mask_longitude, ocean


def _plot_study_area(output_dir: Path, tgrs_dir: Path) -> dict[str, Any]:
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    plt = _configure_matplotlib()
    latitude, longitude, mask_latitude, mask_longitude, ocean = _load_map_assets(tgrs_dir)
    figure, axis = plt.subplots(figsize=(3.5, 4.25), constrained_layout=True)
    axis.pcolormesh(
        mask_longitude, mask_latitude, ocean.astype(np.uint8),
        cmap=ListedColormap(["#E6D8B8", "#DCEFF5"]), vmin=0, vmax=1,
        shading="auto", rasterized=True, zorder=0,
    )
    axis.contour(mask_longitude, mask_latitude, ocean, levels=[0.5], colors="#4D4D4D", linewidths=0.65, zorder=1)
    axis.plot(longitude.T, latitude.T, color="#7F8C8D", lw=0.42, alpha=0.28, zorder=2)
    pad = 0.18
    xmin, xmax = float(longitude.min() - pad), float(longitude.max() + pad)
    ymin, ymax = float(latitude.min() - pad), float(latitude.max() + pad)
    axis.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor="#C73E1D", linewidth=1.25, zorder=5))
    axis.text(xmin + 0.10, ymax - 0.10, "Study domain", color="#C73E1D", fontsize=9, va="top", ha="left", zorder=6)
    axis.set_xlim(float(mask_longitude.min()), float(mask_longitude.max()))
    axis.set_ylim(float(mask_latitude.min()), float(mask_latitude.max()))
    axis.set_xlabel(r"Longitude ($^\circ$E)")
    axis.set_ylabel(r"Latitude ($^\circ$N)")
    axis.tick_params(labelsize=9)
    axis.grid(color="white", linewidth=0.5, alpha=0.85)
    scale_lat = float(mask_latitude.min() + 0.08 * (mask_latitude.max() - mask_latitude.min()))
    scale_lon = float(mask_longitude.min() + 0.08 * (mask_longitude.max() - mask_longitude.min()))
    delta_lon = 100.0 / (111.32 * math.cos(math.radians(scale_lat)))
    axis.plot([scale_lon, scale_lon + delta_lon], [scale_lat, scale_lat], color="black", lw=2.1, zorder=10)
    axis.plot([scale_lon, scale_lon], [scale_lat - 0.015, scale_lat + 0.015], color="black", lw=1.0, zorder=10)
    axis.plot([scale_lon + delta_lon, scale_lon + delta_lon], [scale_lat - 0.015, scale_lat + 0.015], color="black", lw=1.0, zorder=10)
    axis.text(scale_lon + delta_lon / 2.0, scale_lat + 0.045, "100 km", ha="center", va="bottom", fontsize=9, zorder=10)
    axis.annotate("N", xy=(0.93, 0.88), xytext=(0.93, 0.71), xycoords="axes fraction", ha="center", va="center", fontsize=10, fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))
    axis.set_aspect(1.0 / math.cos(math.radians(float(mask_latitude.mean()))))
    _save(figure, output_dir, "study_area_paths")
    plt.close(figure)
    return {"candidate_paths_displayed": int(latitude.shape[0]), "points_per_path": int(latitude.shape[1]), "final_width_in": 3.5, "minimum_font_pt": 9}


def _clean_rows(summary: pd.DataFrame) -> pd.DataFrame:
    clean = summary.loc[summary["condition"].astype(str) == "clean"].copy()
    found = set(clean["model_id"].astype(str))
    missing = [model for model in MODEL_ORDER if model not in found]
    if missing:
        raise ValueError(f"Robustness summary lacks clean rows: {missing}")
    return clean.set_index("model_id").loc[MODEL_ORDER].reset_index()


def _plot_model_comparison(output_dir: Path, summary: pd.DataFrame) -> dict[str, Any]:
    plt = _configure_matplotlib()
    clean = _clean_rows(summary)
    rows = [(MODEL_LABELS[row.model_id], float(row.rmse_m), row.model_id) for row in clean.itertuples(index=False)]
    rows.reverse()
    names = [row[0] for row in rows]
    values = [row[1] for row in rows]
    model_ids = [row[2] for row in rows]
    colors = ["#A7A7A7" if model_id != "physics_rescnn" else "#0072B2" for model_id in model_ids]
    figure, axis = plt.subplots(figsize=(3.5, 3.10), constrained_layout=True)
    bars = axis.barh(names, values, color=colors, edgecolor="#333333", linewidth=0.55)
    axis.set_xlabel("EDH RMSE (m)")
    # Leave a deliberate right margin for the longest value label (RFCFormer: 1.189).
    # This remains safe after the vector figure is embedded at column width in LaTeX.
    axis.set_xlim(0.0, 1.48)
    axis.set_xticks([0.0, 0.4, 0.8, 1.2])
    axis.tick_params(labelsize=9)
    axis.grid(axis="x", color="#D8D8D8", linewidth=0.55)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, values):
        axis.text(value + 0.022, bar.get_y() + bar.get_height() / 2.0, f"{value:.3f}", va="center", ha="left", fontsize=9)
    _save(figure, output_dir, "final_model_comparison")
    plt.close(figure)
    return {"model_count": len(rows), "x_axis_limit_m": 1.48, "minimum_font_pt": 9}


def _series(summary: pd.DataFrame, model_id: str, family: str) -> tuple[np.ndarray, np.ndarray]:
    clean = summary.loc[(summary["model_id"] == model_id) & (summary["condition"] == "clean")]
    family_rows = summary.loc[(summary["model_id"] == model_id) & (summary["family"] == family)].sort_values("level_db")
    if len(clean) != 1 or len(family_rows) != 3:
        raise ValueError(f"Expected one clean and three {family} conditions for {model_id}.")
    x = np.concatenate(([0.0], family_rows["level_db"].to_numpy(dtype=float)))
    y = np.concatenate((clean["rmse_m"].to_numpy(dtype=float), family_rows["rmse_m"].to_numpy(dtype=float)))
    return x, y


def _draw_robustness_panel(axis, summary: pd.DataFrame, family: str, title: str, ymax: float, include_rfcformer: bool) -> None:
    shown = [model for model in MODEL_ORDER if include_rfcformer or model != "rfcformer_adapted"]
    for model_id in shown:
        x, y = _series(summary, model_id, family)
        primary = model_id == "physics_rescnn"
        axis.plot(x, y, marker=MODEL_MARKERS[model_id], markersize=4.5, color=MODEL_COLORS[model_id], linewidth=2.05 if primary else 1.25, label=MODEL_LABELS[model_id], zorder=3 if primary else 2)
    axis.set_title(title, loc="left", pad=5, fontweight="bold")
    axis.set_xlabel("Input perturbation RMSE (dB)")
    axis.set_ylabel("EDH RMSE (m)")
    axis.set_xlim(-0.05, 2.05)
    axis.set_xticks([0.0, 0.5, 1.0, 2.0])
    axis.set_ylim(0.0, ymax)
    axis.tick_params(labelsize=9)
    axis.grid(alpha=0.22, linewidth=0.55)
    if not include_rfcformer:
        axis.text(0.98, 0.94, "RFCFormer shown in full-scale panel", transform=axis.transAxes, ha="right", va="top", fontsize=9, bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.0))


def _plot_robustness(output_dir: Path, summary: pd.DataFrame) -> dict[str, Any]:
    plt = _configure_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(7.16, 5.85))
    figure.subplots_adjust(left=0.09, right=0.985, top=0.93, bottom=0.175, wspace=0.28, hspace=0.43)
    _draw_robustness_panel(axes[0, 0], summary, "white_noise_db", "(a) White perturbation: full scale", 17.5, True)
    _draw_robustness_panel(axes[0, 1], summary, "white_noise_db", "(b) White perturbation: expanded", 2.1, False)
    _draw_robustness_panel(axes[1, 0], summary, "correlated_noise_db", "(c) Correlated perturbation: full scale", 7.0, True)
    _draw_robustness_panel(axes[1, 1], summary, "correlated_noise_db", "(d) Correlated perturbation: expanded", 2.1, False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.018), handlelength=2.1, columnspacing=1.3)
    _save(figure, output_dir, "robustness_noise_curves")
    plt.close(figure)
    return {"models_full_scale": MODEL_ORDER, "models_expanded": [model for model in MODEL_ORDER if model != "rfcformer_adapted"], "minimum_font_pt": 9, "full_and_expanded_panels": True}


def _plot_ambiguity(output_dir: Path, metrics: pd.DataFrame) -> dict[str, Any]:
    plt = _configure_matplotlib()
    required = {"path_class_name", "local_edh_ambiguity_mean_m", "sample_rmse_m"}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"Identifiability table missing: {missing}")
    figure, axis = plt.subplots(figsize=(3.5, 3.15), constrained_layout=True)
    class_style = [
        ("continuous_duct", "Continuous duct", "#0072B2", "o"),
        ("intermittent_transition_duct", "Intermittent transition", "#D55E00", "s"),
        ("no_duct_dominant", "No-duct dominant", "#009E73", "^")
    ]
    for class_name, label, color, marker in class_style:
        subset = metrics.loc[metrics["path_class_name"].astype(str) == class_name]
        axis.scatter(subset["local_edh_ambiguity_mean_m"], subset["sample_rmse_m"], s=12, color=color, marker=marker, alpha=0.62, edgecolors="none", label=label)
    axis.set_xlabel("Local EDH ambiguity (m)")
    axis.set_ylabel("Per-sample EDH RMSE (m)")
    axis.set_xlim(-0.2, 21.0)
    axis.set_ylim(-0.5, 15.7)
    axis.tick_params(labelsize=9)
    axis.grid(alpha=0.22, linewidth=0.55)
    axis.legend(loc="upper left", frameon=False, fontsize=9, handletextpad=0.4, borderaxespad=0.25)
    axis.text(0.985, 0.95, r"Pearson $r=0.845$" + "\n" + r"Spearman $\rho=0.717$", transform=axis.transAxes, ha="right", va="top", fontsize=9, bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.0))
    _save(figure, output_dir, "ambiguity_vs_error")
    plt.close(figure)
    return {"sample_count": int(len(metrics)), "sample_id_labels_removed": True, "minimum_font_pt": 9}


def run(args: argparse.Namespace) -> None:
    required = [args.robustness_csv, args.identifiability_csv, args.tgrs_dir / "path_geometry_100x101.npz", args.tgrs_dir / "east_china_sea_ocean_mask_globe_1km.npz"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required frozen plot inputs are missing:\n  " + "\n  ".join(missing))
    robustness = pd.read_csv(args.robustness_csv, encoding="utf-8-sig")
    metrics = pd.read_csv(args.identifiability_csv, encoding="utf-8-sig")
    audit = {
        "script_version": SCRIPT_VERSION,
        "inference_rerun": False,
        "model_or_checkpoint_selection": False,
        "minimum_display_font_pt": 9,
        "inputs": {"robustness_csv": args.robustness_csv, "identifiability_csv": args.identifiability_csv, "tgrs_dir": args.tgrs_dir},
    }
    if args.dry_run:
        print(json.dumps(_as_json({"dry_run": True, **audit}), ensure_ascii=False, indent=2))
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit["figures"] = {
        "fig2_study_area": _plot_study_area(args.output_dir, args.tgrs_dir),
        "fig3_model_comparison": _plot_model_comparison(args.output_dir, robustness),
        "fig4_robustness": _plot_robustness(args.output_dir, robustness),
        "fig6_identifiability": _plot_ambiguity(args.output_dir, metrics),
    }
    audit["outputs"] = [
        "study_area_paths.pdf", "study_area_paths.png",
        "final_model_comparison.pdf", "final_model_comparison.png",
        "robustness_noise_curves.pdf", "robustness_noise_curves.png",
        "ambiguity_vs_error.pdf", "ambiguity_vs_error.png",
        "manuscript_figures_v6_summary.json",
    ]
    _write_json(args.output_dir / "manuscript_figures_v6_summary.json", audit)
    print(json.dumps(_as_json(audit), ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tgrs-dir", type=Path, default=DEFAULT_TGRS_DIR)
    parser.add_argument("--robustness-csv", type=Path, default=DEFAULT_ROBUSTNESS_CSV)
    parser.add_argument("--identifiability-csv", type=Path, default=DEFAULT_IDENTIFIABILITY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
