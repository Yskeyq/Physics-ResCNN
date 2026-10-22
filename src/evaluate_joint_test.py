#!/usr/bin/env python3
r"""One-shot unified evaluation on the protected ``joint_test`` split.

This script evaluates every frozen inverse checkpoint with exactly the same
data order, normalization rules, mask threshold and metric implementation used
during inverse validation.  It never trains or updates a checkpoint.

Recommended workflow in PowerShell::

    # Does not open joint_test: checks scripts/checkpoints only.
    python .\evaluate_joint_test.py --dry-run

    # After every checkpoint is frozen, generate Pc for the protected role once.
    python .\generate_pc_dataset.py `
      --config .\pc_generation_joint_test_config.json `
      --unlock-joint-test I_UNDERSTAND_JOINT_TEST_IS_FINAL_ONLY

    # Opens joint_test and performs the final evaluation.
    python .\evaluate_joint_test.py --confirm-final-test

If a result directory has a different name, override only that checkpoint::

    python .\evaluate_joint_test.py --dry-run `
      --checkpoint physics_rescnn=artifacts/my_physics_run\best_model.pt

The program expects this file and all ``train_inverse_*.py`` files to be in the
same program directory.  By default it evaluates the proposed physics-guided
ResCNN, its supervised and fixed-radar ablations, CNN, BiLSTM, Transformer and
the adapted RFCFormer.  Use ``--skip-model MODEL_ID`` only for an explicitly
documented experiment exclusion.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from release_paths import DATA_ROOT, MODELS_ROOT, RECORDS_ROOT, OUTPUT_ROOT, SOURCE_ROOT
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd


SCRIPT_VERSION = "joint-test-evaluator-1.0.1"
PROGRAM_DIR = Path(__file__).resolve().parent
PC_DIR = (DATA_ROOT / 'pc_dataset_v1')
RUNS_ROOT = (MODELS_ROOT)
OUTPUT_DIR = (OUTPUT_ROOT / 'joint_test_final')
EXPECTED_JOINT_TEST_SAMPLES = 3571

LOGGER = logging.getLogger("joint_test_evaluator")


def _prime_numpy_openmp_runtime_on_windows() -> None:
    """Initialize NumPy's BLAS/OpenMP runtime before PyTorch on Windows.

    Some Windows Conda environments contain Intel OpenMP through both NumPy/MKL
    and the PyTorch wheel.  Merely importing NumPy does not necessarily load its
    runtime; the duplicate-runtime failure can therefore occur later, after
    PyTorch has already initialized its own copy.  A tiny NumPy BLAS operation
    establishes the process runtime first without enabling the unsafe
    KMP_DUPLICATE_LIB_OK workaround.
    """
    if os.name == "nt":
        probe = np.ones((2, 2), dtype=np.float32)
        result = np.dot(probe, probe)
        if not np.isfinite(result).all():
            raise RuntimeError("NumPy BLAS/OpenMP initialization probe failed.")


_prime_numpy_openmp_runtime_on_windows()


@dataclass(frozen=True)
class ModelDefinition:
    model_id: str
    display_name: str
    script_name: str
    candidate_run_directories: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedModel:
    model_id: str
    display_name: str
    script_path: Path
    checkpoint_path: Path


MODEL_DEFINITIONS = (
    ModelDefinition(
        "physics_rescnn",
        "Proposed Physics-ResCNN",
        "train_inverse_baseline_v2.py",
        ("inverse_physics_w0p0",),
    ),
    ModelDefinition(
        "supervised_rescnn",
        "Supervised ResCNN",
        "train_inverse_baseline_v2.py",
        ("inverse_baseline_v2", "inverse_supervised_rescnn"),
    ),
    ModelDefinition(
        "fixed_radar_rescnn",
        "Fixed-radar ResCNN",
        "train_inverse_baseline_v2.py",
        (
            "inverse_baseline_v2_fixed_radar",
            "inverse_baseline_fixed_radar",
            "inverse_fixed_radar_control",
        ),
    ),
    ModelDefinition(
        "cnn",
        "CNN",
        "train_inverse_cnn.py",
        ("inverse_cnn",),
    ),
    ModelDefinition(
        "bilstm",
        "BiLSTM",
        "train_inverse_bilstm.py",
        ("inverse_bilstm",),
    ),
    ModelDefinition(
        "transformer",
        "Transformer",
        "train_inverse_transformer.py",
        ("inverse_transformer",),
    ),
    ModelDefinition(
        "rfcformer_adapted",
        "RFCFormer-adapted-v2",
        "train_inverse_rfcformer_adapted.py",
        ("inverse_rfcformer_adapted_v2_low_dropout",),
    ),
)


def _load_python_module(path: Path, prefix: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    module_name = f"{prefix}_{digest}"
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import Python module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def _load_core():
    return _load_python_module(
        PROGRAM_DIR / "train_inverse_baseline_v2.py", "joint_test_core"
    )


def _torch_load(path: Path, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar_text(value: Any) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected one scalar value, got shape {array.shape}.")
    item = array.reshape(()).item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _normalization_array(
    normalization: Mapping[str, Any], key: str, expected_points: int
) -> np.ndarray:
    if key not in normalization:
        raise ValueError(f"Checkpoint normalization is missing {key!r}.")
    array = np.asarray(normalization[key], dtype=np.float32)
    if array.shape != (expected_points,):
        raise ValueError(
            f"Checkpoint normalization {key!r} must have shape "
            f"({expected_points},), got {array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"Checkpoint normalization {key!r} contains non-finite values.")
    return array


def _parse_checkpoint_overrides(items: Iterable[str]) -> dict[str, Path]:
    known = {definition.model_id for definition in MODEL_DEFINITIONS}
    result: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --checkpoint value {item!r}; expected MODEL_ID=PATH."
            )
        model_id, raw_path = item.split("=", 1)
        model_id = model_id.strip()
        raw_path = raw_path.strip().strip('"')
        if model_id not in known:
            raise ValueError(
                f"Unknown model id {model_id!r}. Choices: {sorted(known)}"
            )
        if model_id in result:
            raise ValueError(f"Duplicate checkpoint override for {model_id!r}.")
        result[model_id] = Path(raw_path)
    return result


def _summary_matches_model(summary: Mapping[str, Any], model_id: str) -> bool:
    script_version = str(summary.get("script_version", "")).lower()
    input_mode = str(summary.get("input_mode", "era5")).lower()
    train_config = summary.get("train_config", {})
    model_config = summary.get("model_config", {})
    physics_weight = float(train_config.get("physics_loss_weight", 0.0))
    architecture = str(model_config.get("architecture", "")).lower()
    if model_id == "physics_rescnn":
        return not architecture and input_mode == "era5" and physics_weight > 0.0
    if model_id == "supervised_rescnn":
        return not architecture and input_mode == "era5" and physics_weight == 0.0
    if model_id == "fixed_radar_rescnn":
        return not architecture and input_mode == "fixed_radar"
    if model_id == "cnn":
        return "comparison-cnn" in script_version or architecture == "plain_1d_cnn"
    if model_id == "bilstm":
        return "comparison-bilstm" in script_version or architecture == "bidirectional_lstm"
    if model_id == "transformer":
        return (
            "comparison-transformer" in script_version
            or architecture == "transformer_encoder"
        )
    if model_id == "rfcformer_adapted":
        return "rfcformer" in script_version or "rfcformer" in architecture
    return False


def _discover_checkpoint(runs_root: Path, model_id: str) -> Optional[Path]:
    if not runs_root.is_dir():
        return None
    candidates: list[Path] = []
    for directory in sorted(runs_root.iterdir()):
        if not directory.is_dir() or "smoke" in directory.name.lower():
            continue
        checkpoint_path = directory / "best_model.pt"
        summary_path = directory / "training_summary.json"
        if not checkpoint_path.is_file() or not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if _summary_matches_model(summary, model_id):
            candidates.append(checkpoint_path)
    if not candidates:
        return None
    if len(candidates) > 1:
        formatted = "\n  ".join(str(path) for path in candidates)
        raise ValueError(
            f"Several full-run checkpoints match {model_id!r}:\n  {formatted}\n"
            f"Choose one explicitly with --checkpoint {model_id}=PATH."
        )
    return candidates[0]


def resolve_models(args: argparse.Namespace) -> list[ResolvedModel]:
    overrides = _parse_checkpoint_overrides(args.checkpoint)
    skipped = set(args.skip_model)
    known = {definition.model_id for definition in MODEL_DEFINITIONS}
    unknown_skips = sorted(skipped - known)
    if unknown_skips:
        raise ValueError(f"Unknown --skip-model values: {unknown_skips}")
    resolved: list[ResolvedModel] = []
    for definition in MODEL_DEFINITIONS:
        if definition.model_id in skipped:
            continue
        script_path = Path(args.script_dir) / definition.script_name
        if definition.model_id in overrides:
            checkpoint_path = overrides[definition.model_id]
        else:
            exact = [
                Path(args.runs_root) / directory / "best_model.pt"
                for directory in definition.candidate_run_directories
            ]
            checkpoint_path = next((path for path in exact if path.is_file()), None)
            if checkpoint_path is None:
                checkpoint_path = _discover_checkpoint(
                    Path(args.runs_root), definition.model_id
                )
            if checkpoint_path is None:
                checkpoint_path = exact[0]
        resolved.append(
            ResolvedModel(
                definition.model_id,
                definition.display_name,
                script_path,
                checkpoint_path,
            )
        )
    if not resolved:
        raise ValueError("No models remain after applying --skip-model.")
    return resolved


def _checkpoint_audit(resolved: ResolvedModel, device) -> dict[str, Any]:
    if not resolved.script_path.is_file():
        raise FileNotFoundError(
            f"{resolved.model_id}: model script not found: {resolved.script_path}"
        )
    if not resolved.checkpoint_path.is_file():
        raise FileNotFoundError(
            f"{resolved.model_id}: checkpoint not found: {resolved.checkpoint_path}\n"
            f"Use --checkpoint {resolved.model_id}=FULL_PATH to override it."
        )
    checkpoint = _torch_load(resolved.checkpoint_path, device)
    required = {
        "epoch",
        "model_config",
        "normalization",
        "model_state_dict",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(
            f"{resolved.model_id}: checkpoint is missing keys: {missing}"
        )
    normalization = checkpoint["normalization"]
    if not isinstance(normalization, Mapping):
        raise ValueError(f"{resolved.model_id}: invalid normalization object.")
    input_mode = _scalar_text(normalization.get("input_mode", ""))
    if input_mode not in {"era5", "fixed_radar"}:
        raise ValueError(
            f"{resolved.model_id}: unsupported checkpoint input_mode={input_mode!r}."
        )
    pc_mean = _normalization_array(normalization, "pc_mean_dB", 500)
    pc_std = _normalization_array(normalization, "pc_std_dB", 500)
    if np.any(pc_std <= 0.0):
        raise ValueError(f"{resolved.model_id}: pc_std_dB must be positive.")
    edh_scale_m = float(np.asarray(normalization.get("edh_scale_m")).reshape(()))
    if not np.isfinite(edh_scale_m) or edh_scale_m <= 0.0:
        raise ValueError(f"{resolved.model_id}: invalid edh_scale_m={edh_scale_m}.")
    if input_mode == "fixed_radar":
        _normalization_array(normalization, "radar_reference_dB", 500)
    return {
        "model_id": resolved.model_id,
        "display_name": resolved.display_name,
        "script_path": str(resolved.script_path),
        "checkpoint_path": str(resolved.checkpoint_path),
        "checkpoint_sha256": _sha256(resolved.checkpoint_path),
        "script_sha256": _sha256(resolved.script_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_script_version": str(checkpoint.get("script_version", "")),
        "input_mode": input_mode,
        "model_config": checkpoint["model_config"],
        "train_config": checkpoint.get("train_config", {}),
    }


def _load_metadata_ids(pc_directory: Path, role: str) -> np.ndarray:
    path = pc_directory / f"pc_{role}_metadata.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path, encoding="utf-8-sig", usecols=["sample_id"])
    values = table["sample_id"].to_numpy(np.int64)
    if len(np.unique(values)) != len(values):
        raise ValueError(f"{path.name}: duplicate sample_id values.")
    return values


def _audit_role_disjointness(pc_directory: Path, joint_ids: np.ndarray) -> dict[str, int]:
    if len(np.unique(joint_ids)) != len(joint_ids):
        raise ValueError("joint_test contains duplicate sample_id values.")
    result: dict[str, int] = {}
    for role in ("inverse_train", "inverse_validation"):
        other = _load_metadata_ids(pc_directory, role)
        overlap = int(np.intersect1d(joint_ids, other).size)
        result[f"joint_test_vs_{role}_overlap"] = overlap
        if overlap:
            raise ValueError(
                f"Data leakage detected: joint_test overlaps {role} by {overlap} samples."
            )
    return result


def _prepare_model_input(
    test_data: Mapping[str, Any], normalization: Mapping[str, Any]
) -> tuple[np.ndarray, str]:
    input_mode = _scalar_text(normalization.get("input_mode", ""))
    if input_mode == "era5":
        values = np.asarray(test_data["pc"], dtype=np.float32)
    elif input_mode == "fixed_radar":
        if test_data.get("loss") is None:
            raise ValueError("fixed_radar evaluation requires joint_test ls_oneway_dB.")
        radar_reference = _normalization_array(
            normalization, "radar_reference_dB", 500
        )
        raw = -2.0 * np.asarray(test_data["loss"], dtype=np.float32)
        raw = raw + radar_reference[None, :]
        values = np.asarray(
            raw - np.mean(raw, axis=1, keepdims=True), dtype=np.float32
        )
    else:
        raise ValueError(f"Unsupported input_mode={input_mode!r}.")
    if values.shape != np.asarray(test_data["pc"]).shape:
        raise ValueError(f"Prepared input has an invalid shape: {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("Prepared model input contains non-finite values.")
    return values, input_mode


def _make_test_loader(
    core,
    test_data: Mapping[str, Any],
    prepared_input: np.ndarray,
    normalization: Mapping[str, Any],
    batch_size: int,
    workers: int,
    device,
):
    import torch

    pc_mean = _normalization_array(normalization, "pc_mean_dB", 500)
    pc_std = _normalization_array(normalization, "pc_std_dB", 500)
    dataset = core.InverseDataset(
        prepared_input,
        test_data["edh"],
        test_data["mask"],
        test_data["path_code"],
        test_data["height_code"],
        test_data["sample_id"],
        pc_mean,
        pc_std,
        radar=None,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def _flatten_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, Mapping)
    }
    path = metrics.get("path_class", {})
    height = metrics.get("height_class", {})
    row["path_class_accuracy"] = path.get("accuracy", np.nan)
    row["path_class_balanced_accuracy"] = path.get("balanced_accuracy", np.nan)
    row["height_class_accuracy"] = height.get("accuracy", np.nan)
    row["height_class_balanced_accuracy"] = height.get(
        "balanced_accuracy", np.nan
    )
    return row


def _per_sample_rows(
    test_data: Mapping[str, Any],
    predictions: Mapping[str, np.ndarray],
    display_names: Mapping[str, str],
) -> pd.DataFrame:
    metadata = test_data["metadata"].copy()
    keep = [
        "sample_id",
        "pe_row_index_0based",
        "path_class_name",
        "height_relation_class_name",
    ]
    table = metadata[keep].copy()
    truth = np.asarray(test_data["edh"], dtype=np.float64)
    for model_id, prediction in predictions.items():
        error = np.asarray(prediction, dtype=np.float64) - truth
        prefix = model_id
        table[f"{prefix}_sample_rmse_m"] = np.sqrt(np.mean(error * error, axis=1))
        table[f"{prefix}_sample_mae_m"] = np.mean(np.abs(error), axis=1)
        table[f"{prefix}_sample_bias_m"] = np.mean(error, axis=1)
        table[f"{prefix}_maximum_absolute_error_m"] = np.max(
            np.abs(error), axis=1
        )
    table.attrs["display_names"] = dict(display_names)
    return table


def _configure_matplotlib():
    cache = Path(tempfile.gettempdir()) / "matplotlib-joint-test"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_overall(summary: pd.DataFrame, output_path: Path) -> None:
    plt = _configure_matplotlib()
    ordered = summary.sort_values("rmse_m", ascending=True).reset_index(drop=True)
    colors = ["#D55E00" if value == "physics_rescnn" else "#0072B2" for value in ordered["model_id"]]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    y = np.arange(len(ordered))
    axes[0].barh(y, ordered["rmse_m"], color=colors)
    axes[0].set(xlabel="RMSE (m)", title="(a) Overall EDH RMSE")
    axes[1].barh(y, ordered["mae_m"], color=colors)
    axes[1].set(xlabel="MAE (m)", title="(b) Overall EDH MAE")
    axes[2].barh(y, ordered["p95_sample_rmse_m"], color=colors)
    axes[2].set(xlabel="P95 sample RMSE (m)", title="(c) Tail error")
    for axis in axes:
        axis.set_yticks(y, ordered["display_name"])
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_group_heatmap(groups: pd.DataFrame, output_path: Path) -> None:
    plt = _configure_matplotlib()
    path_groups = groups.loc[groups["dimension"] == "path_class_name"].copy()
    pivot = path_groups.pivot(index="display_name", columns="category", values="rmse_m")
    desired = [definition.display_name for definition in MODEL_DEFINITIONS]
    pivot = pivot.reindex([name for name in desired if name in pivot.index])
    figure, axis = plt.subplots(
        figsize=(max(8.5, 2.5 * len(pivot.columns)), 1.0 + 0.65 * len(pivot)),
        constrained_layout=True,
    )
    image = axis.imshow(pivot.to_numpy(), cmap="YlOrRd", aspect="auto")
    axis.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=18, ha="right")
    axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
    axis.set_title("Joint-test RMSE by duct path class (m)")
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            value = pivot.iloc[row, column]
            text = "--" if not np.isfinite(value) else f"{value:.3f}"
            axis.text(column, row, text, ha="center", va="center", fontsize=9)
    figure.colorbar(image, ax=axis, label="RMSE (m)")
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _representative_positions(
    test_data: Mapping[str, Any], primary_prediction: np.ndarray
) -> list[int]:
    truth = np.asarray(test_data["edh"], dtype=np.float64)
    error = np.asarray(primary_prediction, dtype=np.float64) - truth
    sample_rmse = np.sqrt(np.mean(error * error, axis=1))
    labels = test_data["metadata"]["path_class_name"].astype(str).to_numpy()
    positions: list[int] = []
    for category in sorted(np.unique(labels)):
        candidates = np.flatnonzero(labels == category)
        median = np.median(sample_rmse[candidates])
        position = candidates[np.argmin(np.abs(sample_rmse[candidates] - median))]
        positions.append(int(position))
    return positions


def _plot_examples(
    test_data: Mapping[str, Any],
    predictions: Mapping[str, np.ndarray],
    display_names: Mapping[str, str],
    primary_model: str,
    output_path: Path,
) -> None:
    plt = _configure_matplotlib()
    selected = _representative_positions(test_data, predictions[primary_model])
    figure, axes = plt.subplots(
        len(selected), 2, figsize=(15, 3.4 * len(selected)), constrained_layout=True
    )
    if len(selected) == 1:
        axes = np.asarray([axes])
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(predictions))))
    for row, position in enumerate(selected):
        metadata = test_data["metadata"].iloc[position]
        axes[row, 0].plot(
            test_data["range_km"], test_data["pc"][position], color="#0072B2"
        )
        axes[row, 0].set(xlabel="Range (km)", ylabel="Centered Pc (dB)")
        axes[row, 0].grid(alpha=0.2)
        axes[row, 1].plot(
            test_data["input_range_km"],
            test_data["edh"][position],
            color="black",
            linewidth=2.3,
            label="Truth",
        )
        for color, (model_id, prediction) in zip(colors, predictions.items()):
            width = 2.1 if model_id == primary_model else 1.15
            alpha = 1.0 if model_id == primary_model else 0.78
            axes[row, 1].plot(
                test_data["input_range_km"],
                prediction[position],
                color=color,
                linewidth=width,
                alpha=alpha,
                label=display_names[model_id],
            )
        axes[row, 1].set(
            xlabel="Range (km)",
            ylabel="EDH (m)",
            title=(
                f"sample_id={int(metadata['sample_id'])} | "
                f"{metadata['path_class_name']} | "
                f"{metadata['height_relation_class_name']}"
            ),
        )
        axes[row, 1].grid(alpha=0.2)
    axes[0, 1].legend(frameon=False, ncol=2, fontsize=8)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path}\n"
            "The protected final test should normally be evaluated once. "
            "Choose another --output-dir or explicitly pass --overwrite after "
            "confirming that replacement is intended."
        )
    path.mkdir(parents=True, exist_ok=True)


def run_dry_audit(args: argparse.Namespace, models: list[ResolvedModel]) -> None:
    import torch

    device = torch.device("cpu")
    rows = []
    for model in models:
        audit = _checkpoint_audit(model, device)
        module = _load_python_module(
            model.script_path, f"joint_test_dry_model_{model.model_id}"
        )
        checkpoint = _torch_load(model.checkpoint_path, device)
        configuration = module.ModelConfig(**dict(checkpoint["model_config"]))
        normalization = checkpoint["normalization"]
        edh_scale_m = float(np.asarray(normalization["edh_scale_m"]).reshape(()))
        network = module.build_inverse_model(configuration, edh_scale_m).to(device)
        network.load_state_dict(checkpoint["model_state_dict"], strict=True)
        audit["parameter_count"] = int(
            sum(parameter.numel() for parameter in network.parameters())
        )
        audit["strict_state_dict_load_passed"] = True
        rows.append(audit)
        LOGGER.info(
            "AUDIT OK | %-24s | epoch=%d | input=%s | parameters=%s | %s",
            model.model_id,
            audit["checkpoint_epoch"],
            audit["input_mode"],
            f"{audit['parameter_count']:,}",
            model.checkpoint_path,
        )
        del network, checkpoint, module
    print(
        json.dumps(
            {
                "script_version": SCRIPT_VERSION,
                "joint_test_opened": False,
                "model_count": len(rows),
                "models": rows,
                "recommended_next_action": (
                    "Run again with --confirm-final-test only after all checkpoints "
                    "and exclusions are frozen."
                ),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def run_final_evaluation(args: argparse.Namespace, models: list[ResolvedModel]) -> None:
    import torch

    started = time.perf_counter()
    output_directory = Path(args.output_dir)
    _prepare_output_directory(output_directory, args.overwrite)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    LOGGER.info("Device: %s", device)
    core = _load_core()

    audits = [_checkpoint_audit(model, torch.device("cpu")) for model in models]
    # joint_test HDF5 is opened exactly once here and then retained in memory.
    test_data = core.load_role_arrays(
        Path(args.pc_dir), "joint_test", need_radar=True
    )
    sample_count = len(test_data["pc"])
    if args.expected_samples > 0 and sample_count != args.expected_samples:
        raise ValueError(
            f"joint_test sample count mismatch: expected {args.expected_samples}, "
            f"got {sample_count}."
        )
    disjointness = _audit_role_disjointness(
        Path(args.pc_dir), np.asarray(test_data["sample_id"], dtype=np.int64)
    )
    LOGGER.info("joint_test loaded: %d samples", sample_count)

    metric_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    model_metrics: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    prediction_masks: dict[str, np.ndarray] = {}
    display_names = {model.model_id: model.display_name for model in models}

    for number, (resolved, audit) in enumerate(zip(models, audits), start=1):
        LOGGER.info(
            "Evaluating %d/%d: %s", number, len(models), resolved.display_name
        )
        model_module = _load_python_module(
            resolved.script_path, f"joint_test_model_{resolved.model_id}"
        )
        checkpoint = _torch_load(resolved.checkpoint_path, device)
        normalization = checkpoint["normalization"]
        prepared_input, input_mode = _prepare_model_input(test_data, normalization)
        loader = _make_test_loader(
            core,
            test_data,
            prepared_input,
            normalization,
            args.batch_size,
            args.workers,
            device,
        )
        configuration = model_module.ModelConfig(**dict(checkpoint["model_config"]))
        edh_scale_m = float(np.asarray(normalization["edh_scale_m"]).reshape(()))
        model = model_module.build_inverse_model(configuration, edh_scale_m).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
        metrics, rows, arrays = core.evaluate(
            model, loader, device, test_data["metadata"]
        )
        prediction = np.asarray(arrays["prediction_edh"], dtype=np.float32)
        prediction_mask = np.asarray(arrays["prediction_mask"], dtype=bool)
        if prediction.shape != test_data["edh"].shape:
            raise ValueError(
                f"{resolved.model_id}: prediction shape {prediction.shape} does not "
                f"match truth {test_data['edh'].shape}."
            )
        if not np.isfinite(prediction).all():
            raise ValueError(f"{resolved.model_id}: non-finite prediction detected.")
        predictions[resolved.model_id] = prediction
        prediction_masks[resolved.model_id] = prediction_mask
        model_metrics[resolved.model_id] = metrics
        metric_row = {
            "model_id": resolved.model_id,
            "display_name": resolved.display_name,
            "input_mode": input_mode,
            "parameter_count": parameter_count,
            "checkpoint_epoch": int(checkpoint["epoch"]),
            **_flatten_metrics(metrics),
        }
        metric_rows.append(metric_row)
        for row in rows:
            group_rows.append(
                {
                    "model_id": resolved.model_id,
                    "display_name": resolved.display_name,
                    "input_mode": input_mode,
                    **row,
                }
            )
        LOGGER.info(
            "%s | RMSE=%.4f m | MAE=%.4f m | R2=%.6f | mask F1=%.5f",
            resolved.display_name,
            metrics["rmse_m"],
            metrics["mae_m"],
            metrics["r2"],
            metrics["duct_mask_f1"],
        )
        del model, loader, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_table = pd.DataFrame(metric_rows)
    summary_table["rmse_rank"] = summary_table["rmse_m"].rank(
        method="min", ascending=True
    ).astype(int)
    summary_table.to_csv(
        output_directory / "joint_test_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    group_table = pd.DataFrame(group_rows)
    group_table.to_csv(
        output_directory / "joint_test_by_group.csv",
        index=False,
        encoding="utf-8-sig",
    )
    per_sample = _per_sample_rows(test_data, predictions, display_names)
    per_sample.to_csv(
        output_directory / "joint_test_per_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model_ids = [model.model_id for model in models]
    np.savez_compressed(
        output_directory / "joint_test_predictions.npz",
        model_ids=np.asarray(model_ids, dtype="U64"),
        display_names=np.asarray([display_names[key] for key in model_ids], dtype="U96"),
        sample_id=np.asarray(test_data["sample_id"], dtype=np.int64),
        input_range_km=np.asarray(test_data["input_range_km"], dtype=np.float32),
        pc_range_km=np.asarray(test_data["range_km"], dtype=np.float32),
        truth_edh_m=np.asarray(test_data["edh"], dtype=np.float32),
        truth_mask=np.asarray(test_data["mask"] > 0.5, dtype=bool),
        prediction_edh_m=np.stack([predictions[key] for key in model_ids]),
        prediction_mask=np.stack([prediction_masks[key] for key in model_ids]),
    )

    primary_model = args.primary_model
    if primary_model not in predictions:
        raise ValueError(
            f"--primary-model {primary_model!r} was not evaluated. "
            "Do not skip the proposed model, or choose another primary model."
        )
    _plot_overall(
        summary_table, output_directory / "joint_test_overall_comparison.png"
    )
    _plot_group_heatmap(
        group_table, output_directory / "joint_test_group_rmse_heatmap.png"
    )
    _plot_examples(
        test_data,
        predictions,
        display_names,
        primary_model,
        output_directory / "joint_test_prediction_examples.png",
    )

    best_row = summary_table.sort_values("rmse_m", ascending=True).iloc[0]
    summary = {
        "script_version": SCRIPT_VERSION,
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pc_directory": str(args.pc_dir),
        "joint_test_loaded": True,
        "joint_test_sample_count": sample_count,
        "joint_test_point_count": int(np.asarray(test_data["edh"]).size),
        "role_disjointness": disjointness,
        "input_shape": list(np.asarray(test_data["pc"]).shape),
        "target_shape": list(np.asarray(test_data["edh"]).shape),
        "models": audits,
        "metrics": model_metrics,
        "best_overall_rmse_model": str(best_row["model_id"]),
        "best_overall_rmse_m": float(best_row["rmse_m"]),
        "primary_model": primary_model,
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": [
            "joint_test_summary.csv",
            "joint_test_by_group.csv",
            "joint_test_per_sample.csv",
            "joint_test_predictions.npz",
            "joint_test_overall_comparison.png",
            "joint_test_group_rmse_heatmap.png",
            "joint_test_prediction_examples.png",
        ],
        "recommended_next_action": (
            "Freeze these joint-test outputs. Do not tune model parameters or "
            "checkpoint selection using joint-test results."
        ),
    }
    _write_json(output_directory / "joint_test_summary.json", summary)
    completion = {
        "status": "complete",
        "script_version": SCRIPT_VERSION,
        "joint_test_sample_count": sample_count,
        "evaluated_model_ids": model_ids,
        "summary_sha256": _sha256(output_directory / "joint_test_summary.json"),
    }
    _write_json(output_directory / "FINAL_TEST_COMPLETED.json", completion)
    LOGGER.info("Final joint-test evaluation complete: %s", output_directory)
    print(json.dumps(_json_value(summary), ensure_ascii=False, indent=2))


def run_self_test() -> None:
    overrides = _parse_checkpoint_overrides(
        ["cnn=models/cnn/best_model.pt", "bilstm=models/bilstm/best_model.pt"]
    )
    assert set(overrides) == {"cnn", "bilstm"}
    values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    centered = values - np.mean(values, axis=1, keepdims=True)
    assert np.allclose(np.mean(centered, axis=1), 0.0)
    assert re.fullmatch(r"[0-9a-f]{64}", hashlib.sha256(b"test").hexdigest())
    print("Joint-test evaluator self-test passed; joint_test was not opened.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pc-dir", type=Path, default=PC_DIR)
    parser.add_argument("--script-dir", type=Path, default=PROGRAM_DIR)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="MODEL_ID=PATH",
        help="Override one frozen best_model.pt path; may be repeated.",
    )
    parser.add_argument(
        "--skip-model",
        action="append",
        default=[],
        choices=[definition.model_id for definition in MODEL_DEFINITIONS],
        help="Explicitly exclude one model; may be repeated and must be documented.",
    )
    parser.add_argument(
        "--primary-model",
        choices=[definition.model_id for definition in MODEL_DEFINITIONS],
        default="physics_rescnn",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--expected-samples", type=int, default=EXPECTED_JOINT_TEST_SAMPLES
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("--batch-size must be positive and --workers nonnegative.")
    if args.expected_samples < 0:
        raise ValueError("--expected-samples must be nonnegative.")
    models = resolve_models(args)
    if args.dry_run:
        run_dry_audit(args, models)
        return
    if not args.confirm_final_test:
        raise RuntimeError(
            "Refusing to open the protected joint_test without "
            "--confirm-final-test. Run --dry-run first."
        )
    run_final_evaluation(args, models)


if __name__ == "__main__":
    main()
