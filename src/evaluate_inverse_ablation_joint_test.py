#!/usr/bin/env python3
r"""Confirm the frozen inverse-loss ablations on ``joint_test`` exactly once.

This program evaluates five checkpoints with the same data order, preprocessing,
mask threshold and metric implementation:

1. full Physics-ResCNN;
2. Physics-ResCNN without physics loss (the frozen supervised baseline);
3. Physics-ResCNN without mask loss;
4. Physics-ResCNN without smoothness loss;
5. Physics-ResCNN without auxiliary classification losses.

It never trains, updates or selects a checkpoint.  It first audits that the
model architecture, normalization and all controlled loss weights are the
expected frozen values.  ``--dry-run`` and ``--smoke-test`` do not open the
protected ``joint_test`` split.  The full split is opened only when the explicit
confirmation flag is supplied.

Place this file beside ``evaluate_joint_test.py`` and
``train_inverse_baseline_v2.py``.  Recommended PowerShell sequence::

    python .\evaluate_inverse_ablation_joint_test.py --dry-run
    python .\evaluate_inverse_ablation_joint_test.py --smoke-test
    python .\evaluate_inverse_ablation_joint_test.py --confirm-ablation-joint-test

If a run directory has a different name, override only that checkpoint::

    python .\evaluate_inverse_ablation_joint_test.py --dry-run `
      --checkpoint full=artifacts/my_physics_run\best_model.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from release_paths import DATA_ROOT, MODELS_ROOT, RECORDS_ROOT, OUTPUT_ROOT, SOURCE_ROOT
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


SCRIPT_VERSION = "inverse-ablation-joint-test-1.0.0"
PROGRAM_DIR = Path(__file__).resolve().parent
PC_DIR = (DATA_ROOT / 'pc_dataset_v1')
RUNS_ROOT = (MODELS_ROOT)
OUTPUT_DIR = (OUTPUT_ROOT / 'inverse_ablation_joint_test')
SMOKE_OUTPUT_DIR = (OUTPUT_ROOT / 'inverse_ablation_joint_test_smoke')
EXPECTED_JOINT_TEST_SAMPLES = 3571
EXPECTED_SCRIPT_VERSION = "1.2.0"
FULL_MODEL_ID = "full"
PRESELECTED_SIMPLIFIED_ID = "no_aux"

LOGGER = logging.getLogger("inverse_ablation_joint_test")


def _prime_numpy_openmp_runtime_on_windows() -> None:
    """Initialize NumPy BLAS before importing PyTorch in Windows Conda."""
    if os.name == "nt":
        probe = np.ones((2, 2), dtype=np.float32)
        result = np.dot(probe, probe)
        if not np.isfinite(result).all():
            raise RuntimeError("NumPy BLAS/OpenMP initialization probe failed.")


_prime_numpy_openmp_runtime_on_windows()


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    display_name: str
    candidate_checkpoint_paths: tuple[str, ...]
    physics_loss_weight: float
    mask_loss_weight: float = 0.25
    smoothness_weight: float = 0.10
    path_class_weight: float = 0.05
    height_class_weight: float = 0.05

    def expected_weights(self) -> dict[str, float]:
        return {
            "zero_edh_weight": 0.25,
            "mask_loss_weight": self.mask_loss_weight,
            "smoothness_weight": self.smoothness_weight,
            "path_class_weight": self.path_class_weight,
            "height_class_weight": self.height_class_weight,
            "physics_loss_weight": self.physics_loss_weight,
            "physics_scale_db": 5.0,
        }


EXPERIMENTS = (
    ExperimentSpec(
        "full",
        "Full Physics-ResCNN",
        (
            "inverse_physics_w0p01/best_model.pt",
            "inverse_physics_w0p0/best_model.pt",
            "inverse_physics/best_model.pt",
            "inverse_physics_final/best_model.pt",
        ),
        physics_loss_weight=0.01,
    ),
    ExperimentSpec(
        "no_physics",
        "w/o physics loss",
        (
            "inverse_baseline_v2/best_model.pt",
            "inverse_supervised_rescnn/best_model.pt",
        ),
        physics_loss_weight=0.0,
    ),
    ExperimentSpec(
        "no_mask",
        "w/o mask loss",
        ("inverse_ablation_suite/no_mask_loss/best_model.pt",),
        physics_loss_weight=0.01,
        mask_loss_weight=0.0,
    ),
    ExperimentSpec(
        "no_smoothness",
        "w/o smoothness loss",
        ("inverse_ablation_suite/no_smoothness_loss/best_model.pt",),
        physics_loss_weight=0.01,
        smoothness_weight=0.0,
    ),
    ExperimentSpec(
        "no_aux",
        "w/o auxiliary classification losses",
        ("inverse_ablation_suite/no_aux_classification_losses/best_model.pt",),
        physics_loss_weight=0.01,
        path_class_weight=0.0,
        height_class_weight=0.0,
    ),
)


COMMON_EXPECTED_CONFIG: dict[str, Any] = {
    "epochs": 180,
    "batch_size": 128,
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-4,
    "patience": 30,
    "minimum_improvement_m": 1.0e-3,
    "scheduler_name": "cosine_annealing",
    "minimum_learning_rate": 1.0e-5,
    "selection_metric": "positive_point_rmse_m",
    "mask_f1_penalty_m": 1.0,
    "seed": 20260813,
}


def _load_helper_module():
    path = PROGRAM_DIR / "evaluate_joint_test.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing helper program: {path}. Place both evaluators together."
        )
    import importlib.util

    module_name = "ablation_joint_test_helper"
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


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
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_checkpoint_overrides(items: Iterable[str]) -> dict[str, Path]:
    known = {spec.experiment_id for spec in EXPERIMENTS}
    result: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --checkpoint value {item!r}; expected MODEL_ID=PATH."
            )
        experiment_id, raw_path = item.split("=", 1)
        experiment_id = experiment_id.strip()
        if experiment_id not in known:
            raise ValueError(
                f"Unknown model id {experiment_id!r}; choices are {sorted(known)}."
            )
        if experiment_id in result:
            raise ValueError(f"Duplicate checkpoint override for {experiment_id!r}.")
        result[experiment_id] = Path(raw_path.strip().strip('"'))
    return result


def _resolve_checkpoints(args: argparse.Namespace) -> dict[str, Path]:
    overrides = _parse_checkpoint_overrides(args.checkpoint)
    resolved: dict[str, Path] = {}
    for spec in EXPERIMENTS:
        if spec.experiment_id in overrides:
            resolved[spec.experiment_id] = overrides[spec.experiment_id]
            continue
        candidates = [Path(args.runs_root) / value for value in spec.candidate_checkpoint_paths]
        match = next((path for path in candidates if path.is_file()), None)
        resolved[spec.experiment_id] = match if match is not None else candidates[0]
    return resolved


def _float_equal(left: Any, right: float, tolerance: float = 1.0e-10) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def _normalization_fingerprint(normalization: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in ("pc_mean_dB", "pc_std_dB", "edh_scale_m", "input_mode"):
        if key not in normalization:
            raise ValueError(f"Checkpoint normalization is missing {key!r}.")
        value = normalization[key]
        if key == "input_mode":
            array = np.asarray(value)
            item = array.reshape(()).item()
            if isinstance(item, bytes):
                item = item.decode("utf-8")
            digest.update(str(item).encode("utf-8"))
        else:
            array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_config(spec: ExperimentSpec, audit: Mapping[str, Any]) -> None:
    if str(audit.get("checkpoint_script_version", "")) != EXPECTED_SCRIPT_VERSION:
        raise ValueError(
            f"{spec.experiment_id}: checkpoint script_version="
            f"{audit.get('checkpoint_script_version')!r}, expected "
            f"{EXPECTED_SCRIPT_VERSION!r}."
        )
    if audit.get("input_mode") != "era5":
        raise ValueError(
            f"{spec.experiment_id}: expected ERA5-derived Pc input, got "
            f"{audit.get('input_mode')!r}."
        )
    train_config = dict(audit.get("train_config", {}))
    for key, expected in COMMON_EXPECTED_CONFIG.items():
        actual = train_config.get(key)
        if isinstance(expected, float):
            matched = _float_equal(actual, expected)
        else:
            matched = actual == expected
        if not matched:
            raise ValueError(
                f"{spec.experiment_id}: train_config[{key!r}]={actual!r}, "
                f"expected {expected!r}."
            )
    for key, expected in spec.expected_weights().items():
        actual = train_config.get(key)
        if not _float_equal(actual, expected):
            raise ValueError(
                f"{spec.experiment_id}: train_config[{key!r}]={actual!r}, "
                f"expected {expected!r}."
            )


def _audit_checkpoints(helper, paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    import torch

    device = torch.device("cpu")
    audits: list[dict[str, Any]] = []
    model_fingerprint: str | None = None
    normalization_fingerprint: str | None = None
    for spec in EXPERIMENTS:
        path = paths[spec.experiment_id]
        resolved = helper.ResolvedModel(
            spec.experiment_id,
            spec.display_name,
            PROGRAM_DIR / "train_inverse_baseline_v2.py",
            path,
        )
        audit = helper._checkpoint_audit(resolved, device)
        _validate_config(spec, audit)
        checkpoint = helper._torch_load(path, device)
        normalization = checkpoint["normalization"]
        current_normalization = _normalization_fingerprint(normalization)
        current_model = hashlib.sha256(
            json.dumps(audit["model_config"], sort_keys=True).encode("utf-8")
        ).hexdigest()
        if model_fingerprint is None:
            model_fingerprint = current_model
        elif current_model != model_fingerprint:
            raise ValueError(
                f"{spec.experiment_id}: model_config differs from the full model."
            )
        if normalization_fingerprint is None:
            normalization_fingerprint = current_normalization
        elif current_normalization != normalization_fingerprint:
            raise ValueError(
                f"{spec.experiment_id}: input normalization differs from the full model."
            )
        module = helper._load_python_module(
            PROGRAM_DIR / "train_inverse_baseline_v2.py",
            f"ablation_audit_{spec.experiment_id}",
        )
        configuration = module.ModelConfig(**dict(checkpoint["model_config"]))
        edh_scale_m = float(np.asarray(normalization["edh_scale_m"]).reshape(()))
        network = module.build_inverse_model(configuration, edh_scale_m).to(device)
        network.load_state_dict(checkpoint["model_state_dict"], strict=True)
        audit.update(
            {
                "parameter_count": int(
                    sum(parameter.numel() for parameter in network.parameters())
                ),
                "strict_state_dict_load_passed": True,
                "expected_loss_weights": spec.expected_weights(),
                "configuration_audit_passed": True,
                "normalization_sha256": current_normalization,
            }
        )
        audits.append(audit)
        LOGGER.info(
            "AUDIT OK | %-15s | epoch=%d | parameters=%s | %s",
            spec.experiment_id,
            audit["checkpoint_epoch"],
            f"{audit['parameter_count']:,}",
            path,
        )
        del network, checkpoint, module
    return audits


def _configure_matplotlib():
    cache = Path(tempfile.gettempdir()) / "matplotlib-ablation-joint-test"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_overall(summary: pd.DataFrame, output_path: Path) -> None:
    plt = _configure_matplotlib()
    labels = summary["display_name"].tolist()
    positions = np.arange(len(summary))
    colors = ["#D55E00" if value == FULL_MODEL_ID else "#0072B2" for value in summary["model_id"]]
    figure, axes = plt.subplots(1, 4, figsize=(18, 5.2), constrained_layout=True)
    columns = (
        ("rmse_m", "RMSE (m)", "(a) Overall EDH RMSE"),
        ("mae_m", "MAE (m)", "(b) Overall EDH MAE"),
        ("p95_sample_rmse_m", "P95 sample RMSE (m)", "(c) Tail error"),
        ("duct_mask_f1", "F1", "(d) Duct-mask F1"),
    )
    for axis, (column, label, title) in zip(axes, columns):
        axis.barh(positions, summary[column], color=colors)
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set(xlabel=label, title=title)
        axis.grid(axis="x", alpha=0.25)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_group_heatmap(groups: pd.DataFrame, output_path: Path) -> None:
    plt = _configure_matplotlib()
    selected = groups.loc[groups["dimension"] == "path_class_name"].copy()
    pivot = selected.pivot(index="display_name", columns="category", values="rmse_m")
    desired = [spec.display_name for spec in EXPERIMENTS]
    pivot = pivot.reindex([name for name in desired if name in pivot.index])
    figure, axis = plt.subplots(
        figsize=(max(9.0, 2.6 * len(pivot.columns)), 1.5 + 0.72 * len(pivot)),
        constrained_layout=True,
    )
    image = axis.imshow(pivot.to_numpy(), cmap="YlOrRd", aspect="auto")
    axis.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=18, ha="right")
    axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
    axis.set_title("Ablation joint-test RMSE by duct path class (m)")
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            value = pivot.iloc[row, column]
            axis.text(
                column,
                row,
                "--" if not np.isfinite(value) else f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=9,
            )
    figure.colorbar(image, ax=axis, label="RMSE (m)")
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _add_full_model_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    full_rows = result.loc[result["model_id"] == FULL_MODEL_ID]
    if len(full_rows) != 1:
        raise ValueError("Exactly one full model row is required.")
    full = full_rows.iloc[0]
    for column in (
        "rmse_m",
        "mae_m",
        "positive_point_rmse_m",
        "p95_sample_rmse_m",
        "balanced_duct_path_rmse_m",
    ):
        baseline = float(full[column])
        result[f"{column}_change_vs_full"] = result[column] - baseline
        result[f"{column}_change_vs_full_percent"] = (
            100.0 * (result[column] - baseline) / baseline
            if baseline != 0.0
            else np.nan
        )
    result["duct_mask_f1_change_vs_full"] = (
        result["duct_mask_f1"] - float(full["duct_mask_f1"])
    )
    return result


def run_dry_audit(args: argparse.Namespace) -> None:
    helper = _load_helper_module()
    paths = _resolve_checkpoints(args)
    audits = _audit_checkpoints(helper, paths)
    report = {
        "script_version": SCRIPT_VERSION,
        "joint_test_opened": False,
        "configuration_audit_passed": True,
        "model_count": len(audits),
        "models": audits,
        "recommended_next_action": (
            "Run --smoke-test. If it passes, use --confirm-ablation-joint-test "
            "once without changing any checkpoint."
        ),
    }
    print(json.dumps(_json_value(report), ensure_ascii=False, indent=2))


def _select_smoke_subset(core, data: Mapping[str, Any], count: int, seed: int):
    positions = core._stratified_positions(
        np.asarray(data["path_code"], dtype=np.int64),
        min(count, len(data["pc"])),
        seed,
    )
    return core._subset_role(dict(data), positions)


def run_evaluation(args: argparse.Namespace, role: str, output_directory: Path) -> None:
    import torch

    started = time.perf_counter()
    helper = _load_helper_module()
    core = helper._load_core()
    paths = _resolve_checkpoints(args)
    audits = _audit_checkpoints(helper, paths)
    helper._prepare_output_directory(output_directory, args.overwrite)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    LOGGER.info("Device: %s", device)

    # Every ablation consumes only ERA5-derived centered Pc.  radar_term_dB is
    # deliberately not required by this evaluation path.
    test_data = core.load_role_arrays(Path(args.pc_dir), role, need_radar=False)
    if role == "inverse_validation":
        test_data = _select_smoke_subset(
            core, test_data, args.smoke_samples, args.seed
        )
        disjointness: dict[str, int] = {}
    else:
        sample_count = len(test_data["pc"])
        if args.expected_joint_samples > 0 and sample_count != args.expected_joint_samples:
            raise ValueError(
                f"joint_test sample count mismatch: expected "
                f"{args.expected_joint_samples}, got {sample_count}."
            )
        disjointness = helper._audit_role_disjointness(
            Path(args.pc_dir), np.asarray(test_data["sample_id"], dtype=np.int64)
        )
    LOGGER.info("%s loaded: %d samples", role, len(test_data["pc"]))

    display_names = {spec.experiment_id: spec.display_name for spec in EXPERIMENTS}
    predictions: dict[str, np.ndarray] = {}
    prediction_masks: dict[str, np.ndarray] = {}
    model_metrics: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []

    for number, spec in enumerate(EXPERIMENTS, start=1):
        LOGGER.info(
            "Evaluating %d/%d: %s", number, len(EXPERIMENTS), spec.display_name
        )
        checkpoint = helper._torch_load(paths[spec.experiment_id], device)
        normalization = checkpoint["normalization"]
        prepared_input, input_mode = helper._prepare_model_input(
            test_data, normalization
        )
        loader = helper._make_test_loader(
            core,
            test_data,
            prepared_input,
            normalization,
            args.batch_size,
            args.workers,
            device,
        )
        configuration = core.ModelConfig(**dict(checkpoint["model_config"]))
        edh_scale_m = float(np.asarray(normalization["edh_scale_m"]).reshape(()))
        model = core.build_inverse_model(configuration, edh_scale_m).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        metrics, rows, arrays = core.evaluate(
            model, loader, device, test_data["metadata"]
        )
        prediction = np.asarray(arrays["prediction_edh"], dtype=np.float32)
        prediction_mask = np.asarray(arrays["prediction_mask"], dtype=bool)
        if prediction.shape != np.asarray(test_data["edh"]).shape:
            raise ValueError(
                f"{spec.experiment_id}: invalid prediction shape {prediction.shape}."
            )
        if not np.isfinite(prediction).all():
            raise ValueError(f"{spec.experiment_id}: non-finite prediction detected.")
        predictions[spec.experiment_id] = prediction
        prediction_masks[spec.experiment_id] = prediction_mask
        model_metrics[spec.experiment_id] = metrics
        audit = next(
            item for item in audits if item["model_id"] == spec.experiment_id
        )
        metric_rows.append(
            {
                "model_id": spec.experiment_id,
                "display_name": spec.display_name,
                "input_mode": input_mode,
                "parameter_count": audit["parameter_count"],
                "checkpoint_epoch": int(checkpoint["epoch"]),
                **helper._flatten_metrics(metrics),
            }
        )
        for row in rows:
            group_rows.append(
                {
                    "model_id": spec.experiment_id,
                    "display_name": spec.display_name,
                    "input_mode": input_mode,
                    **row,
                }
            )
        LOGGER.info(
            "%s | RMSE=%.4f m | MAE=%.4f m | positive=%.4f m | F1=%.5f",
            spec.display_name,
            metrics["rmse_m"],
            metrics["mae_m"],
            metrics["positive_point_rmse_m"],
            metrics["duct_mask_f1"],
        )
        del model, loader, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_table = _add_full_model_deltas(pd.DataFrame(metric_rows))
    summary_table["rmse_rank_for_reporting_only"] = summary_table["rmse_m"].rank(
        method="min", ascending=True
    ).astype(int)
    group_table = pd.DataFrame(group_rows)
    per_sample = helper._per_sample_rows(test_data, predictions, display_names)
    prefix = "ablation_joint_test" if role == "joint_test" else "ablation_smoke"
    summary_table.to_csv(
        output_directory / f"{prefix}_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    group_table.to_csv(
        output_directory / f"{prefix}_by_group.csv",
        index=False,
        encoding="utf-8-sig",
    )
    per_sample.to_csv(
        output_directory / f"{prefix}_per_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model_ids = [spec.experiment_id for spec in EXPERIMENTS]
    np.savez_compressed(
        output_directory / f"{prefix}_predictions.npz",
        model_ids=np.asarray(model_ids, dtype="U32"),
        display_names=np.asarray(
            [display_names[key] for key in model_ids], dtype="U96"
        ),
        sample_id=np.asarray(test_data["sample_id"], dtype=np.int64),
        input_range_km=np.asarray(test_data["input_range_km"], dtype=np.float32),
        pc_range_km=np.asarray(test_data["range_km"], dtype=np.float32),
        truth_edh_m=np.asarray(test_data["edh"], dtype=np.float32),
        truth_mask=np.asarray(test_data["mask"] > 0.5, dtype=bool),
        prediction_edh_m=np.stack([predictions[key] for key in model_ids]),
        prediction_mask=np.stack([prediction_masks[key] for key in model_ids]),
    )

    _plot_overall(summary_table, output_directory / f"{prefix}_overall_comparison.png")
    _plot_group_heatmap(
        group_table, output_directory / f"{prefix}_group_rmse_heatmap.png"
    )
    helper._plot_examples(
        test_data,
        predictions,
        display_names,
        FULL_MODEL_ID,
        output_directory / f"{prefix}_prediction_examples.png",
    )

    no_aux_row = summary_table.loc[
        summary_table["model_id"] == PRESELECTED_SIMPLIFIED_ID
    ].iloc[0]
    summary = {
        "script_version": SCRIPT_VERSION,
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_role": role,
        "joint_test_loaded": role == "joint_test",
        "sample_count": int(len(test_data["pc"])),
        "point_count": int(np.asarray(test_data["edh"]).size),
        "role_disjointness": disjointness,
        "input_shape": list(np.asarray(test_data["pc"]).shape),
        "target_shape": list(np.asarray(test_data["edh"]).shape),
        "models": audits,
        "metrics": model_metrics,
        "full_model_id": FULL_MODEL_ID,
        "validation_preselected_simplified_candidate": PRESELECTED_SIMPLIFIED_ID,
        "preselected_candidate_joint_test_rmse_change_vs_full_percent": float(
            no_aux_row["rmse_m_change_vs_full_percent"]
        ),
        "joint_test_based_model_selection_performed": False,
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": [
            f"{prefix}_summary.csv",
            f"{prefix}_by_group.csv",
            f"{prefix}_per_sample.csv",
            f"{prefix}_predictions.npz",
            f"{prefix}_overall_comparison.png",
            f"{prefix}_group_rmse_heatmap.png",
            f"{prefix}_prediction_examples.png",
        ],
        "recommended_next_action": (
            "Freeze these outputs and report the deltas versus the full model. "
            "Do not tune loss weights or reselect checkpoints from joint_test."
            if role == "joint_test"
            else "Run the protected split once with --confirm-ablation-joint-test."
        ),
    }
    summary_path = output_directory / f"{prefix}_summary.json"
    _write_json(summary_path, summary)
    completion_name = (
        "ABLATION_JOINT_TEST_COMPLETED.json"
        if role == "joint_test"
        else "ABLATION_SMOKE_COMPLETED.json"
    )
    _write_json(
        output_directory / completion_name,
        {
            "status": "complete",
            "script_version": SCRIPT_VERSION,
            "evaluation_role": role,
            "sample_count": int(len(test_data["pc"])),
            "evaluated_model_ids": model_ids,
            "summary_sha256": _sha256(summary_path),
        },
    )
    LOGGER.info("Evaluation complete: %s", output_directory)
    print(json.dumps(_json_value(summary), ensure_ascii=False, indent=2))


def run_self_test() -> None:
    expected_ids = {"full", "no_physics", "no_mask", "no_smoothness", "no_aux"}
    assert {spec.experiment_id for spec in EXPERIMENTS} == expected_ids
    full = next(spec for spec in EXPERIMENTS if spec.experiment_id == "full")
    for spec in EXPERIMENTS:
        changed = [
            key
            for key, value in spec.expected_weights().items()
            if not _float_equal(value, full.expected_weights()[key])
        ]
        expected_change_count = 0 if spec.experiment_id == "full" else (
            2 if spec.experiment_id == "no_aux" else 1
        )
        assert len(changed) == expected_change_count, (spec.experiment_id, changed)
    synthetic = pd.DataFrame(
        {
            "model_id": ["full", "no_aux"],
            "rmse_m": [1.0, 1.1],
            "mae_m": [0.5, 0.55],
            "positive_point_rmse_m": [1.2, 1.32],
            "p95_sample_rmse_m": [2.0, 2.2],
            "balanced_duct_path_rmse_m": [1.0, 1.1],
            "duct_mask_f1": [0.98, 0.97],
        }
    )
    with_deltas = _add_full_model_deltas(synthetic)
    assert np.isclose(with_deltas.iloc[1]["rmse_m_change_vs_full_percent"], 10.0)
    parsed = _parse_checkpoint_overrides([r"full=models/example/best_model.pt"])
    assert set(parsed) == {"full"}
    print("Ablation joint-test evaluator self-test passed; joint_test was not opened.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pc-dir", type=Path, default=PC_DIR)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="MODEL_ID=PATH",
        help=(
            "Override one frozen checkpoint; MODEL_ID is full, no_physics, "
            "no_mask, no_smoothness or no_aux. May be repeated."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--smoke-samples", type=int, default=256)
    parser.add_argument(
        "--expected-joint-samples", type=int, default=EXPECTED_JOINT_TEST_SAMPLES
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--confirm-ablation-joint-test", action="store_true")
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
    if args.smoke_samples <= 0 or args.expected_joint_samples < 0:
        raise ValueError(
            "--smoke-samples must be positive and --expected-joint-samples nonnegative."
        )
    selected_modes = sum(
        bool(value)
        for value in (
            args.dry_run,
            args.smoke_test,
            args.confirm_ablation_joint_test,
        )
    )
    if selected_modes != 1:
        raise RuntimeError(
            "Choose exactly one mode: --dry-run, --smoke-test, or "
            "--confirm-ablation-joint-test."
        )
    if args.dry_run:
        run_dry_audit(args)
        return
    if args.smoke_test:
        output = Path(args.output_dir) if args.output_dir else SMOKE_OUTPUT_DIR
        run_evaluation(args, "inverse_validation", output)
        return
    output = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    run_evaluation(args, "joint_test", output)


if __name__ == "__main__":
    main()
