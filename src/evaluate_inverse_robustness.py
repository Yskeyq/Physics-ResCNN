#!/usr/bin/env python3
r"""Post-freeze robustness evaluation for Pc(r) -> EDH(r) inverse models.

This script evaluates already-frozen checkpoints.  It never trains, updates or
selects a model.  The same deterministic perturbation realization is applied
to every model so that comparisons are paired sample by sample.

The fixed paper suite contains:

* clean centered clutter;
* zero-mean white log-power perturbations with sigma 0.5, 1.0 and 2.0 dB;
* range-correlated log-power perturbations with the same sigma values and a
  default 5 km Gaussian correlation scale;
* range-dependent calibration/backscatter slopes of -4, -2, +2 and +4 dB
  peak-to-peak across 100 km;
* a +5 dB constant offset sanity check, which should be cancelled by the
  per-profile centering used by the inversion pipeline.

These are perturbations in the measured dB/log-power profile, not voltage-domain
AWGN and not an asserted receiver SNR.  Report them as measurement/clutter-model
perturbations with their explicit dB magnitudes.

Keep this file beside ``evaluate_joint_test.py`` and all training architecture
files.  PowerShell examples are intentionally written on one line so they can
be pasted directly.  A dry audit does not open joint_test::

    python .\evaluate_inverse_robustness.py --dry-run --checkpoint rfcformer_adapted=artifacts/inverse_rfcformer_adapted_v2_low_dropout\best_model.pt

Run a small interface test on inverse_validation::

    python .\evaluate_inverse_robustness.py --smoke-test --checkpoint rfcformer_adapted=artifacts/inverse_rfcformer_adapted_v2_low_dropout\best_model.pt --output-dir artifacts/inverse_robustness_smoke

After all checkpoints and the perturbation suite are frozen, run the complete
joint-test robustness experiment::

    python .\evaluate_inverse_robustness.py --checkpoint rfcformer_adapted=artifacts/inverse_rfcformer_adapted_v2_low_dropout\best_model.pt --output-dir artifacts/inverse_robustness_joint_test --confirm-joint-test-robustness
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from release_paths import DATA_ROOT, MODELS_ROOT, RECORDS_ROOT, OUTPUT_ROOT, SOURCE_ROOT
from typing import Any, Mapping

import numpy as np
import pandas as pd


SCRIPT_VERSION = "inverse-robustness-evaluator-1.0.0"
PROGRAM_DIR = Path(__file__).resolve().parent
JOINT_EVALUATOR_FILENAME = "evaluate_joint_test.py"
PC_DIR = (DATA_ROOT / 'pc_dataset_v1')
RUNS_ROOT = (MODELS_ROOT)
OUTPUT_DIR = (OUTPUT_ROOT / 'inverse_robustness_joint_test')
SMOKE_OUTPUT_DIR = (OUTPUT_ROOT / 'inverse_robustness_smoke')
EXPECTED_JOINT_TEST_SAMPLES = 3571
SELECTED_RFCFORMER_VERSION = "comparison-rfcformer-adapted-2.0.0"

LOGGER = logging.getLogger("inverse_robustness")


def _prime_numpy_openmp_runtime_on_windows() -> None:
    """Establish NumPy/MKL before importing PyTorch on Windows."""
    if os.name == "nt":
        probe = np.ones((2, 2), dtype=np.float32)
        result = np.dot(probe, probe)
        if not np.isfinite(result).all():
            raise RuntimeError("NumPy BLAS/OpenMP initialization probe failed.")


_prime_numpy_openmp_runtime_on_windows()


def _load_joint_evaluator():
    path = PROGRAM_DIR / JOINT_EVALUATOR_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file is missing: {path}. Place "
            f"{JOINT_EVALUATOR_FILENAME} beside this script."
        )
    name = "inverse_robustness_joint_core"
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


joint = _load_joint_evaluator()


@dataclass(frozen=True)
class PerturbationSpec:
    condition: str
    family: str
    level_db: float
    description: str


FULL_SUITE = (
    PerturbationSpec("clean", "clean", 0.0, "Unperturbed centered Pc."),
    PerturbationSpec(
        "white_sigma_0p5dB",
        "white_noise_db",
        0.5,
        "Zero-mean white log-power perturbation; per-sample RMS=0.5 dB.",
    ),
    PerturbationSpec(
        "white_sigma_1dB",
        "white_noise_db",
        1.0,
        "Zero-mean white log-power perturbation; per-sample RMS=1.0 dB.",
    ),
    PerturbationSpec(
        "white_sigma_2dB",
        "white_noise_db",
        2.0,
        "Zero-mean white log-power perturbation; per-sample RMS=2.0 dB.",
    ),
    PerturbationSpec(
        "correlated_sigma_0p5dB",
        "correlated_noise_db",
        0.5,
        "Range-correlated log-power perturbation; per-sample RMS=0.5 dB.",
    ),
    PerturbationSpec(
        "correlated_sigma_1dB",
        "correlated_noise_db",
        1.0,
        "Range-correlated log-power perturbation; per-sample RMS=1.0 dB.",
    ),
    PerturbationSpec(
        "correlated_sigma_2dB",
        "correlated_noise_db",
        2.0,
        "Range-correlated log-power perturbation; per-sample RMS=2.0 dB.",
    ),
    PerturbationSpec(
        "range_slope_minus_4dB",
        "range_slope_pp_db",
        -4.0,
        "Far-minus-near range-dependent mismatch=-4 dB across the path.",
    ),
    PerturbationSpec(
        "range_slope_minus_2dB",
        "range_slope_pp_db",
        -2.0,
        "Far-minus-near range-dependent mismatch=-2 dB across the path.",
    ),
    PerturbationSpec(
        "range_slope_plus_2dB",
        "range_slope_pp_db",
        2.0,
        "Far-minus-near range-dependent mismatch=+2 dB across the path.",
    ),
    PerturbationSpec(
        "range_slope_plus_4dB",
        "range_slope_pp_db",
        4.0,
        "Far-minus-near range-dependent mismatch=+4 dB across the path.",
    ),
    PerturbationSpec(
        "constant_offset_plus_5dB",
        "constant_offset_db",
        5.0,
        "Uniform +5 dB offset before per-profile centering; sanity check.",
    ),
)

SMOKE_CONDITION_NAMES = {
    "clean",
    "white_sigma_1dB",
    "correlated_sigma_1dB",
    "range_slope_minus_2dB",
    "range_slope_plus_2dB",
    "constant_offset_plus_5dB",
}


def _center(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.asarray(
        values - np.mean(values, axis=1, keepdims=True, dtype=np.float64),
        dtype=np.float32,
    )


def _unit_rms(values: np.ndarray) -> np.ndarray:
    centered = _center(values)
    rms = np.sqrt(np.mean(centered.astype(np.float64) ** 2, axis=1, keepdims=True))
    if np.any(rms <= 1.0e-12):
        raise ValueError("Cannot normalize a zero perturbation realization.")
    return np.asarray(centered / rms, dtype=np.float32)


class PerturbationFactory:
    """Generate deterministic, paired perturbations for every model."""

    def __init__(
        self,
        sample_count: int,
        range_km: np.ndarray,
        seed: int,
        correlation_length_km: float,
    ):
        self.sample_count = int(sample_count)
        self.range_km = np.asarray(range_km, dtype=np.float32)
        if self.sample_count <= 0 or self.range_km.ndim != 1:
            raise ValueError("Invalid perturbation dimensions.")
        if len(self.range_km) < 2 or not np.all(np.diff(self.range_km) > 0.0):
            raise ValueError("range_km must be strictly increasing.")
        if correlation_length_km <= 0.0:
            raise ValueError("correlation_length_km must be positive.")
        self.correlation_length_km = float(correlation_length_km)
        rng = np.random.default_rng(int(seed))
        shape = (self.sample_count, len(self.range_km))
        self.white_unit = _unit_rms(rng.standard_normal(shape))

        raw = rng.standard_normal(shape)
        spacing_km = float(np.median(np.diff(self.range_km)))
        frequency = np.fft.rfftfreq(len(self.range_km), d=spacing_km)
        transfer = np.exp(
            -0.5
            * (2.0 * np.pi * frequency * self.correlation_length_km) ** 2
        )
        transfer[0] = 0.0
        correlated = np.fft.irfft(
            np.fft.rfft(raw, axis=1) * transfer[None, :],
            n=len(self.range_km),
            axis=1,
        )
        self.correlated_unit = _unit_rms(correlated)

        span = float(self.range_km[-1] - self.range_km[0])
        if span <= 0.0:
            raise ValueError("Range span must be positive.")
        self.slope_axis = np.asarray(
            (self.range_km - np.mean(self.range_km, dtype=np.float64)) / span,
            dtype=np.float32,
        )

    def raw_delta(self, spec: PerturbationSpec) -> np.ndarray:
        shape = (self.sample_count, len(self.range_km))
        if spec.family == "clean":
            return np.zeros(shape, dtype=np.float32)
        if spec.family == "white_noise_db":
            return np.asarray(self.white_unit * spec.level_db, dtype=np.float32)
        if spec.family == "correlated_noise_db":
            return np.asarray(
                self.correlated_unit * spec.level_db, dtype=np.float32
            )
        if spec.family == "range_slope_pp_db":
            return np.broadcast_to(
                self.slope_axis[None, :] * spec.level_db, shape
            ).astype(np.float32, copy=True)
        if spec.family == "constant_offset_db":
            return np.full(shape, spec.level_db, dtype=np.float32)
        raise ValueError(f"Unknown perturbation family: {spec.family}")

    def apply(self, centered_pc: np.ndarray, spec: PerturbationSpec) -> np.ndarray:
        base = np.asarray(centered_pc, dtype=np.float32)
        expected = (self.sample_count, len(self.range_km))
        if base.shape != expected:
            raise ValueError(f"Expected centered Pc shape {expected}, got {base.shape}.")
        if spec.family == "clean":
            result = base.copy()
        else:
            result = _center(base + self.raw_delta(spec))
        if not np.isfinite(result).all():
            raise ValueError(f"Non-finite perturbed input for {spec.condition}.")
        return result

    def manifest_row(self, spec: PerturbationSpec) -> dict[str, Any]:
        zero = np.zeros(
            (self.sample_count, len(self.range_km)), dtype=np.float32
        )
        effective = self.apply(zero, spec)
        sample_rms = np.sqrt(
            np.mean(effective.astype(np.float64) ** 2, axis=1)
        )
        return {
            **asdict(spec),
            "correlation_length_km": (
                self.correlation_length_km
                if spec.family == "correlated_noise_db"
                else np.nan
            ),
            "effective_mean_rms_db_after_centering": float(np.mean(sample_rms)),
            "effective_max_abs_db_after_centering": float(
                np.max(np.abs(effective))
            ),
        }


def _subset_data(core, data: Mapping[str, Any], total: int, seed: int):
    positions = core._stratified_positions(
        np.asarray(data["path_code"], dtype=np.int64), total, seed
    )
    return core._subset_role(dict(data), positions)


def _resolved_display_names(models):
    result = []
    for model in models:
        display = (
            "RFCFormer-adapted-v2"
            if model.model_id == "rfcformer_adapted"
            else model.display_name
        )
        result.append(
            joint.ResolvedModel(
                model.model_id,
                display,
                model.script_path,
                model.checkpoint_path,
            )
        )
    return result


def _audit_selected_models(models, audits) -> None:
    for model, audit in zip(models, audits):
        if model.model_id == "rfcformer_adapted":
            version = str(audit.get("checkpoint_script_version", ""))
            if version != SELECTED_RFCFORMER_VERSION:
                raise ValueError(
                    "The robustness experiment requires the validation-selected "
                    "RFCFormer-adapted-v2 low-dropout checkpoint. Got script "
                    f"version {version!r}. Use --checkpoint "
                    "rfcformer_adapted=artifacts/inverse_rfcformer_adapted_v2_"
                    "low_dropout\\best_model.pt"
                )


def _plot_noise_curves(summary: pd.DataFrame, output_path: Path) -> None:
    plt = joint._configure_matplotlib()
    figure, axes = plt.subplots(
        1, 2, figsize=(14.5, 5.4), constrained_layout=True
    )
    families = (
        ("white_noise_db", "(a) White log-power perturbation"),
        ("correlated_noise_db", "(b) 5 km-correlated perturbation"),
    )
    model_order = summary["model_id"].drop_duplicates().tolist()
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(model_order))))
    for axis, (family, title) in zip(axes, families):
        for color, model_id in zip(colors, model_order):
            selected = summary.loc[
                (summary["model_id"] == model_id)
                & (summary["family"].isin(("clean", family)))
            ].sort_values("level_db")
            axis.plot(
                selected["level_db"],
                selected["rmse_m"],
                marker="o",
                linewidth=1.8,
                color=color,
                label=str(selected["display_name"].iloc[0]),
            )
        axis.set(
            xlabel="Perturbation RMS (dB)",
            ylabel="EDH RMSE (m)",
            title=title,
        )
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _plot_range_slope(summary: pd.DataFrame, output_path: Path) -> None:
    plt = joint._configure_matplotlib()
    figure, axis = plt.subplots(figsize=(8.8, 5.5), constrained_layout=True)
    model_order = summary["model_id"].drop_duplicates().tolist()
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(model_order))))
    for color, model_id in zip(colors, model_order):
        selected = summary.loc[
            (summary["model_id"] == model_id)
            & (summary["family"].isin(("clean", "range_slope_pp_db")))
        ].sort_values("level_db")
        axis.plot(
            selected["level_db"],
            selected["rmse_m"],
            marker="o",
            linewidth=1.8,
            color=color,
            label=str(selected["display_name"].iloc[0]),
        )
    axis.set(
        xlabel="Far-minus-near profile mismatch (dB across 100 km)",
        ylabel="EDH RMSE (m)",
        title="Range-dependent clutter/radar-term mismatch",
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    joint._write_json(path, payload)


def _run_evaluation(args, models) -> None:
    import torch

    started = time.perf_counter()
    smoke = bool(args.smoke_test)
    role = "inverse_validation" if smoke else "joint_test"
    if not smoke and not args.confirm_joint_test_robustness:
        raise RuntimeError(
            "Refusing to open joint_test without "
            "--confirm-joint-test-robustness. Run --dry-run first."
        )
    output_directory = Path(args.output_dir)
    if smoke and output_directory == OUTPUT_DIR:
        output_directory = SMOKE_OUTPUT_DIR
    joint._prepare_output_directory(output_directory, args.overwrite)

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    LOGGER.info("Device: %s", device)
    core = joint._load_core()
    audits = [joint._checkpoint_audit(model, torch.device("cpu")) for model in models]
    _audit_selected_models(models, audits)

    data = core.load_role_arrays(Path(args.pc_dir), role, need_radar=True)
    if smoke:
        data = _subset_data(core, data, args.smoke_samples, args.seed)
    elif args.expected_samples > 0 and len(data["pc"]) != args.expected_samples:
        raise ValueError(
            f"joint_test sample count mismatch: expected {args.expected_samples}, "
            f"got {len(data['pc'])}."
        )
    disjointness = (
        joint._audit_role_disjointness(
            Path(args.pc_dir), np.asarray(data["sample_id"], dtype=np.int64)
        )
        if role == "joint_test"
        else {}
    )
    LOGGER.info("%s loaded: %d samples", role, len(data["pc"]))

    conditions = (
        tuple(spec for spec in FULL_SUITE if spec.condition in SMOKE_CONDITION_NAMES)
        if smoke
        else FULL_SUITE
    )
    factory = PerturbationFactory(
        len(data["pc"]), data["range_km"], args.seed, args.correlation_length_km
    )
    manifest = pd.DataFrame([factory.manifest_row(spec) for spec in conditions])
    manifest.to_csv(
        output_directory / "robustness_input_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metric_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    primary_sample_table = data["metadata"][[
        "sample_id",
        "pe_row_index_0based",
        "path_class_name",
        "height_relation_class_name",
    ]].copy()

    for model_number, (resolved, audit) in enumerate(zip(models, audits), start=1):
        LOGGER.info(
            "Loading model %d/%d: %s", model_number, len(models), resolved.display_name
        )
        model_module = joint._load_python_module(
            resolved.script_path, f"robustness_model_{resolved.model_id}"
        )
        checkpoint = joint._torch_load(resolved.checkpoint_path, device)
        normalization = checkpoint["normalization"]
        base_input, input_mode = joint._prepare_model_input(data, normalization)
        configuration = model_module.ModelConfig(**dict(checkpoint["model_config"]))
        edh_scale_m = float(np.asarray(normalization["edh_scale_m"]).reshape(()))
        model = model_module.build_inverse_model(configuration, edh_scale_m).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))

        for condition_number, spec in enumerate(conditions, start=1):
            prepared = factory.apply(base_input, spec)
            loader = joint._make_test_loader(
                core,
                data,
                prepared,
                normalization,
                args.batch_size,
                args.workers,
                device,
            )
            metrics, rows, arrays = core.evaluate(
                model, loader, device, data["metadata"]
            )
            metric_rows.append(
                {
                    "model_id": resolved.model_id,
                    "display_name": resolved.display_name,
                    "input_mode": input_mode,
                    "parameter_count": parameter_count,
                    "checkpoint_epoch": int(checkpoint["epoch"]),
                    "condition": spec.condition,
                    "family": spec.family,
                    "level_db": spec.level_db,
                    **joint._flatten_metrics(metrics),
                }
            )
            for row in rows:
                group_rows.append(
                    {
                        "model_id": resolved.model_id,
                        "display_name": resolved.display_name,
                        "condition": spec.condition,
                        "family": spec.family,
                        "level_db": spec.level_db,
                        **row,
                    }
                )
            if resolved.model_id == args.primary_model:
                error = (
                    np.asarray(arrays["prediction_edh"], dtype=np.float64)
                    - np.asarray(data["edh"], dtype=np.float64)
                )
                primary_sample_table[
                    f"{spec.condition}_sample_rmse_m"
                ] = np.sqrt(np.mean(error * error, axis=1))
            LOGGER.info(
                "%s | %02d/%02d %-29s | RMSE=%.4f m | P95=%.4f m",
                resolved.model_id,
                condition_number,
                len(conditions),
                spec.condition,
                metrics["rmse_m"],
                metrics["p95_sample_rmse_m"],
            )
            del loader

        del model, checkpoint, model_module
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.primary_model not in set(row["model_id"] for row in metric_rows):
        raise ValueError(f"Primary model {args.primary_model!r} was not evaluated.")

    summary_table = pd.DataFrame(metric_rows)
    clean = summary_table.loc[summary_table["condition"] == "clean"].set_index(
        "model_id"
    )
    summary_table["clean_rmse_m"] = summary_table["model_id"].map(clean["rmse_m"])
    summary_table["clean_mae_m"] = summary_table["model_id"].map(clean["mae_m"])
    summary_table["rmse_degradation_percent"] = 100.0 * (
        summary_table["rmse_m"] / summary_table["clean_rmse_m"] - 1.0
    )
    summary_table["mae_degradation_percent"] = 100.0 * (
        summary_table["mae_m"] / summary_table["clean_mae_m"] - 1.0
    )
    summary_table.to_csv(
        output_directory / "robustness_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(group_rows).to_csv(
        output_directory / "robustness_by_group.csv",
        index=False,
        encoding="utf-8-sig",
    )
    primary_sample_table.to_csv(
        output_directory / "robustness_per_sample_primary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _plot_noise_curves(
        summary_table, output_directory / "robustness_noise_curves.png"
    )
    _plot_range_slope(
        summary_table, output_directory / "robustness_range_slope.png"
    )

    clean_metrics = {
        row["model_id"]: {
            "rmse_m": float(row["rmse_m"]),
            "mae_m": float(row["mae_m"]),
            "r2": float(row["r2"]),
            "p95_sample_rmse_m": float(row["p95_sample_rmse_m"]),
        }
        for row in metric_rows
        if row["condition"] == "clean"
    }
    payload = {
        "script_version": SCRIPT_VERSION,
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_role": role,
        "smoke_test": smoke,
        "training_or_checkpoint_selection_performed": False,
        "joint_test_loaded": role == "joint_test",
        "sample_count": len(data["pc"]),
        "point_count": int(np.asarray(data["edh"]).size),
        "role_disjointness": disjointness,
        "seed": args.seed,
        "correlation_length_km": args.correlation_length_km,
        "perturbation_domain": "centered radar sea-clutter log power in dB",
        "conditions": [asdict(spec) for spec in conditions],
        "models": audits,
        "clean_metrics": clean_metrics,
        "primary_model": args.primary_model,
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": [
            "robustness_summary.csv",
            "robustness_by_group.csv",
            "robustness_input_manifest.csv",
            "robustness_per_sample_primary.csv",
            "robustness_noise_curves.png",
            "robustness_range_slope.png",
        ],
        "recommended_next_action": (
            "Freeze and report these post-freeze robustness results; do not use "
            "them for model or checkpoint tuning."
        ),
    }
    summary_path = output_directory / "robustness_summary.json"
    _write_json(summary_path, payload)
    completion = {
        "status": "complete",
        "script_version": SCRIPT_VERSION,
        "evaluation_role": role,
        "sample_count": len(data["pc"]),
        "evaluated_model_ids": [model.model_id for model in models],
        "condition_names": [spec.condition for spec in conditions],
        "summary_sha256": joint._sha256(summary_path),
    }
    _write_json(
        output_directory / "ROBUSTNESS_EVALUATION_COMPLETED.json", completion
    )
    LOGGER.info("Robustness evaluation complete: %s", output_directory)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def run_self_test() -> None:
    range_km = np.linspace(0.2, 100.0, 500, dtype=np.float32)
    factory_a = PerturbationFactory(8, range_km, 20260814, 5.0)
    factory_b = PerturbationFactory(8, range_km, 20260814, 5.0)
    base = np.tile(np.linspace(-20.0, 20.0, 500, dtype=np.float32), (8, 1))
    base = _center(base)
    by_name = {spec.condition: spec for spec in FULL_SUITE}
    white = factory_a.apply(base, by_name["white_sigma_1dB"]) - base
    correlated = factory_a.apply(base, by_name["correlated_sigma_1dB"]) - base
    white_rms = np.sqrt(np.mean(white.astype(np.float64) ** 2, axis=1))
    correlated_rms = np.sqrt(
        np.mean(correlated.astype(np.float64) ** 2, axis=1)
    )
    assert np.allclose(white_rms, 1.0, atol=1.0e-5)
    assert np.allclose(correlated_rms, 1.0, atol=1.0e-5)
    assert np.allclose(factory_a.white_unit, factory_b.white_unit)
    offset = factory_a.apply(base, by_name["constant_offset_plus_5dB"])
    assert np.allclose(offset, base, atol=2.0e-6)
    slope = factory_a.apply(base, by_name["range_slope_plus_4dB"]) - base
    assert np.allclose(slope[:, -1] - slope[:, 0], 4.0, atol=1.0e-5)
    assert np.max(np.abs(np.mean(white, axis=1))) < 1.0e-6

    synthetic = []
    for model_id, display in (("a", "Model A"), ("b", "Model B")):
        for spec in FULL_SUITE:
            synthetic.append(
                {
                    "model_id": model_id,
                    "display_name": display,
                    "condition": spec.condition,
                    "family": spec.family,
                    "level_db": spec.level_db,
                    "rmse_m": 0.5 + 0.1 * abs(spec.level_db),
                }
            )
    with tempfile.TemporaryDirectory(prefix="inverse-robustness-selftest-") as tmp:
        table = pd.DataFrame(synthetic)
        noise_path = Path(tmp) / "noise.png"
        slope_path = Path(tmp) / "slope.png"
        _plot_noise_curves(table, noise_path)
        _plot_range_slope(table, slope_path)
        assert noise_path.is_file() and noise_path.stat().st_size > 0
        assert slope_path.is_file() and slope_path.stat().st_size > 0
    print("Inverse robustness self-test passed; no dataset or checkpoint was opened.")


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
        help="Override one frozen checkpoint path; may be repeated.",
    )
    parser.add_argument(
        "--skip-model",
        action="append",
        default=[],
        choices=[definition.model_id for definition in joint.MODEL_DEFINITIONS],
    )
    parser.add_argument(
        "--primary-model",
        choices=[definition.model_id for definition in joint.MODEL_DEFINITIONS],
        default="physics_rescnn",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--expected-samples", type=int, default=EXPECTED_JOINT_TEST_SAMPLES)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--correlation-length-km", type=float, default=5.0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-samples", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-joint-test-robustness", action="store_true")
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
    if args.expected_samples < 0 or args.smoke_samples <= 0:
        raise ValueError("Sample counts must be nonnegative/positive.")
    if args.correlation_length_km <= 0.0:
        raise ValueError("--correlation-length-km must be positive.")

    models = _resolved_display_names(joint.resolve_models(args))
    if args.dry_run:
        joint.run_dry_audit(args, models)
        return
    _run_evaluation(args, models)


if __name__ == "__main__":
    main()
