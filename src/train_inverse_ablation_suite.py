#!/usr/bin/env python3
"""Run the three frozen loss-term ablations for Physics-ResCNN.

This is a sequential experiment launcher, not a training implementation.  It
calls ``train_inverse_baseline_v2.py`` three times with one controlled change
per run:

1. ``no_mask_loss``: mask-loss weight 0 instead of 0.25;
2. ``no_smoothness_loss``: smoothness-loss weight 0 instead of 0.10;
3. ``no_aux_classification_losses``: both auxiliary classification weights 0
   instead of 0.05.

All other settings are explicitly frozen to the selected Physics-ResCNN run,
including the ERA5-derived Pc input, physics-loss weight 0.01, seed, optimizer,
scheduler, model width and checkpoint-selection metric.  The launcher only
loads ``inverse_train`` and ``inverse_validation`` through the underlying
training script.  It never opens ``joint_test``.

Place this file beside ``train_inverse_baseline_v2.py`` and
``train_forward_rescnn.py``.  PowerShell examples::

    python .\train_inverse_ablation_suite.py --dry-run
    python .\train_inverse_ablation_suite.py --smoke-test
    python .\train_inverse_ablation_suite.py

Completed and configuration-matched child runs are skipped by default, making
the suite safe to restart after interruption.  Use ``--overwrite`` only when a
partial or mismatched child directory must intentionally be rerun.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from release_paths import DATA_ROOT, MODELS_ROOT, RECORDS_ROOT, OUTPUT_ROOT, SOURCE_ROOT
from typing import Any, Mapping, Sequence


SCRIPT_VERSION = "inverse-ablation-suite-1.0.0"
PROGRAM_DIR = Path(__file__).resolve().parent
DEFAULT_PC_DIR = (DATA_ROOT / 'pc_dataset_v1')
DEFAULT_FORWARD_CHECKPOINT = (MODELS_ROOT / 'forward_rescnn' / 'best_model.pt')
DEFAULT_OUTPUT_ROOT = (OUTPUT_ROOT / 'inverse_ablation_suite')
DEFAULT_SMOKE_OUTPUT_ROOT = (OUTPUT_ROOT / 'inverse_ablation_suite_smoke')

LOGGER = logging.getLogger("inverse_ablation_suite")


@dataclass(frozen=True)
class AblationSpec:
    experiment_id: str
    display_name: str
    description: str
    mask_loss_weight: float = 0.25
    smoothness_weight: float = 0.10
    path_class_weight: float = 0.05
    height_class_weight: float = 0.05


ABLATIONS = (
    AblationSpec(
        experiment_id="no_mask_loss",
        display_name="Physics-ResCNN w/o mask loss",
        description=(
            "Remove supervised duct-mask loss while retaining the same mask "
            "head, physics loop and inference gate."
        ),
        mask_loss_weight=0.0,
    ),
    AblationSpec(
        experiment_id="no_smoothness_loss",
        display_name="Physics-ResCNN w/o smoothness loss",
        description="Remove the EDH range-gradient matching loss.",
        smoothness_weight=0.0,
    ),
    AblationSpec(
        experiment_id="no_aux_classification_losses",
        display_name="Physics-ResCNN w/o auxiliary classification losses",
        description=(
            "Remove both path-class and antenna-height-relation auxiliary "
            "classification losses."
        ),
        path_class_weight=0.0,
        height_class_weight=0.0,
    ),
)

COMMON_TRAIN_CONFIG: dict[str, Any] = {
    "input_mode": "era5",
    "epochs": 180,
    "batch_size": 128,
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-4,
    "patience": 30,
    "minimum_improvement_m": 1.0e-3,
    "minimum_learning_rate": 1.0e-5,
    "selection_metric": "positive_point_rmse_m",
    "mask_f1_penalty_m": 1.0,
    "seed": 20260813,
    "width": 64,
    "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
    "dropout": 0.08,
    "zero_edh_weight": 0.25,
    "physics_loss_weight": 0.01,
    "physics_scale_db": 5.0,
}


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _format_command(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def _expected_weights(spec: AblationSpec) -> dict[str, float]:
    return {
        "zero_edh_weight": float(COMMON_TRAIN_CONFIG["zero_edh_weight"]),
        "mask_loss_weight": float(spec.mask_loss_weight),
        "smoothness_weight": float(spec.smoothness_weight),
        "path_class_weight": float(spec.path_class_weight),
        "height_class_weight": float(spec.height_class_weight),
        "physics_loss_weight": float(COMMON_TRAIN_CONFIG["physics_loss_weight"]),
        "physics_scale_db": float(COMMON_TRAIN_CONFIG["physics_scale_db"]),
    }


def _float_equal(left: Any, right: float, tolerance: float = 1.0e-10) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def _validate_summary(
    output_directory: Path,
    spec: AblationSpec,
    expected_epochs: int,
) -> dict[str, Any]:
    summary_path = output_directory / "training_summary.json"
    checkpoint_path = output_directory / "best_model.pt"
    history_path = output_directory / "training_history.csv"
    missing = [
        path.name
        for path in (summary_path, checkpoint_path, history_path)
        if not path.is_file()
    ]
    if missing:
        raise ValueError(f"{output_directory} is incomplete; missing {missing}.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if bool(summary.get("joint_test_loaded", True)):
        raise ValueError(f"{summary_path} does not certify joint_test_loaded=false.")
    if str(summary.get("input_mode", "")) != "era5":
        raise ValueError(f"{summary_path} was not trained with input_mode=era5.")
    roles = set(summary.get("roles_loaded", []))
    if roles != {"inverse_train", "inverse_validation"}:
        raise ValueError(f"Unexpected roles in {summary_path}: {sorted(roles)}")
    train_config = dict(summary.get("train_config", {}))
    if int(train_config.get("epochs", -1)) != int(expected_epochs):
        raise ValueError(
            f"{summary_path} has epochs={train_config.get('epochs')}, "
            f"expected {expected_epochs}."
        )
    for key, expected in _expected_weights(spec).items():
        if not _float_equal(train_config.get(key), expected):
            raise ValueError(
                f"{summary_path}: {key}={train_config.get(key)!r}, "
                f"expected {expected}."
            )
    if int(train_config.get("seed", -1)) != int(COMMON_TRAIN_CONFIG["seed"]):
        raise ValueError(f"Unexpected seed in {summary_path}.")
    if not bool(summary.get("plot_rendering_passed", False)):
        raise ValueError(f"Plot rendering did not pass for {output_directory}.")
    metrics = dict(summary.get("validation_metrics", {}))
    for key in ("rmse_m", "mae_m", "positive_point_rmse_m", "duct_mask_f1"):
        try:
            value = float(metrics[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Missing/invalid validation metric {key}.") from exc
        if not (-1.0e30 < value < 1.0e30):
            raise ValueError(f"Non-finite validation metric {key}={value}.")
    return summary


def _completion_state(
    output_directory: Path,
    spec: AblationSpec,
    expected_epochs: int,
) -> tuple[str, dict[str, Any] | None, str | None]:
    if not output_directory.exists() or not any(output_directory.iterdir()):
        return "pending", None, None
    try:
        return "completed", _validate_summary(output_directory, spec, expected_epochs), None
    except Exception as exc:  # preserve diagnostic for partial/mismatched runs
        return "partial_or_mismatched", None, str(exc)


def _build_command(
    args: argparse.Namespace,
    spec: AblationSpec,
    output_directory: Path,
) -> list[str]:
    common = COMMON_TRAIN_CONFIG
    command = [
        str(args.python_executable),
        str(args.baseline_script),
        "--pc-dir",
        str(args.pc_dir),
        "--output-dir",
        str(output_directory),
        "--forward-script",
        str(args.forward_script),
        "--forward-checkpoint",
        str(args.forward_checkpoint),
        "--input-mode",
        str(common["input_mode"]),
        "--epochs",
        str(common["epochs"]),
        "--batch-size",
        str(common["batch_size"]),
        "--learning-rate",
        str(common["learning_rate"]),
        "--weight-decay",
        str(common["weight_decay"]),
        "--patience",
        str(common["patience"]),
        "--minimum-improvement-m",
        str(common["minimum_improvement_m"]),
        "--minimum-learning-rate",
        str(common["minimum_learning_rate"]),
        "--selection-metric",
        str(common["selection_metric"]),
        "--mask-f1-penalty-m",
        str(common["mask_f1_penalty_m"]),
        "--workers",
        str(args.workers),
        "--seed",
        str(common["seed"]),
        "--device",
        str(args.device),
        "--width",
        str(common["width"]),
        "--dilations",
        *[str(value) for value in common["dilations"]],
        "--dropout",
        str(common["dropout"]),
        "--zero-edh-weight",
        str(common["zero_edh_weight"]),
        "--mask-loss-weight",
        str(spec.mask_loss_weight),
        "--smoothness-weight",
        str(spec.smoothness_weight),
        "--path-class-weight",
        str(spec.path_class_weight),
        "--height-class-weight",
        str(spec.height_class_weight),
        "--physics-loss-weight",
        str(common["physics_loss_weight"]),
        "--physics-scale-db",
        str(common["physics_scale_db"]),
    ]
    if args.no_amp:
        command.append("--no-amp")
    if args.smoke_test:
        command.extend(
            [
                "--smoke-test",
                "--smoke-epochs",
                str(args.smoke_epochs),
                "--smoke-train-samples",
                str(args.smoke_train_samples),
                "--smoke-validation-samples",
                str(args.smoke_validation_samples),
            ]
        )
    if args.overwrite:
        command.append("--overwrite")
    return command


def _result_row(
    spec: AblationSpec,
    output_directory: Path,
    state: str,
    summary: dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "experiment_id": spec.experiment_id,
        "display_name": spec.display_name,
        "status": state,
        "output_directory": str(output_directory),
        **_expected_weights(spec),
        "best_epoch": "",
        "validation_rmse_m": "",
        "validation_mae_m": "",
        "positive_point_rmse_m": "",
        "duct_mask_f1": "",
        "error": error or "",
    }
    if summary is not None:
        metrics = summary["validation_metrics"]
        row.update(
            {
                "best_epoch": int(summary["best_epoch"]),
                "validation_rmse_m": float(metrics["rmse_m"]),
                "validation_mae_m": float(metrics["mae_m"]),
                "positive_point_rmse_m": float(metrics["positive_point_rmse_m"]),
                "duct_mask_f1": float(metrics["duct_mask_f1"]),
            }
        )
    return row


def _write_suite_outputs(
    output_root: Path,
    args: argparse.Namespace,
    selected_ids: Sequence[str],
    run_records: list[dict[str, Any]],
    expected_epochs: int,
    started_at: float,
) -> None:
    rows: list[dict[str, Any]] = []
    for spec in ABLATIONS:
        directory = output_root / spec.experiment_id
        state, summary, error = _completion_state(directory, spec, expected_epochs)
        rows.append(_result_row(spec, directory, state, summary, error))
    csv_path = output_root / "ablation_validation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    all_complete = all(row["status"] == "completed" for row in rows)
    payload = {
        "script_version": SCRIPT_VERSION,
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "smoke_test": bool(args.smoke_test),
        "joint_test_loaded": False,
        "selected_experiment_ids": list(selected_ids),
        "all_default_experiments_completed": all_complete,
        "common_train_config": COMMON_TRAIN_CONFIG,
        "ablation_definitions": [asdict(spec) for spec in ABLATIONS],
        "run_records": run_records,
        "current_status": rows,
        "elapsed_seconds": time.perf_counter() - started_at,
        "recommended_next_action": (
            "Run the full suite after smoke tests pass."
            if args.smoke_test
            else (
                "Freeze all three ablation checkpoints, then add them to one "
                "joint_test evaluation without tuning on joint_test."
            )
        ),
    }
    summary_path = output_root / "ablation_suite_summary.json"
    _write_json(summary_path, payload)
    if all_complete:
        marker_name = (
            "ABLATION_SMOKE_SUITE_COMPLETED.json"
            if args.smoke_test
            else "ABLATION_SUITE_COMPLETED.json"
        )
        _write_json(
            output_root / marker_name,
            {
                "status": "complete",
                "script_version": SCRIPT_VERSION,
                "smoke_test": bool(args.smoke_test),
                "joint_test_loaded": False,
                "experiment_ids": [spec.experiment_id for spec in ABLATIONS],
                "summary_file": summary_path.name,
            },
        )


def _preflight(args: argparse.Namespace) -> None:
    required_files = {
        "baseline script": Path(args.baseline_script),
        "forward script": Path(args.forward_script),
        "forward checkpoint": Path(args.forward_checkpoint),
    }
    missing = [f"{label}: {path}" for label, path in required_files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n  " + "\n  ".join(missing))
    if not Path(args.pc_dir).is_dir():
        raise FileNotFoundError(f"Pc dataset directory does not exist: {args.pc_dir}")
    if not Path(args.python_executable).is_file():
        raise FileNotFoundError(f"Python executable does not exist: {args.python_executable}")


def run_suite(args: argparse.Namespace) -> None:
    started_at = time.perf_counter()
    selected_ids = list(args.experiment or [spec.experiment_id for spec in ABLATIONS])
    by_id = {spec.experiment_id: spec for spec in ABLATIONS}
    selected = [by_id[experiment_id] for experiment_id in selected_ids]
    output_root = Path(
        args.output_root
        if args.output_root is not None
        else (DEFAULT_SMOKE_OUTPUT_ROOT if args.smoke_test else DEFAULT_OUTPUT_ROOT)
    )
    expected_epochs = args.smoke_epochs if args.smoke_test else int(COMMON_TRAIN_CONFIG["epochs"])

    if args.dry_run:
        print(f"Script version: {SCRIPT_VERSION}")
        print("joint_test will not be loaded.")
        for spec in selected:
            command = _build_command(args, spec, output_root / spec.experiment_id)
            print(f"\n[{spec.experiment_id}] {spec.description}")
            print(_format_command(command))
        return

    _preflight(args)
    output_root.mkdir(parents=True, exist_ok=True)
    run_records: list[dict[str, Any]] = []
    environment = os.environ.copy()
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    for index, spec in enumerate(selected, start=1):
        output_directory = output_root / spec.experiment_id
        state, summary, error = _completion_state(output_directory, spec, expected_epochs)
        if state == "completed" and not args.overwrite:
            LOGGER.info(
                "[%d/%d] Skipping completed run: %s",
                index,
                len(selected),
                spec.experiment_id,
            )
            run_records.append(
                {
                    "experiment_id": spec.experiment_id,
                    "action": "skipped_completed",
                    "return_code": 0,
                    "output_directory": str(output_directory),
                }
            )
            continue
        if state == "partial_or_mismatched" and not args.overwrite:
            raise RuntimeError(
                f"Refusing to reuse partial/mismatched output {output_directory}: "
                f"{error}\nChoose a clean --output-root or explicitly use --overwrite."
            )
        command = _build_command(args, spec, output_directory)
        LOGGER.info(
            "[%d/%d] Starting %s", index, len(selected), spec.experiment_id
        )
        LOGGER.info("Command: %s", _format_command(command))
        process = subprocess.run(command, env=environment, check=False)
        record = {
            "experiment_id": spec.experiment_id,
            "action": "trained",
            "return_code": int(process.returncode),
            "output_directory": str(output_directory),
        }
        run_records.append(record)
        if process.returncode != 0:
            _write_suite_outputs(
                output_root,
                args,
                selected_ids,
                run_records,
                expected_epochs,
                started_at,
            )
            raise RuntimeError(
                f"Training failed for {spec.experiment_id} with return code "
                f"{process.returncode}. The suite stopped before the next run."
            )
        _validate_summary(output_directory, spec, expected_epochs)
        LOGGER.info("Validated completed run: %s", output_directory)

    _write_suite_outputs(
        output_root,
        args,
        selected_ids,
        run_records,
        expected_epochs,
        started_at,
    )
    LOGGER.info("Ablation suite complete: %s", output_root)
    print(f"Ablation suite complete: {output_root}")


def run_self_test() -> None:
    by_id = {spec.experiment_id: spec for spec in ABLATIONS}
    assert len(by_id) == 3
    assert _expected_weights(by_id["no_mask_loss"])["mask_loss_weight"] == 0.0
    assert _expected_weights(by_id["no_smoothness_loss"])["smoothness_weight"] == 0.0
    aux = _expected_weights(by_id["no_aux_classification_losses"])
    assert aux["path_class_weight"] == 0.0
    assert aux["height_class_weight"] == 0.0
    for spec in ABLATIONS:
        weights = _expected_weights(spec)
        assert weights["physics_loss_weight"] == 0.01
        changed = sum(
            [
                weights["mask_loss_weight"] != 0.25,
                weights["smoothness_weight"] != 0.10,
                weights["path_class_weight"] != 0.05
                or weights["height_class_weight"] != 0.05,
            ]
        )
        assert changed == 1

    with tempfile.TemporaryDirectory(prefix="inverse-ablation-selftest-") as tmp:
        root = Path(tmp)
        for spec in ABLATIONS:
            output = root / spec.experiment_id
            output.mkdir()
            (output / "best_model.pt").write_bytes(b"self-test")
            (output / "training_history.csv").write_text(
                "epoch,train_loss\n1,0.1\n", encoding="utf-8"
            )
            summary = {
                "input_mode": "era5",
                "roles_loaded": ["inverse_train", "inverse_validation"],
                "joint_test_loaded": False,
                "train_config": {
                    "epochs": 2,
                    "seed": COMMON_TRAIN_CONFIG["seed"],
                    **_expected_weights(spec),
                },
                "best_epoch": 1,
                "validation_metrics": {
                    "rmse_m": 1.0,
                    "mae_m": 0.5,
                    "positive_point_rmse_m": 1.1,
                    "duct_mask_f1": 0.9,
                },
                "plot_rendering_passed": True,
            }
            _write_json(output / "training_summary.json", summary)
            validated = _validate_summary(output, spec, expected_epochs=2)
            assert validated["joint_test_loaded"] is False
    print("Ablation-suite self-test passed; no training or dataset was opened.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pc-dir", type=Path, default=DEFAULT_PC_DIR)
    parser.add_argument(
        "--baseline-script",
        type=Path,
        default=PROGRAM_DIR / "train_inverse_baseline_v2.py",
    )
    parser.add_argument(
        "--forward-script",
        type=Path,
        default=PROGRAM_DIR / "train_forward_rescnn.py",
    )
    parser.add_argument(
        "--forward-checkpoint", type=Path, default=DEFAULT_FORWARD_CHECKPOINT
    )
    parser.add_argument(
        "--python-executable", type=Path, default=Path(sys.executable)
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Parent directory for the three child runs. Defaults to "
            "artifacts/inverse_ablation_suite, or the smoke root with --smoke-test."
        ),
    )
    parser.add_argument(
        "--experiment",
        action="append",
        choices=[spec.experiment_id for spec in ABLATIONS],
        help="Run only the selected experiment; may be repeated. Default: all three.",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--smoke-train-samples", type=int, default=512)
    parser.add_argument("--smoke-validation-samples", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
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
    if args.workers < 0:
        raise ValueError("--workers must be nonnegative.")
    if args.smoke_epochs <= 0:
        raise ValueError("--smoke-epochs must be positive.")
    if args.smoke_train_samples <= 0 or args.smoke_validation_samples <= 0:
        raise ValueError("Smoke sample counts must be positive.")
    if args.dry_run and args.overwrite:
        LOGGER.warning("--overwrite has no effect during --dry-run.")
    run_suite(args)


if __name__ == "__main__":
    main()
