#!/usr/bin/env python3
r"""Run a bounded validation-only fairness tuning of RFCFormer-adapted.

This file is the controlled follow-up to ``train_inverse_rfcformer_adapted.py``.
It does not redesign RFCFormer and it never reads ``joint_test``.  Instead, it
provides two predeclared training presets selected before looking at final-test
results:

``low_dropout`` (recommended first run)
    dropout=0.10, learning rate=3e-4, epochs=300, patience=50.

``moderate_dropout`` (recommended second run)
    dropout=0.20, learning rate=5e-4, epochs=300, patience=50.

All other elements remain identical to the existing adapted baseline: the
200-point GASF/GADF representation, parallel Transformer streams, d_model=64,
16 attention heads, FFN dimension 128, range-profile output heads, case-aware
role split, normalization, sampler, losses, validation metrics and checkpoint
selection.  Physics loss and fixed-radar input are forbidden for this
comparison.

Keep these files in the same directory:

* train_inverse_baseline_v2.py
* train_inverse_rfcformer_adapted.py
* train_inverse_rfcformer_adapted_v2.py

Run the checks once::

    python .\train_inverse_rfcformer_adapted_v2.py --self-test
    python .\train_inverse_rfcformer_adapted_v2.py --smoke-test \
      --fairness-preset low_dropout \
      --output-dir artifacts/inverse_rfcformer_adapted_v2_smoke

Then run exactly the two validation experiments::

    python .\train_inverse_rfcformer_adapted_v2.py \
      --fairness-preset low_dropout \
      --output-dir artifacts/inverse_rfcformer_adapted_v2_low_dropout

    python .\train_inverse_rfcformer_adapted_v2.py \
      --fairness-preset moderate_dropout \
      --output-dir artifacts/inverse_rfcformer_adapted_v2_moderate_dropout

Choose the checkpoint only from ``inverse_validation``.  Do not add more
presets after viewing ``joint_test``.  If neither preset improves the original
validation RMSE of 2.57043 m, retain the original adapted baseline.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from release_paths import DATA_ROOT, MODELS_ROOT, RECORDS_ROOT, OUTPUT_ROOT, SOURCE_ROOT


SCRIPT_VERSION = "comparison-rfcformer-adapted-2.0.0"
BASE_FILENAME = "train_inverse_rfcformer_adapted.py"
ORIGINAL_DEFAULT_OUTPUT = (OUTPUT_ROOT / 'inverse_rfcformer_adapted')
DEFAULT_OUTPUTS = {
    "low_dropout": (MODELS_ROOT / 'inverse_rfcformer_adapted_v2_low_dropout'),
    "moderate_dropout": Path(
        r"artifacts/inverse_rfcformer_adapted_v2_moderate_dropout"
    ),
}
PRESETS = {
    "low_dropout": {
        "dropout": 0.10,
        "learning_rate": 3.0e-4,
        "epochs": 300,
        "patience": 50,
    },
    "moderate_dropout": {
        "dropout": 0.20,
        "learning_rate": 5.0e-4,
        "epochs": 300,
        "patience": 50,
    },
}


def _load_base():
    base_path = Path(__file__).resolve().with_name(BASE_FILENAME)
    if not base_path.is_file():
        raise FileNotFoundError(
            f"Required RFCFormer file is missing: {base_path}. "
            f"Place {BASE_FILENAME} beside this script."
        )
    module_name = "inverse_rfcformer_adapted_v2_base"
    specification = importlib.util.spec_from_file_location(module_name, base_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import RFCFormer file: {base_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


base = _load_base()
core = base.core
LOGGER = logging.getLogger(__name__)

_base_run_supervised_only = base._run_supervised_only
_base_build_parser = base.build_parser


def _write_tuning_metadata(output_directory: Path, preset_name: str) -> None:
    """Append the predeclared fairness protocol to reproducibility JSON files."""
    protocol = {
        "protocol_version": "rfcformer-fairness-tuning-1.0",
        "preset": preset_name,
        "preset_values": PRESETS[preset_name],
        "selection_data": "inverse_validation only",
        "joint_test_loaded": False,
        "maximum_additional_full_runs": 2,
        "original_validation_rmse_m": 2.5704295069870415,
        "decision_rule": (
            "Select the lowest validation RMSE among the original adapted "
            "baseline and the two predeclared presets; otherwise retain the "
            "original baseline."
        ),
    }
    for filename in ("run_config.json", "training_summary.json"):
        path = output_directory / filename
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        payload["rfcformer_fairness_tuning"] = protocol
        core._write_json(path, payload)


def _run_fairness_preset(args) -> None:
    preset_name = args.fairness_preset
    preset = PRESETS[preset_name]

    if args.physics_loss_weight != 0.0:
        raise ValueError(
            "RFCFormer fairness tuning requires --physics-loss-weight 0."
        )
    if args.input_mode != "era5":
        raise ValueError(
            "RFCFormer fairness tuning requires --input-mode era5."
        )

    # These four fields are deliberately owned by the selected preset.  This
    # prevents accidental post-hoc mixtures that would invalidate the bounded
    # two-run protocol.
    args.dropout = preset["dropout"]
    args.learning_rate = preset["learning_rate"]
    args.epochs = preset["epochs"]
    args.patience = preset["patience"]

    if Path(args.output_dir) in {
        ORIGINAL_DEFAULT_OUTPUT,
        DEFAULT_OUTPUTS["low_dropout"],
    }:
        args.output_dir = DEFAULT_OUTPUTS[preset_name]

    LOGGER.info(
        "RFCFormer fairness preset=%s | dropout=%.2f | lr=%.1e | "
        "epochs=%d | patience=%d",
        preset_name,
        args.dropout,
        args.learning_rate,
        args.epochs,
        args.patience,
    )
    LOGGER.info(
        "Selection roles are locked to inverse_train + inverse_validation; "
        "joint_test is not loaded."
    )

    _base_run_supervised_only(args)
    _write_tuning_metadata(Path(args.output_dir), preset_name)
    LOGGER.info(
        "Fairness metadata written. Compare validation metrics only before "
        "freezing the RFCFormer-adapted checkpoint."
    )


def build_parser():
    parser = _base_build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--fairness-preset",
        choices=tuple(PRESETS),
        default="low_dropout",
        help=(
            "Predeclared validation-only RFCFormer tuning configuration. "
            "The preset overrides dropout, learning rate, epochs and patience."
        ),
    )
    return parser


core.SCRIPT_VERSION = SCRIPT_VERSION
core.build_parser = build_parser
core.run_training = _run_fairness_preset
core.OUTPUT_DIR = DEFAULT_OUTPUTS["low_dropout"]
core.SMOKE_OUTPUT_DIR = (OUTPUT_ROOT / 'inverse_rfcformer_adapted_v2_smoke')


if __name__ == "__main__":
    core.main()
