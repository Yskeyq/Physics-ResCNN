#!/usr/bin/env python3
r"""Train the BiLSTM comparison model for Pc(r) -> EDH(r).

This script reuses the complete experiment protocol in
``train_inverse_baseline_v2.py`` and replaces only the inverse backbone.  Keep
both files in the same directory.

Smoke test::

    python .\train_inverse_bilstm.py --smoke-test --output-dir artifacts/inverse_bilstm_smoke

Full comparison run::

    python .\train_inverse_bilstm.py --output-dir artifacts/inverse_bilstm

Only ``inverse_train`` and ``inverse_validation`` are opened.  ``joint_test``
remains untouched.  Physics loss is intentionally disabled for this supervised
architecture comparison.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from release_paths import DATA_ROOT, MODELS_ROOT, RECORDS_ROOT, OUTPUT_ROOT, SOURCE_ROOT


SCRIPT_VERSION = "comparison-bilstm-1.0.0"
CORE_FILENAME = "train_inverse_baseline_v2.py"


def _load_core():
    core_path = Path(__file__).resolve().with_name(CORE_FILENAME)
    if not core_path.is_file():
        raise FileNotFoundError(
            f"Required common training file is missing: {core_path}. "
            f"Place {CORE_FILENAME} beside this script."
        )
    module_name = "inverse_comparison_core_bilstm"
    specification = importlib.util.spec_from_file_location(module_name, core_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import common training file: {core_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


core = _load_core()


@dataclass(frozen=True)
class ModelConfig:
    """BiLSTM configuration recorded verbatim in every checkpoint."""

    width: int = 64
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    dropout: float = 0.08
    architecture: str = "bidirectional_lstm"
    hidden_size: int = 48
    recurrent_layers: int = 2


def build_inverse_model(config: ModelConfig, edh_scale_m: float):
    """Build a two-layer bidirectional LSTM with the common output heads."""
    torch, nn, functional = core._require_torch()
    if config.hidden_size <= 0 or config.recurrent_layers <= 0:
        raise ValueError("hidden_size and recurrent_layers must be positive.")

    class BiLSTMInverse(nn.Module):
        def __init__(self):
            super().__init__()
            self.edh_scale_m = float(edh_scale_m)
            self.encoder = nn.LSTM(
                input_size=2,
                hidden_size=config.hidden_size,
                num_layers=config.recurrent_layers,
                batch_first=True,
                bidirectional=True,
                dropout=config.dropout if config.recurrent_layers > 1 else 0.0,
            )
            feature_width = 2 * config.hidden_size
            self.final_norm = nn.LayerNorm(feature_width)
            hidden = max(16, feature_width // 3)
            self.edh_head = nn.Sequential(
                nn.Conv1d(feature_width, hidden, 3, padding=1),
                nn.SiLU(),
                nn.Conv1d(hidden, 1, 1),
            )
            self.mask_head = nn.Conv1d(feature_width, 1, 1)
            self.path_head = nn.Linear(feature_width, core.PATH_CLASS_COUNT)
            self.height_head = nn.Linear(feature_width, core.HEIGHT_CLASS_COUNT)
            coordinate = torch.linspace(0.0, 1.0, 500, dtype=torch.float32)
            self.register_buffer("coordinate", coordinate.view(1, 500, 1))
            self.register_buffer(
                "resampling_matrix",
                core._linear_resampling_matrix(
                    torch, input_points=500, output_points=101
                ),
            )
            nn.init.constant_(self.edh_head[-1].bias, -2.0)

        def forward(self, pc_normalized):
            if pc_normalized.ndim != 2 or pc_normalized.shape[1] != 500:
                raise ValueError("pc_normalized must have shape batch x 500.")
            coordinate = self.coordinate.expand(pc_normalized.shape[0], -1, -1)
            sequence = torch.cat((pc_normalized.unsqueeze(2), coordinate), dim=2)
            encoded, _ = self.encoder(sequence)
            encoded = self.final_norm(encoded)
            pooled = torch.mean(encoded, dim=1)
            features = encoded.transpose(1, 2)
            features_101 = torch.matmul(
                features, self.resampling_matrix.transpose(0, 1)
            )
            edh_m = (
                functional.softplus(self.edh_head(features_101)[:, 0, :])
                * self.edh_scale_m
            )
            return {
                "edh_m": edh_m,
                "mask_logits": self.mask_head(features_101)[:, 0, :],
                "path_logits": self.path_head(pooled),
                "height_logits": self.height_head(pooled),
            }

    return BiLSTMInverse()


def run_self_test() -> None:
    torch, _, _ = core._require_torch()
    model = build_inverse_model(
        ModelConfig(hidden_size=12, recurrent_layers=1, dropout=0.0), 30.0
    )
    test_input = torch.zeros((4, 500), dtype=torch.float32, requires_grad=True)
    output = model(test_input)
    assert output["edh_m"].shape == (4, 101)
    assert output["mask_logits"].shape == (4, 101)
    assert output["path_logits"].shape == (4, core.PATH_CLASS_COUNT)
    assert output["height_logits"].shape == (4, core.HEIGHT_CLASS_COUNT)
    assert torch.isfinite(output["edh_m"]).all()
    output["edh_m"].mean().backward()
    assert test_input.grad is not None and torch.isfinite(test_input.grad).all()
    print("BiLSTM self-test passed.")


_original_run_training = core.run_training


def _run_supervised_only(args) -> None:
    if args.physics_loss_weight != 0.0:
        raise ValueError(
            "The BiLSTM paper comparison must use --physics-loss-weight 0. "
            "Physics guidance is reserved for the proposed ResCNN model."
        )
    _original_run_training(args)


core.SCRIPT_VERSION = SCRIPT_VERSION
core.ModelConfig = ModelConfig
core.build_inverse_model = build_inverse_model
core.run_self_test = run_self_test
core.run_training = _run_supervised_only
core.OUTPUT_DIR = (OUTPUT_ROOT / 'inverse_bilstm')
core.SMOKE_OUTPUT_DIR = (OUTPUT_ROOT / 'inverse_bilstm_smoke')


if __name__ == "__main__":
    core.main()
