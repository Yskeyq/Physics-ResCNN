#!/usr/bin/env python3
r"""Train the plain 1D-CNN comparison model for Pc(r) -> EDH(r).

This script deliberately reuses every data, loss, sampling, checkpoint-selection,
metric and plotting routine from ``train_inverse_baseline_v2.py``.  Only the
inverse backbone is replaced by a conventional non-residual, non-dilated CNN.
Keep both files in the same directory.

Smoke test::

    python .\train_inverse_cnn.py --smoke-test --output-dir artifacts/inverse_cnn_smoke

Full comparison run::

    python .\train_inverse_cnn.py --output-dir artifacts/inverse_cnn

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


SCRIPT_VERSION = "comparison-cnn-1.0.0"
CORE_FILENAME = "train_inverse_baseline_v2.py"


def _load_core():
    core_path = Path(__file__).resolve().with_name(CORE_FILENAME)
    if not core_path.is_file():
        raise FileNotFoundError(
            f"Required common training file is missing: {core_path}. "
            f"Place {CORE_FILENAME} beside this script."
        )
    module_name = "inverse_comparison_core_cnn"
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
    """Plain-CNN configuration recorded verbatim in every checkpoint."""

    width: int = 64
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    dropout: float = 0.08
    architecture: str = "plain_1d_cnn"
    layer_count: int = 6


def build_inverse_model(config: ModelConfig, edh_scale_m: float):
    """Build a local-receptive-field CNN without residual or dilated blocks."""
    torch, nn, functional = core._require_torch()
    if config.kernel_size % 2 != 1:
        raise ValueError("kernel_size must be odd.")
    if config.layer_count <= 0:
        raise ValueError("layer_count must be positive.")

    class PlainInverseCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.edh_scale_m = float(edh_scale_m)
            padding = (config.kernel_size - 1) // 2
            self.stem = nn.Conv1d(2, config.width, kernel_size=7, padding=3)
            groups = 8 if config.width % 8 == 0 else 1
            layers = []
            for _ in range(config.layer_count):
                layers.extend(
                    (
                        nn.GroupNorm(groups, config.width),
                        nn.SiLU(),
                        nn.Conv1d(
                            config.width,
                            config.width,
                            config.kernel_size,
                            padding=padding,
                        ),
                        nn.Dropout(config.dropout),
                    )
                )
            self.encoder = nn.Sequential(*layers)
            self.final_norm = nn.GroupNorm(groups, config.width)
            hidden = max(16, config.width // 2)
            self.edh_head = nn.Sequential(
                nn.Conv1d(config.width, hidden, 3, padding=1),
                nn.SiLU(),
                nn.Conv1d(hidden, 1, 1),
            )
            self.mask_head = nn.Conv1d(config.width, 1, 1)
            self.path_head = nn.Linear(config.width, core.PATH_CLASS_COUNT)
            self.height_head = nn.Linear(config.width, core.HEIGHT_CLASS_COUNT)
            coordinate = torch.linspace(0.0, 1.0, 500, dtype=torch.float32)
            self.register_buffer("coordinate", coordinate.view(1, 1, 500))
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
            x = torch.cat((pc_normalized.unsqueeze(1), coordinate), dim=1)
            x = self.stem(x)
            x = self.encoder(x)
            x = functional.silu(self.final_norm(x))
            pooled = torch.mean(x, dim=2)
            x_101 = torch.matmul(x, self.resampling_matrix.transpose(0, 1))
            edh_m = (
                functional.softplus(self.edh_head(x_101)[:, 0, :])
                * self.edh_scale_m
            )
            return {
                "edh_m": edh_m,
                "mask_logits": self.mask_head(x_101)[:, 0, :],
                "path_logits": self.path_head(pooled),
                "height_logits": self.height_head(pooled),
            }

    return PlainInverseCNN()


def run_self_test() -> None:
    torch, _, _ = core._require_torch()
    model = build_inverse_model(ModelConfig(width=16, layer_count=2), 30.0)
    test_input = torch.zeros((4, 500), dtype=torch.float32, requires_grad=True)
    output = model(test_input)
    assert output["edh_m"].shape == (4, 101)
    assert output["mask_logits"].shape == (4, 101)
    assert output["path_logits"].shape == (4, core.PATH_CLASS_COUNT)
    assert output["height_logits"].shape == (4, core.HEIGHT_CLASS_COUNT)
    assert torch.isfinite(output["edh_m"]).all()
    output["edh_m"].mean().backward()
    assert test_input.grad is not None and torch.isfinite(test_input.grad).all()
    print("Plain 1D-CNN self-test passed.")


_original_run_training = core.run_training


def _run_supervised_only(args) -> None:
    if args.physics_loss_weight != 0.0:
        raise ValueError(
            "The CNN paper comparison must use --physics-loss-weight 0. "
            "Physics guidance is reserved for the proposed ResCNN model."
        )
    _original_run_training(args)


core.SCRIPT_VERSION = SCRIPT_VERSION
core.ModelConfig = ModelConfig
core.build_inverse_model = build_inverse_model
core.run_self_test = run_self_test
core.run_training = _run_supervised_only
core.OUTPUT_DIR = (OUTPUT_ROOT / 'inverse_cnn')
core.SMOKE_OUTPUT_DIR = (OUTPUT_ROOT / 'inverse_cnn_smoke')


if __name__ == "__main__":
    core.main()
