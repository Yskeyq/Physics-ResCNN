#!/usr/bin/env python3
r"""Train the Transformer-Encoder comparison model for Pc(r) -> EDH(r).

This script reuses the complete experiment protocol in
``train_inverse_baseline_v2.py`` and replaces only the inverse backbone.  Keep
both files in the same directory.  A stride-4 patch embedding reduces the
500-point sequence to 125 tokens so the baseline is practical on the same GPU.

Smoke test::

    python .\train_inverse_transformer.py --smoke-test --output-dir artifacts/inverse_transformer_smoke

Full comparison run::

    python .\train_inverse_transformer.py --output-dir artifacts/inverse_transformer

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


SCRIPT_VERSION = "comparison-transformer-1.0.0"
CORE_FILENAME = "train_inverse_baseline_v2.py"


def _load_core():
    core_path = Path(__file__).resolve().with_name(CORE_FILENAME)
    if not core_path.is_file():
        raise FileNotFoundError(
            f"Required common training file is missing: {core_path}. "
            f"Place {CORE_FILENAME} beside this script."
        )
    module_name = "inverse_comparison_core_transformer"
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
    """Transformer configuration recorded verbatim in every checkpoint."""

    width: int = 64
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    dropout: float = 0.08
    architecture: str = "transformer_encoder"
    model_dimension: int = 48
    attention_heads: int = 4
    encoder_layers: int = 4
    feedforward_dimension: int = 96
    patch_size: int = 4


def build_inverse_model(config: ModelConfig, edh_scale_m: float):
    """Build a vanilla patch-based Transformer Encoder with common heads."""
    torch, nn, functional = core._require_torch()
    if config.model_dimension % config.attention_heads != 0:
        raise ValueError("model_dimension must be divisible by attention_heads.")
    if 500 % config.patch_size != 0:
        raise ValueError("patch_size must divide the 500 input points exactly.")
    token_count = 500 // config.patch_size

    class TransformerInverse(nn.Module):
        def __init__(self):
            super().__init__()
            self.edh_scale_m = float(edh_scale_m)
            self.patch_embedding = nn.Conv1d(
                2,
                config.model_dimension,
                kernel_size=config.patch_size,
                stride=config.patch_size,
            )
            self.position_embedding = nn.Parameter(
                torch.zeros(1, token_count, config.model_dimension)
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.model_dimension,
                nhead=config.attention_heads,
                dim_feedforward=config.feedforward_dimension,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=config.encoder_layers,
                norm=nn.LayerNorm(config.model_dimension),
                enable_nested_tensor=False,
            )
            hidden = max(16, config.model_dimension // 2)
            self.edh_head = nn.Sequential(
                nn.Conv1d(config.model_dimension, hidden, 3, padding=1),
                nn.SiLU(),
                nn.Conv1d(hidden, 1, 1),
            )
            self.mask_head = nn.Conv1d(config.model_dimension, 1, 1)
            self.path_head = nn.Linear(
                config.model_dimension, core.PATH_CLASS_COUNT
            )
            self.height_head = nn.Linear(
                config.model_dimension, core.HEIGHT_CLASS_COUNT
            )
            coordinate = torch.linspace(0.0, 1.0, 500, dtype=torch.float32)
            self.register_buffer("coordinate", coordinate.view(1, 1, 500))
            self.register_buffer(
                "resampling_matrix",
                core._linear_resampling_matrix(
                    torch, input_points=token_count, output_points=101
                ),
            )
            nn.init.trunc_normal_(self.position_embedding, std=0.02)
            nn.init.constant_(self.edh_head[-1].bias, -2.0)

        def forward(self, pc_normalized):
            if pc_normalized.ndim != 2 or pc_normalized.shape[1] != 500:
                raise ValueError("pc_normalized must have shape batch x 500.")
            coordinate = self.coordinate.expand(pc_normalized.shape[0], -1, -1)
            x = torch.cat((pc_normalized.unsqueeze(1), coordinate), dim=1)
            tokens = self.patch_embedding(x).transpose(1, 2)
            tokens = self.encoder(tokens + self.position_embedding)
            pooled = torch.mean(tokens, dim=1)
            features = tokens.transpose(1, 2)
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

    return TransformerInverse()


def run_self_test() -> None:
    torch, _, _ = core._require_torch()
    model = build_inverse_model(
        ModelConfig(
            model_dimension=16,
            attention_heads=4,
            encoder_layers=1,
            feedforward_dimension=32,
            dropout=0.0,
        ),
        30.0,
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
    print("Transformer Encoder self-test passed.")


_original_run_training = core.run_training


def _run_supervised_only(args) -> None:
    if args.physics_loss_weight != 0.0:
        raise ValueError(
            "The Transformer paper comparison must use --physics-loss-weight 0. "
            "Physics guidance is reserved for the proposed ResCNN model."
        )
    _original_run_training(args)


core.SCRIPT_VERSION = SCRIPT_VERSION
core.ModelConfig = ModelConfig
core.build_inverse_model = build_inverse_model
core.run_self_test = run_self_test
core.run_training = _run_supervised_only
core.OUTPUT_DIR = (OUTPUT_ROOT / 'inverse_transformer')
core.SMOKE_OUTPUT_DIR = (OUTPUT_ROOT / 'inverse_transformer_smoke')


if __name__ == "__main__":
    core.main()
