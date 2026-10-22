#!/usr/bin/env python3
r"""Train an RFCFormer-style baseline adapted to Pc(r) -> EDH(r).

This is a reproducible *adaptation* of the architecture described in:

    Ji et al., "RFCFormer: A Dual-Stream Transformer Architecture Integrating
    Gramian Angular Field Representations for Retrieving Evaporation Duct
    Refractivity From Radar Sea Clutter," IEEE TGRS, vol. 63, 2025,
    doi: 10.1109/TGRS.2025.3575484.

The original RFCFormer maps a 200-point clutter vector, represented by GASF and
GADF matrices, to four horizontally homogeneous duct parameters.  Our task is
different: it predicts a 101-point range-dependent EDH profile.  Consequently,
this script retains the paper's defining input representation and dual-stream
Transformer settings, while replacing its four-scalar regression layer with
the same EDH, mask, path-class and height-class heads used by our comparison
protocol.

Paper-derived settings retained here:

* 200-point GAF input;
* parallel GASF and GADF streams;
* d_model=64, 16 attention heads and FFN dimension 128;
* one post-norm Transformer encoder block per stream;
* ReLU activation and dropout=0.5;
* batch size 128 and learning rate 1e-3 (provided by the common protocol).

For a fair architecture comparison, all data loading, case-aware role splits,
normalization, weighted sampling, multitask losses, validation metrics,
checkpoint selection and plotting are reused from
``train_inverse_baseline_v2.py``.  Keep that file beside this script.

Recommended checks in the existing ``pytorch`` environment::

    python .\train_inverse_rfcformer_adapted.py --self-test
    python .\train_inverse_rfcformer_adapted.py --smoke-test \
      --output-dir artifacts/inverse_rfcformer_adapted_smoke

Full validation experiment::

    python .\train_inverse_rfcformer_adapted.py \
      --output-dir artifacts/inverse_rfcformer_adapted

The default batch size follows the paper and the other baselines.  If the
200x200 dual GAF attention matrices exceed available GPU memory, use
``--batch-size 64``; this does not change the dataset or model architecture.

Only ``inverse_train`` and ``inverse_validation`` are opened.  ``joint_test``
remains untouched.  Physics loss and fixed-radar inputs are intentionally
forbidden for this supervised comparison model.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from release_paths import DATA_ROOT, MODELS_ROOT, RECORDS_ROOT, OUTPUT_ROOT, SOURCE_ROOT


SCRIPT_VERSION = "comparison-rfcformer-adapted-1.0.0"
CORE_FILENAME = "train_inverse_baseline_v2.py"
DEFAULT_OUTPUT_DIR = (OUTPUT_ROOT / 'inverse_rfcformer_adapted')
DEFAULT_SMOKE_OUTPUT_DIR = (OUTPUT_ROOT / 'inverse_rfcformer_adapted_smoke')


def _load_core():
    core_path = Path(__file__).resolve().with_name(CORE_FILENAME)
    if not core_path.is_file():
        raise FileNotFoundError(
            f"Required common training file is missing: {core_path}. "
            f"Place {CORE_FILENAME} beside this script."
        )
    module_name = "inverse_comparison_core_rfcformer_adapted"
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
    """RFCFormer-style settings stored verbatim in each checkpoint."""

    # ``width`` is d_model.  The first three fields remain compatible with the
    # common v2 command-line/configuration interface.
    width: int = 64
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    dropout: float = 0.5
    architecture: str = "adapted_rfcformer_gasf_gadf_dual_stream"
    gaf_points: int = 200
    attention_heads: int = 16
    encoder_layers: int = 1
    feedforward_dimension: int = 128
    fusion_dimension: int = 128
    gaf_epsilon: float = 1.0e-6
    paper_doi: str = "10.1109/TGRS.2025.3575484"


def _sinusoidal_position_encoding(torch, length: int, dimension: int):
    """Return a fixed Transformer positional encoding of shape 1 x L x D."""
    if length <= 0 or dimension <= 0 or dimension % 2 != 0:
        raise ValueError("Position length must be positive and dimension even.")
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    exponent = torch.arange(0, dimension, 2, dtype=torch.float32)
    divisor = torch.exp(exponent * (-math.log(10000.0) / dimension))
    encoding = torch.zeros((length, dimension), dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * divisor)
    encoding[:, 1::2] = torch.cos(position * divisor)
    return encoding.unsqueeze(0)


def build_inverse_model(config: ModelConfig, edh_scale_m: float):
    """Build the dual GASF/GADF Transformer with range-profile output heads."""
    torch, nn, functional = core._require_torch()
    if config.gaf_points < 2:
        raise ValueError("gaf_points must be at least two.")
    if config.width <= 0 or config.width % config.attention_heads != 0:
        raise ValueError("width must be positive and divisible by attention_heads.")
    if config.encoder_layers <= 0:
        raise ValueError("encoder_layers must be positive.")
    if config.feedforward_dimension <= 0 or config.fusion_dimension <= 0:
        raise ValueError("Feedforward and fusion dimensions must be positive.")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")
    if config.gaf_epsilon <= 0.0:
        raise ValueError("gaf_epsilon must be positive.")

    class GAFTransformerStream(nn.Module):
        def __init__(self):
            super().__init__()
            # Each row of the LxL GAF image is treated as one sequence token.
            self.input_projection = nn.Linear(config.gaf_points, config.width)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.width,
                nhead=config.attention_heads,
                dim_feedforward=config.feedforward_dimension,
                dropout=config.dropout,
                activation="relu",
                batch_first=True,
                norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=config.encoder_layers,
                enable_nested_tensor=False,
            )

        def forward(self, gaf_matrix, position_encoding):
            tokens = self.input_projection(gaf_matrix)
            position = position_encoding.to(
                device=tokens.device, dtype=tokens.dtype
            )
            return self.encoder(tokens + position)

    class AdaptedRFCFormer(nn.Module):
        def __init__(self):
            super().__init__()
            self.edh_scale_m = float(edh_scale_m)
            self.gasf_stream = GAFTransformerStream()
            self.gadf_stream = GAFTransformerStream()
            self.fusion = nn.Sequential(
                nn.Linear(2 * config.width, config.fusion_dimension),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.LayerNorm(config.fusion_dimension),
            )
            hidden = max(16, config.fusion_dimension // 2)
            self.edh_head = nn.Sequential(
                nn.Conv1d(config.fusion_dimension, hidden, 3, padding=1),
                nn.SiLU(),
                nn.Conv1d(hidden, 1, 1),
            )
            self.mask_head = nn.Conv1d(config.fusion_dimension, 1, 1)
            self.path_head = nn.Linear(
                config.fusion_dimension, core.PATH_CLASS_COUNT
            )
            self.height_head = nn.Linear(
                config.fusion_dimension, core.HEIGHT_CLASS_COUNT
            )
            self.register_buffer(
                "input_resampling_matrix",
                core._linear_resampling_matrix(
                    torch, input_points=500, output_points=config.gaf_points
                ),
            )
            self.register_buffer(
                "output_resampling_matrix",
                core._linear_resampling_matrix(
                    torch, input_points=config.gaf_points, output_points=101
                ),
            )
            self.register_buffer(
                "position_encoding",
                _sinusoidal_position_encoding(
                    torch, config.gaf_points, config.width
                ),
            )
            nn.init.constant_(self.edh_head[-1].bias, -2.0)

        def make_gaf(self, pc_normalized):
            """Create standard GASF and GADF matrices from a batch of profiles."""
            if pc_normalized.ndim != 2 or pc_normalized.shape[1] != 500:
                raise ValueError("pc_normalized must have shape batch x 500.")
            # GAF trigonometry is kept in float32 even when the train loop uses
            # AMP.  This prevents round-off near the arccos domain boundaries.
            with torch.autocast(
                device_type=pc_normalized.device.type, enabled=False
            ):
                profile = torch.matmul(
                    pc_normalized.float(),
                    self.input_resampling_matrix.transpose(0, 1).float(),
                )
                minimum = torch.amin(profile, dim=1, keepdim=True)
                maximum = torch.amax(profile, dim=1, keepdim=True)
                span = maximum - minimum
                scaled = (2.0 * profile - maximum - minimum) / torch.clamp(
                    span, min=config.gaf_epsilon
                )
                # A constant input has no angular structure; map it to zero.
                scaled = torch.where(
                    span > config.gaf_epsilon, scaled, torch.zeros_like(scaled)
                )
                cosine = torch.clamp(scaled, min=-1.0, max=1.0)
                sine = torch.sqrt(
                    torch.clamp(
                        1.0 - cosine * cosine, min=config.gaf_epsilon
                    )
                )
                cosine_i = cosine.unsqueeze(2)
                cosine_j = cosine.unsqueeze(1)
                sine_i = sine.unsqueeze(2)
                sine_j = sine.unsqueeze(1)
                # cos(phi_i + phi_j), paper Eq. (29).
                gasf = cosine_i * cosine_j - sine_i * sine_j
                # sin(phi_i - phi_j), paper Eq. (30).
                gadf = sine_i * cosine_j - cosine_i * sine_j
            return gasf, gadf

        def forward(self, pc_normalized):
            gasf, gadf = self.make_gaf(pc_normalized)
            position = self.position_encoding.to(dtype=gasf.dtype)
            gasf_tokens = self.gasf_stream(gasf, position)
            gadf_tokens = self.gadf_stream(gadf, position)
            fused = self.fusion(torch.cat((gasf_tokens, gadf_tokens), dim=2))
            pooled = torch.mean(fused, dim=1)
            features = fused.transpose(1, 2)
            features_101 = torch.matmul(
                features, self.output_resampling_matrix.transpose(0, 1)
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

    return AdaptedRFCFormer()


def run_self_test() -> None:
    """Check GAF identities, tensor interfaces and differentiability."""
    torch, _, _ = core._require_torch()
    torch.manual_seed(20260814)
    test_config = ModelConfig(
        width=16,
        attention_heads=4,
        feedforward_dimension=32,
        fusion_dimension=32,
        gaf_points=40,
        dropout=0.0,
    )
    model = build_inverse_model(test_config, 30.0)
    test_input = torch.randn((3, 500), dtype=torch.float32, requires_grad=True)
    gasf, gadf = model.make_gaf(test_input)
    assert gasf.shape == (3, 40, 40)
    assert gadf.shape == (3, 40, 40)
    assert torch.allclose(gasf, gasf.transpose(1, 2), atol=1.0e-5)
    assert torch.allclose(gadf, -gadf.transpose(1, 2), atol=1.0e-5)
    diagonal = torch.diagonal(gadf, dim1=1, dim2=2)
    assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1.0e-6)
    output = model(test_input)
    assert output["edh_m"].shape == (3, 101)
    assert output["mask_logits"].shape == (3, 101)
    assert output["path_logits"].shape == (3, core.PATH_CLASS_COUNT)
    assert output["height_logits"].shape == (3, core.HEIGHT_CLASS_COUNT)
    assert torch.isfinite(output["edh_m"]).all()
    output["edh_m"].mean().backward()
    assert test_input.grad is not None and torch.isfinite(test_input.grad).all()
    print("Adapted RFCFormer self-test passed.")


_original_run_training = core.run_training
_original_build_parser = core.build_parser


def _run_supervised_only(args) -> None:
    if args.physics_loss_weight != 0.0:
        raise ValueError(
            "The adapted RFCFormer comparison requires --physics-loss-weight 0. "
            "Physics guidance is reserved for the proposed ResCNN model."
        )
    if args.input_mode != "era5":
        raise ValueError(
            "The paper comparison must use the ERA5-derived Pc dataset; "
            "--input-mode fixed_radar is a controlled ablation, not deployment data."
        )
    _original_run_training(args)


def build_parser():
    parser = _original_build_parser()
    parser.description = __doc__
    # The paper specifies dropout=0.5; the common core otherwise defaults to
    # 0.08 for the 1-D baselines.
    parser.set_defaults(dropout=0.5)
    return parser


core.SCRIPT_VERSION = SCRIPT_VERSION
core.ModelConfig = ModelConfig
core.build_inverse_model = build_inverse_model
core.run_self_test = run_self_test
core.run_training = _run_supervised_only
core.build_parser = build_parser
core.OUTPUT_DIR = DEFAULT_OUTPUT_DIR
core.SMOKE_OUTPUT_DIR = DEFAULT_SMOKE_OUTPUT_DIR


if __name__ == "__main__":
    core.main()
