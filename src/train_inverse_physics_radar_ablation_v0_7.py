#!/usr/bin/env python3
r"""Train the first category-aware inverse baseline Pc(r) -> EDH(r).

The deployed inverse network receives only the centered clutter sequence. ERA5
variables and ``radar_term_dB`` are never network inputs.  The latter is used
only for optional training-time forward consistency.

Default full supervised baseline::

    python train_inverse_baseline_v2.py

Recommended first check::

    python train_inverse_baseline_v2.py --smoke-test

Controlled fixed-sea-state experiment (no PE rerun required)::

    python train_inverse_baseline_v2.py --input-mode fixed_radar \
      --output-dir artifacts/inverse_baseline_v2_fixed_radar

After the supervised architecture is frozen, test a small forward-consistency
weight as an ablation::

    python train_inverse_baseline_v2.py --physics-loss-weight 0.02 \
      --output-dir artifacts/inverse_baseline_v2_physics

Only ``inverse_train`` and ``inverse_validation`` are opened. ``joint_test``
remains untouched.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from release_paths import DATA_ROOT, MODELS_ROOT, RECORDS_ROOT, OUTPUT_ROOT, SOURCE_ROOT
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("train_inverse_baseline")
# Must be set before PyTorch creates a CUDA context. Together with the fixed
# resampling matrix below, this removes the two deterministic-algorithm warnings
# emitted by the original implementation.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SCRIPT_VERSION = "1.3.0-v0.7-fixed-physics-radar"
PROGRAM_DIR = SOURCE_ROOT
PC_DIR = (DATA_ROOT / 'pc_dataset_v1')
FORWARD_DIR = (MODELS_ROOT / 'forward_rescnn')
OUTPUT_DIR = (OUTPUT_ROOT / 'inverse_baseline_v2')
SMOKE_OUTPUT_DIR = (OUTPUT_ROOT / 'inverse_baseline_v2_smoke')
ROLES = ("inverse_train", "inverse_validation")
PATH_CLASS_COUNT = 3
HEIGHT_CLASS_COUNT = 4


@dataclass(frozen=True)
class ModelConfig:
    width: int = 64
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    dropout: float = 0.08


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 180
    batch_size: int = 128
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    patience: int = 30
    minimum_improvement_m: float = 1.0e-3
    scheduler_name: str = "cosine_annealing"
    minimum_learning_rate: float = 1.0e-5
    selection_metric: str = "positive_point_rmse_m"
    mask_f1_penalty_m: float = 1.0
    workers: int = 0
    seed: int = 20260813
    use_amp: bool = True
    zero_edh_weight: float = 0.25
    mask_loss_weight: float = 0.25
    smoothness_weight: float = 0.10
    path_class_weight: float = 0.05
    height_class_weight: float = 0.05
    physics_loss_weight: float = 0.0
    physics_scale_db: float = 5.0


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("PyTorch is required in the existing pytorch environment.") from exc
    return torch, nn, functional


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required in the existing pytorch environment.") from exc
    return h5py


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch, _, _ = _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_forward_module(script_path: Path):
    specification = importlib.util.spec_from_file_location(
        "frozen_forward_rescnn_definition", script_path
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import forward script: {script_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _load_torch_checkpoint(path: Path, device):
    torch, _, _ = _require_torch()
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_frozen_forward(script_path: Path, checkpoint_path: Path, device):
    if not script_path.is_file():
        raise FileNotFoundError(script_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    module = _load_forward_module(script_path)
    checkpoint = _load_torch_checkpoint(checkpoint_path, device)
    required = {
        "model_config",
        "edh_scale_m",
        "target_mean_db",
        "target_std_db",
        "model_state_dict",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Forward checkpoint is missing: {missing}")
    configuration = module.ModelConfig(**dict(checkpoint["model_config"]))
    model = module.build_model(
        configuration, float(checkpoint["edh_scale_m"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    torch, _, _ = _require_torch()
    mean = torch.as_tensor(
        np.asarray(checkpoint["target_mean_db"], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    std = torch.as_tensor(
        np.asarray(checkpoint["target_std_db"], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    return model, mean, std, checkpoint


def _h5_path(pc_directory: Path, role: str) -> Path:
    return pc_directory / f"pc_{role}.h5"


def _metadata_path(pc_directory: Path, role: str) -> Path:
    return pc_directory / f"pc_{role}_metadata.csv"


def _load_metadata(path: Path, role: str, sample_count: int) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "pc_dataset_row_0based",
        "sample_id",
        "pe_row_index_0based",
        "role_split_name",
        "path_class_code",
        "path_class_name",
        "height_relation_class_code",
        "height_relation_class_name",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    if len(table) != sample_count:
        raise ValueError(f"{path.name}: expected {sample_count} rows, got {len(table)}.")
    if not np.array_equal(
        table["pc_dataset_row_0based"].to_numpy(np.int64),
        np.arange(sample_count, dtype=np.int64),
    ):
        raise ValueError(f"{path.name}: row order is not 0..N-1.")
    if not table["role_split_name"].astype(str).eq(role).all():
        raise ValueError(f"{path.name}: a non-{role} row is present.")
    path_code = table["path_class_code"].to_numpy(np.int64)
    height_code = table["height_relation_class_code"].to_numpy(np.int64)
    if np.any(path_code < 0) or np.any(path_code >= PATH_CLASS_COUNT):
        raise ValueError("path_class_code must be 0, 1 or 2.")
    if np.any(height_code < 0) or np.any(height_code >= HEIGHT_CLASS_COUNT):
        raise ValueError("height_relation_class_code must be 0..3.")
    return table


def load_role_arrays(pc_directory: Path, role: str, need_radar: bool) -> dict[str, Any]:
    h5py = _require_h5py()
    path = _h5_path(pc_directory, role)
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        if str(handle.attrs.get("role", "")) != role:
            raise ValueError(f"{path.name}: incorrect role attribute.")
        if not bool(handle.attrs.get("complete", False)):
            raise ValueError(f"{path.name}: file is incomplete.")
        required = {
            "pc_centered_dB",
            "edh_m",
            "duct_mask",
            "sample_id",
            "pe_row_index_0based",
            "range_km",
            "input_range_km",
        }
        if need_radar:
            required.update(("ls_oneway_dB", "radar_term_dB"))
        missing = sorted(required - set(handle.keys()))
        if missing:
            raise ValueError(f"{path.name} is missing datasets: {missing}")
        pc = np.asarray(handle["pc_centered_dB"], dtype=np.float32)
        edh = np.asarray(handle["edh_m"], dtype=np.float32)
        mask = np.asarray(handle["duct_mask"], dtype=np.float32)
        sample_id = np.asarray(handle["sample_id"], dtype=np.int64)
        pe_row = np.asarray(handle["pe_row_index_0based"], dtype=np.int64)
        range_km = np.asarray(handle["range_km"], dtype=np.float32)
        input_range_km = np.asarray(handle["input_range_km"], dtype=np.float32)
        radar = (
            np.asarray(handle["radar_term_dB"], dtype=np.float32)
            if need_radar
            else None
        )
        loss = (
            np.asarray(handle["ls_oneway_dB"], dtype=np.float32)
            if need_radar
            else None
        )
    sample_count = pc.shape[0]
    if pc.shape != (sample_count, 500):
        raise ValueError(f"{path.name}: Pc must have shape N x 500.")
    if edh.shape != (sample_count, 101) or mask.shape != (sample_count, 101):
        raise ValueError(f"{path.name}: EDH/mask must have shape N x 101.")
    if not np.isfinite(pc).all() or not np.isfinite(edh).all():
        raise ValueError(f"{path.name}: non-finite Pc or EDH value.")
    if np.any(edh < 0.0) or not np.isin(mask, (0.0, 1.0)).all():
        raise ValueError(f"{path.name}: invalid EDH or duct mask.")
    metadata = _load_metadata(_metadata_path(pc_directory, role), role, sample_count)
    if not np.array_equal(metadata["sample_id"].to_numpy(np.int64), sample_id):
        raise ValueError(f"{role}: metadata/H5 sample_id order differs.")
    if not np.array_equal(
        metadata["pe_row_index_0based"].to_numpy(np.int64), pe_row
    ):
        raise ValueError(f"{role}: metadata/H5 PE row order differs.")
    return {
        "pc": pc,
        "edh": edh,
        "mask": mask,
        "radar": radar,
        "loss": loss,
        "sample_id": sample_id,
        "range_km": range_km,
        "input_range_km": input_range_km,
        "path_code": metadata["path_class_code"].to_numpy(np.int64),
        "height_code": metadata["height_relation_class_code"].to_numpy(np.int64),
        "metadata": metadata,
    }


def _center_profiles(values: np.ndarray) -> np.ndarray:
    return np.asarray(values - np.mean(values, axis=1, keepdims=True), dtype=np.float32)


def prepare_inputs(
    train: dict[str, Any],
    validation: dict[str, Any],
    input_mode: str,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if input_mode == "era5":
        train_input = train["pc"]
        validation_input = validation["pc"]
        radar_reference = None
    elif input_mode == "fixed_radar":
        if train["radar"] is None or train["loss"] is None:
            raise ValueError("fixed_radar requires radar_term_dB and ls_oneway_dB.")
        radar_reference = np.mean(train["radar"], axis=0, dtype=np.float64).astype(
            np.float32
        )
        train_input = _center_profiles(-2.0 * train["loss"] + radar_reference[None, :])
        validation_input = _center_profiles(
            -2.0 * validation["loss"] + radar_reference[None, :]
        )
    else:
        raise ValueError(f"Unknown input mode: {input_mode}")
    return train_input, validation_input, radar_reference


def _linear_resampling_matrix(torch, input_points: int, output_points: int):
    """Return the align-corners linear interpolation operator as a constant matrix."""
    if input_points < 2 or output_points < 2:
        raise ValueError("Linear resampling requires at least two input/output points.")
    positions = torch.linspace(
        0.0, float(input_points - 1), output_points, dtype=torch.float32
    )
    left = torch.floor(positions).to(dtype=torch.long)
    right = torch.clamp(left + 1, max=input_points - 1)
    right_weight = positions - left.to(dtype=positions.dtype)
    matrix = torch.zeros((output_points, input_points), dtype=torch.float32)
    rows = torch.arange(output_points, dtype=torch.long)
    matrix[rows, left] += 1.0 - right_weight
    matrix[rows, right] += right_weight
    return matrix


def build_inverse_model(config: ModelConfig, edh_scale_m: float):
    torch, nn, functional = _require_torch()
    if config.kernel_size % 2 != 1:
        raise ValueError("kernel_size must be odd.")

    class ResidualBlock(nn.Module):
        def __init__(self, width: int, dilation: int):
            super().__init__()
            padding = dilation * (config.kernel_size - 1) // 2
            groups = 8 if width % 8 == 0 else 1
            self.norm1 = nn.GroupNorm(groups, width)
            self.conv1 = nn.Sequential(
                nn.Conv1d(
                    width,
                    width,
                    config.kernel_size,
                    padding=padding,
                    dilation=dilation,
                    groups=width,
                ),
                nn.Conv1d(width, width, 1),
            )
            self.norm2 = nn.GroupNorm(groups, width)
            self.conv2 = nn.Sequential(
                nn.Conv1d(
                    width,
                    width,
                    config.kernel_size,
                    padding=padding,
                    dilation=dilation,
                    groups=width,
                ),
                nn.Conv1d(width, width, 1),
            )
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x):
            residual = x
            x = self.conv1(functional.silu(self.norm1(x)))
            x = self.dropout(x)
            x = self.conv2(functional.silu(self.norm2(x)))
            return residual + x

    class InverseRangeResCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.edh_scale_m = float(edh_scale_m)
            self.stem = nn.Conv1d(2, config.width, kernel_size=5, padding=2)
            self.blocks = nn.ModuleList(
                [ResidualBlock(config.width, dilation) for dilation in config.dilations]
            )
            groups = 8 if config.width % 8 == 0 else 1
            self.final_norm = nn.GroupNorm(groups, config.width)
            hidden = max(16, config.width // 2)
            self.edh_head = nn.Sequential(
                nn.Conv1d(config.width, hidden, 3, padding=1),
                nn.SiLU(),
                nn.Conv1d(hidden, 1, 1),
            )
            self.mask_head = nn.Conv1d(config.width, 1, 1)
            self.path_head = nn.Linear(config.width, PATH_CLASS_COUNT)
            self.height_head = nn.Linear(config.width, HEIGHT_CLASS_COUNT)
            coordinate = torch.linspace(0.0, 1.0, 500, dtype=torch.float32)
            self.register_buffer("coordinate", coordinate.view(1, 1, 500))
            self.register_buffer(
                "resampling_matrix",
                _linear_resampling_matrix(torch, input_points=500, output_points=101),
            )
            nn.init.constant_(self.edh_head[-1].bias, -2.0)

        def forward(self, pc_normalized):
            if pc_normalized.ndim != 2 or pc_normalized.shape[1] != 500:
                raise ValueError("pc_normalized must have shape batch x 500.")
            coordinate = self.coordinate.expand(pc_normalized.shape[0], -1, -1)
            x = torch.cat((pc_normalized.unsqueeze(1), coordinate), dim=1)
            x = self.stem(x)
            for block in self.blocks:
                x = block(x)
            x = functional.silu(self.final_norm(x))
            pooled = torch.mean(x, dim=2)
            # This is exactly align_corners=True linear interpolation, expressed
            # as a constant operator so its CUDA backward pass is deterministic.
            x_101 = torch.matmul(x, self.resampling_matrix.transpose(0, 1))
            edh_m = functional.softplus(self.edh_head(x_101)[:, 0, :]) * self.edh_scale_m
            mask_logits = self.mask_head(x_101)[:, 0, :]
            return {
                "edh_m": edh_m,
                "mask_logits": mask_logits,
                "path_logits": self.path_head(pooled),
                "height_logits": self.height_head(pooled),
            }

    return InverseRangeResCNN()


class InverseDataset:
    def __init__(
        self,
        pc_raw: np.ndarray,
        edh: np.ndarray,
        mask: np.ndarray,
        path_code: np.ndarray,
        height_code: np.ndarray,
        sample_id: np.ndarray,
        pc_mean: np.ndarray,
        pc_std: np.ndarray,
        radar: Optional[np.ndarray],
    ):
        torch, _, _ = _require_torch()
        self.torch = torch
        self.pc_raw = np.ascontiguousarray(pc_raw, dtype=np.float32)
        self.pc = np.ascontiguousarray(
            (self.pc_raw - pc_mean[None, :]) / pc_std[None, :], dtype=np.float32
        )
        self.edh = np.ascontiguousarray(edh, dtype=np.float32)
        self.mask = np.ascontiguousarray(mask, dtype=np.float32)
        self.path_code = np.asarray(path_code, dtype=np.int64)
        self.height_code = np.asarray(height_code, dtype=np.int64)
        self.sample_id = np.asarray(sample_id, dtype=np.int64)
        self.radar = (
            np.ascontiguousarray(radar, dtype=np.float32) if radar is not None else None
        )

    def __len__(self):
        return int(self.pc.shape[0])

    def __getitem__(self, index):
        radar = (
            self.torch.from_numpy(self.radar[index])
            if self.radar is not None
            else self.torch.empty(0, dtype=self.torch.float32)
        )
        return (
            self.torch.from_numpy(self.pc[index]),
            self.torch.from_numpy(self.pc_raw[index]),
            self.torch.from_numpy(self.edh[index]),
            self.torch.from_numpy(self.mask[index]),
            self.torch.tensor(self.path_code[index], dtype=self.torch.long),
            self.torch.tensor(self.height_code[index], dtype=self.torch.long),
            radar,
            int(self.sample_id[index]),
            int(index),
        )


def _class_weights(labels: np.ndarray, class_count: int, device):
    torch, _, _ = _require_torch()
    counts = np.bincount(labels, minlength=class_count).astype(np.float64)
    if np.any(counts <= 0):
        raise ValueError(f"Training labels omit a class: counts={counts.tolist()}")
    weights = 1.0 / np.sqrt(counts)
    weights /= np.mean(weights)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _stratified_positions(labels: np.ndarray, total: int, seed: int) -> np.ndarray:
    if total >= labels.size:
        return np.arange(labels.size, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    unique = np.unique(labels)
    minimum_each = max(1, min(16, total // max(1, len(unique) * 2)))
    for label in unique:
        candidates = np.flatnonzero(labels == label)
        count = min(minimum_each, candidates.size)
        selected.extend(rng.choice(candidates, count, replace=False).tolist())
    remaining = np.setdiff1d(np.arange(labels.size), np.asarray(selected), assume_unique=False)
    fill = min(total - len(selected), remaining.size)
    if fill > 0:
        selected.extend(rng.choice(remaining, fill, replace=False).tolist())
    return np.sort(np.asarray(selected[:total], dtype=np.int64))


def _subset_role(data: dict[str, Any], positions: np.ndarray) -> dict[str, Any]:
    result = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (len(data["pc"]),):
            result[key] = value[positions]
        elif key == "metadata":
            result[key] = value.iloc[positions].reset_index(drop=True)
        else:
            result[key] = value
    return result


def make_loaders(train_dataset, validation_dataset, train_labels, config, device):
    torch, _, _ = _require_torch()
    counts = np.bincount(train_labels, minlength=PATH_CLASS_COUNT).astype(np.float64)
    sample_weight = np.sqrt(np.max(counts) / counts[train_labels])
    sample_weight = np.minimum(sample_weight, 4.0)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(sample_weight, dtype=torch.double),
        num_samples=len(train_dataset),
        replacement=True,
        generator=generator,
    )
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.workers > 0,
    }
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler, **common
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, shuffle=False, **common
    )
    return train_loader, validation_loader


def composite_loss(
    output,
    truth_edh,
    truth_mask,
    path_code,
    height_code,
    pc_target,
    radar_term,
    edh_scale_m: float,
    path_weights,
    height_weights,
    config: TrainConfig,
    frozen_forward=None,
    forward_mean=None,
    forward_std=None,
):
    torch, _, functional = _require_torch()
    prediction = output["edh_m"]
    mask_logits = output["mask_logits"]
    positive = truth_mask > 0.5
    negative = ~positive
    prediction_scaled = prediction / edh_scale_m
    truth_scaled = truth_edh / edh_scale_m
    positive_loss = (
        functional.smooth_l1_loss(prediction_scaled[positive], truth_scaled[positive])
        if torch.any(positive)
        else prediction_scaled.sum() * 0.0
    )
    zero_loss = (
        functional.smooth_l1_loss(
            prediction_scaled[negative], torch.zeros_like(prediction_scaled[negative])
        )
        if torch.any(negative)
        else prediction_scaled.sum() * 0.0
    )
    positive_fraction = torch.mean(truth_mask)
    positive_weight = 0.5 / torch.clamp(positive_fraction, min=1.0e-3)
    negative_weight = 0.5 / torch.clamp(1.0 - positive_fraction, min=1.0e-3)
    point_weight = torch.where(positive, positive_weight, negative_weight)
    mask_loss = functional.binary_cross_entropy_with_logits(
        mask_logits, truth_mask, weight=point_weight
    )
    pair = positive[:, 1:] & positive[:, :-1]
    prediction_gradient = prediction_scaled[:, 1:] - prediction_scaled[:, :-1]
    truth_gradient = truth_scaled[:, 1:] - truth_scaled[:, :-1]
    smoothness_loss = (
        functional.smooth_l1_loss(
            prediction_gradient[pair], truth_gradient[pair]
        )
        if torch.any(pair)
        else prediction_scaled.sum() * 0.0
    )
    path_loss = functional.cross_entropy(
        output["path_logits"], path_code, weight=path_weights
    )
    height_loss = functional.cross_entropy(
        output["height_logits"], height_code, weight=height_weights
    )
    physics_loss = prediction_scaled.sum() * 0.0
    if config.physics_loss_weight > 0.0:
        if frozen_forward is None or radar_term.numel() == 0:
            raise ValueError("Physics loss requires the frozen forward model and radar term.")
        mask_probability = torch.sigmoid(mask_logits)
        hard_mask = (mask_probability >= 0.5).to(mask_probability.dtype)
        straight_through_mask = hard_mask + mask_probability - mask_probability.detach()
        edh_for_forward = prediction * straight_through_mask
        predicted_ls_normalized = frozen_forward(
            edh_for_forward, straight_through_mask
        )
        predicted_ls = predicted_ls_normalized * forward_std + forward_mean
        predicted_pc = -2.0 * predicted_ls + radar_term
        predicted_pc = predicted_pc - torch.mean(predicted_pc, dim=1, keepdim=True)
        physics_loss = functional.smooth_l1_loss(
            predicted_pc / config.physics_scale_db,
            pc_target / config.physics_scale_db,
        )
    total = (
        positive_loss
        + config.zero_edh_weight * zero_loss
        + config.mask_loss_weight * mask_loss
        + config.smoothness_weight * smoothness_loss
        + config.path_class_weight * path_loss
        + config.height_class_weight * height_loss
        + config.physics_loss_weight * physics_loss
    )
    parts = {
        "total": total,
        "edh_positive": positive_loss,
        "edh_zero": zero_loss,
        "mask": mask_loss,
        "smoothness": smoothness_loss,
        "path": path_loss,
        "height": height_loss,
        "physics": physics_loss,
    }
    return total, parts


def _regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth64 = np.asarray(truth, dtype=np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    error = prediction64 - truth64
    denominator = float(np.sum((truth64 - np.mean(truth64)) ** 2))
    sample_rmse = np.sqrt(np.mean(error * error, axis=1))
    return {
        "sample_count": int(truth64.shape[0]),
        "point_count": int(truth64.size),
        "rmse_m": float(np.sqrt(np.mean(error * error))),
        "mae_m": float(np.mean(np.abs(error))),
        "bias_m": float(np.mean(error)),
        "r2": float(1.0 - np.sum(error * error) / denominator)
        if denominator > 0.0
        else np.nan,
        "median_sample_rmse_m": float(np.median(sample_rmse)),
        "p95_sample_rmse_m": float(np.percentile(sample_rmse, 95)),
        "maximum_absolute_error_m": float(np.max(np.abs(error))),
    }


def _classification_metrics(truth: np.ndarray, prediction: np.ndarray, class_count: int):
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    recalls = []
    for category in range(class_count):
        selected = truth == category
        if np.any(selected):
            recalls.append(float(np.mean(prediction[selected] == category)))
    return {
        "accuracy": float(np.mean(prediction == truth)),
        "balanced_accuracy": float(np.mean(recalls)),
    }


def evaluate(model, loader, device, metadata: pd.DataFrame):
    torch, _, _ = _require_torch()
    model.eval()
    truth_edh_rows = []
    truth_mask_rows = []
    prediction_edh_rows = []
    prediction_mask_rows = []
    path_truth_rows = []
    path_prediction_rows = []
    height_truth_rows = []
    height_prediction_rows = []
    pc_rows = []
    sample_ids = []
    positions = []
    with torch.inference_mode():
        for pc, pc_raw, edh, mask, path_code, height_code, _, sample_id, position in loader:
            output = model(pc.to(device, non_blocking=True))
            probability = torch.sigmoid(output["mask_logits"])
            predicted_mask = probability >= 0.5
            predicted_edh = output["edh_m"] * predicted_mask.to(output["edh_m"].dtype)
            truth_edh_rows.append(edh.numpy())
            truth_mask_rows.append(mask.numpy())
            prediction_edh_rows.append(predicted_edh.cpu().numpy())
            prediction_mask_rows.append(predicted_mask.cpu().numpy())
            path_truth_rows.append(path_code.numpy())
            path_prediction_rows.append(torch.argmax(output["path_logits"], dim=1).cpu().numpy())
            height_truth_rows.append(height_code.numpy())
            height_prediction_rows.append(torch.argmax(output["height_logits"], dim=1).cpu().numpy())
            pc_rows.append(pc_raw.numpy())
            sample_ids.append(np.asarray(sample_id, dtype=np.int64))
            positions.append(np.asarray(position, dtype=np.int64))
    truth_edh = np.concatenate(truth_edh_rows)
    truth_mask = np.concatenate(truth_mask_rows).astype(bool)
    prediction_edh = np.concatenate(prediction_edh_rows)
    prediction_mask = np.concatenate(prediction_mask_rows).astype(bool)
    path_truth = np.concatenate(path_truth_rows)
    path_prediction = np.concatenate(path_prediction_rows)
    height_truth = np.concatenate(height_truth_rows)
    height_prediction = np.concatenate(height_prediction_rows)
    pc_raw = np.concatenate(pc_rows)
    sample_id = np.concatenate(sample_ids)
    order = np.concatenate(positions)
    if not np.array_equal(order, np.arange(len(metadata), dtype=np.int64)):
        raise ValueError("Validation loader order differs from metadata order.")
    overall = _regression_metrics(truth_edh, prediction_edh)
    positive = truth_mask
    if np.any(positive):
        positive_error = prediction_edh[positive] - truth_edh[positive]
        overall["positive_point_rmse_m"] = float(np.sqrt(np.mean(positive_error ** 2)))
        overall["positive_point_mae_m"] = float(np.mean(np.abs(positive_error)))
    negative = ~truth_mask
    overall["no_duct_point_mean_predicted_edh_m"] = (
        float(np.mean(prediction_edh[negative])) if np.any(negative) else np.nan
    )
    tp = int(np.sum(prediction_mask & truth_mask))
    fp = int(np.sum(prediction_mask & ~truth_mask))
    fn = int(np.sum(~prediction_mask & truth_mask))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    overall["duct_mask_precision"] = float(precision)
    overall["duct_mask_recall"] = float(recall)
    overall["duct_mask_f1"] = float(2 * precision * recall / max(1.0e-12, precision + recall))
    no_duct_samples = ~np.any(truth_mask, axis=1)
    if np.any(no_duct_samples):
        no_duct_prediction = prediction_mask[no_duct_samples]
        overall["no_duct_sample_count"] = int(np.sum(no_duct_samples))
        overall["no_duct_sample_false_alarm_rate"] = float(
            np.mean(np.any(no_duct_prediction, axis=1))
        )
        overall["no_duct_point_false_positive_rate"] = float(
            np.mean(no_duct_prediction)
        )
    else:
        overall["no_duct_sample_count"] = 0
        overall["no_duct_sample_false_alarm_rate"] = np.nan
        overall["no_duct_point_false_positive_rate"] = np.nan
    overall["path_class"] = _classification_metrics(path_truth, path_prediction, PATH_CLASS_COUNT)
    overall["height_class"] = _classification_metrics(height_truth, height_prediction, HEIGHT_CLASS_COUNT)

    group_rows = []
    balanced_path_values = []
    balanced_duct_path_values = []
    for dimension in ("path_class_name", "height_relation_class_name"):
        labels = metadata[dimension].astype(str).to_numpy()
        for category in sorted(np.unique(labels)):
            selected = labels == category
            row = {"dimension": dimension, "category": category}
            row.update(_regression_metrics(truth_edh[selected], prediction_edh[selected]))
            selected_positive = truth_mask[selected]
            if np.any(selected_positive):
                positive_error = (
                    prediction_edh[selected][selected_positive]
                    - truth_edh[selected][selected_positive]
                )
                row["positive_point_rmse_m"] = float(
                    np.sqrt(np.mean(positive_error * positive_error))
                )
                row["positive_point_mae_m"] = float(np.mean(np.abs(positive_error)))
            else:
                row["positive_point_rmse_m"] = np.nan
                row["positive_point_mae_m"] = np.nan
            group_rows.append(row)
            if dimension == "path_class_name":
                balanced_path_values.append(row["rmse_m"])
                if np.isfinite(row["positive_point_rmse_m"]):
                    balanced_duct_path_values.append(row["positive_point_rmse_m"])
    overall["balanced_path_rmse_m"] = float(np.mean(balanced_path_values))
    overall["balanced_duct_path_rmse_m"] = (
        float(np.mean(balanced_duct_path_values))
        if balanced_duct_path_values
        else np.nan
    )
    arrays = {
        "truth_edh": truth_edh,
        "truth_mask": truth_mask,
        "prediction_edh": prediction_edh,
        "prediction_mask": prediction_mask,
        "pc_raw": pc_raw,
        "sample_id": sample_id,
    }
    return overall, group_rows, arrays


def _atomic_torch_save(payload, path: Path) -> None:
    torch, _, _ = _require_torch()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(
    model,
    train_loader,
    validation_loader,
    validation_metadata,
    device,
    model_config,
    train_config,
    normalization,
    output_directory,
    path_weights,
    height_weights,
    frozen_forward,
    forward_mean,
    forward_std,
):
    torch, _, _ = _require_torch()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, train_config.epochs),
        eta_min=train_config.minimum_learning_rate,
    )
    amp_enabled = train_config.use_amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_scores = {
        "selected": float("inf"),
        "overall": float("inf"),
        "balanced": float("inf"),
    }
    best_epochs = {"selected": 0, "overall": 0, "balanced": 0}
    no_improvement = 0
    history = []
    best_path = output_directory / "best_model.pt"
    best_overall_path = output_directory / "best_overall_model.pt"
    best_balanced_path = output_directory / "best_balanced_model.pt"
    for epoch in range(1, train_config.epochs + 1):
        model.train()
        start_time = time.perf_counter()
        totals = {name: 0.0 for name in ("total", "edh_positive", "edh_zero", "mask", "smoothness", "path", "height", "physics")}
        sample_total = 0
        for pc, pc_raw, edh, mask, path_code, height_code, radar, _, _ in train_loader:
            pc = pc.to(device, non_blocking=True)
            pc_raw = pc_raw.to(device, non_blocking=True)
            edh = edh.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            path_code = path_code.to(device, non_blocking=True)
            height_code = height_code.to(device, non_blocking=True)
            radar = radar.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model(pc)
                total_loss, parts = composite_loss(
                    output,
                    edh,
                    mask,
                    path_code,
                    height_code,
                    pc_raw,
                    radar,
                    normalization["edh_scale_m"],
                    path_weights,
                    height_weights,
                    train_config,
                    frozen_forward,
                    forward_mean,
                    forward_std,
                )
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            batch_count = int(pc.shape[0])
            sample_total += batch_count
            for name, value in parts.items():
                totals[name] += float(value.detach()) * batch_count
        validation_metrics, _, _ = evaluate(
            model, validation_loader, device, validation_metadata
        )
        metric_value = float(validation_metrics[train_config.selection_metric])
        mask_penalty = train_config.mask_f1_penalty_m * (
            1.0 - float(validation_metrics["duct_mask_f1"])
        )
        selection_score = metric_value + mask_penalty
        overall_score = float(validation_metrics["rmse_m"])
        balanced_score = float(validation_metrics["balanced_duct_path_rmse_m"]) + mask_penalty
        if not all(np.isfinite(value) for value in (selection_score, overall_score, balanced_score)):
            raise FloatingPointError("A validation selection score is not finite.")
        improved_selected = (
            selection_score
            < best_scores["selected"] - train_config.minimum_improvement_m
        )
        improved_overall = (
            overall_score
            < best_scores["overall"] - train_config.minimum_improvement_m
        )
        improved_balanced = (
            balanced_score
            < best_scores["balanced"] - train_config.minimum_improvement_m
        )
        if improved_selected:
            best_scores["selected"] = selection_score
            best_epochs["selected"] = epoch
            no_improvement = 0
        else:
            no_improvement += 1
        if improved_overall:
            best_scores["overall"] = overall_score
            best_epochs["overall"] = epoch
        if improved_balanced:
            best_scores["balanced"] = balanced_score
            best_epochs["balanced"] = epoch
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        learning_rate_next = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate_used,
            "learning_rate_next": learning_rate_next,
            **{f"train_{name}_loss": value / sample_total for name, value in totals.items()},
            "validation_rmse_m": validation_metrics["rmse_m"],
            "validation_mae_m": validation_metrics["mae_m"],
            "validation_positive_point_rmse_m": validation_metrics["positive_point_rmse_m"],
            "validation_balanced_path_rmse_m": validation_metrics["balanced_path_rmse_m"],
            "validation_balanced_duct_path_rmse_m": validation_metrics["balanced_duct_path_rmse_m"],
            "validation_mask_f1": validation_metrics["duct_mask_f1"],
            "validation_no_duct_false_alarm_rate": validation_metrics["no_duct_sample_false_alarm_rate"],
            "validation_selection_metric_value_m": metric_value,
            "validation_mask_f1_penalty_m": mask_penalty,
            "validation_selection_score_m": selection_score,
            "is_best_selected": int(improved_selected),
            "is_best_overall": int(improved_overall),
            "is_best_balanced": int(improved_balanced),
            "epoch_seconds": time.perf_counter() - start_time,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            output_directory / "training_history.csv", index=False, encoding="utf-8-sig"
        )
        payload = {
            "script_version": SCRIPT_VERSION,
            "epoch": epoch,
            "selection_metric": train_config.selection_metric,
            "selection_metric_value_m": metric_value,
            "mask_f1_penalty_m": mask_penalty,
            "selection_score_m": selection_score,
            "checkpoint_validation_metrics": validation_metrics,
            "best_scores_m": dict(best_scores),
            "best_epochs": dict(best_epochs),
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "normalization": normalization,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        }
        _atomic_torch_save(payload, output_directory / "last_checkpoint.pt")
        if improved_selected:
            _atomic_torch_save(payload, best_path)
        if improved_overall:
            _atomic_torch_save(payload, best_overall_path)
        if improved_balanced:
            _atomic_torch_save(payload, best_balanced_path)
        LOGGER.info(
            "Epoch %03d/%03d | train=%.5f | val RMSE=%.3f m | positive=%.3f m | balanced-duct=%.3f m | mask F1=%.4f | lr=%.2e | %.1f s%s",
            epoch,
            train_config.epochs,
            row["train_total_loss"],
            row["validation_rmse_m"],
            row["validation_positive_point_rmse_m"],
            row["validation_balanced_duct_path_rmse_m"],
            row["validation_mask_f1"],
            learning_rate_used,
            row["epoch_seconds"],
            " | selected best" if improved_selected else "",
        )
        if no_improvement >= train_config.patience:
            LOGGER.info("Early stopping after %d epochs without improvement.", no_improvement)
            break
    if not best_path.is_file():
        raise RuntimeError("Training did not create best_model.pt.")
    checkpoint = _load_torch_checkpoint(best_path, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    checkpoint_selection = {
        "selected_checkpoint": best_path.name,
        "selection_metric": train_config.selection_metric,
        "mask_f1_penalty_m": train_config.mask_f1_penalty_m,
        "selected_epoch": int(checkpoint["epoch"]),
        "selected_score_m": float(checkpoint["selection_score_m"]),
        "best_overall_checkpoint": best_overall_path.name,
        "best_overall_epoch": int(best_epochs["overall"]),
        "best_overall_rmse_m": float(best_scores["overall"]),
        "best_balanced_checkpoint": best_balanced_path.name,
        "best_balanced_epoch": int(best_epochs["balanced"]),
        "best_balanced_score_m": float(best_scores["balanced"]),
    }
    return pd.DataFrame(history), checkpoint_selection


def save_examples(arrays, metadata, input_range_km, range_km, output_path):
    error = arrays["prediction_edh"] - arrays["truth_edh"]
    sample_rmse = np.sqrt(np.mean(error * error, axis=1))
    selected = []
    labels = metadata["path_class_name"].astype(str).to_numpy()
    for category in sorted(np.unique(labels)):
        candidates = np.flatnonzero(labels == category)
        selected.append(int(candidates[np.argmax(sample_rmse[candidates])]))
    for position in np.argsort(sample_rmse)[::-1]:
        if len(selected) >= 6:
            break
        if int(position) not in selected:
            selected.append(int(position))
    selected = np.asarray(selected[:6], dtype=np.int64)
    titles = np.asarray(
        [
            f"sample_id={arrays['sample_id'][i]} | {metadata.loc[i, 'path_class_name']} | {metadata.loc[i, 'height_relation_class_name']} | RMSE={sample_rmse[i]:.2f} m"
            for i in selected
        ],
        dtype="U240",
    )
    np.savez_compressed(
        output_path,
        pc=arrays["pc_raw"][selected],
        truth_edh=arrays["truth_edh"][selected],
        prediction_edh=arrays["prediction_edh"][selected],
        truth_mask=arrays["truth_mask"][selected],
        prediction_mask=arrays["prediction_mask"][selected],
        input_range_km=input_range_km,
        range_km=range_km,
        titles=titles,
    )


def _configure_matplotlib():
    cache = Path(tempfile.gettempdir()) / "matplotlib-inverse-baseline"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def render_results(output_directory: Path) -> None:
    plt = _configure_matplotlib()
    history = pd.read_csv(output_directory / "training_history.csv", encoding="utf-8-sig")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(history["epoch"], history["train_total_loss"])
    axes[0].set(xlabel="Epoch", ylabel="Composite loss", title="(a) Training loss")
    axes[1].plot(history["epoch"], history["validation_rmse_m"], label="overall")
    axes[1].plot(
        history["epoch"],
        history["validation_positive_point_rmse_m"],
        label="positive duct points",
    )
    axes[1].plot(
        history["epoch"],
        history["validation_balanced_duct_path_rmse_m"],
        label="balanced duct path classes",
    )
    axes[1].set(xlabel="Epoch", ylabel="EDH RMSE (m)", title="(b) Inverse-validation error")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(output_directory / "training_curves.png", dpi=180)
    plt.close(figure)

    with np.load(output_directory / "validation_example_data.npz", allow_pickle=False) as data:
        count = len(data["titles"])
        figure, axes = plt.subplots(count, 2, figsize=(14, 3.2 * count), constrained_layout=True)
        if count == 1:
            axes = np.asarray([axes])
        for row in range(count):
            axes[row, 0].plot(data["range_km"], data["pc"][row], color="#0072B2")
            axes[row, 0].set(xlabel="Range (km)", ylabel="Centered Pc (dB)")
            axes[row, 0].grid(alpha=0.2)
            axes[row, 1].plot(data["input_range_km"], data["truth_edh"][row], label="truth")
            axes[row, 1].plot(data["input_range_km"], data["prediction_edh"][row], label="prediction")
            axes[row, 1].set(xlabel="Range (km)", ylabel="EDH (m)", title=str(data["titles"][row]))
            axes[row, 1].grid(alpha=0.2)
        axes[0, 1].legend(frameon=False)
        figure.savefig(output_directory / "validation_prediction_examples.png", dpi=180)
        plt.close(figure)
    print(f"Plot rendering complete: {output_directory}")


def render_in_isolated_process(output_directory: Path) -> bool:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--render-results", str(output_directory.resolve())],
        check=False,
    )
    return result.returncode == 0


def run_self_test() -> None:
    torch, _, functional = _require_torch()
    model = build_inverse_model(ModelConfig(width=16, dilations=(1, 2), dropout=0.0), 30.0)
    test_input = torch.zeros((4, 500), dtype=torch.float32, requires_grad=True)
    output = model(test_input)
    assert output["edh_m"].shape == (4, 101)
    assert output["mask_logits"].shape == (4, 101)
    assert output["path_logits"].shape == (4, 3)
    assert output["height_logits"].shape == (4, 4)
    assert torch.isfinite(output["edh_m"]).all()
    assert model.resampling_matrix.shape == (101, 500)
    assert torch.allclose(
        torch.sum(model.resampling_matrix, dim=1),
        torch.ones(101),
        atol=1.0e-6,
    )
    reference_input = torch.linspace(-1.0, 1.0, 3000).reshape(2, 3, 500)
    interpolation_reference = functional.interpolate(
        reference_input, size=101, mode="linear", align_corners=True
    )
    matrix_result = torch.matmul(
        reference_input, model.resampling_matrix.transpose(0, 1)
    )
    assert torch.allclose(matrix_result, interpolation_reference, atol=1.0e-6)
    output["edh_m"].mean().backward()
    assert test_input.grad is not None and torch.isfinite(test_input.grad).all()
    print("Self-test passed.")


def run_training(args: argparse.Namespace) -> None:
    torch, _, _ = _require_torch()
    set_reproducibility(args.seed)
    output_directory = Path(args.output_dir)
    if args.smoke_test and output_directory == OUTPUT_DIR:
        output_directory = SMOKE_OUTPUT_DIR
    output_directory.mkdir(parents=True, exist_ok=True)
    if (output_directory / "training_history.csv").exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_directory} already contains a run. Choose another --output-dir or use --overwrite."
        )
    need_radar = args.input_mode == "fixed_radar" or args.physics_loss_weight > 0.0
    train_data = load_role_arrays(Path(args.pc_dir), "inverse_train", need_radar)
    validation_data = load_role_arrays(Path(args.pc_dir), "inverse_validation", need_radar)
    if np.intersect1d(train_data["sample_id"], validation_data["sample_id"]).size:
        raise ValueError("inverse_train and inverse_validation overlap.")
    if args.smoke_test:
        train_positions = _stratified_positions(
            train_data["path_code"], min(args.smoke_train_samples, len(train_data["pc"])), args.seed
        )
        validation_positions = _stratified_positions(
            validation_data["path_code"], min(args.smoke_validation_samples, len(validation_data["pc"])), args.seed + 1
        )
        train_data = _subset_role(train_data, train_positions)
        validation_data = _subset_role(validation_data, validation_positions)
        LOGGER.info("Smoke subsets: train=%d, validation=%d", len(train_positions), len(validation_positions))

    train_input, validation_input, radar_reference = prepare_inputs(
        train_data, validation_data, args.input_mode
    )
    pc_mean = np.mean(train_input, axis=0, dtype=np.float64).astype(np.float32)
    pc_std = np.std(train_input, axis=0, dtype=np.float64).astype(np.float32)
    pc_std = np.maximum(pc_std, np.float32(0.5))
    positive_edh = train_data["edh"][train_data["mask"] > 0.5]
    if positive_edh.size == 0:
        raise ValueError("Training data contain no resolved EDH values.")
    edh_scale_m = float(np.percentile(positive_edh, 99.5))
    if not np.isfinite(edh_scale_m) or edh_scale_m <= 0.0:
        raise ValueError("Failed to derive EDH scale.")
    physics_radar_train = None
    physics_radar_validation = None
    physics_radar_reference = None
    if args.physics_loss_weight > 0.0:
        if args.physics_radar_mode == "observed":
            physics_radar_train = train_data["radar"]
            physics_radar_validation = validation_data["radar"]
        else:
            # The network still receives the original centered Pc sequence.  Only
            # the training-time physics term is deprived of sample-specific radar
            # information, using a reference formed from inverse_train alone.
            physics_radar_reference = np.mean(
                train_data["radar"], axis=0, dtype=np.float64
            ).astype(np.float32)
            physics_radar_train = np.broadcast_to(
                physics_radar_reference, train_input.shape
            ).copy()
            physics_radar_validation = np.broadcast_to(
                physics_radar_reference, validation_input.shape
            ).copy()
    train_dataset = InverseDataset(
        train_input,
        train_data["edh"],
        train_data["mask"],
        train_data["path_code"],
        train_data["height_code"],
        train_data["sample_id"],
        pc_mean,
        pc_std,
        physics_radar_train,
    )
    validation_dataset = InverseDataset(
        validation_input,
        validation_data["edh"],
        validation_data["mask"],
        validation_data["path_code"],
        validation_data["height_code"],
        validation_data["sample_id"],
        pc_mean,
        pc_std,
        physics_radar_validation,
    )
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    LOGGER.info("Device: %s", device)
    model_config = ModelConfig(
        width=args.width, dilations=tuple(args.dilations), dropout=args.dropout
    )
    train_config = TrainConfig(
        epochs=args.smoke_epochs if args.smoke_test else args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        minimum_improvement_m=args.minimum_improvement_m,
        minimum_learning_rate=args.minimum_learning_rate,
        selection_metric=args.selection_metric,
        mask_f1_penalty_m=args.mask_f1_penalty_m,
        workers=args.workers,
        seed=args.seed,
        use_amp=not args.no_amp,
        zero_edh_weight=args.zero_edh_weight,
        mask_loss_weight=args.mask_loss_weight,
        smoothness_weight=args.smoothness_weight,
        path_class_weight=args.path_class_weight,
        height_class_weight=args.height_class_weight,
        physics_loss_weight=args.physics_loss_weight,
        physics_scale_db=args.physics_scale_db,
    )
    train_loader, validation_loader = make_loaders(
        train_dataset, validation_dataset, train_data["path_code"], train_config, device
    )
    path_weights = _class_weights(train_data["path_code"], PATH_CLASS_COUNT, device)
    height_weights = _class_weights(train_data["height_code"], HEIGHT_CLASS_COUNT, device)
    model = build_inverse_model(model_config, edh_scale_m).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    LOGGER.info("Inverse model parameters: %s", f"{parameter_count:,}")
    frozen_forward = forward_mean = forward_std = forward_checkpoint = None
    if train_config.physics_loss_weight > 0.0:
        frozen_forward, forward_mean, forward_std, forward_checkpoint = load_frozen_forward(
            Path(args.forward_script), Path(args.forward_checkpoint), device
        )
    normalization = {
        "input_mode": args.input_mode,
        "pc_mean_dB": pc_mean,
        "pc_std_dB": pc_std,
        "edh_scale_m": edh_scale_m,
        "radar_reference_dB": radar_reference,
    }
    np.savez_compressed(
        output_directory / "normalization.npz",
        pc_mean_dB=pc_mean,
        pc_std_dB=pc_std,
        edh_scale_m=np.asarray(edh_scale_m),
        radar_reference_dB=np.asarray([] if radar_reference is None else radar_reference),
        input_mode=np.asarray(args.input_mode),
    )
    run_config = {
        "script_version": SCRIPT_VERSION,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "input_mode": args.input_mode,
        "network_inputs": ["pc_centered_dB only"],
        "era5_or_radar_term_used_as_network_input": False,
        "physics_radar_mode": args.physics_radar_mode,
        "physics_radar_reference_source": (
            "sample-specific radar_term_dB" if args.physics_radar_mode == "observed"
            else "range-wise inverse_train mean radar_term_dB"
        ),
        "roles_loaded": list(ROLES),
        "joint_test_loaded": False,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "parameter_count": parameter_count,
        "pc_directory": str(args.pc_dir),
        "forward_checkpoint": str(args.forward_checkpoint)
        if train_config.physics_loss_weight > 0.0
        else None,
    }
    _write_json(output_directory / "run_config.json", run_config)
    history, checkpoint_selection = train(
        model,
        train_loader,
        validation_loader,
        validation_data["metadata"],
        device,
        model_config,
        train_config,
        normalization,
        output_directory,
        path_weights,
        height_weights,
        frozen_forward,
        forward_mean,
        forward_std,
    )
    metrics, group_rows, arrays = evaluate(
        model, validation_loader, device, validation_data["metadata"]
    )
    pd.DataFrame(group_rows).to_csv(
        output_directory / "validation_by_group.csv", index=False, encoding="utf-8-sig"
    )
    _write_json(
        output_directory / "validation_metrics.json",
        {"checkpoint_selection": checkpoint_selection, **metrics},
    )
    save_examples(
        arrays,
        validation_data["metadata"],
        validation_data["input_range_km"],
        validation_data["range_km"],
        output_directory / "validation_example_data.npz",
    )
    plot_ok = render_in_isolated_process(output_directory)
    summary = {
        **run_config,
        "best_epoch": checkpoint_selection["selected_epoch"],
        "checkpoint_selection": checkpoint_selection,
        "validation_metrics": metrics,
        "plot_rendering_passed": plot_ok,
        "recommended_next_action": (
            "Freeze this supervised baseline before enabling physics loss."
            if train_config.physics_loss_weight == 0.0
            else "Compare against the frozen supervised baseline as a physics-loss ablation."
        ),
    }
    _write_json(output_directory / "training_summary.json", summary)
    LOGGER.info("Training complete: %s", output_directory)
    print(json.dumps(_json_value(summary), ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pc-dir", type=Path, default=PC_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--forward-script", type=Path, default=PROGRAM_DIR / "train_forward_rescnn.py"
    )
    parser.add_argument(
        "--forward-checkpoint", type=Path, default=FORWARD_DIR / "best_model.pt"
    )
    parser.add_argument("--input-mode", choices=("era5", "fixed_radar"), default="era5")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--minimum-improvement-m", type=float, default=1.0e-3)
    parser.add_argument("--minimum-learning-rate", type=float, default=1.0e-5)
    parser.add_argument(
        "--selection-metric",
        choices=(
            "positive_point_rmse_m",
            "rmse_m",
            "balanced_duct_path_rmse_m",
        ),
        default="positive_point_rmse_m",
        help=(
            "Primary checkpoint/early-stopping metric. A duct-mask F1 penalty is "
            "added so no-duct behavior is not represented by an artificial zero RMSE."
        ),
    )
    parser.add_argument("--mask-f1-penalty-m", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--zero-edh-weight", type=float, default=0.25)
    parser.add_argument("--mask-loss-weight", type=float, default=0.25)
    parser.add_argument("--smoothness-weight", type=float, default=0.10)
    parser.add_argument("--path-class-weight", type=float, default=0.05)
    parser.add_argument("--height-class-weight", type=float, default=0.05)
    parser.add_argument("--physics-loss-weight", type=float, default=0.0)
    parser.add_argument("--physics-scale-db", type=float, default=5.0)
    parser.add_argument(
        "--physics-radar-mode",
        choices=("observed", "fixed_reference"),
        default="observed",
        help=(
            "Training-time radar term used only inside the frozen-forward physics "
            "loss. The inverse network always receives pc_centered_dB only."
        ),
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--smoke-train-samples", type=int, default=512)
    parser.add_argument("--smoke-validation-samples", type=int, default=256)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    if len(sys.argv) == 3 and sys.argv[1] == "--render-results":
        render_results(Path(sys.argv[2]))
        return
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return
    numeric_nonnegative = (
        args.zero_edh_weight,
        args.mask_loss_weight,
        args.smoothness_weight,
        args.path_class_weight,
        args.height_class_weight,
        args.physics_loss_weight,
    )
    if any(value < 0.0 for value in numeric_nonnegative):
        raise ValueError("Loss weights must be nonnegative.")
    if args.batch_size <= 0 or args.epochs <= 0:
        raise ValueError("Batch size and epochs must be positive.")
    if args.learning_rate <= 0.0 or args.minimum_learning_rate <= 0.0:
        raise ValueError("Learning rates must be positive.")
    if args.minimum_learning_rate > args.learning_rate:
        raise ValueError("minimum-learning-rate cannot exceed learning-rate.")
    if args.patience <= 0:
        raise ValueError("patience must be positive.")
    if args.mask_f1_penalty_m < 0.0:
        raise ValueError("mask-f1-penalty-m must be nonnegative.")
    run_training(args)


if __name__ == "__main__":
    main()
