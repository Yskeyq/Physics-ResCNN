#!/usr/bin/env python3
r"""Train the first forward surrogate d(r) -> L_s(r,z_s=0.3 m).

This script is deliberately restricted to the mutually exclusive
``forward_train`` and ``forward_validation`` roles.  It never loads samples
from ``inverse_train``, ``inverse_validation`` or ``joint_test``.

Inputs
------
1. ``pe_ls_dataset_zs_0p3m_v2.mat`` (MATLAB v7.3/HDF5)
2. A model-role directory containing:
   - ``model_role_indices.npz``
   - ``model_role_manifest.csv``

Model
-----
The formal input remains d(r).  The network uses three deterministic channels:

    d(r) / d_scale,  1[d(r)>0],  r / 100 km

The 101 EDH nodes at 0:1:100 km are linearly evaluated at the 500 PE marching
midpoints (0.1:0.2:99.9 km). A dilated residual 1-D CNN then predicts the
pointwise-normalized loss at 0.2:0.2:100 km.

Examples (PowerShell)
---------------------
Self-test without project data::

    python .\train_forward_rescnn.py --self-test

Small real-data smoke test::

    python .\train_forward_rescnn.py `
      .\pe_ls_dataset_zs_0p3m_v2.mat `
      .\model_role_split `
      --smoke-test

Full training (outputs default to artifacts/forward_rescnn)::

    python .\train_forward_rescnn.py `
      .\pe_ls_dataset_zs_0p3m_v2.mat `
      .\model_role_split
"""

from __future__ import annotations

import argparse
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
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("train_forward_rescnn")
SCRIPT_VERSION = "1.0.1"
DEFAULT_OUTPUT_DIRECTORY = (OUTPUT_ROOT / 'forward_rescnn')
DEFAULT_SMOKE_OUTPUT_DIRECTORY = (OUTPUT_ROOT / 'forward_rescnn_smoke')
FORBIDDEN_ROLES = {"inverse_train", "inverse_validation", "joint_test"}


@dataclass(frozen=True)
class ModelConfig:
    input_nodes: int = 101
    output_nodes: int = 500
    width: int = 64
    kernel_size: int = 3
    dilations: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    dropout: float = 0.05


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 150
    batch_size: int = 128
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    gradient_loss_weight: float = 0.10
    gradient_scale_db: float = 5.0
    huber_beta: float = 1.0
    early_stopping_patience: int = 20
    minimum_improvement_db: float = 1.0e-4
    workers: int = 0
    seed: int = 20260812
    use_amp: bool = True


@dataclass(frozen=True)
class Normalization:
    edh_scale_m: float
    target_mean_db: np.ndarray
    target_std_db: np.ndarray


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PyTorch is required. Run the program in your existing pytorch "
            "environment or install requirements_forward_rescnn.txt."
        ) from exc
    return torch, nn, functional


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "h5py is required for the MATLAB v7.3 PE file. Run: "
            "python -m pip install -r requirements_forward_rescnn.txt"
        ) from exc
    return h5py


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch, _, _ = _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic convolution is preferred for the first scientific baseline.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:  # pragma: no cover - older supported PyTorch
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


def _read_vector(handle, name: str, dtype=None) -> np.ndarray:
    if name not in handle:
        raise ValueError(f"PE MAT file is missing {name}.")
    values = np.asarray(handle[name][...]).reshape(-1)
    return values.astype(dtype, copy=False) if dtype is not None else values


def _read_matlab_matrix(handle, name: str, sample_count: int) -> np.ndarray:
    if name not in handle:
        raise ValueError(f"PE MAT file is missing {name}.")
    dataset = handle[name]
    if dataset.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix; found {dataset.shape}.")
    # MATLAB v7.3 normally reverses dimensions in the underlying HDF5 dataset.
    if dataset.shape[1] == sample_count:
        values = np.asarray(dataset[...]).T
    elif dataset.shape[0] == sample_count:
        values = np.asarray(dataset[...])
    else:
        raise ValueError(
            f"Cannot locate N={sample_count} sample dimension in {name}: "
            f"HDF5 shape={dataset.shape}."
        )
    return values


def load_pe_arrays(pe_mat: Path) -> Mapping[str, np.ndarray]:
    h5py = _require_h5py()
    pe_mat = Path(pe_mat)
    if not pe_mat.is_file():
        raise FileNotFoundError(pe_mat)
    try:
        handle = h5py.File(pe_mat, "r")
    except OSError as exc:
        raise ValueError(f"{pe_mat} is not a readable MATLAB v7.3 file.") from exc

    with handle:
        source_rows = _read_vector(handle, "source_mat_row", np.int64)
        sample_count = int(source_rows.size)
        if sample_count == 0:
            raise ValueError("The PE MAT file contains no samples.")
        status = _read_vector(handle, "sample_status_code", np.int16)
        if status.size != sample_count or np.any(status != 1):
            unique, counts = np.unique(status, return_counts=True)
            raise ValueError(
                "Every PE row must be complete (status 1); found "
                f"{dict(zip(unique.tolist(), counts.tolist()))}."
            )
        edh = _read_matlab_matrix(handle, "input_edh_m", sample_count).astype(
            np.float32, copy=False
        )
        duct_mask = _read_matlab_matrix(
            handle, "input_duct_mask", sample_count
        ).astype(np.float32, copy=False)
        loss = _read_matlab_matrix(handle, "Ls_oneway_dB", sample_count).astype(
            np.float32, copy=False
        )
        input_range_km = _read_vector(handle, "input_range_km", np.float64)
        output_range_m = _read_vector(handle, "x_range_m", np.float64)
        zs_values = _read_vector(handle, "zs_m", np.float64)
        if zs_values.size != 1:
            raise ValueError("zs_m must be scalar.")

    if edh.shape != duct_mask.shape:
        raise ValueError(f"EDH/mask shapes differ: {edh.shape} and {duct_mask.shape}.")
    if edh.shape != (sample_count, 101):
        raise ValueError(f"Expected EDH shape ({sample_count},101), got {edh.shape}.")
    if loss.shape != (sample_count, 500):
        raise ValueError(f"Expected L_s shape ({sample_count},500), got {loss.shape}.")
    if input_range_km.size != 101 or output_range_m.size != 500:
        raise ValueError("Expected 101 EDH coordinates and 500 L_s coordinates.")
    if not np.allclose(input_range_km, np.arange(101), atol=1e-6):
        raise ValueError("EDH coordinate must be 0:1:100 km.")
    if not np.allclose(output_range_m, np.arange(1, 501) * 200.0, atol=1e-6):
        raise ValueError("L_s coordinate must be 0.2:0.2:100 km.")
    if not math.isclose(float(zs_values[0]), 0.3, abs_tol=1e-9):
        raise ValueError(f"Expected z_s=0.3 m, found {zs_values[0]:g} m.")
    if not np.isfinite(edh).all() or np.any(edh < 0.0):
        raise ValueError("input_edh_m contains NaN, infinity or negative values.")
    if not np.isin(duct_mask, (0.0, 1.0)).all():
        raise ValueError("input_duct_mask must contain only 0 and 1.")
    if not np.isfinite(loss).all():
        raise ValueError("Ls_oneway_dB contains NaN or infinity.")
    if np.any((duct_mask == 0.0) & (edh != 0.0)):
        raise ValueError("No-duct positions must use d(r)=0 in the PE learning input.")

    LOGGER.info(
        "Loaded PE arrays: EDH=%s, L_s=%s, z_s=%.1f m.",
        edh.shape,
        loss.shape,
        float(zs_values[0]),
    )
    return {
        "source_mat_row": source_rows,
        "edh_m": edh,
        "duct_mask": duct_mask,
        "loss_db": loss,
        "input_range_km": input_range_km,
        "output_range_km": output_range_m / 1000.0,
        "zs_m": np.asarray(float(zs_values[0])),
    }


def load_forward_indices(split_directory: Path, sample_count: int):
    split_directory = Path(split_directory)
    index_file = split_directory / "model_role_indices.npz"
    manifest_file = split_directory / "model_role_manifest.csv"
    if not index_file.is_file():
        raise FileNotFoundError(index_file)
    if not manifest_file.is_file():
        raise FileNotFoundError(manifest_file)
    with np.load(index_file, allow_pickle=False) as archive:
        required = {
            "forward_train_pe_row_0based",
            "forward_validation_pe_row_0based",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Role-index NPZ is missing arrays: {missing}")
        train_indices = np.asarray(
            archive["forward_train_pe_row_0based"], dtype=np.int64
        ).reshape(-1)
        validation_indices = np.asarray(
            archive["forward_validation_pe_row_0based"], dtype=np.int64
        ).reshape(-1)

    for name, values in (
        ("forward_train", train_indices),
        ("forward_validation", validation_indices),
    ):
        if values.size == 0:
            raise ValueError(f"{name} contains no rows.")
        if np.unique(values).size != values.size:
            raise ValueError(f"{name} contains duplicate PE indices.")
        if np.any(values < 0) or np.any(values >= sample_count):
            raise ValueError(f"{name} contains an out-of-range PE index.")
    if np.intersect1d(train_indices, validation_indices).size:
        raise ValueError("Forward train/validation PE indices overlap.")

    manifest = pd.read_csv(manifest_file, encoding="utf-8-sig")
    required_columns = {
        "pe_row_index_0based",
        "role_split_name",
        "path_class_name",
        "height_relation_class_name",
        "f_below_bin",
        "valid_time_utc",
    }
    missing_columns = sorted(required_columns - set(manifest.columns))
    if missing_columns:
        raise ValueError(f"Role manifest is missing columns: {missing_columns}")
    if manifest["pe_row_index_0based"].duplicated().any():
        raise ValueError("Role manifest contains duplicate PE row indices.")
    indexed = manifest.set_index("pe_row_index_0based", drop=False)
    expected_train = indexed.loc[train_indices, "role_split_name"].astype(str)
    expected_validation = indexed.loc[
        validation_indices, "role_split_name"
    ].astype(str)
    if not expected_train.eq("forward_train").all():
        raise ValueError("Train indices do not all map to forward_train.")
    if not expected_validation.eq("forward_validation").all():
        raise ValueError(
            "Validation indices do not all map to forward_validation."
        )
    selected_roles = set(expected_train).union(set(expected_validation))
    if selected_roles.intersection(FORBIDDEN_ROLES):
        raise ValueError("A forbidden inverse/test role entered forward training.")

    LOGGER.info(
        "Role isolation passed: forward_train=%d, forward_validation=%d; "
        "inverse and joint-test roles are untouched.",
        train_indices.size,
        validation_indices.size,
    )
    return train_indices, validation_indices, manifest


def compute_normalization(
    edh_m: np.ndarray,
    duct_mask: np.ndarray,
    loss_db: np.ndarray,
    train_indices: np.ndarray,
) -> Normalization:
    train_edh = edh_m[train_indices]
    train_mask = duct_mask[train_indices] > 0.5
    positive = train_edh[train_mask]
    if positive.size == 0:
        raise ValueError("Forward training set contains no resolved EDH values.")
    # Robust scale only; values above P99.5 are retained rather than clipped.
    edh_scale = float(np.percentile(positive, 99.5))
    if not np.isfinite(edh_scale) or edh_scale <= 0.0:
        raise ValueError("Failed to derive a positive EDH scale.")
    train_loss = loss_db[train_indices].astype(np.float64)
    target_mean = np.mean(train_loss, axis=0).astype(np.float32)
    target_std = np.std(train_loss, axis=0).astype(np.float32)
    target_std = np.maximum(target_std, np.float32(0.5))
    return Normalization(
        edh_scale_m=edh_scale,
        target_mean_db=target_mean,
        target_std_db=target_std,
    )


def build_model(config: ModelConfig, edh_scale_m: float):
    torch, nn, functional = _require_torch()

    if config.kernel_size % 2 != 1:
        raise ValueError("kernel_size must be odd.")

    class ResidualDilatedBlock(nn.Module):
        def __init__(self, width: int, dilation: int, dropout: float):
            super().__init__()
            padding = dilation * (config.kernel_size - 1) // 2
            group_count = 8 if width % 8 == 0 else 1
            self.norm1 = nn.GroupNorm(group_count, width)
            self.conv1 = nn.Sequential(
                nn.Conv1d(
                    width,
                    width,
                    config.kernel_size,
                    padding=padding,
                    dilation=dilation,
                    groups=width,
                ),
                nn.Conv1d(width, width, kernel_size=1),
            )
            self.norm2 = nn.GroupNorm(group_count, width)
            self.conv2 = nn.Sequential(
                nn.Conv1d(
                    width,
                    width,
                    config.kernel_size,
                    padding=padding,
                    dilation=dilation,
                    groups=width,
                ),
                nn.Conv1d(width, width, kernel_size=1),
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, features):
            residual = features
            features = self.conv1(functional.silu(self.norm1(features)))
            features = self.dropout(features)
            features = self.conv2(functional.silu(self.norm2(features)))
            return residual + features

    class RangeResCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.edh_scale_m = float(edh_scale_m)
            self.stem = nn.Conv1d(3, config.width, kernel_size=5, padding=2)
            self.blocks = nn.ModuleList(
                [
                    ResidualDilatedBlock(config.width, dilation, config.dropout)
                    for dilation in config.dilations
                ]
            )
            group_count = 8 if config.width % 8 == 0 else 1
            self.head_norm = nn.GroupNorm(group_count, config.width)
            self.head = nn.Sequential(
                nn.Conv1d(config.width, config.width // 2, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv1d(config.width // 2, 1, kernel_size=1),
            )
            self.direct = nn.Conv1d(3, 1, kernel_size=1)
            coordinate = torch.arange(1, 501, dtype=torch.float32) / 500.0
            self.register_buffer("output_coordinate", coordinate.view(1, 1, -1))

        def forward(self, edh_m_batch, duct_mask_batch):
            if edh_m_batch.ndim != 2 or edh_m_batch.shape[1] != 101:
                raise ValueError("edh_m_batch must have shape batch x 101.")
            if duct_mask_batch.shape != edh_m_batch.shape:
                raise ValueError("duct_mask_batch must match EDH shape.")
            edh_channel = (edh_m_batch / self.edh_scale_m).unsqueeze(1)
            mask_channel = duct_mask_batch.unsqueeze(1)
            # The PE marcher evaluates its environment at step midpoints
            # 0.1:0.2:99.9 km and saves loss after each step at
            # 0.2:0.2:100 km. A 1001-node aligned interpolation gives
            # 0:0.1:100 km exactly; odd positions are the 500 PE midpoints.
            edh_500 = functional.interpolate(
                edh_channel, size=1001, mode="linear", align_corners=True
            )[:, :, 1:1000:2]
            mask_500 = functional.interpolate(
                mask_channel, size=1001, mode="linear", align_corners=True
            )[:, :, 1:1000:2]
            coordinate = self.output_coordinate.expand(edh_m_batch.shape[0], -1, -1)
            inputs = torch.cat((edh_500, mask_500, coordinate), dim=1)
            features = self.stem(inputs)
            for block in self.blocks:
                features = block(features)
            prediction = self.head(functional.silu(self.head_norm(features)))
            prediction = prediction + self.direct(inputs)
            return prediction[:, 0, :]

    return RangeResCNN()


def make_dataset(
    edh_m: np.ndarray,
    duct_mask: np.ndarray,
    loss_db: np.ndarray,
    indices: np.ndarray,
    normalization: Normalization,
):
    torch, _, _ = _require_torch()

    class ForwardArrayDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.indices = np.asarray(indices, dtype=np.int64)
            self.edh = np.ascontiguousarray(edh_m[self.indices], dtype=np.float32)
            self.mask = np.ascontiguousarray(
                duct_mask[self.indices], dtype=np.float32
            )
            raw_target = np.ascontiguousarray(
                loss_db[self.indices], dtype=np.float32
            )
            self.target = np.ascontiguousarray(
                (raw_target - normalization.target_mean_db[None, :])
                / normalization.target_std_db[None, :],
                dtype=np.float32,
            )

        def __len__(self):
            return int(self.indices.size)

        def __getitem__(self, position):
            return (
                torch.from_numpy(self.edh[position]),
                torch.from_numpy(self.mask[position]),
                torch.from_numpy(self.target[position]),
                int(self.indices[position]),
            )

    return ForwardArrayDataset()


def make_loaders(
    train_dataset,
    validation_dataset,
    train_config: TrainConfig,
    device,
):
    torch, _, _ = _require_torch()
    generator = torch.Generator()
    generator.manual_seed(train_config.seed)
    common = {
        "batch_size": train_config.batch_size,
        "num_workers": train_config.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": train_config.workers > 0,
    }
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        shuffle=False,
        **common,
    )
    return train_loader, validation_loader


def composite_loss(
    prediction_normalized,
    target_normalized,
    target_mean,
    target_std,
    config: TrainConfig,
):
    _, _, functional = _require_torch()
    point_loss = functional.smooth_l1_loss(
        prediction_normalized,
        target_normalized,
        beta=config.huber_beta,
    )
    prediction_db = prediction_normalized * target_std + target_mean
    target_db = target_normalized * target_std + target_mean
    prediction_gradient = (
        prediction_db[:, 1:] - prediction_db[:, :-1]
    ) / config.gradient_scale_db
    target_gradient = (
        target_db[:, 1:] - target_db[:, :-1]
    ) / config.gradient_scale_db
    gradient_loss = functional.smooth_l1_loss(
        prediction_gradient,
        target_gradient,
        beta=config.huber_beta,
    )
    total = point_loss + config.gradient_loss_weight * gradient_loss
    return total, point_loss, gradient_loss


def _metrics(truth_db: np.ndarray, prediction_db: np.ndarray) -> Dict[str, float]:
    truth = np.asarray(truth_db, dtype=np.float64)
    prediction = np.asarray(prediction_db, dtype=np.float64)
    error = prediction - truth
    mse = float(np.mean(error * error))
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    r2 = float(1.0 - np.sum(error * error) / denominator) if denominator > 0 else np.nan
    sample_rmse = np.sqrt(np.mean(error * error, axis=1))
    return {
        "sample_count": int(truth.shape[0]),
        "point_count": int(truth.size),
        "rmse_db": math.sqrt(mse),
        "mae_db": float(np.mean(np.abs(error))),
        "bias_db": float(np.mean(error)),
        "r2": r2,
        "median_sample_rmse_db": float(np.median(sample_rmse)),
        "p95_sample_rmse_db": float(np.percentile(sample_rmse, 95)),
        "maximum_absolute_error_db": float(np.max(np.abs(error))),
    }


def evaluate_model(
    model,
    loader,
    device,
    normalization: Normalization,
):
    torch, _, _ = _require_torch()
    model.eval()
    predictions = []
    truths = []
    rows = []
    target_mean = torch.as_tensor(
        normalization.target_mean_db, dtype=torch.float32, device=device
    )
    target_std = torch.as_tensor(
        normalization.target_std_db, dtype=torch.float32, device=device
    )
    with torch.inference_mode():
        for edh, mask, target_normalized, row_index in loader:
            edh = edh.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            prediction_normalized = model(edh, mask)
            prediction_db = prediction_normalized * target_std + target_mean
            truth_db = target_normalized.to(device, non_blocking=True) * target_std + target_mean
            predictions.append(prediction_db.cpu().numpy())
            truths.append(truth_db.cpu().numpy())
            rows.append(np.asarray(row_index, dtype=np.int64))
    prediction = np.concatenate(predictions, axis=0)
    truth = np.concatenate(truths, axis=0)
    row_indices = np.concatenate(rows, axis=0)
    return _metrics(truth, prediction), truth, prediction, row_indices


def _atomic_torch_save(payload, path: Path) -> None:
    torch, _, _ = _require_torch()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _checkpoint_payload(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_validation_rmse: float,
    epochs_without_improvement: int,
    model_config: ModelConfig,
    train_config: TrainConfig,
    normalization: Normalization,
) -> Mapping[str, object]:
    return {
        "script_version": SCRIPT_VERSION,
        "epoch": int(epoch),
        "best_validation_rmse_db": float(best_validation_rmse),
        "epochs_without_improvement": int(epochs_without_improvement),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "edh_scale_m": float(normalization.edh_scale_m),
        "target_mean_db": normalization.target_mean_db,
        "target_std_db": normalization.target_std_db,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }


def _load_checkpoint(path: Path, device):
    torch, _, _ = _require_torch()
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch before weights_only was introduced
        return torch.load(path, map_location=device)


def train_model(
    model,
    train_loader,
    validation_loader,
    device,
    normalization: Normalization,
    model_config: ModelConfig,
    train_config: TrainConfig,
    output_directory: Path,
    resume_checkpoint: Optional[Path],
):
    torch, _, _ = _require_torch()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, train_config.epochs),
        eta_min=train_config.learning_rate * 0.01,
    )
    amp_enabled = bool(train_config.use_amp and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):  # PyTorch 2.0 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    target_mean = torch.as_tensor(
        normalization.target_mean_db, dtype=torch.float32, device=device
    )
    target_std = torch.as_tensor(
        normalization.target_std_db, dtype=torch.float32, device=device
    )

    start_epoch = 1
    best_validation_rmse = float("inf")
    epochs_without_improvement = 0
    history_rows = []
    if resume_checkpoint is not None:
        checkpoint = _load_checkpoint(resume_checkpoint, device)
        if checkpoint.get("script_version") != SCRIPT_VERSION:
            raise ValueError(
                "Resume checkpoint was created by a different script version."
            )
        if checkpoint.get("model_config") != asdict(model_config):
            raise ValueError("Resume checkpoint model configuration does not match.")
        if not math.isclose(
            float(checkpoint["edh_scale_m"]),
            normalization.edh_scale_m,
            rel_tol=0.0,
            abs_tol=1e-8,
        ) or not np.allclose(
            np.asarray(checkpoint["target_mean_db"]),
            normalization.target_mean_db,
            rtol=0.0,
            atol=1e-6,
        ) or not np.allclose(
            np.asarray(checkpoint["target_std_db"]),
            normalization.target_std_db,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError("Resume checkpoint normalization does not match this data.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_rmse = float(checkpoint["best_validation_rmse_db"])
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        history_file = output_directory / "training_history.csv"
        if history_file.exists():
            history_rows = pd.read_csv(history_file).to_dict("records")
        LOGGER.info("Resumed checkpoint at epoch %d.", start_epoch - 1)

    best_path = output_directory / "best_model.pt"
    last_path = output_directory / "last_checkpoint.pt"
    for epoch in range(start_epoch, train_config.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        sample_total = 0
        accumulated_total = 0.0
        accumulated_point = 0.0
        accumulated_gradient = 0.0
        for edh, mask, target_normalized, _ in train_loader:
            edh = edh.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            target_normalized = target_normalized.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                prediction_normalized = model(edh, mask)
                total_loss, point_loss, gradient_loss = composite_loss(
                    prediction_normalized,
                    target_normalized,
                    target_mean,
                    target_std,
                    train_config,
                )
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            batch_count = int(edh.shape[0])
            sample_total += batch_count
            accumulated_total += float(total_loss.detach()) * batch_count
            accumulated_point += float(point_loss.detach()) * batch_count
            accumulated_gradient += float(gradient_loss.detach()) * batch_count

        scheduler.step()
        validation_metrics, _, _, _ = evaluate_model(
            model, validation_loader, device, normalization
        )
        elapsed = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_total_loss": accumulated_total / sample_total,
            "train_point_loss": accumulated_point / sample_total,
            "train_gradient_loss": accumulated_gradient / sample_total,
            "validation_rmse_db": validation_metrics["rmse_db"],
            "validation_mae_db": validation_metrics["mae_db"],
            "validation_r2": validation_metrics["r2"],
            "epoch_seconds": elapsed,
        }
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(
            output_directory / "training_history.csv",
            index=False,
            encoding="utf-8-sig",
        )

        improved = (
            validation_metrics["rmse_db"]
            < best_validation_rmse - train_config.minimum_improvement_db
        )
        if improved:
            best_validation_rmse = validation_metrics["rmse_db"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_validation_rmse,
            epochs_without_improvement,
            model_config,
            train_config,
            normalization,
        )
        _atomic_torch_save(payload, last_path)
        if improved:
            _atomic_torch_save(payload, best_path)

        LOGGER.info(
            "Epoch %03d/%03d | train=%.5f | val RMSE=%.4f dB | "
            "MAE=%.4f dB | R2=%.6f | %.1f s%s",
            epoch,
            train_config.epochs,
            row["train_total_loss"],
            validation_metrics["rmse_db"],
            validation_metrics["mae_db"],
            validation_metrics["r2"],
            elapsed,
            " | best" if improved else "",
        )
        if epochs_without_improvement >= train_config.early_stopping_patience:
            LOGGER.info(
                "Early stopping after %d epochs without improvement.",
                epochs_without_improvement,
            )
            break

    if not best_path.is_file():
        raise RuntimeError("Training did not produce best_model.pt.")
    checkpoint = _load_checkpoint(best_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return pd.DataFrame(history_rows), int(checkpoint["epoch"])


def _range_metric_rows(
    truth_db: np.ndarray,
    prediction_db: np.ndarray,
    range_km: np.ndarray,
) -> list[dict]:
    error = prediction_db.astype(np.float64) - truth_db.astype(np.float64)
    rows = []
    for index, range_value in enumerate(range_km):
        values = error[:, index]
        rows.append(
            {
                "range_km": float(range_value),
                "rmse_db": float(np.sqrt(np.mean(values * values))),
                "mae_db": float(np.mean(np.abs(values))),
                "bias_db": float(np.mean(values)),
                "p95_absolute_error_db": float(np.percentile(np.abs(values), 95)),
            }
        )
    return rows


def _summary_metric_rows(
    truth_db: np.ndarray,
    prediction_db: np.ndarray,
    range_km: np.ndarray,
) -> list[dict]:
    bands = {
        "all_0.2_to_100_km": np.ones(range_km.size, dtype=bool),
        "near_0.2_to_20_km": range_km <= 20.0,
        "middle_20.2_to_60_km": (range_km > 20.0) & (range_km <= 60.0),
        "far_60.2_to_100_km": range_km > 60.0,
    }
    rows = []
    for name, selected in bands.items():
        row = {"scope": "range_band", "category": name}
        row.update(_metrics(truth_db[:, selected], prediction_db[:, selected]))
        rows.append(row)
    return rows


def _group_metric_rows(
    truth_db: np.ndarray,
    prediction_db: np.ndarray,
    validation_rows: np.ndarray,
    manifest: pd.DataFrame,
) -> list[dict]:
    indexed = manifest.set_index("pe_row_index_0based", drop=False)
    metadata = indexed.loc[validation_rows].reset_index(drop=True)
    if not metadata["role_split_name"].astype(str).eq("forward_validation").all():
        raise ValueError("Non-forward-validation sample reached validation metrics.")
    dimensions = (
        "path_class_name",
        "height_relation_class_name",
        "f_below_bin",
        "month",
    )
    if "month" not in metadata.columns:
        metadata["month"] = pd.to_datetime(
            metadata["valid_time_utc"], utc=True
        ).dt.strftime("%Y-%m")
    rows = []
    for dimension in dimensions:
        for category in sorted(metadata[dimension].astype(str).unique()):
            selected = metadata[dimension].astype(str).eq(category).to_numpy()
            row = {"dimension": dimension, "category": category}
            row.update(_metrics(truth_db[selected], prediction_db[selected]))
            rows.append(row)
    return rows


def _configure_matplotlib():
    cache = Path(tempfile.gettempdir()) / "matplotlib-forward-rescnn"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_training_history(history: pd.DataFrame, output_path: Path) -> None:
    plt = _configure_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(history["epoch"], history["train_total_loss"], label="training")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Composite loss")
    axes[0].set_title("(a) Training loss")
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        history["epoch"], history["validation_rmse_db"], color="#D55E00"
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation RMSE (dB)")
    axes[1].set_title("(b) Forward-validation error")
    axes[1].grid(alpha=0.25)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_range_errors(range_table: pd.DataFrame, output_path: Path) -> None:
    plt = _configure_matplotlib()
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.plot(range_table["range_km"], range_table["rmse_db"], label="RMSE")
    axis.plot(range_table["range_km"], range_table["mae_db"], label="MAE")
    axis.axvline(20.0, color="0.5", linestyle="--", linewidth=1)
    axis.axvline(60.0, color="0.5", linestyle="--", linewidth=1)
    axis.set_xlabel("Range (km)")
    axis.set_ylabel("Error (dB)")
    axis.set_title("1D-ResCNN error versus range (forward validation only)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _select_example_positions(
    validation_rows: np.ndarray,
    manifest: pd.DataFrame,
    maximum_examples: int,
) -> np.ndarray:
    indexed = manifest.set_index("pe_row_index_0based", drop=False)
    metadata = indexed.loc[validation_rows].reset_index(drop=True)
    selected = []
    for column in ("path_class_name", "height_relation_class_name"):
        for category in sorted(metadata[column].astype(str).unique()):
            candidates = np.flatnonzero(metadata[column].astype(str).eq(category))
            if candidates.size:
                selected.append(int(candidates[0]))
            if len(selected) >= maximum_examples:
                break
        if len(selected) >= maximum_examples:
            break
    for position in range(len(metadata)):
        if len(selected) >= maximum_examples:
            break
        if position not in selected:
            selected.append(position)
    return np.asarray(selected[:maximum_examples], dtype=np.int64)


def save_prediction_example_data(
    truth_db: np.ndarray,
    prediction_db: np.ndarray,
    validation_rows: np.ndarray,
    manifest: pd.DataFrame,
    range_km: np.ndarray,
    output_path: Path,
) -> None:
    positions = _select_example_positions(validation_rows, manifest, 6)
    indexed = manifest.set_index("pe_row_index_0based", drop=False)
    metadata = indexed.loc[validation_rows].reset_index(drop=True)
    titles = []
    for position in positions:
        path_label = str(metadata.loc[position, "path_class_name"])
        height_label = str(metadata.loc[position, "height_relation_class_name"])
        titles.append(f"{path_label}\n{height_label}")
    np.savez_compressed(
        output_path,
        truth_db=np.asarray(truth_db[positions], dtype=np.float32),
        prediction_db=np.asarray(prediction_db[positions], dtype=np.float32),
        range_km=np.asarray(range_km, dtype=np.float32),
        titles=np.asarray(titles, dtype="U160"),
    )


def plot_prediction_examples(
    truth_db: np.ndarray,
    prediction_db: np.ndarray,
    range_km: np.ndarray,
    titles: np.ndarray,
    output_path: Path,
) -> None:
    plt = _configure_matplotlib()
    figure, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    for axis, truth, prediction, title in zip(
        axes.flat, truth_db, prediction_db, titles
    ):
        axis.plot(range_km, truth, label="PE", linewidth=1.2)
        axis.plot(
            range_km,
            prediction,
            label="1D-ResCNN",
            linewidth=1.0,
        )
        axis.set_title(str(title), fontsize=9)
        axis.set_xlabel("Range (km)")
        axis.set_ylabel("One-way $L_s$ (dB)")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False)
    figure.suptitle("Forward-validation prediction examples", fontsize=14)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def render_saved_results(output_directory: Path) -> None:
    """Render plots in a process that never imports PyTorch.

    This process isolation avoids the Windows libiomp5md.dll collision between
    PyTorch and Matplotlib/compiled numerical dependencies.
    """
    output_directory = Path(output_directory)
    history_file = output_directory / "training_history.csv"
    range_file = output_directory / "validation_by_range.csv"
    example_file = output_directory / "validation_example_data.npz"
    for required in (history_file, range_file, example_file):
        if not required.is_file():
            raise FileNotFoundError(required)
    history = pd.read_csv(history_file, encoding="utf-8-sig")
    range_table = pd.read_csv(range_file, encoding="utf-8-sig")
    with np.load(example_file, allow_pickle=False) as examples:
        truth_db = np.asarray(examples["truth_db"])
        prediction_db = np.asarray(examples["prediction_db"])
        range_km = np.asarray(examples["range_km"])
        titles = np.asarray(examples["titles"])
    plot_training_history(history, output_directory / "training_curves.png")
    plot_range_errors(range_table, output_directory / "range_error_curves.png")
    plot_prediction_examples(
        truth_db,
        prediction_db,
        range_km,
        titles,
        output_directory / "validation_prediction_examples.png",
    )
    print(f"Plot rendering complete: {output_directory}")


def render_in_isolated_process(output_directory: Path) -> bool:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--render-results",
        str(Path(output_directory).resolve()),
    ]
    result = subprocess.run(command, check=False)
    return result.returncode == 0


def measure_inference_time(model, device, repeats: int = 300) -> float:
    torch, _, _ = _require_torch()
    model.eval()
    edh = torch.full((1, 101), 12.0, dtype=torch.float32, device=device)
    mask = torch.ones((1, 101), dtype=torch.float32, device=device)
    with torch.inference_mode():
        for _ in range(20):
            model(edh, mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(edh, mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - start) / repeats


def _prepare_output_directory(
    requested: Path,
    smoke_test: bool,
    overwrite: bool,
    resume: Optional[Path],
) -> Path:
    requested = Path(requested)
    if smoke_test and requested == DEFAULT_OUTPUT_DIRECTORY:
        requested = DEFAULT_SMOKE_OUTPUT_DIRECTORY
    requested.mkdir(parents=True, exist_ok=True)
    protected = requested / "training_history.csv"
    if protected.exists() and not overwrite and resume is None:
        raise FileExistsError(
            f"{requested} already contains a training run. Use --resume with "
            "last_checkpoint.pt, choose another --output-dir, or use --overwrite."
        )
    return requested


def run_training(args: argparse.Namespace) -> Mapping[str, Path]:
    torch, _, _ = _require_torch()
    set_reproducibility(args.seed)
    output_directory = _prepare_output_directory(
        args.output_dir, args.smoke_test, args.overwrite, args.resume
    )
    pe = load_pe_arrays(args.pe_mat)
    train_indices, validation_indices, manifest = load_forward_indices(
        args.split_directory, pe["edh_m"].shape[0]
    )
    if args.smoke_test:
        rng = np.random.default_rng(args.seed)
        train_indices = np.sort(
            rng.choice(
                train_indices,
                size=min(args.smoke_train_samples, train_indices.size),
                replace=False,
            )
        )
        validation_indices = np.sort(
            rng.choice(
                validation_indices,
                size=min(args.smoke_validation_samples, validation_indices.size),
                replace=False,
            )
        )
        LOGGER.info(
            "Smoke-test subset: train=%d, validation=%d.",
            train_indices.size,
            validation_indices.size,
        )

    normalization = compute_normalization(
        pe["edh_m"], pe["duct_mask"], pe["loss_db"], train_indices
    )
    np.savez_compressed(
        output_directory / "normalization.npz",
        edh_scale_m=np.asarray(normalization.edh_scale_m),
        target_mean_db=normalization.target_mean_db,
        target_std_db=normalization.target_std_db,
    )
    model_config = ModelConfig(
        width=args.width,
        dropout=args.dropout,
        dilations=tuple(args.dilations),
    )
    train_config = TrainConfig(
        epochs=args.smoke_epochs if args.smoke_test else args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_loss_weight=args.gradient_loss_weight,
        gradient_scale_db=args.gradient_scale_db,
        huber_beta=args.huber_beta,
        early_stopping_patience=args.patience,
        minimum_improvement_db=args.minimum_improvement_db,
        workers=args.workers,
        seed=args.seed,
        use_amp=not args.no_amp,
    )
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        LOGGER.info("Using CUDA device: %s", torch.cuda.get_device_name(device))
    else:
        LOGGER.warning("CUDA is unavailable or disabled; training will use CPU.")

    train_dataset = make_dataset(
        pe["edh_m"],
        pe["duct_mask"],
        pe["loss_db"],
        train_indices,
        normalization,
    )
    validation_dataset = make_dataset(
        pe["edh_m"],
        pe["duct_mask"],
        pe["loss_db"],
        validation_indices,
        normalization,
    )
    train_loader, validation_loader = make_loaders(
        train_dataset, validation_dataset, train_config, device
    )
    model = build_model(model_config, normalization.edh_scale_m).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    LOGGER.info(
        "1D-ResCNN parameters: %s; output directory: %s",
        f"{parameter_count:,}",
        output_directory,
    )
    history, best_epoch = train_model(
        model,
        train_loader,
        validation_loader,
        device,
        normalization,
        model_config,
        train_config,
        output_directory,
        args.resume,
    )
    validation_metrics, truth_db, prediction_db, validation_rows = evaluate_model(
        model, validation_loader, device, normalization
    )
    if not np.array_equal(validation_rows, validation_indices):
        raise ValueError("Validation prediction order differs from role indices.")

    summary_rows = _summary_metric_rows(
        truth_db, prediction_db, pe["output_range_km"]
    )
    summary_table = pd.DataFrame(summary_rows)
    range_table = pd.DataFrame(
        _range_metric_rows(truth_db, prediction_db, pe["output_range_km"])
    )
    group_table = pd.DataFrame(
        _group_metric_rows(
            truth_db, prediction_db, validation_rows, manifest
        )
    )
    summary_table.to_csv(
        output_directory / "validation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    range_table.to_csv(
        output_directory / "validation_by_range.csv",
        index=False,
        encoding="utf-8-sig",
    )
    group_table.to_csv(
        output_directory / "validation_by_group.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_prediction_example_data(
        truth_db,
        prediction_db,
        validation_rows,
        manifest,
        pe["output_range_km"],
        output_directory / "validation_example_data.npz",
    )
    # Finish every numerical result before launching the isolated plot process.
    inference_ms = measure_inference_time(model, device)

    configuration = {
        "script_version": SCRIPT_VERSION,
        "model": asdict(model_config),
        "training": asdict(train_config),
        "device": str(device),
        "parameter_count": int(parameter_count),
        "pe_mat": str(Path(args.pe_mat).resolve()),
        "split_directory": str(Path(args.split_directory).resolve()),
        "output_directory": str(output_directory.resolve()),
        "forward_train_samples": int(train_indices.size),
        "forward_validation_samples": int(validation_indices.size),
        "forbidden_roles_loaded": [],
        "normalization_scope": "forward_train only",
        "best_epoch": int(best_epoch),
        "single_sample_inference_ms": float(inference_ms),
    }
    (output_directory / "run_config.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_lines = [
        "FORWARD 1D-RESCNN TRAINING SUMMARY",
        "=================================",
        f"Script version: {SCRIPT_VERSION}",
        f"PE mapping: d(r), 101 nodes -> L_s(r,z_s=0.3m), 500 nodes",
        f"Training role: forward_train ({train_indices.size} samples)",
        f"Validation role: forward_validation ({validation_indices.size} samples)",
        "Inverse roles loaded: no",
        "Joint test loaded: no",
        f"Best epoch: {best_epoch}",
        f"Parameters: {parameter_count:,}",
        f"Validation RMSE: {validation_metrics['rmse_db']:.6f} dB",
        f"Validation MAE: {validation_metrics['mae_db']:.6f} dB",
        f"Validation R2: {validation_metrics['r2']:.8f}",
        f"Validation bias: {validation_metrics['bias_db']:.6f} dB",
        f"Median sample RMSE: {validation_metrics['median_sample_rmse_db']:.6f} dB",
        f"P95 sample RMSE: {validation_metrics['p95_sample_rmse_db']:.6f} dB",
        f"Single-sample inference: {inference_ms:.6f} ms ({device})",
        f"Output directory: {output_directory.resolve()}",
        "",
        "Do not evaluate joint_test until the architecture, loss and training",
        "settings have been frozen. This run is a forward baseline only.",
    ]
    (output_directory / "training_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    plotting_complete = render_in_isolated_process(output_directory)
    with (output_directory / "training_summary.txt").open(
        "a", encoding="utf-8"
    ) as stream:
        stream.write(
            "Plot rendering in isolated process: "
            + ("passed\n" if plotting_complete else "failed; numerical outputs preserved\n")
        )
    if not plotting_complete:
        LOGGER.warning(
            "Plot subprocess failed, but model weights, metrics and summary were "
            "saved successfully. Re-run with --render-results %s.",
            output_directory,
        )
    LOGGER.info(
        "Training complete: validation RMSE=%.4f dB, MAE=%.4f dB, R2=%.6f.",
        validation_metrics["rmse_db"],
        validation_metrics["mae_db"],
        validation_metrics["r2"],
    )
    return {
        "best_model": output_directory / "best_model.pt",
        "summary": output_directory / "training_summary.txt",
        "metrics": output_directory / "validation_metrics.csv",
        "group_metrics": output_directory / "validation_by_group.csv",
        "range_metrics": output_directory / "validation_by_range.csv",
        "training_plot": output_directory / "training_curves.png",
        "range_plot": output_directory / "range_error_curves.png",
        "examples": output_directory / "validation_prediction_examples.png",
    }


def self_test() -> None:
    torch, _, _ = _require_torch()
    set_reproducibility(20260812)
    model_config = ModelConfig(width=16, dilations=(1, 2, 4), dropout=0.0)
    model = build_model(model_config, edh_scale_m=20.0)
    edh = torch.linspace(5.0, 20.0, 101).repeat(4, 1)
    mask = torch.ones_like(edh)
    edh[1, 40:55] = 0.0
    mask[1, 40:55] = 0.0
    prediction = model(edh, mask)
    if tuple(prediction.shape) != (4, 500):
        raise AssertionError(f"Unexpected model output shape: {prediction.shape}")
    if not torch.isfinite(prediction).all():
        raise AssertionError("Model self-test produced a non-finite output.")
    target = torch.zeros_like(prediction)
    mean = torch.linspace(110.0, 160.0, 500)
    std = torch.full((500,), 8.0)
    config = TrainConfig(epochs=1, batch_size=4)
    loss, point, gradient = composite_loss(
        prediction, target, mean, std, config
    )
    loss.backward()
    if not all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise AssertionError("Model self-test produced a non-finite gradient.")
    print(
        "Self-test passed: 101-to-500 range mapping, no-duct mask channel, "
        f"1D-ResCNN forward/backward, point loss={float(point):.6f}, "
        f"gradient loss={float(gradient):.6f}."
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pe_mat", type=Path, nargs="?")
    parser.add_argument("split_directory", type=Path, nargs="?")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-loss-weight", type=float, default=0.10)
    parser.add_argument("--gradient-scale-db", type=float, default=5.0)
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--minimum-improvement-db", type=float, default=1.0e-4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--dilations",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8, 16, 32, 64, 128),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Keep at 0: in-memory datasets and Windows-safe deterministic loading.",
    )
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=3)
    parser.add_argument("--smoke-train-samples", type=int, default=512)
    parser.add_argument("--smoke-validation-samples", type=int, default=256)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--render-results",
        type=Path,
        help="Render an existing run in a process that does not import PyTorch.",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.self_test or args.render_results is not None:
        return
    if args.pe_mat is None or args.split_directory is None:
        raise SystemExit(
            "pe_mat and split_directory are required unless --self-test is used."
        )
    positive_integer_names = (
        "epochs",
        "batch_size",
        "patience",
        "width",
        "smoke_epochs",
        "smoke_train_samples",
        "smoke_validation_samples",
    )
    for name in positive_integer_names:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.workers != 0:
        raise ValueError(
            "This Windows-safe in-memory baseline requires --workers 0."
        )
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay nonnegative.")
    if args.gradient_loss_weight < 0 or args.gradient_scale_db <= 0:
        raise ValueError("Gradient-loss settings are invalid.")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0,1).")
    if any(value < 1 for value in args.dilations):
        raise ValueError("Every dilation must be positive.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    _validate_args(args)
    if args.render_results is not None:
        render_saved_results(args.render_results)
    elif args.self_test:
        self_test()
    else:
        run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
