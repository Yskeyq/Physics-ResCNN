"""Clutter-only inference using the frozen model; no targets, PE, or forward model."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

from release_paths import MODELS_ROOT, OUTPUT_ROOT, SOURCE_ROOT

MODULES = {
    "physics_rescnn": ("train_inverse_baseline_v2.py", "inverse_physics_w0p0"),
    "supervised_rescnn": ("train_inverse_baseline_v2.py", "inverse_baseline_v2"),
    "cnn": ("train_inverse_cnn.py", "inverse_cnn"),
    "bilstm": ("train_inverse_bilstm.py", "inverse_bilstm"),
    "transformer": ("train_inverse_transformer.py", "inverse_transformer"),
    "rfcformer_adapted": ("train_inverse_rfcformer_adapted.py", "inverse_rfcformer_adapted_v2_low_dropout"),
}


def load_module(filename):
    name = "release_predict_" + Path(filename).stem
    spec = importlib.util.spec_from_file_location(name, SOURCE_ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def predict(pc_centered_db, checkpoint_path, model_id="physics_rescnn", device="cpu", batch_size=128):
    if model_id not in MODULES:
        raise ValueError("Unsupported model. Fixed-radar input diagnostics are not clutter-only deployment models.")
    values = np.asarray(pc_centered_db, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 500 or not np.isfinite(values).all():
        raise ValueError("Expected finite centered-clutter values of shape N x 500, in dB.")
    if len(values) == 0 or batch_size < 1:
        raise ValueError("At least one profile and a positive batch size are required.")
    if np.max(np.abs(values.mean(axis=1, dtype=np.float64))) > 1e-3:
        raise ValueError("Input is not profile-centered. Subtract each 500-point profile mean before calling predict.")
    # The release checkpoints contain only tensors and primitive metadata.
    # Never deserialize untrusted third-party pickle files.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    normalization = checkpoint["normalization"]
    if str(normalization.get("input_mode", "era5")) != "era5":
        raise ValueError("This checkpoint expects regenerated diagnostic inputs, not ordinary clutter.")
    module = load_module(MODULES[model_id][0])
    config = module.ModelConfig(**dict(checkpoint["model_config"]))
    model = module.build_inverse_model(config, float(normalization["edh_scale_m"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    target_device = torch.device(device)
    model.to(target_device).eval()
    mean = np.asarray(normalization["pc_mean_dB"], dtype=np.float32)
    std = np.asarray(normalization["pc_std_dB"], dtype=np.float32)
    if mean.shape != (500,) or std.shape != (500,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid checkpoint input normalization.")
    heights, masks, probabilities = [], [], []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            x = torch.from_numpy((values[start:start + batch_size] - mean) / std).to(target_device)
            result = model(x)
            probability = torch.sigmoid(result["mask_logits"])
            mask = probability >= .5
            heights.append((result["edh_m"] * mask.to(result["edh_m"].dtype)).cpu().numpy())
            masks.append(mask.cpu().numpy().astype(np.uint8))
            probabilities.append(probability.cpu().numpy())
    return {"edh_m": np.concatenate(heights), "duct_mask": np.concatenate(masks),
            "duct_probability": np.concatenate(probabilities), "range_km": np.arange(101, dtype=np.float32)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="NPZ with pc_centered_dB, N x 500.")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "predictions.npz")
    parser.add_argument("--model", choices=tuple(MODULES), default="physics_rescnn")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output path; existing predictions are not overwritten.")
    checkpoint = args.checkpoint or MODELS_ROOT / MODULES[args.model][1] / "best_model.pt"
    with np.load(args.input, allow_pickle=False) as source:
        result = predict(source["pc_centered_dB"], checkpoint, args.model, args.device, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print(json.dumps({"profiles": len(result["edh_m"]), "output_nodes": 101,
                      "uses_forward_model_or_pe": False, "output": str(args.output)}))


if __name__ == "__main__":
    main()
