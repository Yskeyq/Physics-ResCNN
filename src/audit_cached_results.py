"""Reaggregate frozen caches. Never trains, selects a model, or calls a PE solver."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

from release_paths import DATA_ROOT


def cosine(a, b):
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def directional(root):
    folder = root / "records/directional_audit"
    protocol = json.loads((folder / "PROTOCOL.json").read_text(encoding="utf-8"))
    inp = loadmat(folder / "input.mat")
    net = np.load(folder / "surrogate.npz", allow_pickle=False)
    std, gate = inp["pc_std"].reshape(-1), protocol["gate"]
    records = []
    def centered(v):
        return -2 * (v - v.mean())
    for i, sample in enumerate(protocol["samples"]):
        pe = loadmat(folder / f"pe_sample_{i+1:02d}.mat")
        assert pe["replay_rmse"].item() <= gate["nominal_ls_max_rmse_db"]
        response = pe["response"]
        for j, name in enumerate(protocol["directions"]):
            raw = centered((response[j, 0, 1] - response[j, 0, 0]) / .2)
            a = raw / std
            b = centered((response[j, 1, 1] - response[j, 1, 0]) / .5) / std
            pred = centered(net["derivative_ls"][i, j]) / std
            stable_cos = cosine(a, b)
            stable_rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))
            rms = float(np.sqrt(np.mean(raw * raw)))
            stable = (stable_cos >= gate["stability_cosine_min"]
                      and stable_rel <= gate["stability_relative_difference_max"]
                      and rms >= gate["minimum_pe_derivative_rms_db_per_m"])
            overlap = len(set(np.argsort(np.abs(a))[-100:]) & set(np.argsort(np.abs(pred))[-100:])) / 100.
            records.append({**sample, "direction": name, "stable": stable, "PE_step_cosine": stable_cos,
                            "PE_step_relative_difference": stable_rel, "PE_derivative_rms_db_per_m": rms,
                            "signed_cosine": cosine(pred, a), "absolute_cosine": cosine(np.abs(pred), np.abs(a)),
                            "relative_l2_error": float(np.linalg.norm(pred - a) / max(np.linalg.norm(a), 1e-30)),
                            "top20percent_overlap": overlap,
                            "surrogate_step_cosine": cosine(pred, centered(net["finite_difference_ls"][i, j, 0]) / std)})
    actual = pd.DataFrame(records)
    expected = pd.read_csv(folder / "derivative_metrics.csv")
    assert len(actual) == len(expected) == 32
    for key in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[key]) and key != "stable":
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-10, atol=1e-10)
        else:
            assert actual[key].tolist() == expected[key].tolist(), key
    summary = json.loads((folder / "RESULT.json").read_text(encoding="utf-8"))["roles"]
    values = {}
    for role, group in actual.groupby("role"):
        stable = group[group.stable]
        values[role] = {key: float(stable[key].median()) for key in ("signed_cosine", "relative_l2_error", "top20percent_overlap")}
        for key, value in values[role].items():
            np.testing.assert_allclose(value, summary[role][key], rtol=1e-10, atol=1e-10)
    return {"pairs": len(actual), "matches_all_original_per_direction_records": True, "role_medians": values}


def posterior(root):
    folder = root / "records/direct_pe_posterior"
    truth = loadmat(folder / "direct_pe_posterior_input_v0_7.mat")
    original = pd.read_csv(folder / "direct_pe_per_sample_metrics_v0_7.csv")
    np.testing.assert_array_equal(truth["sample_id"].reshape(-1), original.sample_id)
    result = {}
    with h5py.File(folder / "direct_pe_posterior_output_v0_7.mat", "r") as h:
        ids = ["".join(chr(c) for c in h[ref][...].reshape(-1)) for ref in h["model_ids"][...].reshape(-1)]
        for i, name in enumerate(ids):
            # MATLAB HDF5 dimensions are reversed: model, range, sample.
            residual = np.asarray(h["pc_residual_dB"][i], dtype=np.float64).T
            reconstructed = np.asarray(h["reconstructed_pc_centered_dB"][i], dtype=np.float64).T
            loss = np.asarray(h["predicted_ls_oneway_dB"][i], dtype=np.float64).T
            np.testing.assert_allclose(residual, reconstructed - truth["pc_centered_dB"], rtol=1e-4, atol=1e-5)
            pc = truth["radar_term_dB"] - 2 * loss
            pc -= pc.mean(axis=1, keepdims=True)
            np.testing.assert_allclose(pc, reconstructed, rtol=1e-4, atol=1e-4)
            per_sample = np.sqrt(np.mean(residual ** 2, axis=1))
            np.testing.assert_allclose(per_sample, original[name + "_pc_rmse_dB"], rtol=2e-6, atol=1e-6)
            pooled = float(np.sqrt(np.mean(residual ** 2)))
            pooled_csv = float(np.sqrt(np.mean(original[name + "_pc_rmse_dB"].to_numpy() ** 2)))
            np.testing.assert_allclose(pooled, pooled_csv, rtol=2e-6, atol=1e-6)
            result[name] = {"pooled_rmse_dB": pooled, "profiles": len(residual)}
    return {"uses_stored_sample_specific_radar_term": True, "all_cached_traces_match": True, "models": result}


def grouped(root, name, source, target, groups, metrics, sd_suffix):
    folder = root / "records" / name
    rows, means = pd.read_csv(folder / source), pd.read_csv(folder / target)
    count = 0
    for keys, group in rows.groupby(groups):
        keys = keys if isinstance(keys, tuple) else (keys,)
        selected = means
        for key, value in zip(groups, keys):
            selected = selected[selected[key] == value]
        assert len(selected) == 1 and len(group) == 3
        for metric in metrics:
            for suffix, value in (("_mean", group[metric].mean()), (sd_suffix, group[metric].std(ddof=1))):
                np.testing.assert_allclose(value, selected.iloc[0][metric + suffix], rtol=1e-9, atol=1e-12)
        count += 1
    return {"run_records": len(rows), "groups": count, "three_seeds_per_group": True, "original_mean_and_sample_sd_match": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, help="Optional new JSON file; existing files are never overwritten")
    args = parser.parse_args()
    result = {"directional_audit": directional(args.data), "direct_pe_posterior": posterior(args.data),
              "sensitivity": grouped(args.data, "sensitivity", "all_seed_validation.csv", "mean_sd_validation.csv",
                                     ["setting"], ["rmse_m", "mae_m", "positive_point_rmse_m", "duct_mask_f1"], "_sample_sd"),
              "spatial": grouped(args.data, "spatial_evaluation", "all_seed_metrics.csv", "seed_summary.csv",
                                 ["direction", "cohort", "model"], ["rmse_m", "mae_m", "p95_profile_rmse_m", "mask_f1"], "_std"),
              "new_training": False, "new_pe_simulation": False, "fresh_test_inference": False,
              "reaggregation_of_frozen_records_only": True}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
