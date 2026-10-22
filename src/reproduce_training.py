"""Print frozen development training plans; execute only with --execute.

No plan opens joint_test or chooses settings from test performance. Retraining
need not be bitwise identical across hardware or CUDA versions. Original frozen
checkpoints and histories remain the evidence for the manuscript.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from release_paths import DATA_ROOT, OUTPUT_ROOT, SOURCE_ROOT

SETTINGS = {"supervised": (0., 5.), "nominal": (.01, 5.),
            "lambda_0p005_scale_5": (.005, 5.), "lambda_0p02_scale_5": (.02, 5.),
            "lambda_0p01_scale_2p5": (.01, 2.5), "lambda_0p01_scale_10": (.01, 10.)}


def inverse(pc, forward, output, seed, weight, scale, fixed=False):
    source = "train_inverse_physics_radar_ablation_v0_7.py" if fixed else "train_inverse_baseline_v2.py"
    command = [sys.executable, str(SOURCE_ROOT / source), "--pc-dir", str(pc), "--forward-checkpoint", str(forward),
               "--output-dir", str(output), "--seed", str(seed), "--physics-loss-weight", str(weight),
               "--physics-scale-db", str(scale)]
    if fixed:
        command += ["--physics-radar-mode", "fixed_reference"]
    return command


def plan(suite, data, output):
    commands = []
    forward = data / "models/forward_rescnn/best_model.pt"
    if suite in {"nominal", "sensitivity", "fixed-reference"}:
        settings = SETTINGS if suite == "sensitivity" else {"supervised": (0., 5.), "nominal": (.01, 5.)}
        if suite == "fixed-reference":
            settings = {"fixed_reference": (.01, 5.)}
        for seed in ((20260813, 20260814, 20260815) if suite == "sensitivity" else (20260813,)):
            for name, (weight, scale) in settings.items():
                commands.append(inverse(data / "pc_dataset_v1", forward, output / f"seed_{seed}" / name,
                                        seed, weight, scale, suite == "fixed-reference"))
    elif suite == "spatial":
        for direction in ("south_to_north", "north_to_south"):
            inputs = data / "spatial_inputs" / direction
            for seed in (20260908, 20260909, 20260910):
                run = output / direction / f"seed_{seed}"
                regional_f = run / "forward"
                commands.append([sys.executable, str(SOURCE_ROOT / "train_forward_rescnn.py"),
                                 str(inputs / "forward_development.h5"), str(inputs / "forward_split"),
                                 "--output-dir", str(regional_f), "--seed", str(seed)])
                for name, weight in (("supervised", 0.), ("physics", .01)):
                    commands.append(inverse(inputs / "pc", regional_f / "best_model.pt", run / name, seed, weight, 5.))
    elif suite == "forward":
        commands.append([sys.executable, str(SOURCE_ROOT / "train_forward_rescnn.py"),
                         str(data / "pe_reference/pe_ls_dataset_zs_0p3m_v2.mat"), str(data / "model_role_split"),
                         "--output-dir", str(output / "forward"), "--seed", "20260812"])
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("nominal", "sensitivity", "spatial", "forward", "fixed-reference"), default="nominal")
    parser.add_argument("--data", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "retraining")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    data, output = args.data.resolve(), (args.output / args.suite).resolve()
    if output == data or data in output.parents:
        parser.error("Training outputs must be outside the frozen data directory.")
    commands = plan(args.suite, data, output)
    print(json.dumps({"suite": args.suite, "execute": args.execute, "joint_test_access": False,
                      "commands": commands}, indent=2))
    if args.execute:
        if output.exists():
            parser.error("Refusing to overwrite an existing output tree; choose a new --output.")
        env = os.environ.copy()
        env["PHYSICS_RESCNN_DATA"] = str(data)
        for command in commands:
            subprocess.run(command, env=env, cwd=SOURCE_ROOT.parent, check=True)


if __name__ == "__main__":
    main()
