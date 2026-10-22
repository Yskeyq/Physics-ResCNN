"""Small, development-only release checks; never train or open joint_test."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))
from release_paths import DATA_ROOT, MODELS_ROOT
import predict
import train_inverse_baseline_v2 as core


class ReleaseTests(unittest.TestCase):
    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            predict.predict(np.zeros((1, 499)), "unused.pt")

    def test_nonfinite_validation(self):
        x = np.zeros((1, 500), dtype=np.float32)
        x[0, 0] = np.nan
        with self.assertRaises(ValueError):
            predict.predict(x, "unused.pt")

    def test_centering_validation(self):
        with self.assertRaises(ValueError):
            predict.predict(np.ones((1, 500)), "unused.pt")

    def test_no_silent_fixed_radar_deployment(self):
        with self.assertRaises(ValueError):
            predict.predict(np.zeros((1, 500)), "unused.pt", "fixed_radar_rescnn")

    def test_training_role_contract(self):
        self.assertEqual(core.ROLES, ("inverse_train", "inverse_validation"))

    @unittest.skipUnless((DATA_ROOT / "pc_dataset_v1/pc_inverse_validation.h5").is_file(), "Separate dataset not installed")
    def test_validation_alignment_and_inference(self):
        torch.set_num_threads(2)
        data = core.load_role_arrays(DATA_ROOT / "pc_dataset_v1", "inverse_validation", need_radar=True)
        self.assertEqual(data["pc"].shape, (1778, 500))
        self.assertEqual(data["edh"].shape, (1778, 101))
        self.assertEqual(len(np.unique(data["sample_id"])), 1778)
        selected = data["pc"][:8]
        output = predict.predict(selected, MODELS_ROOT / "inverse_physics_w0p0/best_model.pt")
        self.assertEqual(output["edh_m"].shape, (8, 101))
        self.assertTrue(np.isfinite(output["edh_m"]).all())
        self.assertTrue(np.all(output["edh_m"][output["duct_mask"] == 0] == 0))
        checkpoint = torch.load(MODELS_ROOT / "inverse_physics_w0p0/best_model.pt", map_location="cpu", weights_only=True)
        norm = checkpoint["normalization"]
        model = core.build_inverse_model(core.ModelConfig(**checkpoint["model_config"]), float(norm["edh_scale_m"]))
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        x = (selected - np.asarray(norm["pc_mean_dB"])) / np.asarray(norm["pc_std_dB"])
        with torch.inference_mode():
            raw = model(torch.from_numpy(x))
            expected = (raw["edh_m"] * (torch.sigmoid(raw["mask_logits"]) >= .5)).numpy()
        np.testing.assert_array_equal(output["edh_m"], expected)


if __name__ == "__main__":
    unittest.main()
