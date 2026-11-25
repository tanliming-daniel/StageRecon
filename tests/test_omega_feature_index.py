"""Regressions for persisted Omega feature coordinates and cache invalidation."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
from tools.omega_feature_index import save_index, load_index


class OmegaFeatureIndexTests(unittest.TestCase):
    def reference_sets(self):
        rng = np.random.default_rng(31)
        row = ("ref.jpg", rng.random((20, 2)), rng.random((20, 128)).astype(np.float32),
               rng.random((20, 3)))
        return {"sift": [row], "disk": [(row[0], *(a.copy() for a in row[1:]))]}

    def test_serialized_descriptors_keep_corresponding_world_points(self):
        original = self.reference_sets()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "index.npz"
            save_index(path, original, {"prediction_sha256": "a"})
            loaded, _ = load_index(path, {"prediction_sha256": "a"})
            for backend in original:
                self.assertEqual(loaded[backend][0][0], "ref.jpg")
                for field in range(1, 4):
                    np.testing.assert_array_equal(loaded[backend][0][field], original[backend][0][field])

    def test_updated_map_rejects_old_index(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "index.npz"
            save_index(path, self.reference_sets(), {"prediction_sha256": "a"})
            with self.assertRaisesRegex(ValueError, "signature differs"):
                load_index(path, {"prediction_sha256": "b"})

    def test_omega_pose_is_independent_of_previous_opencv_rng_calls(self):
        import cv2
        from tools.omega_clip_localizer import OmegaPnPLocalizer
        rng = np.random.default_rng(18)
        xyz = rng.uniform([-1, -.6, 3], [1, .6, 7], size=(160, 3))
        intrinsic = np.array([[900., 0., 383.5], [0., 900., 215.5], [0., 0., 1.]])
        pixels = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), intrinsic, None)[0][:, 0]
        pixels += rng.normal(0, .15, pixels.shape)
        refs = np.asarray(["a", "b"] * 80)
        cv2.setRNGSeed(1)
        first = OmegaPnPLocalizer.solve(xyz, pixels, refs)
        cv2.setRNGSeed(999)
        second = OmegaPnPLocalizer.solve(xyz, pixels, refs)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        for a, b in zip(first[:3], second[:3]):
            np.testing.assert_allclose(a, b, atol=1e-10)
        self.assertEqual(first[3], second[3])

    def test_nonfinite_world_points_rejected(self):
        sets = self.reference_sets()
        sets["disk"][0][3][0, 0] = np.nan
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "index.npz"
            save_index(path, sets, {})
            with self.assertRaisesRegex(ValueError, "Invalid disk"):
                load_index(path, {})


if __name__ == "__main__":
    unittest.main()
