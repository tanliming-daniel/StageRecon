"""Regressions for telephoto focal search and camera endpoint quality."""
import unittest
from tools.pose_quality import assess_pose_quality, require_usable_anchor
try:
    import cv2
    import numpy as np
    from tools.pnp_focal import pnp_with_focal_search
except ModuleNotFoundError:
    cv2 = None


class CameraQualityTests(unittest.TestCase):
    def pose(self, count=60):
        return {'pnp': {'inlier_count': count, 'inlier_rmse_pixels': 2.0}}

    def test_boundary_solution_is_rejected_despite_many_inliers(self):
        pose = self.pose(200)
        pose['pnp']['focal_search'] = {'at_search_boundary': True}
        self.assertEqual(assess_pose_quality(pose)['status'], 'rejected')
        with self.assertRaises(ValueError):
            require_usable_anchor(pose)

    def test_review_does_not_silently_anchor_depth(self):
        pose = self.pose(35)
        with self.assertRaises(ValueError):
            require_usable_anchor(pose)
        self.assertEqual(require_usable_anchor(pose, allow_review=True)['status'], 'review')

    def test_small_background_patch_requires_review(self):
        pose = self.pose(100)
        pose['pnp']['inlier_spatial_support'] = {'hull_fraction': .02}
        self.assertEqual(assess_pose_quality(pose)['status'], 'review')
        with self.assertRaises(ValueError):
            require_usable_anchor(pose)

    def test_accepted_endpoint_passes(self):
        self.assertEqual(require_usable_anchor(self.pose())['status'], 'accepted')


@unittest.skipIf(cv2 is None, 'PnP tests require numpy and OpenCV')
class FocalSearchTests(unittest.TestCase):
    def test_recovers_telephoto_focal_beyond_old_grid(self):
        rng = np.random.default_rng(3)
        xyz = rng.uniform([-.4, -.25, 3.], [.4, .25, 7.], size=(160, 3))
        focal = 5872.
        k = np.array([[focal, 0., 640.], [0., focal, 360.], [0., 0., 1.]])
        xy = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), k, None)[0].reshape(-1, 2)
        xy += rng.normal(0, .15, xy.shape)
        cv2.setRNGSeed(7)
        result, _ = pnp_with_focal_search(xyz, xy, (1280, 720), [1024., 1536., 2048., 2688.], 3.)
        self.assertLess(abs(result['focal']/focal - 1), .01)
        r = cv2.Rodrigues(np.array(result['rotation_vector']))[0]
        center = -r.T @ np.array(result['translation'])
        self.assertLess(np.linalg.norm(center), .04)

    def test_frontoparallel_plane_does_not_return_arbitrary_depth(self):
        rng = np.random.default_rng(5)
        xyz = rng.uniform([-.4, -.25, 5.], [.4, .25, 5.], size=(80, 3))
        k = np.array([[5872., 0., 640.], [0., 5872., 360.], [0., 0., 1.]])
        xy = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), k, None)[0].reshape(-1, 2)
        with self.assertRaisesRegex(RuntimeError, 'underconstrained'):
            pnp_with_focal_search(xyz, xy, (1280, 720), [1024., 1536., 2048., 2688.], 3.)

    def test_single_fixed_focal_is_respected(self):
        rng = np.random.default_rng(4)
        xyz = rng.uniform([-.4, -.25, 3.], [.4, .25, 7.], size=(50, 3))
        k = np.array([[5900., 0., 640.], [0., 5900., 360.], [0., 0., 1.]])
        xy = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), k, None)[0].reshape(-1, 2)
        result, _ = pnp_with_focal_search(xyz, xy, (1280, 720), [5900.], 3.)
        self.assertEqual(result['focal'], 5900.)
        self.assertEqual(result['focal_search']['evaluated_candidates'], 1)
