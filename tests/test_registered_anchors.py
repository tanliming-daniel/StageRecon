"""Verify source-frame mapping and safe reuse of existing camera calibration."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
try:
    import cv2
    import numpy as np
    from tools.registered_anchors import source_image_name, frame_identity, registered_anchor, intrinsics_between_anchors
    from tools.pose_quality import assess_pose_quality
except ModuleNotFoundError:
    cv2 = None


@unittest.skipIf(cv2 is None, 'Registered-camera tests require numpy/OpenCV')
class RegisteredAnchorTests(unittest.TestCase):
    def test_intrinsics_retain_principal_point_and_pixel_aspect(self):
        first=np.array([[800,0,310],[0,750,170],[0,0,1]],dtype=float)
        last=np.array([[1200,0,315],[0,1125,175],[0,0,1]],dtype=float)
        np.testing.assert_allclose(intrinsics_between_anchors(first,last,800,0),first)
        np.testing.assert_allclose(intrinsics_between_anchors(first,last,1200,1),last)
        middle=intrinsics_between_anchors(first,last,1000,.5)
        np.testing.assert_allclose(middle,[[1000,0,312.5],[0,937.5,172.5],[0,0,1]])

    def test_mapping_and_decoded_count(self):
        video = '/clips/S08E04/reference_clips/clean/general_0196_f00032018-f00032041.mp4'
        self.assertEqual(source_image_name(video, 23, 24), 'Friends_S08E04_00032041.jpg')
        with self.assertRaises(ValueError):
            source_image_name(video, 23, 25)
        with self.assertRaises(ValueError):
            source_image_name(video, 24, 24)

    def test_image_identity_accepts_resize_and_rejects_another_scene(self):
        rng = np.random.default_rng(72)
        a = rng.integers(0, 256, (180, 320, 3), dtype=np.uint8)
        b = rng.integers(0, 256, (180, 320, 3), dtype=np.uint8)
        self.assertTrue(frame_identity(cv2.resize(a, (160, 90)), a)['matches'])
        self.assertFalse(frame_identity(a, b)['matches'])

    def fixture(self, root):
        camera = SimpleNamespace(model='SIMPLE_PINHOLE', params=[300., 160., 90.], width=320, height=180)
        xyz = np.random.default_rng(1).uniform([-1, -.5, 4], [1, .5, 8], (25,3))
        xy = xyz[:, :2] / xyz[:, 2:3] * 300 + [160,90]
        image = SimpleNamespace(id=7, camera_id=3, tvec=np.zeros(3), point3D_ids=np.arange(25), xys=xy, qvec2rotmat=lambda: np.eye(3))
        pixels = np.random.default_rng(3).integers(0,256,(180,320,3),dtype=np.uint8)
        cv2.imwrite(str(root/'Friends_S08E04_00032018.jpg'), pixels)
        cv2.imwrite(str(root/'query.jpg'), cv2.resize(pixels,(160,90)))
        return {3:camera}, {'Friends_S08E04_00032018':image}, {i:SimpleNamespace(xyz=p) for i,p in enumerate(xyz)}

    def test_reuse_scales_intrinsics_and_preserves_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cameras,images,points=self.fixture(root)
            pose, reason=registered_anchor('/S08E04/general_f00032018-f00032041.mp4',0,24,root/'query.jpg',cameras,images,points,[root])
            self.assertIsNone(reason)
            self.assertEqual(pose['pose_source'],'registered_colmap')
            self.assertEqual(pose['pnp']['intrinsics'],[[150.,0.,80.],[0.,150.,45.],[0.,0.,1.]])
            self.assertEqual(assess_pose_quality(pose)['status'],'accepted')
            np.testing.assert_allclose(pose['camera_center'],[0,0,0])

    def test_missing_registered_frame_falls_back(self):
        pose,reason=registered_anchor('/S08E04/general_f00032018-f00032041.mp4',0,24,'query.jpg',{}, {}, {}, [])
        self.assertIsNone(pose)
        self.assertIn('not registered',reason)

    def test_mismatched_frame_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cameras,images,points=self.fixture(root)
            cv2.imwrite(str(root/'query.jpg'), np.random.default_rng(4).integers(0,256,(90,160,3),dtype=np.uint8))
            pose,reason=registered_anchor('/S08E04/general_f00032018-f00032041.mp4',0,24,root/'query.jpg',cameras,images,points,[root])
            self.assertIsNone(pose)
            self.assertIn('identity check failed',reason)

    def test_unverified_registered_pose_is_rejected(self):
        pose={'pose_source':'registered_colmap','pnp':{'inlier_count':100,'inlier_rmse_pixels':0.2}}
        self.assertEqual(assess_pose_quality(pose)['status'],'rejected')
