"""Regressions for mask support and fixed-size proxy anchoring."""
import json
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np
    from tools.estimate_actor_boxes import (
        robust_body_core, robust_mask_depth, smooth_localization_bboxes,
        proxy_center_from_body_anchor, proxy_support_quality,
    )
except ModuleNotFoundError:
    cv2 = None



@unittest.skipIf(cv2 is None, "Actor geometry tests require numpy and OpenCV")
class ActorGeometryTests(unittest.TestCase):
    def quality(self, bbox, visible_height, boundary=False):
        return proxy_support_quality([{'bbox_xyxy':bbox,'bbox_3d':{'size_world':[.1,visible_height,.1]},
                                        'touches_image_boundary':boundary}],1.7,(1280,720))

    def test_tiny_photo_sized_detection_requires_review(self):
        q=self.quality([1006,155,1031,203],.13)
        self.assertFalse(q['eligible'])
        self.assertTrue(q['review_required'])

    def test_large_closeup_head_is_preserved(self):
        self.assertTrue(self.quality([100,0,600,500],.20,True)['eligible'])

    def test_distant_full_body_is_preserved(self):
        self.assertTrue(self.quality([100,100,120,150],1.7)['eligible'])

    def test_binary_mask_depth_is_not_discarded(self):
        mask = np.zeros((120, 160), np.uint8)
        mask[15:110, 50:100] = 1
        depth = np.full(mask.shape, 100., np.float32)
        depth[mask > 0] = 3.
        result = robust_mask_depth({}, depth, None, (160, 120), mask)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['median'], 3.)

    def test_head_top_does_not_move_when_lower_body_is_cropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            anchors = []
            for bottom in (115, 180):
                mask = np.zeros((200, 160), np.uint8)
                cv2.circle(mask, (80, 30), 12, 255, -1)
                cv2.rectangle(mask, (61, 36), (99, bottom), 255, -1)
                # Detached object must not move the person bbox or head.
                cv2.rectangle(mask, (130, 2), (140, 10), 255, -1)
                path = Path(tmp) / f'{bottom}.png'
                cv2.imwrite(str(path), mask)
                result = robust_body_core({'mask': str(path), 'bbox_xyxy': [61, 2, 140, bottom]}, (160, 200))
                anchors.append(result['anchor_xy'])
                self.assertEqual(result['component_bbox_xyxy'][2], 99)
            self.assertLess(np.linalg.norm(anchors[0] - anchors[1]), 1.)
            self.assertLess(anchors[0][1], 22.)

    def test_component_extent_is_kept_separate_from_torso_axis(self):
        observations = [{'bbox_xyxy': [0, 20, 130, 199],
                         'mask_component_bbox_xyxy': [0, 20, 100, 199]}]
        actual = smooth_localization_bboxes(observations, 7, (160, 200))
        np.testing.assert_allclose(actual, [[0, 20, 100, 199]], atol=1e-10)

    def test_nonzero_frame_ids_use_depth_array_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask = np.zeros((120, 160), np.uint8)
            cv2.rectangle(mask, (55, 10), (100, 110), 255, -1)
            cv2.imwrite(str(root / 'mask.png'), mask)
            frames = []
            observations = []
            for frame_id in (10, 20):
                frames.append({'frame_index': frame_id, 'frame_name': f'{frame_id}.jpg',
                               'intrinsics': [[200, 0, 80], [0, 200, 60], [0, 0, 1]],
                               'world_to_camera_rotation': np.eye(3).tolist(),
                               'camera_center': [0, 0, 0], 'focal_pixels': 200})
                observations.append({'frame_index': frame_id, 'frame_name': f'{frame_id}.jpg',
                                     'mask': 'mask.png', 'bbox_xyxy': [55, 10, 100, 110]})
            (root / 'camera.json').write_text(json.dumps({'frames': frames, 'resolution': {'width': 160, 'height': 120}, 'da3': {'translation_scale': 1}}))
            (root / 'people.json').write_text(json.dumps({'tracks': [{'track_id': 0, 'frames': observations}]}))
            np.savez(root / 'depth.npz', depth=np.stack([np.full((120, 160), 3.), np.full((120, 160), 4.)]))
            script = Path(__file__).resolve().parents[1] / 'tools/estimate_actor_boxes.py'
            subprocess.run([sys.executable, str(script), '--person-tracks', str(root/'people.json'),
                            '--camera-track', str(root/'camera.json'), '--depth-prediction', str(root/'depth.npz'),
                            '--output', str(root/'actors.json'), '--depth-window', '1'], check=True, capture_output=True)
            result = json.loads((root/'actors.json').read_text())
            track = result['tracks'][0]['frames']
            self.assertEqual([f['frame_index'] for f in track], [10, 20])
            np.testing.assert_allclose([f['depth']['visible_surface_world'] for f in track], [3, 4])
            np.testing.assert_allclose(track[0]['proxy_up_world'], track[1]['proxy_up_world'])
            for frame in track:
                proxy=frame['proxy_bbox_3d']
                np.testing.assert_allclose(proxy['size_world'],[.36,.36,1.7])
                np.testing.assert_allclose(proxy['center_world'],frame['proxy_center_world'])
                top=np.asarray(proxy['center_world']) + .85*np.asarray(proxy['axes_world'])[:,2]
                np.testing.assert_allclose(top,frame['head_anchor_world'])

    def test_fixed_up_preserves_head_anchor_under_camera_tilt(self):
        up = np.array([0., -1., 0.])
        for tilt in (0., .3):
            r = np.array([[np.cos(tilt), -np.sin(tilt), 0.],
                          [np.sin(tilt), np.cos(tilt), 0.], [0., 0., 1.]])
            k = np.array([[700., 0., 320.], [0., 700., 180.], [0., 0., 1.]])
            camera = {'intrinsics': k.tolist(), 'world_to_camera_rotation': r.tolist(), 'camera_center': [1., 2., 3.]}
            anchor = np.array([123., 45.])
            center = proxy_center_from_body_anchor(anchor, np.array([0., 0., 4.]), camera, 1.7, up)
            head = center + .85 * up
            q = k @ (r @ (head - np.array(camera['camera_center'])))
            np.testing.assert_allclose(q[:2] / q[2], anchor, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
