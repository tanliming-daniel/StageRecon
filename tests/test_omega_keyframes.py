import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.omega_actor_keyframes import stabilize
from tools.omega_depth_alignment import calibrate


def camera():
    return {"camera_center":[0.,0.,0.], "world_to_camera_rotation":np.eye(3).tolist(),
            "world_to_camera_translation":[0.,0.,0.], "intrinsics":[[80.,0.,48.],[0.,80.,36.],[0.,0.,1.]]}


class OmegaKeyframeTests(unittest.TestCase):
    def test_masked_plane_known_scale(self):
        h,w=72,96
        y,x=np.mgrid[:h,:w]
        xyz=np.stack(((x-48)*2/80,(y-36)*2/80,np.full_like(x,2)),axis=-1).reshape(-1,3)
        depth=np.ones((5,h,w),np.float32)
        depth[:,20:50,20:40]=0.2
        mask=np.zeros((h,w),np.uint8)
        mask[20:50,20:40]=255
        cams=[dict(camera(),frame_name=f"{i:06}.jpg") for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(5):
                cv2.imwrite(str(Path(tmp)/f"{i:06}.png"),mask)
            report, scales=calibrate(xyz,cams,depth,None,Path(tmp),(w,h),1,1.5)
        np.testing.assert_allclose(scales,2.,rtol=1e-6)
        self.assertFalse(report["temporal"])
        self.assertEqual(len(report["keyframes"]),5)

    def test_missing_observations_and_clip_boundaries(self):
        ids=[0,1,3,4,20,21,23]
        fs=[{"frame_index":i,"head_anchor_world":[0.,0.,2.],"body_anchor_xy":[48.,36.],
             "depth":{"visible_surface_world":1.9,"median_world":2.,"confidence":.8},
             "touches_image_boundary":i>=21,"facing_world":[1.,0.,0.]} for i in ids]
        a={"tracks":[{"track_id":1,"frames":fs}],"proxy_convention":{"up_world":[0.,1.,0.],
           "height_world":1.7,"radius_world":.18},"proxy_box_convention":{}}
        c={"frames":[camera() for _ in range(24)],"scene_frame":{"source_blend_sha256":"test",
           "units_per_meter":1.,"world_to_blender":np.eye(4).tolist()}}
        stabilize(a,c,{"scene_sha256":"other","clips":{}},"other",24)
        segs=a["tracks"][0]["animation_segments"]
        self.assertEqual([(s["first"],s["last"]) for s in segs],[(0,23)])
        self.assertEqual(len(a["tracks"][0]["frames"]),24)
        self.assertEqual(a["tracks"][0]["frames"][12]["observation_status"],"interpolated")
        self.assertNotIn("seat_contact",a["tracks"][0])

    def test_id_merge_requires_unambiguous_world_and_image_agreement(self):
        from tools.actor_presence import stitch_tracks
        def make(n, start, end, x):
            return {"track_id":n,"frames":[{"frame_index":i,"head_anchor_world":[x,0,2],
                "body_anchor_xy":[48+40*x,36],"bbox_xyxy":[20,10,75,70]} for i in range(start,end)]}
        cams=[camera() for _ in range(24)]
        found=stitch_tracks([make(0,0,5,0),make(3,7,12,0)],24,1,cams)
        self.assertEqual(found[0]["source_track_ids"],[0,3])
        found=stitch_tracks([make(0,0,5,0),make(3,7,12,0),make(4,7,12,.02)],24,1,cams)
        self.assertEqual(len(found),3)

    def test_stationary_actor_persists_before_and_after_observations(self):
        from tools.actor_presence import complete
        frames=[{"frame_index":i,"head_anchor_world":[0,0,2],"facing_world":[1,0,0],
                 "support":"depth","animation_keyframe":True} for i in (4,6,8)]
        t={"frames":frames}
        complete(t,[camera() for _ in range(14)],24,1,{"height_world":1.7,"up_world":[0,1,0]})
        self.assertEqual(t["observed_ranges"],[[4,4],[6,6],[8,8]])
        self.assertEqual([f["frame_index"] for f in t["frames"]],list(range(14)))
        self.assertEqual([t["frames"][i]["observation_status"] for i in (0,5,13)],["hold","interpolated","hold"])
        self.assertTrue(all(f["head_anchor_world"]==[0,0,2] for f in t["frames"]))
