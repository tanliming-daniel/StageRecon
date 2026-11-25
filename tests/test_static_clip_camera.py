import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.static_clip_camera import background_motion


class StaticCameraTests(unittest.TestCase):
    def test_intermediate_pan_and_missing_support_prevent_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            masks=d/"masks"
            masks.mkdir()
            rng=np.random.default_rng(19)
            texture=rng.integers(0,256,(432,768),dtype=np.uint8)
            paths=[d/f"{i:06}.jpg" for i in range(3)]
            for p in paths:
                cv2.imwrite(str(p),texture)
                cv2.imwrite(str(masks/p.with_suffix(".png").name),np.zeros_like(texture))
            self.assertIsNotNone(background_motion(paths,masks))
            cv2.imwrite(str(paths[1]),np.roll(texture,10,axis=1))
            self.assertIsNone(background_motion(paths,masks))
            cv2.imwrite(str(paths[1]),texture)
            cv2.imwrite(str(masks/"000001.png"),np.full_like(texture,255))
            self.assertIsNone(background_motion(paths,masks))
