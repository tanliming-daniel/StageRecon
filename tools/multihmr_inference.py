"""Calibrated Multi-HMR inference on RGB images or SAM3-guided RGB crops."""

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps


@contextmanager
def model_directory(path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


class MultiHMR:
    def __init__(self, source, checkpoint, backbone=None):
        source = source.resolve()
        sys.path.insert(0, str(source))
        from model import Model
        from utils import normalize_rgb

        self.normalize = normalize_rgb
        torch.serialization.add_safe_globals([argparse.Namespace])
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        options = vars(data["args"]).copy()
        self.size = options["img_size"] = options["img_size"][0]
        options["type"] = options["train_return_type"]
        load = torch.hub.load

        def load_backbone(repo, name, *args, **kwargs):
            if backbone and repo == "facebookresearch/dinov2":
                return load(
                    str(backbone),
                    name,
                    *args,
                    **dict(kwargs, source="local", pretrained=False),
                )
            return load(repo, name, *args, **kwargs)

        with model_directory(source), patch.object(torch.hub, "load", load_backbone):
            model = Model(**options)
        state = model.load_state_dict(data["model_state_dict"], strict=False)
        missing = [k for k in state.missing_keys if not k.startswith("smpl_layer.")]
        if missing:
            raise ValueError(f"Missing checkpoint parameters: {missing}")
        self.model = model.cuda().eval()
        weights = model.smpl_layer[f"neutral_{model.num_betas}"].bm_x.lbs_weights
        self.head = weights[:, [15, 22, 23, 24]].sum(dim=1) > 0.6

    @torch.inference_mode()
    def predict(self, image, K, threshold):
        rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        scaled = ImageOps.contain(rgb, (self.size, self.size))
        ox, oy = (self.size - scaled.width) // 2, (self.size - scaled.height) // 2
        canvas = Image.new("RGB", (self.size, self.size))
        canvas.paste(scaled, (ox, oy))
        pixels = (
            torch.from_numpy(self.normalize(np.asarray(canvas))).unsqueeze(0).cuda()
        )
        affine = np.array(
            [
                [scaled.width / rgb.width, 0, ox],
                [0, scaled.height / rgb.height, oy],
                [0, 0, 1],
            ]
        )
        camera = torch.as_tensor(affine @ K, dtype=torch.float32, device="cuda")[None]
        with torch.autocast("cuda", dtype=torch.float16):
            people = self.model(
                pixels,
                K=camera,
                det_thresh=threshold,
                nms_kernel_size=3,
                is_training=False,
            )
        result = []
        for p in people:
            joints = p["j3d"][:22].float().cpu().numpy()
            vertices = p["v3d"][self.head].float().cpu().numpy()
            rot = p["rotvec"][:22].float().cpu().numpy()
            if (
                not all(np.isfinite(v).all() for v in (joints, vertices, rot))
                or min(joints[:, 2].min(), vertices[:, 2].min()) <= 0
            ):
                continue

            def project(points):
                q = points @ K.T
                return q[:, :2] / q[:, 2:]

            box = np.percentile(project(vertices), [1, 99], axis=0).ravel()
            result.append(
                dict(
                    joints_original_px=project(joints),
                    joints_camera_m=joints,
                    head_bounds=box,
                    rotvecs=rot,
                )
            )
        return result
