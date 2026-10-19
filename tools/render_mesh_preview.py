#!/usr/bin/env python3
"""Render a flat-colour camera preview from temporary Blender mesh geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--camera-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    return parser.parse_args()


def world_to_open3d_camera(matrix_world: np.ndarray) -> np.ndarray:
    world_to_blender_camera = np.linalg.inv(matrix_world)
    # Blender camera coordinates are +X right, +Y up, -Z forward.  Open3D's
    # pinhole camera uses +X right, +Y down, +Z forward.
    blender_to_open3d = np.diag((1.0, -1.0, -1.0, 1.0))
    return blender_to_open3d @ world_to_blender_camera


def main() -> None:
    args = parse_args()
    cache = np.load(args.mesh)
    vertices = np.asarray(cache["vertices"], dtype=np.float32)
    triangles = np.asarray(cache["triangles"], dtype=np.int32)
    colours = np.asarray(cache["colours"], dtype=np.float32)
    info = json.loads(args.camera_info.resolve().read_text())

    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(triangles, dtype=o3d.core.Dtype.Int32),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    source_width = float(info["resolution"]["width"])
    source_height = float(info["resolution"]["height"])
    scale_x = args.width / source_width
    scale_y = args.height / source_height
    intrinsics = np.asarray(info["intrinsics"], dtype=np.float32).copy()
    intrinsics[0, 0] *= scale_x
    intrinsics[0, 2] *= scale_x
    intrinsics[1, 1] *= scale_y
    intrinsics[1, 2] *= scale_y
    matrix_world = np.asarray(info["blender"]["matrix_world"], dtype=np.float32)
    extrinsics = world_to_open3d_camera(matrix_world).astype(np.float32)
    rays = scene.create_rays_pinhole(
        o3d.core.Tensor(intrinsics),
        o3d.core.Tensor(extrinsics),
        args.width,
        args.height,
    )
    hits = scene.cast_rays(rays)
    primitive_ids = hits["primitive_ids"].numpy().reshape(args.height, args.width)
    image = np.empty((args.height, args.width, 3), dtype=np.uint8)
    image[:] = (18, 22, 30)
    hit_mask = primitive_ids != np.iinfo(np.uint32).max
    image[hit_mask] = np.rint(
        np.clip(colours[primitive_ids[hit_mask]], 0.0, 1.0) * 255.0
    ).astype(np.uint8)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(output)
    print(f"preview_saved={output} hit_pixels={int(hit_mask.sum())}/{hit_mask.size}")


if __name__ == "__main__":
    main()
