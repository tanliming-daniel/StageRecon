#!/usr/bin/env python3
"""Export a predicted camera as importable JSON without writing a Blender file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def cli_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--localization", type=Path, required=True)
    parser.add_argument("--point-cloud-object", default="pointcloud")
    parser.add_argument("--camera-name", default="PredictedCamera_query_0001")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def camera_matrix_blender(pose: dict, pointcloud_matrix: Matrix) -> Matrix:
    # COLMAP: +X right, +Y down, +Z forward. Blender: +X right, +Y up, -Z forward.
    colmap_to_blender = Matrix(
        ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))
    )
    pointcloud_rotation = pointcloud_matrix.to_3x3().to_quaternion().to_matrix()
    rotation = (
        pointcloud_rotation
        @ Matrix(pose["world_to_camera_rotation"]).transposed()
        @ colmap_to_blender
    )
    center = pointcloud_matrix @ Vector(pose["camera_center"])
    return Matrix.Translation(center) @ rotation.to_4x4()


def matrix_to_lists(matrix: Matrix) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def main() -> None:
    args = cli_args()
    localization = json.loads(args.localization.resolve().read_text())
    pointcloud = bpy.data.objects.get(args.point_cloud_object)
    if pointcloud is None:
        raise RuntimeError(f"Missing alignment object: {args.point_cloud_object}")

    intrinsics = localization["pnp"]["intrinsics"]
    width = int(round(2.0 * float(intrinsics[0][2])))
    height = int(round(2.0 * float(intrinsics[1][2])))
    fx = float(intrinsics[0][0])
    camera_matrix = camera_matrix_blender(localization, pointcloud.matrix_world.copy())
    info = {
        "schema_version": 1,
        "camera_name": args.camera_name,
        "query_key": localization.get("query_key"),
        "query_image": localization.get("query_image"),
        "coordinate_system": "raw_colmap_world",
        "source": "DISK matches -> COLMAP tracks -> RANSAC PnP",
        "resolution": {"width": width, "height": height},
        "intrinsics": intrinsics,
        "world_to_camera_rotation": localization["world_to_camera_rotation"],
        "world_to_camera_translation": localization["world_to_camera_translation"],
        "camera_center_colmap": localization["camera_center"],
        "blender": {
            "matrix_world": matrix_to_lists(camera_matrix),
            "lens_mm": fx * 36.0 / width,
            "sensor_width_mm": 36.0,
            "sensor_fit": "HORIZONTAL",
            "clip_start": 0.01,
            "clip_end": 1000.0,
        },
        "alignment": {
            "object": args.point_cloud_object,
            "matrix_world": matrix_to_lists(pointcloud.matrix_world),
        },
        "pnp": localization.get("pnp", {}),
        "inlier_source_images": localization.get("inlier_source_images", {}),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(info, indent=2) + "\n")
    print(f"Camera info: {output}")


if __name__ == "__main__":
    main()
