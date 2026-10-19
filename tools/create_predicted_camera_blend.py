#!/usr/bin/env python3
"""Create a predicted active camera in an existing Blender scene.

Run with Blender. The script maps COLMAP coordinates with the existing
point-cloud object's matrix_world and writes a separate blend file by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCALIZATION = REPO_ROOT / "outputs/Friends-monica_apartment/new_image_localization/localization_calibrated_colmap.json"
DEFAULT_OUTPUT = REPO_ROOT / "sitcoms3D/data/sparse_reconstruction_and_nerf_data/Friends-monica_apartment/3dscene_predicted_camera.blend"
CAMERA_NAME = "PredictedCamera_query_0001"
COLLECTION_NAME = "PredictedCamera_query_0001"


def cli_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--localization", type=Path, default=DEFAULT_LOCALIZATION)
    parser.add_argument("--point-cloud-object", default="pointcloud")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def delete_object(name: str) -> None:
    obj = bpy.data.objects.get(name)
    if obj is None:
        return
    data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is not None and data.users == 0:
        if isinstance(data, bpy.types.Camera):
            bpy.data.cameras.remove(data)
        elif isinstance(data, bpy.types.Curve):
            bpy.data.curves.remove(data)


def get_collection(name: str):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def read_pose(path: Path) -> dict:
    data = json.loads(path.read_text())
    r_world_to_camera = Matrix(data["world_to_camera_rotation"])
    translation = Vector(data["world_to_camera_translation"])
    center = Vector(data.get("camera_center", -(r_world_to_camera.transposed() @ translation)))
    intrinsics = data["pnp"]["intrinsics"]
    fx, fy = float(intrinsics[0][0]), float(intrinsics[1][1])
    cx, cy = float(intrinsics[0][2]), float(intrinsics[1][2])
    if min(fx, fy, cx, cy) <= 0:
        raise ValueError("Invalid PnP intrinsics in localization JSON")
    return {
        "data": data,
        "camera_to_world": r_world_to_camera.transposed(),
        "center": center,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": 2.0 * cx,
        "height": 2.0 * cy,
    }


def camera_matrix_blender(pose: dict, pointcloud_matrix: Matrix) -> Matrix:
    # COLMAP camera: +X right, +Y down, +Z forward.
    # Blender camera: +X right, +Y up, -Z forward.
    colmap_camera_to_blender_camera = Matrix(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)))
    pointcloud_rotation = pointcloud_matrix.to_3x3().to_quaternion().to_matrix()
    rotation = pointcloud_rotation @ pose["camera_to_world"] @ colmap_camera_to_blender_camera
    center = pointcloud_matrix @ pose["center"]
    return Matrix.Translation(center) @ rotation.to_4x4()


def make_camera(pose: dict, matrix_world: Matrix, collection):
    data = bpy.data.cameras.new(CAMERA_NAME)
    data.type = "PERSP"
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 36.0
    data.lens = pose["fx"] * data.sensor_width / pose["width"]
    data.clip_start = 0.01
    data.clip_end = 1000.0
    data.display_size = 0.25
    camera = bpy.data.objects.new(CAMERA_NAME, data)
    collection.objects.link(camera)
    camera.matrix_world = matrix_world
    camera.show_name = True
    camera.show_in_front = True
    camera.color = (1.0, 0.18, 0.02, 1.0)
    pnp = pose["data"].get("pnp", {})
    camera["source"] = "DISK matches -> COLMAP tracks -> RANSAC PnP"
    camera["query_image"] = str(pose["data"].get("query_image", ""))
    camera["pnp_inliers"] = int(pnp.get("inlier_count", 0))
    camera["pnp_rmse_pixels"] = float(pnp.get("inlier_rmse_pixels", 0.0))
    camera["colmap_center"] = tuple(float(value) for value in pose["center"])
    camera["blender_center"] = tuple(float(value) for value in matrix_world.translation)
    return camera



def main() -> None:
    args = cli_args()
    pose = read_pose(args.localization.resolve())
    pointcloud = bpy.data.objects.get(args.point_cloud_object)
    if pointcloud is None:
        raise RuntimeError(f"Missing point-cloud object: {args.point_cloud_object}")
    delete_object(CAMERA_NAME)
    delete_object("PredictedCameraFrustum_query_0001")
    collection = get_collection(COLLECTION_NAME)
    matrix_world = camera_matrix_blender(pose, pointcloud.matrix_world.copy())
    camera = make_camera(pose, matrix_world, collection)
    bpy.context.scene.camera = camera
    bpy.ops.object.select_all(action="DESELECT")
    camera.select_set(True)
    bpy.context.view_layer.objects.active = camera
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    print(f"Created {camera.name}")
    print(f"Point cloud: {pointcloud.name}")
    print(f"COLMAP center: {tuple(round(float(v), 6) for v in pose['center'])}")
    print(f"Blender center: {tuple(round(float(v), 6) for v in camera.matrix_world.translation)}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
