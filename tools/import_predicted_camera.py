#!/usr/bin/env python3
"""Import camera-only JSON into the currently opened Blender scene.

This script updates the predicted camera, or creates one in a
small predicted-camera collection. It never saves a blend file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix


def cli_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-info", type=Path, required=True)
    parser.add_argument("--camera-name")
    return parser.parse_args(argv)


def collection_for_new_camera(name: str):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def import_camera(info_path: Path, camera_name: str | None = None):
    info = json.loads(info_path.resolve().read_text())
    name = camera_name or info["camera_name"]
    camera = bpy.data.objects.get(name)
    if camera is None:
        data = bpy.data.cameras.new(name)
        camera = bpy.data.objects.new(name, data)
        collection_for_new_camera("PredictedCameras").objects.link(camera)
    elif camera.type != "CAMERA":
        raise RuntimeError(f"Object exists but is not a camera: {name}")

    settings = info["blender"]
    camera.data.lens = float(settings["lens_mm"])
    camera.data.sensor_width = float(settings["sensor_width_mm"])
    camera.data.sensor_fit = settings.get("sensor_fit", "HORIZONTAL")
    camera.data.clip_start = float(settings.get("clip_start", 0.01))
    camera.data.clip_end = float(settings.get("clip_end", 1000.0))
    camera.matrix_world = Matrix(settings["matrix_world"])
    camera["source"] = info.get("source", "camera-only import")
    camera["query_key"] = info.get("query_key", "")
    camera["query_image"] = info.get("query_image", "")
    camera["pnp_inliers"] = int(info.get("pnp", {}).get("inlier_count", 0))
    camera["pnp_rmse_pixels"] = float(info.get("pnp", {}).get("inlier_rmse_pixels", 0.0))
    camera["camera_info"] = str(info_path.resolve())
    bpy.context.scene.camera = camera
    bpy.ops.object.select_all(action="DESELECT")
    camera.select_set(True)
    bpy.context.view_layer.objects.active = camera
    return camera


def main() -> None:
    args = cli_args()
    camera = import_camera(args.camera_info, args.camera_name)
    print(f"Imported active camera: {camera.name}")
    print("The Blender scene was not saved.")


if __name__ == "__main__":
    main()
