#!/usr/bin/env python3
"""Render the Point Cloud Visualizer object from a selected Blender camera.

Run this through Blender's Python interpreter after opening a .blend containing
the ``pointcloud`` PCV controller. Unlike Workbench, this uses the add-on's
own offscreen renderer, so the PLY vertex colors remain visible.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import bpy


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADDON = REPO_ROOT / "third_party/space_view3d_point_cloud_visualizer.py"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/Friends-monica_apartment/new_image_localization/pcv_preview_query_0001.png"
DEFAULT_PLY = REPO_ROOT / "sitcoms3D/data/sparse_reconstruction_and_nerf_data/Friends-monica_apartment/colmap/pc.ply"


def cli_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="PredictedCamera_query_0001")
    parser.add_argument("--point-cloud-object", default="pointcloud")
    parser.add_argument("--addon", type=Path, default=DEFAULT_ADDON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ply", type=Path, default=DEFAULT_PLY)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--point-size", type=float, default=2.0)
    return parser.parse_args(argv)


def load_addon(path: Path):
    module_name = "space_view3d_point_cloud_visualizer"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load Point Cloud Visualizer from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        module.register()
    return module


def main() -> None:
    args = cli_args()
    if not args.addon.is_file():
        raise FileNotFoundError(args.addon)
    if not args.ply.is_file():
        raise FileNotFoundError(args.ply)
    if args.width <= 0 or args.height <= 0 or args.point_size <= 0:
        raise ValueError("Preview dimensions and point size must be positive")

    addon = load_addon(args.addon)
    cloud = bpy.data.objects.get(args.point_cloud_object)
    camera = bpy.data.objects.get(args.camera)
    if cloud is None:
        raise KeyError(f"Point-cloud controller not found: {args.point_cloud_object}")
    if camera is None or camera.type != "CAMERA":
        raise KeyError(f"Camera not found or not a camera: {args.camera}")

    scene = bpy.context.scene
    scene.camera = camera
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    bpy.context.view_layer.objects.active = cloud
    cloud.select_set(True)

    pcv = cloud.point_cloud_visualizer
    # The source .blend was authored on Windows, so its saved PCV path is stale.
    # Override it in-memory for this render only.
    pcv.filepath = str(args.ply)
    if not pcv.uuid or pcv.uuid not in addon.PCVManager.cache:
        if not addon.PCVManager.initialized:
            addon.PCVManager.init()
        if not addon.PCVManager.load_ply_to_cache(None, bpy.context):
            raise RuntimeError(f"Could not load PLY: {pcv.filepath}")

    cache = addon.PCVManager.cache[pcv.uuid]
    cache["draw"] = True
    pcv.render_resolution_linked = True
    pcv.render_display_percent = 100.0
    pcv.render_point_size = args.point_size
    pcv.render_path = str(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    result = bpy.ops.point_cloud_visualizer.render()
    if "FINISHED" not in result:
        raise RuntimeError(f"Point Cloud Visualizer render failed: {result}")
    print(args.output)


if __name__ == "__main__":
    main()
