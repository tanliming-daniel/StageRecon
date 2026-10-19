#!/usr/bin/env python3
"""Capture one tracked-camera frame with the standard Blender viewport preview."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--camera-name", default="PredictedClipCamera")
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Blender frame to preview; 0 selects the shot midpoint.",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--hide-collection", action="append", default=[])
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.width < 1 or args.height < 1:
        raise ValueError("width and height must be positive")
    source_scene = bpy.context.scene
    camera = bpy.data.objects.get(args.camera_name)
    if camera is None or camera.type != "CAMERA":
        raise RuntimeError(f"Missing animated camera: {args.camera_name}")
    preview_frame = args.frame or (
        source_scene.frame_start + source_scene.frame_end
    ) // 2
    if not source_scene.frame_start <= preview_frame <= source_scene.frame_end:
        raise ValueError(
            f"Preview frame {preview_frame} is outside "
            f"[{source_scene.frame_start}, {source_scene.frame_end}]"
        )
    source_scene.frame_set(preview_frame)
    source_scene.camera = camera

    for name in args.hide_collection:
        collection = bpy.data.collections.get(name)
        if collection is not None:
            collection.hide_viewport = True

    preview_scene = bpy.data.scenes.new("__clip_viewport_preview")
    for collection in source_scene.collection.children:
        preview_scene.collection.children.link(collection)
    preview_scene.collection.objects.link(camera)
    preview_scene.camera = camera
    preview_scene.frame_set(preview_frame)
    bpy.context.window.scene = preview_scene
    preview_scene.view_layers[0].update()

    area = next((item for item in bpy.context.screen.areas if item.type == "VIEW_3D"), None)
    if area is None:
        raise RuntimeError("The Blender UI has no VIEW_3D area")
    region = next((item for item in area.regions if item.type == "WINDOW"), None)
    if region is None:
        raise RuntimeError("The Blender VIEW_3D area has no WINDOW region")
    space = area.spaces.active
    for shading in (space.shading, preview_scene.display.shading):
        shading.type = "SOLID"
        shading.light = "STUDIO"
        shading.color_type = "TEXTURE"
        shading.show_shadows = False
        shading.show_cavity = False
        shading.show_specular_highlight = False
    if hasattr(space, "camera"):
        space.camera = camera
    space.region_3d.view_perspective = "CAMERA"
    bpy.ops.object.select_all(action="DESELECT")
    camera.select_set(True)
    bpy.context.view_layer.objects.active = camera
    bpy.context.view_layer.update()

    with bpy.context.temp_override(
        window=bpy.context.window,
        area=area,
        region=region,
        scene=preview_scene,
        space_data=space,
    ):
        bpy.ops.view3d.view_camera()
        # Do not wait for Xvfb window swaps here.  On a headless display this
        # synchronization can block for tens of seconds per frame; the OpenGL
        # viewport render below evaluates the active camera directly.

    preview_scene.render.resolution_x = args.width
    preview_scene.render.resolution_y = args.height
    preview_scene.render.resolution_percentage = 100
    preview_scene.render.image_settings.file_format = "PNG"
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    preview_scene.render.filepath = str(output)
    with bpy.context.temp_override(
        window=bpy.context.window,
        area=area,
        region=region,
        scene=preview_scene,
        space_data=space,
    ):
        bpy.ops.render.opengl(write_still=True, view_context=False)
    if not output.is_file():
        raise RuntimeError(f"Viewport preview did not create: {output}")

    metadata = {
        "active_camera": preview_scene.camera.name,
        "viewport_camera": getattr(space.camera, "name", None),
        "preview_blender_frame": preview_frame,
        "source_frame_index": preview_frame - 1,
        "source_frame_range": [source_scene.frame_start, source_scene.frame_end],
        "resolution": {"width": args.width, "height": args.height},
        "camera_matrix_world": [list(row) for row in camera.matrix_world],
        "camera_lens_mm": camera.data.lens,
        "viewport_color_type": space.shading.color_type,
        "render_color_type": preview_scene.display.shading.color_type,
    }
    args.metadata.resolve().write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    main()
