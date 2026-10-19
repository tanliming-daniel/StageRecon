#!/usr/bin/env python3
"""Export a Blender Solid/Texture/Studio viewport PNG without saving a blend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--camera-name")
    parser.add_argument("--hide-collection", action="append", default=[])
    return parser.parse_args(argv)


def import_camera(info: dict, camera_name: str):
    camera = bpy.data.objects.get(camera_name)
    if camera is None:
        data = bpy.data.cameras.new(camera_name)
        camera = bpy.data.objects.new(camera_name, data)
        bpy.context.scene.collection.objects.link(camera)
    if camera.type != "CAMERA":
        raise RuntimeError(f"Object exists but is not a camera: {camera_name}")
    settings = info["blender"]
    camera.matrix_world = Matrix(settings["matrix_world"])
    camera.data.lens = float(settings["lens_mm"])
    camera.data.sensor_width = float(settings["sensor_width_mm"])
    camera.data.sensor_fit = settings.get("sensor_fit", "HORIZONTAL")
    camera.data.clip_start = float(settings.get("clip_start", 0.01))
    camera.data.clip_end = float(settings.get("clip_end", 1000.0))
    return camera


def main() -> None:
    args = parse_args()
    info = json.loads(args.camera_info.resolve().read_text())
    source_scene = bpy.context.scene
    camera = import_camera(info, args.camera_name or info["camera_name"])
    source_scene.camera = camera

    for name in args.hide_collection:
        collection = bpy.data.collections.get(name)
        if collection is not None:
            collection.hide_viewport = True

    # The source file has an animation-only image-output setting. A transient
    # scene supplies PNG output settings while sharing the source collections;
    # it is never saved and its active camera is explicitly the imported pose.
    preview_scene = bpy.data.scenes.new("__viewport_preview")
    for collection in source_scene.collection.children:
        preview_scene.collection.children.link(collection)
    preview_scene.collection.objects.link(camera)
    preview_scene.camera = camera
    bpy.context.window.scene = preview_scene
    preview_scene.view_layers[0].update()

    area = next((item for item in bpy.context.screen.areas if item.type == "VIEW_3D"), None)
    if area is None:
        raise RuntimeError("The Blender UI has no VIEW_3D area for a viewport capture")
    region = next((item for item in area.regions if item.type == "WINDOW"), None)
    if region is None:
        raise RuntimeError("The Blender VIEW_3D area has no WINDOW region")
    space = area.spaces.active
    # ``render.opengl(view_context=False)`` reads the scene Workbench state;
    # keep it in sync with the 3D View state so materials use image textures,
    # never the source file's ATTRIBUTE display color.
    for shading in (space.shading, preview_scene.display.shading):
        shading.type = "SOLID"
        shading.light = "STUDIO"
        shading.color_type = "TEXTURE"
        shading.show_shadows = False
        shading.show_cavity = False
        shading.show_specular_highlight = False

    # Match the previously verified capture path exactly: render the original
    # scene, with its active camera replaced only for this Blender process.
    # A temporary scene can retain an unrelated viewport camera state.
    if hasattr(space, "camera"):
        space.camera = camera
    space.region_3d.view_perspective = "CAMERA"
    bpy.ops.object.select_all(action="DESELECT")
    camera.select_set(True)
    bpy.context.view_layer.objects.active = camera
    preview_scene.camera = camera
    bpy.context.view_layer.update()

    with bpy.context.temp_override(
        window=bpy.context.window,
        area=area,
        region=region,
        scene=preview_scene,
        space_data=space,
    ):
        bpy.ops.view3d.view_camera()
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=3)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    preview_scene.render.resolution_x = 960
    preview_scene.render.resolution_y = 540
    preview_scene.render.resolution_percentage = 100
    preview_scene.render.image_settings.file_format = "PNG"
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
        raise RuntimeError(f"Viewport render did not create: {output}")
    metadata = {
        "render_width": preview_scene.render.resolution_x,
        "render_height": preview_scene.render.resolution_y,
        "image_format": "PNG",
        "active_camera": preview_scene.camera.name,
        "viewport_camera": getattr(space.camera, "name", None),
        "camera_matrix_world": [list(row) for row in camera.matrix_world],
        "camera_lens_mm": camera.data.lens,
        "viewport_color_type": space.shading.color_type,
        "render_color_type": preview_scene.display.shading.color_type,
    }
    args.metadata.resolve().write_text(json.dumps(metadata) + "\n")
    print(f"viewport_render={output}")
    print(json.dumps(metadata))
    print("The Blender scene was not saved.")
    bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    main()
