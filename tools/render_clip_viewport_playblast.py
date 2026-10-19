#!/usr/bin/env python3
"""Create one cached Blender viewport playblast through an animated camera."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import bpy
import gpu


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--camera-name", default="PredictedClipCamera")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--hide-collection", action="append", default=[])
    parser.add_argument("--require-hardware-gpu", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    gpu_info = {
        "vendor": gpu.platform.vendor_get(),
        "renderer": gpu.platform.renderer_get(),
        "version": gpu.platform.version_get(),
        "device_type": gpu.platform.device_type_get(),
        "backend_type": gpu.platform.backend_type_get(),
    }
    if args.require_hardware_gpu and gpu_info["device_type"] == "SOFTWARE":
        raise RuntimeError(
            "Hardware GPU is required, but Blender is using software OpenGL: "
            f"{gpu_info['vendor']} / {gpu_info['renderer']}"
        )
    scene = bpy.context.scene
    camera = bpy.data.objects.get(args.camera_name)
    if camera is None or camera.type != "CAMERA":
        raise RuntimeError(f"Missing animated camera: {args.camera_name}")
    start = args.start_frame or int(scene.frame_start)
    end = args.end_frame or int(scene.frame_end)
    if start < scene.frame_start or end > scene.frame_end or start > end:
        raise ValueError(
            f"Requested range [{start}, {end}] is outside "
            f"[{scene.frame_start}, {scene.frame_end}]"
        )

    scene.camera = camera
    scene.frame_start = start
    scene.frame_end = end
    scene.frame_set(start)
    for name in args.hide_collection:
        collection = bpy.data.collections.get(name)
        if collection is not None:
            collection.hide_viewport = True

    area = next((item for item in bpy.context.screen.areas if item.type == "VIEW_3D"), None)
    if area is None:
        raise RuntimeError("The Blender UI has no VIEW_3D area")
    region = next((item for item in area.regions if item.type == "WINDOW"), None)
    if region is None:
        raise RuntimeError("The Blender VIEW_3D area has no WINDOW region")
    space = area.spaces.active
    space.shading.type = "SOLID"
    space.shading.light = "STUDIO"
    space.shading.color_type = "TEXTURE"
    space.shading.show_shadows = False
    space.shading.show_cavity = False
    space.shading.show_specular_highlight = False
    # bpy.ops.render.opengl(animation=True) reads the scene display shading
    # settings in addition to the VIEW_3D space.  Setting only space.shading
    # silently falls back to flat grey materials in a headless playblast,
    # while the single-frame preview correctly uses studio+texture.
    scene.display.shading.type = "SOLID"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.display.shading.show_shadows = False
    scene.display.shading.show_cavity = False
    scene.display.shading.show_specular_highlight = False
    if hasattr(space, "camera"):
        space.camera = camera
    space.region_3d.view_perspective = "CAMERA"

    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output)

    bpy.ops.object.select_all(action="DESELECT")
    camera.select_set(True)
    bpy.context.view_layer.objects.active = camera
    bpy.context.view_layer.update()
    started = time.perf_counter()
    with bpy.context.temp_override(
        window=bpy.context.window,
        area=area,
        region=region,
        scene=scene,
        space_data=space,
    ):
        bpy.ops.view3d.view_camera()
        # One animation operator keeps the viewport draw cache alive across
        # all frames.  Avoid redraw_timer: Xvfb swap synchronization stalls.
        bpy.ops.render.opengl(animation=True, view_context=False)
    elapsed = time.perf_counter() - started
    if not output.is_file():
        raise RuntimeError(f"Viewport playblast did not create: {output}")

    metadata = {
        "active_camera": scene.camera.name,
        "viewport_camera": getattr(space.camera, "name", None),
        "frame_range": [start, end],
        "source_frame_range_inclusive": [start - 1, end - 1],
        "frame_count": end - start + 1,
        "fps": scene.render.fps / scene.render.fps_base,
        "resolution": {"width": args.width, "height": args.height},
        "viewport_shading_type": space.shading.type,
        "viewport_light": space.shading.light,
        "viewport_color_type": space.shading.color_type,
        "scene_display_shading_type": scene.display.shading.type,
        "scene_display_light": scene.display.shading.light,
        "scene_display_color_type": scene.display.shading.color_type,
        "camera_has_animation": camera.animation_data is not None,
        "lens_has_animation": camera.data.animation_data is not None,
        "elapsed_seconds": elapsed,
        "seconds_per_frame": elapsed / (end - start + 1),
        "gpu": gpu_info,
        "output": str(output),
    }
    metadata_path = args.metadata.resolve()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    main()
