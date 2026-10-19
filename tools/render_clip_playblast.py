#!/usr/bin/env python3
"""Render a lightweight MP4 playblast through an animated clip camera."""

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
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--hide-collection", action="append", default=[])
    return parser.parse_args(argv)


def set_workbench_engine(scene: bpy.types.Scene) -> str:
    for engine in ("BLENDER_WORKBENCH_NEXT", "BLENDER_WORKBENCH"):
        try:
            scene.render.engine = engine
            return engine
        except TypeError:
            continue
    raise RuntimeError("This Blender build does not expose the Workbench render engine")


def main() -> None:
    args = parse_args()
    if args.width < 1 or args.height < 1:
        raise ValueError("width and height must be positive")

    scene = bpy.context.scene
    camera = bpy.data.objects.get(args.camera_name)
    if camera is None or camera.type != "CAMERA":
        raise RuntimeError(f"Missing animated camera: {args.camera_name}")
    scene.camera = camera

    start = args.start_frame or int(scene.frame_start)
    end = args.end_frame or int(scene.frame_end)
    if start < scene.frame_start or end > scene.frame_end or start > end:
        raise ValueError(
            f"Requested range [{start}, {end}] is outside "
            f"[{scene.frame_start}, {scene.frame_end}]"
        )
    scene.frame_start = start
    scene.frame_end = end
    scene.frame_set(start)

    for name in args.hide_collection:
        collection = bpy.data.collections.get(name)
        if collection is not None:
            collection.hide_render = True
            collection.hide_viewport = True

    engine = set_workbench_engine(scene)
    shading = scene.display.shading
    shading.type = "SOLID"
    shading.light = "STUDIO"
    shading.color_type = "TEXTURE"
    shading.show_shadows = False
    shading.show_cavity = False
    shading.show_specular_highlight = False

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.filepath = str(output)
    scene.render.use_file_extension = True

    print(
        json.dumps(
            {
                "active_camera": scene.camera.name,
                "frame_range": [start, end],
                "fps": scene.render.fps / scene.render.fps_base,
                "resolution": [args.width, args.height],
                "engine": engine,
                "output": str(output),
            },
            indent=2,
        ),
        flush=True,
    )
    bpy.ops.render.render(animation=True)
    if not output.is_file():
        raise RuntimeError(f"Playblast did not create: {output}")

    metadata = {
        "active_camera": scene.camera.name,
        "frame_range": [start, end],
        "source_frame_range_inclusive": [start - 1, end - 1],
        "frame_count": end - start + 1,
        "fps": scene.render.fps / scene.render.fps_base,
        "resolution": {"width": args.width, "height": args.height},
        "engine": engine,
        "output": str(output),
        "camera_has_animation": camera.animation_data is not None,
        "lens_has_animation": camera.data.animation_data is not None,
    }
    metadata_path = args.metadata.resolve()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
