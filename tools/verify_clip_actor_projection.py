#!/usr/bin/env python3
"""Check fixed proxy dimensions and head-anchor reprojection inside Blender.

Run after loading an imported .blend with --python-exit-code 1, then pass
-- --output report.json [--max-error-pixels 0.1]. This checks consistency
with saved 2D anchors; it does not measure accuracy against real anatomy.
"""
import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-error-pixels', type=float, default=0.1)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    scene = bpy.context.scene
    cameras = json.loads(Path(scene['clip_camera_track']).read_text())
    actors = json.loads(Path(scene['clip_actor_tracks']).read_text())
    width, height = cameras['resolution']['width'], cameras['resolution']['height']
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    proxy_height = float(actors.get('proxy_convention', {}).get('height_world', 1.70))
    errors = []
    sizes = {}
    for track in actors['tracks']:
        if not track.get("proxy_quality", {}).get("eligible", True):
            continue
        name = f"ClipActor_{track['track_id']:04d}"
        root = bpy.data.objects[name]
        for item in track['frames']:
            if 'body_anchor_xy' not in item:
                continue
            scene.frame_set(int(item['frame_index']) + 1)
            bpy.context.view_layer.update()
            top = root.matrix_world @ Vector((0, 0, 0.5 * proxy_height))
            projected = world_to_camera_view(scene, scene.camera, top)
            if projected.z <= 0:
                raise AssertionError(f'{name} is behind the camera')
            pixel = Vector((projected.x * width, (1 - projected.y) * height))
            errors.append((pixel - Vector(item['body_anchor_xy'])).length)
            # All generated mesh pieces, including the head, must keep size.
            for obj in root.children:
                dimensions = tuple(obj.dimensions)
                sizes.setdefault(obj.name, dimensions)
                if max(abs(a-b) for a, b in zip(dimensions, sizes[obj.name])) > 1e-5:
                    raise AssertionError(f'{obj.name} changes dimensions within the clip')
    if not errors:
        raise ValueError('No body_anchor_xy observations to verify')
    result = {
        'observations': len(errors),
        'max_head_anchor_error_pixels': max(errors),
        'mean_head_anchor_error_pixels': sum(errors) / len(errors),
        'fixed_mesh_dimensions_all_frames': True,
        'scope': 'reprojection consistency with saved anchors; not anatomical or depth ground truth',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if max(errors) > args.max_error_pixels:
        raise AssertionError(f'Head anchor error exceeds {args.max_error_pixels} px')


if __name__ == '__main__':
    main()
