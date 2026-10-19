#!/usr/bin/env python3
"""Create an animated Blender camera and coarse actor proxies for one clip."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-track", type=Path, required=True)
    parser.add_argument("--actor-tracks", type=Path, required=True)
    parser.add_argument("--output-blend", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--point-cloud-object", default="pointcloud")
    parser.add_argument("--camera-name", default="PredictedClipCamera")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--hide-collection", action="append", default=[])
    return parser.parse_args(argv)


def ensure_collection(name: str):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def remove_generated_animation(camera_name: str) -> None:
    """Remove only camera/actor objects produced by an earlier clip import."""
    objects_to_remove = set()
    for name in ("ClipCamera", "ClipActors"):
        collection = bpy.data.collections.get(name)
        if collection is None:
            continue
        objects_to_remove.update(collection.all_objects)
        bpy.data.collections.remove(collection)
    # Older generated scenes may contain the camera outside ClipCamera.
    camera = bpy.data.objects.get(camera_name)
    if camera is not None:
        objects_to_remove.add(camera)
    for obj in objects_to_remove:
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)


def link_only(obj, collection) -> None:
    for owner in list(obj.users_collection):
        owner.objects.unlink(obj)
    collection.objects.link(obj)


def camera_matrix_blender(rotation_rows, center_values, pointcloud_matrix: Matrix) -> Matrix:
    colmap_to_blender = Matrix(
        ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))
    )
    pointcloud_rotation = pointcloud_matrix.to_3x3().to_quaternion().to_matrix()
    rotation = pointcloud_rotation @ Matrix(rotation_rows).transposed() @ colmap_to_blender
    center = pointcloud_matrix @ Vector(center_values)
    return Matrix.Translation(center) @ rotation.to_4x4()


def world_point(point, pointcloud_matrix: Matrix) -> Vector:
    return pointcloud_matrix @ Vector(point)


def world_direction(direction, pointcloud_matrix: Matrix) -> Vector:
    transformed = pointcloud_matrix.to_3x3() @ Vector(direction)
    return transformed.normalized()


def actor_basis(floor_x: Vector, floor_y: Vector, normal: Vector, yaw: float) -> Matrix:
    forward = math.cos(yaw) * floor_x + math.sin(yaw) * floor_y
    forward.normalize()
    side = normal.cross(forward).normalized()
    # Columns are local X (forward), local Y (side), local Z (up).
    return Matrix(
        (
            (forward.x, side.x, normal.x),
            (forward.y, side.y, normal.y),
            (forward.z, side.z, normal.z),
        )
    )


def actor_material(track_id: int):
    name = f"ClipActorMaterial_{track_id:04d}"
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    hue = (track_id * 0.38196601125) % 1.0
    # Compact HSV to RGB conversion through Blender's mathutils is not
    # available in every supported version; this palette is deterministic.
    colors = [
        (0.90, 0.20, 0.18, 1.0),
        (0.18, 0.65, 0.95, 1.0),
        (0.20, 0.82, 0.34, 1.0),
        (0.96, 0.62, 0.12, 1.0),
        (0.65, 0.30, 0.92, 1.0),
        (0.15, 0.82, 0.76, 1.0),
    ]
    material.diffuse_color = colors[int(hue * len(colors)) % len(colors)]
    return material


def main() -> None:
    args = parse_args()
    camera_track = json.loads(args.camera_track.resolve().read_text())
    actor_tracks = json.loads(args.actor_tracks.resolve().read_text())
    pointcloud = bpy.data.objects.get(args.point_cloud_object)
    if pointcloud is None:
        raise RuntimeError(f"Missing alignment object: {args.point_cloud_object}")
    alignment = pointcloud.matrix_world.copy()

    remove_generated_animation(args.camera_name)

    for name in args.hide_collection:
        collection = bpy.data.collections.get(name)
        if collection is not None:
            collection.hide_viewport = True
            collection.hide_render = True

    camera_collection = ensure_collection("ClipCamera")
    actor_collection = ensure_collection("ClipActors")
    camera = bpy.data.objects.get(args.camera_name)
    if camera is None:
        camera_data = bpy.data.cameras.new(args.camera_name)
        camera = bpy.data.objects.new(args.camera_name, camera_data)
    if camera.type != "CAMERA":
        raise RuntimeError(f"Object exists but is not a camera: {args.camera_name}")
    link_only(camera, camera_collection)
    camera.rotation_mode = "QUATERNION"
    camera.data.sensor_width = 36.0
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.clip_start = 0.01
    camera.data.clip_end = 1000.0

    width = int(camera_track["resolution"]["width"])
    height = int(camera_track["resolution"]["height"])
    for item in camera_track["frames"]:
        blender_frame = int(item["frame_index"]) + 1
        camera.matrix_world = camera_matrix_blender(
            item["world_to_camera_rotation"], item["camera_center"], alignment
        )
        camera.data.lens = float(item["focal_pixels"]) * 36.0 / width
        camera.keyframe_insert(data_path="location", frame=blender_frame)
        camera.keyframe_insert(data_path="rotation_quaternion", frame=blender_frame)
        camera.data.keyframe_insert(data_path="lens", frame=blender_frame)
    camera["camera_track"] = str(args.camera_track.resolve())
    camera["model"] = camera_track["model"]
    bpy.context.scene.camera = camera

    bbox_representation = "3d_bbox" in actor_tracks.get("representation", "")
    if not bbox_representation:
        floor = actor_tracks["floor"]
        floor_x = world_direction(floor["basis_x"], alignment)
        floor_y = world_direction(floor["basis_y"], alignment)
        floor_normal = world_direction(floor["normal"], alignment)
    proxy_names = []
    for track in actor_tracks.get("tracks", []):
        track_id = int(track["track_id"])
        root_name = f"ClipActor_{track_id:04d}"
        root = bpy.data.objects.get(root_name) or bpy.data.objects.new(root_name, None)
        link_only(root, actor_collection)
        root.rotation_mode = "QUATERNION"
        root["track_id"] = track_id
        root["actor_track"] = str(args.actor_tracks.resolve())

        if bbox_representation:
            bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.18, depth=1.70)
            body = bpy.context.object
            body.name = f"{root_name}_Body"
            body.display_type = "SOLID"
            body.color = actor_material(track_id).diffuse_color
            link_only(body, actor_collection)
            body.parent = root
            body.location = (0.0, 0.0, 0.0)
            body.rotation_euler = (0.0, 0.0, 0.0)
            body.data.materials.append(actor_material(track_id))
        else:
            bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.18, depth=1.70)
            body = bpy.context.object
            body.name = f"{root_name}_Body"
            link_only(body, actor_collection)
            body.parent = root
            body.location = (0.0, 0.0, 0.85)
            body.data.materials.append(actor_material(track_id))

            bpy.ops.mesh.primitive_cone_add(vertices=16, radius1=0.12, radius2=0.0, depth=0.55)
            arrow = bpy.context.object
            arrow.name = f"{root_name}_Facing"
            link_only(arrow, actor_collection)
            arrow.parent = root
            arrow.location = (0.28, 0.0, 1.30)
            arrow.rotation_euler[1] = math.pi / 2.0
            arrow.data.materials.append(actor_material(track_id))

        first_frame = int(track["first_frame"]) + 1
        last_frame = int(track["last_frame"]) + 1
        if first_frame > 1:
            root.scale = (0.0, 0.0, 0.0)
            root.keyframe_insert(data_path="scale", frame=first_frame - 1)
        for item in track["frames"]:
            blender_frame = int(item["frame_index"]) + 1
            if bbox_representation:
                box = item["bbox_3d"]
                position = world_point(
                    item.get("proxy_center_world", item.get("position_world", box["center_world"])),
                    alignment,
                )
                # The proxy position is stable when the visible bbox extent
                # changes; bbox dimensions never resize the fixed cylinder.
                root.matrix_world = Matrix.Translation(position)
            else:
                position = world_point(item["position_world"], alignment)
                basis = actor_basis(
                    floor_x, floor_y, floor_normal, float(item["yaw_radians_floor_basis"])
                )
                root.matrix_world = Matrix.Translation(position) @ basis.to_4x4()
            root.scale = (1.0, 1.0, 1.0)
            root.keyframe_insert(data_path="location", frame=blender_frame)
            root.keyframe_insert(data_path="rotation_quaternion", frame=blender_frame)
            root.keyframe_insert(data_path="scale", frame=blender_frame)
            if not bbox_representation:
                posture_scale = 0.56 if item["posture"] == "sitting" else 1.0
                body.scale = (1.0, 1.0, posture_scale)
                body.location.z = 0.85 * posture_scale
                body.keyframe_insert(data_path="scale", frame=blender_frame)
                body.keyframe_insert(data_path="location", frame=blender_frame)
        if last_frame < len(camera_track["frames"]):
            root.scale = (0.0, 0.0, 0.0)
            root.keyframe_insert(data_path="scale", frame=last_frame + 1)
        proxy_names.append(root_name)

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = len(camera_track["frames"])
    scene.render.fps = max(1, int(round(args.fps)))
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene["clip_camera_track"] = str(args.camera_track.resolve())
    scene["clip_actor_tracks"] = str(args.actor_tracks.resolve())
    scene.frame_set(1)

    output = args.output_blend.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    metadata = {
        "output_blend": str(output),
        "camera": camera.name,
        "actor_proxies": proxy_names,
        "actor_proxy_geometry": (
            {
                "type": "fixed_cylinder",
                "radius": 0.18,
                "height": 1.70,
                "position_source": "proxy_center_world",
                "uses_bbox_size": False,
                "uses_posture_scale": False,
            }
            if bbox_representation else
            {
                "type": "legacy_floor_cylinder_with_facing_arrow",
                "uses_posture_scale": True,
            }
        ),
        "frame_start": scene.frame_start,
        "frame_end": scene.frame_end,
        "fps": scene.render.fps,
        "resolution": {"width": width, "height": height},
    }
    args.metadata.resolve().write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
