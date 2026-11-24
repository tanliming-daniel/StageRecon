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
    parser.add_argument("--no-save-blend", action="store_true", help="Import in memory for validation/rendering without duplicating the base scene.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--point-cloud-object", default="pointcloud")
    parser.add_argument("--camera-name", default="PredictedClipCamera")
    parser.add_argument("--actor-proxy-style", choices=("cylinder", "head-cylinder"), default="head-cylinder")
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


def proxy_pose(item: dict, camera: dict, convention: dict, alignment: Matrix):
    """Keep fixed physical dimensions and place the TOP at the observed head.

    Legacy bundles encoded their centre using per-frame camera up. Recover
    the head before switching to scene up; otherwise tilt produces x/y drift.
    """
    height = float(convention.get("height_world", 1.70))
    radius = float(convention.get("radius_world", 0.18))
    if height <= 0 or radius <= 0:
        raise ValueError("Proxy dimensions must be positive")
    scales = alignment.to_scale()
    scale = sum(scales) / 3.0
    if scale <= 0 or max(abs(v - scale) for v in scales) > 1e-5 * scale:
        raise ValueError("Pointcloud alignment must have a positive uniform scale")
    if alignment.to_3x3().determinant() <= 0:
        raise ValueError("Pointcloud alignment must preserve orientation")
    if "head_anchor_world" in item:
        head = Vector(item["head_anchor_world"])
    elif "proxy_center_world" in item:
        old_up = -Matrix(camera["world_to_camera_rotation"]).transposed().col[1]
        head = Vector(item["proxy_center_world"]) + 0.5 * height * old_up
    else:
        # Older bbox-only data has no head anchor: use
        # the visible box top as the only available anchor.
        box = item["bbox_3d"]
        old_up = -Matrix(box["axes_world"]).col[1]
        head = Vector(box["center_world"]) + 0.5 * box["size_world"][1] * old_up
    raw_up = item.get("proxy_up_world", convention.get("up_world"))
    up = world_direction(raw_up, alignment) if raw_up is not None else Vector((0, 0, 1))
    center = world_point(head, alignment) - 0.5 * height * scale * up
    forward = world_direction(item.get("facing_world", [1, 0, 0]), alignment)
    forward -= up * forward.dot(up)
    if forward.length < 1e-6:
        forward = up.orthogonal()
    forward.normalize()
    side = up.cross(forward).normalized()
    basis = Matrix((forward, side, up)).transposed()
    return center, basis, scale, height, radius


def actor_material(track_id: int):
    name = f"ClipActorMaterial_{track_id:04d}"
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    import colorsys
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
    colors.extend([(0.96,.33,.64,1),(.42,.30,.12,1),(.67,.88,.12,1),(.22,.26,.90,1)])
    if 0 <= track_id < len(colors):
        material.diffuse_color = colors[track_id]
    else:
        hue = (track_id * 0.61803398875) % 1.0
        material.diffuse_color = (*colorsys.hsv_to_rgb(hue, 0.85, 0.58), 1.0)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    if shader is not None:
        shader.inputs["Base Color"].default_value = material.diffuse_color
        shader.inputs["Roughness"].default_value = 0.75
    return material


def main() -> None:
    args = parse_args()
    camera_track = json.loads(args.camera_track.resolve().read_text())
    actor_tracks = json.loads(args.actor_tracks.resolve().read_text())
    pointcloud = bpy.data.objects.get(args.point_cloud_object)
    if pointcloud is None:
        raise RuntimeError(f"Missing alignment object: {args.point_cloud_object}")
    if camera_track.get("coordinate_system") == "omega_ba_world":
        if actor_tracks.get("coordinate_system") != "omega_ba_world":
            raise ValueError("Camera and actor tracks use different coordinate systems")
        frame = camera_track.get("scene_frame")
        if frame:
            expected = Matrix(frame["world_to_blender"])
            if max(abs(expected[r][c]-pointcloud.matrix_world[r][c]) for r in range(4) for c in range(4)) > 1e-5:
                raise ValueError("The Blender pointcloud transform differs from the checked Omega scene")
            pointcloud["coordinate_system"] = "omega_ba_world"
            bpy.context.scene["coordinate_system"] = "omega_ba_world"
        if pointcloud.get("coordinate_system") != "omega_ba_world":
            raise ValueError("Omega tracks require the Omega-aligned base Blender scene")
    alignment = pointcloud.matrix_world.copy()

    remove_generated_animation(args.camera_name)
    # Authoring copies of the apartment have visible set pieces disabled for render.
    frame_config = camera_track.get("scene_frame", {})
    if "render_objects" in frame_config:
        for name in frame_config["render_objects"]:
            obj = bpy.context.scene.objects.get(name)
            if obj is None:
                raise ValueError(f"Missing configured render object: {name}")
            obj.hide_render = False
    else:
        # Bundles exported before scene layouts carried visibility overrides.
        scene_hash = camera_track.get("scene_frame", {}).get("source_blend_sha256")
        has_tv_doors = scene_hash in {"3c6206e48e7da4458256f16d276f3c3b71419f5cbc19a1da663a23ef791a8e4e",
                                      "366923a76853bd956b0db8ed5e829245c6c998750b600834345d569ae135a457"}
        if has_tv_doors or scene_hash == "61edcc2915340c6d8e1e2a1dc05aa577640ebc6bfc11356347543c61bd63f52c":
            for name in ("Layout_Window_Console", "Layout_Window_Desk_Chair"):
                bpy.data.objects[name].hide_render = False
        if has_tv_doors:
            for obj in bpy.context.scene.objects:
                if obj.name.startswith(("Layout_TV_Left_Door", "Layout_TV_Left_Transom_", "Layout_TV_Purple_Door_")):
                    obj.hide_render = False

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
    cameras_by_frame = {int(c["frame_index"]): c for c in camera_track["frames"]}
    pixel_ratios = [float(c["intrinsics"][0][0]) / float(c["intrinsics"][1][1]) for c in camera_track["frames"]]
    scene = bpy.context.scene
    aspect = sorted(pixel_ratios)[len(pixel_ratios)//2]
    varying = max(pixel_ratios)-min(pixel_ratios)>1e-6
    if varying and camera_track.get("model") != "da3_with_review_only_omega_joint_anchors":
        raise ValueError("Verified camera has changing pixel aspect")
    # Blender fixes output pixel aspect for the shot. Joint-model review
    # cameras retain their measured fx and use the shot median fx/fy.
    scene.render.pixel_aspect_x = max(1., 1/aspect)
    scene.render.pixel_aspect_y = max(1., aspect)
    max_edge_error = max(abs(aspect/r-1)*height/2 for r in pixel_ratios)
    for item in camera_track["frames"]:
        blender_frame = int(item["frame_index"]) + 1
        camera.matrix_world = camera_matrix_blender(
            item["world_to_camera_rotation"], item["camera_center"], alignment
        )
        k = item["intrinsics"]
        fx, fy = float(k[0][0]), float(k[1][1])
        camera.data.lens = fx * 36.0 / width
        camera.data.shift_x = (width / 2.0 - float(k[0][2])) / width
        camera.data.shift_y = (float(k[1][2]) - height / 2.0) * aspect / width
        camera.data.keyframe_insert(data_path="shift_x", frame=blender_frame)
        camera.data.keyframe_insert(data_path="shift_y", frame=blender_frame)
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
    convention = actor_tracks.get("proxy_convention", {})
    proxy_names = []
    skipped_proxy_tracks = []
    for track in actor_tracks.get("tracks", []):
        if not track.get("proxy_quality", {}).get("eligible", True):
            skipped_proxy_tracks.append({"track_id": int(track["track_id"]), "quality": track["proxy_quality"]})
            continue
        track_id = int(track["track_id"])
        material = actor_material(track_id)
        root_name = f"ClipActor_{track_id:04d}"
        root = bpy.data.objects.get(root_name) or bpy.data.objects.new(root_name, None)
        link_only(root, actor_collection)
        root.rotation_mode = "QUATERNION"
        root["track_id"] = track_id
        root["actor_track"] = str(args.actor_tracks.resolve())

        seated = bool(track.get("seat_contact"))
        if bbox_representation:
            proxy_height = float(convention.get("height_world", 1.70))
            proxy_radius = float(track.get("proxy_radius_world",convention.get("radius_world", 0.18)))
            head_radius = float(track.get("proxy_head_radius_world", min(0.07 * proxy_height, 0.8 * proxy_radius)))
            head_style = args.actor_proxy_style == "head-cylinder"
            body_height = proxy_height - 2 * head_radius if head_style else proxy_height
            bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=proxy_radius, depth=body_height)
            body = bpy.context.object
            body.name = f"{root_name}_Body"
            body.display_type = "SOLID"
            body.color = material.diffuse_color
            link_only(body, actor_collection)
            body.parent = root
            body.location = (0.0, 0.0, -head_radius if head_style else 0.0)
            body.rotation_euler = (0.0, 0.0, 0.0)
            body.data.materials.append(material)
            if head_style:
                bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=head_radius)
                head = bpy.context.object
                head.name = f"{root_name}_Head"
                link_only(head, actor_collection)
                head.parent = root
                head.location = (0, 0, 0.5 * proxy_height - head_radius)
                head.data.materials.append(material)
                for polygon in head.data.polygons:
                    polygon.use_smooth = True
                # A shallow cream band follows the sphere's FRONT (+local X).
                # It rotates with the actor basis; no protruding nose/arrow.
                vertices, faces = [], []
                columns, rows = 24, 4
                marker_radius = head_radius * 1.012
                for row in range(rows + 1):
                    latitude = 0.08 + (row / rows - 0.5) * 0.36
                    for column in range(columns + 1):
                        longitude = (column / columns - 0.5) * 1.45
                        vertices.append((
                            marker_radius * math.cos(latitude) * math.cos(longitude),
                            marker_radius * math.cos(latitude) * math.sin(longitude),
                            marker_radius * math.sin(latitude),
                        ))
                for row in range(rows):
                    for column in range(columns):
                        index = row * (columns + 1) + column
                        faces.append((index, index + 1, index + columns + 2, index + columns + 1))
                mesh = bpy.data.meshes.new(f"{root_name}_FacingBandMesh")
                mesh.from_pydata(vertices, [], faces)
                mesh.update()
                marker = bpy.data.objects.new(f"{root_name}_Facing", mesh)
                actor_collection.objects.link(marker)
                marker.parent = root
                marker.location = head.location.copy()
                marker["meaning"] = "front_of_head_positive_local_x"
                cream = bpy.data.materials.get("ClipActorFacingBand") or bpy.data.materials.new("ClipActorFacingBand")
                cream.diffuse_color = (0.90, 0.88, 0.72, 1.0)
                cream.use_nodes = True
                shader = cream.node_tree.nodes.get("Principled BSDF")
                if shader is not None:
                    shader.inputs["Base Color"].default_value = cream.diffuse_color
                    shader.inputs["Roughness"].default_value = 0.75
                marker.color = cream.diffuse_color
                marker.data.materials.append(cream)
                for polygon in mesh.polygons:
                    polygon.use_smooth = True
        else:
            bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.18, depth=1.70)
            body = bpy.context.object
            body.name = f"{root_name}_Body"
            link_only(body, actor_collection)
            body.parent = root
            body.location = (0.0, 0.0, 0.85)
            body.data.materials.append(material)

            bpy.ops.mesh.primitive_cone_add(vertices=16, radius1=0.12, radius2=0.0, depth=0.55)
            arrow = bpy.context.object
            arrow.name = f"{root_name}_Facing"
            link_only(arrow, actor_collection)
            arrow.parent = root
            arrow.location = (0.28, 0.0, 1.30)
            arrow.rotation_euler[1] = math.pi / 2.0
            arrow.data.materials.append(material)

        previous_orientation = None
        previous_head = None
        for item in track["frames"]:
            animate_head = bbox_representation and head_style and "head_rotation_world" in item
            if not item.get("animation_keyframe", True) and not animate_head:
                continue
            blender_frame = int(item["frame_index"]) + 1
            if bbox_representation:
                position, basis, proxy_scale, _, _ = proxy_pose(
                    item, cameras_by_frame[int(item["frame_index"])], convention, alignment
                )
                root.matrix_world = Matrix.Translation(position) @ basis.to_4x4()
                if seated or "body_height_world" in item:
                    if "body_height_world" in item:
                        height_above_bottom = item["body_height_world"]
                    else:
                        up = basis.col[2]
                        head_top = world_point(item["head_anchor_world"], alignment)
                        pelvis = world_point(item["pelvis_world"], alignment)
                        units = camera_track["scene_frame"]["units_per_meter"]
                        height_above_bottom = (head_top - pelvis).dot(up) / proxy_scale + .16 * units
                    units = camera_track["scene_frame"]["units_per_meter"]
                    top = .5 * proxy_height - (2 * head_radius if head_style else 0)
                    bottom = .5 * proxy_height - height_above_bottom
                    body.scale.z = max(.02 * units, top-bottom) / body_height
                    body.location.z = top - .5 * body_height * body.scale.z
                    body.keyframe_insert(data_path="scale", frame=blender_frame)
                    body.keyframe_insert(data_path="location", frame=blender_frame)
            else:
                position = world_point(item["position_world"], alignment)
                basis = actor_basis(
                    floor_x, floor_y, floor_normal, float(item["yaw_radians_floor_basis"])
                )
                root.matrix_world = Matrix.Translation(position) @ basis.to_4x4()
            root.scale = (proxy_scale,) * 3 if bbox_representation else (1.0, 1.0, 1.0)
            if previous_orientation is not None:
                rotation = root.rotation_quaternion.copy()
                rotation.make_compatible(previous_orientation)
                root.rotation_quaternion = rotation
            previous_orientation = root.rotation_quaternion.copy()
            if animate_head:
                desired = alignment.to_quaternion() @ Matrix(item["head_rotation_world"]).to_quaternion()
                relative = basis.to_quaternion().inverted() @ desired
                if previous_head is not None:
                    relative.make_compatible(previous_head)
                for part in (head, marker):
                    part.rotation_mode = "QUATERNION"
                    part.rotation_quaternion = relative
                    part.keyframe_insert(data_path="rotation_quaternion", frame=blender_frame)
                previous_head = relative.copy()
            root.keyframe_insert(data_path="location", frame=blender_frame)
            root.keyframe_insert(data_path="rotation_quaternion", frame=blender_frame)
            if not bbox_representation:
                posture_scale = 0.56 if item["posture"] == "sitting" else 1.0
                body.scale = (1.0, 1.0, posture_scale)
                body.location.z = 0.85 * posture_scale
                body.keyframe_insert(data_path="scale", frame=blender_frame)
                body.keyframe_insert(data_path="location", frame=blender_frame)
        proxy_names.append(root_name)

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = len(camera_track["frames"])
    scene.render.fps = max(1, int(round(args.fps)))
    scene.render.fps_base = scene.render.fps / args.fps
    # Frame observations are samples, not Bezier control points.
    for block in (camera, camera.data, scene, *list(actor_collection.objects)):
        if block.animation_data and block.animation_data.action:
            action = block.animation_data.action
            if hasattr(action, "fcurves"):
                curves = list(action.fcurves)
            else:
                curves = [curve for layer in action.layers for strip in layer.strips
                          for bag in strip.channelbags for curve in bag.fcurves]
            for curve in curves:
                for key in curve.keyframe_points:
                    key.interpolation = "LINEAR" if curve.data_path != "scale" or block.name.endswith("_Body") else "CONSTANT"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene["clip_camera_track"] = str(args.camera_track.resolve())
    scene["clip_actor_tracks"] = str(args.actor_tracks.resolve())
    scene.frame_set(1)

    output = args.output_blend.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(output))
    metadata = {
        "output_blend": None if args.no_save_blend else str(output),
        "camera": camera.name,
        "fixed_pixel_aspect_fx_fy": aspect,
        "max_vertical_edge_error_source_pixels": max_edge_error,
        "actor_proxies": proxy_names,
        "seated_proxies": [t["track_id"] for t in actor_tracks["tracks"] if t.get("seat_contact")],
        "skipped_proxy_tracks": skipped_proxy_tracks,
        "actor_proxy_geometry": (
            {
                "type": "fixed_head_cylinder" if args.actor_proxy_style == "head-cylinder" else "fixed_cylinder",
                "radius": float(convention.get("radius_world", 0.18)),
                "height": float(convention.get("height_world", 1.70)),
                "size_policy": "fixed_for_entire_clip",
                "facing_marker": "cream_horizontal_spherical_band" if args.actor_proxy_style == "head-cylinder" else None,
                "anchor": "head_top_world",
                "position_source": "head_anchor_world_or_recovered_legacy_head",
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
        "fps": scene.render.fps / scene.render.fps_base,
        "resolution": {"width": width, "height": height},
    }
    if metadata["seated_proxies"]:
        metadata["actor_proxy_geometry"] = {
            "type": "seat_center_head_cylinder", "seated_anchor": "pelvis_world",
            "seated_head": "head_anchor_world", "other_tracks": args.actor_proxy_style,
        }
    args.metadata.resolve().write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
