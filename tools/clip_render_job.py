#!/usr/bin/env python3
"""Write a portable render job from saved clip camera and actor data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def read_json(path: Path, label: str) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def relative_file(path: Path, job_dir: Path) -> str:
    return os.path.relpath(path.resolve(), job_dir.resolve())


def actor_proxy_settings(actor_tracks: dict) -> dict:
    if actor_tracks.get("pose_estimator") == "sam3_multihmr_visibility_v1":
        return {"type": "fitted_head_torso", "orientation": "visibility_aware"}
    if "3d_bbox" not in str(actor_tracks.get("representation", "")):
        return {
            "type": "legacy_floor_cylinder_with_facing_arrow",
            "uses_posture_scale": True,
        }
    frames = next(
        (
            track.get("frames", [])
            for track in actor_tracks.get("tracks", [])
            if track.get("frames")
        ),
        [],
    )
    position_source = (
        "proxy_center_world"
        if frames and "proxy_center_world" in frames[0]
        else "bbox_3d.center_world"
    )
    return {
        "type": "fixed_head_cylinder",
        "facing_marker": "cream_horizontal_spherical_band",
        "radius": float(actor_tracks.get("proxy_convention", {}).get("radius_world", 0.18)),
        "height": float(actor_tracks.get("proxy_convention", {}).get("height_world", 1.70)),
        "size_policy": "fixed_for_entire_clip",
        "anchor": "head_top_world",
        "position_source": position_source,
        "uses_bbox_size": False,
        "uses_posture_scale": False,
    }


def validate_tracks(camera_track: dict, actor_tracks: dict, frames: dict) -> int:
    camera_frames = camera_track.get("frames", [])
    if not camera_frames:
        raise ValueError("Camera track has no frames")
    frame_count = len(camera_frames)
    declared_count = int(frames.get("frame_count", frame_count))
    if declared_count != frame_count:
        raise ValueError(
            f"frames.json declares {declared_count} frames but "
            f"camera_track.json has {frame_count}"
        )
    indices = [int(item["frame_index"]) for item in camera_frames]
    if indices != list(range(frame_count)):
        raise ValueError("Camera frame indices must be contiguous and start at zero")
    for track in actor_tracks.get("tracks", []):
        for item in track.get("frames", []):
            frame_index = int(item["frame_index"])
            if not 0 <= frame_index < frame_count:
                raise ValueError(
                    f"Actor track {track.get('track_id')} has out-of-range "
                    f"frame {frame_index}"
                )
    return frame_count


def write_render_job(
    *,
    output: Path,
    camera_track_path: Path,
    actor_tracks_path: Path,
    frames_path: Path,
    base_blend: Path,
    camera_name: str = "PredictedClipCamera",
    point_cloud_object: str = "pointcloud",
    hide_collections: tuple[str, ...] = ("actor1", "camera"),
    preview_width: int = 960,
    preview_height: int = 540,
) -> dict:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    camera_track_path = camera_track_path.resolve()
    actor_tracks_path = actor_tracks_path.resolve()
    frames_path = frames_path.resolve()
    camera_track = read_json(camera_track_path, "Camera track")
    actor_tracks = read_json(actor_tracks_path, "Actor tracks")
    frames = read_json(frames_path, "Frame metadata")
    frame_count = validate_tracks(camera_track, actor_tracks, frames)
    source_resolution = camera_track.get("resolution") or frames.get("resolution")
    if not source_resolution:
        raise ValueError("Camera track and frame metadata do not contain a resolution")
    job = {
        "schema_version": 1,
        "kind": "stagerecon_clip_render_job",
        "assumption": "one_clip_is_one_shot",
        "data": {
            "camera_track": relative_file(camera_track_path, output.parent),
            "actor_tracks": relative_file(actor_tracks_path, output.parent),
            "frames": relative_file(frames_path, output.parent),
            "sha256": {
                "camera_track": sha256(camera_track_path),
                "actor_tracks": sha256(actor_tracks_path),
                "frames": sha256(frames_path),
            },
        },
        "coordinate_system": {
            "camera": camera_track.get("coordinate_system"),
            "actors": actor_tracks.get("coordinate_system"),
            "alignment_object": point_cloud_object,
        },
        "source_scene": {
            "suggested_path": str(base_blend.resolve()),
            "filename": base_blend.name,
            "supply_on_render_machine": True,
        },
        "timeline": {
            "source_frame_range_inclusive": [0, frame_count - 1],
            "blender_frame_range_inclusive": [1, frame_count],
            "frame_count": frame_count,
            "fps": float(frames.get("fps", 30.0)),
        },
        "scene_import": {
            "camera_name": camera_name,
            "hide_collections": list(hide_collections),
            "actor_proxy": actor_proxy_settings(actor_tracks),
            "active_camera": camera_name,
        },
        "preview": {
            "resolution": {"width": preview_width, "height": preview_height},
            "source_resolution": {
                "width": int(source_resolution["width"]),
                "height": int(source_resolution["height"]),
            },
            "shading": {
                "type": "SOLID",
                "light": "STUDIO",
                "color_type": "TEXTURE",
                "show_shadows": False,
                "show_cavity": False,
                "show_specular_highlight": False,
            },
        },
    }
    output.write_text(json.dumps(job, indent=2) + "\n")
    return job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--base-blend", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--camera-name", default="PredictedClipCamera")
    parser.add_argument("--point-cloud-object", default="pointcloud")
    parser.add_argument("--hide-collection", action="append", default=[])
    parser.add_argument("--preview-width", type=int, default=960)
    parser.add_argument("--preview-height", type=int, default=540)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clip_dir = args.clip_dir.resolve()
    output = (args.output or clip_dir / "render_job.json").resolve()
    job = write_render_job(
        output=output,
        camera_track_path=clip_dir / "camera_track.json",
        actor_tracks_path=clip_dir / "actor_tracks.json",
        frames_path=clip_dir / "frames.json",
        base_blend=args.base_blend,
        camera_name=args.camera_name,
        point_cloud_object=args.point_cloud_object,
        hide_collections=tuple(args.hide_collection or ("actor1", "camera")),
        preview_width=args.preview_width,
        preview_height=args.preview_height,
    )
    print(json.dumps({"render_job": str(output), **job["timeline"]}, indent=2))


if __name__ == "__main__":
    main()
