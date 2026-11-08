"""Package StageRecon-derived geometry without asking the caption model to infer it."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .io import read_json, relative_path, sha256_file, write_json


GEOMETRY_INPUTS = (
    ("camera_trajectory", "camera_track.json"),
    ("actor_root_trajectories", "actor_tracks.json"),
    ("frame_timeline", "frames.json"),
    ("portable_render_job", "render_job.json"),
)


def package_geometry(
    *, clip_run: Path, output_dir: Path, caption_root: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for role, name in GEOMETRY_INPUTS:
        source = clip_run / name
        if not source.is_file():
            continue
        destination = output_dir / name
        if not destination.is_file() or sha256_file(destination) != sha256_file(source):
            shutil.copy2(source, destination)
        files.append(
            {
                "role": role,
                "path": relative_path(destination, caption_root),
                "sha256": sha256_file(destination),
            }
        )

    camera = read_json(clip_run / "camera_track.json", "camera track")
    actors = read_json(clip_run / "actor_tracks.json", "actor tracks")
    coordinate_system = str(
        actors.get("coordinate_system") or camera.get("coordinate_system") or "unknown"
    )
    producer = "+".join(
        value
        for value in (
            str(camera.get("model", "")),
            str(actors.get("representation", "")),
        )
        if value
    )
    geometry = {
        "status": "estimated" if files else "missing",
        "producer": producer or None,
        "coordinate_system": coordinate_system if files else None,
        "files": files,
    }
    write_json(output_dir / "manifest.json", geometry)
    return geometry


def build_blocking_motion(
    *,
    clip_run: Path,
    bindings: list[dict[str, Any]],
    fps: float,
    frame_count: int,
    caption_root: Path,
) -> list[dict[str, Any]]:
    actors = read_json(clip_run / "actor_tracks.json", "actor tracks")
    actor_by_id = {int(track["track_id"]): track for track in actors.get("tracks", [])}
    trajectory_ref = relative_path(
        caption_root / "geometry/actor_tracks.json", caption_root
    )
    motions = []
    for index, binding in enumerate(bindings, start=1):
        track_id = int(binding["source_track"]["sam3_track_id"])
        actor = actor_by_id.get(track_id)
        if not actor:
            continue
        frames = actor.get("frames", [])
        if frames:
            first_frame = max(0, int(frames[0]["frame_index"]))
            last_frame = min(frame_count - 1, int(frames[-1]["frame_index"]))
        else:
            first_frame = max(0, int(actor.get("first_frame", 0)))
            last_frame = min(frame_count - 1, int(actor.get("last_frame", first_frame)))
        end_frame_exclusive = min(frame_count, last_frame + 1)
        motions.append(
            {
                "motion_id": f"BLOCKING_{index}",
                "ref_id": str(binding["ref_id"]),
                "source_track_id": track_id,
                "time_range": {
                    "start_frame": first_frame,
                    "end_frame_exclusive": end_frame_exclusive,
                    "start_sec": first_frame / fps,
                    "end_sec": end_frame_exclusive / fps,
                    "temporal_precision": "frame_exact",
                },
                "trajectory_ref": trajectory_ref,
                "meaning": (
                    "Actor root position and route only; proxy limb pose, facial "
                    "expression, and animation artifacts are not target performance."
                ),
            }
        )
    return motions
