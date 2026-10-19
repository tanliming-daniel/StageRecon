#!/usr/bin/env python3
"""Build and optionally playblast a saved StageRecon clip render job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_IMPORTER = TOOLS_DIR / "import_clip_animation.py"
DEFAULT_PLAYBLAST = TOOLS_DIR / "render_clip_viewport_playblast.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--base-blend", type=Path)
    parser.add_argument("--blender-executable", default="blender")
    parser.add_argument("--output-blend", type=Path)
    parser.add_argument("--blender-metadata", type=Path)
    parser.add_argument("--importer", type=Path, default=DEFAULT_IMPORTER)
    parser.add_argument("--playblast-script", type=Path, default=DEFAULT_PLAYBLAST)
    parser.add_argument("--playblast", type=Path)
    parser.add_argument("--playblast-metadata", type=Path)
    parser.add_argument("--display", default="")
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=0)
    parser.add_argument("--reuse-scene", action="store_true")
    parser.add_argument("--allow-software-opengl", action="store_true")
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("$", shlex.join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_job_file(job_path: Path, value: str, label: str) -> Path:
    return require_file(job_path.parent / value, label)


def main() -> None:
    args = parse_args()
    job_path = require_file(args.job, "Render job")
    job = json.loads(job_path.read_text())
    if job.get("kind") != "stagerecon_clip_render_job":
        raise ValueError(f"Unsupported render job: {job.get('kind')!r}")
    data = job["data"]
    camera_track = resolve_job_file(job_path, data["camera_track"], "Camera track")
    actor_tracks = resolve_job_file(job_path, data["actor_tracks"], "Actor tracks")
    frames = resolve_job_file(job_path, data["frames"], "Frame metadata")
    for name, path in (
        ("camera_track", camera_track),
        ("actor_tracks", actor_tracks),
        ("frames", frames),
    ):
        expected = data.get("sha256", {}).get(name)
        if expected and file_sha256(path) != expected:
            raise ValueError(f"Checksum mismatch for {name}: {path}")

    suggested_blend = Path(job["source_scene"]["suggested_path"])
    base_blend = require_file(args.base_blend or suggested_blend, "Base Blender scene")
    importer = require_file(args.importer, "Clip animation importer")
    scene_import = job["scene_import"]
    timeline = job["timeline"]
    output_blend = (
        args.output_blend or job_path.parent / "scene_with_clip_animation.blend"
    ).resolve()
    blender_metadata = (
        args.blender_metadata or job_path.parent / "blender_animation.json"
    ).resolve()
    output_blend.parent.mkdir(parents=True, exist_ok=True)
    blender_metadata.parent.mkdir(parents=True, exist_ok=True)

    if not (args.reuse_scene and output_blend.is_file()):
        command = [
            args.blender_executable,
            "--background",
            str(base_blend),
            "--python",
            str(importer),
            "--",
            "--camera-track",
            str(camera_track),
            "--actor-tracks",
            str(actor_tracks),
            "--output-blend",
            str(output_blend),
            "--metadata",
            str(blender_metadata),
            "--point-cloud-object",
            job["coordinate_system"]["alignment_object"],
            "--camera-name",
            scene_import["camera_name"],
            "--fps",
            str(timeline["fps"]),
        ]
        for collection in scene_import.get("hide_collections", []):
            command.extend(("--hide-collection", collection))
        run(command)

    result = {
        "render_job": str(job_path),
        "base_blend": str(base_blend),
        "animated_blend": str(output_blend),
        "blender_metadata": str(blender_metadata),
        "playblast": None,
    }
    if args.playblast is not None:
        playblast_script = require_file(args.playblast_script, "Viewport playblast script")
        playblast = args.playblast.resolve()
        playblast_metadata = (
            args.playblast_metadata or playblast.with_suffix(".json")
        ).resolve()
        preview = job["preview"]
        width = args.width or int(preview["resolution"]["width"])
        height = args.height or int(preview["resolution"]["height"])
        command = [
            args.blender_executable,
            str(output_blend),
            "--python",
            str(playblast_script),
            "--",
            "--output",
            str(playblast),
            "--metadata",
            str(playblast_metadata),
            "--camera-name",
            scene_import["camera_name"],
            "--start-frame",
            str(args.start_frame),
            "--end-frame",
            str(args.end_frame),
            "--width",
            str(width),
            "--height",
            str(height),
        ]
        if not args.allow_software_opengl:
            command.append("--require-hardware-gpu")
        for collection in scene_import.get("hide_collections", []):
            command.extend(("--hide-collection", collection))
        child_env = os.environ.copy()
        if args.display:
            child_env["DISPLAY"] = args.display
            child_env["GHOST_BACKEND"] = "x11"
        run(command, env=child_env)
        result["playblast"] = str(playblast)
        result["playblast_metadata"] = str(playblast_metadata)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
