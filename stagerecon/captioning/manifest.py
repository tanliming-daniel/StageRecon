"""Normalize a StageRecon clip run into a captioning input manifest."""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from .io import file_asset, read_json, relative_path, sha256_file, write_json


def _probe_target_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,nb_frames,duration:format=duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    value = json.loads(result.stdout)
    streams = value.get("streams", [])
    if not streams:
        raise ValueError(f"Target video has no video stream: {path}")
    stream = streams[0]
    rate = str(stream.get("avg_frame_rate") or "0/1")
    fps = float(Fraction(rate)) if rate not in {"0/0", "N/A"} else 0.0
    duration_value = value.get("format", {}).get("duration") or stream.get("duration")
    if duration_value in {None, "N/A"}:
        raise ValueError(f"Target video duration is unavailable: {path}")
    duration = float(duration_value)
    frame_value = stream.get("nb_frames")
    frame_count = int(frame_value) if frame_value not in {None, "N/A"} else int(round(duration * fps))
    if fps <= 0 or frame_count <= 0 or duration <= 0:
        raise ValueError(
            f"Invalid target video timing: fps={fps}, frames={frame_count}, duration={duration}"
        )
    return {"fps": fps, "frame_count": frame_count, "duration_sec": duration}


def _existing_path(value: str | None, fallback: Path) -> Path:
    if value:
        candidate = Path(value)
        if candidate.is_file():
            return candidate.resolve()
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(f"Required clip-run file does not exist: {fallback}")


def load_clip_run(clip_run: Path) -> dict[str, Any]:
    clip_run = clip_run.resolve()
    frames_path = clip_run / "frames.json"
    frames = read_json(frames_path, "frames metadata")
    manifest_path = clip_run / "clip_manifest.json"
    manifest = read_json(manifest_path, "clip manifest") if manifest_path.is_file() else {}

    video_value = manifest.get("video") or frames.get("video")
    if not video_value:
        raise ValueError("Neither clip_manifest.json nor frames.json specifies the video")
    source_video = Path(str(video_value)).resolve()
    if not source_video.is_file():
        raise FileNotFoundError(f"Source video does not exist: {source_video}")

    return {
        "clip_run": clip_run,
        "clip_manifest_path": manifest_path if manifest_path.is_file() else None,
        "clip_manifest": manifest,
        "frames_path": frames_path,
        "frames": frames,
        "frames_dir": clip_run / "frames",
        "source_video": source_video,
        "person_tracks_path": _existing_path(
            manifest.get("person_tracks"), clip_run / "masks/person_tracks.json"
        ),
        "camera_track_path": _existing_path(
            manifest.get("camera_track"), clip_run / "camera_track.json"
        ),
        "actor_tracks_path": _existing_path(
            manifest.get("actor_tracks"), clip_run / "actor_tracks.json"
        ),
        "render_job_path": (
            _existing_path(manifest.get("render_job"), clip_run / "render_job.json")
            if (manifest.get("render_job") or (clip_run / "render_job.json").is_file())
            else None
        ),
    }


def build_input_manifest(
    *,
    clip_run: Path,
    previs_video: Path,
    output_dir: Path,
    sample_id: str | None = None,
    sequence_id: str | None = None,
    target_video: Path | None = None,
    audio: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    loaded = load_clip_run(clip_run)
    output_dir = output_dir.resolve()
    previs_video = previs_video.resolve()
    if not previs_video.is_file():
        raise FileNotFoundError(f"Previs video does not exist: {previs_video}")
    target_video = (target_video or loaded["source_video"]).resolve()
    if not target_video.is_file():
        raise FileNotFoundError(f"Target video does not exist: {target_video}")
    audio = (audio or target_video).resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"Audio does not exist: {audio}")

    frames = loaded["frames"]
    target_timeline = _probe_target_video(target_video)
    fps = float(target_timeline["fps"])
    frame_count = int(target_timeline["frame_count"])
    duration = float(target_timeline["duration_sec"])
    sample_id = sample_id or loaded["clip_run"].name

    manifest = {
        "sample_id": sample_id,
        "sequence_id": sequence_id,
        "clip_run": str(loaded["clip_run"]),
        "inputs": {
            "source_video": str(loaded["source_video"]),
            "previs_video": str(previs_video),
            "target_video": str(target_video),
            "audio": str(audio),
            "frames": str(loaded["frames_path"]),
            "person_tracks": str(loaded["person_tracks_path"]),
            "camera_track": str(loaded["camera_track_path"]),
            "actor_tracks": str(loaded["actor_tracks_path"]),
            "render_job": str(loaded["render_job_path"]) if loaded["render_job_path"] else None,
        },
        "input_hashes": {
            "source_video": sha256_file(loaded["source_video"]),
            "previs_video": sha256_file(previs_video),
            "target_video": sha256_file(target_video),
            "audio": sha256_file(audio),
            "frames": sha256_file(loaded["frames_path"]),
            "person_tracks": sha256_file(loaded["person_tracks_path"]),
            "camera_track": sha256_file(loaded["camera_track_path"]),
            "actor_tracks": sha256_file(loaded["actor_tracks_path"]),
        },
        "timeline": {
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": duration,
            "time_unit": "seconds",
            "interval_convention": "[start_sec,end_sec)",
        },
        "stage_geometry_timeline": {
            "fps": float(frames["fps"]),
            "frame_count": int(frames["frame_count"]),
            "duration_sec": float(
                frames.get("duration_seconds", int(frames["frame_count"]) / float(frames["fps"]))
            ),
        },
    }
    assets = {
        "source_video": file_asset(loaded["source_video"], output_dir),
        "previs_video": file_asset(
            previs_video, output_dir, role="coarse_3d_previs"
        ),
        "target_video": file_asset(target_video, output_dir),
        "audio": file_asset(audio, output_dir, role="original_audio_source"),
    }
    write_json(output_dir / "input_manifest.json", manifest)
    return manifest, assets
