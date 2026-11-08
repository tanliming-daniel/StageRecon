"""Deterministic key-frame sampling with exact frame/time anchors."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

from .io import read_json, relative_path, write_json


def _uniform_indices(frame_count: int, fps: float, sample_fps: float) -> set[int]:
    indices = {0, frame_count - 1}
    if sample_fps <= 0:
        return indices
    step = fps / sample_fps
    cursor = 0.0
    while round(cursor) < frame_count:
        indices.add(min(frame_count - 1, max(0, round(cursor))))
        cursor += step
    return indices


def _limit_indices(indices: list[int], maximum: int) -> list[int]:
    if maximum <= 0 or len(indices) <= maximum:
        return indices
    if maximum == 1:
        return [indices[0]]
    positions = [round(i * (len(indices) - 1) / (maximum - 1)) for i in range(maximum)]
    return [indices[position] for position in positions]


def sample_observations(
    *,
    frames_path: Path,
    frames_dir: Path,
    output_dir: Path,
    caption_root: Path,
    sample_fps: float = 2.0,
    max_observations: int = 16,
    extra_frame_indices: list[int] | None = None,
) -> dict[str, Any]:
    frames_metadata = read_json(frames_path, "frames metadata")
    fps = float(frames_metadata["fps"])
    frame_count = int(frames_metadata["frame_count"])
    indices = _uniform_indices(frame_count, fps, sample_fps)
    indices.update(extra_frame_indices or [])
    valid_indices = sorted(index for index in indices if 0 <= index < frame_count)
    valid_indices = _limit_indices(valid_indices, max_observations)
    records_by_index = {
        int(record["frame_index"]): record for record in frames_metadata["frames"]
    }

    frame_output = output_dir / "frames"
    frame_output.mkdir(parents=True, exist_ok=True)
    observations = []
    for observation_index, frame_index in enumerate(valid_indices, start=1):
        record = records_by_index[frame_index]
        source = frames_dir / str(record["frame_name"])
        if not source.is_file():
            raise FileNotFoundError(f"Extracted frame does not exist: {source}")
        destination = frame_output / source.name
        if not destination.is_file():
            shutil.copy2(source, destination)
        reasons = []
        if frame_index == 0:
            reasons.append("shot_start")
        if frame_index == frame_count - 1:
            reasons.append("shot_end")
        if not reasons:
            reasons.append("uniform")
        observations.append(
            {
                "observation_id": f"OBS_{observation_index:04d}",
                "frame_index": frame_index,
                "timestamp_sec": float(record.get("timestamp_seconds", frame_index / fps)),
                "image": relative_path(destination, caption_root),
                "reason": reasons,
            }
        )
    payload = {
        "sample_fps": sample_fps,
        "max_observations": max_observations,
        "source_frame_count": frame_count,
        "source_fps": fps,
        "observations": observations,
    }
    write_json(output_dir / "keyframes.json", payload)
    return payload
