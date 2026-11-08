"""Prompt assembly for the Omni-first and legacy split proposal calls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import PROMPT_ROOT


def load_prompt(relative: str) -> str:
    path = PROMPT_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f"Prompt does not exist: {path}")
    return path.read_text(encoding="utf-8").strip()


def visual_request_text(
    *, observations: dict[str, Any], bindings: dict[str, Any], duration_sec: float
) -> str:
    reference_bank = [
        {
            "ref_id": item["ref_id"],
            "appearance_image": item["appearance_image"],
            "source_track_id": item["source_track"]["sam3_track_id"],
        }
        for item in bindings.get("bindings", [])
    ]
    observation_index = [
        {
            "observation_id": item["observation_id"],
            "frame_index": item["frame_index"],
            "timestamp_sec": item["timestamp_sec"],
        }
        for item in observations["observations"]
    ]
    payload = {
        "duration_sec": duration_sec,
        "reference_bank": reference_bank,
        "observations": observation_index,
    }
    return (
        "以下是程序提供且不可修改的输入索引。图像将按照 reference_bank 后 "
        "observations 的顺序附加。\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def audio_request_text(
    *, transcript: dict[str, Any], bindings: dict[str, Any], duration_sec: float
) -> str:
    payload = {
        "duration_sec": duration_sec,
        "allowed_ref_ids": [item["ref_id"] for item in bindings.get("bindings", [])],
        "transcript": transcript,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def omni_request_text(
    *, bindings: dict[str, Any], duration_sec: float, fps: float, frame_count: int
) -> str:
    reference_bank = [
        {
            "ref_id": item["ref_id"],
            "appearance_image": item["appearance_image"],
            "canonical_identity": item.get("canonical_identity"),
        }
        for item in bindings.get("bindings", [])
    ]
    payload = {
        "timeline": {
            "duration_sec": duration_sec,
            "fps": fps,
            "frame_count": frame_count,
            "interval_convention": "[start_sec,end_sec)",
        },
        "reference_bank": reference_bank,
        "input_order": {
            "video": "TARGET_VIDEO_WITH_ORIGINAL_AUDIO",
            "reference_images": [item["ref_id"] for item in reference_bank],
        },
    }
    return (
        "Analyze the attached target video and its original audio jointly. "
        "The labeled reference images are the episode candidate bank and define the only "
        "allowed person ref_id values. Match visible people directly from the complete video. "
        "In references, include only candidates who actually appear in this target video; "
        "omit unused candidates. Use selected ref_id values consistently in shots and speakers. "
        "Return the complete qwen_omni_mtss_proposal_v1 JSON object.\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
