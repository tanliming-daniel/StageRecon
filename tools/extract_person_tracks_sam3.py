#!/usr/bin/env python3
"""Track every person in a single-shot frame sequence with one SAM3 session.

The script preserves instance IDs.  It also writes one dilated union mask per
frame for static-scene camera tracking.  A mask value of 255 means dynamic and
must be excluded from background feature extraction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch


DEFAULT_SAM3_ROOT = Path("/root/userdata/300workspace/sam3")
DEFAULT_CHECKPOINT = Path("/root/userdata/300workspace/ckpt/sam3/sam3.pt")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam3-root", type=Path, default=DEFAULT_SAM3_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--version", choices=("sam3", "sam3.1"), default="sam3")
    parser.add_argument("--prompt", default="person")
    parser.add_argument("--prob-threshold", type=float, default=0.45)
    parser.add_argument("--dilate-pixels", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def frame_paths(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def as_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def scalar_at(values, index: int) -> float | None:
    if values is None or len(values) <= index:
        return None
    value = np.asarray(values[index]).reshape(-1)
    return float(value[0]) if len(value) else None


def main() -> None:
    args = parse_args()
    frames_dir = args.frames_dir.resolve()
    output_dir = args.output_dir.resolve()
    sam3_root = args.sam3_root.resolve()
    checkpoint = args.checkpoint.resolve()
    if not frames_dir.is_dir():
        raise NotADirectoryError(frames_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not (sam3_root / "sam3").is_dir():
        raise FileNotFoundError(f"SAM3 package not found under {sam3_root}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("SAM3 video tracking requires CUDA")
    if not 0.0 <= args.prob_threshold <= 1.0:
        raise ValueError("--prob-threshold must be in [0, 1]")
    if args.dilate_pixels < 0:
        raise ValueError("--dilate-pixels must be non-negative")

    frames = frame_paths(frames_dir)
    if not frames:
        raise ValueError(f"No frames found in {frames_dir}")
    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f"Could not decode {frames[0]}")
    height, width = first.shape[:2]

    union_dir = output_dir / "union"
    instance_root = output_dir / "instances"
    union_dir.mkdir(parents=True, exist_ok=True)
    instance_root.mkdir(parents=True, exist_ok=True)

    if str(sam3_root) not in sys.path:
        sys.path.insert(0, str(sam3_root))
    from sam3 import build_sam3_predictor

    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/stagerecon_sam3_cache")
    started = time.perf_counter()
    predictor = build_sam3_predictor(
        checkpoint_path=str(checkpoint),
        version=args.version,
        compile=False,
        async_loading_frames=False,
    )
    response = predictor.handle_request(
        {"type": "start_session", "resource_path": str(frames_dir)}
    )
    session_id = response["session_id"]
    predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": args.prompt,
            "output_prob_thresh": args.prob_threshold,
        }
    )

    tracks: dict[int, list[dict]] = defaultdict(list)
    seen_frames: set[int] = set()
    kernel = None
    if args.dilate_pixels:
        diameter = 2 * args.dilate_pixels + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))

    try:
        stream = predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "output_prob_thresh": args.prob_threshold,
                "evict_cached_frame_outputs": True,
            }
        )
        for item in stream:
            frame_index = int(item.get("frame_index", -1))
            if frame_index < 0 or frame_index >= len(frames) or frame_index in seen_frames:
                continue
            seen_frames.add(frame_index)
            outputs = item.get("outputs", {})
            object_ids = as_numpy(outputs.get("out_obj_ids"))
            masks = as_numpy(outputs.get("out_binary_masks"))
            scores = as_numpy(
                outputs.get("out_probs", outputs.get("out_scores", outputs.get("scores")))
            )
            union = np.zeros((height, width), dtype=np.uint8)
            if object_ids is not None and masks is not None:
                object_ids = np.asarray(object_ids).reshape(-1)
                masks = np.asarray(masks)
                for instance_index, object_id_value in enumerate(object_ids):
                    mask = masks[instance_index]
                    while mask.ndim > 2:
                        mask = mask[0]
                    mask = mask.astype(bool)
                    if mask.shape != (height, width):
                        mask = cv2.resize(
                            mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                    if not mask.any():
                        continue
                    object_id = int(object_id_value)
                    object_dir = instance_root / f"{object_id:04d}"
                    object_dir.mkdir(parents=True, exist_ok=True)
                    mask_path = object_dir / f"{frames[frame_index].stem}.png"
                    cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
                    union[mask] = 255
                    ys, xs = np.where(mask)
                    x1, x2 = int(xs.min()), int(xs.max())
                    y1, y2 = int(ys.min()), int(ys.max())
                    tracks[object_id].append(
                        {
                            "frame_index": frame_index,
                            "frame_name": frames[frame_index].name,
                            "mask": str(mask_path),
                            "bbox_xyxy": [x1, y1, x2, y2],
                            "centroid_xy": [float(xs.mean()), float(ys.mean())],
                            "area_pixels": int(mask.sum()),
                            "score": scalar_at(scores, instance_index),
                        }
                    )
            if kernel is not None:
                union = cv2.dilate(union, kernel, iterations=1)
            cv2.imwrite(str(union_dir / f"{frames[frame_index].stem}.png"), union)
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})

    # The predictor should emit every frame.  Explicit empty masks make a
    # partial result visible to downstream stages rather than shifting names.
    for frame_index, frame in enumerate(frames):
        union_path = union_dir / f"{frame.stem}.png"
        if not union_path.exists():
            cv2.imwrite(str(union_path), np.zeros((height, width), dtype=np.uint8))

    result = {
        "schema_version": 1,
        "frames_dir": str(frames_dir),
        "frame_count": len(frames),
        "image_size": {"width": width, "height": height},
        "prompt": args.prompt,
        "version": args.version,
        "checkpoint": str(checkpoint),
        "prob_threshold": args.prob_threshold,
        "union_dilate_pixels": args.dilate_pixels,
        "emitted_frame_count": len(seen_frames),
        "union_mask_dir": str(union_dir),
        "tracks": [
            {
                "track_id": object_id,
                "frame_count": len(records),
                "first_frame": min(record["frame_index"] for record in records),
                "last_frame": max(record["frame_index"] for record in records),
                "frames": sorted(records, key=lambda record: record["frame_index"]),
            }
            for object_id, records in sorted(tracks.items())
            if records
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "person_tracks.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "frame_count": result["frame_count"],
                "emitted_frame_count": result["emitted_frame_count"],
                "track_count": len(result["tracks"]),
                "track_frame_counts": [track["frame_count"] for track in result["tracks"]],
                "elapsed_seconds": result["elapsed_seconds"],
                "manifest": str(manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
