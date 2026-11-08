#!/usr/bin/env python3
"""Long-lived SAM3 worker used by export_episode_character_gallery.py."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROTOCOL_PREFIX = "__STAGERECON_SAM3__"


def emit(payload: dict[str, Any]) -> None:
    print(PROTOCOL_PREFIX + json.dumps(payload), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam3-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--prompt", default="person")
    return parser.parse_args()


def choose_person_mask(
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    face_bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray | None, float | None, list[float] | None]:
    x, y, width, height = face_bbox
    centre_x, centre_y = x + 0.5 * width, y + 0.55 * height
    ranked: list[tuple[float, np.ndarray, float, list[float]]] = []
    for mask, box, score in zip(masks, boxes, scores):
        while mask.ndim > 2:
            mask = mask[0]
        binary = mask > 0.5
        x1, y1, x2, y2 = [float(value) for value in box]
        px = min(binary.shape[1] - 1, max(0, int(round(centre_x))))
        py = min(binary.shape[0] - 1, max(0, int(round(centre_y))))
        contains = x1 <= centre_x <= x2 and y1 <= centre_y <= y2
        hits_face = bool(binary[py, px])
        if not (hits_face or contains):
            continue
        box_area = max(1.0, (x2 - x1) * (y2 - y1))
        specificity = min(1.0, (width * height * 20.0) / box_area)
        rank = float(score) + 0.22 * specificity + 0.35 * hits_face + 0.15 * contains
        ranked.append((rank, binary, float(score), [x1, y1, x2, y2]))
    if not ranked:
        return None, None, None
    _, mask, score, box = max(ranked, key=lambda item: item[0])
    return mask, score, box


def completeness_metrics(
    mask: np.ndarray,
    face_bbox: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
) -> dict[str, Any]:
    ys, xs = np.where(mask)
    image_height, image_width = image_shape[:2]
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    _, face_y, _, face_height = face_bbox
    face_height = max(1, face_height)
    height_ratio = (y2 - y1) / face_height
    below_face_ratio = max(0.0, y2 - (face_y + face_height)) / face_height
    margin = max(3, int(round(min(image_height, image_width) * 0.008)))
    touches_top = y1 <= margin
    touches_bottom = y2 >= image_height - margin
    touches_side = x1 <= margin or x2 >= image_width - margin
    completeness = (
        0.58 * min(1.0, height_ratio / 7.5)
        + 0.24 * min(1.0, below_face_ratio / 5.5)
        + 0.10 * float(not touches_bottom)
        + 0.05 * float(not touches_top)
        + 0.03 * float(not touches_side)
    )
    return {
        "person_bbox_xyxy": [x1, y1, x2, y2],
        "person_to_face_height_ratio": round(float(height_ratio), 6),
        "below_face_height_ratio": round(float(below_face_ratio), 6),
        "touches_top": touches_top,
        "touches_bottom": touches_bottom,
        "touches_side": touches_side,
        "full_body_likely": bool(
            height_ratio >= 5.5 and below_face_ratio >= 3.7
        ),
        "completeness_score": round(float(completeness), 6),
    }


def torso_colour_histogram(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    face_bbox: tuple[int, int, int, int],
) -> list[float] | None:
    x, y, width, height = face_bbox
    image_height, image_width = image_bgr.shape[:2]
    x1 = max(0, int(round(x - 0.75 * width)))
    x2 = min(image_width, int(round(x + 1.75 * width)))
    y1 = max(0, int(round(y + 0.90 * height)))
    y2 = min(image_height, int(round(y + 4.20 * height)))
    if x2 <= x1 or y2 <= y1:
        return None
    torso_mask = mask[y1:y2, x1:x2]
    pixels = image_bgr[y1:y2, x1:x2][torso_mask]
    if len(pixels) < 80:
        return None
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    histogram, _, _ = np.histogram2d(
        hsv[:, 0], hsv[:, 1], bins=(12, 4), range=((0, 180), (0, 256))
    )
    vector = histogram.astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return None
    return (vector / norm).round(7).tolist()


def crop_rgb_from_mask(
    image_bgr: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Keep full source height and crop only to the SAM mask width."""

    if not np.any(mask):
        raise ValueError("empty person mask")
    _, xs = np.where(mask)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    return image_bgr[:, x1:x2].copy()


def main() -> None:
    args = parse_args()
    root = args.sam3_root.resolve()
    sys.path.insert(0, str(root))
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint.resolve()),
        load_from_HF=False,
        device=args.device,
        eval_mode=True,
        compile=False,
    )
    processor = Sam3Processor(
        model, device=args.device, confidence_threshold=args.threshold
    )
    emit({"status": "ready"})
    for raw_line in sys.stdin:
        try:
            request = json.loads(raw_line)
            if request.get("command") == "close":
                emit({"status": "closed"})
                return
            image_path = Path(request["image_path"])
            output_path = Path(request["output_path"])
            face_bbox = tuple(int(value) for value in request["face_bbox_xywh"])
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                emit({"ok": False, "error": f"could not read {image_path}"})
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if args.device.startswith("cuda")
                else nullcontext()
            )
            with context:
                state = processor.set_image(Image.fromarray(image_rgb))
                state = processor.set_text_prompt(prompt=args.prompt, state=state)
            mask, score, _box = choose_person_mask(
                state["masks"].detach().float().cpu().numpy(),
                state["boxes"].detach().float().cpu().numpy(),
                state["scores"].detach().float().cpu().numpy(),
                face_bbox,
            )
            if mask is None:
                emit({"ok": False, "error": "no person mask matched face"})
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), crop_rgb_from_mask(image_bgr, mask)):
                raise RuntimeError(f"could not write {output_path}")
            emit(
                {
                    "ok": True,
                    "score": score,
                    "torso_colour_histogram": torso_colour_histogram(
                        image_bgr, mask, face_bbox
                    ),
                    **completeness_metrics(mask, face_bbox, image_bgr.shape),
                }
            )
        except Exception as error:
            emit({"ok": False, "error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
