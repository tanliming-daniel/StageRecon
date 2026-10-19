#!/usr/bin/env python3
"""Create a dynamic-person mask for a query frame with local SAM3 weights.

The output is a single-channel PNG in which 255 denotes pixels that must be
excluded from static-scene localization.  It is intentionally independent from
the localization runner so masks can be inspected or supplied to other tools.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


DEFAULT_SAM3_ROOT = Path("/root/userdata/300workspace/sam3")
DEFAULT_CHECKPOINT = Path("/root/userdata/300workspace/ckpt/sam3/sam3.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sam3-root", type=Path, default=DEFAULT_SAM3_ROOT)
    parser.add_argument("--prompt", default="person")
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--dilate-pixels", type=int, default=12)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def import_sam3(sam3_root: Path):
    sam3_root = sam3_root.resolve()
    if not (sam3_root / "sam3").is_dir():
        raise FileNotFoundError(f"SAM3 source package not found under: {sam3_root}")
    if str(sam3_root) not in sys.path:
        sys.path.insert(0, str(sam3_root))
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    return build_sam3_image_model, Sam3Processor


def write_overlay(image: np.ndarray, dynamic_mask: np.ndarray, path: Path) -> None:
    overlay = image.copy()
    tint = np.zeros_like(overlay)
    tint[:, :, 2] = 255
    selected = dynamic_mask.astype(bool)
    overlay[selected] = cv2.addWeighted(overlay[selected], 0.42, tint[selected], 0.58, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), overlay):
        raise RuntimeError(f"Could not write overlay: {path}")


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be in [0, 1]")
    if args.dilate_pixels < 0:
        raise ValueError("--dilate-pixels must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("SAM3 person masks require CUDA in this pipeline")

    image_path = require_file(args.image, "Input image")
    checkpoint = require_file(args.checkpoint, "SAM3 checkpoint")
    print("sam3_stage=read_image", flush=True)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV could not decode input image: {image_path}")

    print("sam3_stage=import_source", flush=True)
    build_sam3_image_model, Sam3Processor = import_sam3(args.sam3_root)
    started = time.perf_counter()
    print("sam3_stage=build_model", flush=True)
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        device=args.device,
        load_from_HF=False,
    )
    processor = Sam3Processor(
        model,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )
    image_rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    use_bfloat16 = args.device.startswith("cuda") and torch.cuda.is_bf16_supported()
    # SAM3's image forward is designed for CUDA BF16 autocast.  Keeping model
    # weights in FP32 and entering this context lets PyTorch select compatible
    # kernels while preserving the decoder's explicit FP32 sections.
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
        print("sam3_stage=encode_image", flush=True)
        state = processor.set_image(image_rgb)
        print("sam3_stage=text_prompt", flush=True)
        state = processor.set_text_prompt(prompt=args.prompt, state=state)
    masks = state["masks"].detach().cpu().numpy().astype(bool)
    scores = state["scores"].float().detach().cpu().numpy()
    boxes = state["boxes"].float().detach().cpu().numpy()
    if masks.size:
        dynamic_mask = np.any(masks.reshape((-1, masks.shape[-2], masks.shape[-1])), axis=0)
    else:
        dynamic_mask = np.zeros(image_bgr.shape[:2], dtype=bool)
    if args.dilate_pixels:
        diameter = 2 * args.dilate_pixels + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        dynamic_mask = cv2.dilate(dynamic_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print("sam3_stage=write_outputs", flush=True)
    if not cv2.imwrite(str(args.output), dynamic_mask.astype(np.uint8) * 255):
        raise RuntimeError(f"Could not write mask: {args.output}")
    if args.overlay is not None:
        write_overlay(image_bgr, dynamic_mask, args.overlay)

    metadata = {
        "image": str(image_path),
        "output": str(args.output.resolve()),
        "checkpoint": str(checkpoint),
        "prompt": args.prompt,
        "confidence_threshold": args.confidence_threshold,
        "dilate_pixels": args.dilate_pixels,
        "inference_autocast_dtype": "bfloat16" if use_bfloat16 else "float32",
        "image_size": {"width": int(image_bgr.shape[1]), "height": int(image_bgr.shape[0])},
        "detected_instance_count": int(len(scores)),
        "scores": [float(score) for score in scores],
        "boxes_xyxy": [[float(value) for value in box] for box in boxes],
        "dynamic_pixel_count": int(dynamic_mask.sum()),
        "dynamic_area_fraction": float(dynamic_mask.mean()),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    metadata_path = args.metadata or args.output.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
