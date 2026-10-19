#!/usr/bin/env python3
"""Crop a Blender UI screenshot to its VIEW_3D region and resize it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata.resolve().read_text())
    image = Image.open(args.input).convert("RGB")
    sx = image.width / metadata["window_width"]
    sy = image.height / metadata["window_height"]
    left = round(metadata["viewport_x"] * sx)
    top = round((metadata["window_height"] - metadata["viewport_y"] - metadata["viewport_height"]) * sy)
    right = round((metadata["viewport_x"] + metadata["viewport_width"]) * sx)
    bottom = round((metadata["window_height"] - metadata["viewport_y"]) * sy)
    viewport = image.crop((left, top, right, bottom))
    viewport = viewport.resize((args.width, args.height), Image.Resampling.LANCZOS)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    viewport.save(output)
    print(f"preview_saved={output} source_crop=({left},{top},{right},{bottom})")


if __name__ == "__main__":
    main()
