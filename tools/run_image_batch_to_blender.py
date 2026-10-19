#!/usr/bin/env python3
"""Batch localize images and export predicted-camera viewport previews.

Each input image is processed independently by ``run_new_image_to_blender.py``.
The generated camera is ``PredictedCamera_query_0001`` and Blender is never
saved: each item contains only localization artifacts, camera JSON/import code,
and a transient native viewport PNG.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "tools/run_new_image_to_blender.py"
DEFAULT_ENVIRONMENT = (
    REPO_ROOT
    / "sitcoms3D/data/sparse_reconstruction_and_nerf_data/Friends-monica_apartment"
)
DEFAULT_BLEND = DEFAULT_ENVIRONMENT / "scene_with_predicted_camera.blend"
DEFAULT_BLENDER = Path("/root/userdata/300workspace/zdj/blender/blender")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs/Friends-monica_apartment/batch_localization",
    )
    parser.add_argument("--blender-input", type=Path, default=DEFAULT_BLEND)
    parser.add_argument("--environment-dir", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--blender-executable", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--viewport-display", default=":88")
    parser.add_argument("--camera-name", default="PredictedCamera_query_0001")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-features", type=int, default=2048)
    parser.add_argument("--ratio", type=float, default=0.90)
    parser.add_argument("--reprojection-error", type=float, default=8.0)
    parser.add_argument("--focal-pixels", default="")
    parser.add_argument(
        "--person-mask",
        choices=("auto", "off"),
        default="auto",
        help="Use SAM3 to suppress people in each query frame (default: auto).",
    )
    parser.add_argument("--sam3-root", type=Path)
    parser.add_argument("--sam3-checkpoint", type=Path)
    parser.add_argument("--sam3-confidence-threshold", type=float)
    parser.add_argument("--sam3-dilate-pixels", type=int)
    parser.add_argument("--disable-reference-person-mask", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip inputs already marked successful in batch_manifest.json.",
    )
    return parser.parse_args()


def image_paths(input_dir: Path) -> list[Path]:
    return sorted(
        path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if not RUNNER.is_file():
        raise FileNotFoundError(RUNNER)

    inputs = image_paths(input_dir)
    if not inputs:
        raise ValueError(f"No supported images under {input_dir}")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    batch_manifest_path = output_root / "batch_manifest.json"
    previous_items: dict[Path, dict] = {}
    if args.resume and batch_manifest_path.is_file():
        previous_manifest = json.loads(batch_manifest_path.read_text())
        for item in previous_manifest.get("items", []):
            if item.get("status") == "ok" and item.get("query_image"):
                previous_items[Path(item["query_image"]).resolve()] = item
    manifest = {
        "input_dir": str(input_dir),
        "blender_input": str(args.blender_input.resolve()),
        "camera_name": args.camera_name,
        "preview": "native_viewport_texture_studio",
        "items": [],
    }

    for index, image in enumerate(inputs, start=1):
        previous = previous_items.get(image.resolve())
        if previous is not None and not args.overwrite:
            resumed = dict(previous)
            resumed["resumed"] = True
            manifest["items"].append(resumed)
            continue
        item_dir = output_root / f"{index:03d}_{image.stem}"
        command = [
            str(args.python_executable),
            str(RUNNER),
            "--query-image",
            str(image),
            "--blender-input",
            str(args.blender_input),
            "--environment-dir",
            str(args.environment_dir),
            "--output-dir",
            str(item_dir),
            "--blender-executable",
            str(args.blender_executable),
            "--python-executable",
            str(args.python_executable),
            "--viewport-display",
            args.viewport_display,
            "--camera-name",
            args.camera_name,
            "--top-k",
            str(args.top_k),
            "--max-features",
            str(args.max_features),
            "--ratio",
            str(args.ratio),
            "--reprojection-error",
            str(args.reprojection_error),
            "--person-mask",
            args.person_mask,
            "--preview",
        ]
        if args.focal_pixels.strip():
            command.extend(("--focal-pixels", args.focal_pixels))
        if args.sam3_root is not None:
            command.extend(("--sam3-root", str(args.sam3_root)))
        if args.sam3_checkpoint is not None:
            command.extend(("--sam3-checkpoint", str(args.sam3_checkpoint)))
        if args.sam3_confidence_threshold is not None:
            command.extend(("--sam3-confidence-threshold", str(args.sam3_confidence_threshold)))
        if args.sam3_dilate_pixels is not None:
            command.extend(("--sam3-dilate-pixels", str(args.sam3_dilate_pixels)))
        if args.disable_reference_person_mask:
            command.append("--disable-reference-person-mask")
        # Failed/interrupted items already contain their extracted features.
        # The single-image runner can safely refresh those artifacts in place.
        if args.overwrite or (item_dir.exists() and any(item_dir.iterdir())):
            command.append("--overwrite")

        print(f"[{index}/{len(inputs)}] {image}", flush=True)
        completed = subprocess.run(command, check=False)
        item = {
            "index": index,
            "query_image": str(image),
            "output_dir": str(item_dir),
            "returncode": completed.returncode,
        }
        run_manifest = item_dir / "run_manifest.json"
        if completed.returncode == 0 and run_manifest.is_file():
            result = json.loads(run_manifest.read_text())
            item.update(
                {
                    "status": "ok",
                    "camera_info": result["camera_info"],
                    "preview": result["preview"],
                    "camera_center_colmap": result["camera_center_colmap"],
                    "pnp_inlier_count": result["pnp_inlier_count"],
                    "pnp_rmse_pixels": result["pnp_rmse_pixels"],
                    "pose_quality": result.get("pose_quality"),
                }
            )
        else:
            item["status"] = "failed"
        manifest["items"].append(item)
        write_manifest(batch_manifest_path, manifest)

    failed = [item for item in manifest["items"] if item["status"] != "ok"]
    print(f"Batch complete: {len(inputs) - len(failed)}/{len(inputs)} succeeded")
    print(batch_manifest_path)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
