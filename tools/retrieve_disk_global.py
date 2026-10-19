#!/usr/bin/env python3
"""Retrieve registered frames using a cached pooled-DISK descriptor index."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
SITCOMS3D_ROOT = REPO_ROOT / "sitcoms3D"
import sys

if str(SITCOMS3D_ROOT) not in sys.path:
    sys.path.insert(0, str(SITCOMS3D_ROOT))

from sitcoms3D.utils.colmap_utils import read_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-h5", type=Path, required=True)
    parser.add_argument("--query-key", default="query")
    parser.add_argument(
        "--query-mask",
        type=Path,
        help="Optional dynamic-object mask. Non-zero query pixels are excluded.",
    )
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--environment-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with h5py.File(args.query_h5 / "descriptors.h5", "r") as descriptor_h5, h5py.File(
        args.query_h5 / "keypoints.h5", "r"
    ) as keypoint_h5:
        descriptors = np.asarray(descriptor_h5[args.query_key], dtype=np.float32)
        keypoints = np.asarray(keypoint_h5[args.query_key], dtype=np.float32)
    if len(descriptors) != len(keypoints):
        raise RuntimeError("Query DISK descriptor/keypoint counts do not match")
    static = np.ones(len(keypoints), dtype=bool)
    if args.query_mask is not None:
        mask = cv2.imread(str(args.query_mask), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read query mask: {args.query_mask}")
        x = np.clip(np.rint(keypoints[:, 0]).astype(np.int64), 0, mask.shape[1] - 1)
        y = np.clip(np.rint(keypoints[:, 1]).astype(np.int64), 0, mask.shape[0] - 1)
        static = mask[y, x] == 0
    if int(static.sum()) < 20:
        raise RuntimeError("The query mask leaves too few DISK descriptors for global retrieval")
    query = descriptors[static].mean(axis=0)
    query /= max(float(np.linalg.norm(query)), 1e-12)
    index = np.load(args.index)
    names = index["names"]
    vectors = index["vectors"].astype(np.float32, copy=False)
    _, registered_images, _ = read_model(str(args.environment_dir / "colmap"), ext="")
    registered_stems = {Path(image.name).stem for image in registered_images.values()}
    keep = np.asarray([str(name) in registered_stems for name in names], dtype=bool)
    names = names[keep]
    vectors = vectors[keep]
    scores = vectors @ query
    selected = np.argsort(-scores)[: args.top_k]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["rank\tscore\timage"]
    lines.extend(f"{rank}\t{scores[item]:.8f}\t{names[item]}.jpg" for rank, item in enumerate(selected, 1))
    args.output.write_text("\n".join(lines) + "\n")
    print(f"static_query_descriptors={int(static.sum())}/{len(static)}")
    print(args.output)


if __name__ == "__main__":
    main()
