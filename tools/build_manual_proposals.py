#!/usr/bin/env python3
"""Build manual 3D asset proposals from bbox annotations.

Pipeline:
  bbox annotation
  -> bbox intersect semantic segmentation(category)
  -> COLMAP tracks inside that semantic region
  -> accumulate point3D ids across views with the same asset id
  -> write manual proposal point ids, PLY, and manifest
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


REPO_ROOT = Path(__file__).resolve().parents[1]
SITCOMS3D_ROOT = REPO_ROOT / "sitcoms3D"
if str(SITCOMS3D_ROOT) not in sys.path:
    sys.path.insert(0, str(SITCOMS3D_ROOT))

from sitcoms3D.utils.colmap_utils import read_model  # noqa: E402

DEFAULT_ENV = REPO_ROOT / "sitcoms3D/data/sparse_reconstruction_and_nerf_data/Friends-monica_apartment"
DEFAULT_ANNOTATIONS = REPO_ROOT / "outputs/Friends-monica_apartment/manual_asset_marks/annotations.json"
DEFAULT_PROPOSAL_DIR = REPO_ROOT / "outputs/Friends-monica_apartment/semantic_proposals_instances"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/Friends-monica_apartment/manual_proposals"


def normalize_name(name: str) -> str:
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


def safe_id(value: str) -> str:
    cleaned = "_".join(normalize_name(value).split())
    return cleaned or "asset"


def load_category_labels(proposal_dir: Path) -> Mapping[str, Sequence[Mapping[str, object]]]:
    manifest_path = proposal_dir / "proposals.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing proposal manifest with category labels: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    return manifest["category_labels"]


def semantic_mask(environment_dir: Path, image_name: str, labels: Sequence[Mapping[str, object]]) -> np.ndarray:
    stem = Path(image_name).stem
    stuff_path = environment_dir / "segmentations" / "stuff" / f"{stem}.png"
    thing_path = environment_dir / "segmentations" / "thing" / f"{stem}.png"
    if not stuff_path.exists() or not thing_path.exists():
        raise FileNotFoundError(f"Missing segmentation masks for {image_name}")
    stuff = np.asarray(Image.open(stuff_path))
    thing = np.asarray(Image.open(thing_path))
    if stuff.shape != thing.shape:
        raise ValueError(f"stuff/thing shape mismatch for {image_name}")
    mask = np.zeros(stuff.shape, dtype=bool)
    for item in labels:
        source = str(item["source"])
        label_id = int(item["label_id"])
        if source == "thing":
            mask |= thing == label_id
        elif source == "stuff":
            mask |= (thing == 0) & (stuff == label_id)
        else:
            raise ValueError(f"Unknown label source: {source}")
    return mask


def collect_annotation_points(
    environment_dir: Path,
    image,
    annotation: Mapping[str, object],
    labels: Sequence[Mapping[str, object]],
) -> Tuple[set[int], dict]:
    mask = semantic_mask(environment_dir, str(annotation["source_image"]), labels)
    height, width = mask.shape
    x, y, w, h = [int(v) for v in annotation["bbox_xywh"]]
    x1 = max(0, min(width - 1, x))
    y1 = max(0, min(height - 1, y))
    x2 = max(0, min(width, x + max(0, w)))
    y2 = max(0, min(height, y + max(0, h)))
    if x2 <= x1 or y2 <= y1:
        return set(), {"error": "invalid bbox", "bbox_xywh": annotation["bbox_xywh"]}

    xy = np.asarray(image.xys, dtype=np.float64)
    point_ids = np.asarray(image.point3D_ids, dtype=np.int64)
    px = np.rint(xy[:, 0]).astype(np.int64)
    py = np.rint(xy[:, 1]).astype(np.int64)
    in_image = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    in_bbox = in_image & (point_ids >= 0) & (px >= x1) & (px < x2) & (py >= y1) & (py < y2)
    semantic_track = np.zeros(len(point_ids), dtype=bool)
    semantic_track[in_image] = mask[py[in_image], px[in_image]]
    selected = in_bbox & semantic_track
    ids = set(int(item) for item in point_ids[selected].tolist())
    stats = {
        "source_image": annotation["source_image"],
        "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
        "raw_bbox_track_count": int(np.count_nonzero(in_bbox)),
        "semantic_track_count": int(np.count_nonzero(selected)),
        "semantic_pixel_area_in_bbox": int(np.count_nonzero(mask[y1:y2, x1:x2])),
        "unique_point3d_count": len(ids),
    }
    return ids, stats


def write_ply(path: Path, points3d: Mapping[int, object], point_ids: Sequence[int]) -> None:
    rows = []
    for point_id in point_ids:
        point = points3d.get(int(point_id))
        if point is None:
            continue
        x, y, z = point.xyz
        r, g, b = point.rgb
        rows.append((x, y, z, r, g, b, int(point_id)))
    dtype = np.dtype([
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("point3D_id", "i4"),
    ])
    arr = np.asarray(rows, dtype=dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))


def proposal_metadata(points3d: Mapping[int, object], point_ids: Sequence[int]) -> dict:
    xyz = np.asarray([points3d[int(point_id)].xyz for point_id in point_ids if int(point_id) in points3d], dtype=np.float64)
    if len(xyz) == 0:
        return {"point_count": 0}
    minimum = xyz.min(axis=0)
    maximum = xyz.max(axis=0)
    return {
        "point_count": int(len(xyz)),
        "centroid": xyz.mean(axis=0).tolist(),
        "bbox_min": minimum.tolist(),
        "bbox_max": maximum.tolist(),
        "extent": (maximum - minimum).tolist(),
    }


def build(args: argparse.Namespace) -> dict:
    environment_dir = args.environment_dir.resolve()
    annotations_path = args.annotations.resolve()
    proposal_dir = args.proposal_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = json.loads(annotations_path.read_text()).get("assets", [])
    annotations = [item for item in annotations if item.get("priority") != "reject"]
    category_labels = load_category_labels(proposal_dir)
    _, images, points3d = read_model(str(environment_dir / "colmap"), ext=args.colmap_format)
    image_by_name = {image.name: image for image in images.values()}

    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for item in annotations:
        if item.get("id") and item.get("source_image") and item.get("bbox_xywh"):
            grouped[str(item["id"])].append(item)

    proposals = []
    for asset_id, items in sorted(grouped.items()):
        category = normalize_name(str(items[0].get("category", "")))
        if category not in category_labels:
            print(f"[WARN] skip {asset_id}: unknown category '{category}'")
            continue
        all_ids: set[int] = set()
        view_stats = []
        for item in items:
            image = image_by_name.get(str(item["source_image"]))
            if image is None:
                view_stats.append({"source_image": item.get("source_image"), "error": "image not registered in COLMAP"})
                continue
            ids, stats = collect_annotation_points(environment_dir, image, item, category_labels[category])
            all_ids.update(ids)
            view_stats.append(stats)
        sorted_ids = np.asarray(sorted(all_ids), dtype=np.int64)
        proposal_id = safe_id(asset_id)
        np.save(output_dir / f"{proposal_id}_point3d_ids.npy", sorted_ids)
        write_ply(output_dir / f"{proposal_id}.ply", points3d, sorted_ids)
        metadata = proposal_metadata(points3d, sorted_ids)
        proposals.append({
            "id": proposal_id,
            "asset_id": asset_id,
            "category": category,
            **metadata,
            "point3d_ids": f"{proposal_id}_point3d_ids.npy",
            "pointcloud": f"{proposal_id}.ply",
            "view_count": len(items),
            "views": view_stats,
            "source_annotations": [item.get("uid", item.get("id")) for item in items],
        })
        print(f"[INFO] {proposal_id}: {len(sorted_ids)} points from {len(items)} views")

    manifest = {
        "schema_version": 1,
        "environment_dir": str(environment_dir),
        "annotations": str(annotations_path),
        "category_labels": category_labels,
        "proposal_count": len(proposals),
        "proposals": proposals,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
    return manifest


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build manual asset proposals from 2D bbox annotations.")
    parser.add_argument("--environment-dir", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--proposal-dir", type=Path, default=DEFAULT_PROPOSAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--colmap-format", choices=["", ".bin", ".txt"], default="")
    return parser


def main() -> None:
    manifest = build(make_parser().parse_args())
    print(json.dumps({"proposal_count": manifest["proposal_count"]}, indent=2))


if __name__ == "__main__":
    main()
