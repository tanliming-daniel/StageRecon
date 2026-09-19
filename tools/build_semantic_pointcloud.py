#!/usr/bin/env python3
"""Project 2D semantic masks onto a COLMAP point cloud.

The script uses COLMAP tracks to transfer semantic labels from the
``segmentations/stuff`` and ``segmentations/thing`` PNGs onto 3D points.
It writes:

* a labeled point cloud;
* one PLY per requested semantic category;
* optional voxel-connected 3D proposals per category;
* a JSON manifest with metric-free scene proposals and provenance.

The default output stays in COLMAP world coordinates. Use ``--space nerf``
to apply the transform stored in ``cameras.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


REPO_ROOT = Path(__file__).resolve().parents[1]
SITCOMS3D_ROOT = REPO_ROOT / "sitcoms3D"
if str(SITCOMS3D_ROOT) not in sys.path:
    sys.path.insert(0, str(SITCOMS3D_ROOT))

from sitcoms3D.utils.colmap_utils import read_model  # noqa: E402


DEFAULT_CATEGORIES = ("couch", "table", "cabinet", "floor", "wall")
ALIASES = {
    "sofa": ("couch",),
    "dining_table": ("dining table", "table"),
    "table": ("table", "dining table"),
    "dining-table": ("dining table", "table"),
}
PALETTE = {
    "couch": (226, 88, 90),
    "table": (236, 157, 62),
    "cabinet": (71, 145, 212),
    "floor": (92, 181, 102),
    "wall": (160, 113, 194),
}


def normalize_name(name: str) -> str:
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_model_dir(environment_dir: Path, value: str | None) -> Path:
    if value is None:
        return environment_dir / "colmap"
    path = Path(value)
    return path if path.is_absolute() else environment_dir / path


def load_classes(environment_dir: Path) -> Mapping[str, Sequence[str]]:
    path = environment_dir / "panoptic_classes.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing semantic class file: {path}")
    data = json.loads(path.read_text())
    for key in ("stuff", "thing"):
        if key not in data or not isinstance(data[key], list):
            raise ValueError(f"{path} must contain a list named '{key}'")
    return {"stuff": data["stuff"], "thing": data["thing"]}


def build_category_labels(
    classes: Mapping[str, Sequence[str]],
    categories: Sequence[str],
) -> Dict[str, List[Tuple[str, int]]]:
    labels: Dict[str, List[Tuple[str, int]]] = {}
    for requested in categories:
        canonical = normalize_name(requested)
        names = ALIASES.get(canonical, (canonical,))
        matches: List[Tuple[str, int]] = []
        for source in ("thing", "stuff"):
            for label_id, label_name in enumerate(classes[source]):
                if normalize_name(label_name) in names:
                    matches.append((source, label_id))
        if not matches:
            available = sorted(
                {normalize_name(name) for values in classes.values() for name in values}
            )
            raise ValueError(
                f"Category '{requested}' was not found in panoptic_classes.json. "
                f"Available examples: {', '.join(available[:30])}"
            )
        labels[canonical] = matches
    return labels


def image_mask_path(root: Path, source: str, image_name: str) -> Path:
    return root / source / f"{Path(image_name).stem}.png"


def load_mask(path: Path, expected_shape: Tuple[int, int]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing segmentation mask: {path}")
    mask = np.asarray(Image.open(path))
    if mask.ndim != 2:
        raise ValueError(f"Expected a single-channel mask at {path}, got {mask.shape}")
    if tuple(mask.shape) != expected_shape:
        raise ValueError(
            f"Mask/image size mismatch for {path}: mask={mask.shape}, "
            f"expected={expected_shape}"
        )
    return mask


def points_in_space(
    xyz: np.ndarray,
    environment_dir: Path,
    space: str,
) -> Tuple[np.ndarray, dict]:
    if space == "colmap":
        return xyz.copy(), {"name": "colmap_world", "scale_factor": 1.0}

    cameras_path = environment_dir / "cameras.json"
    if not cameras_path.exists():
        raise FileNotFoundError(f"--space nerf requires {cameras_path}")
    cameras = json.loads(cameras_path.read_text())
    transform = np.asarray(cameras["point_cloud_transform"], dtype=np.float64)
    scale_factor = float(cameras["scale_factor"])
    if transform.shape != (4, 4) or scale_factor <= 0:
        raise ValueError(f"Invalid point_cloud_transform/scale_factor in {cameras_path}")
    xyz_h = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=np.float64)], axis=1)
    transformed = (xyz_h @ transform.T)[:, :3] / scale_factor
    return transformed, {
        "name": "sitcoms3D_nerf",
        "scale_factor": scale_factor,
        "point_cloud_transform": transform.tolist(),
    }


def semantic_color(category: str) -> Tuple[int, int, int]:
    if category in PALETTE:
        return PALETTE[category]
    # Stable fallback color for user-provided categories.
    seed = sum((index + 1) * ord(char) for index, char in enumerate(category))
    return (
        64 + (seed * 53) % 160,
        64 + (seed * 97) % 160,
        64 + (seed * 139) % 160,
    )


def write_pointcloud(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    labels: np.ndarray | None = None,
    confidences: np.ndarray | None = None,
    point_ids: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.clip(np.asarray(rgb), 0, 255).astype(np.uint8)
    fields = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    columns: List[np.ndarray] = [xyz[:, 0], xyz[:, 1], xyz[:, 2], rgb[:, 0], rgb[:, 1], rgb[:, 2]]
    if labels is not None:
        fields.append(("label_id", "i4"))
        columns.append(np.asarray(labels, dtype=np.int32))
    if confidences is not None:
        fields.append(("confidence", "f4"))
        columns.append(np.asarray(confidences, dtype=np.float32))
    if point_ids is not None:
        fields.append(("point3D_id", "i4"))
        columns.append(np.asarray(point_ids, dtype=np.int32))

    vertices = np.empty(len(xyz), dtype=np.dtype(fields))
    for field, values in zip(vertices.dtype.names, columns):
        vertices[field] = values
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))


def connected_voxel_clusters(points: np.ndarray, voxel_size: float) -> List[np.ndarray]:
    """Split points by 26-connected occupied voxels without scipy/sklearn."""
    if len(points) == 0:
        return []
    if voxel_size <= 0:
        return [np.arange(len(points), dtype=np.int64)]

    voxels = np.floor(points / voxel_size).astype(np.int64)
    unique_voxels, inverse = np.unique(voxels, axis=0, return_inverse=True)
    voxel_to_index = {tuple(voxel): index for index, voxel in enumerate(unique_voxels)}
    neighbors = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]

    components: List[np.ndarray] = []
    visited = np.zeros(len(unique_voxels), dtype=bool)
    for start in range(len(unique_voxels)):
        if visited[start]:
            continue
        queue = deque([start])
        visited[start] = True
        component_voxels: List[int] = []
        while queue:
            current = queue.popleft()
            component_voxels.append(current)
            base = unique_voxels[current]
            for delta in neighbors:
                neighbor = voxel_to_index.get(
                    (int(base[0] + delta[0]), int(base[1] + delta[1]), int(base[2] + delta[2]))
                )
                if neighbor is not None and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        component_voxels_array = np.asarray(component_voxels, dtype=np.int64)
        components.append(np.flatnonzero(np.isin(inverse, component_voxels_array)))
    return components


def proposal_dict(
    proposal_id: str,
    category: str,
    points: np.ndarray,
    confidences: np.ndarray,
    observed: np.ndarray,
    point_ids_path: str,
    pointcloud_path: str,
) -> dict:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    return {
        "id": proposal_id,
        "category": category,
        "point_count": int(len(points)),
        "centroid": points.mean(axis=0).tolist(),
        "bbox_min": minimum.tolist(),
        "bbox_max": maximum.tolist(),
        "extent": (maximum - minimum).tolist(),
        "mean_confidence": float(confidences.mean()),
        "mean_observations": float(observed.mean()),
        "point3d_ids": point_ids_path,
        "pointcloud": pointcloud_path,
    }


def build(args: argparse.Namespace) -> dict:
    environment_dir = args.environment_dir.resolve()
    model_dir = resolve_model_dir(environment_dir, args.colmap_dir)
    mask_dir = environment_dir / "segmentations"
    output_dir = args.output_dir.resolve() if args.output_dir else environment_dir / "semantic_proposals"
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = load_classes(environment_dir)
    categories = [normalize_name(item) for item in args.categories]
    category_labels = build_category_labels(classes, categories)
    category_to_index = {category: index for index, category in enumerate(categories)}

    model = read_model(str(model_dir), ext=args.colmap_format)
    if model is None:
        raise RuntimeError(f"Could not read COLMAP model from {model_dir}")
    _, images, points3d = model
    point_ids = np.asarray(sorted(points3d.keys()), dtype=np.int64)
    point_index = {int(point_id): index for index, point_id in enumerate(point_ids)}
    xyz_colmap = np.stack([np.asarray(points3d[int(point_id)].xyz) for point_id in point_ids], axis=0)
    rgb = np.stack([np.asarray(points3d[int(point_id)].rgb) for point_id in point_ids], axis=0)
    xyz, space_metadata = points_in_space(xyz_colmap, environment_dir, args.space)

    target_votes = np.zeros((len(point_ids), len(categories)), dtype=np.uint32)
    observed = np.zeros(len(point_ids), dtype=np.uint32)
    processed_images = 0
    skipped_images = 0
    invalid_tracks = 0

    sorted_images = sorted(images.values(), key=lambda image: image.name)
    if args.max_images is not None:
        sorted_images = sorted_images[: args.max_images]

    for image in sorted_images:
        stuff_path = image_mask_path(mask_dir, "stuff", image.name)
        thing_path = image_mask_path(mask_dir, "thing", image.name)
        if not stuff_path.exists() or not thing_path.exists():
            skipped_images += 1
            continue
        stuff_raw = np.asarray(Image.open(stuff_path))
        if args.image_width is not None or args.image_height is not None:
            expected_height = args.image_height or stuff_raw.shape[0]
            expected_width = args.image_width or stuff_raw.shape[1]
            stuff = load_mask(stuff_path, (expected_height, expected_width))
        else:
            stuff = stuff_raw
        thing = load_mask(thing_path, stuff.shape)
        if stuff.shape != thing.shape:
            raise ValueError(f"stuff/thing mask mismatch for {image.name}")

        point3d_ids = np.asarray(image.point3D_ids, dtype=np.int64)
        valid = point3d_ids >= 0
        if not np.any(valid):
            continue
        point3d_ids = point3d_ids[valid]
        xy = np.asarray(image.xys, dtype=np.float64)[valid]
        point_indices = np.asarray([point_index.get(int(point_id), -1) for point_id in point3d_ids])
        valid = point_indices >= 0
        if not np.any(valid):
            invalid_tracks += len(point_indices)
            continue
        point_indices = point_indices[valid]
        xy = xy[valid]
        x = np.rint(xy[:, 0]).astype(np.int64)
        y = np.rint(xy[:, 1]).astype(np.int64)
        in_bounds = (x >= 0) & (x < stuff.shape[1]) & (y >= 0) & (y < stuff.shape[0])
        if not np.any(in_bounds):
            continue
        point_indices = point_indices[in_bounds]
        x = x[in_bounds]
        y = y[in_bounds]
        observed[point_indices] += 1

        stuff_labels = stuff[y, x].astype(np.int64)
        thing_labels = thing[y, x].astype(np.int64)
        for category_index, category in enumerate(categories):
            matches = np.zeros(len(point_indices), dtype=bool)
            for source, label_id in category_labels[category]:
                source_labels = thing_labels if source == "thing" else stuff_labels
                if source == "thing":
                    matches |= source_labels == label_id
                else:
                    # A non-background thing pixel occludes the stuff label.
                    matches |= (thing_labels == 0) & (source_labels == label_id)
            if np.any(matches):
                np.add.at(target_votes[:, category_index], point_indices[matches], 1)
        processed_images += 1
        if processed_images % 25 == 0:
            print(f"[INFO] processed {processed_images}/{len(sorted_images)} images")

    max_votes = target_votes.max(axis=1)
    best_category_index = target_votes.argmax(axis=1)
    confidence = np.divide(
        max_votes,
        np.maximum(observed, 1),
        out=np.zeros_like(max_votes, dtype=np.float32),
    )
    accepted = (
        (observed >= args.min_observations)
        & (max_votes >= args.min_votes)
        & (confidence >= args.min_confidence)
    )
    labels = np.full(len(point_ids), -1, dtype=np.int32)
    labels[accepted] = best_category_index[accepted]

    semantic_rgb = rgb.copy()
    for category_index, category in enumerate(categories):
        semantic_rgb[labels == category_index] = semantic_color(category)
    write_pointcloud(
        output_dir / "semantic_pointcloud.ply",
        xyz[observed > 0],
        semantic_rgb[observed > 0],
        labels=labels[observed > 0],
        confidences=confidence[observed > 0],
        point_ids=point_ids[observed > 0],
    )

    np.savez_compressed(
        output_dir / "point_labels.npz",
        point3d_ids=point_ids,
        labels=labels,
        confidence=confidence,
        observed=observed,
        target_votes=target_votes,
    )

    proposals: List[dict] = []
    proposal_dir = output_dir / "proposals"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    for category_index, category in enumerate(categories):
        category_mask = labels == category_index
        category_indices = np.flatnonzero(category_mask)
        if len(category_indices) == 0:
            print(f"[WARN] no accepted points for category '{category}'")
            continue
        clusters = connected_voxel_clusters(xyz[category_indices], args.cluster_voxel_size)
        clusters = [
            cluster
            for cluster in clusters
            if len(cluster) >= args.min_cluster_points
        ]
        clusters.sort(key=len, reverse=True)
        if args.max_clusters_per_category > 0:
            clusters = clusters[: args.max_clusters_per_category]
        for cluster_index, cluster in enumerate(clusters):
            selected = category_indices[cluster]
            proposal_id = f"{category.replace(' ', '_')}_{cluster_index:03d}"
            cloud_path = proposal_dir / f"{proposal_id}.ply"
            ids_path = proposal_dir / f"{proposal_id}_point3d_ids.npy"
            np.save(ids_path, point_ids[selected])
            write_pointcloud(
                cloud_path,
                xyz[selected],
                rgb[selected],
                labels=labels[selected],
                confidences=confidence[selected],
                point_ids=point_ids[selected],
            )
            proposals.append(
                proposal_dict(
                    proposal_id,
                    category,
                    xyz[selected],
                    confidence[selected],
                    observed[selected],
                    str(ids_path.relative_to(output_dir)),
                    str(cloud_path.relative_to(output_dir)),
                )
            )

    manifest = {
        "schema_version": 1,
        "environment_dir": str(environment_dir),
        "colmap_dir": str(model_dir),
        "segmentation_dir": str(mask_dir),
        "space": space_metadata,
        "categories": categories,
        "category_labels": {
            category: [
                {"source": source, "label_id": int(label_id)}
                for source, label_id in category_labels[category]
            ]
            for category in categories
        },
        "parameters": {
            "min_observations": args.min_observations,
            "min_votes": args.min_votes,
            "min_confidence": args.min_confidence,
            "cluster_voxel_size": args.cluster_voxel_size,
            "min_cluster_points": args.min_cluster_points,
            "max_clusters_per_category": args.max_clusters_per_category,
        },
        "statistics": {
            "num_colmap_points": int(len(point_ids)),
            "num_observed_points": int(np.count_nonzero(observed)),
            "num_accepted_points": int(np.count_nonzero(accepted)),
            "processed_images": int(processed_images),
            "skipped_images": int(skipped_images),
            "invalid_tracks": int(invalid_tracks),
            "accepted_points_by_category": {
                category: int(np.count_nonzero(labels == category_index))
                for category_index, category in enumerate(categories)
            },
        },
        "proposals": proposals,
    }
    (output_dir / "proposals.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n"
    )
    return manifest


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project sitcoms3D 2D semantic masks onto the COLMAP point cloud."
    )
    parser.add_argument(
        "--environment-dir",
        type=Path,
        required=True,
        help="Path to an environment such as sitcoms3D/data/.../Friends-monica_apartment.",
    )
    parser.add_argument(
        "--colmap-dir",
        type=str,
        default=None,
        help="COLMAP model directory, relative to --environment-dir by default.",
    )
    parser.add_argument(
        "--colmap-format",
        choices=["", ".bin", ".txt"],
        default="",
        help="COLMAP format. Empty means auto-detect.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <environment-dir>/semantic_proposals.",
    )
    parser.add_argument(
        "--categories",
        type=parse_csv,
        default=list(DEFAULT_CATEGORIES),
        help="Comma-separated semantic categories.",
    )
    parser.add_argument(
        "--space",
        choices=["colmap", "nerf"],
        default="colmap",
        help="Output coordinate system. COLMAP is recommended for ShowMak3r.",
    )
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--min-votes", type=int, default=2)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument(
        "--cluster-voxel-size",
        type=float,
        default=0.0,
        help="Voxel size for 26-connected proposals. 0 emits one proposal per category.",
    )
    parser.add_argument("--min-cluster-points", type=int, default=10)
    parser.add_argument(
        "--max-clusters-per-category",
        type=int,
        default=0,
        help="Maximum proposals per category. 0 keeps all clusters.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Debug limit. Process only the first N sorted COLMAP images.",
    )
    parser.add_argument(
        "--image-width",
        dest="image_width",
        type=int,
        default=None,
        help="Optional expected segmentation width.",
    )
    parser.add_argument(
        "--image-height",
        dest="image_height",
        type=int,
        default=None,
        help="Optional expected segmentation height.",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.min_observations < 1 or args.min_votes < 1:
        raise ValueError("min-observations and min-votes must be positive")
    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("min-confidence must be in [0, 1]")
    if args.cluster_voxel_size < 0:
        raise ValueError("cluster-voxel-size must be >= 0")
    manifest = build(args)
    print(json.dumps(manifest["statistics"], indent=2))
    print(f"[INFO] wrote proposals to {args.output_dir or args.environment_dir / 'semantic_proposals'}")


if __name__ == "__main__":
    main()
