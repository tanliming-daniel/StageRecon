#!/usr/bin/env python3
"""Localize a new image in an existing COLMAP point-cloud coordinate system.

The input query features must be DISK features generated with the same
``depth-save.pth`` checkpoint as the reference H5 files.  The script matches a
query image to retrieved reference frames, converts their matched keypoints to
existing COLMAP ``point3D_id`` values, then estimates the query pose with
RANSAC PnP.  It never reads a pose for the query image.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SITCOMS3D_ROOT = REPO_ROOT / "sitcoms3D"
if str(SITCOMS3D_ROOT) not in sys.path:
    sys.path.insert(0, str(SITCOMS3D_ROOT))

from sitcoms3D.utils.colmap_utils import read_model  # noqa: E402
from pnp_focal import pnp_with_focal_search  # noqa: E402


DEFAULT_ENVIRONMENT = (
    REPO_ROOT
    / "sitcoms3D/data/sparse_reconstruction_and_nerf_data/Friends-monica_apartment"
)


def parse_candidates(path: Path, count: int, excluded: set[str]) -> list[str]:
    candidates = []
    for line in path.read_text().splitlines()[1:]:
        fields = line.split("\t")
        # Color retrieval emits rank/score/color/structure/image while DISK
        # global retrieval emits rank/score/image.  In both formats the
        # registered COLMAP image name is the final column.
        if len(fields) < 3:
            continue
        image_name = fields[-1]
        if image_name not in excluded:
            candidates.append(image_name)
        if len(candidates) >= count:
            break
    if not candidates:
        raise ValueError("No reference candidates remain after exclusions")
    return candidates


def static_query_indices(keypoints: np.ndarray, mask_path: Path | None) -> np.ndarray:
    if mask_path is None:
        return np.arange(len(keypoints), dtype=np.int64)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    x = np.clip(np.rint(keypoints[:, 0]).astype(np.int64), 0, mask.shape[1] - 1)
    y = np.clip(np.rint(keypoints[:, 1]).astype(np.int64), 0, mask.shape[0] - 1)
    return np.flatnonzero(mask[y, x] == 0)


def static_reference_indices(
    keypoints: np.ndarray,
    mask_path: Path | None,
    person_label: int,
) -> tuple[np.ndarray, bool]:
    """Return reference keypoints outside a known person-class mask.

    The existing Sitcoms3D masks store the human class as value 1. Missing
    masks are expected for part of this scene, and are intentionally treated as
    "not available" rather than silently applying a malformed filter.
    """
    if mask_path is None or not mask_path.is_file():
        return np.ones(len(keypoints), dtype=bool), False
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.ones(len(keypoints), dtype=bool), False
    x = np.clip(np.rint(keypoints[:, 0]).astype(np.int64), 0, mask.shape[1] - 1)
    y = np.clip(np.rint(keypoints[:, 1]).astype(np.int64), 0, mask.shape[0] - 1)
    return mask[y, x] != person_label, True


def mutual_ratio_matches(
    query_descriptors: np.ndarray,
    reference_descriptors: np.ndarray,
    ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return query/ref descriptor indices passing mutual ratio matching."""
    query = torch.from_numpy(query_descriptors).cuda(non_blocking=True)
    reference = torch.from_numpy(reference_descriptors).cuda(non_blocking=True)
    query = torch.nn.functional.normalize(query.float(), dim=1)
    reference = torch.nn.functional.normalize(reference.float(), dim=1)

    # Squared L2 distance for normalized descriptors is 2 - 2 dot product.
    similarity = query @ reference.T
    q_values, q_indices = torch.topk(similarity, k=2, dim=1, largest=True)
    r_values, r_indices = torch.topk(similarity, k=2, dim=0, largest=True)
    q_distance = torch.clamp(2 - 2 * q_values, min=1e-8)
    r_distance = torch.clamp(2 - 2 * r_values, min=1e-8)
    q_pass = q_distance[:, 0] / q_distance[:, 1] < ratio * ratio
    r_pass = r_distance[0] / r_distance[1] < ratio * ratio
    query_ids = torch.arange(len(query), device=query.device)
    reference_ids = q_indices[:, 0]
    mutual = (r_indices[0, reference_ids] == query_ids) & q_pass & r_pass[reference_ids]
    return (
        torch.nonzero(mutual, as_tuple=False).squeeze(1).cpu().numpy(),
        reference_ids[mutual].cpu().numpy(),
    )


def verify_pair_geometry(
    query_keypoints: np.ndarray,
    reference_keypoints: np.ndarray,
    query_ids: np.ndarray,
    reference_ids: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep descriptor matches consistent with one epipolar geometry."""
    if len(query_ids) < 8:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    _, inlier_mask = cv2.findFundamentalMat(
        query_keypoints[query_ids],
        reference_keypoints[reference_ids],
        cv2.FM_RANSAC,
        ransacReprojThreshold=threshold,
        confidence=0.999,
    )
    if inlier_mask is None:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    keep = inlier_mask.reshape(-1).astype(bool)
    return query_ids[keep], reference_ids[keep]



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-h5", type=Path, required=True)
    parser.add_argument("--query-key", required=True)
    parser.add_argument("--query-image", type=Path, required=True)
    parser.add_argument("--candidate-list", type=Path, required=True)
    parser.add_argument("--query-mask", type=Path)
    parser.add_argument(
        "--reference-person-mask-dir",
        type=Path,
        help="Sitcoms3D thing masks; value 1 is excluded as the person class.",
    )
    parser.add_argument("--reference-person-label", type=int, default=1)
    parser.add_argument("--disable-reference-person-mask", action="store_true")
    parser.add_argument("--environment-dir", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--ratio", type=float, default=0.90)
    parser.add_argument("--pair-geometry-threshold", type=float, default=2.0)
    parser.add_argument("--disable-pair-geometry", action="store_true")
    parser.add_argument("--reprojection-error", type=float, default=8.0)
    parser.add_argument("--focal-multipliers", default="0.8,0.9,1.0,1.1,1.2,1.35,1.5,1.7,1.9,2.1")
    parser.add_argument("--focal-pixels", default="", help="Optional comma-separated focal-length candidates in pixels; overrides multiplier grid when provided.")
    parser.add_argument("--exclude-reference", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for efficient DISK reference matching")
    query_image = cv2.imread(str(args.query_image), cv2.IMREAD_COLOR)
    if query_image is None:
        raise FileNotFoundError(args.query_image)
    image_size = (int(query_image.shape[1]), int(query_image.shape[0]))
    excluded = set(args.exclude_reference)
    candidates = parse_candidates(args.candidate_list, args.top_k, excluded)

    with h5py.File(args.query_h5 / "keypoints.h5", "r") as keypoint_h5, h5py.File(
        args.query_h5 / "descriptors.h5", "r"
    ) as descriptor_h5:
        query_keypoints = np.asarray(keypoint_h5[args.query_key], dtype=np.float32)
        query_descriptors = np.asarray(descriptor_h5[args.query_key], dtype=np.float32)
    query_feature_count_before_mask = int(len(query_keypoints))
    keep_query = static_query_indices(query_keypoints, args.query_mask)
    query_keypoints = query_keypoints[keep_query]
    query_descriptors = query_descriptors[keep_query]
    if len(query_keypoints) < 20:
        raise RuntimeError("The query mask leaves too few DISK keypoints")

    environment = args.environment_dir.resolve()
    reference_person_mask_dir = args.reference_person_mask_dir
    if reference_person_mask_dir is None:
        reference_person_mask_dir = environment / "segmentations" / "thing"
    _, images, points3d = read_model(str(environment / "colmap"), ext="")
    images_by_stem = {Path(image.name).stem: image for image in images.values()}
    reference_h5 = environment / "h5"
    observations: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, str]] = {}
    observation_votes: dict[tuple[int, int], int] = defaultdict(int)
    candidate_stats = []
    with h5py.File(reference_h5 / "descriptors.h5", "r") as descriptor_h5, h5py.File(
        reference_h5 / "keypoints.h5", "r"
    ) as reference_keypoint_h5:
        for image_name in candidates:
            stem = Path(image_name).stem
            image = images_by_stem.get(stem)
            if image is None or stem not in descriptor_h5 or stem not in reference_keypoint_h5:
                continue
            query_ids, reference_ids = mutual_ratio_matches(
                query_descriptors, np.asarray(descriptor_h5[stem], dtype=np.float32), args.ratio
            )
            raw_mutual_matches = len(query_ids)
            reference_keypoints = np.asarray(reference_keypoint_h5[stem], dtype=np.float32)
            reference_mask_available = False
            excluded_reference_person_matches = 0
            if not args.disable_reference_person_mask:
                reference_static, reference_mask_available = static_reference_indices(
                    reference_keypoints,
                    reference_person_mask_dir / f"{stem}.png",
                    args.reference_person_label,
                )
                static_match = reference_static[reference_ids]
                excluded_reference_person_matches = int((~static_match).sum())
                query_ids = query_ids[static_match]
                reference_ids = reference_ids[static_match]
            point_ids = np.asarray(image.point3D_ids, dtype=np.int64)
            usable = point_ids[reference_ids] >= 0
            query_ids = query_ids[usable]
            reference_ids = reference_ids[usable]
            if not args.disable_pair_geometry:
                query_ids, reference_ids = verify_pair_geometry(
                    query_keypoints,
                    reference_keypoints,
                    query_ids,
                    reference_ids,
                    args.pair_geometry_threshold,
                )
            valid_matches = 0
            for query_id, reference_id in zip(query_ids, reference_ids):
                point_id = int(point_ids[reference_id])
                point = points3d.get(point_id)
                if point is None:
                    continue
                observations[(int(query_id), point_id)] = (
                    query_keypoints[query_id], np.asarray(point.xyz, dtype=np.float64), image_name
                )
                observation_votes[(int(query_id), point_id)] += 1
                valid_matches += 1
            candidate_stats.append(
                {
                    "image": image_name,
                    "mutual_matches": int(raw_mutual_matches),
                    "reference_person_mask_available": reference_mask_available,
                    "reference_person_matches_excluded": excluded_reference_person_matches,
                    "geometry_inliers": int(len(query_ids)),
                    "matches_with_3d": valid_matches,
                }
            )

    # A query feature must vote for a single scene track. Keeping every
    # descriptor collision creates contradictory 2D-to-many-3D constraints,
    # which is especially damaging in a repetitive sitcom set.
    best_keys: dict[int, tuple[int, int]] = {}
    for key, votes in observation_votes.items():
        query_id = key[0]
        previous = best_keys.get(query_id)
        if previous is None or votes > observation_votes[previous]:
            best_keys[query_id] = key
    selected_observations = [observations[key] for _, key in sorted(best_keys.items())]
    if len(selected_observations) < 6:
        raise RuntimeError(
            f"Only {len(selected_observations)} query features have a unique 3D-track vote"
        )
    xy = np.asarray([item[0] for item in selected_observations], dtype=np.float64)
    xyz = np.asarray([item[1] for item in selected_observations], dtype=np.float64)
    provenance = [item[2] for item in selected_observations]
    if args.focal_pixels.strip():
        focal_values = [float(item) for item in args.focal_pixels.split(",") if item.strip()]
        if not focal_values or any(value <= 0 for value in focal_values):
            raise ValueError("--focal-pixels must contain positive focal lengths")
    else:
        focal_values = [
            float(item) * max(image_size) for item in args.focal_multipliers.split(",") if item.strip()
        ]
    pnp, inlier_ids = pnp_with_focal_search(xyz, xy, image_size, focal_values, args.reprojection_error, adaptive=not bool(args.focal_pixels.strip()))
    rotation, _ = cv2.Rodrigues(np.asarray(pnp["rotation_vector"], dtype=np.float64))
    translation = np.asarray(pnp["translation"], dtype=np.float64)
    center = -rotation.T @ translation

    inlier_sources: dict[str, int] = defaultdict(int)
    for index in inlier_ids:
        inlier_sources[provenance[int(index)]] += 1
    result = {
        "query_image": str(args.query_image),
        "query_key": args.query_key,
        "coordinate_system": "raw_colmap_world (same as environment_dir/colmap/pc.ply)",
        "excluded_reference_images": sorted(excluded),
        "query_feature_count_before_mask": query_feature_count_before_mask,
        "query_static_feature_count": int(len(query_keypoints)),
        "query_dynamic_mask": str(args.query_mask.resolve()) if args.query_mask else None,
        "reference_person_mask_dir": (
            str(reference_person_mask_dir.resolve())
            if not args.disable_reference_person_mask and reference_person_mask_dir.is_dir()
            else None
        ),
        "retrieved_candidate_count": int(len(candidates)),
        "candidate_match_stats": candidate_stats,
        "unique_2d_3d_correspondence_count": int(len(observations)),
        "selected_query_feature_count": int(len(xy)),
        "pnp": pnp,
        "world_to_camera_rotation": rotation.tolist(),
        "world_to_camera_translation": translation.tolist(),
        "camera_center": center.tolist(),
        "inlier_source_images": dict(sorted(inlier_sources.items(), key=lambda item: -item[1])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "query_static_feature_count": result["query_static_feature_count"],
                "selected_query_feature_count": result["selected_query_feature_count"],
                "pnp": result["pnp"],
                "inlier_source_images": result["inlier_source_images"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
