#!/usr/bin/env python3
"""Recover a single-shot PTZ camera track from one absolute first-frame pose.

The expensive scene lookup and 2D-to-3D matching happen only for frame zero.
Every later frame is related to its predecessor with masked background KLT
tracks.  A pure-rotation plus focal-change model is fitted from the robust
homography, keeping the first-frame camera centre fixed for the whole clip.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--anchor-localization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-features", type=int, default=1600)
    parser.add_argument("--min-tracks", type=int, default=24)
    parser.add_argument("--ransac-threshold", type=float, default=2.5)
    parser.add_argument("--focal-step-limit", type=float, default=0.05)
    parser.add_argument("--focal-grid-size", type=int, default=31)
    parser.add_argument("--smooth-window", type=int, default=9)
    return parser.parse_args()


def image_paths(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode {path}")
    return image


def load_static_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    dynamic = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if dynamic is None:
        return np.full(shape, 255, dtype=np.uint8)
    if dynamic.shape != shape:
        dynamic = cv2.resize(dynamic, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    static = np.where(dynamic == 0, 255, 0).astype(np.uint8)
    static[:8] = 0
    static[-8:] = 0
    static[:, :8] = 0
    static[:, -8:] = 0
    return static


def background_tracks(
    previous: np.ndarray,
    current: np.ndarray,
    previous_static: np.ndarray,
    current_static: np.ndarray,
    max_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    corners = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=max_features,
        qualityLevel=0.008,
        minDistance=7,
        mask=previous_static,
        blockSize=7,
        useHarrisDetector=False,
    )
    if corners is None or len(corners) < 8:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    lk = dict(
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    forward, status_forward, _ = cv2.calcOpticalFlowPyrLK(previous, current, corners, None, **lk)
    if forward is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None, **lk)
    if backward is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    source = corners.reshape(-1, 2)
    target = forward.reshape(-1, 2)
    backward = backward.reshape(-1, 2)
    valid = status_forward.reshape(-1).astype(bool) & status_backward.reshape(-1).astype(bool)
    valid &= np.linalg.norm(backward - source, axis=1) < 1.5
    height, width = current.shape
    x = np.rint(target[:, 0]).astype(np.int64)
    y = np.rint(target[:, 1]).astype(np.int64)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    valid &= inside
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices):
        valid[valid_indices] &= current_static[y[valid_indices], x[valid_indices]] > 0
    return source[valid], target[valid]


def closest_rotation(matrix: np.ndarray) -> np.ndarray:
    determinant = float(np.linalg.det(matrix))
    scale = np.cbrt(abs(determinant)) if abs(determinant) > 1e-12 else 1.0
    normalized = matrix / scale
    u, _, vt = np.linalg.svd(normalized)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def project_rotation(
    points: np.ndarray,
    rotation: np.ndarray,
    previous_intrinsic: np.ndarray,
    current_intrinsic: np.ndarray,
) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    rays = (np.linalg.inv(previous_intrinsic) @ homogeneous.T).T
    rotated = (rotation @ rays.T).T
    projected = (current_intrinsic @ rotated.T).T
    return projected[:, :2] / np.maximum(projected[:, 2:3], 1e-9)


def fit_ptz_step(
    source: np.ndarray,
    target: np.ndarray,
    previous_intrinsic: np.ndarray,
    principal_point: tuple[float, float],
    focal_step_limit: float,
    focal_grid_size: int,
    ransac_threshold: float,
) -> tuple[np.ndarray, float, dict] | None:
    if len(source) < 8:
        return None
    homography, inlier_mask = cv2.findHomography(
        source, target, cv2.RANSAC, ransac_threshold, confidence=0.999
    )
    if homography is None or inlier_mask is None:
        return None
    keep = inlier_mask.reshape(-1).astype(bool)
    source = source[keep]
    target = target[keep]
    if len(source) < 8:
        return None

    previous_focal = float(previous_intrinsic[0, 0])
    cx, cy = principal_point
    best = None
    for log_ratio in np.linspace(-focal_step_limit, focal_step_limit, focal_grid_size):
        focal = previous_focal * float(np.exp(log_ratio))
        current_intrinsic = np.array(
            [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        normalized_h = np.linalg.inv(current_intrinsic) @ homography @ previous_intrinsic
        rotation = closest_rotation(normalized_h)
        predicted = project_rotation(source, rotation, previous_intrinsic, current_intrinsic)
        residuals = np.linalg.norm(predicted - target, axis=1)
        score = float(np.median(residuals))
        # A tiny prior avoids focal jitter when neighboring grid points explain
        # the same background tracks equally well.
        score += 0.03 * abs(float(log_ratio)) * previous_focal
        if best is None or score < best[0]:
            best = (score, rotation, focal, residuals)
    if best is None:
        return None
    _, rotation, focal, residuals = best
    return rotation, focal, {
        "track_count": int(len(inlier_mask)),
        "homography_inlier_count": int(keep.sum()),
        "median_reprojection_error": float(np.median(residuals)),
        "p90_reprojection_error": float(np.percentile(residuals, 90)),
    }


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values.copy()
    window = min(window, len(values) if len(values) % 2 else len(values) - 1)
    window = max(1, window)
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return values.copy()
    half = window // 2
    weights = np.arange(1, half + 2, dtype=np.float64)
    weights = np.concatenate([weights, weights[-2::-1]])
    weights /= weights.sum()
    padded = np.pad(values, ((half, half), (0, 0)), mode="edge")
    result = np.empty_like(values, dtype=np.float64)
    for index in range(len(values)):
        result[index] = np.sum(padded[index:index + window] * weights[:, None], axis=0)
    result[0] = values[0]
    return result


def main() -> None:
    args = parse_args()
    frames = image_paths(args.frames_dir.resolve())
    if not frames:
        raise ValueError(f"No frames found under {args.frames_dir}")
    masks = [args.mask_dir.resolve() / f"{frame.stem}.png" for frame in frames]
    anchor = json.loads(args.anchor_localization.resolve().read_text())
    anchor_intrinsic = np.asarray(anchor["pnp"]["intrinsics"], dtype=np.float64)
    anchor_rotation = np.asarray(anchor["world_to_camera_rotation"], dtype=np.float64)
    camera_center = np.asarray(anchor["camera_center"], dtype=np.float64)

    first_gray = load_gray(frames[0])
    height, width = first_gray.shape
    cx = float(anchor_intrinsic[0, 2])
    cy = float(anchor_intrinsic[1, 2])
    rotations = [anchor_rotation]
    focals = [float(anchor_intrinsic[0, 0])]
    quality = [{
        "status": "anchor",
        "pnp_inlier_count": int(anchor["pnp"]["inlier_count"]),
        "pnp_rmse_pixels": float(anchor["pnp"]["inlier_rmse_pixels"]),
    }]

    previous_gray = first_gray
    previous_static = load_static_mask(masks[0], first_gray.shape)
    for frame_index in range(1, len(frames)):
        current_gray = load_gray(frames[frame_index])
        current_static = load_static_mask(masks[frame_index], current_gray.shape)
        source, target = background_tracks(
            previous_gray,
            current_gray,
            previous_static,
            current_static,
            args.max_features,
        )
        previous_intrinsic = np.array(
            [[focals[-1], 0.0, cx], [0.0, focals[-1], cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        fitted = fit_ptz_step(
            source,
            target,
            previous_intrinsic,
            (cx, cy),
            args.focal_step_limit,
            args.focal_grid_size,
            args.ransac_threshold,
        )
        if fitted is None or fitted[2]["homography_inlier_count"] < args.min_tracks:
            rotations.append(rotations[-1].copy())
            focals.append(focals[-1])
            quality.append({
                "status": "held",
                "track_count": int(len(source)),
                "homography_inlier_count": int(
                    fitted[2]["homography_inlier_count"] if fitted is not None else 0
                ),
            })
        else:
            relative_rotation, focal, metrics = fitted
            rotations.append(relative_rotation @ rotations[-1])
            focals.append(float(focal))
            quality.append({"status": "tracked", **metrics})
        previous_gray = current_gray
        previous_static = current_static

    relative_vectors = []
    for rotation in rotations:
        relative = rotation @ anchor_rotation.T
        vector, _ = cv2.Rodrigues(relative)
        relative_vectors.append(vector.reshape(3))
    relative_vectors = smooth_series(np.asarray(relative_vectors), args.smooth_window)
    log_focal = np.log(np.asarray(focals) / focals[0]).reshape(-1, 1)
    log_focal = smooth_series(log_focal, args.smooth_window).reshape(-1)
    log_focal[0] = 0.0

    smoothed_rotations = []
    for vector in relative_vectors:
        relative, _ = cv2.Rodrigues(vector.astype(np.float64))
        smoothed_rotations.append(relative @ anchor_rotation)
    smoothed_rotations[0] = anchor_rotation
    smoothed_focals = focals[0] * np.exp(log_focal)
    smoothed_focals[0] = focals[0]

    track_frames = []
    for frame_index, (frame, rotation, focal, frame_quality) in enumerate(
        zip(frames, smoothed_rotations, smoothed_focals, quality)
    ):
        translation = -rotation @ camera_center
        intrinsic = np.array(
            [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        track_frames.append(
            {
                "frame_index": frame_index,
                "frame_name": frame.name,
                "intrinsics": intrinsic.tolist(),
                "world_to_camera_rotation": rotation.tolist(),
                "world_to_camera_translation": translation.tolist(),
                "camera_center": camera_center.tolist(),
                "focal_pixels": float(focal),
                "quality": frame_quality,
            }
        )

    total_relative = smoothed_rotations[-1] @ anchor_rotation.T
    total_vector, _ = cv2.Rodrigues(total_relative)
    tracked_count = sum(item["status"] == "tracked" for item in quality)
    result = {
        "schema_version": 1,
        "model": "fixed_camera_center_ptz",
        "coordinate_system": "raw_colmap_world",
        "anchor_frame_index": 0,
        "anchor_frame_name": frames[0].name,
        "anchor_localization": str(args.anchor_localization.resolve()),
        "resolution": {"width": width, "height": height},
        "camera_center": camera_center.tolist(),
        "summary": {
            "frame_count": len(frames),
            "tracked_transition_count": tracked_count,
            "held_transition_count": len(frames) - 1 - tracked_count,
            "total_rotation_degrees": float(np.linalg.norm(total_vector) * 180.0 / np.pi),
            "focal_min_pixels": float(np.min(smoothed_focals)),
            "focal_max_pixels": float(np.max(smoothed_focals)),
            "focal_ratio": float(np.max(smoothed_focals) / np.min(smoothed_focals)),
        },
        "frames": track_frames,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
