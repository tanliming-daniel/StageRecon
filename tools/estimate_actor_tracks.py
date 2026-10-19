#!/usr/bin/env python3
"""Estimate coarse world-space actor tracks without fitting full SMPL motion.

Each SAM3 instance is represented by a floor contact point, a horizontal
facing direction, and a temporally filtered posture label.  The first-frame
absolute camera and the PTZ track put every observation in the existing
COLMAP world coordinate system.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SITCOMS3D_ROOT = REPO_ROOT / "sitcoms3D"
if str(SITCOMS3D_ROOT) not in sys.path:
    sys.path.insert(0, str(SITCOMS3D_ROOT))

from sitcoms3D.utils.colmap_utils import read_model  # noqa: E402


DEFAULT_ENVIRONMENT = (
    REPO_ROOT / "sitcoms3D/data/sparse_reconstruction_and_nerf_data/Friends-monica_apartment"
)
DEFAULT_FLOOR_IDS = (
    REPO_ROOT
    / "outputs/Friends-monica_apartment/semantic_proposals/proposals/floor_000_point3d_ids.npy"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person-tracks", type=Path, required=True)
    parser.add_argument("--camera-track", type=Path, required=True)
    parser.add_argument("--environment-dir", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--floor-point-ids", type=Path, default=DEFAULT_FLOOR_IDS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--posture-window", type=int, default=15)
    parser.add_argument("--position-window", type=int, default=9)
    parser.add_argument("--motion-window", type=int, default=5)
    return parser.parse_args()


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < 2 or window <= 1:
        return values.copy()
    window = min(window, len(values) if len(values) % 2 else len(values) - 1)
    window = max(1, window)
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return values.copy()
    half = window // 2
    padded = np.pad(values, ((half, half), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.stack(
        [np.sum(padded[index:index + window] * kernel[:, None], axis=0) for index in range(len(values))]
    )


def fit_floor(points: np.ndarray, camera_center: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    centroid = np.median(points, axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    normal /= np.linalg.norm(normal)
    if np.dot(normal, camera_center - centroid) < 0:
        normal *= -1
    offset = -float(np.dot(normal, centroid))
    axis_x = np.array([1.0, 0.0, 0.0])
    axis_x -= normal * np.dot(axis_x, normal)
    if np.linalg.norm(axis_x) < 1e-5:
        axis_x = np.array([0.0, 1.0, 0.0])
        axis_x -= normal * np.dot(axis_x, normal)
    axis_x /= np.linalg.norm(axis_x)
    axis_y = np.cross(normal, axis_x)
    axis_y /= np.linalg.norm(axis_y)
    return normal, offset, axis_x, axis_y


def bottom_pixel(record: dict) -> tuple[float, float]:
    mask_path = Path(record["mask"])
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is not None and np.any(mask):
        ys, xs = np.where(mask > 0)
        lower = ys >= np.percentile(ys, 98.0)
        return float(np.median(xs[lower])), float(np.max(ys))
    x1, y1, x2, y2 = record["bbox_xyxy"]
    return 0.5 * (x1 + x2), float(y2)


def ray_floor_intersection(
    pixel: tuple[float, float],
    intrinsic: np.ndarray,
    world_to_camera_rotation: np.ndarray,
    camera_center: np.ndarray,
    plane_normal: np.ndarray,
    plane_offset: float,
) -> np.ndarray | None:
    ray_camera = np.linalg.inv(intrinsic) @ np.array([pixel[0], pixel[1], 1.0])
    ray_world = world_to_camera_rotation.T @ ray_camera
    ray_world /= np.linalg.norm(ray_world)
    denominator = float(np.dot(plane_normal, ray_world))
    if abs(denominator) < 1e-8:
        return None
    distance = -float(np.dot(plane_normal, camera_center) + plane_offset) / denominator
    if distance <= 0:
        return None
    return camera_center + distance * ray_world


def posture_shape_prior(record: dict, image_width: int, image_height: int) -> tuple[float, float, float]:
    x1, y1, x2, y2 = record["bbox_xyxy"]
    box_width = max(1.0, float(x2 - x1 + 1))
    box_height = max(1.0, float(y2 - y1 + 1))
    aspect = box_height / box_width
    # Shape alone is ambiguous for cropped TV shots, so it remains a weak
    # prior.  Track-relative angular height below is the primary cue.
    sitting = float(np.clip((1.18 - aspect) / 0.35, 0.0, 1.0))
    area_fraction = float(record["area_pixels"]) / float(image_width * image_height)
    confidence = float(np.clip(area_fraction / 0.025, 0.15, 1.0))
    if x1 <= 2 or y1 <= 2 or x2 >= image_width - 3 or y2 >= image_height - 3:
        confidence *= 0.55
    return sitting, confidence, aspect


def main() -> None:
    args = parse_args()
    people = json.loads(args.person_tracks.resolve().read_text())
    camera_data = json.loads(args.camera_track.resolve().read_text())
    camera_frames = {int(item["frame_index"]): item for item in camera_data["frames"]}
    camera_center = np.asarray(camera_data["camera_center"], dtype=np.float64)
    image_width = int(camera_data["resolution"]["width"])
    image_height = int(camera_data["resolution"]["height"])

    floor_ids = np.load(args.floor_point_ids.resolve()).astype(np.int64)
    _, _, points3d = read_model(str(args.environment_dir.resolve() / "colmap"), ext="")
    floor_points = np.stack(
        [np.asarray(points3d[int(point_id)].xyz, dtype=np.float64) for point_id in floor_ids if int(point_id) in points3d]
    )
    if len(floor_points) < 3:
        raise RuntimeError("At least three semantic floor points are required")
    normal, offset, floor_x, floor_y = fit_floor(floor_points, camera_center)
    scene_scale = float(np.linalg.norm(np.ptp(floor_points, axis=0)))

    actor_tracks = []
    for track in people.get("tracks", []):
        observations = []
        for record in sorted(track["frames"], key=lambda item: item["frame_index"]):
            frame_index = int(record["frame_index"])
            camera = camera_frames.get(frame_index)
            if camera is None:
                continue
            position = ray_floor_intersection(
                bottom_pixel(record),
                np.asarray(camera["intrinsics"], dtype=np.float64),
                np.asarray(camera["world_to_camera_rotation"], dtype=np.float64),
                np.asarray(camera["camera_center"], dtype=np.float64),
                normal,
                offset,
            )
            shape_sitting, posture_confidence, aspect = posture_shape_prior(
                record, image_width, image_height
            )
            bbox = record["bbox_xyxy"]
            angular_height = float(bbox[3] - bbox[1] + 1) / float(camera["focal_pixels"])
            observations.append(
                {
                    "frame_index": frame_index,
                    "frame_name": record["frame_name"],
                    "bbox_xyxy": record["bbox_xyxy"],
                    "mask": record["mask"],
                    "position": position,
                    "position_source": "floor_ray" if position is not None else "interpolated",
                    "shape_sitting": shape_sitting,
                    "bbox_aspect": aspect,
                    "angular_height": angular_height,
                    "posture_confidence": posture_confidence,
                }
            )
        if not observations:
            continue

        frame_indices = np.asarray(
            [item["frame_index"] for item in observations], dtype=np.float64
        )
        valid_position = np.asarray(
            [item["position"] is not None for item in observations], dtype=bool
        )
        if not np.any(valid_position):
            continue
        known_positions = np.stack(
            [item["position"] for item in observations if item["position"] is not None]
        )
        known_frames = frame_indices[valid_position]
        positions = np.stack(
            [
                np.interp(frame_indices, known_frames, known_positions[:, axis])
                for axis in range(3)
            ],
            axis=1,
        )
        positions = moving_average(positions, args.position_window)
        angular_heights = np.asarray(
            [item["angular_height"] for item in observations], dtype=np.float64
        )
        upright_reference = max(float(np.percentile(angular_heights, 90)), 1e-6)
        relative_heights = angular_heights / upright_reference
        relative_sitting = np.clip((0.84 - relative_heights) / 0.24, 0.0, 1.0)
        shape_sitting = np.asarray(
            [item["shape_sitting"] for item in observations], dtype=np.float64
        )
        raw_scores = np.maximum(relative_sitting, 0.45 * shape_sitting)
        aspects = np.asarray(
            [item["bbox_aspect"] for item in observations], dtype=np.float64
        )
        # A tall visible silhouette is strong evidence for standing even when
        # occlusion or a noisy focal estimate shrinks its track-relative height.
        raw_scores = np.where(aspects >= 1.50, np.minimum(raw_scores, 0.25), raw_scores)
        raw_scores = np.where(
            (aspects >= 1.30) & (aspects < 1.50),
            np.minimum(raw_scores, 0.45),
            raw_scores,
        )
        scores = raw_scores.reshape(-1, 1)
        scores = moving_average(scores, args.posture_window).reshape(-1)
        output_frames = []
        for index, (item, position, sitting_score) in enumerate(zip(observations, positions, scores)):
            lo = max(0, index - args.motion_window)
            hi = min(len(positions) - 1, index + args.motion_window)
            velocity = positions[hi] - positions[lo]
            velocity -= normal * np.dot(velocity, normal)
            motion = float(np.linalg.norm(velocity))
            if motion > max(scene_scale * 0.002, 1e-5):
                facing = velocity / motion
                facing_source = "trajectory"
                facing_confidence = float(np.clip(motion / max(scene_scale * 0.02, 1e-6), 0.35, 0.9))
            else:
                facing = camera_center - position
                facing -= normal * np.dot(facing, normal)
                facing_norm = float(np.linalg.norm(facing))
                if facing_norm < 1e-8:
                    facing = floor_y.copy()
                else:
                    facing /= facing_norm
                facing_source = "camera_facing_prior"
                facing_confidence = 0.2
            yaw = float(np.arctan2(np.dot(facing, floor_y), np.dot(facing, floor_x)))
            posture_confidence = float(item["posture_confidence"])
            if sitting_score >= 0.65:
                posture = "sitting"
                posture_confidence *= float((sitting_score - 0.5) * 2.0)
            elif sitting_score <= 0.35:
                posture = "standing"
                posture_confidence *= float((0.5 - sitting_score) * 2.0)
            else:
                posture = "transition"
                posture_confidence *= 0.5
            output_frames.append(
                {
                    "frame_index": item["frame_index"],
                    "frame_name": item["frame_name"],
                    "mask": item["mask"],
                    "bbox_xyxy": item["bbox_xyxy"],
                    "position_world": position.tolist(),
                    "position_source": item["position_source"],
                    "facing_world": facing.tolist(),
                    "yaw_radians_floor_basis": yaw,
                    "facing_source": facing_source,
                    "facing_confidence": facing_confidence,
                    "posture": posture,
                    "posture_confidence": float(np.clip(posture_confidence, 0.0, 1.0)),
                    "sitting_score": float(sitting_score),
                    "angular_height_relative_to_track_upright": float(
                        item["angular_height"] / upright_reference
                    ),
                    "support_surface": "floor",
                }
            )
        actor_tracks.append(
            {
                "track_id": int(track["track_id"]),
                "first_frame": output_frames[0]["frame_index"],
                "last_frame": output_frames[-1]["frame_index"],
                "frame_count": len(output_frames),
                "frames": output_frames,
            }
        )

    result = {
        "schema_version": 1,
        "coordinate_system": "raw_colmap_world",
        "representation": "floor_contact_plus_facing_and_posture",
        "camera_track": str(args.camera_track.resolve()),
        "person_tracks": str(args.person_tracks.resolve()),
        "floor": {
            "source_point_ids": str(args.floor_point_ids.resolve()),
            "point_count": int(len(floor_points)),
            "normal": normal.tolist(),
            "offset": offset,
            "basis_x": floor_x.tolist(),
            "basis_y": floor_y.tolist(),
            "scene_scale": scene_scale,
        },
        "tracks": actor_tracks,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"track_count": len(actor_tracks), "tracks": [item["frame_count"] for item in actor_tracks]}, indent=2))


if __name__ == "__main__":
    main()
