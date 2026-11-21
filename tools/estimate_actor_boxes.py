#!/usr/bin/env python3
"""Estimate tracked 3D person boxes from SAM3 masks and DA3 depth.

This estimator does not use foot points or a floor intersection. For every
SAM3 observation it obtains a robust foreground depth from the mask, uses the
tracked 2D box to estimate visible width and height at that depth, and projects
the resulting camera-oriented 3D box into the tracked camera's world frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person-tracks", type=Path, required=True)
    parser.add_argument("--camera-track", type=Path, required=True)
    parser.add_argument(
        "--depth-prediction",
        type=Path,
        help="DA3 prediction .npz. Defaults to camera_track.da3.raw_output.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--position-window", "--depth-window", dest="depth_window", type=int, default=9)
    parser.add_argument("--size-window", type=int, default=9)
    parser.add_argument(
        "--bbox-window",
        type=int,
        default=7,
        help="Rolling-median window for the 2D box used by 3D localization.",
    )
    parser.add_argument(
        "--proxy-height",
        type=float,
        default=1.70,
        help="Fixed proxy height used to place its centre below the head ray.",
    )
    parser.add_argument(
        "--surface-to-center-offset",
        type=float,
        default=0.36,
        help=(
            "Move the actor centre this many world units behind the visible "
            "DA3 body surface along the viewing ray."
        ),
    )
    parser.add_argument("--proxy-radius", type=float, default=0.18)
    parser.add_argument(
        "--world-up", type=float, nargs=3,
        help="Calibrated COLMAP world up. Otherwise use one fixed clip camera-up prior.",
    )
    parser.add_argument("--posture-window", type=int, default=15)
    parser.add_argument("--motion-window", type=int, default=5)
    return parser.parse_args()


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < 2 or window <= 1:
        return values.copy()
    maximum = len(values) if len(values) % 2 else len(values) - 1
    window = min(window if window % 2 else window - 1, maximum)
    if window <= 1:
        return values.copy()
    half = window // 2
    padding = [(half, half)] + [(0, 0)] * (values.ndim - 1)
    padded = np.pad(values, padding, mode="edge")
    kernel_shape = (-1,) + (1,) * (values.ndim - 1)
    kernel = (np.ones(window, dtype=np.float64) / window).reshape(kernel_shape)
    return np.stack(
        [np.sum(padded[index:index + window] * kernel, axis=0) for index in range(len(values))]
    )


def moving_median(values: np.ndarray, window: int) -> np.ndarray:
    """Robust temporal filter that removes isolated segmentation-box spikes."""
    if len(values) < 2 or window <= 1:
        return values.copy()
    maximum = len(values) if len(values) % 2 else len(values) - 1
    window = min(window if window % 2 else window - 1, maximum)
    if window <= 1:
        return values.copy()
    half = window // 2
    padding = [(half, half)] + [(0, 0)] * (values.ndim - 1)
    padded = np.pad(values, padding, mode="edge")
    return np.stack(
        [np.median(padded[index:index + window], axis=0) for index in range(len(values))]
    )


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    midpoint = 0.5 * float(np.sum(sorted_weights))
    return float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), midpoint)])


def robust_body_core(
    record: dict,
    image_size: tuple[int, int],
) -> dict:
    """Find a torso-led body axis while rejecting thin mask protrusions."""
    image_width, image_height = image_size
    raw = cv2.imread(str(record["mask"]), cv2.IMREAD_GRAYSCALE)
    x1, y1, x2, y2 = [float(value) for value in record["bbox_xyxy"]]
    fallback_anchor = np.asarray([0.5 * (x1 + x2), y1], dtype=np.float64)
    if raw is None or not np.any(raw):
        return {
            "anchor_xy": fallback_anchor,
            "depth_mask": None,
            "torso_radius_pixels": 0.0,
            "component_bbox_xyxy": [x1, y1, x2, y2],
            "method": "bbox_fallback",
        }
    if raw.shape != (image_height, image_width):
        raw = cv2.resize(raw, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    binary = (raw > (0 if raw.max() <= 1 else 127)).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count > 1:
        # SAM masks can contain detached carried objects. The largest connected
        # component is the most reliable person support.
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        body = (labels == label).astype(np.uint8)
    else:
        body = binary
    ys, xs = np.where(body > 0)
    if len(xs) < 32:
        return {
            "anchor_xy": fallback_anchor,
            "depth_mask": body,
            "torso_radius_pixels": 0.0,
            "component_bbox_xyxy": [x1, y1, x2, y2],
            "method": "bbox_fallback",
        }

    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    body_height = max(ymax - ymin + 1, 1)
    distance = cv2.distanceTransform(body, cv2.DIST_L2, 5)
    maximum_radius = float(distance.max())

    # The widest interior point on each upper-body scanline follows the torso
    # medial axis. Thin horizontal arms and carried objects contribute little
    # distance-to-boundary and therefore cannot pull this axis like a bbox.
    row_start = int(round(ymin + 0.16 * (body_height - 1)))
    row_end = int(round(ymin + 0.66 * (body_height - 1)))
    row_centers = []
    row_radii = []
    minimum_radius = max(1.5, 0.10 * maximum_radius)
    for row_index in range(row_start, row_end + 1):
        row = distance[row_index]
        radius = float(row.max())
        if radius < minimum_radius:
            continue
        near_peak = np.flatnonzero(row >= 0.92 * radius)
        if len(near_peak):
            row_centers.append(float(np.median(near_peak)))
            row_radii.append(radius)
    if row_centers:
        row_centers_array = np.asarray(row_centers, dtype=np.float64)
        row_radii_array = np.asarray(row_radii, dtype=np.float64)
        axis_x = weighted_median(row_centers_array, np.square(row_radii_array))
        torso_radius = weighted_median(row_radii_array, np.square(row_radii_array))
        method = "upper_body_distance_transform_axis"
    else:
        axis_x = float(np.median(xs))
        torso_radius = max(maximum_radius, 1.0)
        method = "component_median_fallback"

    # Estimate the head/top only near the body axis. A raised hand elsewhere in
    # the mask is ignored, while a slightly offset/profile head remains inside
    # this strip. A low percentile is more stable than the single topmost pixel.
    central_half_width = max(3.0, 0.9 * torso_radius)
    central = (
        (body > 0)
        & (np.arange(image_width)[None, :] >= axis_x - central_half_width)
        & (np.arange(image_width)[None, :] <= axis_x + central_half_width)
    )
    # Quantiles over all mask pixels bias the top down as the visible body
    # gets longer. Use the top envelope of central columns instead.
    top_columns = np.flatnonzero(np.any(central, axis=0))
    tops = [np.flatnonzero(central[:, column])[0] for column in top_columns]
    head_y = float(np.percentile(tops, 20.0)) if tops else float(ymin)

    # DA3 depth comes from the thick central upper-body support. This rejects
    # most hands, arms, and objects extending away from the torso.
    yy, xx = np.indices(body.shape)
    torso_band = (
        (yy >= ymin + 0.18 * body_height)
        & (yy <= ymin + 0.68 * body_height)
        & (np.abs(xx - axis_x) <= max(6.0, 1.7 * torso_radius))
    )
    depth_core = (
        (body > 0)
        & torso_band
        & (distance >= max(1.0, 0.14 * torso_radius))
    )
    if int(depth_core.sum()) < 128:
        depth_core = (body > 0) & torso_band

    return {
        "anchor_xy": np.asarray([axis_x, head_y], dtype=np.float64),
        "depth_mask": depth_core.astype(np.uint8) * 255,
        "torso_radius_pixels": float(torso_radius),
        "component_bbox_xyxy": [xmin, ymin, xmax, ymax],
        "method": method,
    }


def smooth_localization_bboxes(
    observations: list[dict], window: int, image_size: tuple[int, int]
) -> np.ndarray:
    """Filter center and log-size while retaining raw SAM3 boxes as metadata."""
    bboxes = np.asarray(
        [item.get("mask_component_bbox_xyxy", item["bbox_xyxy"]) for item in observations],
        dtype=np.float64,
    )
    centers = np.column_stack(
        (0.5 * (bboxes[:, 0] + bboxes[:, 2]), 0.5 * (bboxes[:, 1] + bboxes[:, 3]))
    )
    sizes = np.column_stack(
        (np.maximum(bboxes[:, 2] - bboxes[:, 0], 1.0),
         np.maximum(bboxes[:, 3] - bboxes[:, 1], 1.0))
    )
    centers = moving_median(centers, window)
    sizes = np.exp(moving_median(np.log(sizes), window))
    filtered = np.column_stack(
        (centers[:, 0] - 0.5 * sizes[:, 0], centers[:, 1] - 0.5 * sizes[:, 1],
         centers[:, 0] + 0.5 * sizes[:, 0], centers[:, 1] + 0.5 * sizes[:, 1])
    )
    image_width, image_height = image_size
    filtered[:, [0, 2]] = np.clip(filtered[:, [0, 2]], 0.0, image_width - 1.0)
    filtered[:, [1, 3]] = np.clip(filtered[:, [1, 3]], 0.0, image_height - 1.0)
    return filtered


def robust_mask_depth(
    record: dict,
    depth: np.ndarray,
    confidence: np.ndarray | None,
    image_size: tuple[int, int],
    sampling_mask: np.ndarray | None = None,
) -> dict | None:
    """Return robust DA3 z-depth statistics from the interior of one mask."""
    image_width, image_height = image_size
    depth_height, depth_width = depth.shape
    mask = sampling_mask
    if mask is None:
        mask = cv2.imread(str(record["mask"]), cv2.IMREAD_GRAYSCALE)
    if mask is None or not np.any(mask):
        return None
    if mask.shape != (image_height, image_width):
        mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    mask = (mask > (0 if mask.max() <= 1 else 127)).astype(np.uint8) * 255
    small = cv2.resize(mask, (depth_width, depth_height), interpolation=cv2.INTER_AREA) >= 128
    # Removing one low-resolution boundary pixel suppresses depth bleeding from
    # the background while retaining small and partially visible people.
    interior = cv2.erode(small.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    if int(interior.sum()) < 32:
        interior = small
    valid = interior & np.isfinite(depth) & (depth > 1e-5)
    confidence_threshold = None
    if confidence is not None and np.any(valid):
        confidence_threshold = float(np.nanpercentile(confidence[valid], 30.0))
        valid &= np.isfinite(confidence) & (confidence >= confidence_threshold)
    if int(valid.sum()) < 24:
        return None

    values = depth[valid].astype(np.float64)
    log_values = np.log(values)
    median_log = float(np.median(log_values))
    mad = float(np.median(np.abs(log_values - median_log)))
    # Reject secondary surfaces introduced at occlusion boundaries. Log depth
    # makes the threshold behave consistently for near and far people.
    radius = float(np.clip(3.0 * 1.4826 * mad, 0.04, 0.18))
    foreground = np.abs(log_values - median_log) <= radius
    values = values[foreground]
    if len(values) < 24:
        return None
    p10, median, p90 = np.percentile(values, [10.0, 50.0, 90.0])
    return {
        "median": float(median),
        "p10": float(p10),
        "p90": float(p90),
        "sample_count": int(len(values)),
        "mask_depth_pixel_count": int(small.sum()),
        "confidence_threshold": confidence_threshold,
        "log_depth_mad": mad,
    }


def camera_box_from_observation(
    record: dict,
    camera: dict,
    depth_stats: dict,
    depth_scale: float,
    surface_to_center_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a camera-axis-aligned 3D box from a 2D bbox and z-depth."""
    intrinsic = np.asarray(camera["intrinsics"], dtype=np.float64)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    x1, y1, x2, y2 = [float(value) for value in record["bbox_xyxy"]]
    u = 0.5 * (x1 + x2)
    v = 0.5 * (y1 + y2)
    surface_z = float(depth_stats["median"] * depth_scale)
    # DA3 measures the first visible body surface. The proxy and the coarse
    # box represent the person's centre, so move them farther on the same ray.
    z = surface_z + max(float(surface_to_center_offset), 0.0)
    center_camera = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])

    width = max((x2 - x1 + 1.0) * z / fx, 1e-4)
    height = max((y2 - y1 + 1.0) * z / fy, 1e-4)
    measured_depth = max((depth_stats["p90"] - depth_stats["p10"]) * depth_scale, 0.0)
    # A monocular map observes a body's front surface rather than a full solid.
    # Add a conservative thickness tied to visible width and cap depth leaks.
    thickness = float(np.clip(measured_depth, 0.28 * width, 0.80 * width))
    return center_camera, np.array([width, height, thickness], dtype=np.float64)


def camera_box_to_world(
    center_camera: np.ndarray,
    camera: dict,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(camera["world_to_camera_rotation"], dtype=np.float64)
    center_world = np.asarray(camera["camera_center"], dtype=np.float64) + rotation.T @ center_camera
    # Columns map camera x/right, y/down, z/forward into the world frame.
    return center_world, rotation.T


def proxy_center_from_body_anchor(
    body_anchor_xy: np.ndarray,
    center_camera: np.ndarray,
    camera: dict,
    proxy_height: float,
    world_up: np.ndarray | None = None,
) -> np.ndarray:
    """Place a fixed proxy from the torso-led head anchor without a foot ray."""
    intrinsic = np.asarray(camera["intrinsics"], dtype=np.float64)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    u, v = [float(value) for value in body_anchor_xy]
    z = float(center_camera[2])
    head_camera = np.asarray([(u - cx) * z / fx, (v - cy) * z / fy, z])
    rotation = np.asarray(camera["world_to_camera_rotation"], dtype=np.float64)
    camera_center = np.asarray(camera["camera_center"], dtype=np.float64)
    head_world = camera_center + rotation.T @ head_camera
    camera_up_world = -rotation.T[:, 1].copy() if world_up is None else world_up.copy()
    camera_up_world /= max(float(np.linalg.norm(camera_up_world)), 1e-8)
    return head_world - 0.5 * proxy_height * camera_up_world


def interpolate_vectors(
    values: list[np.ndarray | None], frame_indices: np.ndarray
) -> np.ndarray | None:
    valid = np.asarray([value is not None for value in values], dtype=bool)
    if not np.any(valid):
        return None
    known = np.stack([value for value in values if value is not None])
    known_frames = frame_indices[valid]
    return np.stack(
        [np.interp(frame_indices, known_frames, known[:, axis]) for axis in range(known.shape[1])],
        axis=1,
    )


def world_aabb(center: np.ndarray, axes: np.ndarray, size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    extent = np.abs(axes) @ (0.5 * size)
    return center - extent, center + extent


def fixed_proxy_box(center, facing, up, height, radius):
    """Export the same fixed-size proxy box used by Blender, in raw world units."""
    up = np.asarray(up, dtype=np.float64).copy()
    up /= np.linalg.norm(up)
    forward = np.asarray(facing, dtype=np.float64).copy()
    forward -= up * np.dot(forward, up)
    if np.linalg.norm(forward) < 1e-8:
        forward = np.array([1., 0., 0.]) if abs(up[0]) < .9 else np.array([0., 1., 0.])
        forward -= up * np.dot(forward, up)
    forward /= np.linalg.norm(forward)
    side = np.cross(up, forward)
    axes = np.column_stack([forward, side, up])
    size = np.array([2 * radius, 2 * radius, height], dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    minimum, maximum = world_aabb(center, axes, size)
    return {"type": "fixed_actor_proxy_box", "center_world": center.tolist(),
            "size_world": size.tolist(), "axes_world": axes.tolist(),
            "aabb_min_world": minimum.tolist(), "aabb_max_world": maximum.tolist()}


def proxy_support_quality(output_frames, proxy_height, image_size):
    """Flag tiny interior detections for review without deleting their tracks.

    A scene-photo detection should not become a full-height actor. Requiring
    both tiny image support and implausible physical extent protects large
    close-up heads and small-but-distant full bodies. The decision is fixed
    for the complete track, never animated frame by frame.
    """
    width, height = image_size
    heights = [f["bbox_3d"]["size_world"][1] for f in output_frames]
    areas = [(f["bbox_xyxy"][2]-f["bbox_xyxy"][0]+1)*(f["bbox_xyxy"][3]-f["bbox_xyxy"][1]+1)/(width*height)
             for f in output_frames]
    height_ratio = float(np.median(heights) / proxy_height)
    area_fraction = float(np.median(areas))
    boundary_fraction = float(np.mean([f.get("touches_image_boundary", False) for f in output_frames]))
    insufficient = height_ratio < .15 and area_fraction < .003 and boundary_fraction < .25
    return {"eligible": not insufficient, "review_required": insufficient,
            "reason": "tiny_interior_mask_with_insufficient_physical_extent" if insufficient else None,
            "median_visible_height_ratio": height_ratio, "median_bbox_area_fraction": area_fraction,
            "boundary_observation_fraction": boundary_fraction, "decision_scope": "entire_track"}


def main() -> None:
    args = parse_args()
    people = json.loads(args.person_tracks.resolve().read_text())
    camera_data = json.loads(args.camera_track.resolve().read_text())
    if args.proxy_height <= 0 or args.proxy_radius <= 0:
        raise ValueError("Proxy height and radius must be positive")
    camera_frames = {int(item["frame_index"]): item for item in camera_data["frames"]}
    depth_indices = {int(item["frame_index"]): i for i, item in enumerate(camera_data["frames"])}
    if len(camera_frames) != len(camera_data["frames"]):
        raise ValueError("Camera frame indices must be unique")
    up_prior = (
        np.asarray(args.world_up, dtype=np.float64) if args.world_up is not None else
        np.median([-np.asarray(c["world_to_camera_rotation"])[1] for c in camera_frames.values()], axis=0)
    )
    if not np.all(np.isfinite(up_prior)) or np.linalg.norm(up_prior) < 1e-8:
        raise ValueError("World up must be a finite nonzero vector")
    up_prior /= np.linalg.norm(up_prior)
    image_width = int(camera_data["resolution"]["width"])
    image_height = int(camera_data["resolution"]["height"])

    depth_path = args.depth_prediction or Path(camera_data["da3"]["raw_output"])
    prediction = np.load(depth_path.resolve())
    depths = np.asarray(prediction["depth"], dtype=np.float32)
    confidence = (
        None
        if "conf" not in prediction or prediction["conf"].size == 0
        else np.asarray(prediction["conf"], dtype=np.float32)
    )
    if len(depths) != len(camera_data["frames"]):
        raise ValueError(
            f"Depth frame count {len(depths)} does not match camera frame count "
            f"{len(camera_data['frames'])}"
        )
    depth_scale = float(camera_data["da3"].get("actor_depth_alignment", {}).get("scale", camera_data["da3"]["translation_scale"]))
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("Depth-to-world scale must be positive and finite")

    actor_tracks = []
    for track in people.get("tracks", []):
        observations = []
        for record in sorted(track["frames"], key=lambda item: item["frame_index"]):
            frame_index = int(record["frame_index"])
            camera = camera_frames.get(frame_index)
            if camera is None:
                continue
            record = dict(record)
            mask_path = Path(record["mask"])
            if not mask_path.is_absolute():
                mask_path = args.person_tracks.resolve().parent / mask_path
            record["mask"] = str(mask_path)
            depth_index = depth_indices[frame_index]
            body_core = robust_body_core(record, (image_width, image_height))
            stats = robust_mask_depth(
                record,
                depths[depth_index],
                None if confidence is None else confidence[depth_index],
                (image_width, image_height),
                sampling_mask=body_core["depth_mask"],
            )
            center_world = None
            size_camera = None
            center_camera = None
            axes_world = np.asarray(camera["world_to_camera_rotation"], dtype=np.float64).T
            if stats is not None:
                center_camera, size_camera = camera_box_from_observation(
                    record, camera, stats, camera.get("actor_depth_scale", depth_scale), args.surface_to_center_offset
                )
                center_world, axes_world = camera_box_to_world(center_camera, camera)
            x1, y1, x2, y2 = [float(value) for value in record["bbox_xyxy"]]
            observations.append(
                {
                    "frame_index": frame_index,
                    "frame_name": record["frame_name"],
                    "mask": record["mask"],
                    "bbox_xyxy": record["bbox_xyxy"],
                    "body_anchor_xy_raw": body_core["anchor_xy"],
                    "body_anchor_method": body_core["method"],
                    "torso_radius_pixels": body_core["torso_radius_pixels"],
                    "mask_component_bbox_xyxy": body_core["component_bbox_xyxy"],
                    "center_world": center_world,
                    "center_camera": center_camera,
                    "size_camera": size_camera,
                    "axes_world": axes_world,
                    "depth": stats,
                    "touches_image_boundary": bool(
                        x1 <= 1 or y1 <= 1 or x2 >= image_width - 2 or y2 >= image_height - 2
                    ),
                }
            )
        if not observations:
            continue

        frame_indices = np.asarray([item["frame_index"] for item in observations], dtype=np.float64)
        depth_values = interpolate_vectors(
            [
                None if item["depth"] is None
                else np.asarray([item["depth"]["median"] * camera_frames[item["frame_index"]].get("actor_depth_scale", depth_scale)])
                for item in observations
            ],
            frame_indices,
        )
        depth_spans = interpolate_vectors(
            [
                None if item["depth"] is None
                else np.asarray([(item["depth"]["p90"] - item["depth"]["p10"]) * camera_frames[item["frame_index"]].get("actor_depth_scale", depth_scale)])
                for item in observations
            ],
            frame_indices,
        )
        if depth_values is None or depth_spans is None:
            continue
        depth_values = np.exp(
            moving_average(np.log(np.maximum(depth_values, 1e-5)), args.depth_window)
        )[:, 0]
        depth_spans = np.exp(
            moving_average(np.log(np.maximum(depth_spans, 1e-5)), args.size_window)
        )[:, 0]

        localization_bboxes = smooth_localization_bboxes(
            observations,
            args.bbox_window,
            (image_width, image_height),
        )
        body_anchors = moving_median(
            np.stack([item["body_anchor_xy_raw"] for item in observations]),
            args.bbox_window,
        )
        # Bbox extent and torso/head anchor serve different purposes. Recentring
        # the box on the torso shifts it away from its observed silhouette,
        # especially at image boundaries. Keep the measured component extent.

        # Rebuild every box from robustly filtered 2D center/extent and depth.
        # Raw SAM3 boxes remain in the output for diagnosis.  This prevents a
        # transient arm, occluder, or mask merge from moving the actor proxy by
        # a large distance for one frame.
        centers_list = []
        proxy_centers_list = []
        sizes_list = []
        camera_centers_list = []
        for item, localization_bbox, body_anchor, depth_value, depth_span in zip(
            observations, localization_bboxes, body_anchors, depth_values, depth_spans
        ):
            synthetic_stats = {
                "median": float(depth_value / depth_scale),
                "p10": float((depth_value - 0.5 * depth_span) / depth_scale),
                "p90": float((depth_value + 0.5 * depth_span) / depth_scale),
            }
            camera = camera_frames[item["frame_index"]]
            localization_record = {**item, "bbox_xyxy": localization_bbox.tolist()}
            center_camera, size_camera = camera_box_from_observation(
                localization_record,
                camera,
                synthetic_stats,
                depth_scale,
                args.surface_to_center_offset,
            )
            center_world, axes_world = camera_box_to_world(center_camera, camera)
            proxy_center = proxy_center_from_body_anchor(
                body_anchor,
                center_camera,
                camera,
                args.proxy_height,
                up_prior,
            )
            item["axes_world"] = axes_world
            item["localization_bbox_xyxy"] = localization_bbox
            item["body_anchor_xy"] = body_anchor
            centers_list.append(center_world)
            proxy_centers_list.append(proxy_center)
            sizes_list.append(size_camera)
            camera_centers_list.append(center_camera)
        centers = np.stack(centers_list)
        proxy_centers = np.stack(proxy_centers_list)
        sizes = np.stack(sizes_list)
        camera_centers = np.stack(camera_centers_list)

        heights = sizes[:, 1]
        widths = sizes[:, 0]
        upright_reference = max(float(np.percentile(heights, 90.0)), 1e-6)
        relative_height = heights / upright_reference
        relative_sitting = np.clip((0.84 - relative_height) / 0.24, 0.0, 1.0)
        aspects = heights / np.maximum(widths, 1e-6)
        shape_sitting = np.clip((1.18 - aspects) / 0.35, 0.0, 1.0)
        raw_sitting = np.maximum(relative_sitting, 0.45 * shape_sitting)
        raw_sitting = np.where(aspects >= 1.50, np.minimum(raw_sitting, 0.25), raw_sitting)
        sitting_scores = moving_average(raw_sitting[:, None], args.posture_window)[:, 0]

        output_frames = []
        median_height = max(float(np.median(heights)), 1e-5)
        for index, (item, center, size, center_cam, sitting_score) in enumerate(
            zip(observations, centers, sizes, camera_centers, sitting_scores)
        ):
            axes = item["axes_world"]
            lo = max(0, index - args.motion_window)
            hi = min(len(centers) - 1, index + args.motion_window)
            velocity = proxy_centers[hi] - proxy_centers[lo]
            up = up_prior
            velocity -= up * float(np.dot(velocity, up))
            motion = float(np.linalg.norm(velocity))
            if motion > 0.03 * median_height:
                facing = velocity / motion
                facing_source = "track_motion"
                facing_confidence = float(np.clip(motion / (0.25 * median_height), 0.35, 0.85))
            else:
                camera_center = np.asarray(
                    camera_frames[item["frame_index"]]["camera_center"], dtype=np.float64
                )
                facing = camera_center - proxy_centers[index]
                facing -= up * float(np.dot(facing, up))
                norm = float(np.linalg.norm(facing))
                facing = -axes[:, 2] if norm < 1e-8 else facing / norm
                facing_source = "camera_facing_prior"
                facing_confidence = 0.2

            if sitting_score >= 0.65:
                posture = "sitting"
            elif sitting_score <= 0.35:
                posture = "standing"
            else:
                posture = "transition"
            minimum, maximum = world_aabb(center, axes, size)
            depth_stats = item["depth"]
            depth_confidence = 0.0
            if depth_stats is not None:
                coverage = min(
                    depth_stats["sample_count"] / max(depth_stats["mask_depth_pixel_count"], 1), 1.0
                )
                stability = float(np.exp(-8.0 * depth_stats["log_depth_mad"]))
                depth_confidence = float(np.clip(coverage * stability, 0.0, 1.0))
            output_frames.append(
                {
                    "frame_index": item["frame_index"],
                    "frame_name": item["frame_name"],
                    "mask": item["mask"],
                    "bbox_xyxy": item["bbox_xyxy"],
                    "localization_bbox_xyxy": item["localization_bbox_xyxy"].tolist(),
                    "body_anchor_xy_raw": item["body_anchor_xy_raw"].tolist(),
                    "body_anchor_xy": item["body_anchor_xy"].tolist(),
                    "body_anchor_method": item["body_anchor_method"],
                    "torso_radius_pixels": item["torso_radius_pixels"],
                    "mask_component_bbox_xyxy": item["mask_component_bbox_xyxy"],
                    "position_world": proxy_centers[index].tolist(),
                    "position_source": (
                        "smoothed_torso_axis_head_ray_fixed_height_center_"
                        "with_surface_depth_offset"
                    ),
                    "proxy_center_world": proxy_centers[index].tolist(),
                    "head_anchor_world": (proxy_centers[index] + 0.5 * args.proxy_height * up_prior).tolist(),
                    "proxy_up_world": up_prior.tolist(),
                    "proxy_bbox_3d": fixed_proxy_box(proxy_centers[index], facing, up_prior,
                                                     args.proxy_height, args.proxy_radius),
                    "bbox_3d": {
                        "type": "camera_oriented_box",
                        "center_world": center.tolist(),
                        "size_world": size.tolist(),
                        "axes_world": axes.tolist(),
                        "aabb_min_world": minimum.tolist(),
                        "aabb_max_world": maximum.tolist(),
                    },
                    "depth": {
                        "median_raw": None if depth_stats is None else depth_stats["median"],
                        "median_world": float(center_cam[2]),
                        "visible_surface_world": float(
                            center_cam[2] - args.surface_to_center_offset
                        ),
                        "p10_raw": None if depth_stats is None else depth_stats["p10"],
                        "p90_raw": None if depth_stats is None else depth_stats["p90"],
                        "sample_count": 0 if depth_stats is None else depth_stats["sample_count"],
                        "confidence": depth_confidence,
                    },
                    "facing_world": facing.tolist(),
                    "facing_source": facing_source,
                    "facing_confidence": facing_confidence,
                    "posture": posture,
                    "posture_confidence": float(abs(sitting_score - 0.5) * 2.0),
                    "sitting_score": float(sitting_score),
                    "touches_image_boundary": item["touches_image_boundary"],
                }
            )
        actor_tracks.append(
            {
                "track_id": int(track["track_id"]),
                "first_frame": output_frames[0]["frame_index"],
                "last_frame": output_frames[-1]["frame_index"],
                "frame_count": len(output_frames),
                "proxy_quality": proxy_support_quality(output_frames, args.proxy_height, (image_width, image_height)),
                "frames": output_frames,
            }
        )

    result = {
        "schema_version": 2,
        "coordinate_system": "omega_ba_world" if camera_data.get("coordinate_system") == "omega_ba_world" else "raw_colmap_world",
        "representation": "sam3_track_da3_depth_camera_oriented_3d_bbox",
        "camera_track": str(args.camera_track.resolve()),
        "person_tracks": str(args.person_tracks.resolve()),
        "depth_prediction": str(depth_path.resolve()),
        "depth_to_world_scale": depth_scale,
        "surface_to_center_offset_world": float(args.surface_to_center_offset),
        "box_convention": {
            "axes_world_columns": ["camera_right", "camera_down", "camera_forward"],
            "size_world_order": ["width", "height", "depth"],
            "extent": "visible connected-component bbox at robust DA3 foreground depth",
        },
        "proxy_box_convention": {
            "axes_world_columns": ["actor_forward", "actor_side", "scene_up"],
            "size_world_order": ["forward_extent", "side_extent", "height"],
            "size_policy": "fixed_for_entire_clip",
            "extent": "coarse head-and-cylinder proxy; not fitted anatomical extent",
        },
        "proxy_convention": {
            "height_world": args.proxy_height,
            "radius_world": args.proxy_radius,
            "up_world": up_prior.tolist(),
            "up_source": "explicit_world_up" if args.world_up is not None else "fixed_clip_camera_up_prior",
            "size_policy": "fixed_for_entire_clip",
            "surface_to_center_offset_world": args.surface_to_center_offset,
            "position": (
                "smoothed_torso_axis_head_ray_at_surface_plus_depth_offset_"
                "minus_half_height_along_fixed_clip_up"
            ),
            "uses_foot_pixel": False,
            "uses_floor_intersection": False,
        },
        "mask_localization": {
            "body_axis": "upper_body_scanline_distance_transform_weighted_median",
            "depth_support": "thick_central_upper_body_core",
            "rejects": ["detached_components", "thin_arm_extensions", "off_axis_carried_objects"],
        },
        "smoothing": {
            "log_depth_window_frames": args.depth_window,
            "log_thickness_window_frames": args.size_window,
            "bbox_center_and_log_size_median_window_frames": args.bbox_window,
        },
        "tracks": actor_tracks,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "representation": result["representation"],
                "track_count": len(actor_tracks),
                "tracks": [item["frame_count"] for item in actor_tracks],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
