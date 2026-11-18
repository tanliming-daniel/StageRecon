#!/usr/bin/env python3
"""Estimate one-shot camera motion with DA3 between endpoint PnP anchors.

DA3 sees the complete clip in one inference call and predicts a coherent set of
OpenCV world-to-camera poses, intrinsics and depths.  Its coordinate frame and
translation scale are arbitrary.  This script therefore:

1. expresses every DA3 camera relative to DA3 frame zero;
2. estimates one translation scale from frame-zero DA3 depth and the existing
   COLMAP point cloud projected by the already-computed frame-zero PnP pose;
3. composes the relative trajectory onto the frame-zero PnP pose;
4. distributes the terminal position, rotation, and focal drift continuously
   so the last pose is exactly the independently localized last-frame PnP pose.

Only the first and last frames are sent through PnP.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from depth_anything_3.api import DepthAnything3
from pose_quality import require_usable_anchor
from registered_anchors import intrinsics_between_anchors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-review-anchors", action="store_true")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--anchor-localization", type=Path, required=True)
    parser.add_argument("--tail-anchor-localization", type=Path, required=True)
    parser.add_argument("--environment-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument(
        "--raw-prediction",
        type=Path,
        help="Reuse a cached DA3 .npz and only rebuild the anchored trajectory.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--use-ray-pose", action="store_true")
    parser.add_argument(
        "--ref-view-strategy",
        choices=("first", "middle", "saddle_balanced", "saddle_sim_range"),
        default="first",
    )
    parser.add_argument(
        "--smoothing-mode",
        choices=("none", "gaussian", "savgol"),
        default="savgol",
        help="Temporal camera smoothing; savgol is the cinematic default.",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.35,
        help="Gaussian sigma in frames when --smoothing-mode=gaussian.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=21,
        help="Odd Savitzky-Golay window in frames.",
    )
    parser.add_argument(
        "--smooth-polyorder",
        type=int,
        default=3,
        help="Savitzky-Golay polynomial order.",
    )
    parser.add_argument(
        "--translation-scale",
        type=float,
        help="Override automatic COLMAP-depth scale fitting with this value.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def read_xyz_ply(path: Path) -> np.ndarray:
    """Read xyz from the binary/ascii PLY used by the Sitcoms3D scenes."""
    with path.open("rb") as stream:
        header_lines: list[str] = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"Incomplete PLY header: {path}")
            decoded = line.decode("ascii").rstrip("\r\n")
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        fmt = next(line.split()[1] for line in header_lines if line.startswith("format "))
        count = int(next(line.split()[2] for line in header_lines if line.startswith("element vertex ")))
        properties: list[tuple[str, str]] = []
        in_vertices = False
        scalar_types = {
            "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
            "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
            "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
            "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
        }
        for line in header_lines:
            if line.startswith("element "):
                in_vertices = line.startswith("element vertex ")
            elif in_vertices and line.startswith("property "):
                fields = line.split()
                if fields[1] == "list":
                    raise ValueError("List-valued vertex properties are not supported")
                properties.append((fields[2], scalar_types[fields[1]]))
        if fmt == "ascii":
            data = np.loadtxt(stream, max_rows=count)
            names = [name for name, _ in properties]
            return data[:, [names.index("x"), names.index("y"), names.index("z")]].astype(np.float64)
        if fmt not in ("binary_little_endian", "binary_big_endian"):
            raise ValueError(f"Unsupported PLY format: {fmt}")
        endian = "<" if fmt == "binary_little_endian" else ">"
        dtype = np.dtype([(name, endian + code) for name, code in properties])
        vertices = np.fromfile(stream, dtype=dtype, count=count)
        if len(vertices) != count:
            raise ValueError(f"Expected {count} PLY vertices, got {len(vertices)}")
        return np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float64)


def homogeneous(extrinsics: np.ndarray) -> np.ndarray:
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics.astype(np.float64, copy=True)
    if extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"Unsupported extrinsics shape: {extrinsics.shape}")
    result = np.broadcast_to(np.eye(4), extrinsics.shape[:-2] + (4, 4)).copy()
    result[..., :3, :] = extrinsics
    return result.astype(np.float64)


def effective_savgol_window(length: int, requested: int, polyorder: int) -> int:
    if requested < 3:
        raise ValueError("--smooth-window must be at least 3")
    window = requested if requested % 2 else requested + 1
    maximum = length if length % 2 else length - 1
    window = min(window, maximum)
    if window <= polyorder:
        raise ValueError(
            f"Smoothing window {window} must exceed polynomial order {polyorder}"
        )
    return window


def smooth_series(
    values: np.ndarray,
    mode: str,
    sigma: float,
    window: int,
    polyorder: int,
) -> np.ndarray:
    if mode == "none" or len(values) < 3:
        return values.copy()
    if mode == "gaussian":
        if sigma <= 0:
            return values.copy()
        return gaussian_filter1d(values, sigma=sigma, axis=0, mode="nearest")
    if mode == "savgol":
        actual_window = effective_savgol_window(len(values), window, polyorder)
        return savgol_filter(
            values,
            window_length=actual_window,
            polyorder=polyorder,
            axis=0,
            mode="interp",
        )
    raise ValueError(f"Unsupported smoothing mode: {mode}")


def smooth_relative_poses(
    relative_c2w: np.ndarray,
    mode: str,
    sigma: float,
    window: int,
    polyorder: int,
) -> np.ndarray:
    if mode == "none" or len(relative_c2w) < 3:
        return relative_c2w
    rotvec = Rotation.from_matrix(relative_c2w[:, :3, :3]).as_rotvec()
    centers = relative_c2w[:, :3, 3]
    smooth_rotvec = smooth_series(rotvec, mode, sigma, window, polyorder)
    smooth_centers = smooth_series(centers, mode, sigma, window, polyorder)
    smoothed = np.broadcast_to(np.eye(4), relative_c2w.shape).copy()
    smoothed[:, :3, :3] = Rotation.from_rotvec(smooth_rotvec).as_matrix()
    smoothed[:, :3, 3] = smooth_centers
    # Filtering moves frame zero slightly. Re-anchor it exactly to identity.
    return np.linalg.inv(smoothed[0])[None] @ smoothed


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def estimate_scene_scale(
    xyz_world: np.ndarray,
    anchor_rotation: np.ndarray,
    anchor_translation: np.ndarray,
    anchor_intrinsics: np.ndarray,
    da3_depth: np.ndarray,
    da3_conf: np.ndarray | None,
    original_size: tuple[int, int],
    dynamic_mask: np.ndarray | None,
) -> tuple[float, dict]:
    """Fit scene-depth / DA3-depth using a sparse z-buffer at frame zero."""
    height, width = da3_depth.shape
    original_width, original_height = original_size
    points_camera = xyz_world @ anchor_rotation.T + anchor_translation
    z = points_camera[:, 2]
    valid = np.isfinite(points_camera).all(axis=1) & (z > 1e-4)
    points_camera = points_camera[valid]
    z = z[valid]

    u_original = anchor_intrinsics[0, 0] * points_camera[:, 0] / z + anchor_intrinsics[0, 2]
    v_original = anchor_intrinsics[1, 1] * points_camera[:, 1] / z + anchor_intrinsics[1, 2]
    visible = (
        (u_original >= 0) & (u_original < original_width)
        & (v_original >= 0) & (v_original < original_height)
    )
    u_original = u_original[visible]
    v_original = v_original[visible]
    z = z[visible]
    if dynamic_mask is not None:
        mx = np.clip(np.rint(u_original).astype(np.int64), 0, original_width - 1)
        my = np.clip(np.rint(v_original).astype(np.int64), 0, original_height - 1)
        static = dynamic_mask[my, mx] == 0
        u_original, v_original, z = u_original[static], v_original[static], z[static]

    x = np.clip(np.rint(u_original * width / original_width).astype(np.int64), 0, width - 1)
    y = np.clip(np.rint(v_original * height / original_height).astype(np.int64), 0, height - 1)
    flat = y * width + x
    scene_zbuffer = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(scene_zbuffer, flat, z)
    populated = np.isfinite(scene_zbuffer)
    predicted = da3_depth.reshape(-1).astype(np.float64)
    valid = populated & np.isfinite(predicted) & (predicted > 1e-5)
    confidence_threshold = None
    if da3_conf is not None:
        confidence = da3_conf.reshape(-1).astype(np.float64)
        confidence_threshold = float(np.nanpercentile(confidence[valid], 35.0))
        valid &= np.isfinite(confidence) & (confidence >= confidence_threshold)
    ratios = scene_zbuffer[valid] / predicted[valid]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if len(ratios) < 100:
        raise RuntimeError(f"Only {len(ratios)} depth samples remain for scene-scale fitting")

    log_ratios = np.log(ratios)
    median_log = float(np.median(log_ratios))
    mad = float(np.median(np.abs(log_ratios - median_log)))
    robust_sigma = max(1.4826 * mad, 0.08)
    keep = np.abs(log_ratios - median_log) <= 2.5 * robust_sigma
    filtered = ratios[keep]
    if len(filtered) < 100:
        raise RuntimeError(f"Only {len(filtered)} robust depth samples remain for scene-scale fitting")
    scale = float(np.exp(np.median(np.log(filtered))))
    stats = {
        "method": "frame0_colmap_sparse_zbuffer_to_da3_depth_log_median",
        "projected_static_point_count": int(len(z)),
        "zbuffer_sample_count": int(populated.sum()),
        "confidence_threshold": confidence_threshold,
        "candidate_ratio_count": int(len(ratios)),
        "robust_ratio_count": int(len(filtered)),
        "scale": scale,
        "ratio_p10_p50_p90": np.percentile(filtered, [10, 50, 90]).astype(float).tolist(),
        "log_ratio_mad": mad,
    }
    return scale, stats


def smooth_focal_ratios(
    intrinsics: np.ndarray,
    processed_size: tuple[int, int],
    mode: str,
    sigma: float,
    window: int,
    polyorder: int,
) -> np.ndarray:
    height, width = processed_size
    focal = 0.5 * (intrinsics[:, 0, 0] / width + intrinsics[:, 1, 1] / height)
    focal = np.maximum(focal.astype(np.float64), 1e-6)
    log_focal = np.log(focal)
    log_focal = smooth_series(log_focal, mode, sigma, window, polyorder)
    ratios = np.exp(log_focal - log_focal[0])
    return np.clip(ratios, 0.5, 2.0)


def localization_c2w(localization: dict) -> tuple[np.ndarray, float]:
    """Convert one existing PnP localization to C2W and return its focal."""
    rotation = np.asarray(localization["world_to_camera_rotation"], dtype=np.float64)
    translation = np.asarray(localization["world_to_camera_translation"], dtype=np.float64)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation.T
    c2w[:3, 3] = -rotation.T @ translation
    return c2w, float(localization["pnp"]["focal"])


def constrain_trajectory_to_tail_anchor(
    predicted_c2w: np.ndarray,
    predicted_focals: np.ndarray,
    tail_c2w: np.ndarray,
    tail_focal: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Remove endpoint drift while retaining DA3's local frame-to-frame path."""
    count = len(predicted_c2w)
    if count < 2:
        raise ValueError("Dual-anchor camera tracking requires at least two frames")
    alpha = np.linspace(0.0, 1.0, count, dtype=np.float64)

    predicted_tail_center = predicted_c2w[-1, :3, 3].copy()
    target_tail_center = tail_c2w[:3, 3]
    translation_residual = target_tail_center - predicted_tail_center
    corrected = predicted_c2w.copy()
    corrected[:, :3, 3] += alpha[:, None] * translation_residual[None]

    predicted_tail_rotation = predicted_c2w[-1, :3, :3]
    rotation_residual = Rotation.from_matrix(
        tail_c2w[:3, :3] @ predicted_tail_rotation.T
    )
    residual_rotvec = rotation_residual.as_rotvec()
    corrections = Rotation.from_rotvec(alpha[:, None] * residual_rotvec[None])
    corrected[:, :3, :3] = (
        corrections.as_matrix() @ predicted_c2w[:, :3, :3]
    )

    if tail_focal <= 0 or predicted_focals[-1] <= 0:
        raise ValueError("PnP and predicted focal lengths must be positive")
    log_focal_residual = float(np.log(tail_focal / predicted_focals[-1]))
    corrected_focals = predicted_focals * np.exp(alpha * log_focal_residual)

    # Assign exact anchors after floating-point interpolation.
    corrected[0] = predicted_c2w[0]
    corrected[-1] = tail_c2w
    corrected_focals[0] = predicted_focals[0]
    corrected_focals[-1] = tail_focal
    stats = {
        "method": "linear_endpoint_drift_distribution",
        "translation_residual_world": translation_residual.tolist(),
        "translation_residual_norm_world": float(np.linalg.norm(translation_residual)),
        "rotation_residual_degrees": float(np.degrees(rotation_residual.magnitude())),
        "predicted_tail_focal_pixels": float(predicted_focals[-1]),
        "target_tail_focal_pixels": float(tail_focal),
        "focal_correction_ratio": float(tail_focal / predicted_focals[-1]),
    }
    return corrected, corrected_focals, stats


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    frames_dir = args.frames_dir.resolve()
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No JPG frames found in {frames_dir}")
    anchor_path = require_file(args.anchor_localization, "Anchor localization")
    tail_anchor_path = require_file(
        args.tail_anchor_localization, "Tail anchor localization"
    )
    anchor = json.loads(anchor_path.read_text())
    tail_anchor = json.loads(tail_anchor_path.read_text())
    require_usable_anchor(anchor, allow_review=args.allow_review_anchors)
    require_usable_anchor(tail_anchor, allow_review=args.allow_review_anchors)
    model_path = args.model.resolve()
    if not (model_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"DA3 model.safetensors not found in {model_path}")
    first_image = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first_image is None:
        raise FileNotFoundError(frame_paths[0])
    original_height, original_width = first_image.shape[:2]

    raw_output = (args.raw_output or args.output.with_name("da3_prediction.npz")).resolve()
    if args.raw_prediction is not None:
        raw_output = args.raw_prediction.resolve()
        print(f"[DA3] Reusing cached prediction {raw_output}", flush=True)
        cached = np.load(raw_output)
        depth = np.asarray(cached["depth"], dtype=np.float32)
        confidence = (
            None
            if "conf" not in cached or cached["conf"].size == 0
            else np.asarray(cached["conf"], dtype=np.float32)
        )
        extrinsics = homogeneous(np.asarray(cached["extrinsics"]))
        intrinsics = np.asarray(cached["intrinsics"], dtype=np.float64)
    else:
        device = torch.device(args.device)
        print(f"[DA3] Loading {model_path} on {device}", flush=True)
        model = DepthAnything3.from_pretrained(str(model_path)).to(device).eval()
        print(f"[DA3] Inferring {len(frame_paths)} frames in one call", flush=True)
        with torch.inference_mode():
            prediction = model.inference(
                image=[str(path) for path in frame_paths],
                use_ray_pose=args.use_ray_pose,
                ref_view_strategy=args.ref_view_strategy,
                process_res=args.process_res,
                process_res_method="upper_bound_resize",
            )
        if prediction.extrinsics is None or prediction.intrinsics is None:
            raise RuntimeError("DA3 prediction does not contain camera poses and intrinsics")
        depth = np.asarray(prediction.depth, dtype=np.float32)
        confidence = None if prediction.conf is None else np.asarray(prediction.conf, dtype=np.float32)
        extrinsics = homogeneous(np.asarray(prediction.extrinsics))
        intrinsics = np.asarray(prediction.intrinsics, dtype=np.float64)
    if len(extrinsics) != len(frame_paths):
        raise RuntimeError(f"DA3 returned {len(extrinsics)} poses for {len(frame_paths)} frames")
    processed_height, processed_width = depth.shape[1:]

    raw_output.parent.mkdir(parents=True, exist_ok=True)
    if args.raw_prediction is None:
        np.savez_compressed(
            raw_output,
            depth=depth.astype(np.float16),
            conf=(confidence.astype(np.float16) if confidence is not None else np.empty(0, np.float16)),
            extrinsics=extrinsics.astype(np.float32),
            intrinsics=intrinsics.astype(np.float32),
        )

    da3_c2w = np.linalg.inv(extrinsics)
    relative_c2w = extrinsics[0][None] @ da3_c2w
    relative_c2w = smooth_relative_poses(
        relative_c2w,
        args.smoothing_mode,
        args.smooth_sigma,
        args.smooth_window,
        args.smooth_polyorder,
    )

    anchor_rotation = np.asarray(anchor["world_to_camera_rotation"], dtype=np.float64)
    anchor_translation = np.asarray(anchor["world_to_camera_translation"], dtype=np.float64)
    anchor_center = -anchor_rotation.T @ anchor_translation
    anchor_intrinsics = np.asarray(anchor["pnp"]["intrinsics"], dtype=np.float64)

    mask = None
    if args.mask_dir is not None:
        mask_path = args.mask_dir.resolve() / "000000.png"
        if mask_path.is_file():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.shape != (original_height, original_width):
                mask = resize_mask(mask, original_width, original_height)

    if args.translation_scale is not None:
        if args.translation_scale <= 0:
            raise ValueError("--translation-scale must be positive")
        translation_scale = float(args.translation_scale)
        scale_stats = {"method": "explicit_override", "scale": translation_scale}
    else:
        pointcloud = require_file(args.environment_dir.resolve() / "colmap/pc.ply", "COLMAP PLY")
        xyz_world = read_xyz_ply(pointcloud)
        translation_scale, scale_stats = estimate_scene_scale(
            xyz_world,
            anchor_rotation,
            anchor_translation,
            anchor_intrinsics,
            depth[0],
            None if confidence is None else confidence[0],
            (original_width, original_height),
            mask,
        )

    anchor_c2w, anchor_focal = localization_c2w(anchor)
    tail_anchor_c2w, tail_anchor_focal = localization_c2w(tail_anchor)
    focal_ratios = smooth_focal_ratios(
        intrinsics,
        (processed_height, processed_width),
        args.smoothing_mode,
        args.smooth_sigma,
        args.smooth_window,
        args.smooth_polyorder,
    )
    predicted_c2w = []
    for relative_pose in relative_c2w:
        relative = relative_pose.copy()
        relative[:3, 3] *= translation_scale
        predicted_c2w.append(anchor_c2w @ relative)
    predicted_c2w = np.stack(predicted_c2w)
    predicted_focals = anchor_focal * focal_ratios
    absolute_c2w, absolute_focals, endpoint_correction = (
        constrain_trajectory_to_tail_anchor(
            predicted_c2w,
            predicted_focals,
            tail_anchor_c2w,
            tail_anchor_focal,
        )
    )

    frames = []
    raw_centers = relative_c2w[:, :3, 3]
    for index, frame_path in enumerate(frame_paths):
        frame_c2w = absolute_c2w[index]
        absolute_w2c = np.linalg.inv(frame_c2w)
        rotation = absolute_w2c[:3, :3]
        translation = absolute_w2c[:3, 3]
        center = frame_c2w[:3, 3]
        focal = float(absolute_focals[index])
        intrinsic = intrinsics_between_anchors(
            anchor["pnp"]["intrinsics"], tail_anchor["pnp"]["intrinsics"],
            focal, index / max(len(frame_paths) - 1, 1),
        )
        if index == 0:
            quality = {
                "status": "anchor",
                "pose_source": anchor.get("pose_source", "estimated_disk_pnp"),
                "pnp_inlier_count": int(anchor["pnp"]["inlier_count"]),
                "pnp_rmse_pixels": float(anchor["pnp"]["inlier_rmse_pixels"]),
            }
        elif index == len(frame_paths) - 1:
            quality = {
                "status": "tail_anchor",
                "pose_source": tail_anchor.get("pose_source", "estimated_disk_pnp"),
                "pnp_inlier_count": int(tail_anchor["pnp"]["inlier_count"]),
                "pnp_rmse_pixels": float(tail_anchor["pnp"]["inlier_rmse_pixels"]),
            }
        else:
            quality = {
                "status": "tracked_da3_dual_anchor",
                "da3_depth_confidence_median": (
                    float(np.nanmedian(confidence[index])) if confidence is not None else None
                ),
                "raw_relative_translation_norm": float(np.linalg.norm(raw_centers[index])),
            }
        frames.append(
            {
                "frame_index": index,
                "frame_name": frame_path.name,
                "intrinsics": intrinsic.tolist(),
                "world_to_camera_rotation": rotation.tolist(),
                "world_to_camera_translation": translation.tolist(),
                "camera_center": center.tolist(),
                "focal_pixels": focal,
                "quality": quality,
            }
        )

    absolute_centers = np.asarray([frame["camera_center"] for frame in frames])
    step_distances = np.linalg.norm(np.diff(absolute_centers, axis=0), axis=1)
    corrected_relative_rotations = (
        absolute_c2w[0, :3, :3].T[None] @ absolute_c2w[:, :3, :3]
    )
    relative_rotations = Rotation.from_matrix(corrected_relative_rotations)
    rotation_from_first = np.degrees(relative_rotations.magnitude())
    result = {
        "schema_version": 2,
        "model": "depth_anything_3_large_full_clip_dual_endpoint_pnp",
        "coordinate_system": "raw_colmap_world (same as environment_dir/colmap/pc.ply)",
        "anchor_frame_index": 0,
        "anchor_frame_name": frame_paths[0].name,
        "anchor_localization": str(anchor_path),
        "tail_anchor_frame_index": len(frame_paths) - 1,
        "tail_anchor_frame_name": frame_paths[-1].name,
        "tail_anchor_localization": str(tail_anchor_path),
        "absolute_localization_frames": [0, len(frame_paths) - 1],
        "absolute_anchor_policy": anchor.get("requested_anchor_source", "estimated"),
        "absolute_anchor_provenance": [
            {"frame_index": i, "pose_source": p.get("pose_source", "estimated_disk_pnp"),
             "registration": p.get("registration"), "quality": p.get("quality")}
            for i, p in ((0, anchor), (len(frame_paths) - 1, tail_anchor))
        ],
        "resolution": {"width": original_width, "height": original_height},
        "camera_center": frames[0]["camera_center"],
        "da3": {
            "model": str(model_path),
            "raw_output": str(raw_output),
            "process_resolution": args.process_res,
            "processed_resolution": {"width": processed_width, "height": processed_height},
            "reference_view_strategy": args.ref_view_strategy,
            "use_ray_pose": args.use_ray_pose,
            "pose_smoothing": {
                "mode": args.smoothing_mode,
                "gaussian_sigma_frames": args.smooth_sigma,
                "savgol_window_frames": args.smooth_window,
                "savgol_polyorder": args.smooth_polyorder,
            },
            "translation_scale": translation_scale,
            "translation_scale_fit": scale_stats,
            "endpoint_correction": endpoint_correction,
        },
        "summary": {
            "frame_count": len(frames),
            "tracked_transition_count": len(frames) - 1,
            "held_transition_count": 0,
            "translation_scale_to_colmap": translation_scale,
            "camera_path_length_world": float(step_distances.sum()),
            "camera_displacement_from_first_world": float(np.linalg.norm(absolute_centers[-1] - absolute_centers[0])),
            "camera_max_displacement_from_first_world": float(
                np.max(np.linalg.norm(absolute_centers - absolute_centers[0], axis=1))
            ),
            "max_rotation_from_first_degrees": float(np.max(rotation_from_first)),
            "focal_min_pixels": float(absolute_focals.min()),
            "focal_max_pixels": float(absolute_focals.max()),
            "focal_ratio": float(absolute_focals.max() / absolute_focals.min()),
            "tail_endpoint_translation_error_world": float(
                np.linalg.norm(absolute_c2w[-1, :3, 3] - tail_anchor_c2w[:3, 3])
            ),
            "tail_endpoint_rotation_error_degrees": float(
                np.degrees(
                    Rotation.from_matrix(
                        tail_anchor_c2w[:3, :3] @ absolute_c2w[-1, :3, :3].T
                    ).magnitude()
                )
            ),
            "tail_endpoint_focal_error_pixels": float(
                abs(absolute_focals[-1] - tail_anchor_focal)
            ),
            "inference_seconds": round(time.perf_counter() - started, 3),
        },
        "frames": frames,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
