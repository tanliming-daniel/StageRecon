#!/usr/bin/env python3
"""Long-lived SAM3 + DISK/PnP worker for clip render-data export."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
import traceback
from batch_protocol import stage_result
from registered_anchors import registered_anchor
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
SITCOMS3D_ROOT = REPO_ROOT / "sitcoms3D"
DISK_ROOT = SITCOMS3D_ROOT / "external/disk"
for path in (TOOLS_ROOT, SITCOMS3D_ROOT, DISK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from disk import DISK  # noqa: E402
from sitcoms3D.utils.colmap_utils import read_model  # noqa: E402

import extract_person_tracks_sam3 as sam3_tracker  # noqa: E402
import localize_query_from_disk as localizer  # noqa: E402
from run_clip_to_blender import extract_frames  # noqa: E402
from run_new_image_to_blender import assess_pose_quality, compact_visual_features  # noqa: E402


def emit(payload: dict) -> None:
    sys.__stdout__.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()


class DiskPnPLocalizer:
    def __init__(
        self,
        *,
        environment: Path,
        checkpoint: Path,
        global_index: Path,
        max_features: int = 2048,
        retrieval_pool: int = 200,
        top_k: int = 20,
        ratio: float = 0.90,
        global_top_k: int = 200,
        global_ratio: float = 0.95,
        reprojection_error: float = 8.0,
    ) -> None:
        self.environment = environment.resolve()
        self.max_features = max_features
        self.retrieval_pool = retrieval_pool
        self.top_k = top_k
        self.ratio = ratio
        self.global_top_k = global_top_k
        self.global_ratio = global_ratio
        self.reprojection_error = reprojection_error
        self.focal_multipliers = (0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.7, 1.9, 2.1)

        state = torch.load(str(checkpoint.resolve()), map_location="cpu")
        weights = state["extractor"] if "extractor" in state else state["disk"]
        self.disk = DISK(window=8, desc_dim=128)
        self.disk.load_state_dict(weights)
        # SAM3 enables BF16-oriented CUDA paths. The validated standalone DISK
        # detector runs in FP32, so pin this model and its forward pass to FP32
        # even though both models share one long-lived process.
        self.disk = self.disk.float().cuda()

        self.cameras, images, points3d = read_model(str(self.environment / "colmap"), ext="")
        self.images_by_stem = {Path(image.name).stem: image for image in images.values()}
        self.points3d = points3d
        self.reference_mask_dir = self.environment / "segmentations" / "thing"
        reference_h5 = self.environment / "h5"
        self.descriptor_h5 = h5py.File(reference_h5 / "descriptors.h5", "r")
        self.keypoint_h5 = h5py.File(reference_h5 / "keypoints.h5", "r")
        self.reference_descriptors: dict[str, torch.Tensor] = {}
        self.reference_keypoints: dict[str, np.ndarray] = {}
        self.reference_static: dict[str, tuple[np.ndarray, bool]] = {}

        self.appearance_references: list[tuple[str, np.ndarray, np.ndarray]] = []
        image_dir = self.environment / "images"
        for image in images.values():
            name = Path(image.name).name
            path = image_dir / name
            if not path.is_file():
                continue
            reference = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
            if reference is None:
                continue
            color, structure = compact_visual_features(reference)
            self.appearance_references.append((name, color, structure))
        if len(self.appearance_references) < 2:
            raise RuntimeError("Too few readable registered reference images")

        index = np.load(global_index.resolve())
        names = np.asarray(index["names"]).astype(str)
        vectors = np.asarray(index["vectors"], dtype=np.float32)
        keep = np.asarray([name in self.images_by_stem for name in names], dtype=bool)
        self.global_names = names[keep]
        self.global_vectors = vectors[keep]

    def close(self) -> None:
        self.descriptor_h5.close()
        self.keypoint_h5.close()

    def extract_disk(self, image_path: Path) -> tuple[np.ndarray, np.ndarray]:
        rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(image_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        original_height, original_width = rgb.shape[:2]
        target_height = (original_height + 15) // 16 * 16
        target_width = (original_width + 15) // 16 * 16
        scale = min(target_height / original_height, target_width / original_width)
        resized_height = max(1, int(scale * original_height))
        resized_width = max(1, int(scale * original_width))
        bitmap = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        bitmap = F.interpolate(
            bitmap.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )
        bitmap = F.pad(bitmap, (0, target_width - resized_width, 0, target_height - resized_height))
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            batch = self.disk.features(
                bitmap.float().cuda(non_blocking=True),
                kind="nms",
                window_size=5,
                cutoff=0.0,
                n=self.max_features,
            )
        feature = batch.flat[0].to(torch.device("cpu"))
        keypoints = feature.kp.T.float().numpy().T / scale
        descriptors = feature.desc.float().numpy()
        scores = feature.kp_logp.float().numpy()
        valid = (
            (keypoints[:, 0] >= 0)
            & (keypoints[:, 0] < original_width)
            & (keypoints[:, 1] >= 0)
            & (keypoints[:, 1] < original_height)
        )
        keypoints = keypoints[valid]
        descriptors = descriptors[valid]
        scores = scores[valid]
        order = np.argsort(scores)[::-1]
        return keypoints[order].astype(np.float32), descriptors[order].astype(np.float32)

    def coarse_candidates(self, image_path: Path, mask_path: Path) -> list[str]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Missing endpoint image or mask: {image_path}, {mask_path}")
        query_color, query_structure = compact_visual_features(image)
        query_static = cv2.resize(mask, (160, 90), interpolation=cv2.INTER_NEAREST) == 0
        if int(query_static.sum()) < 576:
            query_static = None
        scores = []
        for name, color, structure in self.appearance_references:
            if query_static is None:
                color_error = float(np.mean(np.abs(query_color - color)))
                structure_error = float(np.mean(np.abs(query_structure - structure)))
            else:
                color_error = float(np.mean(np.abs(query_color[query_static] - color[query_static])))
                structure_error = float(
                    np.mean(np.abs(query_structure[query_static] - structure[query_static]))
                )
            scores.append((color_error + 0.60 * structure_error, name))
        scores.sort(key=lambda item: item[0])
        return [name for _, name in scores[: self.retrieval_pool]]

    def global_candidates(
        self, descriptors: np.ndarray, keypoints: np.ndarray, mask_path: Path
    ) -> list[str]:
        keep = localizer.static_query_indices(keypoints, mask_path)
        if len(keep) < 20:
            raise RuntimeError("The query mask leaves too few DISK descriptors for global retrieval")
        query = descriptors[keep].mean(axis=0)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        scores = self.global_vectors @ query.astype(np.float32)
        selected = np.argsort(-scores)[: self.global_top_k]
        return [f"{self.global_names[index]}.jpg" for index in selected]

    def _reference_descriptor(self, stem: str) -> torch.Tensor:
        value = self.reference_descriptors.get(stem)
        if value is None:
            array = np.asarray(self.descriptor_h5[stem], dtype=np.float32)
            value = F.normalize(torch.from_numpy(array).cuda(non_blocking=True), dim=1)
            self.reference_descriptors[stem] = value
        return value

    def _reference_keypoint(self, stem: str) -> np.ndarray:
        value = self.reference_keypoints.get(stem)
        if value is None:
            value = np.asarray(self.keypoint_h5[stem], dtype=np.float32)
            self.reference_keypoints[stem] = value
        return value

    def _reference_static(self, stem: str, keypoints: np.ndarray) -> tuple[np.ndarray, bool]:
        value = self.reference_static.get(stem)
        if value is None:
            value = localizer.static_reference_indices(
                keypoints, self.reference_mask_dir / f"{stem}.png", 1
            )
            self.reference_static[stem] = value
        return value

    @staticmethod
    def mutual_matches(
        query: torch.Tensor, reference: torch.Tensor, ratio: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(query) < 2 or len(reference) < 2:
            return np.empty(0, np.int64), np.empty(0, np.int64)
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

    def localize(
        self,
        *,
        image_path: Path,
        mask_path: Path,
        keypoints: np.ndarray,
        descriptors: np.ndarray,
        candidates: list[str],
        ratio: float,
        strategy: str,
    ) -> dict:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        image_size = (int(image.shape[1]), int(image.shape[0]))
        query_feature_count_before_mask = len(keypoints)
        keep_query = localizer.static_query_indices(keypoints, mask_path)
        keypoints = keypoints[keep_query]
        descriptors = descriptors[keep_query]
        if len(keypoints) < 20:
            raise RuntimeError("The query mask leaves too few DISK keypoints")
        query_tensor = F.normalize(torch.from_numpy(descriptors).cuda(non_blocking=True), dim=1)

        observations: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, str]] = {}
        votes: dict[tuple[int, int], int] = defaultdict(int)
        candidate_stats = []
        for image_name in candidates:
            stem = Path(image_name).stem
            image_record = self.images_by_stem.get(stem)
            if image_record is None or stem not in self.descriptor_h5 or stem not in self.keypoint_h5:
                continue
            query_ids, reference_ids = self.mutual_matches(
                query_tensor, self._reference_descriptor(stem), ratio
            )
            raw_mutual = len(query_ids)
            reference_keypoints = self._reference_keypoint(stem)
            reference_static, mask_available = self._reference_static(stem, reference_keypoints)
            static_match = reference_static[reference_ids]
            excluded = int((~static_match).sum())
            query_ids = query_ids[static_match]
            reference_ids = reference_ids[static_match]
            point_ids = np.asarray(image_record.point3D_ids, dtype=np.int64)
            usable = point_ids[reference_ids] >= 0
            query_ids = query_ids[usable]
            reference_ids = reference_ids[usable]
            query_ids, reference_ids = localizer.verify_pair_geometry(
                keypoints, reference_keypoints, query_ids, reference_ids, 2.0
            )
            valid_matches = 0
            for query_id, reference_id in zip(query_ids, reference_ids):
                point_id = int(point_ids[reference_id])
                point = self.points3d.get(point_id)
                if point is None:
                    continue
                key = (int(query_id), point_id)
                observations[key] = (
                    keypoints[query_id], np.asarray(point.xyz, dtype=np.float64), image_name
                )
                votes[key] += 1
                valid_matches += 1
            candidate_stats.append(
                {
                    "image": image_name,
                    "mutual_matches": int(raw_mutual),
                    "reference_person_mask_available": mask_available,
                    "reference_person_matches_excluded": excluded,
                    "geometry_inliers": int(len(query_ids)),
                    "matches_with_3d": valid_matches,
                }
            )

        best_keys: dict[int, tuple[int, int]] = {}
        for key, count in votes.items():
            query_id = key[0]
            previous = best_keys.get(query_id)
            if previous is None or count > votes[previous]:
                best_keys[query_id] = key
        selected = [observations[key] for _, key in sorted(best_keys.items())]
        if len(selected) < 6:
            raise RuntimeError(f"Only {len(selected)} query features have a unique 3D-track vote")
        xy = np.asarray([item[0] for item in selected], dtype=np.float64)
        xyz = np.asarray([item[1] for item in selected], dtype=np.float64)
        provenance = [item[2] for item in selected]
        focal_values = [factor * max(image_size) for factor in self.focal_multipliers]
        pnp, inlier_ids = localizer.pnp_with_focal_search(
            xyz, xy, image_size, focal_values, self.reprojection_error
        )
        rotation, _ = cv2.Rodrigues(np.asarray(pnp["rotation_vector"], dtype=np.float64))
        translation = np.asarray(pnp["translation"], dtype=np.float64)
        center = -rotation.T @ translation
        sources: dict[str, int] = defaultdict(int)
        for index in inlier_ids:
            sources[provenance[int(index)]] += 1
        return {
            "query_image": str(image_path.resolve()),
            "query_key": image_path.stem,
            "coordinate_system": "raw_colmap_world (same as environment_dir/colmap/pc.ply)",
            "localization_strategy": strategy,
            "query_feature_count_before_mask": int(query_feature_count_before_mask),
            "query_static_feature_count": int(len(keypoints)),
            "query_dynamic_mask": str(mask_path.resolve()),
            "reference_person_mask_dir": str(self.reference_mask_dir.resolve()),
            "retrieved_candidate_count": int(len(candidates)),
            "candidate_match_stats": candidate_stats,
            "unique_2d_3d_correspondence_count": int(len(observations)),
            "selected_query_feature_count": int(len(xy)),
            "pnp": pnp,
            "world_to_camera_rotation": rotation.tolist(),
            "world_to_camera_translation": translation.tolist(),
            "camera_center": center.tolist(),
            "inlier_source_images": dict(sorted(sources.items(), key=lambda item: -item[1])),
        }

    def endpoint_once(
        self, image_path: Path, mask_path: Path, output: Path, strategy_suffix: str = ""
    ) -> dict:
        keypoints, descriptors = self.extract_disk(image_path)
        coarse_pose = None
        coarse_error = None
        try:
            coarse_pose = self.localize(
                image_path=image_path,
                mask_path=mask_path,
                keypoints=keypoints,
                descriptors=descriptors,
                candidates=self.coarse_candidates(image_path, mask_path)[: self.top_k],
                ratio=self.ratio,
                strategy=f"coarse_appearance{strategy_suffix}",
            )
        except Exception as error:  # global retrieval is the tested fallback
            coarse_error = f"{type(error).__name__}: {error}"
        coarse_quality = assess_pose_quality(coarse_pose) if coarse_pose is not None else None
        if coarse_quality is not None and coarse_quality["status"] == "accepted":
            selected = coarse_pose
        else:
            global_pose = self.localize(
                image_path=image_path,
                mask_path=mask_path,
                keypoints=keypoints,
                descriptors=descriptors,
                candidates=self.global_candidates(descriptors, keypoints, mask_path),
                ratio=self.global_ratio,
                strategy=f"global_disk_fallback{strategy_suffix}",
            )
            candidates = [global_pose] + ([coarse_pose] if coarse_pose is not None else [])

            def rank(pose: dict) -> tuple[int, int, float]:
                quality = assess_pose_quality(pose)
                return (
                    {"rejected": 0, "review": 1, "accepted": 2}[quality["status"]],
                    int(quality["pnp_inlier_count"]),
                    -float(quality["pnp_rmse_pixels"]),
                )

            selected = max(candidates, key=rank)
        selected["quality"] = assess_pose_quality(selected)
        if coarse_error is not None:
            selected["coarse_localization_error"] = coarse_error
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(selected, indent=2) + "\n")
        return selected

    def endpoint_for_clip(self, image_path, mask_path, output, video, frame_index,
                          frame_count, anchor_source, reference_dirs):
        reason = None
        if anchor_source == "registered-first":
            pose, reason = registered_anchor(video, frame_index, frame_count, image_path,
                                             self.cameras, self.images_by_stem, self.points3d,
                                             reference_dirs)
            if pose is not None:
                pose["quality"] = assess_pose_quality(pose)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(pose, indent=2) + "\n")
                return pose
        pose = self.endpoint(image_path, mask_path, output)
        pose["pose_source"] = "estimated_disk_pnp"
        pose["requested_anchor_source"] = anchor_source
        if reason is not None:
            pose["registered_lookup_fallback"] = reason
        output.write_text(json.dumps(pose, indent=2) + "\n")
        return pose

    def endpoint(self, image_path: Path, mask_path: Path, output: Path) -> dict:
        attempts: list[tuple[str, Path]] = [("", mask_path)]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        # The union mask is intentionally dilated for camera tracking. If an
        # actor fills most of the frame, that conservative mask can remove too
        # many static DISK keypoints for PnP. Retry with the tracked person
        # support eroded back toward its original extent, then with an empty
        # dynamic mask. The tested PnP/retrieval algorithm itself is unchanged.
        retry_dir = output.parent / "mask_retries"
        retry_dir.mkdir(parents=True, exist_ok=True)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        undilated = cv2.erode(mask, kernel, iterations=1)
        undilated_path = retry_dir / f"{image_path.stem}_undilated.png"
        empty_path = retry_dir / f"{image_path.stem}_empty.png"
        cv2.imwrite(str(undilated_path), undilated)
        cv2.imwrite(str(empty_path), np.zeros_like(mask))
        attempts.extend((("_undilated_mask_retry", undilated_path), ("_empty_mask_retry", empty_path)))
        errors = []
        for suffix, active_mask in attempts:
            try:
                pose = self.endpoint_once(image_path, active_mask, output, suffix)
                if errors:
                    pose["localization_retry_errors"] = errors
                    output.write_text(json.dumps(pose, indent=2) + "\n")
                return pose
            except Exception as error:
                errors.append(
                    {"strategy_suffix": suffix, "error": f"{type(error).__name__}: {error}"}
                )
        raise RuntimeError(f"Endpoint localization failed after retries: {errors}")


def run_sam3_tracker(
    *,
    predictor,
    sam3_module,
    frames_dir: Path,
    masks_dir: Path,
    sam3_root: Path,
    checkpoint: Path,
    version: str,
    probability: float,
    dilate: int,
) -> None:
    original_builder = sam3_module.build_sam3_predictor
    sam3_module.build_sam3_predictor = lambda *args, **kwargs: predictor
    original_argv = sys.argv
    sys.argv = [
        str(Path(sam3_tracker.__file__)),
        "--frames-dir", str(frames_dir),
        "--output-dir", str(masks_dir),
        "--sam3-root", str(sam3_root),
        "--checkpoint", str(checkpoint),
        "--version", version,
        "--prob-threshold", str(probability),
        "--dilate-pixels", str(dilate),
        "--device", "cuda",
    ]
    try:
        with contextlib.redirect_stdout(sys.stderr):
            sam3_tracker.main()
    finally:
        sys.argv = original_argv
        sam3_module.build_sam3_predictor = original_builder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-source", choices=("registered-first", "estimated"), default="registered-first")
    parser.add_argument("--registered-images-dir", type=Path, action="append", default=[])
    parser.add_argument("--keep-failed-output", action="store_true")
    parser.add_argument("--environment-dir", type=Path, required=True)
    parser.add_argument("--sam3-root", type=Path, required=True)
    parser.add_argument("--sam3-checkpoint", type=Path, required=True)
    parser.add_argument("--sam3-version", default="sam3")
    parser.add_argument("--sam3-prob-threshold", type=float, default=0.45)
    parser.add_argument("--sam3-dilate-pixels", type=int, default=12)
    parser.add_argument("--disk-checkpoint", type=Path, required=True)
    parser.add_argument("--global-index", type=Path, required=True)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Reserve the real stdout pipe for JSONL IPC. Third-party model code can
    # still print diagnostics, but those messages belong in the worker log.
    sys.stdout = sys.stderr
    sam3_root = args.sam3_root.resolve()
    if str(sam3_root) not in sys.path:
        sys.path.insert(0, str(sam3_root))
    import sam3

    print("[stage1] loading SAM3 predictor", file=sys.stderr, flush=True)
    predictor = sam3.build_sam3_predictor(
        checkpoint_path=str(args.sam3_checkpoint.resolve()),
        version=args.sam3_version,
        compile=False,
        async_loading_frames=False,
    )
    print("[stage1] loading DISK and scene localization cache", file=sys.stderr, flush=True)
    pnp = DiskPnPLocalizer(
        environment=args.environment_dir,
        checkpoint=args.disk_checkpoint,
        global_index=args.global_index,
    )
    reference_dirs = [args.environment_dir / "images",
                      REPO_ROOT / "data/sparse_reconstruction_and_image_data/Friends-monica_apartment/images",
                      *args.registered_images_dir]
    emit({"type": "ready", "stage": 1})
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            job = json.loads(line)
            started = time.perf_counter()
            output_dir = Path(job["output_dir"]).resolve()
            try:
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                frames_dir = output_dir / "frames"
                masks_dir = output_dir / "masks"
                with contextlib.redirect_stdout(sys.stderr):
                    metadata = extract_frames(
                        Path(job["video"]).resolve(), frames_dir, args.jpeg_quality, 0
                    )
                run_sam3_tracker(
                    predictor=predictor,
                    sam3_module=sam3,
                    frames_dir=frames_dir,
                    masks_dir=masks_dir,
                    sam3_root=sam3_root,
                    checkpoint=args.sam3_checkpoint.resolve(),
                    version=args.sam3_version,
                    probability=args.sam3_prob_threshold,
                    dilate=args.sam3_dilate_pixels,
                )
                first_index = 0
                last_index = int(metadata["frame_count"]) - 1
                first = frames_dir / f"{first_index:06d}.jpg"
                last = frames_dir / f"{last_index:06d}.jpg"
                union = masks_dir / "union"
                with contextlib.redirect_stdout(sys.stderr):
                    first_pose = pnp.endpoint_for_clip(
                        first, union / f"{first_index:06d}.png",
                        output_dir / "anchor" / "localization_colmap.json",
                        job["video"], first_index, int(metadata["frame_count"]), args.anchor_source, reference_dirs,
                    )
                    last_pose = pnp.endpoint_for_clip(
                        last, union / f"{last_index:06d}.png",
                        output_dir / "tail_anchor" / "localization_colmap.json",
                        job["video"], last_index, int(metadata["frame_count"]), args.anchor_source, reference_dirs,
                    )
                emit(stage_result(job, stage=1, status="ok",
                        frame_count=int(metadata["frame_count"]),
                        first_quality=first_pose["quality"],
                        last_quality=last_pose["quality"],
                        anchor_source_policy=args.anchor_source,
                        first_anchor_source=first_pose.get("pose_source", "estimated_disk_pnp"),
                        last_anchor_source=last_pose.get("pose_source", "estimated_disk_pnp"),
                        stage1_seconds=round(time.perf_counter() - started, 3),
                ))
            except Exception as error:
                if not args.keep_failed_output:
                    shutil.rmtree(output_dir, ignore_errors=True)
                emit(stage_result(job, stage=1, status="failed",
                        error=f"{type(error).__name__}: {error}",
                        traceback=traceback.format_exc(),
                        stage1_seconds=round(time.perf_counter() - started, 3),
                ))
    finally:
        pnp.close()


if __name__ == "__main__":
    main()
