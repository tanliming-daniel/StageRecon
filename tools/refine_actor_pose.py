#!/usr/bin/env python3
"""Add visibility-aware Multi-HMR orientations to existing SAM3/DA3 actor tracks."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from multihmr_proxy_shape import fit_shapes


def mask_hits(mask, points, margin=0):
    h, w = mask.shape
    x, y = np.rint(points).astype(int).T
    valid = (x >= -margin) & (x < w + margin) & (y >= -margin) & (y < h + margin)
    return valid & (mask[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)] != 0)


def affinity(p, mask, box):
    points = p["joints_original_px"][[15, 12, 9, 13, 14, 16, 17, 6, 3, 0]]
    x1, y1, x2, y2 = box
    tol = max(5, min(24, 0.05 * (x2 - x1 + 1)))
    if not (
        y1 - tol <= points[0, 1] <= y1 + 0.55 * (y2 - y1) + tol
        and mask_hits(mask, points[:1], tol)[0]
    ):
        return 0.0
    h, w = mask.shape
    valid = np.all((points >= 0) & (points < [w, h]), axis=1)
    if not np.any(valid):
        return 0.0
    weights = np.array([4, 3, 2, 1, 1, 1, 1, 1, 1, 1])
    return 0.55 + 0.45 * np.average(
        mask_hits(mask, points[valid]), weights=weights[valid]
    )


def crop_prediction(model, image, mask, box, K, expanded):
    x1, y1, x2, y2 = box
    h, w = mask.shape
    pad = int(max(x2 - x1, y2 - y1) * 0.14) + 8
    l, t, r, b = (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w, x2 + pad + 1),
        min(h, y2 + pad + 1),
    )
    region = image[t:b, l:r].copy()
    bg = cv2.dilate(mask[t:b, l:r], np.ones((13, 13), np.uint8)) == 0
    region[bg] = 0.2 * region[bg] + 85
    local = K.copy()
    local[:2, 2] -= [l, t]
    candidates = model.predict(region, local, 0.12)
    for p in candidates:
        p["joints_original_px"] += [l, t]
        p["head_bounds"] += [l, t, l, t]
    best = max(candidates, key=lambda p: affinity(p, expanded, box), default=None)
    return best if best is not None and affinity(best, expanded, box) > 0.69 else None


def read_json(path):
    return json.loads(path.read_text())


def associate(model, actors, cameras, sam, frames_dir, mask_base):
    masks = {
        (t["track_id"], f["frame_index"]): f for t in sam["tracks"] for f in t["frames"]
    }
    frames = {}
    for tr in actors["tracks"]:
        if tr.get("proxy_quality", {}).get("eligible", True):
            for f in tr["frames"]:
                frames.setdefault(f["frame_index"], []).append((tr["track_id"], f))
    rows = []
    for i, cam in sorted(cameras.items()):
        row = {"frame_index": i, "people": []}
        rows.append(row)
        tracks = frames.get(i, [])
        if not tracks:
            continue
        image = cv2.imread(str(frames_dir / cam["frame_name"]))
        if image is None or image.shape[:2] != (
            sam["image_size"]["height"],
            sam["image_size"]["width"],
        ):
            raise ValueError(f"Missing frame or SAM3 image size mismatch at {i}")
        K = np.asarray(cam["intrinsics"], dtype=float)
        candidates = model.predict(image, K, 0.3)
        entries = []
        for tid, f in tracks:
            obs = masks[tid, i]
            if obs["frame_name"] != cam["frame_name"]:
                raise ValueError(f"Frame mismatch for ID {tid} at {i}")
            path = mask_base / obs["mask"]
            m = cv2.imread(str(path), 0)
            if m is None or m.shape != image.shape[:2]:
                raise ValueError(f"Invalid mask {path}")
            expanded = cv2.dilate(m, np.ones((7, 7), np.uint8))
            entries.append((tid, f, path, m, expanded, obs["bbox_xyxy"]))
        scores = np.array(
            [
                [affinity(p, m, box) for p in candidates]
                for _, _, _, _, m, box in entries
            ]
        )
        a, b = linear_sum_assignment(
            np.column_stack((scores, np.full((len(entries), len(entries)), 0.69))),
            maximize=True,
        )
        matched = {
            j: k for j, k in zip(a, b) if k < len(candidates) and scores[j, k] > 0.69
        }
        for j, (tid, f, path, m, expanded, box) in enumerate(entries):
            pose = (
                candidates[matched[j]]
                if j in matched
                else crop_prediction(model, image, m, box, K, expanded)
            )
            source = (
                "whole" if j in matched else ("crop" if pose is not None else "missing")
            )
            shoulders = False
            if pose is not None:
                pts = pose["joints_original_px"]
                # Face-only crops can still regress plausible shoulders outside the visible body.
                left, top, right, bottom = pose["head_bounds"]
                shoulders = bool(
                    mask_hits(expanded, pts[[16, 17, 9]]).all()
                    and min(pts[16, 1], pts[17, 1]) > top + 0.65 * (bottom - top)
                    and np.linalg.norm(pts[16] - pts[17]) > 0.65 * (right - left)
                )
            anchor = f.get("head_anchor_world")
            if anchor is None:
                R = np.asarray(cam["world_to_camera_rotation"])
                C = np.asarray(cam["camera_center"])
                z = (R @ (np.asarray(f["bbox_3d"]["center_world"]) - C))[2]
                anchor = C + R.T @ (np.linalg.solve(K, [*f["body_anchor_xy"], 1]) * z)
            row["people"].append(
                dict(
                    track_id=tid,
                    actor=f,
                    mask=str(path),
                    anchor_px=f["body_anchor_xy"],
                    head_anchor_world=anchor,
                    fit=pose,
                    source=source,
                    shoulders=shoulders,
                )
            )
    return rows


def world_rotations(pose, cam):
    local = Rotation.from_rotvec(pose["rotvecs"]).as_matrix()
    basis = np.asarray(cam["world_to_camera_rotation"]).T
    for j in [0, 3, 6, 9]:
        basis = basis @ local[j]
    return basis, basis @ local[12] @ local[15]


def apply_orientations(rows, cameras, actors):
    body_history = {}
    head_history = {}
    for row in rows:
        i = row["frame_index"]
        for p in row["people"]:
            tid = p["track_id"]
            f = p["actor"]
            shape = p.get("shape")
            head_source = "actor_track"
            head = None
            body_source = "actor_track"
            body_facing = f["facing_world"]
            if p["fit"] is not None:
                body, head = world_rotations(p["fit"], cameras[i])
                head_history[tid] = (i, head)
                head_source = p["source"]
                if p["shoulders"]:
                    offset = (
                        None
                        if not shape
                        else np.subtract(shape["pelvis_world"], shape["center_world"])
                    )
                    body_history[tid] = (
                        i,
                        body,
                        offset,
                        shape and shape["torso_width_world"],
                    )
            elif tid in head_history and i - head_history[tid][0] <= 3:
                head = head_history[tid][1]
                head_source = "previous_head"
            if tid in body_history:
                seen, body, offset, width = body_history[tid]
                body_facing = body[:, 2].tolist()
                body_source = p["source"] if seen == i else "previous_shoulders"
                if shape and seen != i:
                    shape["shoulder_axis_world"] = body[:, 0].tolist()
                    if offset is not None:
                        shape["pelvis_world"] = (
                            shape["center_world"] + offset
                        ).tolist()
                        shape["torso_width_world"] = width
            elif shape:
                up = np.asarray(
                    actors.get("proxy_convention", {}).get(
                        "up_world",
                        -np.asarray(cameras[i]["world_to_camera_rotation"])[1],
                    ),
                    dtype=float,
                )
                up /= np.linalg.norm(up)
                side = np.cross(up, f["facing_world"])
                if np.linalg.norm(side) < 1e-6:
                    side = np.asarray(cameras[i]["world_to_camera_rotation"])[0]
                shape["shoulder_axis_world"] = (side / np.linalg.norm(side)).tolist()
                # Only the head is observed. Use a neutral torso beneath it until both shoulders appear.
                shape["pelvis_world"] = (
                    np.asarray(shape["center_world"])
                    - 2.4 * shape["head_wh_world"][1] * up
                ).tolist()
            f["proxy_orientation"] = {
                "head_source": head_source,
                "body_source": body_source,
                "shoulders_visible": p["shoulders"],
                "body_facing_world": body_facing,
                "body_reference_frame": (
                    body_history[tid][0] if tid in body_history else None
                ),
            }
            if head is not None:
                f["proxy_orientation"]["head_basis_world"] = head[:, [2, 0, 1]].tolist()
            if shape:
                f["fitted_proxy"] = shape
    actors["pose_estimator"] = "sam3_multihmr_visibility_v1"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in [
        "actor-tracks",
        "camera-track",
        "person-tracks",
        "frames-dir",
        "multihmr-source",
        "checkpoint",
        "output",
    ]:
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--dinov2-source", type=Path)
    args = ap.parse_args()
    from multihmr_inference import MultiHMR

    model = MultiHMR(
        args.multihmr_source,
        args.checkpoint,
        args.dinov2_source and args.dinov2_source.resolve(),
    )
    actors = read_json(args.actor_tracks)
    camera = read_json(args.camera_track)
    if "3d_bbox" not in actors.get("representation", ""):
        raise ValueError("Expected SAM3/DA3 actor boxes")
    cams = {f["frame_index"]: f for f in camera["frames"]}
    rows = associate(
        model,
        actors,
        cams,
        read_json(args.person_tracks),
        args.frames_dir,
        args.person_tracks.parent,
    )
    fit_shapes(rows, cams, camera["resolution"]["width"])
    apply_orientations(rows, cams, actors)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
        try:
            json.dump(actors, tmp, allow_nan=False)
            tmp.flush()
            temporary.replace(args.output)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
