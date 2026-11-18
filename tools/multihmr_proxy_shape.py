"""Fit head and torso dimensions to SAM3 silhouettes at existing DA3 depths."""

import cv2
import numpy as np
from scipy.ndimage import median_filter


def norm(x):
    return np.linalg.norm(x)


def span_near(mask, y, x):
    row = mask[int(np.clip(round(y), 0, mask.shape[0] - 1))] > 0
    edges = np.diff(np.r_[0, row.astype(int), 0])
    spans = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    if not spans:
        return None
    return min(spans, key=lambda s: max(s[0] - x, x - s[1], 0))


def sample(obs, cam):
    m = cv2.imread(obs["mask"], 0)
    x, y = obs["anchor_px"]
    fit = obs["fit"]
    if not fit or m is None or not m.any():
        return None
    K = np.array(cam["intrinsics"])
    f = K[0, 0]
    j = np.array(fit["joints_camera_m"])
    uv = np.array(fit["joints_original_px"])
    x0, y0, x1, y1 = fit["head_bounds"]
    if not np.isfinite([*fit["head_bounds"], *j.ravel()]).all():
        return None
    w, h = x1 - x0, y1 - y0
    if min(w, h) < 5 or h > m.shape[0] * 1.4:
        return None
    mx = (x0 + x1) / 2
    # Hair alters the visible crown. Use the instance's upper silhouette and
    # retain the mesh width on a side cropped by the frame.
    # Estimate skull width from a contiguous upper-head cross section of this
    # instance. Exclude border-truncated spans; the SMPL-X head then supplies
    # the hidden portion. Refine the row heights using the measured diameter.
    diameter = np.clip(0.20 * (np.ptp(np.where(m)[0]) + 1), 40, 260)
    valid = []
    for _ in range(3):
        spans = [
            span_near(m, y + t * diameter, x) for t in (0.31, 0.38, 0.45, 0.52, 0.59)
        ]
        usable = [
            (l, r)
            for s in spans
            if s is not None
            for l, r in [s]
            if l > 0
            and r < m.shape[1]
            and l - 0.3 * diameter < x < r + 0.3 * diameter
            and r - l > 16
        ]
        if len(usable) < 3:
            break
        valid = usable
        diameter = np.median([r - l for l, r in valid])
    if valid:
        width = diameter * 0.96
        center_x = np.median([(l + r - 1) / 2 for l, r in valid])
        height = width * np.clip(h / w, 1.20, 1.37)
    else:
        width = w
        height = np.clip(y1 - y, 1.14 * w, 1.45 * w)
        center_x = mx
    center = np.array([center_x, y + height * 0.5])
    mesh_h = max(h, 1)
    delta_pelvis = (uv[0] - [(x0 + x1) / 2, (y0 + y1) / 2]) / mesh_h
    if not (0.55 < delta_pelvis[1] < 6 and abs(delta_pelvis[0]) < 3.5):
        return None
    shoulder = j[16] - j[17]
    body_width = norm(shoulder) / j[15, 2] * f * 1.14
    width_ratio = np.clip(body_width / mesh_h, 1.55, 2.02)
    if norm(shoulder) < 1e-6 or j[15, 2] <= 0:
        return None
    return np.array(
        [
            *(center - [x, y]),
            width,
            height,
            *delta_pelvis,
            (j[0, 2] - j[15, 2]) / j[15, 2],
            width_ratio,
            *list(shoulder / norm(shoulder)),
        ]
    )


def fit_shapes(rows, cams, image_width):
    by = {}
    for row in rows:
        i = row["frame_index"]
        for ob in row["people"]:
            by.setdefault(ob["track_id"], []).append((i, ob))
    for tid, frames in by.items():
        measured = [sample(ob, cams[i]) for i, ob in frames]
        valid = [k for k, m in enumerate(measured) if m is not None]
        if not valid:
            continue
        values = np.array([measured[k] for k in valid])
        times = np.array([i for i, ob in frames])
        result = np.stack([np.interp(times, times[valid], x) for x in values.T], axis=1)
        result = median_filter(result, size=(5, 1), mode="nearest")
        for k, ((i, ob), v) in enumerate(zip(frames, result)):
            c = cams[i]
            K = np.array(c["intrinsics"])
            R = np.array(c["world_to_camera_rotation"])
            C = np.array(c["camera_center"])
            z = (R @ (np.asarray(ob["head_anchor_world"]) - C))[2]
            uv = np.array(ob["anchor_px"]) + v[:2]
            width, height = v[2:4]
            # If a head is cropped, preserve its estimated full size and centre;
            # visible ownership is checked independently using the actual mask.
            pelvis_uv = uv + height * v[4:6]
            pelvis_z = z * (1 + v[6])

            def world(uvpoint, depth):
                return C + R.T @ (np.linalg.solve(K, np.r_[uvpoint, 1.0]) * depth)

            center = world(uv, z)
            pelvis = world(pelvis_uv, pelvis_z)
            ob["shape"] = {
                "center_px": uv.tolist(),
                "center_world": center.tolist(),
                "head_wh_world": (v[2:4] * z / np.diag(K)[:2]).tolist(),
                "pelvis_world": pelvis.tolist(),
                "shoulder_axis_world": (R.T @ v[8:11]).tolist(),
                "torso_width_world": float(v[7] * height * z / K[0, 0]),
                "head_wh_px": v[2:4].tolist(),
                "source": "frame" if k in valid else "track_interpolation_or_hold",
            }
    # An image boundary or foreground occluder can remove half a head. Use
    # shot-specific physical head widths from fully visible tracks to constrain
    # the scale of those partial tracks; preserve their own calibrated depth.
    healthy = []
    for tid, frames in by.items():
        measurements = [
            o["shape"]["head_wh_world"][0]
            for _, o in frames
            if "shape" in o
            and 0.16875 * image_width < o["anchor_px"][0] < 0.83125 * image_width
            and o["shape"]["source"] == "frame"
        ]
        if len(measurements) >= len(frames) / 2:
            healthy.append(float(np.median(measurements)))
    typical = float(np.median(healthy)) if healthy else None
    for tid, frames in by.items():
        if not typical or "shape" not in frames[0][1]:
            continue
        widths = [o["shape"]["head_wh_world"][0] for _, o in frames]
        partial = (
            np.mean(
                [
                    min(o["anchor_px"][0], image_width - o["anchor_px"][0])
                    < 0.16875 * image_width
                    for _, o in frames
                ]
            )
            > 0.5
        )
        if not partial:
            continue
        spread = np.percentile(widths, 90) / max(np.percentile(widths, 10), 1e-6)
        bad = spread > 1.40 or not (0.77 < np.median(widths) / typical < 1.28)
        if not bad:
            continue
        reliable_width = typical
        for i, o in frames:
            s = o["shape"]
            c = cams[i]
            K = np.array(c["intrinsics"])
            R = np.array(c["world_to_camera_rotation"])
            C = np.array(c["camera_center"])
            prev = np.array(s["center_world"])
            z = (R @ (prev - C))[2]
            width = reliable_width * K[0, 0] / z
            height = 1.29 * width
            x, y = o["anchor_px"]
            m = cv2.imread(o["mask"], 0)
            spans = [span_near(m, y + t * width, x) for t in (0.29, 0.36, 0.43, 0.50)]
            side = [v for v in spans if v is not None]
            if x > image_width / 2:
                xs = [l + width * 0.5 for l, r in side]
            else:
                xs = [r - width * 0.5 for l, r in side]
            u = float(np.median(xs)) if xs else s["center_px"][0]
            xy = [u, y + height * 0.5]
            center = C + R.T @ (np.linalg.solve(K, np.r_[xy, 1]) * z)
            ratio = height / s["head_wh_px"][1]
            s["pelvis_world"] = (
                center + (np.array(s["pelvis_world"]) - prev) * ratio
            ).tolist()
            s["center_px"] = xy
            s["center_world"] = center.tolist()
            s["head_wh_world"] = [reliable_width, height * z / K[1, 1]]
            s["torso_width_world"] *= ratio
            s["head_wh_px"] = [width, height]
            s["scale_constraint"] = "partial_head_shot_prior"
