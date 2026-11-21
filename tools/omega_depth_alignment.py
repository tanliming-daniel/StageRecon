"""Calibrate actor depth against the fixed Omega camera trajectory."""
import cv2
import numpy as np


def depth_samples(xyz, camera, depth, confidence, mask, size):
    h, w = depth.shape
    k = np.array(camera["intrinsics"], dtype=float)
    k[:2] *= np.array([[w / size[0]], [h / size[1]]])
    r = np.asarray(camera["world_to_camera_rotation"], np.float32)
    t = np.asarray(camera["world_to_camera_translation"], np.float32)
    zbuf = np.full((h, w), np.inf, np.float32)
    for start in range(0, len(xyz), 250000):
        p = xyz[start:start + 250000] @ r.T + t
        p = p[np.isfinite(p).all(axis=1) & (p[:, 2] > 1e-4)]
        uv = np.rint(p[:, :2] / p[:, 2:] * [k[0, 0], k[1, 1]] + k[:2, 2]).astype(int)
        x, y = uv.T
        ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        np.minimum.at(zbuf, (y[ok], x[ok]), p[ok, 2])
    occupied = np.isfinite(zbuf)
    z = np.where(occupied, zbuf, 0)
    kernel = np.ones((3, 3), np.uint8)
    valid = occupied & ((cv2.dilate(z, kernel) - cv2.erode(z, kernel)) < 0.06 * z)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    valid &= (cv2.dilate(mask, np.ones((7, 7), np.uint8)) == 0)
    valid &= np.isfinite(depth) & (depth > 1e-5)
    if confidence is not None and valid.any():
        valid &= np.isfinite(confidence)
        if valid.any():
            valid &= confidence >= np.percentile(confidence[valid], 35)
    y, x = np.where(valid)
    ratio = np.log(z[y, x] / depth[y, x])
    hold = ((x // 12 + 3 * (y // 12)) % 5 == 0)
    near = cv2.dilate(mask, np.ones((31, 31), np.uint8))[y, x] != 0
    return ratio[~hold], ratio[hold], ratio[hold & near]


def calibrate(xyz, cameras, depths, confidence, mask_dir, size, stride, baseline):
    indices = sorted(set(range(0, len(cameras), stride)) | {len(cameras) - 1})
    samples = []
    for i in indices:
        path = mask_dir / (cameras[i]["frame_name"].rsplit(".", 1)[0] + ".png")
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        train, test, near = depth_samples(xyz, cameras[i], depths[i],
            None if confidence is None else confidence[i], mask, size)
        if len(train) >= 100 and len(test) >= 50:
            samples.append((i, float(np.median(train)), test, near))
    if not samples:
        return {"method": "first_frame_fallback", "scale": baseline, "keyframes": []}, np.full(len(cameras), baseline)
    ids = np.array([s[0] for s in samples])
    logs = np.array([s[1] for s in samples])
    global_log = np.median(logs)
    def errors(pred):
        return np.array([np.median(abs(s[2] - v)) for s, v in zip(samples, pred)])
    constant = np.full(len(ids), global_log)
    # Predict each interior key without fitting it. Only accept temporal variation
    # if it predicts unseen keyframes as well as unseen image tiles.
    smooth = np.array([np.median(logs[max(0,j-1):j+2]) for j in range(len(logs))])
    dynamic = False
    if len(ids) >= 4:
        loo = [np.interp(ids[j], np.delete(ids,j), np.delete(logs,j)) for j in range(len(ids))]
        static_err = errors(constant)[1:-1]
        dynamic_err = errors(loo)[1:-1]
        dynamic = (np.median(static_err - dynamic_err) > 0.003
                   and np.mean(dynamic_err < static_err) >= 0.65
                   and np.median(errors(smooth)) < np.median(errors(constant)))
    selected = smooth if dynamic else constant
    report = {"method": "omega_keyframes_log_scale", "temporal": bool(dynamic),
              "scale": float(np.exp(global_log)), "keyframes": []}
    for (i, log, test, near), selected_log in zip(samples, selected):
        report["keyframes"].append({"frame_index": i, "estimated_scale": float(np.exp(log)),
            "applied_scale": float(np.exp(selected_log)), "test_pixels": len(test),
            "baseline_median_abs_log_error": float(np.median(abs(test-np.log(baseline)))),
            "median_abs_log_error": float(np.median(abs(test-selected_log))),
            "near_person_static_median_log_residual": float(np.median(near-selected_log)) if len(near) else None})
    return report, np.exp(np.interp(np.arange(len(cameras)), ids, selected))
