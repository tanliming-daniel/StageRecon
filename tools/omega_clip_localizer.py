"""Person-masked endpoint localization in the refined VGGT-Omega reconstruction."""
from collections import Counter
from pathlib import Path
import json

import cv2
import numpy as np


OMEGA_WORLD = "omega_ba_world"
WIDTH, HEIGHT = 768, 432
FOCAL_MULTIPLIERS = (.65, .85, 1.10, 1.40, 1.80, 2.35, 3.05, 4.0, 5.3, 7.0)


class OmegaPnPLocalizer:
    def __init__(self, reconstruction, images):
        self.reconstruction = Path(reconstruction).resolve()
        self.sift = cv2.SIFT_create(nfeatures=1400)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.reference_images = Path(images)
        self.flow_reference_cache = {}
        self.references = []
        cv2.setNumThreads(4)
        with np.load(self.reconstruction / "predictions.npz", allow_pickle=False) as data:
            for name, depth, valid, k, extrinsic in zip(
                data["names"], data["depth"], data["static_mask"], data["intrinsics"], data["extrinsics"]
            ):
                image = cv2.imread(str(images / str(name)), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise FileNotFoundError(images / str(name))
                height, width = depth.shape
                static = cv2.resize(valid.astype(np.uint8)*255, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
                pixels, descriptors = self.features(image, static)
                scale = np.array([width / WIDTH, height / HEIGHT])
                native = (pixels + .5) * scale - .5
                x, y = np.clip(np.rint(native).astype(int), 0, [width-1, height-1]).T
                z = depth[y, x]
                keep = valid[y, x] & np.isfinite(z) & (z > 0)
                if descriptors is None or keep.sum() < 16:
                    continue
                rays = np.column_stack((native[keep], np.ones(keep.sum()))) @ np.linalg.inv(k).T
                xyz = (rays * z[keep, None] - extrinsic[:, 3]) @ extrinsic[:, :3]
                self.references.append((str(name), pixels[keep], descriptors[keep], xyz))
        if len(self.references) < 2:
            raise ValueError("Omega scene has fewer than two usable static references")

    def features(self, gray, static):
        gray = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        static = cv2.erode(static, np.ones((5, 5), np.uint8))
        points, features = self.sift.detectAndCompute(gray, static)
        return np.asarray([p.pt for p in points], np.float32).reshape(-1, 2), features

    def match(self, pixels, descriptors):
        candidates = []
        for name, ref_xy, ref_desc, xyz in self.references:
            pairs = [(a.queryIdx, a.trainIdx) for a, b in self.matcher.knnMatch(descriptors, ref_desc, k=2)
                     if a.distance < .78*b.distance]
            if len(pairs) >= 16:
                candidates.append((len(pairs), name, ref_xy, ref_desc, xyz, pairs))
        observations = []
        for _, name, ref_xy, ref_desc, xyz, pairs in sorted(candidates, key=lambda item: -item[0])[:24]:
            reverse = {a.queryIdx: a.trainIdx for a, b in self.matcher.knnMatch(ref_desc, descriptors, k=2)
                       if a.distance < .78*b.distance}
            mutual = [(q, r) for q, r in pairs if reverse.get(r) == q]
            if len(mutual) < 16:
                continue
            qid, rid = np.asarray(mutual).T
            try:
                _, good = cv2.findFundamentalMat(pixels[qid], ref_xy[rid], cv2.USAC_MAGSAC, 1.6, .999, 4000)
            except cv2.error:
                good = None
            if good is None or good.sum() < 16:
                _, good = cv2.findHomography(pixels[qid], ref_xy[rid], cv2.USAC_MAGSAC, 2.0)
            if good is not None and good.sum() >= 16:
                keep = good.ravel() != 0
                observations.append((name, qid[keep], xyz[rid[keep]]))
        return sorted(observations, key=lambda row: -len(row[1]))

    @staticmethod
    def solve(xyz, pixels, references):
        ids = np.random.default_rng(52).permutation(len(pixels))
        fit, test = np.split(ids, [int(.70*len(ids))])
        best = None
        for multiplier in FOCAL_MULTIPLIERS:
            focal = multiplier * WIDTH
            k = np.array([[focal, 0, (WIDTH-1)/2], [0, focal, (HEIGHT-1)/2], [0, 0, 1]])
            success, r, t, inliers = cv2.solvePnPRansac(
                xyz[fit], pixels[fit], k, None, iterationsCount=450, reprojectionError=4,
                confidence=.999, flags=cv2.SOLVEPNP_EPNP,
            )
            if not success or inliers is None or len(inliers) < 12:
                continue
            idx = fit[inliers.ravel()]
            r, t = cv2.solvePnPRefineLM(xyz[idx], pixels[idx], k, None, r, t)
            fit_error = np.linalg.norm(cv2.projectPoints(xyz[idx],r,t,k,None)[0][:,0]-pixels[idx], axis=1)
            score = (len(idx), -float(np.median(fit_error)))
            if best is None or score > best[0]:
                best = score, r, t, k, idx
        if best is None:
            return None
        _, r, t, k, idx = best
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT |
                 cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST |
                 cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
                 cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
        for _ in range(2):
            try:
                _, refined_k, _, rotations, translations = cv2.calibrateCamera(
                    [xyz[idx].astype(np.float32)], [pixels[idx].astype(np.float32)],
                    (WIDTH, HEIGHT), k.copy(), np.zeros(5), flags=flags,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-8),
                )
            except cv2.error:
                return None
            k, r, t = refined_k, rotations[0], translations[0]
            if not np.isfinite(k).all() or not (.4 * WIDTH <= k[0,0] <= 10 * WIDTH):
                return None
            training_error = np.linalg.norm(cv2.projectPoints(xyz[fit],r,t,k,None)[0][:,0]-pixels[fit],axis=1)
            idx = fit[training_error <= 4]
            if len(idx) < 12:
                return None
        rotation = cv2.Rodrigues(r)[0]
        projected = cv2.projectPoints(xyz,r,t,k,None)[0][:,0]
        error = np.linalg.norm(projected - pixels, axis=1)
        positive = (xyz @ rotation[2] + t[2,0]) > 0
        supported = positive & (error <= 4)
        test_median = float(np.median(np.where(positive[test], error[test], np.inf)))
        test_fraction = float(supported[test].mean())
        ref_support = Counter(references[supported])
        if test_median > 4 or test_fraction < .60 or supported.sum() < 30:
            return None
        feature_area = cv2.contourArea(cv2.convexHull(pixels[supported].astype(np.float32))) / (WIDTH*HEIGHT)
        if feature_area < .025:
            return None
        return rotation, t.ravel(), k, {
            "passed": True, "matched_points": len(pixels), "inlier_count": int(supported.sum()),
            "test_count": len(test), "test_median_pixels_at_768": test_median,
            "test_fraction_within_4_pixels_at_768": test_fraction,
            "static_feature_area_fraction": feature_area, "reference_inliers": dict(ref_support),
            "focal_refinement": "training correspondences only",
            "inlier_rmse_at_768": float(np.sqrt(np.mean(error[supported]**2))),
        }

    def flow_recovery(self, gray, static, choices, video, frame_index):
        """Track static corners from a visually verified Omega reference."""
        import re
        query = cv2.resize(gray,(WIDTH,HEIGHT),interpolation=cv2.INTER_AREA)
        key = re.search(r"_f(\d+)-f",Path(video).stem)
        episode = next((v for v in Path(video).parts if re.fullmatch(r"S\d+E\d+",v)),"")
        ranked = [row[0] for row in choices[:6]]
        if key and episode:
            target = int(key[1])+frame_index
            nearby = sorted((abs(int(m[1])-target),name) for name,*_ in self.references
                            if episode in name and (m:=re.search(r"_(\d+)\.jpg",name)))
            ranked.extend(name for dist,name in nearby[:3] if dist<=160)
        for name in dict.fromkeys(ranked):
            if name not in self.flow_reference_cache:
                with np.load(self.reconstruction/"predictions.npz",allow_pickle=False) as data:
                    ix = list(map(str,data["names"])).index(name)
                    depth=data["depth"][ix]; valid=data["static_mask"][ix]
                    k=data["intrinsics"][ix]; ext=data["extrinsics"][ix]
                ref=cv2.resize(cv2.imread(str(self.reference_images/name),0),(WIDTH,HEIGHT),interpolation=cv2.INTER_AREA)
                mask=cv2.erode(cv2.resize(valid.astype(np.uint8),(WIDTH,HEIGHT),interpolation=cv2.INTER_NEAREST),np.ones((5,5),np.uint8))
                corners=cv2.goodFeaturesToTrack(ref,1800,.006,5,mask=mask)
                if corners is None:continue
                pts=corners[:,0]; h,w=depth.shape; native=(pts+.5)*[w/WIDTH,h/HEIGHT]-.5
                x,y=np.clip(np.rint(native).astype(int),0,[w-1,h-1]).T; z=depth[y,x]
                ok=valid[y,x].astype(bool)&np.isfinite(z)&(z>0)
                rays=np.column_stack((native[ok],np.ones(ok.sum())))@np.linalg.inv(k).T
                xyz=(rays*z[ok,None]-ext[:,3])@ext[:,:3]
                self.flow_reference_cache[name]=(ref,pts[ok],xyz)
            ref,pts,xyz=self.flow_reference_cache[name]
            if len(pts)<60:continue
            opts=dict(winSize=(21,21),maxLevel=4,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,.01))
            uv,status,_=cv2.calcOpticalFlowPyrLK(ref,query,pts,None,**opts)
            back,reverse,_=cv2.calcOpticalFlowPyrLK(query,ref,uv,None,**opts)
            x,y=np.clip(np.rint(uv).astype(int),0,[WIDTH-1,HEIGHT-1]).T
            ok=status.ravel().astype(bool)&reverse.ravel().astype(bool)&(np.linalg.norm(pts-back,axis=1)<.65)&(static[y,x]>0)
            ok &= (uv[:,0]>=5)&(uv[:,0]<WIDTH-5)&(uv[:,1]>=5)&(uv[:,1]<HEIGHT-5)
            verified=[]
            for j in np.flatnonzero(ok):
                a=cv2.getRectSubPix(ref,(9,9),tuple(pts[j])).astype(float).ravel()
                b=cv2.getRectSubPix(query,(9,9),tuple(uv[j])).astype(float).ravel()
                a-=a.mean();b-=b.mean()
                if a@b>.88*max(1.,np.linalg.norm(a)*np.linalg.norm(b)):verified.append(j)
            if len(verified)<65:continue
            idx=np.array(verified); points=xyz[idx]
            if np.linalg.svd(points-points.mean(0),compute_uv=False)[-1] < .012*np.linalg.norm(points-points.mean(0)):continue
            result=self.solve(points,uv[idx].astype(float),np.repeat(name,len(idx)))
            if result is not None and result[3]["inlier_count"]>=60 and result[3]["static_feature_area_fraction"]>=.04:
                result[3]["recovery"]="omega_reference_bidirectional_patch_tracks"
                return result
        return None

    def endpoint_for_clip(self, image_path, mask_path, output, video, frame_index,
                          frame_count, anchor_source, reference_dirs):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Missing endpoint image or SAM3 mask: {image_path}, {mask_path}")
        h, w = image.shape
        static = (cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST) == 0).astype(np.uint8)*255
        pixels, descriptors = self.features(image, static)
        choices = [] if descriptors is None or len(pixels) < 30 else self.match(pixels, descriptors)
        attempted = []
        for count in (8, 16, 4, len(choices)):
            used, xyz, uv, references = set(), [], [], []
            for ref, ids, points in choices[:count]:
                for index, point in zip(ids, points):
                    if index in used:
                        continue
                    used.add(index)
                    xyz.append(point)
                    uv.append(pixels[index])
                    references.append(ref)
            attempted.append(len(uv))
            if len(uv) < 30 or len(set(references)) < 2:
                continue
            result = self.solve(np.asarray(xyz), np.asarray(uv, np.float64), np.asarray(references))
            if result is not None:
                break
        else:
            result = self.flow_recovery(image, static, choices, video, frame_index)
            if result is None:
                raise RuntimeError(f"Omega PnP failed static validation for {Path(video).name} frame {frame_index}: {attempted} matches")
        rotation, translation, k, quality = result
        scale = np.array([w/WIDTH, h/HEIGHT])
        k[:2, 2] = (k[:2, 2]+.5)*scale-.5
        k[0, 0] *= scale[0]
        k[1, 1] *= scale[1]
        payload = {
            "image": str(image_path), "coordinate_system": OMEGA_WORLD,
            "pose_source": "estimated_omega_ba_sift_pnp", "requested_anchor_source": "estimated",
            "quality": quality,
            "world_to_camera_rotation": rotation.tolist(), "world_to_camera_translation": translation.tolist(),
            "camera_center": (-rotation.T @ translation).tolist(),
            "pnp": {"intrinsics": k.tolist(), "focal": float(k[0,0]), "inlier_count": quality["inlier_count"],
                    "inlier_rmse_pixels": quality["inlier_rmse_at_768"]*float(np.mean(scale))},
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2)+"\n")
        return payload

    def close(self):
        pass
