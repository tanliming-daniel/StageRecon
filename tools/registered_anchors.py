"""Reuse registered COLMAP anchors after checking clip mapping and image identity."""
from pathlib import Path
import re
import cv2
import numpy as np


def source_image_name(video, frame_index, frame_count):
    video = Path(video)
    episode = re.search(r'S\d{2}E\d{2}', str(video))
    bounds = re.search(r'_f(\d+)-f(\d+)$', video.stem)
    if episode is None or bounds is None:
        raise ValueError('Source episode/frame range is absent from clip path')
    first, last = map(int, bounds.groups())
    if last - first + 1 != frame_count or not 0 <= frame_index < frame_count:
        raise ValueError('Decoded clip length disagrees with source frame range')
    return f'Friends_{episode.group()}_{first + frame_index:08d}.jpg'


def frame_identity(query, reference):
    if query is None or reference is None:
        return {'matches': False, 'reason': 'Reference/query image could not be read'}
    if abs(query.shape[1] / query.shape[0] - reference.shape[1] / reference.shape[0]) > .01:
        return {'matches': False, 'reason': 'Reference/query aspect ratio differs'}
    a = cv2.resize(cv2.cvtColor(query, cv2.COLOR_BGR2GRAY), (160, 90), interpolation=cv2.INTER_AREA).astype(float)
    b = cv2.resize(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), (160, 90), interpolation=cv2.INTER_AREA).astype(float)
    mae = float(np.mean(np.abs(a-b)) / 255.)
    correlation = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if min(a.std(), b.std()) > 1 else 0.
    return {'matches': bool(correlation >= .90 and mae <= .08),
            'grayscale_correlation': correlation, 'normalized_mae': mae}


def camera_intrinsics(camera, width, height):
    params = np.asarray(camera.params, dtype=float)
    if camera.model == 'SIMPLE_PINHOLE':
        fx, cx, cy = params; fy = fx
    elif camera.model == 'PINHOLE':
        fx, fy, cx, cy = params
    else:
        raise ValueError(f'Registered camera model {camera.model} requires distortion handling')
    if min(width, height, camera.width, camera.height) <= 0 or not np.isfinite(params).all() or min(fx, fy) <= 0:
        raise ValueError('Invalid registered camera dimensions or intrinsics')
    sx, sy = width / camera.width, height / camera.height
    if abs(sx/sy - 1) > .01:
        raise ValueError('Clip is cropped/stretched relative to registered camera')
    return np.array([[fx*sx, 0., cx*sx], [0., fy*sy, cy*sy], [0., 0., 1.]])


def intrinsics_between_anchors(first, last, focal, alpha):
    """Preserve calibrated principal point and pixel aspect during zoom."""
    first, last = np.asarray(first, dtype=float), np.asarray(last, dtype=float)
    if not np.isfinite(first).all() or not np.isfinite(last).all() or min(first[0,0],first[1,1],last[0,0],last[1,1],focal) <= 0:
        raise ValueError('Invalid anchor intrinsics')
    ratio = np.exp((1-alpha)*np.log(first[1,1]/first[0,0]) + alpha*np.log(last[1,1]/last[0,0]))
    center = (1-alpha)*first[:2,2] + alpha*last[:2,2]
    return np.array([[focal,0.,center[0]],[0.,focal*ratio,center[1]],[0.,0.,1.]])


def registered_anchor(video, frame_index, frame_count, query_path, cameras,
                      images_by_stem, points3d, reference_dirs):
    """Return (pose, reason). Missing/unsafe registry entries fall back to PnP."""
    try:
        name = source_image_name(video, frame_index, frame_count)
        image = images_by_stem.get(Path(name).stem)
        if image is None:
            return None, f'{name} is not registered'
        reference_path = next((Path(d)/name for d in reference_dirs if (Path(d)/name).is_file()), None)
        if reference_path is None:
            return None, f'No source image is available to verify {name}'
        query = cv2.imread(str(query_path)); reference = cv2.imread(str(reference_path))
        identity = frame_identity(query, reference)
        if not identity['matches']:
            return None, f'Source image identity check failed: {identity}'
        height, width = query.shape[:2]
        camera = cameras[image.camera_id]
        intrinsic = camera_intrinsics(camera, width, height)
        rotation = np.asarray(image.qvec2rotmat(), dtype=float)
        translation = np.asarray(image.tvec, dtype=float)
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all() or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < .999:
            return None, 'Registered camera transform is invalid'
        known = [i for i, pid in enumerate(image.point3D_ids) if int(pid) >= 0 and int(pid) in points3d]
        if len(known) < 6:
            return None, 'Registered camera has fewer than six scene observations'
        xyz = np.array([points3d[int(image.point3D_ids[i])].xyz for i in known], dtype=float)
        xy = np.asarray(image.xys, dtype=float)[known] * [width/camera.width, height/camera.height]
        camera_xyz = xyz @ rotation.T + translation
        positive = camera_xyz[:, 2] > 1e-5
        projected = camera_xyz @ intrinsic.T
        projected = projected[:, :2] / np.maximum(camera_xyz[:, 2:3], 1e-8)
        errors = np.linalg.norm(projected-xy, axis=1)
        support = positive & np.isfinite(errors) & (errors <= 8.)
        count = int(support.sum())
        rmse = float(np.sqrt(np.mean(errors[support] ** 2))) if count else float('inf')
        if count < 6 or rmse > 5:
            return None, f'Registered reprojection check failed: count={count}, rmse={rmse}'
        rvec = cv2.Rodrigues(rotation)[0].reshape(3)
        pose = {
            'query_image': str(Path(query_path).resolve()),
            'coordinate_system': 'raw_colmap_world (same as environment_dir/colmap/pc.ply)',
            'pose_source': 'registered_colmap',
            'requested_anchor_source': 'registered-first',
            'registration': {'image_name': name, 'image_id': int(image.id), 'camera_id': int(image.camera_id),
                             'reference_image': str(reference_path), 'source_frame_index': int(re.search(r'_(\d+)\.jpg$', name).group(1)),
                             'frame_count_validated': True, 'identity': identity,
                             'reprojection_validated': True, 'original_resolution': [camera.width, camera.height],
                             'query_resolution': [width, height]},
            'pnp': {'focal': float(intrinsic[0,0]), 'intrinsics': intrinsic.tolist(),
                    'rotation_vector': rvec.tolist(), 'translation': translation.tolist(),
                    'inlier_count': count, 'inlier_rmse_pixels': rmse},
            'world_to_camera_rotation': rotation.tolist(),
            'world_to_camera_translation': translation.tolist(),
            'camera_center': (-rotation.T @ translation).tolist(),
            'inlier_source_images': {name: count},
        }
        return pose, None
    except (ValueError, KeyError, OSError, cv2.error) as error:
        return None, f'{type(error).__name__}: {error}'


def rotation_zoom_anchor(reference, image, points, frames, mask_dir):
    """Track registered static landmarks through a shot and test a tripod pan/zoom."""

    def gray(path):
        value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if value is None:
            raise FileNotFoundError(path)
        return value

    prev = gray(frames[0])
    h, w = prev.shape
    source_size = reference["registration"]["original_resolution"]
    known = image.point3D_ids >= 0
    xy = (image.xys[known] * (np.array([w,h]) / source_size)).astype(np.float32)
    xyz = np.array([points[k].xyz for k in image.point3D_ids[known]])

    def static(frame, pixels):
        mask = gray(mask_dir / frame.with_suffix(".png").name)
        if mask.shape != (h,w):
            raise ValueError("Unexpected mask resolution")
        x,y = np.rint(pixels).astype(int).T
        return ((x>=0) & (x<w) & (y>=0) & (y<h)
                & (mask[y.clip(0,h-1), x.clip(0,w-1)]==0))

    # Independently test the pure-rotation image warp outside the map points.
    # These background corners have rays but no depth and never enter the fit.
    k = np.asarray(reference["pnp"]["intrinsics"])
    base = np.asarray(reference["world_to_camera_rotation"])
    center = np.asarray(reference["camera_center"])
    allowed = (gray(mask_dir / frames[0].with_suffix(".png").name)==0).astype(np.uint8)
    for pt in np.rint(xy).astype(int):
        cv2.circle(allowed,tuple(pt),8,0,-1)
    corners = cv2.goodFeaturesToTrack(prev,800,.005,12,mask=allowed)
    mapped = np.ones(len(xy), dtype=bool)
    if corners is not None:
        extra = corners.reshape(-1,2)
        rays = np.column_stack((extra,np.ones(len(extra)))) @ np.linalg.inv(k).T
        xyz = np.vstack((xyz,center + rays @ base))
        xy = np.vstack((xy,extra))
        mapped = np.r_[mapped,np.zeros(len(extra),dtype=bool)]
    keep = static(frames[0], xy)
    xy, xyz, mapped = xy[keep], xyz[keep], mapped[keep]
    for frame in frames[1:]:
        if len(xy) < 55:
            return None
        nxt = gray(frame)
        tracked, ok, _ = cv2.calcOpticalFlowPyrLK(prev,nxt,xy,None)
        back, reverse_ok, _ = cv2.calcOpticalFlowPyrLK(nxt,prev,tracked,None)
        keep = (ok.ravel().astype(bool) & reverse_ok.ravel().astype(bool)
                & (np.linalg.norm(back-xy,axis=1)<.7) & static(frame,tracked))
        xy, xyz, mapped, prev = tracked[keep], xyz[keep], mapped[keep], nxt
    if len(xy) < 55:
        return None
    center = np.asarray(reference["camera_center"])
    base = np.asarray(reference["world_to_camera_rotation"])
    k = np.asarray(reference["pnp"]["intrinsics"])
    if mapped.sum()<55:
        return None
    train,test = np.split(np.random.default_rng(52).permutation(np.flatnonzero(mapped)), [int(.7*mapped.sum())])

    def errors(v, idx):
        r = cv2.Rodrigues(v[:3])[0] @ base
        cam = (xyz[idx]-center) @ r.T
        pixels = cam[:,:2] / np.maximum(cam[:,2:3],1.e-8) * k.diagonal()[:2]*np.exp(v[3]) + k[:2,2]
        return (pixels-xy[idx]).ravel()

    # Four parameters, with soft-L1 weights in pixel space. Use OpenCV and
    # NumPy since the SAM3 worker does not depend on SciPy.
    v = np.zeros(4)
    steps = np.eye(4)*1.e-6
    for iteration in range(30):
        residual = errors(v,train)
        jac = np.column_stack([(errors(v+step,train)-residual)/1.e-6 for step in steps])
        weights = (1+(residual/2.)**2)**(-.25)
        delta, _, rank, _ = np.linalg.lstsq(jac*weights[:,None], -residual*weights, rcond=None)
        if rank != 4 or not np.isfinite(delta).all():
            return None
        v += delta
        if abs(v[3])>=np.log(3.) or np.linalg.norm(v[:3])>np.pi:
            return None
        if np.linalg.norm(delta)<1.e-7:
            break
    else:
        return None
    held = np.linalg.norm(errors(v,test).reshape(-1,2),axis=1)*768/w
    residuals = np.linalg.norm(errors(v,np.arange(len(xy))).reshape(-1,2),axis=1)
    inliers = (residuals<=4) & mapped
    warp_test = np.flatnonzero(~mapped)
    if len(warp_test)<16:
        return None
    warp_errors = residuals[warp_test]*768/w
    if np.quantile(warp_errors,.9)>4:
        return None
    supported = warp_test[warp_errors<=4]
    area = cv2.contourArea(cv2.convexHull(xy[supported]))/(w*h)
    if np.median(held)>2 or np.quantile(held,.9)>4 or area<.05 or sum(inliers)<40:
        return None
    r = cv2.Rodrigues(v[:3])[0] @ base
    k = k.copy()
    k[0,0] *= np.exp(v[3])
    k[1,1] *= np.exp(v[3])
    source = reference["registration"]["image_name"]
    return dict(query_image=str(frames[-1]), coordinate_system=reference["coordinate_system"],
                pose_source="registered_static_flow_rotation_zoom",requested_anchor_source="registered-first",
                world_to_camera_rotation=r.tolist(), world_to_camera_translation=(-r@center).tolist(),
                camera_center=center.tolist(), inlier_source_images={source:int(sum(inliers))},
                flow_validation=dict(reference=source,test_count=len(test),median_at_768=float(np.median(held)),
                                     p90_at_768=float(np.quantile(held,.9)),test_area=area, warp_test_count=len(warp_test),
                                     warp_p90_at_768=float(np.quantile(warp_errors,.9))),
                pnp=dict(intrinsics=k.tolist(),focal=float(k[0,0]),rotation_vector=cv2.Rodrigues(r)[0].ravel().tolist(),
                         translation=(-r@center).tolist(),inlier_count=int(sum(inliers)),
                         inlier_rmse_pixels=float(np.sqrt(np.mean(residuals[inliers]**2))),
                         inlier_spatial_support=dict(hull_fraction=cv2.contourArea(cv2.convexHull(xy[inliers]))/(w*h))))
