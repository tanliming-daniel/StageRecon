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
