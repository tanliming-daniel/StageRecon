"""CPU-only PnP focal search with boundary expansion and local refinement."""
import cv2
import numpy as np


def pnp_with_focal_search(xyz, xy, image_size, focal_values, reprojection_error,
                          *, adaptive=True):
    width, height = image_size
    grid = sorted(set(float(f) for f in focal_values))
    if not grid or any(not np.isfinite(f) or f <= 0 for f in grid):
        raise ValueError('Focal candidates must be finite and positive')
    xyz, xy = np.asarray(xyz, np.float64), np.asarray(xy, np.float64)
    best = None
    tried = set()
    solutions = []

    def evaluate(focal):
        nonlocal best
        if focal in tried:
            return
        tried.add(focal)
        k = np.array([[focal, 0., width/2], [0., focal, height/2], [0., 0., 1.]])
        ok, rvec, tvec, ids = cv2.solvePnPRansac(
            xyz, xy, k, None, iterationsCount=5000,
            reprojectionError=reprojection_error, confidence=.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or ids is None or len(ids) < 6:
            return
        ids = ids.reshape(-1)
        rvec, tvec = cv2.solvePnPRefineLM(xyz[ids], xy[ids], k, None, rvec, tvec)
        # Recompute consensus after refinement; stale RANSAC inliers can
        # otherwise favor an incorrect focal with a handful of extra matches.
        projected = cv2.projectPoints(xyz, rvec, tvec, k, None)[0].reshape(-1, 2)
        error = np.linalg.norm(projected - xy, axis=1)
        rotation = cv2.Rodrigues(rvec)[0]
        positive_depth = (xyz @ rotation.T + tvec.reshape(3))[:, 2] > 0
        ids = np.flatnonzero((error <= reprojection_error) & positive_depth)
        if len(ids) < 6:
            return
        rmse = float(np.sqrt(np.mean(error[ids] ** 2)))
        solutions.append((focal, len(ids), rmse))
        key = (len(ids), -rmse)
        if best is None or key > best[0]:
            best = (key, {'focal': focal, 'intrinsics': k.tolist(),
                         'rotation_vector': rvec.reshape(-1).tolist(),
                         'translation': tvec.reshape(-1).tolist(),
                         'inlier_count': len(ids), 'inlier_rmse_pixels': rmse}, ids)

    for focal in grid:
        evaluate(focal)
    boundary = best is None or best[1]['focal'] in (grid[0], grid[-1])
    if adaptive and len(grid) > 1:
        # Search wider even when the wrong grid has an interior local optimum.
        # Forward translation can compensate for focal error on a small patch.
        for factor in (0.5, 0.65, 2.5, 3.0, 3.6, 4.3, 5.2, 6.2, 7.5, 9.0):
            evaluate(factor * max(image_size))
    if adaptive and len(grid) > 1 and boundary:
        # A narrow default grid previously ended at 2.1 * image width, so
        # telephoto sitcom shots silently acquired a wrong translation.
        low, high = min(tried), max(tried)
        for _ in range(6):
            edge = None if best is None else best[1]['focal']
            if edge is None or edge == high:
                new_high = min(high * 1.4, 16. * max(image_size))
                if new_high == high:
                    break
                evaluate(new_high)
                high = new_high
            elif edge == low:
                new_low = max(low / 1.4, .15 * max(image_size))
                if new_low == low:
                    break
                evaluate(new_low)
                low = new_low
            else:
                break
    if best is None:
        raise RuntimeError('PnP failed for every focal-length candidate')
    if adaptive and len(grid) > 1:
        # Refine focal as well as pose; solvePnPRefineLM alone keeps K fixed.
        for fraction in (.15, .04, .01):
            center = best[1]['focal']
            for focal in np.linspace(center*(1-fraction), center*(1+fraction), 9):
                if .15 * max(image_size) <= focal <= 16 * max(image_size):
                    evaluate(float(focal))
    result = best[1]
    equivalent = [f for f, n, error in solutions
                  if n >= .99 * result['inlier_count']
                  and error <= result['inlier_rmse_pixels'] + max(.15, .1 * result['inlier_rmse_pixels'])]
    focal_interval = [min(equivalent), max(equivalent)]
    ambiguous = focal_interval[1] / focal_interval[0] > 1.5
    support = xy[best[2]].astype(np.float32)
    span = np.ptp(support, axis=0)
    result['inlier_spatial_support'] = {
        'hull_fraction': float(cv2.contourArea(cv2.convexHull(support)) / (width * height)),
        'width_fraction': float(span[0] / width),
        'height_fraction': float(span[1] / height),
    }
    result['focal_search'] = {
        'initial_min': grid[0], 'initial_max': grid[-1],
        'initial_boundary_hit': boundary, 'adaptive': adaptive,
        'evaluated_candidates': len(tried),
        'equivalent_focal_interval': focal_interval,
        'focal_ambiguous': ambiguous,
        'at_search_boundary': len(grid) > 1 and result['focal'] in (min(tried), max(tried)),
    }
    if adaptive and ambiguous:
        raise RuntimeError(f'PnP focal is underconstrained: equivalent focal interval {focal_interval}')
    if adaptive and result['focal_search']['at_search_boundary']:
        raise RuntimeError('PnP focal remains at search boundary; localization is unreliable')
    return result, best[2]
