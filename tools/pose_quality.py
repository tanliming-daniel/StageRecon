"""Shared endpoint quality gates, independent of ML/Blender dependencies."""
import math

def assess_pose_quality(pose):
    pnp = pose['pnp']
    inliers = int(pnp['inlier_count'])
    rmse = float(pnp['inlier_rmse_pixels'])
    sources = pose.get('inlier_source_images', {})
    concentration = max(sources.values(), default=0) / max(inliers, 1)
    coverage = pnp.get('inlier_spatial_support', {}).get('hull_fraction')
    reasons = []
    if pose.get("pose_source") in ("estimated_omega_ba_sift_pnp", "estimated_omega_ba_disk_pnp"):
        quality = pose.get("quality", {})
        metrics = [float(quality.get(key, float("nan"))) for key in (
            "test_median_pixels_at_768", "test_fraction_within_4_pixels_at_768",
            "static_feature_area_fraction", "inlier_rmse_at_768")]
        median, fraction, area, normalized_rmse = metrics
        verified = (
            pose.get("coordinate_system") == "omega_ba_world"
            and quality.get("passed") is True
            and all(math.isfinite(value) for value in metrics)
            and inliers >= 30 and quality.get("inlier_count") == inliers
            and quality.get("test_count", 0) >= 8
            and 0 <= median <= 4 and .60 <= fraction <= 1
            and .025 <= area <= 1 and 0 <= normalized_rmse <= 4
        )
        status = "accepted" if verified else "rejected"
        coverage = area
        sources = quality.get("reference_inliers", {})
        concentration = max(sources.values(), default=0) / max(inliers, 1)
        if not verified:
            reasons.append("Omega endpoint failed normalized held-out reprojection and coverage checks.")
    elif pose.get("pose_source") == "registered_static_flow_rotation_zoom":
        held = pose.get("flow_validation", {})
        accepted = (inliers >= 40 and 0 <= rmse <= 4
                    and held.get("test_count", 0) >= 16
                    and 0 <= held.get("median_at_768", float("inf")) <= 2
                    and 0 <= held.get("p90_at_768", float("inf")) <= 4
                    and .05 <= held.get("test_area", 0) <= 1
                    and held.get("reference") in sources)
        status = "accepted" if accepted else "rejected"
        if not accepted:
            reasons.append("Tracked static landmarks failed independent pan/zoom validation.")
    elif pose.get('pose_source') == 'registered_colmap':
        registration = pose.get('registration', {})
        verified = (registration.get('frame_count_validated')
                    and registration.get('identity', {}).get('matches')
                    and registration.get('reprojection_validated'))
        status = 'accepted' if verified and inliers >= 6 and rmse <= 5 else 'rejected'
        if status != 'accepted':
            reasons.append('Registered anchor failed mapping, identity, or reprojection validation.')
    elif pnp.get('focal_search', {}).get('focal_ambiguous'):
        status = 'rejected'
        reasons.append('Focal is underconstrained by the matched scene geometry.')
    elif pnp.get('focal_search', {}).get('at_search_boundary'):
        status = 'rejected'
        reasons.append('Focal estimate remains at the search boundary.')
    elif inliers >= 20 and rmse <= 5 and coverage is not None and coverage < .05:
        status = 'review'
        reasons.append('Inliers cover less than 5% of the image; a small patch can hide focal/depth ambiguity.')
    elif inliers >= 40 and rmse <= 4.5:
        status = 'accepted'
    elif inliers >= 20 and rmse <= 5.0:
        status = 'review'
        reasons.append('Endpoint requires review before camera/depth reconstruction.')
    else:
        status = 'rejected'
        reasons.append('Too few PnP inliers or excessive reprojection error.')
    return {'status': status, 'dataset_eligible': status == 'accepted',
            'pnp_inlier_count': inliers, 'pnp_rmse_pixels': rmse,
            'dominant_source_inlier_fraction': round(concentration, 4),
            'inlier_hull_fraction': coverage, 'reasons': reasons}


def require_usable_anchor(pose, *, allow_review=False):
    quality = assess_pose_quality(pose)
    allowed = ('accepted', 'review') if allow_review else ('accepted',)
    if quality['status'] not in allowed:
        raise ValueError(f'Unsafe camera anchor: {quality}')
    return quality
