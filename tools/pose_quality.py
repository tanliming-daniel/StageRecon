"""Shared endpoint quality gates, independent of ML/Blender dependencies."""

def assess_pose_quality(pose):
    pnp = pose['pnp']
    inliers = int(pnp['inlier_count'])
    rmse = float(pnp['inlier_rmse_pixels'])
    sources = pose.get('inlier_source_images', {})
    concentration = max(sources.values(), default=0) / max(inliers, 1)
    coverage = pnp.get('inlier_spatial_support', {}).get('hull_fraction')
    reasons = []
    if pose.get('pose_source') == 'registered_colmap':
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
