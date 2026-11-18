# Camera localization audit: S08E04 general_0196

The frame identifiers are Friends_S08E04_00032018.jpg and
Friends_S08E04_00032041.jpg. Both exist in the original COLMAP reconstruction.
Their existing poses are used only for this audit; production localization
still matches query features and does not read the query's registered pose.

At the render-bundle resolution of 1280x720:

| Endpoint | Saved focal | Registered focal | Camera-centre discrepancy |
| --- | ---: | ---: | ---: |
| First | 2688 | 5872.2143 | 4.4451 world units |
| Last | 2688 | 5885.3497 | 4.4482 world units |

The first/last rotation discrepancies are 3.8722 and 4.1287 degrees. The old
maximum focal was exactly 2.1 times image width: the solution hit the fixed
search-grid ceiling. A low inlier RMSE did not establish a correct pose.

## Propagation

SAM3 person mask -> DISK matching to scene tracks -> endpoint PnP/focal search
-> DA3 relative motion aligned to those endpoints -> scene-depth/DA3-depth
scale fit using the first PnP camera -> actor depth and world bbox -> Blender.

The PnP grid excluded this telephoto shot's true focal. Translation could
compensate on a limited background patch. The 35-inlier first endpoint was
classified `review` but was accepted by the clip runner; `review` coarse poses
also bypassed the global retrieval fallback. DA3 then fitted scale against the
wrong projected scene and forced its trajectory to the wrong endpoints.
The endpoint interpolation is not an independent check on endpoint accuracy.

## Changes

- A shared CPU PnP helper searches wider focal values, including telephoto
  ranges, even when the narrow grid has an interior local optimum. It refines
  the focal locally, rechecks consensus and positive depth, rejects underconstrained focal ranges, records search
  diagnostics, and rejects unresolved boundary solutions. Explicit focal
  values remain configurable; a single fixed focal remains fixed.
- Both CLI and persistent stage-1 worker use that helper. Coarse `review`
  results now trigger global retrieval rather than bypassing it.
- Endpoint quality is shared across runners. `review` is not permitted to
  anchor DA3 by default. `--allow-review-anchors` explicitly restores that
  behavior when an operator has reviewed the estimate.
- The clip runner's resumed endpoint cache depends on the localization and
  quality-check source files, so updating the solver invalidates old anchors.
- DA3 checks both endpoint qualities before inference/depth-scale fitting.

A solver-only regression on 303 registered correspondences from the first
frame recovered focal 5872.27608, within 0.0011% of the registered focal, and
camera-centre error 0.00004743 world units. This validates the focal-search fix
on known correspondences. It is not a rerun of DISK retrieval, SAM3 or DA3.
Synthetic telephoto/fixed-focal tests and endpoint-quality tests are included.

The original frames contain prominent credits over actors and background.
Text overlays and limited static feature support remain a matching-quality
concern; neither a low RMSE nor a successful render is sufficient alone.

## Existing exported data

Do not reuse the old `actor_tracks.json` after correcting the camera. Recompute
scene depth scale, actor depth/world positions, bbox metadata, render job
checksums and the imported animation. Cached raw DA3 predictions can be reused
if they belong to the same unmodified clip; the world alignment and scale must
be recomputed. Portable bundles in this checkout omit the raw DA3 predictions,
SAM3 masks and original clip video, so they cannot supply those stages alone.
Batch exports already considered complete also need a targeted rerun with
`--overwrite`; source changes do not rewrite existing exported bundles.

Original source images used for review are available locally under:
`/Users/liming/Documents/project/sitcoms3D/data/Friends-monica_apartment/images`.

## Multi-case validation (2026-11-12)

Audited 2,065 exported clips against available registered endpoints: 3,786
endpoints matched source frame names. 2,250 of those hit the old 2.1-width
focal ceiling. The median saved-pose centre discrepancy was 2.5852 raw COLMAP
units (90th percentile 5.9468). These are comparisons to existing COLMAP poses,
not independent metric ground truth.

Selected 13 distinct first-frame cases across the registered focal/width
range 0.36–9.78, spanning several episodes. Each solver received the same
registered 2D–3D correspondences; no query pose was supplied to the solver.
The final wide-search solver recovered 11 cases and rejected 2 as focal-
underconstrained. This is preferable to exporting a seemingly precise pose
for the far-telephoto and low-depth-variation cases. The equivalent-focal
interval is a search diagnostic, not a statistically calibrated confidence
interval.

Added 26 stress variants: 1px coordinate noise plus 30% random correspondence
outliers, and a 60-feature localized background patch. Final quality counts
were 11 accepted, 3 requiring review, and 12 rejected. No accepted result in
this sample had centre discrepancy above 0.5 world units. This limited stress
test does not validate actual DISK retrieval or eliminate systematic false
matches/text-overlay contamination.

A small-patch stress case initially had 59 inliers and 0.725 world-unit centre
error while passing the count/RMSE gate. The shared gate now flags inlier
convex-hull coverage below 5% of image area for review. The threshold is a
conservative heuristic; use explicit operator review for legitimate narrow-
support cases. A frontoparallel-plane regression also checks that focal/depth
ambiguity is rejected instead of returning arbitrary translation.

28 local unit tests passed with numpy/OpenCV available. The remote system
Python lacks OpenCV, so its complete unit suite cannot run in that runtime;
Blender imports, 48 actor-anchor checks, and GPU renders were validated there.

Six additional camera-only renders used registered cameras as controls. They
show that individual scene assets, particularly the sofa lamp in the S08E04
case, still differ in placement from source images. The scene's `pointcloud`
alignment object is an Empty with a manually stored transform; it contains no
mesh/children with which to independently verify raw-point alignment. These
renders therefore test the scene/import side, not production pose recovery.
Do not adjust a correct camera solely to compensate for a misplaced asset.
The geometry and pointcloud-to-scene alignment require separate validation.
