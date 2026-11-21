# Production camera anchors and fixed actor proxies

Production batch jobs now default to `--anchor-source registered-first`.
`--anchor-source estimated` retains independent DISK/PnP localization for
new-image evaluation. These modes must be reported separately: reusing an
existing registered pose is not an independent localization result.

For each endpoint, production checks:

1. The episode/source frame interval can be parsed from the clip filename.
2. The decoded clip count matches that inclusive interval.
3. The registered source image exists and visually matches the decoded frame.
4. Resolution/aspect ratio and pinhole intrinsics can be converted safely.
5. The registered pose passes a scene-point reprojection check.

If any check fails, the endpoint falls back to the independent estimator and
records the lookup/fallback reason. Estimated anchors must pass focal
ambiguity, inlier count, reprojection error and spatial coverage gates before
DA3 can use them. Review anchors require explicit operator opt-in in the
single-clip tools. Existing calibration is accepted only with its mapping,
identity and reprojection validation; a source label alone is insufficient.

`pose_source`, `registration`, the requested policy, and absolute endpoint
provenance distinguish reused calibration from estimated poses. Intrinsic
scaling preserves principal point and pixel aspect through the camera track.
After an anchor changes, DA3-to-world scale and actor trajectories are rebuilt.
Raw DA3 predictions may be reused for an unchanged clip; old world-space
actor boxes cannot be reused just by replacing the camera JSON.

## H100 usage

The batch runner defaults both model stages to physical GPU 4. An explicit
production command, from `/root/userdata/300workspace/StageRecon`, is:

```bash
/root/userdata/300workspace/sm3r/bin/python tools/run_clip_render_data_batch.py \
  --anchor-source registered-first \
  --stage1-gpu 4 --stage2-gpu 4 \
  --output-root outputs/Friends-monica_apartment/clip_render_data_s08_registered_v2
```

Use `--include general_0196_global_0350 --limit 1` for a targeted pilot.
Use a new output root to retain existing datasets. Completed old bundles are
not retroactively recalibrated by changing the source policy; `--overwrite`
is needed when deliberately rebuilding the same output location.

The runner first looks for reference images in the scene environment, then
`data/sparse_reconstruction_and_image_data/Friends-monica_apartment/images`.
`--registered-images-dir PATH` supplies additional reference locations.
If source images are absent, pose reuse is withheld and localization falls
back to estimation instead of trusting the filename alone.

## Actor geometry

Each actor uses one height/radius for the complete clip. Bbox changes and
posture labels do not animate mesh size. The exported `proxy_bbox_3d`
uses that same fixed size, centre and actor/scene-up axes. The legacy `bbox_3d`
retains observed visible extent and can vary with occlusion; use
`proxy_bbox_3d` when fixed actor size is required. The new box is a coarse
proxy, not an anatomical body fit. The default `head-cylinder` preview
has a sphere head and a cream horizontal band on its front (+local X).
The direction is a motion/camera-facing prior, not measured anatomical head
pose. The Friends scene configuration fixes up using the inverse Blender alignment
rotation applied to scene +Z. The batch/single-clip runners load it only when
scene and base-blend names match, and pass it through to actor estimation.
`--actor-world-up X Y Z` overrides it; the direct actor estimator accepts
`--world-up X Y Z`. Other scenes without matching configuration retain an
explicit camera-up prior. Update the scene configuration if the alignment
rotation changes. This corrects the roughly 35-degree camera-up/scene-up
mismatch measured in the wide case while preserving head rays and fixed size.

Tiny interior tracks with BOTH a median visible-height/proxy-height ratio
below 0.15 and image bbox fraction below 0.003 are flagged when fewer than
25% of observations touch the image boundary. The track remains in JSON with
`proxy_quality.eligible=false` and a review reason. Blender skips its proxy.
This whole-track heuristic prevents small photo detections from becoming
full-height actors; it is not a semantic photo/person classifier. Large
close-ups, boundary-truncated people, and small distant full-body detections
are protected by the combined checks. Review the flags for unusual scenes.

## Batch integrity

Stage-2 type/status fields can no longer be overwritten by upstream job
metadata. A stage-2 exception is reported as failed. A completed bundle must
contain parseable camera/actor/frame/job JSON with matching frame indices,
counts and SHA-256 hashes. Four filenames alone are not sufficient evidence
of a valid dataset. Source submodules and account configuration are unchanged.

## Validation

The retained H100 validation workspace is:
`outputs/validation_camera_bbox_20261112`.

- 46 unit tests passed locally and in the isolated H100 test environment.
- Seven complete clips, 300 frames total, were tested with actual SAM3/DISK/DA3.
- Independent estimation completed six; one was rejected for insufficient
  camera support. Some completed estimates still differed substantially from
  registered cameras, confirming the limits of count/RMSE-only assurance.
- Registered-first controls completed all seven, checking 14 source images.
  Six reused raw DA3 predictions and one ran inference after blind estimation
  had been rejected. Their zero endpoint discrepancy is by construction and
  must not be presented as independent pose-estimation accuracy.
- The multi-person S08E04 general_0144 example changed world depth scale from
  7.3207 to 3.5373. A 25x48px background-frame detection was flagged for proxy
  review while all five principal-person proxies remained eligible.

Camera consistency does not establish true actor depth, anatomical head
location, or perfect asset alignment. DA3 foreground depth, occlusion,
segmentation and manually placed scene geometry still require visual review.


The batch `--dry-run` prints the chosen GPU devices, anchor policy, actor up
vector and pending jobs without launching model workers. It is useful for
reviewing a run while GPU4 is occupied.
