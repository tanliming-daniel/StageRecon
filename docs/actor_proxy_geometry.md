# Actor bbox estimation and fixed-size proxies

A SAM visible bbox is an observation, not a person's physical dimensions.
Each clip uses fixed `proxy_convention.height_world` and `radius_world`
(default 1.70 and 0.18 in COLMAP world units, not necessarily metres).
Neither framewise bbox changes nor posture labels resize the mesh.

The estimator keeps the connected person's visible bbox separate from the
upper-body axis/head-top anchor. Detached components no longer expand the
localization bbox, and the bbox is not shifted sideways onto the torso axis.
The robust top envelope replaces a percentile over all central body pixels,
which moved the top down when more of the lower body was visible.
Both 0/1 and 0/255 masks are supported. Relative masks resolve against the
person-tracks JSON directory. DA3 array positions map to camera frame IDs
explicitly, including nonzero/sparse IDs.

`head_anchor_world` is backprojected at the filtered DA3 depth plus the
configured surface offset. `proxy_center_world` lies half the fixed height
below it along one fixed clip up vector. Supply `--world-up X Y Z` to the
estimator when calibrated scene up is available; otherwise the median camera
up is an explicit prior, not a recovered floor normal.

Blender imports the same anchor and up direction, including the pointcloud
alignment's uniform scale. For legacy bundles, it first recovers the head
using the old camera-up convention and then places the proxy along scene up.
This removes the mismatch between a camera-up-derived centre and a world-Z
cylinder. The importer also respects principal-point offsets and fx/fy,
preserves fractional FPS, and uses constant scale keyframes.

The render job runner accepts `--actor-proxy-style cylinder` or
`--actor-proxy-style head-cylinder` (default). The latter adds a sphere head and a
cream horizontal band on the front of the sphere (+local X) while retaining the same total height and body radius. Facing
comes from saved motion/camera-facing priors and is not observed head pose.
Blender Python failures now cause a nonzero runner exit.

Example, from the repository root:

```bash
python3 tools/run_clip_render_job.py \
  --job data/clip_render_data_s08/S08E04/general_0196_global_0350_f00032018-f00032041/render_job.json \
  --base-blend data/3dscene_living_room_aligned.blend \
  --blender-executable /data5/cy/tlm/blender-5.2.3-linux-x64/blender \
  --output-blend outputs/previews/S08E04_general_0196_bbox_fix/scene_with_clip_animation.blend \
  --blender-metadata outputs/previews/S08E04_general_0196_bbox_fix/blender_animation.json \
  --actor-proxy-style head-cylinder
```

Use `tools/render_clip_headless_preview.py` to render this scene as described
in `blender_5_2_headless_preview.md`. To check all observed frames independently
using Blender's projection implementation:

```bash
/data5/cy/tlm/blender-5.2.3-linux-x64/blender --background --disable-autoexec \
  outputs/previews/S08E04_general_0196_bbox_fix/scene_with_clip_animation.blend \
  --python-exit-code 1 --python tools/verify_clip_actor_projection.py -- \
  --output outputs/previews/S08E04_general_0196_bbox_fix/verification.json
```

This verifies fixed mesh dimensions and agreement with stored 2D anchors.
It does not prove human-head or depth accuracy. Cropped heads, torso-axis
ambiguity, monocular depth scale, and carried/occluding objects still need
source-video/mask/depth validation. Portable render bundles omit those inputs;
rendering them exercises the importer fix but cannot rerun the mask estimator.

Tiny interior tracks can carry a track-level `proxy_quality` review flag. The
raw track is retained, but Blender omits an ineligible proxy for the complete
clip. See `production_registered_anchors.md` for criteria and validation.

## Multi-HMR for half-body shots and close-ups

Pass `--multihmr-checkpoint` and `--multihmr-source` to `tools/run_clip_to_blender.py` to enable the optional pose stage with the DA3 camera tracker. Use `--multihmr-python` for the Python environment containing PyTorch, SciPy, OpenCV and the official Multi-HMR dependencies. The source checkout needs the official SMPL-X assets. An offline DINOv2 checkout can be passed with `--dinov2-source`. The calibrated RGB inference path was exercised with the original 672_B checkpoint.

For an existing shot, run the refinement stage directly:

```sh
python tools/refine_actor_pose.py \
  --actor-tracks "$BUNDLE/actor_tracks.json" --camera-track "$BUNDLE/camera_track.json" \
  --person-tracks "$RUN/masks/person_tracks.json" --frames-dir "$RUN/frames" \
  --multihmr-source "$MULTIHMR" --checkpoint "$CHECKPOINT" \
  --output "$BUNDLE/actor_tracks_multihmr.json"
```

Give the resulting actor JSON to the usual Blender clip importer. The single-clip runner puts it in the portable render job automatically. SAM3 IDs and DA3 positions remain the reference. Projected joints associate whole-image detections with instance masks one-to-one. For unmatched IDs, the same model gets a padded RGB crop with subdued background and adjusted camera intrinsics.

Each actor sample records separate head and body sources in `proxy_orientation`. The head basis uses the SMPL-X chest, neck and head rotation chain. A torso update requires the chest and both projected shoulders inside the same mask with plausible separation below the head. This is a visibility heuristic. In a face-only view, the importer uses the latest supported torso orientation from this ID in the shot, or the original actor direction if none has been observed. When the head itself is missing, its orientation is held for up to three frames. The resulting band represents head direction; gaze and pixel-accurate eyes require separate evidence.

The Blender head-cylinder style animates independent head and torso meshes. The fitted head uses the visible upper-mask width, and the original DA3 depth anchors its position. Reliable shoulder observations determine torso orientation. When shape fitting has no usable sample for an entire actor, the proxy uses default dimensions.
