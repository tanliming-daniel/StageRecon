# Clip-to-Blender pipeline

`tools/run_clip_to_blender.py` treats one input clip as one already-separated shot. It performs absolute localization only on source frames 0 and N-1, then uses a video-level model for every intermediate frame:

1. Decode the selected source range (`--max-frames 99` selects frames 0 through 98).
2. Load SAM3 once and propagate the text prompt `person` through the whole frame folder. The saved instance IDs are stable within the clip.
3. Run Depth Anything 3 once on the complete shot. Express its full SE(3) camera poses relative to frame 0, estimate one translation scale by aligning frame-0 DA3 depth with the COLMAP scene point cloud, and compose the result onto the unchanged frame-0 PnP pose. SAM3 masks exclude people from the scale fit. By default, relative rotation, translation, and log focal length receive zero-phase, third-order Savitzky-Golay smoothing over a 21-frame window. The remaining terminal position, rotation, and focal drift is distributed continuously across the shot so frame 0 exactly equals the first PnP result and frame N-1 exactly equals the last PnP result. The old fixed-camera-centre PTZ tracker is still available through `--camera-tracker ptz`.
4. Combine every SAM3 instance mask with the corresponding DA3 depth. Localization does not use the mask centroid or the 2D bbox midpoint. The largest connected person support is converted to a distance transform; on upper-body scanlines, the thickest interior locations form a torso-led medial axis. A weighted median of that axis rejects thin extended arms and off-axis carried objects. DA3 depth is sampled only from the thick central upper-body core, then confidence-filtered and robustly trimmed in log-depth space. The raw SAM3 bbox remains in the JSON for diagnosis. The body anchor, depth, and visible extent receive temporal filtering per track before conversion into the corrected dual-anchor camera's world frame.
5. Import the camera and fixed-size actor cylinders into one Blender scene. The camera gets location, rotation, and lens keyframes. Each cylinder uses `proxy_center_world`, obtained from the temporally filtered torso-axis head ray and half of the fixed 1.70-world-unit proxy height. This avoids shifts caused by arms, carried objects, or legs appearing after an occlusion, and does not use a foot pixel or floor intersection. The visible 3D box remains localization metadata and never changes the preview geometry's size. A coarse facing direction and posture label (`standing`, `sitting`, or `transition`) remain in `actor_tracks.json` as optional metadata.

The maintained layout input is `data/scene_with_clip_animation.blend`. Each
clip import keeps that scene's layout, materials, and lighting, removes only
the previously generated `ClipCamera` and `ClipActors` objects, then creates
the current camera and actor animation.

The output directory contains `clip_manifest.json`, `masks/person_tracks.json`, `anchor/localization_colmap.json`, `tail_anchor/localization_colmap.json`, `camera_track.json`, `da3_prediction.npz`, `actor_tracks.json`, `diagnostic_overlay.mp4`, and `scene_with_clip_animation.blend`. Add `--preview` to create the same native Workbench PNG used by the image pipeline at `predicted_camera_preview.png`. The middle tracked-camera frame is used by default; `--preview-frame` selects a Blender frame explicitly. `absolute_localization_frames` is `[0, N-1]`; no intermediate frame invokes the DISK/PnP localizer.

The video playblast path is `tools/render_clip_viewport_playblast.py`. It sets both the active `VIEW_3D` shading and `scene.display.shading` to `SOLID` + `STUDIO` + `TEXTURE`; setting only the viewport space can produce flat grey materials under Xvfb. It renders from the active `PredictedClipCamera` and keeps the viewport draw cache for the complete animation.

Example for the two supplied clips:

```bash
SM3R=/root/userdata/300workspace/sm3r/bin/python

$SM3R tools/run_clip_to_blender.py \
  --video data/clips/1028_1791454112.mp4 \
  --output-dir outputs/Friends-monica_apartment/clip_runs/1028_1791454112 \
  --max-frames 99 \
  --python-executable "$SM3R"

$SM3R tools/run_clip_to_blender.py \
  --video data/clips/1029_1791454119.mp4 \
  --output-dir outputs/Friends-monica_apartment/clip_runs/1029_1791454119 \
  --python-executable "$SM3R"
```

Use `--no-blender` when only masks, camera JSON, actor JSON, and the diagnostic overlay are needed. If a cached output was created with a different frame selection, use `--overwrite` or another output directory. The first-frame anchor keeps the existing single-image localization defaults (`top-k=20`, ratio `0.90`); the resulting `anchor_quality` is recorded in the manifest. DA3 is the default camera tracker and uses the local `DA3-LARGE-1.1` checkpoint. `--camera-tracker ptz` selects the legacy fixed-centre propagation without changing the PnP implementation.

## Render on a different workstation

Every successful analysis run now writes `render_job.json` beside
`camera_track.json`, `actor_tracks.json`, and `frames.json`. Those four files are
the portable rendering input. The render job stores relative data paths,
timeline settings, active camera and collection names, fixed actor proxy
settings, viewport shading settings, and SHA-256 checksums. Masks, extracted
frames, DA3 arrays, and diagnostic videos are not required to build the Blender
animation.

Run analysis on the compute machine without creating the final `.blend` file:

```bash
$SM3R tools/run_clip_to_blender.py \
  --video data/clips/1028_1791454112.mp4 \
  --output-dir outputs/Friends-monica_apartment/clip_runs/1028_1791454112 \
  --max-frames 99 \
  --python-executable "$SM3R" \
  --no-blender
```

Copy the complete clip output directory to the graphics workstation. Supply its
local base scene path because the development-machine path in the render job is
only a hint:

```bash
python tools/run_clip_render_job.py \
  --job /data/clip_runs/1028_1791454112/render_job.json \
  --base-blend /data/scenes/scene_with_clip_animation.blend \
  --blender-executable /opt/blender/blender \
  --playblast /data/clip_runs/1028_1791454112/camera_track_playblast.mp4 \
  --display :0
```

Omit `--playblast` to create only `scene_with_clip_animation.blend`. Add
`--reuse-scene` to keep an existing animated scene and render it again. The
playblast command requires hardware OpenGL by default and reports the Blender
GPU vendor, renderer, backend, and device type in its metadata. This prevents an
Xvfb display backed by Mesa `llvmpipe` from silently starting a very slow
software render. `--allow-software-opengl` is available for deliberate CPU
tests.

The actor representation is intentionally coarse. `bbox_3d.center_world`, `size_world` (`width`, visible `height`, `depth`), and `axes_world` describe each camera-oriented box. The box estimates the visible segmented extent, so an actor cropped by the image boundary does not receive an invented full-body height. It does not estimate SMPL pose, articulated joints, or exact body orientation. Motion supplies a coarse facing estimate; a nearly static actor retains a low-confidence camera-facing prior.
