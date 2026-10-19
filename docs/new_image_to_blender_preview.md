# New Image To Camera Info

> Current runnable reference: [`image_to_blender_preview_pipeline.md`](image_to_blender_preview_pipeline.md).
> This older note predates the `PredictedCamera_query_0001`, global-DISK fallback,
> and `Solid + Texture + Studio` preview workflow.

This is the runnable pipeline for placing one **new image** in the existing
Friends Monica apartment coordinate system and exporting a camera-only
artifact that can be imported into an already opened Blender scene.

Runner:

```text
tools/run_new_image_to_blender.py
```

It is a localization pipeline, not a reconstruction pipeline. The query image
is never inserted into COLMAP. The existing COLMAP tracks and the already
instantiated Blender scene remain the source of geometry.

## Inputs

| Input | Purpose |
| --- | --- |
| New image | Query to localize. PNG, JPG, and other OpenCV-readable formats work. |
| Scene directory | Existing Friends images, h5, and colmap data. |
| Blender input | Your instantiated scene; it must retain the pointcloud controller because its transform aligns COLMAP and Blender coordinates. |
| DISK/CUDA Python | Extracts query local features and matches them to the scene H5 descriptors. |
| Blender executable | Should be the same version or newer than the version that saved the input blend. |

For this workspace, the useful Blender input is:

```text
sitcoms3D/data/sparse_reconstruction_and_nerf_data/Friends-monica_apartment/scene_with_predicted_camera.blend
```

## Run

Use a separate output directory per query:

```bash
 /root/userdata/300workspace/sm3r/bin/python \
  tools/run_new_image_to_blender.py \
  --query-image /absolute/path/to/new_frame.png \
  --blender-input /root/userdata/300workspace/StageRecon/sitcoms3D/data/sparse_reconstruction_and_nerf_data/Friends-monica_apartment/scene_with_predicted_camera.blend \
  --output-dir /root/userdata/300workspace/StageRecon/outputs/Friends-monica_apartment/new_image_runs/my_frame \
  --blender-executable /root/userdata/300workspace/blender-5.1.1-linux-x64/blender \
  --camera-name PredictedCamera_query_0001
```

If the query comes from a calibrated camera rig, pass its pixel focal length:

```bash
  --focal-pixels 2558.567097579417
```

If focal length is unknown, omit --focal-pixels. The runner falls back to a
coarse focal multiplier search, which is usable for a rough pose but is less
reliable for camera translation.

For an exact uniform resize of a reference image, scale its focal length by:

```text
f_query = f_reference * query_width / reference_width
```

To remove people from PnP constraints, provide a query mask. Mask value 0
keeps a keypoint; nonzero pixels reject it:

```bash
  --query-mask /absolute/path/to/new_frame_dynamic_mask.png
```

## Stages

1. The query is copied to query_images/query.jpg without resizing.
2. Coarse YCrCb and edge descriptors retrieve registered Friends frames.
   This only narrows matching; it is not the final pose.
3. The existing depth-save.pth DISK model extracts query keypoints and
   descriptors.
4. Query descriptors are mutually ratio-matched against retrieved registered
   frames. Each valid reference keypoint maps through its existing COLMAP
   point3D_id, producing query 2D to world 3D correspondences.
5. RANSAC PnP plus LM refinement estimates R, t, intrinsics, and the camera
   center in raw COLMAP world coordinates.
6. Blender opens the supplied scene read-only and uses pointcloud.matrix_world
   to convert the pose to Blender coordinates. It writes camera_info.json only;
   it never writes a new blend file.
7. import_predicted_camera.py applies camera_info.json to an existing camera
such as PredictedCamera_query_0001 and marks it active. The script never saves the scene.
8. A viewport PNG is optional: pass --preview when an Xvfb/OpenGL capture is
   needed. It is disabled by default so localization cannot block on preview.

## Outputs

| File | Contents |
| --- | --- |
| retrieval_candidates.tsv | Coarse registered-frame ranking. |
| query_h5/ | Query DISK keypoints and descriptors. |
| localization_colmap.json | PnP intrinsics, pose, camera center, inliers, RMSE, and source frames. |
| camera_info.json | Blender matrix, focal settings, COLMAP pose, PnP metrics, and alignment transform. |
| import_predicted_camera.py | Import script that updates shot1 (or another requested camera) without saving the blend. |
| predicted_camera_preview.png | Optional temporary OpenGL capture; emitted only with --preview. |
| run_manifest.json | Paths and PnP summary for the run. |

## Import Into Blender

Open the target scene in Blender and run the generated script. It updates the
requested target camera (by default shot1), marks it active, and does not save
the blend:

~~~bash
/root/userdata/300workspace/blender-5.1.1-linux-x64/blender \
  /absolute/path/to/scene_with_predicted_camera.blend \
  --python /absolute/path/to/import_predicted_camera.py -- \
  --camera-info /absolute/path/to/camera_info.json \
  --camera-name PredictedCamera_query_0001
~~~

Use --preview on the runner only when a transient viewport PNG is wanted.

Use the JSON quality indicators before accepting a sample. A practical baseline
is at least 20 PnP inliers and inlier RMSE below about five pixels. A poor match
or unknown focal length can still return a mathematically valid but visually
incorrect pose.

## Controls

- --top-k 80: retrieved frames entering DISK matching.
- --retrieval-pool 200: frames recorded by appearance retrieval.
- --focal-pixels: calibrated focal candidates in pixels, comma-separated.
- --exclude-reference frame.jpg: repeat to forbid known source frames.
- --hide-collection name: repeat to hide more collections in the preview.
- --preview: opt-in transient viewport capture; it does not save the Blender scene.
- --preview-width 960 --preview-height 540: preview dimensions.

## Blender Version Note

The current input blend has reported that it was written by a newer Blender
binary than the local 5.1.1 executable. For real data generation, use the
Blender version that created the scene or a newer compatible build. The runner
always writes a new output blend and does not overwrite the input.
