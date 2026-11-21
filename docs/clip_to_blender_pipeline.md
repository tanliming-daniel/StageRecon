# Clip-to-Blender pipeline

Omega 场景发布、24 镜头回归和新场景接入见 [场景更新流程](omega_scene_workflow.md)。

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

## Omega 精修场景的批量数据产出

批量入口默认使用 `--camera-backend omega`。172 张筛选剧照的精修重建位于
`/root/userdata/300workspace/merg3r_work/omega_20261119_reviewed172_ba_regularized`。
`batch_scene.json` 记录了 P70 点云哈希、场景文件哈希、Omega 到 Blender 的相似变换和人物尺度。
当前场景为 `data/Living_Room_Frame_Checked.blend`，其中的点云与此 P70 点云 SHA-256 一致。

SAM3 生成视频人物遮罩和轨迹。首尾帧对筛选后的 Omega 静态参考图进行 SIFT 匹配和 PnP，
使用 70% 对应点拟合相机及焦距，并用剩余 30% 验证。验证失败会记录失败原因。
中间帧继续使用 DA3，通过同一份 Omega 点云拟合深度标尺，并以两个 PnP 端点约束相机轨迹。
人物代理及相机统一使用 `omega_ba_world`，导入 Blender 时应用已核对的 `pointcloud` 变换。

默认输出为 `outputs/Friends-monica_apartment/clip_render_data_s08_omega_persistent`。
每个完成的镜头包含 `camera_track.json`、`actor_tracks.json`、`frames.json` 和 `render_job.json`。
渲染任务记录场景 SHA-256，导入器检查场景中的点云变换。

在 H100 项目目录运行单镜头验收或指定剧集的批次。

```bash
python3 tools/run_clip_render_data_batch.py   --episode S08E04 --include general_0091_global_0218 --limit 1   --run-name omega_checked_scene_smoke

python3 tools/run_clip_render_data_batch.py   --episode S08E04 --run-name omega_s08e04
```

`--omega-reconstruction` 和 `--omega-images` 可指定其他已配置的 Omega 场景及参考图目录。
`--camera-backend colmap` 可调用原有后端。

Omega 人物深度按半秒采样，使用最终相机将静态点云投影到 DA3 图像，排除人物和深度边缘。相机运动继续使用原端点校准。查看 camera_track.json 的 da3.actor_depth_alignment 可见各帧留出区域的残差、人物周围静物残差和实际尺度。时间变化需通过留帧预测和留出像素验证。


每个已确认的人物在单镜头全时间轴只创建一个球头圆柱对象。相邻 ID 片段仅在时间不重叠且世界位置、速度、头部投影均唯一匹配时合并。actor_tracks.json 保留 source_track_ids、observed_ranges 和每帧 observation_status（observed、interpolated、hold、motion_extrapolation、offscreen_extrapolation）。出画与家具遮挡由空间位置和相机自然产生。

支撑候选使用绑定场景哈希的坐垫中带、地面和家具深度采样。每半秒估计深度、站立或具体座面候选，按整段证据平滑选择；近景证据不足时交付深度估计。人物分配与镜头名称、人物左右次序无关。Blender 导入只为运动和代理几何插值，人物全程保持正缩放。

验证工具 tools/verify_clip_presence_blender.py 在 Blender 后台接受 --bundles 和 --output，检查每帧对象身份、渲染可见性和真实几何变换，并导出预演和每帧轮廓投影。每个输入视频应对应单个连续镜头。
