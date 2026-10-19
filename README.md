# StageRecon

StageRecon is an experimental pipeline for reconstructing TV stages and placing
shot cameras and coarse actor proxies into an existing 3D set. A clip is treated
as one pre-cut shot. The repository contains the orchestration and conversion
tools; datasets, checkpoints, Blender scenes, extracted frames, and generated
outputs stay outside Git.

The current clip pipeline combines:

- DISK-to-COLMAP absolute camera localization at shot anchors.
- SAM3 video person masks with stable track IDs.
- Depth Anything 3 relative camera motion and depth.
- Coarse world-space actor tracks and fixed-size Blender proxy cylinders.
- Blender camera animation and Studio + Texture viewport previews.

## Repository layout

- `tools/`: localization, tracking, Blender import, preview, and shot-cut tools.
- `docs/`: workflow details and environment-specific examples.
- `ShowMak3r/`: upstream reconstruction pipeline, pinned as a submodule.
- `sitcoms3D/`: upstream scene utilities, pinned as a submodule.
- `external/OmniShotCut/`: upstream shot-boundary model, pinned as a submodule.

## Clone

```bash
git clone --recurse-submodules git@github.com:tanliming-daniel/StageRecon.git
cd StageRecon
```

For an existing clone:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

Model checkpoints and scene data are intentionally untracked. Pass their paths
through the command-line options documented in each tool. The checked-in
defaults reflect the development machine and are examples rather than portable
installation paths.

## Clip analysis and rendering

Generate tracking data without building a final animated Blender scene:

```bash
python tools/run_clip_to_blender.py \
  --video data/clips/example.mp4 \
  --output-dir outputs/example \
  --no-blender
```

The output includes `camera_track.json`, `actor_tracks.json`, `frames.json`, and
`render_job.json`. Move the output directory to a workstation with a graphics
GPU, then build the animated scene and optionally create a viewport playblast:

```bash
python tools/run_clip_render_job.py \
  --job outputs/example/render_job.json \
  --base-blend /path/to/scene_with_clip_animation.blend \
  --blender-executable /path/to/blender \
  --playblast outputs/example/camera_track_playblast.mp4 \
  --display :0
```

The render job uses paths relative to the clip output directory and verifies
the saved camera, actor, and frame metadata with SHA-256 checksums. Viewport
playblast rejects software OpenGL by default so an accidental Xvfb/llvmpipe run
does not consume hours. Use `--allow-software-opengl` only when that behavior is
intentional.

## Documentation

- [Clip to Blender pipeline](docs/clip_to_blender_pipeline.md)
- [New image camera localization](docs/new_image_camera_localization.md)
- [New image to Blender preview](docs/new_image_to_blender_preview.md)
- [Image to Blender pipeline notes](docs/image_to_blender_preview_pipeline.md)

## Development checks

The lightweight repository check compiles every Python source file without
importing optional ML or Blender dependencies:

```bash
python -m compileall -q tools
```
