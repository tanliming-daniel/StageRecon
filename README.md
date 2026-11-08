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
- MTSS-3D caption generation with Reference, Shot, Event, and Global streams.
- SAM3-track identity binding, conservative anonymous fallback, and geometry references.

## Repository layout

- `tools/`: localization, tracking, Blender import, preview, and shot-cut tools.
- `stagerecon/captioning/`: MTSS-3D schemas, identity binding, Qwen adapter, validation, and training prompt assembly.
- `schemas/`, `prompts/`, `configs/`: versioned caption contracts and defaults.
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
- [MTSS-3D video caption pipeline](docs/mtss_3d_caption_pipeline.md)
- [Qwen3.8-Omni 视频 Caption 数据处理说明](docs/qwen38_omni_video_caption_data_guide.md)
- [Friends 分集角色候选图库](docs/episode_character_gallery.md)

## MTSS-3D caption dry-run

Run the full caption data path without calling a model API:

```bash
python tools/run_mtss3d_caption.py \
  --clip-run outputs/scene/clip_runs/example \
  --previs-video outputs/scene/clip_runs/example/camera_track_playblast.mp4 \
  --registry configs/characters/friends.json \
  --dry-run \
  --no-resume
```

### Qwen3.8-Omni joint audio-video captioning

The default caption path sends exactly one real target MP4 with its embedded
original audio, person reference crops, stable `ref_id` mappings, duration, and
FPS to `qwen3.8-omni-flash`. The previs/playblast is not sent to Qwen. It is
retained only as the downstream training control video.

```bash
export DASHSCOPE_API_KEY=...
python tools/run_mtss3d_caption.py \
  --clip-run outputs/sample \
  --previs-video outputs/sample/camera_track_playblast.mp4 \
  --target-video data/clips/sample.mp4 \
  --registry configs/characters/friends.json \
  --no-resume
```

Omni jointly produces Reference, Shot, Event, and Global streams, including
verbatim dialogue, timing, speaker-to-`PERSON_*` binding, sound effects, music,
and audio-visual links.

### Optional ASR fallback (MOSS 0.9B)

DA3 and MOSS intentionally use separate Python runtimes because their official
requirements conflict (`torch 2.7 / transformers 4.x` versus
`torch 2.8 / transformers 5.x`). StageRecon keeps the user-facing workflow
unified and switches runtimes internally.

Run transcription alone:

```bash
python tools/run_moss_transcribe.py input.mp4 \
  --output outputs/sample/captions/moss_transcript.json \
  --cuda-visible-devices 4
```

Passing a transcript or split proposal explicitly selects the legacy fallback.
MOSS can generate the timestamped transcript used by that path:

```bash
python tools/run_mtss3d_caption.py \
  --clip-run outputs/sample \
  --previs-video input.mp4 \
  --audio-from-source \
  --transcribe-moss \
  --moss-cuda-visible-devices 4
```

MOSS emits timestamped `S01`, `S02`, ... speaker labels. The transcript keeps
`person_id` unresolved until a visual identity/lip-activity binding stage maps
each acoustic speaker to a stable `PERSON_*` reference.

The output contains `mtss_3d.json`, identity RGBA crops, frozen geometry files,
a validation report, an HTML review page, and a versioned training record.

## Development checks

The lightweight repository check compiles every Python source file without
importing optional ML or Blender dependencies:

```bash
python -m compileall -q stagerecon tools tests
python -m unittest discover -s tests -v
```
