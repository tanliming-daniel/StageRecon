#!/usr/bin/env python3
"""Run the single-shot clip localization and coarse actor pipeline.

Frames zero and N-1 are sent through the existing DISK-to-COLMAP PnP path.
SAM3 is loaded once for the clip. By default, Depth Anything 3 estimates the
complete relative SE(3) trajectory in one video inference call; its terminal
drift is distributed between the two exact PnP endpoints. The legacy
fixed-centre PTZ tracker remains available as an explicit fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from clip_render_job import write_render_job
from pose_quality import assess_pose_quality
from actor_proxy_config import resolve_actor_world_up


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENVIRONMENT = (
    REPO_ROOT / "sitcoms3D/data/sparse_reconstruction_and_nerf_data/Friends-monica_apartment"
)
DEFAULT_BLEND = REPO_ROOT / "data/scene_with_clip_animation.blend"
DEFAULT_BLENDER = Path("/root/userdata/300workspace/zdj/blender/blender")
DEFAULT_SAM3_ROOT = Path("/root/userdata/300workspace/sam3")
DEFAULT_SAM3_CHECKPOINT = Path("/root/userdata/300workspace/ckpt/sam3/sam3.pt")
SINGLE_IMAGE_RUNNER = REPO_ROOT / "tools/run_new_image_to_blender.py"
SAM3_TRACKER = REPO_ROOT / "tools/extract_person_tracks_sam3.py"
CAMERA_TRACKER = REPO_ROOT / "tools/estimate_clip_camera.py"
CAMERA_TRACKER_DA3 = REPO_ROOT / "tools/estimate_clip_camera_da3.py"
ACTOR_TRACKER = REPO_ROOT / "tools/estimate_actor_boxes.py"
ACTOR_TRACKER_FLOOR = REPO_ROOT / "tools/estimate_actor_tracks.py"
BLENDER_IMPORTER = REPO_ROOT / "tools/import_clip_animation.py"
CLIP_PREVIEW = REPO_ROOT / "tools/capture_clip_viewport_preview.py"
DEFAULT_DA3_PYTHON = Path("/root/userdata/300workspace/da3/bin/python")
DEFAULT_DA3_SOURCE = Path("/root/userdata/300workspace/Depth-Anything-3/src")
DEFAULT_DA3_MODEL = Path(
    "/root/userdata/300workspace/zzf/server/Depth-Anything-3/DA3-LARGE-1.1"
)
DEFAULT_BLENDER_TEMP = REPO_ROOT / ".cache/blender/tmp"
DEFAULT_BLENDER_RUNTIME = REPO_ROOT / ".cache/blender/xdg-runtime"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--environment-dir", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--blender-input", type=Path, default=DEFAULT_BLEND)
    parser.add_argument("--blender-executable", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--sam3-root", type=Path, default=DEFAULT_SAM3_ROOT)
    parser.add_argument("--sam3-checkpoint", type=Path, default=DEFAULT_SAM3_CHECKPOINT)
    parser.add_argument("--sam3-version", choices=("sam3", "sam3.1"), default="sam3")
    parser.add_argument("--sam3-prob-threshold", type=float, default=0.75)
    parser.add_argument("--sam3-dilate-pixels", type=int, default=12)
    parser.add_argument(
        "--actor-depth-offset",
        type=float,
        default=0.36,
        help="World-space offset from visible person surface to proxy centre.",
    )
    parser.add_argument("--actor-world-up", type=float, nargs=3)
    parser.add_argument("--allow-review-anchors", action="store_true", help="Explicitly allow review-quality camera endpoints.")
    parser.add_argument("--focal-pixels", default="")
    parser.add_argument("--anchor-top-k", type=int, default=20)
    parser.add_argument("--anchor-ratio", type=float, default=0.90)
    parser.add_argument("--camera-name", default="PredictedClipCamera")
    parser.add_argument("--camera-tracker", choices=("da3", "ptz"), default="da3")
    parser.add_argument("--da3-python", type=Path, default=DEFAULT_DA3_PYTHON)
    parser.add_argument("--da3-source", type=Path, default=DEFAULT_DA3_SOURCE)
    parser.add_argument("--da3-model", type=Path, default=DEFAULT_DA3_MODEL)
    parser.add_argument("--da3-device", default="cuda:0")
    parser.add_argument("--da3-process-res", type=int, default=504)
    parser.add_argument(
        "--da3-smoothing-mode",
        choices=("none", "gaussian", "savgol"),
        default="savgol",
    )
    parser.add_argument("--da3-smooth-sigma", type=float, default=0.35)
    parser.add_argument("--da3-smooth-window", type=int, default=21)
    parser.add_argument("--da3-smooth-polyorder", type=int, default=3)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Decode at most this many frames from the start; 0 keeps the full input.",
    )
    parser.add_argument(
        "--no-blender",
        action="store_true",
        help=(
            "Defer final Blender scene construction and rendering. Camera/actor "
            "JSON and a portable render_job.json are still written."
        ),
    )
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-frame", type=int, default=0)
    parser.add_argument("--viewport-display", default=":88")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def blender_environment() -> dict[str, str]:
    DEFAULT_BLENDER_TEMP.mkdir(parents=True, exist_ok=True)
    DEFAULT_BLENDER_RUNTIME.mkdir(parents=True, exist_ok=True)
    DEFAULT_BLENDER_RUNTIME.chmod(0o700)
    environment = os.environ.copy()
    environment.update(
        {
            "TMPDIR": str(DEFAULT_BLENDER_TEMP),
            "TEMP": str(DEFAULT_BLENDER_TEMP),
            "TMP": str(DEFAULT_BLENDER_TEMP),
            "XDG_RUNTIME_DIR": str(DEFAULT_BLENDER_RUNTIME),
        }
    )
    return environment


def run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    print("$", shlex.join(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def measure(name: str, durations: dict[str, float], action) -> None:
    started = time.perf_counter()
    action()
    durations[name] = round(time.perf_counter() - started, 3)


def up_to_date(output: Path, *inputs: Path) -> bool:
    if not output.is_file():
        return False
    output_time = output.stat().st_mtime
    return all(path.is_file() and path.stat().st_mtime <= output_time for path in inputs)


def camera_track_matches(path: Path, tracker: str) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text())
        model = str(data.get("model", ""))
    except (OSError, json.JSONDecodeError):
        return False
    if tracker == "da3":
        return (
            model == "depth_anything_3_large_full_clip_dual_endpoint_pnp"
            and len(data.get("absolute_localization_frames", [])) == 2
        )
    return model.startswith("fixed_center")


def anchor_quality(localization_path: Path, frame_index: int) -> dict:
    pose = json.loads(localization_path.read_text())
    pnp = pose["pnp"]
    inliers = int(pnp["inlier_count"])
    rmse = float(pnp["inlier_rmse_pixels"])
    status = assess_pose_quality(pose)["status"]
    return {
        "frame_index": frame_index,
        "status": status,
        "pnp_inlier_count": inliers,
        "pnp_rmse_pixels": rmse,
        "focal_pixels": float(pnp["focal"]),
    }


def extract_frames(video: Path, frames_dir: Path, quality: int, max_frames: int) -> dict:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_records = []
    frame_index = 0
    while True:
        if max_frames > 0 and frame_index >= max_frames:
            break
        ok, frame = capture.read()
        if not ok:
            break
        path = frames_dir / f"{frame_index:06d}.jpg"
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
            raise RuntimeError(f"Could not write {path}")
        frame_records.append(
            {
                "frame_index": frame_index,
                "frame_name": path.name,
                "timestamp_seconds": frame_index / fps,
            }
        )
        frame_index += 1
    capture.release()
    if not frame_records:
        raise ValueError(f"No frames decoded from {video}")
    metadata = {
        "video": str(video),
        "fps": fps,
        "frame_count": len(frame_records),
        "source_frame_count": source_frame_count,
        "source_frame_range_inclusive": [0, len(frame_records) - 1],
        "max_frames": max_frames,
        "duration_seconds": len(frame_records) / fps,
        "resolution": {"width": width, "height": height},
        "frames": frame_records,
    }
    (frames_dir.parent / "frames.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def diagnostic_video(
    frames_dir: Path,
    people_path: Path,
    actors_path: Path,
    camera_path: Path,
    fps: float,
    output: Path,
) -> None:
    people = json.loads(people_path.read_text())
    actors = json.loads(actors_path.read_text())
    camera = json.loads(camera_path.read_text())
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    height, width = first.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open diagnostic video writer: {output}")
    people_by_frame: dict[int, list[tuple[int, dict]]] = {}
    for track in people.get("tracks", []):
        for item in track["frames"]:
            people_by_frame.setdefault(int(item["frame_index"]), []).append(
                (int(track["track_id"]), item)
            )
    actors_by_frame: dict[int, dict[int, dict]] = {}
    for track in actors.get("tracks", []):
        for item in track["frames"]:
            actors_by_frame.setdefault(int(item["frame_index"]), {})[
                int(track["track_id"])
            ] = item
    camera_by_frame = {int(item["frame_index"]): item for item in camera["frames"]}
    palette = [(48, 80, 255), (255, 160, 40), (60, 210, 80), (210, 70, 210), (40, 210, 220)]
    for frame_index, frame_path in enumerate(frame_paths):
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        for track_id, item in people_by_frame.get(frame_index, []):
            color = palette[track_id % len(palette)]
            mask = cv2.imread(item["mask"], cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, color, 2)
            x1, y1, x2, y2 = [int(value) for value in item["bbox_xyxy"]]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            actor = actors_by_frame.get(frame_index, {}).get(track_id)
            label = f"ID {track_id}"
            if actor is not None:
                body_anchor = actor.get("body_anchor_xy")
                if body_anchor is not None:
                    anchor_x, anchor_y = [int(round(value)) for value in body_anchor]
                    cv2.circle(image, (anchor_x, anchor_y), 7, (40, 255, 40), -1)
                    cv2.line(
                        image,
                        (anchor_x, max(0, y1)),
                        (anchor_x, min(height - 1, y2)),
                        (40, 255, 40),
                        2,
                    )
                box = actor.get("bbox_3d", {})
                size = box.get("size_world", [0.0, 0.0, 0.0])
                label += (
                    f" {actor['posture']} "
                    f"3D={size[0]:.2f}x{size[1]:.2f}x{size[2]:.2f}"
                )
            cv2.putText(image, label, (x1, max(24, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
        cam = camera_by_frame[frame_index]
        text = (
            f"frame {frame_index:04d}  f={cam['focal_pixels']:.1f}px  "
            f"camera={cam['quality']['status']}"
        )
        cv2.rectangle(image, (0, 0), (width, 38), (0, 0, 0), -1)
        cv2.putText(image, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
        writer.write(image)
    writer.release()


def main() -> None:
    args = parse_args()
    args.actor_world_up = resolve_actor_world_up(args.environment_dir, args.blender_input, args.actor_world_up, REPO_ROOT / "configs/scenes")
    if args.no_blender and args.preview:
        raise ValueError("--preview cannot be combined with --no-blender")
    started = time.perf_counter()
    durations: dict[str, float] = {}
    video = require_file(args.video, "Input video")
    environment = args.environment_dir.resolve()
    blender_input = require_file(args.blender_input, "Blender scene")
    require_file(args.sam3_checkpoint, "SAM3 checkpoint")
    require_file(args.blender_executable, "Blender executable")
    required_scripts = [
        SINGLE_IMAGE_RUNNER,
        SAM3_TRACKER,
        CAMERA_TRACKER,
        CAMERA_TRACKER_DA3,
        ACTOR_TRACKER,
        ACTOR_TRACKER_FLOOR,
    ]
    if not args.no_blender:
        required_scripts.append(BLENDER_IMPORTER)
    for script in required_scripts:
        require_file(script, "Pipeline script")
    if args.camera_tracker == "da3":
        require_file(args.da3_python, "DA3 Python executable")
        require_file(args.da3_model / "model.safetensors", "DA3 model weights")
        require_file(args.da3_source / "depth_anything_3/api.py", "DA3 source tree")
    if args.preview:
        require_file(CLIP_PREVIEW, "Clip preview script")

    output_dir = (
        args.output_dir
        or REPO_ROOT / "outputs/Friends-monica_apartment/clip_runs" / video.stem
    ).resolve()
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "clip_manifest.json"
    previous_durations: dict[str, float] = {}
    if args.resume and manifest_path.is_file():
        try:
            previous_manifest = json.loads(manifest_path.read_text())
            previous_durations = previous_manifest.get("stage_durations_seconds", {})
        except (json.JSONDecodeError, OSError):
            previous_durations = {}

    frames_dir = output_dir / "frames"
    frames_metadata_path = output_dir / "frames.json"
    if args.resume and frames_metadata_path.is_file() and list(frames_dir.glob("*.jpg")):
        frames_metadata = json.loads(frames_metadata_path.read_text())
        if int(frames_metadata.get("max_frames", 0)) != args.max_frames:
            raise ValueError(
                "Existing frame selection does not match --max-frames; use --overwrite "
                "or a different output directory."
            )
    else:
        measure(
            "extract_frames",
            durations,
            lambda: extract_frames(video, frames_dir, args.jpeg_quality, args.max_frames),
        )
        frames_metadata = json.loads(frames_metadata_path.read_text())

    masks_dir = output_dir / "masks"
    person_tracks = masks_dir / "person_tracks.json"
    if not (args.resume and person_tracks.is_file()):
        measure(
            "sam3_video_tracking",
            durations,
            lambda: run(
                [
                    str(args.python_executable), str(SAM3_TRACKER),
                    "--frames-dir", str(frames_dir),
                    "--output-dir", str(masks_dir),
                    "--sam3-root", str(args.sam3_root),
                    "--checkpoint", str(args.sam3_checkpoint),
                    "--version", args.sam3_version,
                    "--prob-threshold", str(args.sam3_prob_threshold),
                    "--dilate-pixels", str(args.sam3_dilate_pixels),
                ]
            ),
        )

    first_frame = frames_dir / "000000.jpg"
    first_mask = masks_dir / "union/000000.png"
    anchor_dir = output_dir / "anchor"
    anchor_pose = anchor_dir / "localization_colmap.json"
    localization_sources = [SINGLE_IMAGE_RUNNER, REPO_ROOT / "tools/localize_query_from_disk.py",
                            REPO_ROOT / "tools/pnp_focal.py", REPO_ROOT / "tools/pose_quality.py"]
    if not (args.resume and up_to_date(anchor_pose, first_frame, first_mask, *localization_sources)):
        command = [
            str(args.python_executable), str(SINGLE_IMAGE_RUNNER),
            "--query-image", str(first_frame),
            "--query-mask", str(first_mask),
            "--person-mask", "off",
            "--environment-dir", str(environment),
            "--blender-input", str(blender_input),
            "--output-dir", str(anchor_dir),
            "--blender-executable", str(args.blender_executable),
            "--python-executable", str(args.python_executable),
            "--camera-name", args.camera_name,
            "--top-k", str(args.anchor_top_k),
            "--ratio", str(args.anchor_ratio),
            "--no-preview",
        ]
        if args.focal_pixels.strip():
            command.extend(("--focal-pixels", args.focal_pixels))
        measure("first_frame_absolute_pnp", durations, lambda: run(command))

    last_frame_index = int(frames_metadata["frame_count"]) - 1
    last_frame = frames_dir / f"{last_frame_index:06d}.jpg"
    last_mask = masks_dir / f"union/{last_frame_index:06d}.png"
    tail_anchor_dir = output_dir / "tail_anchor"
    tail_anchor_pose = tail_anchor_dir / "localization_colmap.json"
    if args.camera_tracker == "da3" and not (
        args.resume and up_to_date(tail_anchor_pose, last_frame, last_mask, *localization_sources)
    ):
        command = [
            str(args.python_executable), str(SINGLE_IMAGE_RUNNER),
            "--query-image", str(last_frame),
            "--query-mask", str(last_mask),
            "--person-mask", "off",
            "--environment-dir", str(environment),
            "--blender-input", str(blender_input),
            "--output-dir", str(tail_anchor_dir),
            "--blender-executable", str(args.blender_executable),
            "--python-executable", str(args.python_executable),
            "--camera-name", args.camera_name,
            "--top-k", str(args.anchor_top_k),
            "--ratio", str(args.anchor_ratio),
            "--no-preview",
        ]
        if args.focal_pixels.strip():
            command.extend(("--focal-pixels", args.focal_pixels))
        measure("last_frame_absolute_pnp", durations, lambda: run(command))

    first_anchor_quality = anchor_quality(anchor_pose, 0)
    allowed_anchor_statuses = ("accepted", "review") if args.allow_review_anchors else ("accepted",)
    if first_anchor_quality["status"] not in allowed_anchor_statuses:
        raise RuntimeError(
            "First-frame PnP failed the pose-quality gate: "
            f"{first_anchor_quality}"
        )
    tail_quality = None
    if args.camera_tracker == "da3":
        tail_quality = anchor_quality(tail_anchor_pose, last_frame_index)
        if tail_quality["status"] not in allowed_anchor_statuses:
            raise RuntimeError(
                "Last-frame PnP failed the pose-quality gate: "
                f"{tail_quality}"
            )

    camera_track = output_dir / "camera_track.json"
    camera_inputs = [anchor_pose]
    if args.camera_tracker == "da3":
        camera_inputs.extend((tail_anchor_pose, CAMERA_TRACKER_DA3))
    else:
        camera_inputs.append(CAMERA_TRACKER)
    if not (
        args.resume
        and camera_track_matches(camera_track, args.camera_tracker)
        and up_to_date(camera_track, *camera_inputs)
    ):
        if args.camera_tracker == "da3":
            camera_command = [
                "env",
                f"PYTHONPATH={args.da3_source.resolve()}",
                f"HF_HOME={(REPO_ROOT / '.cache/huggingface').resolve()}",
                f"XDG_CACHE_HOME={(REPO_ROOT / '.cache').resolve()}",
                str(args.da3_python.resolve()),
                str(CAMERA_TRACKER_DA3),
                "--frames-dir", str(frames_dir),
                "--mask-dir", str(masks_dir / "union"),
                "--anchor-localization", str(anchor_pose),
                "--tail-anchor-localization", str(tail_anchor_pose),
                "--environment-dir", str(environment),
                "--model", str(args.da3_model.resolve()),
                "--device", args.da3_device,
                "--process-res", str(args.da3_process_res),
                "--smoothing-mode", args.da3_smoothing_mode,
                "--smooth-sigma", str(args.da3_smooth_sigma),
                "--smooth-window", str(args.da3_smooth_window),
                "--smooth-polyorder", str(args.da3_smooth_polyorder),
                "--output", str(camera_track),
            ]
            if args.allow_review_anchors:
                camera_command.append("--allow-review-anchors")
            raw_prediction = output_dir / "da3_prediction.npz"
            if args.resume and raw_prediction.is_file():
                camera_command.extend(("--raw-prediction", str(raw_prediction)))
            else:
                camera_command.extend(("--raw-output", str(raw_prediction)))
        else:
            camera_command = [
                str(args.python_executable), str(CAMERA_TRACKER),
                "--frames-dir", str(frames_dir),
                "--mask-dir", str(masks_dir / "union"),
                "--anchor-localization", str(anchor_pose),
                "--output", str(camera_track),
            ]
        measure(
            "clip_camera_tracking",
            durations,
            lambda: run(camera_command),
        )

    actor_tracks = output_dir / "actor_tracks.json"
    actor_estimator = ACTOR_TRACKER if args.camera_tracker == "da3" else ACTOR_TRACKER_FLOOR
    actor_inputs = [person_tracks, camera_track, actor_estimator]
    if args.camera_tracker == "da3":
        actor_inputs.append(output_dir / "da3_prediction.npz")
    if not (args.resume and up_to_date(actor_tracks, *actor_inputs)):
        actor_command = [
            str(args.python_executable), str(actor_estimator),
            "--person-tracks", str(person_tracks),
            "--camera-track", str(camera_track),
        ]
        if args.camera_tracker == "da3":
            actor_command.extend(
                ("--depth-prediction", str(output_dir / "da3_prediction.npz"))
            )
        else:
            actor_command.extend(("--environment-dir", str(environment)))
        actor_command.extend(("--output", str(actor_tracks)))
        if args.camera_tracker == "da3" and args.actor_world_up is not None:
            actor_command.extend(("--world-up", *map(str, args.actor_world_up)))
        if args.camera_tracker == "da3":
            actor_command.extend(
                ("--surface-to-center-offset", str(args.actor_depth_offset))
            )
        measure(
            "coarse_actor_tracking",
            durations,
            lambda: run(actor_command),
        )

    render_job_path = output_dir / "render_job.json"
    render_job = write_render_job(
        output=render_job_path,
        camera_track_path=camera_track,
        actor_tracks_path=actor_tracks,
        frames_path=frames_metadata_path,
        base_blend=blender_input,
        camera_name=args.camera_name,
        hide_collections=("actor1", "camera"),
    )

    diagnostic_path = output_dir / "diagnostic_overlay.mp4"
    if not (
        args.resume
        and up_to_date(diagnostic_path, person_tracks, actor_tracks, camera_track)
    ):
        measure(
            "diagnostic_video",
            durations,
            lambda: diagnostic_video(
                frames_dir,
                person_tracks,
                actor_tracks,
                camera_track,
                float(frames_metadata["fps"]),
                diagnostic_path,
            ),
        )

    blender_output = output_dir / "scene_with_clip_animation.blend"
    blender_metadata = output_dir / "blender_animation.json"
    if not args.no_blender and not (
        args.resume
        and up_to_date(
            blender_output,
            blender_input,
            camera_track,
            actor_tracks,
            BLENDER_IMPORTER,
        )
    ):
        command = [
            str(args.blender_executable), "--background", str(blender_input),
            "--python", str(BLENDER_IMPORTER), "--",
            "--camera-track", str(camera_track),
            "--actor-tracks", str(actor_tracks),
            "--output-blend", str(blender_output),
            "--metadata", str(blender_metadata),
            "--camera-name", args.camera_name,
            "--fps", str(frames_metadata["fps"]),
            "--hide-collection", "actor1",
            "--hide-collection", "camera",
        ]
        measure(
            "blender_animation",
            durations,
            lambda: run(command, environment=blender_environment()),
        )

    preview_path = output_dir / "predicted_camera_preview.png"
    preview_metadata = output_dir / "viewport_capture.json"
    if args.preview and not (
        args.resume and up_to_date(preview_path, blender_output)
    ):
        if not blender_output.is_file():
            raise FileNotFoundError("Blender scene is required for --preview")
        preview_command = [
            "env",
            f"DISPLAY={args.viewport_display}",
            "GHOST_BACKEND=x11",
            str(args.blender_executable), str(blender_output),
            "--python", str(CLIP_PREVIEW), "--",
            "--output", str(preview_path),
            "--metadata", str(preview_metadata),
            "--camera-name", args.camera_name,
            "--frame", str(args.preview_frame),
            "--hide-collection", "actor1",
            "--hide-collection", "camera",
        ]
        measure(
            "blender_viewport_preview",
            durations,
            lambda: run(preview_command, environment=blender_environment()),
        )

    camera_summary = json.loads(camera_track.read_text())["summary"]
    people_summary = json.loads(person_tracks.read_text())
    actors_summary = json.loads(actor_tracks.read_text())
    manifest = {
        "schema_version": 1,
        "video": str(video),
        "assumption": "selected_frame_range_is_one_shot",
        "source_frame_range_inclusive": frames_metadata.get(
            "source_frame_range_inclusive", [0, int(frames_metadata["frame_count"]) - 1]
        ),
        "absolute_localization_frames": (
            [0, last_frame_index] if args.camera_tracker == "da3" else [0]
        ),
        "camera_tracker": args.camera_tracker,
        "blender_input": str(blender_input),
        "anchor_quality": first_anchor_quality,
        "tail_anchor_localization": (
            str(tail_anchor_pose) if args.camera_tracker == "da3" else None
        ),
        "tail_anchor_quality": (
            tail_quality if args.camera_tracker == "da3" else None
        ),
        "frames": str(frames_metadata_path),
        "person_tracks": str(person_tracks),
        "anchor_localization": str(anchor_pose),
        "camera_track": str(camera_track),
        "actor_tracks": str(actor_tracks),
        "render_job": str(render_job_path),
        "render_stage_deferred": bool(args.no_blender),
        "diagnostic_overlay": str(diagnostic_path),
        "blender_scene": str(blender_output) if blender_output.is_file() else None,
        "blender_metadata": str(blender_metadata) if blender_metadata.is_file() else None,
        "blender_viewport_preview": str(preview_path) if preview_path.is_file() else None,
        "frame_count": int(frames_metadata["frame_count"]),
        "fps": float(frames_metadata["fps"]),
        "sam3_track_count": len(people_summary.get("tracks", [])),
        "actor_track_count": len(actors_summary.get("tracks", [])),
        "actor_representation": actors_summary.get("representation"),
        "actor_depth_prediction": actors_summary.get("depth_prediction"),
        "actor_proxy_geometry": render_job["scene_import"]["actor_proxy"],
        "camera_summary": camera_summary,
        "stage_durations_seconds": {**previous_durations, **durations},
        "total_duration_seconds": round(time.perf_counter() - started, 3),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
