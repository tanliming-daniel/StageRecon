#!/usr/bin/env python3
"""Long-lived DA3 worker producing portable clip render-data bundles."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import tempfile
import shutil
import sys
import time
import traceback
from batch_protocol import stage_result
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import estimate_actor_boxes  # noqa: E402
import estimate_clip_camera_da3  # noqa: E402
from clip_render_job import write_render_job  # noqa: E402
from depth_anything_3.api import DepthAnything3  # noqa: E402


def emit(payload: dict) -> None:
    sys.__stdout__.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()


def invoke_main(module, argv: list[str]) -> None:
    original = sys.argv
    sys.argv = [str(Path(module.__file__).resolve()), *argv]
    try:
        with contextlib.redirect_stdout(sys.stderr):
            module.main()
    finally:
        sys.argv = original


def sanitize_portable_bundle(output_dir: Path) -> None:
    camera_path = output_dir / "camera_track.json"
    actor_path = output_dir / "actor_tracks.json"
    frames_path = output_dir / "frames.json"

    camera = json.loads(camera_path.read_text())
    camera["anchor_localization"] = None
    camera["tail_anchor_localization"] = None
    if isinstance(camera.get("da3"), dict):
        camera["da3"]["raw_output"] = None
    camera_path.write_text(json.dumps(camera, indent=2) + "\n")

    actors = json.loads(actor_path.read_text())
    actors["camera_track"] = "camera_track.json"
    actors["person_tracks"] = None
    actors["depth_prediction"] = None
    for track in actors.get("tracks", []):
        for frame in track.get("frames", []):
            frame.pop("mask", None)
    actor_path.write_text(json.dumps(actors, indent=2) + "\n")

    frames = json.loads(frames_path.read_text())
    frames["video"] = None
    frames["portable_render_bundle"] = True
    frames_path.write_text(json.dumps(frames, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-dir", type=Path, required=True)
    parser.add_argument("--omega-pointcloud", type=Path)
    parser.add_argument("--omega-scene-config", type=Path)
    parser.add_argument("--omega-units-per-meter", type=float, default=1.0)
    parser.add_argument("--da3-model", type=Path, required=True)
    parser.add_argument("--da3-device", default="cuda:0")
    parser.add_argument("--da3-process-res", type=int, default=504)
    parser.add_argument("--da3-smoothing-mode", default="savgol")
    parser.add_argument("--da3-smooth-sigma", type=float, default=0.35)
    parser.add_argument("--da3-smooth-window", type=int, default=21)
    parser.add_argument("--da3-smooth-polyorder", type=int, default=3)
    parser.add_argument("--actor-world-up", type=float, nargs=3)
    parser.add_argument("--actor-depth-offset", type=float, default=0.36)
    parser.add_argument("--base-blend", type=Path, required=True)
    parser.add_argument("--camera-name", default="PredictedClipCamera")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    anchor_filename = "localization_omega.json" if args.omega_pointcloud else "localization_colmap.json"
    # Keep stdout machine-readable for the batch orchestrator.
    sys.stdout = sys.stderr
    print(f"[stage2] loading DA3 model {args.da3_model}", file=sys.stderr, flush=True)
    model = DepthAnything3.from_pretrained(str(args.da3_model.resolve()))
    model = model.to(args.da3_device).eval()
    original_loader = estimate_clip_camera_da3.DepthAnything3.from_pretrained
    estimate_clip_camera_da3.DepthAnything3.from_pretrained = lambda *a, **k: model
    emit({"type": "ready", "stage": 2})
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            job = json.loads(line)
            started = time.perf_counter()
            output_dir = Path(job["output_dir"]).resolve()
            raw_prediction = output_dir / "da3_prediction.npz"
            camera_path = output_dir / "camera_track.json"
            actor_path = output_dir / "actor_tracks.json"
            try:
                invoke_main(
                    estimate_clip_camera_da3,
                    [
                        "--frames-dir", str(output_dir / "frames"),
                        "--mask-dir", str(output_dir / "masks" / "union"),
                        "--anchor-localization", str(output_dir / "anchor" / anchor_filename),
                        "--tail-anchor-localization", str(output_dir / "tail_anchor" / anchor_filename),
                        "--environment-dir", str(args.environment_dir.resolve()),
                        *(["--scene-pointcloud", str(args.omega_pointcloud.resolve())] if args.omega_pointcloud else []),
                        "--model", str(args.da3_model.resolve()),
                        "--output", str(camera_path),
                        "--raw-output", str(raw_prediction),
                        "--device", args.da3_device,
                        "--process-res", str(args.da3_process_res),
                        "--ref-view-strategy", "first",
                        "--smoothing-mode", args.da3_smoothing_mode,
                        "--smooth-sigma", str(args.da3_smooth_sigma),
                        "--smooth-window", str(args.da3_smooth_window),
                        "--smooth-polyorder", str(args.da3_smooth_polyorder),
                    ],
                )
                invoke_main(
                    estimate_actor_boxes,
                    [
                        "--person-tracks", str(output_dir / "masks" / "person_tracks.json"),
                        "--camera-track", str(camera_path),
                        "--depth-prediction", str(raw_prediction),
                        "--output", str(actor_path),
                        "--surface-to-center-offset", str(args.actor_depth_offset * args.omega_units_per_meter),
                        "--proxy-height", str(1.70 * args.omega_units_per_meter),
                        "--proxy-radius", str(.18 * args.omega_units_per_meter),
                        *(["--world-up", *[format(value, ".17f") for value in args.actor_world_up]] if args.actor_world_up is not None else []),
                    ],
                )
                if args.omega_scene_config:
                    scene = json.loads(args.omega_scene_config.read_text())
                    camera = json.loads(camera_path.read_text())
                    camera["scene_frame"] = {
                        "coordinate_system": scene["coordinate_system"],
                        "world_to_blender": scene["omega_to_blender"],
                        "source_blend_filename": args.base_blend.name,
                        "source_blend_sha256": scene["source_blend_sha256"],
                        "pointcloud_sha256": scene["pointcloud_sha256"],
                        "units_per_meter": scene["omega_units_per_meter"],
                    }
                    camera["scene_frame"].update({k:scene[k] for k in ("render_objects","hide_collections") if k in scene})
                    camera_path.write_text(json.dumps(camera, indent=2) + "\n")
                    from omega_body_pose import stabilize_body
                    import torch
                    actors = json.loads(actor_path.read_text())
                    contacts = (json.loads((REPO_ROOT / scene["contacts"]).read_text()) if scene.get("contacts") else {"scene_sha256":scene["source_blend_sha256"],"seats":{},"floors":[],"occluders":[]})
                    if contacts.get("scene_sha256") != scene["source_blend_sha256"]:
                        raise ValueError("Contact geometry belongs to another Blender scene")
                    fps = float(json.loads((output_dir / "frames.json").read_text())["fps"])
                    flow = Path(os.environ.get("STAGERECON_FLOWHMR", "/root/userdata/300workspace/flowhmr"))
                    checkpoint = os.environ.get("STAGERECON_BODY_CHECKPOINT", "/root/userdata/300workspace/ckpt/sam-3d-body-dinov3/model.ckpt")
                    with tempfile.TemporaryDirectory(dir=output_dir) as tmp:
                        pose_path = Path(tmp) / "measurements.json"
                        model.to("cpu")
                        torch.cuda.empty_cache()
                        try:
                            subprocess.run([str(flow / ".venv/bin/python"), str(TOOLS_ROOT / "measure_actor_body.py"),
                                            "--input-dir", str(output_dir), "--camera", str(camera_path),
                                            "--flow", str(flow), "--checkpoint", checkpoint, "--output", str(pose_path)],
                                           check=True, stdout=sys.stderr, env={**os.environ, "PYOPENGL_PLATFORM":"egl"})
                        finally:
                            model.to(args.da3_device)
                        from actor_depth_review import correct_large_proxies
                        from omega_review import exclude_reviewed_regions
                        measurements = json.loads(pose_path.read_text())
                        exclude_reviewed_regions(actors, measurements, job, scene)
                        stabilize_body(actors, camera, contacts, Path(job["video"]).stem, fps, measurements)
                        correct_large_proxies(actors,camera,measurements,args.base_blend,scene,tmp)
                    actor_path.write_text(json.dumps(actors, indent=2) + "\n")

                sanitize_portable_bundle(output_dir)
                write_render_job(
                    output=output_dir / "render_job.json",
                    camera_track_path=camera_path,
                    actor_tracks_path=actor_path,
                    frames_path=output_dir / "frames.json",
                    base_blend=args.base_blend,
                    camera_name=args.camera_name,
                )
                for temporary in (
                    output_dir / "frames",
                    output_dir / "masks",
                    output_dir / "anchor",
                    output_dir / "tail_anchor",
                    raw_prediction,
                ):
                    if temporary.is_dir():
                        shutil.rmtree(temporary)
                    elif temporary.exists():
                        temporary.unlink()
                expected = {
                    "camera_track.json", "actor_tracks.json", "frames.json", "render_job.json"
                }
                actual = {path.name for path in output_dir.iterdir()}
                if actual != expected:
                    raise RuntimeError(f"Portable bundle has unexpected files: {sorted(actual)}")
                camera = json.loads(camera_path.read_text())
                actors = json.loads(actor_path.read_text())
                emit(stage_result(job, stage=2, status="ok",
                        frame_count=len(camera.get("frames", [])),
                        actor_track_count=len(actors.get("tracks", [])),
                        stage2_seconds=round(time.perf_counter() - started, 3),
                ))
            except Exception as error:
                shutil.rmtree(output_dir, ignore_errors=True)
                emit(stage_result(job, stage=2, status="failed",
                        error=f"{type(error).__name__}: {error}",
                        traceback=traceback.format_exc(),
                        stage2_seconds=round(time.perf_counter() - started, 3),
                ))
    finally:
        estimate_clip_camera_da3.DepthAnything3.from_pretrained = original_loader


if __name__ == "__main__":
    main()
