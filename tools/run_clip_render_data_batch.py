#!/usr/bin/env python3
"""Batch-export portable Clip-to-Blender data without launching Blender."""

from __future__ import annotations

import argparse
import json
import os
import selectors
import shutil
import subprocess
import sys
import time
from pathlib import Path
from batch_protocol import portable_bundle_is_valid
from omega_review import sha256, read, copy_bundle, scene_signature
from actor_proxy_config import resolve_actor_world_up


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIPS_ROOT = REPO_ROOT / "data/clips/Friends_S08_PublicHD_OmniShotCut_v1_5"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/Friends-monica_apartment/clip_render_data_s08"
DEFAULT_ENVIRONMENT = (
    REPO_ROOT / "sitcoms3D/data/sparse_reconstruction_and_nerf_data/Friends-monica_apartment"
)
DEFAULT_BLEND = REPO_ROOT / "data/scene_with_clip_animation.blend"
DEFAULT_OMEGA = Path("/root/userdata/300workspace/merg3r_work/omega_20261119_reviewed172_ba_regularized")
DEFAULT_OMEGA_IMAGES = REPO_ROOT / "data/sparse_reconstruction_and_image_data/Friends-monica_apartment/images"
DEFAULT_SAM3_PYTHON = Path("/root/userdata/300workspace/sm3r/bin/python")
DEFAULT_DA3_PYTHON = Path("/root/userdata/300workspace/da3/bin/python")
DEFAULT_SAM3_ROOT = Path("/root/userdata/300workspace/sam3")
DEFAULT_SAM3_CHECKPOINT = Path("/root/userdata/300workspace/ckpt/sam3/sam3.pt")
DEFAULT_DISK_CHECKPOINT = (
    REPO_ROOT / "outputs/Friends-monica_apartment/new_image_localization/models/depth-save.pth"
)
DEFAULT_GLOBAL_INDEX = (
    REPO_ROOT / "outputs/Friends-monica_apartment/models/disk_global_retrieval_index.npz"
)
DEFAULT_DA3_MODEL = Path("/root/userdata/300workspace/zzf/server/Depth-Anything-3/DA3-LARGE-1.1")
STAGE1_WORKER = REPO_ROOT / "tools/clip_data_stage1_worker.py"
STAGE2_WORKER = REPO_ROOT / "tools/clip_data_stage2_worker.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-root", type=Path, default=DEFAULT_CLIPS_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-name", default="main")
    parser.add_argument("--manifest-name", default="batch_manifest.json")
    parser.add_argument("--environment-dir", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--base-blend", type=Path)
    parser.add_argument("--camera-backend", choices=("omega", "colmap"), default="colmap")
    parser.add_argument("--omega-reconstruction", type=Path, default=DEFAULT_OMEGA)
    parser.add_argument("--omega-scene-config", type=Path, help="Scene profile for this reconstruction; default: its batch_scene.json")
    parser.add_argument("--omega-images", type=Path)
    parser.add_argument("--experimental-colmap-registration", action="store_true", help="Use the fitted static-point similarity instead of the scene asset alignment")
    parser.add_argument("--omega-feature-index", type=Path)
    parser.add_argument("--sam3-python", type=Path, default=DEFAULT_SAM3_PYTHON)
    parser.add_argument("--da3-python", type=Path, default=DEFAULT_DA3_PYTHON)
    parser.add_argument("--sam3-root", type=Path, default=DEFAULT_SAM3_ROOT)
    parser.add_argument("--sam3-checkpoint", type=Path, default=DEFAULT_SAM3_CHECKPOINT)
    parser.add_argument("--disk-checkpoint", type=Path, default=DEFAULT_DISK_CHECKPOINT)
    parser.add_argument("--global-index", type=Path, default=DEFAULT_GLOBAL_INDEX)
    parser.add_argument("--da3-model", type=Path, default=DEFAULT_DA3_MODEL)
    parser.add_argument("--actor-world-up", type=float, nargs=3)
    parser.add_argument("--anchor-source", choices=("registered-first", "estimated"))
    parser.add_argument("--registered-images-dir", type=Path, action="append", default=[])
    parser.add_argument("--stage1-gpu", default="4")
    parser.add_argument("--stage2-gpu", default="4")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--case-list",type=Path,help="Ordered JSON array of clip paths or records with a clip field")
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--include", default="", help="Only paths containing this text")
    parser.add_argument("--dry-run", action="store_true", help="Print the execution plan without starting model workers.")
    parser.add_argument("--use-reviewed", action=argparse.BooleanOptionalAction, default=True, help="Use published results for matching source clips and scene")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failed", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def valid_bundle(directory: Path) -> bool:
    return portable_bundle_is_valid(directory)


def clip_output(video: Path, clips_root: Path, output_root: Path) -> Path:
    relative = video.relative_to(clips_root)
    episode = relative.parts[0]
    return output_root / episode / video.stem


def read_manifest(path: Path) -> dict:
    if not path.is_file():
        return {"schema_version": 1, "clips": {}}
    return json.loads(path.read_text())


def write_manifest(path: Path, manifest: dict) -> None:
    clips = manifest["clips"]
    statuses: dict[str, int] = {}
    for item in clips.values():
        status = item.get("status", "unknown")
        statuses[status] = statuses.get(status, 0) + 1
    manifest["updated_at_unix"] = time.time()
    manifest["summary"] = {"clip_count": len(clips), "statuses": statuses}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def append_failure(path: Path, message: dict, failed_stage: int, run_name: str) -> None:
    record = {
        "recorded_at_unix": time.time(),
        "run_name": run_name,
        "clip_key": message.get("clip_key"),
        "video": message.get("video"),
        "output_dir": message.get("output_dir"),
        "failed_stage": failed_stage,
        "error": message.get("error"),
        "traceback": message.get("traceback"),
        "stage1_seconds": message.get("stage1_seconds"),
        "stage2_seconds": message.get("stage2_seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def spawn(command: list[str], gpu: str, cache_root: Path, log_path: Path) -> subprocess.Popen:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / f"torchinductor_gpu{gpu}")
    environment["TMPDIR"] = str(cache_root / "tmp")
    environment["TEMP"] = environment["TMPDIR"]
    environment["TMP"] = environment["TMPDIR"]
    Path(environment["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(environment["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", buffering=1)
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=log,
        text=True,
        bufsize=1,
        env=environment,
        cwd=str(REPO_ROOT),
    )


def send(process: subprocess.Popen, payload: dict) -> None:
    if process.stdin is None:
        raise RuntimeError("Worker stdin is closed")
    process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
    process.stdin.flush()


def read_line(process: subprocess.Popen, label: str) -> dict:
    if process.stdout is None:
        raise RuntimeError(f"{label} stdout is closed")
    while True:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"{label} exited before responding (code={process.poll()})")
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            # Some native/model libraries write directly to fd=1 and bypass
            # Python's redirected stdout. Treat those lines as worker logs.
            print(f"[{label}] {line.rstrip()}", file=sys.stderr, flush=True)
            continue
        if isinstance(value, dict) and "type" in value:
            return value
        print(f"[{label}] {line.rstrip()}", file=sys.stderr, flush=True)


def main() -> None:
    args = parse_args()
    use_omega = args.camera_backend == "omega"
    omega_pointcloud = None
    omega_units_per_meter = 1.0
    review_index = None
    reviewed_jobs = []
    if args.omega_feature_index is not None:
        if not use_omega:
            raise ValueError("--omega-feature-index requires the Omega camera backend")
        if not args.omega_feature_index.is_file():
            raise FileNotFoundError(args.omega_feature_index)
        if not args.disk_checkpoint.is_file():
            raise FileNotFoundError(args.disk_checkpoint)
    if use_omega and args.anchor_source not in (None, "estimated"):
        raise ValueError("Omega requires --anchor-source estimated")
    args.anchor_source = "estimated" if use_omega else args.anchor_source or "registered-first"
    omega = args.omega_reconstruction.resolve()
    scene_path = (args.omega_scene_config or omega / "batch_scene.json").resolve()
    scene = read(scene_path)
    args.omega_images = args.omega_images or (omega / scene["reference_images"] if scene.get("reference_images") else DEFAULT_OMEGA_IMAGES)
    if scene["coordinate_system"] != "omega_ba_world":
        raise ValueError("Invalid Omega batch scene coordinate system")
    colmap_alignment_path = REPO_ROOT / "configs/scenes/Friends-monica_apartment.json"
    registration = None
    if args.experimental_colmap_registration:
        if use_omega:
            raise ValueError("Experimental COLMAP registration requires --camera-backend colmap")
        registration = read(colmap_alignment_path)["registered_omega_alignment"]
        if (registration["pointcloud_sha256"] != scene["pointcloud_sha256"]
                or registration["images_sha256"] != sha256(args.environment_dir / "colmap/images.bin")):
            raise ValueError("COLMAP and Omega models differ from the calibrated scene pair")
    args.base_blend = args.base_blend or omega / scene["blend"]
    args.output_root = args.output_root or DEFAULT_OUTPUT_ROOT.with_name(DEFAULT_OUTPUT_ROOT.name + ("_colmap_registered_v2_shoulders_p075_" if registration else f"_{args.camera_backend}_static_camera_shoulders_p075_") + scene["source_blend_sha256"][:8])
    omega_pointcloud = omega / scene["pointcloud"]
    if scene.get("predictions_sha256") and sha256(omega/"predictions.npz")!=scene["predictions_sha256"]:
        raise ValueError("Omega predictions differ from the registered reconstruction")
    if sha256(omega_pointcloud)!=scene["pointcloud_sha256"]:
        raise ValueError("Point cloud differs from the registered Omega scene")
    if sha256(args.base_blend)!=scene["source_blend_sha256"]:
        raise ValueError("Base blend differs from the registered Omega scene; prepare a new scene profile")
    if (use_omega and args.use_reviewed and scene.get("review_index") and args.actor_world_up is None
            and args.da3_model.resolve()==DEFAULT_DA3_MODEL.resolve()
            and args.sam3_checkpoint.resolve()==DEFAULT_SAM3_CHECKPOINT.resolve()
            and args.omega_images.resolve()==(omega / scene.get("reference_images", args.omega_images)).resolve()):
        index_path = omega / scene["review_index"]
        review_index = read(index_path)
        if (review_index["profile_sha256"]!=scene_signature(scene) or
                review_index["scene_sha256"]!=scene["source_blend_sha256"] or
                review_index["pointcloud_sha256"]!=scene["pointcloud_sha256"]):
            raise ValueError("Reviewed bundle index does not match the selected scene")
    omega_units_per_meter = float(scene["omega_units_per_meter"])
    if not (0 < omega_units_per_meter < float("inf")):
        raise ValueError("Invalid Omega scene units per meter")
    args.actor_world_up = args.actor_world_up or scene["world_up"]
    for path in (omega_pointcloud, omega / "predictions.npz", args.base_blend):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.omega_images.is_dir():
        raise FileNotFoundError(args.omega_images)
    args.actor_world_up = resolve_actor_world_up(args.environment_dir, args.base_blend, args.actor_world_up, REPO_ROOT / "configs/scenes")
    clips_root = args.clips_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = output_root / "_runtime_cache" / args.run_name
    log_root = output_root / "_logs" / args.run_name
    cache_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / args.manifest_name
    failure_history_path = output_root / "failed_clips_history.jsonl"
    manifest = read_manifest(manifest_path)

    if args.case_list:
        selected = read(args.case_list)
        videos = [clips_root / (c if isinstance(c,str) else c["clip"]) for c in selected]
        for video in videos:
            if not video.is_file():raise FileNotFoundError(video)
    else:
        videos = sorted(clips_root.rglob("*.mp4"))
    if args.episode:
        episodes = set(args.episode)
        videos = [video for video in videos if video.relative_to(clips_root).parts[0] in episodes]
    if args.include:
        videos = [video for video in videos if args.include in str(video.relative_to(clips_root))]
    if args.limit > 0:
        videos = videos[: args.limit]

    pending = []
    for video in videos:
        relative = str(video.relative_to(clips_root))
        output_dir = clip_output(video, clips_root, output_root)
        current = manifest["clips"].get(relative, {})
        entry = review_index["clips"].get(Path(relative).as_posix()) if review_index else None
        if entry:
            video_hash = sha256(video)
            if video_hash == entry["video_sha256"]:
                approved = index_path.parent / entry["bundle"]
                if not valid_bundle(approved) or sha256(approved/"render_job.json") != entry["job_sha256"]:
                    raise ValueError(f"Invalid published review: {approved}")
                reviewed_jobs.append(relative)
                if not args.dry_run:
                    copy_bundle(approved, output_dir, args.overwrite)
                    manifest["clips"][relative] = dict(status="completed",source="published_review",
                        video=str(video.resolve()),output_dir=str(output_dir.resolve()),video_sha256=video_hash)
                continue
            if valid_bundle(output_dir) and not args.overwrite:
                raise ValueError(f"Video changed since review: {video}; use a fresh output root or --overwrite")
        if valid_bundle(output_dir):
            prior = json.loads((output_dir / "render_job.json").read_text())
            prior_omega = prior["coordinate_system"]["camera"] == "omega_ba_world"
            if not prior_omega:
                raise ValueError(f"Bundle uses different scene coordinates: {output_dir}; select another --output-root")
            if not args.overwrite:
                saved_camera = read(output_dir / "camera_track.json")
                if saved_camera.get("camera_backend", "omega") != args.camera_backend:
                    raise ValueError(f"Existing bundle uses another camera backend: {output_dir}")
                if registration and saved_camera.get("colmap_to_omega") != registration["colmap_to_omega"]:
                    raise ValueError(f"Existing bundle uses another COLMAP alignment: {output_dir}")
                if prior.get("source_scene", {}).get("sha256") != scene["source_blend_sha256"]:
                    raise ValueError(f"Existing bundle uses a different base scene: {output_dir}")
                method = json.loads((output_dir / "actor_tracks.json").read_text()).get("animation_method")
                if method != "omega_sam3d_shoulders_v2":
                    raise ValueError(f"Existing bundle predates actor keyframes: {output_dir}; select another --output-root")
        if not args.use_reviewed and current.get("source")=="published_review" and not args.overwrite:
            raise ValueError("Use a fresh output root or --overwrite to recompute a published review")
        if valid_bundle(output_dir) and not args.overwrite:
            manifest["clips"][relative] = {
                **current,
                "status": "completed",
                "video": str(video.resolve()),
                "output_dir": str(output_dir.resolve()),
            }
            continue
        if current.get("status") == "failed" and not args.retry_failed and not args.overwrite:
            continue
        pending.append(
            {"clip_key": relative, "video": str(video.resolve()), "output_dir": str(output_dir.resolve())}
        )

    if args.dry_run:
        print(json.dumps({"dry_run": True, "reviewed_jobs": reviewed_jobs, "selected_clips": len(videos),
                          "anchor_source": args.anchor_source, "camera_backend": args.camera_backend,
                          "base_blend": str(args.base_blend),
                          "stage1_gpu": args.stage1_gpu, "stage2_gpu": args.stage2_gpu,
                          "omega_feature_index": str(args.omega_feature_index.resolve()) if args.omega_feature_index else None,
                          "actor_world_up": args.actor_world_up, "jobs": pending}, indent=2))
        return

    write_manifest(manifest_path, manifest)
    print(
        json.dumps(
            {
                "selected_clips": len(videos),
                "pending_clips": len(pending),
                "output_root": str(output_root),
                "manifest": str(manifest_path),
            },
            indent=2,
        ),
        flush=True,
    )
    if not pending:
        return

    stage1 = spawn(
        [
            str(args.sam3_python.resolve()), str(STAGE1_WORKER),
            "--environment-dir", str(args.environment_dir.resolve()),
            "--anchor-source", args.anchor_source,
            *[item for directory in args.registered_images_dir for item in ("--registered-images-dir", str(directory.resolve()))],
            "--sam3-root", str(args.sam3_root.resolve()),
            "--sam3-checkpoint", str(args.sam3_checkpoint.resolve()),
            *(["--omega-reconstruction", str(args.omega_reconstruction.resolve()), "--omega-images", str(args.omega_images.resolve())]
              if use_omega else ["--disk-checkpoint", str(args.disk_checkpoint.resolve()), "--global-index", str(args.global_index.resolve())]),
            *(["--omega-feature-index", str(args.omega_feature_index.resolve()), "--disk-checkpoint", str(args.disk_checkpoint.resolve())]
              if use_omega and args.omega_feature_index else []),
        ],
        args.stage1_gpu,
        cache_root,
        log_root / "stage1.log",
    )
    stage2 = spawn(
        [
            str(args.da3_python.resolve()), str(STAGE2_WORKER),
            "--environment-dir", str(args.environment_dir.resolve()),
            "--da3-model", str(args.da3_model.resolve()),
            "--da3-device", "cuda:0",
            "--base-blend", str(args.base_blend.resolve()),
            "--omega-scene-config", str(scene_path),
            *(["--experimental-colmap-registration"] if registration else []),
            *(["--colmap-alignment", str(colmap_alignment_path)] if not use_omega else []),
            *(["--omega-pointcloud", str(omega_pointcloud), "--omega-units-per-meter", str(omega_units_per_meter)]
              if omega_pointcloud else []),
            *(["--actor-world-up", *[format(value, ".17f") for value in args.actor_world_up]] if args.actor_world_up is not None else []),
        ],
        args.stage2_gpu,
        cache_root,
        log_root / "stage2.log",
    )
    try:
        if read_line(stage1, "stage1").get("type") != "ready":
            raise RuntimeError("Stage1 worker did not become ready")
        if read_line(stage2, "stage2").get("type") != "ready":
            raise RuntimeError("Stage2 worker did not become ready")

        selector = selectors.DefaultSelector()
        assert stage1.stdout is not None and stage2.stdout is not None
        selector.register(stage1.stdout, selectors.EVENT_READ, "stage1")
        selector.register(stage2.stdout, selectors.EVENT_READ, "stage2")
        next_job = 0
        stage1_busy = False
        stage1_finished = False
        stage2_busy = False
        stage2_queue: list[dict] = []
        completed = 0
        failed = 0
        run_started = time.perf_counter()

        def start_stage1() -> None:
            nonlocal next_job, stage1_busy
            if stage1_busy or next_job >= len(pending):
                return
            job = pending[next_job]
            next_job += 1
            manifest["clips"][job["clip_key"]] = {
                **job, "status": "stage1_running", "started_at_unix": time.time()
            }
            write_manifest(manifest_path, manifest)
            send(stage1, job)
            stage1_busy = True

        def start_stage2() -> None:
            nonlocal stage2_busy
            if stage2_busy or not stage2_queue:
                return
            job = stage2_queue.pop(0)
            manifest["clips"][job["clip_key"]] = {**job, "status": "stage2_running"}
            write_manifest(manifest_path, manifest)
            send(stage2, job)
            stage2_busy = True

        start_stage1()
        while completed + failed < len(pending):
            events = selector.select(timeout=30.0)
            if not events:
                if (not stage1_finished and stage1.poll() is not None) or stage2.poll() is not None:
                    raise RuntimeError(
                        f"A worker exited unexpectedly: stage1={stage1.poll()} stage2={stage2.poll()}"
                    )
                print(
                    f"progress completed={completed} failed={failed} "
                    f"stage1_queue={len(pending) - next_job} stage2_queue={len(stage2_queue)}",
                    flush=True,
                )
                continue
            for key, _ in events:
                process = stage1 if key.data == "stage1" else stage2
                message = read_line(process, key.data)
                clip_key = message["clip_key"]
                if key.data == "stage1":
                    stage1_busy = False
                    if message["status"] == "ok":
                        manifest["clips"][clip_key] = {**message, "status": "stage1_completed"}
                        stage2_queue.append(message)
                    else:
                        failed += 1
                        manifest["clips"][clip_key] = {**message, "status": "failed", "failed_stage": 1}
                        append_failure(failure_history_path, message, 1, args.run_name)
                    if next_job==len(pending):
                        selector.unregister(stage1.stdout)
                        stage1.stdin.close()
                        stage1.wait()
                        stage1_finished=True
                    else:
                        start_stage1()
                    start_stage2()
                else:
                    stage2_busy = False
                    if message["status"] == "ok" and not valid_bundle(Path(message["output_dir"])):
                        message = {**message, "status": "failed", "error": "Stage2 reported success but output bundle failed validation"}
                    if message["status"] == "ok":
                        completed += 1
                        manifest["clips"][clip_key] = {
                            **message,
                            "status": "completed",
                            "completed_at_unix": time.time(),
                        }
                    else:
                        failed += 1
                        manifest["clips"][clip_key] = {**message, "status": "failed", "failed_stage": 2}
                        append_failure(failure_history_path, message, 2, args.run_name)
                    start_stage2()
                write_manifest(manifest_path, manifest)
                elapsed = time.perf_counter() - run_started
                print(
                    f"[{completed + failed}/{len(pending)}] {clip_key}: "
                    f"{manifest['clips'][clip_key]['status']} elapsed={elapsed:.1f}s",
                    flush=True,
                )
    finally:
        for process in (stage1, stage2):
            if process.stdin is not None:
                process.stdin.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
        write_manifest(manifest_path, manifest)
        if not args.limit:
            shutil.rmtree(cache_root, ignore_errors=True)


if __name__ == "__main__":
    main()
