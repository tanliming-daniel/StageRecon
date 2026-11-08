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


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIPS_ROOT = REPO_ROOT / "data/clips/Friends_S08_PublicHD_OmniShotCut_v1_5"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/Friends-monica_apartment/clip_render_data_s08"
DEFAULT_ENVIRONMENT = (
    REPO_ROOT / "sitcoms3D/data/sparse_reconstruction_and_nerf_data/Friends-monica_apartment"
)
DEFAULT_BLEND = REPO_ROOT / "data/scene_with_clip_animation.blend"
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
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="main")
    parser.add_argument("--manifest-name", default="batch_manifest.json")
    parser.add_argument("--environment-dir", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--base-blend", type=Path, default=DEFAULT_BLEND)
    parser.add_argument("--sam3-python", type=Path, default=DEFAULT_SAM3_PYTHON)
    parser.add_argument("--da3-python", type=Path, default=DEFAULT_DA3_PYTHON)
    parser.add_argument("--sam3-root", type=Path, default=DEFAULT_SAM3_ROOT)
    parser.add_argument("--sam3-checkpoint", type=Path, default=DEFAULT_SAM3_CHECKPOINT)
    parser.add_argument("--disk-checkpoint", type=Path, default=DEFAULT_DISK_CHECKPOINT)
    parser.add_argument("--global-index", type=Path, default=DEFAULT_GLOBAL_INDEX)
    parser.add_argument("--da3-model", type=Path, default=DEFAULT_DA3_MODEL)
    parser.add_argument("--stage1-gpu", default="1")
    parser.add_argument("--stage2-gpu", default="4")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--include", default="", help="Only paths containing this text")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failed", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def valid_bundle(directory: Path) -> bool:
    expected = {
        "camera_track.json", "actor_tracks.json", "frames.json", "render_job.json"
    }
    return directory.is_dir() and {path.name for path in directory.iterdir()} == expected


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
            "--sam3-root", str(args.sam3_root.resolve()),
            "--sam3-checkpoint", str(args.sam3_checkpoint.resolve()),
            "--disk-checkpoint", str(args.disk_checkpoint.resolve()),
            "--global-index", str(args.global_index.resolve()),
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
                if stage1.poll() is not None or stage2.poll() is not None:
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
                    start_stage1()
                    start_stage2()
                else:
                    stage2_busy = False
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
