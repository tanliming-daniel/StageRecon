#!/usr/bin/env python3
"""Generate one Qwen character reference sheet per Friends gallery folder."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from qwen21_client import request, submit


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GALLERY = REPO_ROOT / "data/raw/friends_gallery"
OUTPUT_NAME = "reference_bank.png"
METADATA_NAME = "reference_bank.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
PROMPT = """基于输入参考图中的同一个人物，生成一张干净、写实、可用于后续生图的人物参考设定图。
输入参考图是美国电视剧《老友记》（Friends）的真实剧集截图。请准确识别并保持截图中对应真实演员在该集、该时期的实际长相与年龄特征，重点忠实还原演员本人独特的脸型、五官比例、眼睛、鼻子、嘴唇、发际线、发型和自然皮肤质感；不要网红化、年轻化、过度美化，也不要替换成仅仅相似的陌生人。
如果提供了图1和图2，必须把两张图视为同一个人物的互补参考，综合两图确认人物身份、脸部、发型、体型、服装、鞋子和配饰；不要把两张参考图画成两个人。
画面必须严格横向分成四列、每列只出现一个视图且互不遮挡：第一列只能是人物正面脸部超近特写，人物直视镜头，从完整发顶到下巴和少量颈部，脸部占该列画面的百分之八十以上，清晰呈现眼睛、鼻子、嘴唇、脸型、皮肤和发际线，不能出现胸部或全身；第二列是完整全身正面；第三列是完整全身九十度侧面；第四列是完整全身背面，身体与头部均背对镜头，不要四十五度侧身。
严格保持参考人物的身份特征、脸型、五官、发型发色、年龄、体型、肤色、服装、鞋子和配饰一致，不要换装。
人物表情自然，采用中性直立站姿。第二、三、四列的全身视图都必须从头顶到鞋底完整入画，四肢和手脚完整清晰，不裁切、不遮挡、不残缺、不增加肢体；第一列必须是以脸部容貌识别为目的的正脸超近景，不要头肩照。
使用纯白或浅灰色摄影棚背景、均匀柔光；不要文字、标签、水印、道具或其他人物。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY)
    parser.add_argument("--url", default=os.getenv("IMAGE_API_URL"))
    parser.add_argument("--api-key", default=os.getenv("IMAGE_API_KEY"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1088)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--only",
        action="append",
        help="Only process this gallery-relative person directory; repeat as needed",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key.strip()
    raise RuntimeError("No API key found. Set IMAGE_API_KEY or pass --api-key.")


def source_images(person_dir: Path) -> list[Path]:
    images = sorted(
        path
        for path in person_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.name != OUTPUT_NAME
    )
    if not 1 <= len(images) <= 2:
        raise RuntimeError(f"{person_dir}: expected one or two input images, found {len(images)}")
    return images


def is_valid_png(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def wait_for_task(
    base_url: str,
    api_key: str,
    task_id: str,
    poll_interval: float,
    timeout: float,
) -> dict:
    status_url = f"{base_url}/v1/tasks/{task_id}"
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() > deadline:
            raise RuntimeError(f"task {task_id} timed out after {timeout:.0f}s")
        task = json.loads(request(status_url, api_key))
        state = task.get("status")
        if state == "succeeded":
            return task
        if state == "failed":
            raise RuntimeError(task.get("error") or f"task {task_id} failed")
        time.sleep(poll_interval)


def validate_image_bytes(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
        return image.size


def make_idempotency_key(
    relative_dir: str,
    sources: list[Path],
    width: int,
    height: int,
    steps: int,
    attempt: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(relative_dir.encode())
    for source in sources:
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    digest.update(PROMPT.encode())
    digest.update(f"{width}x{height}:{steps}:attempt={attempt}".encode())
    return "friends-ref-" + digest.hexdigest()[:64]


def generate_one(
    person_dir: Path,
    gallery: Path,
    api_key: str,
    args: argparse.Namespace,
) -> dict:
    relative_dir = person_dir.relative_to(gallery).as_posix()
    sources = source_images(person_dir)
    output = person_dir / OUTPUT_NAME
    metadata = person_dir / METADATA_NAME

    if not args.overwrite and is_valid_png(output):
        return {"status": "skipped", "directory": relative_dir, "output": str(output)}

    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            key = make_idempotency_key(
                relative_dir, sources, args.width, args.height, args.steps, attempt
            )
            task_id = submit(
                args.url.rstrip("/"),
                api_key,
                "qwen",
                PROMPT,
                sources,
                key,
                {
                    "width": args.width,
                    "height": args.height,
                    "steps": args.steps,
                    "transparent": False,
                },
            )
            task = wait_for_task(
                args.url.rstrip("/"),
                api_key,
                task_id,
                args.poll_interval,
                args.timeout,
            )
            result_url = task["result_url"]
            data = request(args.url.rstrip("/") + result_url, api_key)
            generated_size = validate_image_bytes(data)

            temp_path = person_dir / f".{OUTPUT_NAME}.{uuid.uuid4().hex}.tmp"
            temp_path.write_bytes(data)
            temp_path.replace(output)

            record = {
                "status": "generated",
                "directory": relative_dir,
                "source_images": [source.name for source in sources],
                "source_image_count": len(sources),
                "output_image": OUTPUT_NAME,
                "prompt": PROMPT,
                "requested_size": [args.width, args.height],
                "generated_size": list(generated_size),
                "steps": args.steps,
                "api_url": args.url.rstrip("/"),
                "task_id": task_id,
                "task_parameters": task.get("parameters", {}),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "attempt": attempt,
            }
            metadata.write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return record
        except Exception as exc:
            last_error = exc
            if attempt < args.retries:
                time.sleep(min(5 * attempt, 15))
    raise RuntimeError(f"{relative_dir}: {last_error}")


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.width % 32 or args.height % 32:
        raise SystemExit("--width and --height must be multiples of 32")
    if args.width * args.height > 4_200_000:
        raise SystemExit("requested image exceeds the API's 4,200,000-pixel limit")

    gallery = args.gallery.resolve()
    person_dirs = sorted(gallery.glob("S*/person_*"))
    if args.only:
        requested = set(args.only)
        person_dirs = [
            directory
            for directory in person_dirs
            if directory.relative_to(gallery).as_posix() in requested
        ]
        missing = requested - {
            directory.relative_to(gallery).as_posix() for directory in person_dirs
        }
        if missing:
            raise SystemExit("Unknown --only directory: " + ", ".join(sorted(missing)))
    if args.limit is not None:
        person_dirs = person_dirs[: args.limit]

    errors = []
    for person_dir in person_dirs:
        try:
            source_images(person_dir)
        except Exception as exc:
            errors.append(str(exc))
    if errors:
        raise SystemExit("Input validation failed:\n" + "\n".join(errors))

    pending = [
        directory
        for directory in person_dirs
        if args.overwrite or not is_valid_png(directory / OUTPUT_NAME)
    ]
    print(
        f"Found {len(person_dirs)} directories: {len(pending)} pending, "
        f"{len(person_dirs) - len(pending)} already complete.",
        flush=True,
    )
    if args.dry_run or not pending:
        return 0

    if not args.url:
        raise SystemExit("Set IMAGE_API_URL or pass --url.")
    api_key = resolve_api_key(args)
    counters = {"generated": 0, "skipped": 0, "failed": 0}
    counter_lock = threading.Lock()
    log_lock = threading.Lock()
    log_path = gallery / "reference_bank_generation.jsonl"

    def append_log(record: dict) -> None:
        with log_lock:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
        futures = {
            executor.submit(generate_one, directory, gallery, api_key, args): directory
            for directory in pending
        }
        for future in as_completed(futures):
            directory = futures[future]
            relative_dir = directory.relative_to(gallery).as_posix()
            try:
                record = future.result()
                state = record["status"]
                with counter_lock:
                    counters[state] += 1
                append_log(record)
                print(
                    f"[{sum(counters.values())}/{len(pending)}] {state}: {relative_dir}",
                    flush=True,
                )
            except Exception as exc:
                record = {
                    "status": "failed",
                    "directory": relative_dir,
                    "error": str(exc),
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                }
                with counter_lock:
                    counters["failed"] += 1
                append_log(record)
                print(
                    f"[{sum(counters.values())}/{len(pending)}] failed: {relative_dir}: {exc}",
                    flush=True,
                )

    print("Summary: " + json.dumps(counters, ensure_ascii=False), flush=True)
    return 1 if counters["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
