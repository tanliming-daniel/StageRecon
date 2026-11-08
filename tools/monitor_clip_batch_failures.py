#!/usr/bin/env python3
"""Persist failure history from live Clip-to-Blender batch manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any


FAILED_LINE = re.compile(r"\[\d+/\d+\] (?P<clip>.+?\.mp4): failed elapsed=")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--console-log", action="append", default=[])
    parser.add_argument("--interval", type=float, default=15.0)
    return parser.parse_args()


def fingerprint(record: dict) -> str:
    payload = "\0".join(
        str(record.get(key, ""))
        for key in ("manifest", "clip_key", "failed_stage", "error", "traceback")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def load_history(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            value.setdefault("fingerprint", fingerprint(value))
            records.append(value)
            seen.add(value["fingerprint"])
    return records, seen


def append_record(path: Path, record: dict[str, Any], seen: set[str]) -> bool:
    record.setdefault("recorded_at_unix", time.time())
    record["fingerprint"] = fingerprint(record)
    if record["fingerprint"] in seen:
        return False
    with path.open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    seen.add(record["fingerprint"])
    return True


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    history = root / "failed_clips_history.jsonl"
    latest_path = root / "failed_clips_latest.json"
    index_path = root / "failed_clips_index.json"
    manifests = [root / name for name in args.manifest]
    console_logs = [root / name for name in args.console_log]
    console_offsets = {path: 0 for path in console_logs}
    _, seen = load_history(history)
    while True:
        latest = []
        current_statuses: dict[str, str] = {}
        for path in manifests:
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            for clip_key, item in data.get("clips", {}).items():
                current_statuses[clip_key] = item.get("status", "unknown")
                if item.get("status") != "failed":
                    continue
                record = {
                    "recorded_at_unix": time.time(),
                    "manifest": path.name,
                    "clip_key": clip_key,
                    "video": item.get("video"),
                    "output_dir": item.get("output_dir"),
                    "failed_stage": item.get("failed_stage"),
                    "error": item.get("error"),
                    "traceback": item.get("traceback"),
                    "stage1_seconds": item.get("stage1_seconds"),
                    "stage2_seconds": item.get("stage2_seconds"),
                }
                record["fingerprint"] = fingerprint(record)
                latest.append(record)
                append_record(history, record, seen)

        for path in console_logs:
            if not path.is_file():
                continue
            size = path.stat().st_size
            if console_offsets[path] > size:
                console_offsets[path] = 0
            with path.open(errors="replace") as stream:
                stream.seek(console_offsets[path])
                for line in stream:
                    match = FAILED_LINE.search(line)
                    if not match:
                        continue
                    append_record(
                        history,
                        {
                            "source": "console_log",
                            "console_log": path.name,
                            "clip_key": match.group("clip"),
                            "failed_stage": None,
                            "error": "failed in a batch attempt; see worker logs for details",
                        },
                        seen,
                    )
                console_offsets[path] = stream.tell()

        records, disk_seen = load_history(history)
        seen.update(disk_seen)
        by_clip: dict[str, dict[str, Any]] = {}
        for record in records:
            clip_key = record.get("clip_key")
            if not clip_key:
                continue
            timestamp = float(record.get("recorded_at_unix", 0.0) or 0.0)
            item = by_clip.setdefault(
                clip_key,
                {
                    "clip_key": clip_key,
                    "first_recorded_at_unix": timestamp,
                    "last_recorded_at_unix": timestamp,
                    "failure_record_count": 0,
                    "latest_error": None,
                },
            )
            item["first_recorded_at_unix"] = min(item["first_recorded_at_unix"], timestamp)
            item["last_recorded_at_unix"] = max(item["last_recorded_at_unix"], timestamp)
            item["failure_record_count"] += 1
            if timestamp >= item["last_recorded_at_unix"] and record.get("error"):
                item["latest_error"] = record["error"]
        for clip_key, item in by_clip.items():
            item["current_status"] = current_statuses.get(clip_key, "unknown")
            item["resolved"] = item["current_status"] == "completed"

        write_json_atomic(
            latest_path,
            {
                "updated_at_unix": time.time(),
                "failed_clip_count": len(latest),
                "failed_clips": latest,
            },
        )
        write_json_atomic(
            index_path,
            {
                "updated_at_unix": time.time(),
                "unique_failed_clip_count": len(by_clip),
                "currently_failed_clip_count": sum(
                    item["current_status"] == "failed" for item in by_clip.values()
                ),
                "failed_clips": sorted(by_clip.values(), key=lambda item: item["clip_key"]),
            },
        )
        time.sleep(max(args.interval, 1.0))


if __name__ == "__main__":
    main()
