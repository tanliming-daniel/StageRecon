#!/usr/bin/env python3
"""Run MTSS-3D captioning for a JSONL manifest with resumable workers."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stagerecon.captioning.manifest import load_clip_run
from stagerecon.captioning.pipeline import run_pipeline
from stagerecon.captioning.io import write_json


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    items = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Line {line_number} must contain a JSON object")
        items.append(value)
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--identity-backend", choices=("histogram", "insightface"))
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    items = read_jsonl(args.manifest.resolve())

    def run(item: dict[str, Any]) -> dict[str, Any]:
        clip_run = Path(item["clip_run"]).resolve()
        sample_id = str(item.get("sample_id") or clip_run.name)
        output_dir = (
            args.output_root.resolve() / sample_id
            if args.output_root
            else Path(item.get("output_dir") or clip_run / "captions/mtss_3d_v1").resolve()
        )
        audio = Path(item["audio"]).resolve() if item.get("audio") else None
        if item.get("audio_from_source"):
            audio = load_clip_run(clip_run)["source_video"]
        try:
            document, report = run_pipeline(
                clip_run=clip_run,
                previs_video=Path(item["previs_video"]),
                output_dir=output_dir,
                registry_path=(Path(item["registry"]) if item.get("registry") else args.registry),
                config_path=args.config,
                sample_id=sample_id,
                sequence_id=item.get("sequence_id"),
                target_video=Path(item["target_video"]) if item.get("target_video") else None,
                audio=audio,
                transcript_path=Path(item["transcript"]) if item.get("transcript") else None,
                manual_bindings_path=(
                    Path(item["manual_bindings"]) if item.get("manual_bindings") else None
                ),
                identity_backend=args.identity_backend,
                qwen_base_url=args.base_url,
                qwen_model=args.model,
                dry_run=args.dry_run,
                resume=args.resume,
            )
            return {
                "sample_id": sample_id,
                "output_dir": str(output_dir),
                "status": document["status"],
                "validation": report["status"],
                "error": None,
            }
        except Exception as error:
            return {
                "sample_id": sample_id,
                "output_dir": str(output_dir),
                "status": "failed",
                "validation": "failed",
                "error": f"{type(error).__name__}: {error}",
            }

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = [executor.submit(run, item) for item in items]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    results.sort(key=lambda item: item["sample_id"])
    report_path = args.manifest.with_suffix(args.manifest.suffix + ".caption_report.json")
    report = {
        "samples": len(results),
        "succeeded": sum(item["validation"] == "passed" for item in results),
        "failed": sum(item["validation"] != "passed" for item in results),
        "results": results,
    }
    write_json(report_path, report)
    return 0 if report["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
