#!/usr/bin/env python3
"""Probe Alibaba Bailian Qwen3.8-Omni-Flash with local video files."""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-url",
        default="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",
    )
    parser.add_argument("--model", default="qwen3.8-omni-flash")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    parser.add_argument("--reasoning-effort", default="none")
    return parser.parse_args()


def video_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:;base64,{encoded}"


def run_case(
    *, base_url: str, api_key: str, timeout_sec: float,
    case: dict[str, Any], model: str, reasoning_effort: str
) -> dict[str, Any]:
    video_path = Path(case["video"]).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    record: dict[str, Any] = {
        "case_id": case["case_id"],
        "video": str(video_path),
        "video_bytes": video_path.stat().st_size,
        "prompt": case["prompt"],
        "expected": case.get("expected"),
    }
    started = time.perf_counter()
    answer: list[str] = []
    reasoning: list[str] = []
    usage = None
    response_model = None
    try:
        body = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_data_uri(video_path)}},
                    {"type": "text", "text": str(case["prompt"])},
                ],
            }],
            "modalities": ["text"],
            "reasoning_effort": reasoning_effort,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        endpoint = base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload_text = line[5:].strip()
                if payload_text == "[DONE]":
                    break
                chunk = json.loads(payload_text)
                response_model = chunk.get("model", response_model)
                usage = chunk.get("usage") or usage
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    answer.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
        record.update({
            "status": "ok",
            "latency_sec": round(time.perf_counter() - started, 3),
            "response_model": response_model,
            "usage": usage,
            "text": "".join(answer),
            "reasoning_content": "".join(reasoning),
        })
    except urllib.error.HTTPError as error:
        record.update({
            "status": "http_error",
            "latency_sec": round(time.perf_counter() - started, 3),
            "http_status": error.code,
            "error": error.read().decode("utf-8", errors="replace")[:4000],
        })
    except Exception as error:
        record.update({
            "status": "error",
            "latency_sec": round(time.perf_counter() - started, 3),
            "error_type": type(error).__name__,
            "error": str(error),
        })
    return record


def main() -> int:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"Missing environment variable: {args.api_key_env}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = []
    for case in manifest["cases"]:
        record = run_case(
            base_url=args.base_url, api_key=api_key, timeout_sec=args.timeout_sec,
            case=case, model=args.model, reasoning_effort=args.reasoning_effort,
        )
        records.append(record)
        print(json.dumps({
            "case_id": record["case_id"],
            "status": record["status"],
            "latency_sec": record["latency_sec"],
        }, ensure_ascii=False), flush=True)
    report = {
        "schema_version": "qwen38_omni_video_probe_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "requested_model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
