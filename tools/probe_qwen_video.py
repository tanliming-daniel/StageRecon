#!/usr/bin/env python3
"""Probe an OpenAI-compatible Qwen endpoint with local video inputs."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
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
    parser.add_argument("--base-url", default="https://models.sjtu.edu.cn/api/v1")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--api-key-env", default="QWEN_API_KEY")
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--modalities", nargs="+", default=None)
    return parser.parse_args()


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "video/mp4"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content)


def endpoint(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    return base_url if base_url.endswith("/chat/completions") else base_url + "/chat/completions"


def run_case(
    *, case: dict[str, Any], base_url: str, model: str, api_key: str,
    timeout_sec: float, max_tokens: int, reasoning_effort: str | None,
    modalities: list[str] | None,
) -> dict[str, Any]:
    video_path = Path(case["video"]).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": data_uri(video_path)}},
                    {"type": "text", "text": str(case["prompt"])},
                ],
            }
        ],
        "stream": False,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    if modalities:
        body["modalities"] = modalities
    request = urllib.request.Request(
        endpoint(base_url),
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    record: dict[str, Any] = {
        "case_id": case["case_id"],
        "video": str(video_path),
        "video_bytes": video_path.stat().st_size,
        "prompt": case["prompt"],
        "expected": case.get("expected"),
    }
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
        record.update(
            {
                "status": "ok",
                "latency_sec": round(time.perf_counter() - started, 3),
                "response_model": payload.get("model"),
                "response_id": payload.get("id"),
                "usage": payload.get("usage"),
                "text": content_text(payload["choices"][0]["message"]["content"]),
            }
        )
    except urllib.error.HTTPError as error:
        record.update(
            {
                "status": "http_error",
                "latency_sec": round(time.perf_counter() - started, 3),
                "http_status": error.code,
                "error": error.read().decode("utf-8", errors="replace")[:4000],
            }
        )
    except Exception as error:  # preserve exact boundary failures in the report
        record.update(
            {
                "status": "error",
                "latency_sec": round(time.perf_counter() - started, 3),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
    return record


def main() -> int:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise ValueError(f"Missing API key environment variable: {args.api_key_env}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = []
    for case in manifest["cases"]:
        record = run_case(
            case=case, base_url=args.base_url, model=args.model, api_key=api_key,
            timeout_sec=args.timeout_sec, max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort, modalities=args.modalities,
        )
        records.append(record)
        print(json.dumps({"case_id": record["case_id"], "status": record["status"], "latency_sec": record["latency_sec"]}, ensure_ascii=False), flush=True)
    report = {
        "schema_version": "qwen_video_probe_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": args.base_url,
        "requested_model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "modalities": args.modalities,
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "cases": len(records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
