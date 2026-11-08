#!/usr/bin/env python3
"""Submit Qwen-Image-2.1 tasks, poll status and download PNG results.

Use IMAGE_API_KEY and --url for the existing gateway. Repeat --image for editing.
"""

from __future__ import annotations

import argparse
import http.client
import json
import mimetypes
import os
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

def request(url: str, api_key: str, *, method: str = "GET", data: bytes | None = None, content_type: str | None = None) -> bytes:
    headers = {"Authorization": f"Bearer {api_key}"}
    if content_type:
        headers["Content-Type"] = content_type
    for attempt in range(4):
        try:
            with urlopen(Request(url, data=data, headers=headers, method=method), timeout=60) as response:
                payload = response.read()
            if not payload and method == "GET":
                raise URLError("empty HTTP response")
            return payload
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            if method != "GET" or exc.code not in (500, 502, 503, 504) or attempt == 3:
                raise RuntimeError(f"API returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, http.client.IncompleteRead, http.client.RemoteDisconnected) as exc:
            if method != "GET" or attempt == 3:
                raise RuntimeError(f"Unable to connect to API: {exc}") from exc
        time.sleep(min(.5 * 2**attempt, 2))
    raise RuntimeError("GET retries exhausted")


def multipart(fields: dict[str, str], image: list[Path] | None) -> tuple[bytes, str]:
    boundary = "----A100ImageClient" + uuid.uuid4().hex
    chunks: list[bytes] = []

    def add(value: bytes) -> None:
        chunks.append(value)

    for name, value in fields.items():
        add(f"--{boundary}\r\n".encode())
        add(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        add(value.encode("utf-8"))
        add(b"\r\n")
    for image in image or []:
        content_type = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
        add(f"--{boundary}\r\n".encode())
        add((f'Content-Disposition: form-data; name="images"; filename="{image.name}"\r\n').encode())
        add(f"Content-Type: {content_type}\r\n\r\n".encode())
        add(image.read_bytes())
        add(b"\r\n")
    add(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def submit(base_url: str, api_key: str, model: str, prompt: str, image: list[Path] | None, idempotency_key: str | None = None, parameters: dict | None = None) -> str:
    operation = "image_edit" if image else "text_to_image"
    fields = {"model": model, "operation": operation, "prompt": prompt}
    if idempotency_key:
        fields["idempotency_key"] = idempotency_key
    for name, value in (parameters or {}).items():
        if value is not None:
            fields[name] = str(value).lower() if isinstance(value, bool) else str(value)
    body, content_type = multipart(fields, image)
    payload = json.loads(request(base_url + "/v1/tasks", api_key, method="POST", data=body, content_type=content_type))
    print(f"Task submitted: {payload['id']}")
    return payload["id"]


def wait_for_result(base_url: str, api_key: str, task_id: str, interval: float) -> str:
    status_url = f"{base_url}/v1/tasks/{task_id}"
    invalid_responses = 0
    while True:
        try:
            task = json.loads(request(status_url, api_key))
            invalid_responses = 0
        except json.JSONDecodeError:
            invalid_responses += 1
            if invalid_responses > 3:
                raise RuntimeError("gateway returned invalid task status repeatedly")
            time.sleep(interval)
            continue
        state = task["status"]
        print(f"Status: {state}")
        if state == "succeeded":
            return task["result_url"]
        if state == "failed":
            raise RuntimeError(task.get("error") or "inference failed")
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description="H100 Qwen-Image-2.1 generation and editing client")
    parser.add_argument("--prompt", required=True, help="Generation or editing instruction")
    parser.add_argument("--image", type=Path, action="append", help="Reference image; repeat up to 10 times for image_edit")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--idempotency-key")
    parser.add_argument("--model", choices=("qwen",), default="qwen")
    parser.add_argument("--output", type=Path, default=Path("result.png"))
    parser.add_argument("--url", default=os.getenv("IMAGE_API_URL"))
    parser.add_argument("--api-key", default=os.getenv("IMAGE_API_KEY"))
    parser.add_argument("--poll-interval", type=float, default=5.0)
    args = parser.parse_args()

    if not args.api_key:
        parser.error("set IMAGE_API_KEY or pass --api-key")
    if not args.url:
        parser.error("set IMAGE_API_URL or pass --url")
    for image in args.image or []:
        if not image.is_file():
            parser.error(f"image does not exist: {image}")

    base_url = args.url.rstrip("/")
    task_id = submit(base_url, args.api_key, args.model, args.prompt, args.image, args.idempotency_key, {name: getattr(args, name) for name in ("width", "height", "seed", "steps", "transparent")})
    result_url = wait_for_result(base_url, args.api_key, task_id, args.poll_interval)
    result = request(base_url + result_url, args.api_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(result)
    print(f"Saved result: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        raise SystemExit(130)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
