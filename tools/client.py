#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys
import time
from typing import Any
from urllib import error, parse, request


TERMINAL_STATUSES = {"succeeded", "failed"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload an image to the TRELLIS.2 web_v2 service and wait for the generated GLB."
    )
    parser.add_argument(
        "--server",
        default="http://192.168.4.30:12837",
        help="Base URL of the web_v2 service. Default: http://192.168.4.30:12837",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Local image file to upload.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Local directory that will receive downloaded artifacts.",
    )
    parser.add_argument(
        "--latent_res",
        "--latent-res",
        choices=("512", "1024", "1536"),
        default="1024",
        help="TRELLIS latent resolution. 512 -> pipeline_type 512, 1024 -> 1024, 1536 -> 1536_cascade. Default: 1024",
    )
    parser.add_argument(
        "--glb-name",
        default="sample",
        help="Output GLB base name on the server. Default: sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed sent to the server. Default: 42",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between job status polls. Default: 2",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=7200.0,
        help="Total seconds to wait for the job; 0 means no limit. Default: 7200",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=300.0,
        help="Per-request HTTP timeout in seconds. Default: 300",
    )
    return parser


def _join_url(server: str, route_path: str) -> str:
    return server.rstrip("/") + "/" + route_path.lstrip("/")


def _read_json_response(response: Any) -> dict[str, Any]:
    body = response.read().decode("utf-8")
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise RuntimeError("Server response is not a JSON object")
    return payload


def _request_json(req: request.Request, timeout: float) -> dict[str, Any]:
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return _read_json_response(response)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
            message = payload.get("message", body)
            code = payload.get("code", "http_error")
        except json.JSONDecodeError:
            message = body
            code = "http_error"
        raise RuntimeError(f"HTTP {exc.code} {code}: {message}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Failed to reach server: {exc.reason}") from exc


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return _request_json(req, timeout)


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    req = request.Request(url, method="GET")
    return _request_json(req, timeout)


def _download_file(url: str, destination: Path, timeout: float) -> None:
    try:
        with request.urlopen(url, timeout=timeout) as response, destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    except error.HTTPError as exc:
        raise RuntimeError(f"Download failed with HTTP {exc.code}: {url}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Failed to download {url}: {exc.reason}") from exc


def _print_new_events(job_payload: dict[str, Any], last_seq: int) -> int:
    events = job_payload.get("events")
    if not isinstance(events, list):
        return last_seq

    for event in events:
        if not isinstance(event, dict):
            continue
        try:
            seq = int(event.get("seq", 0))
        except (TypeError, ValueError):
            continue
        if seq <= last_seq:
            continue
        message = event.get("message") or event.get("stage") or "updated"
        timestamp = event.get("time")
        if isinstance(timestamp, str) and timestamp:
            print(f"{timestamp}  {message}", flush=True)
        else:
            print(str(message), flush=True)
        last_seq = max(last_seq, seq)
    return last_seq


def _wait_for_job(
    *,
    server: str,
    job_id: str,
    poll_interval: float,
    timeout: float,
    request_timeout: float,
    initial_payload: dict[str, Any],
) -> dict[str, Any]:
    last_seq = _print_new_events(initial_payload, 0)
    deadline = None if timeout <= 0 else time.monotonic() + timeout

    while True:
        if deadline is not None and time.monotonic() > deadline:
            raise RuntimeError(f"Timed out waiting for job {job_id}")

        time.sleep(max(0.1, poll_interval))
        job_payload = _get_json(_join_url(server, f"jobs/{parse.quote(job_id)}"), request_timeout)
        last_seq = _print_new_events(job_payload, last_seq)

        status = job_payload.get("status")
        if status == "succeeded":
            return job_payload
        if status == "failed":
            error_payload = job_payload.get("error")
            if isinstance(error_payload, dict):
                message = error_payload.get("message", "job failed")
            else:
                message = "job failed"
            raise RuntimeError(f"Job {job_id} failed: {message}")
        if status in TERMINAL_STATUSES:
            raise RuntimeError(f"Job {job_id} ended with unexpected status: {status}")


def _download_glb(job_payload: dict[str, Any], *, server: str, output_dir: Path, timeout: float) -> Path:
    artifacts = job_payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("Server response does not contain an artifacts object")

    glb_artifact = artifacts.get("glb")
    if not isinstance(glb_artifact, dict):
        raise RuntimeError("Server response does not contain a GLB artifact")

    filename = glb_artifact.get("filename")
    url = glb_artifact.get("url") or glb_artifact.get("download_path")
    if not isinstance(filename, str) or not filename:
        raise RuntimeError("GLB artifact filename is missing")
    if not isinstance(url, str) or not url:
        raise RuntimeError("GLB artifact URL is missing")

    destination = output_dir / filename
    resolved_url = parse.urljoin(server.rstrip("/") + "/", url)
    _download_file(resolved_url, destination, timeout)
    return destination


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise SystemExit(f"image does not exist or is not a file: {image_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "image_base64": base64.b64encode(image_path.read_bytes()).decode("ascii"),
        "image_name": image_path.name,
        "glb_name": args.glb_name,
        "latent_res": args.latent_res,
        "seed": args.seed,
    }

    accepted_payload = _post_json(
        _join_url(args.server, "generate"),
        payload,
        args.request_timeout,
    )

    job_id = accepted_payload.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise RuntimeError(f"Server response does not contain a job_id: {accepted_payload}")

    print(f"job_id={job_id}", flush=True)
    final_payload = _wait_for_job(
        server=args.server,
        job_id=job_id,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
        request_timeout=args.request_timeout,
        initial_payload=accepted_payload,
    )

    downloaded_glb = _download_glb(
        final_payload,
        server=args.server,
        output_dir=output_dir,
        timeout=args.request_timeout,
    )

    response_json_path = output_dir / "response.json"
    response_json_path.write_text(
        json.dumps(final_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"glb={downloaded_glb}", flush=True)
    print(f"response={response_json_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
