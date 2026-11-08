"""OpenAI-compatible Qwen adapter for joint video, audio, and image annotation."""

from __future__ import annotations

import base64
import json
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..io import read_json


MAX_INLINE_DATA_URI_BYTES = 10 * 1024 * 1024


def _data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    value = f"data:{mime};base64,{encoded}"
    if len(value.encode("ascii")) >= MAX_INLINE_DATA_URI_BYTES:
        raise ValueError(
            f"Inline media exceeds Qwen's 10 MiB Base64 limit: {path} "
            f"({path.stat().st_size} raw bytes)"
        )
    return value


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                pieces.append(str(item.get("text", "")))
        return "".join(pieces)
    return str(content)


def parse_json_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Qwen response must be a JSON object")
    return value


class QwenProvider:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout_sec: float = 180.0,
        max_retries: int = 2,
        structured_output: str = "json_schema",
        reasoning_effort: str | None = "none",
        modalities: list[str] | None = None,
        max_tokens: int = 8192,
    ) -> None:
        if not model:
            raise ValueError("Qwen model ID is required")
        if not api_key:
            raise ValueError("Qwen API key is empty")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.structured_output = structured_output
        self.reasoning_effort = reasoning_effort
        self.modalities = modalities or ["text"]
        self.max_tokens = max_tokens

    def _endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"

    def _response_format(self, schema: dict[str, Any], schema_name: str) -> dict[str, Any] | None:
        if self.structured_output == "none":
            return None
        if self.structured_output == "json_object":
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        }

    def request_json(
        self,
        *,
        system_prompt: str,
        user_text: str,
        images: list[tuple[str, Path]],
        schema: dict[str, Any],
        schema_name: str,
        temperature: float = 0.0,
        videos: list[tuple[str, Path]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for label, path in videos or []:
            if not path.is_file():
                raise FileNotFoundError(f"Qwen video does not exist: {path}")
            content.append({"type": "text", "text": f"[{label}]"})
            content.append(
                {"type": "video_url", "video_url": {"url": _data_uri(path)}}
            )
        for label, path in images:
            if not path.is_file():
                raise FileNotFoundError(f"Qwen image does not exist: {path}")
            content.append({"type": "text", "text": f"[{label}]"})
            content.append(
                {"type": "image_url", "image_url": {"url": _data_uri(path)}}
            )
        content.append({"type": "text", "text": user_text})
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        }
        if self.modalities:
            body["modalities"] = self.modalities
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        response_format = self._response_format(schema, schema_name)
        if response_format:
            body["response_format"] = response_format
        request = urllib.request.Request(
            self._endpoint(),
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                    response_body = json.loads(response.read().decode("utf-8"))
                content_text = _content_text(response_body["choices"][0]["message"]["content"])
                parsed = parse_json_response(content_text)
                metadata = {
                    "model": self.model,
                    "attempt": attempt + 1,
                    "latency_sec": round(time.perf_counter() - started, 3),
                    "usage": response_body.get("usage"),
                    "response_id": response_body.get("id"),
                    "video_count": len(videos or []),
                    "image_count": len(images),
                }
                return parsed, metadata
            except urllib.error.HTTPError as error:
                payload = error.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"Qwen HTTP {error.code}: {payload[:1000]}")
                if error.code not in {408, 409, 429} and error.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as error:
                last_error = error
            if attempt < self.max_retries:
                time.sleep(min(8.0, 2.0 ** attempt))
        raise RuntimeError(f"Qwen request failed after retries: {last_error}") from last_error


def load_provider_from_config(
    config: dict[str, Any], *, api_key: str | None = None
) -> QwenProvider:
    import os

    qwen = config.get("qwen", config)
    key = api_key or os.environ.get(str(qwen.get("api_key_env", "QWEN_API_KEY")), "")
    return QwenProvider(
        base_url=str(qwen["base_url"]),
        model=str(qwen.get("model", "")),
        api_key=key,
        timeout_sec=float(qwen.get("timeout_sec", 180)),
        max_retries=int(qwen.get("max_retries", 2)),
        structured_output=str(qwen.get("structured_output", "json_schema")),
        reasoning_effort=(
            str(qwen["reasoning_effort"])
            if qwen.get("reasoning_effort") is not None
            else None
        ),
        modalities=[str(value) for value in qwen.get("modalities", ["text"])],
        max_tokens=int(qwen.get("max_tokens", 8192)),
    )
