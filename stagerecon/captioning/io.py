"""Small, dependency-light I/O helpers used by the captioning pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ROOT = REPO_ROOT / "schemas"
PROMPT_ROOT = REPO_ROOT / "prompts"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, label: str = "JSON") -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    """Atomically write UTF-8 JSON with stable, review-friendly formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def relative_path(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def resolve_reference(path: str, base: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def file_asset(path: Path, base: Path, *, role: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value: dict[str, Any] = {
        "path": relative_path(path, base),
        "sha256": sha256_file(path),
    }
    if role is not None:
        value["role"] = role
    return value


def validate_with_schema(value: dict[str, Any], schema_name: str) -> list[str]:
    """Return JSON-Schema errors; report a clear dependency error if unavailable."""

    schema = read_json(SCHEMA_ROOT / schema_name, "JSON Schema")
    try:
        import jsonschema
    except ImportError:
        return [
            "jsonschema is not installed; install requirements-caption.txt to run "
            "structural validation"
        ]
    validator = jsonschema.Draft202012Validator(schema)
    errors = []
    for error in sorted(validator.iter_errors(value), key=lambda item: list(item.path)):
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"{location}: {error.message}")
    return errors
