#!/usr/bin/env python3
"""Run speaker-aware MOSS transcription through StageRecon's unified CLI."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = REPO_ROOT / ".envs/moss-transcribe-diarize/bin/python"
DEFAULT_WORKER = REPO_ROOT / "tools/_moss_transcribe_worker.py"
DEFAULT_CACHE = REPO_ROOT / ".cache/huggingface"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runtime-python", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument(
        "--model", default="OpenMOSS-Team/MOSS-Transcribe-Diarize"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    parser.add_argument("--prompt")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input audio/video does not exist: {input_path}")
    runtime = args.runtime_python.resolve()
    if not runtime.is_file():
        raise FileNotFoundError(
            f"MOSS runtime does not exist: {runtime}. Run the MOSS setup first."
        )
    output = (
        args.output.resolve()
        if args.output
        else input_path.with_suffix(".moss_transcript.json")
    )
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(runtime),
        str(DEFAULT_WORKER),
        "--input",
        str(input_path),
        "--output",
        str(output),
        "--model",
        args.model,
        "--device",
        args.device,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--attn-implementation",
        args.attn_implementation,
    ]
    if args.prompt:
        command.extend(("--prompt", args.prompt))

    environment = os.environ.copy()
    environment.update(
        {
            "HF_HOME": str(cache_dir),
            "HUGGINGFACE_HUB_CACHE": str(cache_dir / "hub"),
            "HF_HUB_DISABLE_XET": "1",
            "TORCH_HOME": str(REPO_ROOT / ".cache/torch"),
        }
    )
    if args.cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    print(
        json.dumps(
            {
                "stage": "moss_transcribe",
                "runtime": str(runtime),
                "input": str(input_path),
                "output": str(output),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    subprocess.run(command, check=True, env=environment, cwd=REPO_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
