#!/usr/bin/env python3
"""Generate one MTSS-3D caption bundle from a StageRecon clip run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stagerecon.captioning.manifest import load_clip_run
from stagerecon.captioning.pipeline import run_pipeline


def load_project_env(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding shell variables."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-run", type=Path, required=True)
    parser.add_argument("--previs-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--sample-id")
    parser.add_argument("--sequence-id")
    parser.add_argument("--target-video", type=Path)
    audio_group = parser.add_mutually_exclusive_group()
    audio_group.add_argument("--audio", type=Path)
    audio_group.add_argument("--audio-from-source", action="store_true")
    parser.add_argument("--transcript", type=Path)
    parser.add_argument(
        "--omni-proposal",
        type=Path,
        help="Use a precomputed qwen_omni_mtss_proposal_v1 JSON response.",
    )
    parser.add_argument(
        "--transcribe-moss",
        action="store_true",
        help="Generate a speaker-aware transcript with the isolated MOSS runtime.",
    )
    parser.add_argument(
        "--moss-runtime-python",
        type=Path,
        default=REPO_ROOT / ".envs/moss-transcribe-diarize/bin/python",
    )
    parser.add_argument(
        "--moss-model", default="OpenMOSS-Team/MOSS-Transcribe-Diarize"
    )
    parser.add_argument("--moss-device", default="auto")
    parser.add_argument("--moss-cuda-visible-devices")
    parser.add_argument("--moss-max-new-tokens", type=int, default=8192)
    parser.add_argument("--visual-proposal", type=Path)
    parser.add_argument("--audio-proposal", type=Path)
    parser.add_argument("--manual-bindings", type=Path)
    parser.add_argument(
        "--identity-mode",
        choices=("auto", "qwen_direct", "track"),
        default="auto",
        help=(
            "auto uses direct Qwen video-to-reference matching when the registry has "
            "gallery images; track preserves the legacy SAM3/ReID path."
        ),
    )
    parser.add_argument("--identity-backend", choices=("histogram", "insightface"))
    parser.add_argument("--identity-model-name", default="buffalo_l")
    parser.add_argument("--identity-model-root", type=Path)
    parser.add_argument("--identity-device", default="cpu")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    load_project_env(REPO_ROOT / ".env")
    args = parse_args()
    clip_run = args.clip_run.resolve()
    output_dir = (args.output_dir or clip_run / "captions/mtss_3d_v1").resolve()
    audio = args.audio
    if args.audio_from_source:
        audio = load_clip_run(clip_run)["source_video"]
    transcript = args.transcript
    if args.transcribe_moss:
        if transcript is not None:
            raise ValueError("--transcribe-moss cannot be combined with --transcript")
        if audio is None:
            raise ValueError(
                "--transcribe-moss requires --audio or --audio-from-source"
            )
        transcript = output_dir / "audio/moss_transcript.json"
        command = [
            sys.executable,
            str(REPO_ROOT / "tools/run_moss_transcribe.py"),
            str(audio),
            "--output",
            str(transcript),
            "--runtime-python",
            str(args.moss_runtime_python),
            "--model",
            args.moss_model,
            "--device",
            args.moss_device,
            "--max-new-tokens",
            str(args.moss_max_new_tokens),
        ]
        if args.moss_cuda_visible_devices is not None:
            command.extend(
                ("--cuda-visible-devices", args.moss_cuda_visible_devices)
            )
        subprocess.run(command, check=True, cwd=REPO_ROOT)
    document, report = run_pipeline(
        clip_run=clip_run,
        previs_video=args.previs_video,
        output_dir=output_dir,
        registry_path=args.registry,
        config_path=args.config,
        sample_id=args.sample_id,
        sequence_id=args.sequence_id,
        target_video=args.target_video,
        audio=audio,
        transcript_path=transcript,
        omni_proposal_path=args.omni_proposal,
        visual_proposal_path=args.visual_proposal,
        audio_proposal_path=args.audio_proposal,
        manual_bindings_path=args.manual_bindings,
        identity_mode=args.identity_mode,
        identity_backend=args.identity_backend,
        identity_model_name=args.identity_model_name,
        identity_model_root=args.identity_model_root,
        identity_device=args.identity_device,
        qwen_base_url=args.base_url,
        qwen_model=args.model,
        qwen_api_key=args.api_key,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    summary = {
        "sample_id": document["sample_id"],
        "status": document["status"],
        "validation": report["status"],
        "output_dir": str(output_dir),
        "errors": report["error_count"],
        "warnings": report["warning_count"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
