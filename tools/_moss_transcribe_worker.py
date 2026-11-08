#!/usr/bin/env python3
"""Run MOSS-Transcribe-Diarize inside its pinned runtime environment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages,
    generate_transcription,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", default="OpenMOSS-Team/MOSS-Transcribe-Diarize"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    parser.add_argument("--prompt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input audio/video does not exist: {input_path}")

    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype="auto",
        attn_implementation=args.attn_implementation,
    ).to(dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    if args.prompt:
        messages = build_transcription_messages(str(input_path), prompt=args.prompt)
    else:
        messages = build_transcription_messages(str(input_path))
    result = generate_transcription(
        model,
        processor,
        messages,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        device=device,
        dtype=dtype,
    )
    parsed = list(parse_transcript(result["text"]))
    speaker_labels = sorted({segment.speaker for segment in parsed})
    document = {
        "schema_version": "stagerecon_moss_transcript_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "media_path": str(input_path),
            "media_type": input_path.suffix.lower().lstrip(".") or None,
        },
        "model": {
            "model_id": args.model,
            "device": str(device),
            "dtype": str(dtype).removeprefix("torch."),
            "attention": args.attn_implementation,
        },
        "speaker_registry": [
            {
                "speaker_label": label,
                "person_id": None,
                "binding_status": "unresolved",
            }
            for label in speaker_labels
        ],
        "segments": [
            {
                "segment_id": f"SEG_{index:04d}",
                "start_sec": float(segment.start),
                "end_sec": float(segment.end),
                "speaker_label": segment.speaker,
                "person_id": None,
                "text": segment.text,
                "evidence": "observed_audio",
            }
            for index, segment in enumerate(parsed, start=1)
        ],
        "raw_text": result["text"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "segments": len(parsed),
                "speakers": speaker_labels,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
