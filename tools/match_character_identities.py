#!/usr/bin/env python3
"""Bind SAM3 person tracks to a character registry or anonymous References."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stagerecon.captioning.identity import bind_identities
from stagerecon.captioning.manifest import load_clip_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-run", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("histogram", "insightface"), default="histogram")
    parser.add_argument("--accept-threshold", type=float, default=0.90)
    parser.add_argument("--margin-threshold", type=float, default=0.08)
    parser.add_argument("--representative-crops", type=int, default=5)
    parser.add_argument("--min-frame-gap", type=int, default=8)
    args = parser.parse_args()
    loaded = load_clip_run(args.clip_run)
    output = args.output_dir.resolve()
    result = bind_identities(
        clip_run=args.clip_run.resolve(),
        person_tracks_path=loaded["person_tracks_path"],
        registry_path=args.registry.resolve() if args.registry else None,
        output_dir=output,
        caption_root=output.parent,
        backend=args.backend,
        accept_threshold=args.accept_threshold,
        margin_threshold=args.margin_threshold,
        representative_crops=args.representative_crops,
        min_frame_gap=args.min_frame_gap,
    )
    print(json.dumps({
        "bindings": len(result["bindings"]),
        "matched": sum(item["identity_status"] != "anonymous" for item in result["bindings"]),
        "output": str(output / "bindings.json"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
