#!/usr/bin/env python3
"""Scan per-character gallery folders and write a Friends character registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stagerecon.captioning.identity import build_registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument(
        "--template", type=Path,
        default=REPO_ROOT / "configs/characters/friends.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = build_registry(
        gallery_root=args.gallery,
        template_path=args.template,
        output=args.output.resolve(),
    )
    summary = {
        "output": str(args.output.resolve()),
        "characters": len(registry["characters"]),
        "gallery_images": sum(len(item["gallery"]) for item in registry["characters"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
