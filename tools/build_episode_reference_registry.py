#!/usr/bin/env python3
"""Build an episode-local registry from ``person_*/reference_bank.png`` files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stagerecon.captioning.identity import build_episode_registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-name", default="reference_bank.png")
    args = parser.parse_args()
    registry = build_episode_registry(
        episode_dir=args.episode_dir,
        output=args.output,
        reference_name=args.reference_name,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "episode_id": registry["episode_id"],
                "characters": len(registry["characters"]),
                "ref_ids": [
                    item["canonical_identity"].split(":")[-1].upper()
                    for item in registry["characters"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
