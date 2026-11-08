#!/usr/bin/env python3
"""Validate MTSS-3D structure, references, time ranges, paths, and hashes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stagerecon.captioning.io import read_json, write_json
from stagerecon.captioning.validation import validate_mtss3d


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--check-assets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-hashes", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    source = args.input.resolve()
    report = validate_mtss3d(
        read_json(source, "MTSS-3D document"),
        base_dir=source.parent,
        check_assets=args.check_assets,
        check_hashes=args.check_hashes,
    )
    if args.report:
        write_json(args.report.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
