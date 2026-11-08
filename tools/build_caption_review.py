#!/usr/bin/env python3
"""Build an HTML review page for an existing MTSS-3D sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stagerecon.captioning.io import read_json
from stagerecon.captioning.review import write_review
from stagerecon.captioning.validation import validate_mtss3d


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.input.resolve()
    document = read_json(source, "MTSS-3D document")
    validation = validate_mtss3d(document, base_dir=source.parent)
    output = (args.output or source.parent / "review.html").resolve()
    write_review(output, document, validation)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
