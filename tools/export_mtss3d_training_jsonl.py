#!/usr/bin/env python3
"""Collect validated training_record.json files into one JSONL dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stagerecon.captioning.io import read_json, write_json, validate_with_schema


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    records = []
    rejected = []
    for path in sorted(args.input_root.resolve().rglob("training_record.json")):
        record = read_json(path, "training record")
        errors = validate_with_schema(record, "training_record_v1.schema.json")
        mtss_path = path.parent / record["mtss_path"]
        if not mtss_path.is_file():
            errors.append(f"missing MTSS file: {mtss_path}")
        elif args.require_complete:
            mtss = read_json(mtss_path, "MTSS document")
            if mtss.get("status") != "complete":
                errors.append(f"MTSS status is {mtss.get('status')}")
        if errors:
            rejected.append({"path": str(path), "errors": errors})
        else:
            records.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = {
        "output": str(args.output.resolve()),
        "accepted": len(records),
        "rejected": len(rejected),
        "rejected_items": rejected,
    }
    write_json(args.output.with_suffix(args.output.suffix + ".report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not rejected else 2


if __name__ == "__main__":
    raise SystemExit(main())
