"""Render a versioned MiniMax H3 prompt from validated MTSS-3D data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import PROMPT_ROOT, relative_path, sha256_text, validate_with_schema, write_json
from .h3_prompt import render_h3_full_reference


def build_training_record(
    *, document: dict[str, Any], mtss_path: Path, output: Path
) -> tuple[dict[str, Any], list[str]]:
    contract_path = PROMPT_ROOT / "training/previs_control_v1.txt"
    contract = contract_path.read_text(encoding="utf-8").strip()
    references = document["mtss"]["references"]
    reference_bank = []
    for reference in references:
        images = list(reference["appearance_anchor"]["image_refs"])
        reference_bank.append({"ref_id": reference["ref_id"], "images": images})
    prompt = render_h3_full_reference(document)
    record = {
        "schema_version": "training_record_v1",
        "sample_id": document["sample_id"],
        "control_contract": "previs_control_v1",
        "control_contract_sha256": sha256_text(contract),
        "previs_video": document["assets"]["previs_video"]["path"],
        "reference_bank": reference_bank,
        "prompt": prompt,
        "target_video": document["assets"]["target_video"]["path"],
        "mtss_path": relative_path(mtss_path, output.parent),
    }
    errors = validate_with_schema(record, "training_record_v1.schema.json")
    write_json(output, record)
    return record, errors
