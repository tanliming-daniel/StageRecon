"""Structural and cross-reference validation for MTSS-3D artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import resolve_reference, sha256_file, validate_with_schema, write_json


def validate_visual_proposal(
    proposal: dict[str, Any], *, allowed_ref_ids: set[str], observation_ids: set[str]
) -> list[str]:
    errors = validate_with_schema(proposal, "qwen_visual_proposal_v1.schema.json")
    reference_ids = [str(item.get("ref_id")) for item in proposal.get("references", [])]
    if len(reference_ids) != len(set(reference_ids)):
        errors.append("references contains duplicate ref_id values")
    unknown_refs = set(reference_ids) - allowed_ref_ids
    if unknown_refs:
        errors.append(f"references contains unknown IDs: {sorted(unknown_refs)}")
    shot = proposal.get("shot", {})
    unknown_shot_refs = set(shot.get("references_in_shot", [])) - allowed_ref_ids
    if unknown_shot_refs:
        errors.append(f"shot.references_in_shot contains unknown IDs: {sorted(unknown_shot_refs)}")
    for segment in shot.get("description_segments", []):
        observation_id = str(segment.get("observation_id"))
        if observation_id not in observation_ids:
            errors.append(f"description segment references unknown observation: {observation_id}")
    return errors


def validate_audio_proposal(
    proposal: dict[str, Any], *, allowed_ref_ids: set[str], duration_sec: float
) -> list[str]:
    errors = validate_with_schema(proposal, "qwen_audio_proposal_v1.schema.json")
    for index, event in enumerate(proposal.get("events", [])):
        start = float(event.get("start_sec", -1))
        end = float(event.get("end_sec", -1))
        if not 0 <= start < end <= duration_sec:
            errors.append(
                f"events[{index}] time range [{start}, {end}) is outside [0, {duration_sec})"
            )
        speaker = event.get("speaker")
        if speaker is not None and speaker not in allowed_ref_ids:
            errors.append(f"events[{index}] has unknown speaker: {speaker}")
        if event.get("type") == "dialogue" and not event.get("line"):
            errors.append(f"events[{index}] dialogue has no verbatim line")
    return errors


def validate_omni_proposal(
    proposal: dict[str, Any], *, allowed_ref_ids: set[str], duration_sec: float,
    require_all_refs: bool = True,
) -> list[str]:
    errors = validate_with_schema(proposal, "qwen_omni_mtss_proposal_v1.schema.json")

    reference_ids = [str(item.get("ref_id")) for item in proposal.get("references", [])]
    if len(reference_ids) != len(set(reference_ids)):
        errors.append("references contains duplicate ref_id values")
    unknown_refs = set(reference_ids) - allowed_ref_ids
    if unknown_refs:
        errors.append(f"references contains unknown IDs: {sorted(unknown_refs)}")
    returned_ref_ids = set(reference_ids)
    missing_refs = allowed_ref_ids - returned_ref_ids
    if require_all_refs and missing_refs:
        errors.append(f"references omitted supplied IDs: {sorted(missing_refs)}")

    event_ids = [str(item.get("event_id")) for item in proposal.get("events", [])]
    event_id_set = set(event_ids)
    if len(event_ids) != len(event_id_set):
        errors.append("events contains duplicate event_id values")
    for index, event in enumerate(proposal.get("events", [])):
        start = float(event.get("start_sec", -1))
        end = float(event.get("end_sec", -1))
        if not 0 <= start < end <= duration_sec + 1e-6:
            errors.append(
                f"events[{index}] time range [{start}, {end}) is outside [0, {duration_sec})"
            )
        speaker = event.get("speaker")
        if speaker is not None and speaker not in returned_ref_ids:
            errors.append(f"events[{index}] has unknown speaker: {speaker}")
        if event.get("type") == "dialogue" and not event.get("line"):
            errors.append(f"events[{index}] dialogue has no verbatim line")
        if event.get("type") != "dialogue" and speaker is not None:
            errors.append(f"events[{index}] non-dialogue event has a speaker")

    shot_ids = [str(item.get("shot_id")) for item in proposal.get("shots", [])]
    if len(shot_ids) != len(set(shot_ids)):
        errors.append("shots contains duplicate shot_id values")
    previous_end = 0.0
    covered_event_ids: set[str] = set()
    shots = proposal.get("shots", [])
    for index, shot in enumerate(shots):
        start = float(shot.get("start_sec", -1))
        end = float(shot.get("end_sec", -1))
        if not 0 <= start < end <= duration_sec + 1e-6:
            errors.append(
                f"shots[{index}] time range [{start}, {end}) is outside [0, {duration_sec})"
            )
        if index and start < previous_end - 1e-6:
            errors.append(f"shots[{index}] overlaps the previous shot")
        previous_end = max(previous_end, end)
        unknown_shot_refs = set(shot.get("references_in_shot", [])) - returned_ref_ids
        if unknown_shot_refs:
            errors.append(
                f"shots[{index}].references_in_shot contains unknown IDs: {sorted(unknown_shot_refs)}"
            )
        active_ids = set(shot.get("active_event_ids", []))
        unknown_events = active_ids - event_id_set
        if unknown_events:
            errors.append(
                f"shots[{index}].active_event_ids contains unknown IDs: {sorted(unknown_events)}"
            )
        covered_event_ids.update(active_ids)
        for event in proposal.get("events", []):
            if event.get("event_id") not in active_ids:
                continue
            event_start = float(event.get("start_sec", -1))
            event_end = float(event.get("end_sec", -1))
            if event_end <= start or event_start >= end:
                errors.append(
                    f"shots[{index}] lists non-overlapping event {event.get('event_id')}"
                )

    missing_event_links = event_id_set - covered_event_ids
    if missing_event_links:
        errors.append(f"events are not linked to any shot: {sorted(missing_event_links)}")
    if shots:
        if abs(float(shots[0].get("start_sec", -1))) > 0.05:
            errors.append("shots do not start near 0 seconds")
        if abs(float(shots[-1].get("end_sec", -1)) - duration_sec) > 0.05:
            errors.append("shots do not cover the video end")
    return errors


def _validate_time_range(
    value: dict[str, Any], *, label: str, frame_count: int, duration: float,
    errors: list[str],
) -> None:
    start_frame = int(value.get("start_frame", -1))
    end_frame = int(value.get("end_frame_exclusive", -1))
    start = float(value.get("start_sec", -1))
    end = float(value.get("end_sec", -1))
    if not 0 <= start_frame < end_frame <= frame_count:
        errors.append(
            f"{label} frame range [{start_frame}, {end_frame}) is outside [0, {frame_count})"
        )
    if not 0 <= start < end <= duration + 1e-6:
        errors.append(f"{label} time range [{start}, {end}) is outside [0, {duration})")


def validate_mtss3d(
    document: dict[str, Any], *, base_dir: Path, check_assets: bool = True,
    check_hashes: bool = True,
) -> dict[str, Any]:
    errors = validate_with_schema(document, "mtss_3d_v1.schema.json")
    warnings: list[str] = []
    timeline = document.get("timeline", {})
    frame_count = int(timeline.get("frame_count", 0))
    duration = float(timeline.get("duration_sec", 0))

    references = document.get("mtss", {}).get("references", [])
    ref_ids = [str(item.get("ref_id")) for item in references]
    ref_id_set = set(ref_ids)
    if len(ref_ids) != len(ref_id_set):
        errors.append("Reference Stream contains duplicate ref_id values")

    events = document.get("mtss", {}).get("events", [])
    event_ids = [str(item.get("event_id")) for item in events]
    event_id_set = set(event_ids)
    if len(event_ids) != len(event_id_set):
        errors.append("Event Stream contains duplicate event_id values")

    for index, shot in enumerate(document.get("mtss", {}).get("shots", [])):
        _validate_time_range(
            shot.get("time_range", {}), label=f"shots[{index}]", frame_count=frame_count,
            duration=duration, errors=errors,
        )
        unknown_refs = set(shot.get("references_in_shot", [])) - ref_id_set
        if unknown_refs:
            errors.append(f"shots[{index}] references unknown IDs: {sorted(unknown_refs)}")
        unknown_events = set(shot.get("active_events", [])) - event_id_set
        if unknown_events:
            errors.append(f"shots[{index}] references unknown events: {sorted(unknown_events)}")
        description = str(shot.get("visual_description", "")).lower()
        forbidden = [word for word in ("圆柱体", "方块", "cylinder", "proxy block") if word in description]
        if forbidden:
            warnings.append(f"shots[{index}] describes proxy geometry: {forbidden}")

    for index, event in enumerate(events):
        _validate_time_range(
            event.get("time_range", {}), label=f"events[{index}]", frame_count=frame_count,
            duration=duration, errors=errors,
        )
        speaker = event.get("content", {}).get("speaker")
        if speaker is not None and speaker not in ref_id_set:
            errors.append(f"events[{index}] has unknown speaker: {speaker}")

    audio_asset = document.get("assets", {}).get("audio")
    if audio_asset is None and events:
        errors.append("assets.audio is null but Event Stream is not empty")
    if audio_asset is None:
        global_audio = document.get("mtss", {}).get("global", {}).get("global_audio", {})
        if global_audio.get("background_music") or global_audio.get("ambient_sound"):
            errors.append("assets.audio is null but Global Stream contains audio descriptions")

    bindings = document.get("extensions", {}).get("mtss_3d", {}).get("entity_bindings", [])
    binding_ids = [str(item.get("ref_id")) for item in bindings]
    if set(binding_ids) != ref_id_set:
        errors.append("entity_bindings ref_id set does not match Reference Stream")
    canonical_ids = [
        str(item["canonical_identity"])
        for item in bindings
        if item.get("canonical_identity") is not None
    ]
    if len(canonical_ids) != len(set(canonical_ids)):
        errors.append("Multiple tracks bind to the same canonical identity within one shot")
    if any(item.get("identity_status") == "anonymous" for item in bindings):
        warnings.append("At least one person uses an anonymous fallback Reference")

    if check_assets:
        for role, asset in document.get("assets", {}).items():
            if asset is None:
                continue
            path = resolve_reference(str(asset["path"]), base_dir)
            if not path.is_file():
                errors.append(f"Missing {role} asset: {path}")
            elif check_hashes and sha256_file(path) != asset["sha256"]:
                errors.append(f"SHA-256 mismatch for {role}: {path}")
        for reference in references:
            for image_ref in reference.get("appearance_anchor", {}).get("image_refs", []):
                path = resolve_reference(str(image_ref), base_dir)
                if not path.is_file():
                    errors.append(f"Missing appearance anchor: {path}")
        geometry_files = (
            document.get("extensions", {}).get("mtss_3d", {}).get("geometry", {}).get("files", [])
        )
        for item in geometry_files:
            path = resolve_reference(str(item["path"]), base_dir)
            if not path.is_file():
                errors.append(f"Missing geometry file: {path}")
            elif check_hashes and sha256_file(path) != item["sha256"]:
                errors.append(f"SHA-256 mismatch for geometry file: {path}")

    status = "passed" if not errors else "failed"
    return {
        "valid": not errors,
        "status": status,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }


def write_validation_report(
    path: Path, document: dict[str, Any], *, base_dir: Path,
    check_assets: bool = True, check_hashes: bool = True,
) -> dict[str, Any]:
    report = validate_mtss3d(
        document,
        base_dir=base_dir,
        check_assets=check_assets,
        check_hashes=check_hashes,
    )
    write_json(path, report)
    return report
