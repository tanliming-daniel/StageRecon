"""Qwen-driven matching from SAM3 person tracks to episode reference banks."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from .identity import _person_ref_id
from .io import read_json, relative_path, resolve_reference, write_json


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def _registry_candidates(registry_path: Path) -> list[dict[str, Any]]:
    registry = read_json(registry_path, "character registry")
    candidates = []
    for index, character in enumerate(registry.get("characters", []), start=1):
        canonical = str(character["canonical_identity"])
        gallery_paths = []
        for item in character.get("gallery", []):
            path = Path(str(item["path"]))
            if not path.is_absolute():
                path = registry_path.parent / path
            path = path.resolve()
            if path.is_file():
                gallery_paths.append(path)
        if not gallery_paths:
            continue
        candidates.append(
            {
                "ref_id": _person_ref_id(canonical, index),
                "canonical_identity": canonical,
                "display_name": str(character.get("display_name") or canonical),
                "gallery_paths": gallery_paths,
            }
        )
    return candidates


def qwen_identity_request(
    *,
    identity: dict[str, Any],
    registry_path: Path,
    caption_root: Path,
    crops_per_track: int = 3,
) -> tuple[str, list[tuple[str, Path]], list[dict[str, Any]]]:
    candidates = _registry_candidates(registry_path)
    if not candidates:
        raise ValueError(f"Registry contains no usable gallery images: {registry_path}")

    images: list[tuple[str, Path]] = []
    candidate_payload = []
    for candidate in candidates:
        labels = []
        for index, path in enumerate(candidate["gallery_paths"], start=1):
            label = f"CANDIDATE {candidate['ref_id']} IMAGE {index}"
            labels.append(label)
            images.append((label, path))
        candidate_payload.append(
            {
                "ref_id": candidate["ref_id"],
                "display_name": candidate["display_name"],
                "image_labels": labels,
            }
        )

    track_payload = []
    for binding in identity.get("bindings", []):
        track_id = int(binding["source_track"]["sam3_track_id"])
        labels = []
        for index, crop in enumerate(binding.get("representative_crops", [])[:crops_per_track], start=1):
            label = f"TRACK {track_id} CROP {index}"
            labels.append(label)
            images.append((label, resolve_reference(str(crop), caption_root)))
        track_payload.append(
            {
                "track_id": track_id,
                "first_frame": int(binding["source_track"]["first_frame"]),
                "last_frame": int(binding["source_track"]["last_frame"]),
                "crop_labels": labels,
            }
        )

    payload = {
        "candidate_reference_bank": candidate_payload,
        "sam3_tracks": track_payload,
        "rules": {
            "every_track_once": True,
            "candidate_used_at_most_once": True,
            "unmatched_ref_id": None,
        },
    }
    text = (
        "Match every SAM3 track in the target video to at most one candidate person "
        "reference. Compare face identity first, then hairstyle, body shape, clothing and "
        "temporal continuity. The candidate images may be clean generated reference sheets "
        "derived from Friends episode stills, while the track crops come from the original "
        "episode video. Do not reject a correct identity merely because the background, pose "
        "or rendering style differs. Return null when genuinely uncertain. Return the complete "
        "qwen_identity_match_v1 JSON object.\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return text, images, candidates


def validate_qwen_identity_matches(
    proposal: dict[str, Any],
    *,
    track_ids: set[int],
    allowed_ref_ids: set[str],
) -> list[str]:
    errors = []
    bindings = proposal.get("bindings", [])
    returned_track_ids = [int(item.get("track_id", -1)) for item in bindings]
    if len(returned_track_ids) != len(set(returned_track_ids)):
        errors.append("bindings contains duplicate track_id values")
    if set(returned_track_ids) != track_ids:
        errors.append(
            f"bindings track_id set {sorted(set(returned_track_ids))} does not match "
            f"supplied tracks {sorted(track_ids)}"
        )
    selected = []
    for item in bindings:
        ref_id = item.get("ref_id")
        if ref_id is not None and str(ref_id) not in allowed_ref_ids:
            errors.append(f"track {item.get('track_id')} selected unknown ref_id: {ref_id}")
        if ref_id is not None:
            selected.append(str(ref_id))
    if len(selected) != len(set(selected)):
        errors.append("the same candidate ref_id was assigned to multiple tracks")
    return errors


def apply_qwen_identity_matches(
    *,
    identity: dict[str, Any],
    proposal: dict[str, Any],
    candidates: list[dict[str, Any]],
    output_dir: Path,
    caption_root: Path,
    model_name: str,
    accept_threshold: float = 0.55,
) -> dict[str, Any]:
    candidate_by_ref = {str(item["ref_id"]): item for item in candidates}
    match_by_track = {int(item["track_id"]): item for item in proposal.get("bindings", [])}
    reference_dir = output_dir / "references"
    reference_dir.mkdir(parents=True, exist_ok=True)

    for binding in identity.get("bindings", []):
        track_id = int(binding["source_track"]["sam3_track_id"])
        match = match_by_track.get(track_id, {})
        ref_id = match.get("ref_id")
        confidence = float(match.get("confidence", 0.0))
        candidate = candidate_by_ref.get(str(ref_id)) if ref_id is not None else None
        if candidate is None or confidence < accept_threshold:
            continue
        source = Path(candidate["gallery_paths"][0])
        destination = reference_dir / (
            _safe_slug(str(candidate["canonical_identity"])) + source.suffix.lower()
        )
        if not destination.is_file():
            shutil.copy2(source, destination)
        binding.update(
            {
                "ref_id": str(candidate["ref_id"]),
                "canonical_identity": str(candidate["canonical_identity"]),
                "identity_status": "matched",
                "appearance_image": relative_path(destination, caption_root),
                "anchor_origin": "character_gallery",
                "match": {
                    "method": "qwen_omni_identity_selector_v1",
                    "score": confidence,
                    "second_best_score": None,
                    "margin": None,
                    "observation_count": len(binding.get("representative_crops", [])),
                    "model_version": model_name,
                },
                "candidate_matches": [
                    {
                        "canonical_identity": str(candidate["canonical_identity"]),
                        "score": confidence,
                    }
                ],
            }
        )
    identity["backend"] = "qwen_omni_identity_selector_v1"
    write_json(output_dir / "bindings.json", identity)
    return identity
