"""Normalize common Qwen Omni JSON aliases into the frozen proposal contract."""

from __future__ import annotations

from typing import Any


def _nullable_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        value = "; ".join(str(item) for item in value if item is not None)
    text = str(value).strip()
    if not text or text.lower().rstrip(".") in {"none", "null", "n/a"}:
        return None
    return text


def _evidence_flag(value: Any) -> bool | None:
    if isinstance(value, bool) or value is None:
        return value
    text = str(value).strip().lower()
    if not text or text in {"unknown", "uncertain", "n/a"}:
        return None
    if any(token in text for token in ("not visible", "no mouth", "not speaking", "no continuity")):
        return False
    return True


def _speaker_evidence(value: Any, *, dialogue: bool) -> dict[str, bool | None]:
    if not dialogue:
        return {
            "visible_on_screen": None,
            "mouth_activity": None,
            "voice_continuity": None,
        }
    if isinstance(value, dict):
        return {
            "visible_on_screen": _evidence_flag(value.get("visible_on_screen")),
            "mouth_activity": _evidence_flag(value.get("mouth_activity")),
            "voice_continuity": _evidence_flag(value.get("voice_continuity")),
        }
    text = str(value or "").lower()
    return {
        "visible_on_screen": True if any(token in text for token in ("visible", "stands", "approaches", "person_")) else None,
        "mouth_activity": True if any(token in text for token in ("mouth", "speaks", "speaking", "singing")) else None,
        "voice_continuity": True if any(token in text for token in ("same voice", "continues", "continuity")) else None,
    }


def normalize_omni_proposal(
    raw: dict[str, Any], *, allowed_ref_ids: set[str]
) -> dict[str, Any]:
    annotations = raw.get("annotations") if isinstance(raw.get("annotations"), dict) else {}
    references = []
    reference_source = raw.get("references") or raw.get("reference_bank") or []
    for item in reference_source:
        attributes = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
        references.append(
            {
                "ref_id": str(item.get("ref_id", "")),
                "semantic_description": str(
                    item.get("semantic_description") or item.get("description") or "Visible person"
                ),
                "id_features": _nullable_text(item.get("id_features")) or "Appearance visible in the reference image",
                "attributes": {
                    "clothing": _nullable_text(attributes.get("clothing")),
                    "hairstyle": _nullable_text(attributes.get("hairstyle")),
                    "accessories": _nullable_text(attributes.get("accessories")),
                },
                "confidence": float(item.get("confidence", 0.5)),
            }
        )

    events = []
    event_source = raw.get("events") or annotations.get("events") or []
    for index, item in enumerate(event_source, start=1):
        dialogue_value = item.get("dialogue") if isinstance(item.get("dialogue"), dict) else {}
        raw_type = str(item.get("type", "sfx")).lower()
        event_type = {
            "sound_effect": "sfx",
            "sound effect": "sfx",
            "effect": "sfx",
            "speech": "dialogue",
        }.get(raw_type, raw_type)
        dialogue = event_type == "dialogue"
        time_range = item.get("time_range")
        if isinstance(time_range, list) and len(time_range) >= 2:
            start_sec, end_sec = time_range[0], time_range[1]
        else:
            start_sec = item.get("start_sec", item.get("start_time", 0.0))
            end_sec = item.get("end_sec", item.get("end_time", 0.0))
        events.append(
            {
                "event_id": str(item.get("event_id") or f"EVENT_{index}"),
                "type": event_type,
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "speaker": (
                    item.get("speaker", item.get("speaker_ref_id", dialogue_value.get("speaker_ref_id")))
                    if dialogue else None
                ),
                "line": (
                    _nullable_text(
                        item.get("line", item.get("text", item.get("transcript", dialogue_value.get("line"))))
                    )
                    if dialogue else None
                ),
                "description": (
                    _nullable_text(item.get("description", item.get("audio_description")))
                    or ("Audible sound effect." if event_type == "sfx" else None)
                ),
                "speaker_evidence": _speaker_evidence(
                    item.get("speaker_evidence", dialogue_value.get("speaker_evidence")),
                    dialogue=dialogue,
                ),
                "confidence": float(
                    item.get("confidence", item.get("speaker_confidence", 0.5))
                ),
            }
        )

    event_ids = {item["event_id"] for item in events}
    returned_ref_ids = {item["ref_id"] for item in references}
    shots = []
    shot_source = raw.get("shots") or annotations.get("shots") or []
    for index, item in enumerate(shot_source, start=1):
        camera = item.get("camera") if isinstance(item.get("camera"), dict) else {}
        time_range = item.get("time_range")
        if isinstance(time_range, list) and len(time_range) >= 2:
            start_sec, end_sec = time_range[0], time_range[1]
        else:
            start_sec, end_sec = item.get("start_sec", 0.0), item.get("end_sec", 0.0)
        requested_refs = [str(value) for value in item.get("references_in_shot", [])]
        if not requested_refs:
            description = str(item.get("visual_description") or item.get("description") or "")
            requested_refs = [value for value in sorted(allowed_ref_ids) if value in description]
        if not requested_refs:
            requested_refs = sorted(returned_ref_ids)
        shots.append(
            {
                "shot_id": str(item.get("shot_id") or f"SHOT_{index}"),
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "visual_description": str(
                    item.get("visual_description") or item.get("description") or "Video action"
                ),
                "camera": {
                    "shot_type": _nullable_text(
                        camera.get("shot_type", camera.get("shot_scale", item.get("shot_scale")))
                    ),
                    "angle": _nullable_text(camera.get("angle", item.get("angle"))),
                    "movement": _nullable_text(
                        camera.get("movement", camera.get("motion", item.get("camera_motion")))
                    ),
                },
                "references_in_shot": [
                    value for value in requested_refs if value in allowed_ref_ids
                ],
                "active_event_ids": [
                    str(value) for value in item.get("active_event_ids", [])
                    if str(value) in event_ids
                ],
                "confidence": float(item.get("confidence", 0.5)),
            }
        )

    global_raw = raw.get("global") if isinstance(raw.get("global"), dict) else {}
    if not global_raw and isinstance(annotations.get("global"), dict):
        global_raw = annotations["global"]
    if not global_raw and isinstance(raw.get("global_notes"), dict):
        global_raw = raw["global_notes"]
    global_audio = global_raw.get("global_audio") if isinstance(global_raw.get("global_audio"), dict) else {}
    audio_raw = raw.get("audio") if isinstance(raw.get("audio"), dict) else {}
    if not audio_raw and isinstance(annotations.get("audio"), dict):
        audio_raw = annotations["audio"]
    ambient = (
        global_audio.get("ambient_sound")
        or global_audio.get("ambience_and_effects")
        or global_raw.get("ambient_sound")
        or audio_raw.get("ambient_sound")
        or audio_raw.get("ambient")
    )
    music = (
        global_audio.get("background_music")
        or global_audio.get("music_overview")
        or global_raw.get("background_music")
        or audio_raw.get("music")
    )
    return {
        "references": references,
        "shots": shots,
        "events": events,
        "global": {
            "scene_description": str(
                global_raw.get("scene_description")
                or global_raw.get("setting_summary")
                or global_raw.get("setting")
                or "Scene visible in the target video"
            ),
            "global_style": str(
                global_raw.get("global_style")
                or global_raw.get("visual_style")
                or "Live-action video"
            ),
            "global_audio": {
                "background_music": _nullable_text(music),
                "ambient_sound": _nullable_text(ambient),
            },
        },
    }
