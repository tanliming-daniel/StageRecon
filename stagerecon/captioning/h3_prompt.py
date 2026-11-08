"""Render MiniMax H3 full-reference Context-IR from MTSS data."""

from __future__ import annotations

import re
from typing import Any


def _clock(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:06.3f}"


def _seconds(item: dict[str, Any]) -> tuple[float, float]:
    value = item["time_range"]
    if isinstance(value, dict):
        return float(value["start_sec"]), float(value["end_sec"])
    return float(value[0]), float(value[1])


def _replace_refs(text: str, labels: dict[str, str]) -> str:
    for ref_id, label in labels.items():
        text = text.replace(ref_id, label)
    return text


def _visual_segments(text: str) -> list[tuple[float, str]]:
    matches = list(re.finditer(r"\[(\d+(?:\.\d+)?)s\]\s*", text))
    if not matches:
        return [(0.0, text.strip())]
    segments = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segments.append((float(match.group(1)), text[match.end():end].strip()))
    return segments


def render_h3_full_reference(document: dict[str, Any]) -> str:
    mtss = document["mtss"]
    references = mtss["references"]
    shots = mtss["shots"]
    events = mtss["events"]
    global_info = mtss["global"]

    subjects = {
        item["ref_id"]: f"<Subject {index}>"
        for index, item in enumerate(references, start=1)
    }
    pictures: dict[str, list[str]] = {}
    picture_index = 1
    for item in references:
        pictures[item["ref_id"]] = []
        for _ in item["appearance_anchor"]["image_refs"]:
            pictures[item["ref_id"]].append(f"<Picture {picture_index}>")
            picture_index += 1

    definitions = []
    for item in references:
        subject = subjects[item["ref_id"]]
        source = " and ".join(pictures[item["ref_id"]])
        detail = item["appearance_anchor"]["id_features"]["detail_description"]
        definitions.append(
            f"{subject} is the {item['type']} shown in {source}. "
            f"Preserve this identity and appearance: {detail}"
        )
    definitions.append(
        "<Video 1> is the coarse 3D previs. It defines only composition, approximate "
        "spatial layout, initial positions and travel paths, relative blocking, framing, "
        "camera trajectory, and overall timing."
    )

    visible_subjects = []
    for shot in shots:
        for ref_id in shot.get("references_in_shot", []):
            label = subjects.get(ref_id)
            if label and label not in visible_subjects:
                visible_subjects.append(label)
    summary = (
        "[reference generation] The target is a live-action audiovisual scene featuring "
        f"{', '.join(visible_subjects)}. It uses <Video 1> only for structural guidance: "
        "camera movement, framing, shot scale, timing, subject root positions, movement "
        "paths, and relative screen-space layout. Replace every proxy model with the "
        "referenced people and re-synthesize natural performance."
    )

    retention = []
    for item in references:
        ref_id = item["ref_id"]
        appearances = [
            f"[Shot {index}]"
            for index, shot in enumerate(shots, start=1)
            if ref_id in shot.get("references_in_shot", [])
        ]
        retention.append(
            f"{subjects[ref_id]} (appears in {', '.join(appearances)}): fully_preserved - "
            "preserve the identity and appearance defined by its reference picture."
        )
    retention.append(
        "<Video 1> (structural guidance only): partially_preserved - preserve the camera "
        "trajectory, framing, shot scale, shot boundaries, overall timing, subject root "
        "positions, movement paths, and relative screen-space layout. Do not preserve "
        "proxy geometry, proxy appearance, body poses, limb motion, facial expressions, "
        "gait, or animation artifacts. Re-synthesize all local body movement and "
        "performance naturally according to the scene events, dialogue, and character "
        "interactions."
    )

    speech_order: list[str] = []
    for event in events:
        speaker = event.get("content", {}).get("speaker")
        if event.get("type") == "dialogue" and speaker and speaker not in speech_order:
            speech_order.append(speaker)
    speaker_ids = {ref_id: f"S{i}" for i, ref_id in enumerate(speech_order, start=1)}

    description = [f"{global_info['global_style']} {global_info['scene_description']}"]
    for shot_index, shot in enumerate(shots, start=1):
        shot_start, _ = _seconds(shot)
        prefix = f"[Shot {shot_index}]"
        if shot_index > 1:
            prefix += f" At {_clock(shot_start)}, the shot cuts to"
        camera = shot.get("camera", {})
        camera_text = ", ".join(
            str(camera[key]).replace("_", " ")
            for key in ("shot_type", "angle", "movement")
            if camera.get(key)
        )
        visual = _replace_refs(shot["visual_description"], subjects)
        active_events = set(shot.get("active_events", []))
        timeline_items: list[tuple[float, int, str]] = []
        for segment_index, (segment_time, segment_text) in enumerate(_visual_segments(visual)):
            timeline_items.append((segment_time, segment_index, segment_text))
        for event in events:
            if active_events and event["event_id"] not in active_events:
                continue
            event_start, _ = _seconds(event)
            content = event.get("content", {})
            if event.get("type") == "dialogue":
                ref_id = content.get("speaker")
                subject = subjects.get(ref_id, str(ref_id))
                delivery = content.get("description") or "natural conversational delivery"
                timeline_items.append(
                    (event_start, 100, f"{subject} ({speaker_ids.get(ref_id)}) speaks "
                    f"with {delivery}: <d>[English] {content.get('line', '')}</d>")
                )
            else:
                detail = content.get("description") or content.get("line")
                if detail:
                    timeline_items.append((event_start, 100, str(detail)))

        timeline_items.sort(key=lambda item: (item[0], item[1]))
        for item_index, (item_time, _, item_text) in enumerate(timeline_items):
            if item_index == 0:
                if shot_index == 1:
                    description.append(f"{prefix} {camera_text}. {item_text}".strip())
                else:
                    description.append(f"{prefix} a {camera_text} view. {item_text}".strip())
            else:
                description.append(f"At {_clock(item_time)}, {item_text}")

    description.append(
        "Use <Video 1> only for its structural guidance. Do not copy its proxy characters, "
        "specific poses, rigid motion, or placeholder geometry. Do not add subtitles, "
        "voice-over, extra people, or new movement paths. Keep faces, bodies, clothing, "
        "lip movement, and identity stable, with natural non-mechanical acting."
    )

    global_audio = global_info.get("global_audio", {})
    soundscape = global_audio.get("ambient_sound") or "N/A"
    music = global_audio.get("background_music") or "N/A"
    return (
        "subject_definitions:\n"
        + "\n".join(definitions)
        + "\n\nsummary:\n"
        + summary
        + "\n\nretention_analysis:\n"
        + "\n".join(retention)
        + "\n\ndetailed_description:\n"
        + " ".join(description)
        + "\n\noverall_soundscape:\n"
        + str(soundscape)
        + "\n\nnon_diegetic_music:\n"
        + str(music)
    )
