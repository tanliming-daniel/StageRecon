"""Deterministically merge semantic proposals with StageRecon ground rules."""

from __future__ import annotations

import math
from typing import Any

from . import CONTROL_CONTRACT, SCHEMA_VERSION
from .io import now_iso


PRESERVE_SCOPE = [
    "composition",
    "approximate_scene_layout",
    "actor_initial_positions",
    "actor_root_trajectories",
    "screen_space_relative_positions",
    "shot_scale",
    "camera_trajectory",
    "overall_timing",
]

IGNORE_SCOPE = [
    "proxy_appearance",
    "proxy_body_pose",
    "proxy_limb_motion",
    "proxy_facial_expression",
    "proxy_animation_artifacts",
]


def _event_time_range(
    event: dict[str, Any], fps: float
) -> dict[str, Any]:
    start_sec = float(event["start_sec"])
    end_sec = float(event["end_sec"])
    return {
        "start_frame": max(0, int(math.floor(start_sec * fps + 1e-8))),
        "end_frame_exclusive": max(1, int(math.ceil(end_sec * fps - 1e-8))),
        "start_sec": start_sec,
        "end_sec": end_sec,
        "temporal_precision": "asr_exact",
    }


def _binding_for_schema(binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "ref_id": binding["ref_id"],
        "canonical_identity": binding.get("canonical_identity"),
        "identity_status": binding["identity_status"],
        "source_track": binding.get("source_track"),
        "appearance_image": binding["appearance_image"],
        "anchor_origin": binding["anchor_origin"],
        "match": binding.get("match"),
        "candidate_matches": binding.get("candidate_matches", []),
    }


def _model_time_range(
    *, start_sec: float, end_sec: float, fps: float, frame_count: int
) -> dict[str, Any]:
    start_frame = min(frame_count - 1, max(0, int(math.floor(start_sec * fps + 1e-8))))
    end_frame = min(frame_count, max(start_frame + 1, int(math.ceil(end_sec * fps - 1e-8))))
    return {
        "start_frame": start_frame,
        "end_frame_exclusive": end_frame,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "temporal_precision": "sampled_estimate",
    }


def _merge_omni_mtss3d(
    *,
    sample_id: str,
    sequence_id: str | None,
    assets: dict[str, Any],
    timeline: dict[str, Any],
    identity: dict[str, Any],
    omni_proposal: dict[str, Any],
    geometry: dict[str, Any],
    blocking_motion: list[dict[str, Any]],
    caption_model: str | None,
    request_hashes: dict[str, str],
    dry_run: bool,
) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []
    bindings = identity.get("bindings", [])
    known_ref_ids = {str(binding["ref_id"]) for binding in bindings}
    proposals_by_ref = {
        str(item["ref_id"]): item for item in omni_proposal.get("references", [])
    }
    target_video_ref = str(assets["target_video"]["path"])
    fps = float(timeline["fps"])
    frame_count = int(timeline["frame_count"])

    evidence: list[dict[str, Any]] = [
        {
            "evidence_id": "EVID_OMNI_TARGET_VIDEO",
            "kind": "observed",
            "source_refs": [target_video_ref],
            "confidence": None,
        }
    ]
    references = []
    for index, binding in enumerate(bindings, start=1):
        ref_id = str(binding["ref_id"])
        proposal = proposals_by_ref.get(ref_id)
        if proposal is None:
            notes.append(f"Omni proposal omitted {ref_id}; inserted conservative placeholder")
            proposal = {
                "semantic_description": f"Visible person identified as {ref_id}",
                "id_features": "Appearance requires manual review",
                "attributes": {
                    "clothing": None,
                    "hairstyle": None,
                    "accessories": None,
                },
                "confidence": 0.0,
            }
        evidence_id = f"EVID_REFERENCE_{index}"
        source_refs = [binding["appearance_image"], target_video_ref]
        if binding.get("source_track") is not None:
            source_refs.insert(
                1, f"sam3_track:{binding['source_track']['sam3_track_id']}"
            )
        evidence.append(
            {
                "evidence_id": evidence_id,
                "kind": "derived",
                "source_refs": source_refs,
                "confidence": float(proposal.get("confidence", 0.0)),
            }
        )
        references.append(
            {
                "ref_id": ref_id,
                "type": "person",
                "semantic_description": str(proposal["semantic_description"]),
                "timestamp": float(binding.get("anchor_timestamp_sec", 0.0)),
                "appearance_anchor": {
                    "image_refs": [str(binding["appearance_image"])],
                    "id_features": {
                        "detail_description": str(proposal["id_features"])
                    },
                    "attributes": {
                        "clothing": proposal["attributes"].get("clothing"),
                        "hairstyle": proposal["attributes"].get("hairstyle"),
                        "accessories": proposal["attributes"].get("accessories"),
                    },
                },
                "evidence_ids": [evidence_id],
            }
        )

    events = []
    for index, proposal in enumerate(omni_proposal.get("events", []), start=1):
        event_id = str(proposal.get("event_id") or f"EVENT_{index}")
        evidence_id = f"EVID_OMNI_EVENT_{index}"
        evidence.append(
            {
                "evidence_id": evidence_id,
                "kind": "observed",
                "source_refs": [target_video_ref],
                "confidence": float(proposal.get("confidence", 0.0)),
            }
        )
        events.append(
            {
                "event_id": event_id,
                "type": proposal["type"],
                "time_range": _model_time_range(
                    start_sec=float(proposal["start_sec"]),
                    end_sec=float(proposal["end_sec"]),
                    fps=fps,
                    frame_count=frame_count,
                ),
                "content": {
                    "speaker": proposal.get("speaker"),
                    "line": proposal.get("line"),
                    "description": proposal.get("description"),
                    "speaker_evidence": proposal.get("speaker_evidence", {
                        "visible_on_screen": None,
                        "mouth_activity": None,
                        "voice_continuity": None,
                    }),
                },
                "evidence_ids": [evidence_id],
            }
        )

    event_ids = {event["event_id"] for event in events}
    shots = []
    for index, proposal in enumerate(omni_proposal.get("shots", []), start=1):
        requested_refs = [str(value) for value in proposal.get("references_in_shot", [])]
        references_in_shot = [value for value in requested_refs if value in known_ref_ids]
        active_events = [
            str(value) for value in proposal.get("active_event_ids", [])
            if str(value) in event_ids
        ]
        shots.append(
            {
                "shot_id": str(proposal.get("shot_id") or f"SHOT_{index}"),
                "time_range": _model_time_range(
                    start_sec=float(proposal["start_sec"]),
                    end_sec=float(proposal["end_sec"]),
                    fps=fps,
                    frame_count=frame_count,
                ),
                "visual_description": str(proposal["visual_description"]),
                "camera": {
                    "shot_type": proposal["camera"].get("shot_type"),
                    "angle": proposal["camera"].get("angle"),
                    "movement": proposal["camera"].get("movement"),
                },
                "references_in_shot": references_in_shot,
                "active_events": active_events,
                "evidence_ids": ["EVID_OMNI_TARGET_VIDEO"],
            }
        )

    global_proposal = omni_proposal["global"]
    global_audio = global_proposal["global_audio"]
    global_stream = {
        "scene_description": str(global_proposal["scene_description"]),
        "global_style": str(global_proposal["global_style"]),
        "global_audio": {
            "background_music": global_audio.get("background_music"),
            "ambient_sound": global_audio.get("ambient_sound"),
        },
        "evidence_ids": ["EVID_OMNI_TARGET_VIDEO"],
    }

    if geometry.get("files"):
        evidence.append(
            {
                "evidence_id": "EVID_GEOMETRY",
                "kind": "derived",
                "source_refs": [item["path"] for item in geometry["files"]],
                "confidence": None,
            }
        )

    needs_review = bool(
        notes
        or any(binding["identity_status"] == "anonymous" for binding in bindings)
        or any(
            event["type"] == "dialogue" and event["content"]["speaker"] is None
            for event in events
        )
    )
    status = "dry_run_placeholder" if dry_run else ("needs_review" if needs_review else "complete")
    document = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "sequence_id": sequence_id,
        "status": status,
        "assets": assets,
        "timeline": timeline,
        "mtss": {
            "references": references,
            "shots": shots,
            "events": events,
            "global": global_stream,
        },
        "extensions": {
            "mtss_3d": {
                "control_contract": CONTROL_CONTRACT,
                "control_scope": {"preserve": PRESERVE_SCOPE, "ignore": IGNORE_SCOPE},
                "entity_bindings": [_binding_for_schema(binding) for binding in bindings],
                "blocking_motion": blocking_motion,
                "geometry": geometry,
                "evidence": evidence,
            }
        },
        "annotation_meta": {
            "caption_model": caption_model,
            "visual_prompt_version": "omni_joint_v1",
            "audio_prompt_version": "omni_joint_v1",
            "identity_registry_version": identity.get("registry_version"),
            "created_at": now_iso(),
            "request_hashes": request_hashes,
            "notes": notes,
        },
    }
    return document, notes


def merge_mtss3d(
    *,
    sample_id: str,
    sequence_id: str | None,
    assets: dict[str, Any],
    timeline: dict[str, Any],
    observations: dict[str, Any],
    identity: dict[str, Any],
    visual_proposal: dict[str, Any],
    audio_proposal: dict[str, Any] | None,
    geometry: dict[str, Any],
    blocking_motion: list[dict[str, Any]],
    caption_model: str | None,
    request_hashes: dict[str, str],
    omni_proposal: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    if omni_proposal is not None:
        return _merge_omni_mtss3d(
            sample_id=sample_id,
            sequence_id=sequence_id,
            assets=assets,
            timeline=timeline,
            identity=identity,
            omni_proposal=omni_proposal,
            geometry=geometry,
            blocking_motion=blocking_motion,
            caption_model=caption_model,
            request_hashes=request_hashes,
            dry_run=dry_run,
        )
    notes: list[str] = []
    bindings = identity.get("bindings", [])
    known_ref_ids = {str(binding["ref_id"]) for binding in bindings}
    visual_references = {
        str(item["ref_id"]): item for item in visual_proposal.get("references", [])
    }
    observation_map = {
        str(item["observation_id"]): item for item in observations["observations"]
    }

    evidence = []
    for observation in observations["observations"]:
        evidence.append(
            {
                "evidence_id": f"EVID_{observation['observation_id']}",
                "kind": "observed",
                "source_refs": [observation["image"]],
                "confidence": 1.0,
            }
        )

    references = []
    for index, binding in enumerate(bindings, start=1):
        ref_id = str(binding["ref_id"])
        proposal = visual_references.get(ref_id)
        if proposal is None:
            notes.append(f"Visual proposal omitted {ref_id}; inserted conservative placeholder")
            proposal = {
                "semantic_description": f"输入视频中由{ref_id}标识的可见人物",
                "id_features": "外观细节需要人工复核",
                "attributes": {
                    "clothing": None,
                    "hairstyle": None,
                    "accessories": None,
                },
                "confidence": 0.0,
            }
        evidence_id = f"EVID_REFERENCE_{index}"
        evidence.append(
            {
                "evidence_id": evidence_id,
                "kind": "derived",
                "source_refs": [
                    binding["appearance_image"],
                    f"sam3_track:{binding['source_track']['sam3_track_id']}",
                ],
                "confidence": float(proposal.get("confidence", 0.0)),
            }
        )
        references.append(
            {
                "ref_id": ref_id,
                "type": "person",
                "semantic_description": str(proposal["semantic_description"]),
                "timestamp": float(binding.get("anchor_timestamp_sec", 0.0)),
                "appearance_anchor": {
                    "image_refs": [str(binding["appearance_image"])],
                    "id_features": {
                        "detail_description": str(proposal["id_features"])
                    },
                    "attributes": {
                        "clothing": proposal["attributes"].get("clothing"),
                        "hairstyle": proposal["attributes"].get("hairstyle"),
                        "accessories": proposal["attributes"].get("accessories"),
                    },
                },
                "evidence_ids": [evidence_id],
            }
        )

    events = []
    audio_proposal = audio_proposal or {
        "events": [],
        "global_audio": {"background_music": None, "ambient_sound": None},
    }
    for index, proposal in enumerate(audio_proposal.get("events", []), start=1):
        evidence_id = f"EVID_AUDIO_EVENT_{index}"
        confidence = float(proposal.get("confidence", 0.0))
        evidence.append(
            {
                "evidence_id": evidence_id,
                "kind": "observed",
                "source_refs": ["audio_transcript"],
                "confidence": confidence,
            }
        )
        events.append(
            {
                "event_id": f"EVENT_{index}",
                "type": proposal["type"],
                "time_range": _event_time_range(proposal, float(timeline["fps"])),
                "content": {
                    "speaker": proposal.get("speaker"),
                    "line": proposal.get("line"),
                    "description": proposal.get("description"),
                },
                "evidence_ids": [evidence_id],
            }
        )

    shot_proposal = visual_proposal["shot"]
    valid_segments = []
    for segment in shot_proposal.get("description_segments", []):
        observation = observation_map.get(str(segment["observation_id"]))
        if observation is None:
            notes.append(
                f"Unknown observation ID {segment['observation_id']} was omitted from description"
            )
            continue
        valid_segments.append((float(observation["timestamp_sec"]), str(segment["description"])))
    valid_segments.sort(key=lambda item: item[0])
    if valid_segments:
        visual_description = " ".join(
            f"[{timestamp:.3f}s] {description}" for timestamp, description in valid_segments
        )
    else:
        visual_description = "[0.000s] 该镜头的具体视觉动作需要人工复核。"
        notes.append("No valid visual description segment was returned")

    requested_refs = [str(value) for value in shot_proposal.get("references_in_shot", [])]
    unknown_requested_refs = sorted(set(requested_refs) - known_ref_ids)
    if unknown_requested_refs:
        notes.append(f"Visual proposal referenced unknown IDs: {unknown_requested_refs}")
    references_in_shot = [value for value in requested_refs if value in known_ref_ids]

    frame_count = int(timeline["frame_count"])
    fps = float(timeline["fps"])
    shot = {
        "shot_id": "SHOT_1",
        "time_range": {
            "start_frame": 0,
            "end_frame_exclusive": frame_count,
            "start_sec": 0.0,
            "end_sec": float(timeline["duration_sec"]),
            "temporal_precision": "frame_exact",
        },
        "visual_description": visual_description,
        "camera": {
            "shot_type": shot_proposal["camera"].get("shot_type"),
            "angle": shot_proposal["camera"].get("angle"),
            "movement": shot_proposal["camera"].get("movement"),
        },
        "references_in_shot": references_in_shot,
        "active_events": [event["event_id"] for event in events],
        "evidence_ids": [
            f"EVID_{observation['observation_id']}"
            for observation in observations["observations"]
        ],
    }

    global_proposal = visual_proposal["global"]
    global_audio = audio_proposal.get("global_audio", {})
    global_stream = {
        "scene_description": str(global_proposal["scene_description"]),
        "global_style": str(global_proposal["global_style"]),
        "global_audio": {
            "background_music": global_audio.get("background_music"),
            "ambient_sound": global_audio.get("ambient_sound"),
        },
        "evidence_ids": shot["evidence_ids"],
    }

    if geometry.get("files"):
        evidence.append(
            {
                "evidence_id": "EVID_GEOMETRY",
                "kind": "derived",
                "source_refs": [item["path"] for item in geometry["files"]],
                "confidence": None,
            }
        )

    needs_review = bool(
        notes
        or any(binding["identity_status"] == "anonymous" for binding in bindings)
    )
    status = "dry_run_placeholder" if dry_run else ("needs_review" if needs_review else "complete")
    document = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "sequence_id": sequence_id,
        "status": status,
        "assets": assets,
        "timeline": timeline,
        "mtss": {
            "references": references,
            "shots": [shot],
            "events": events,
            "global": global_stream,
        },
        "extensions": {
            "mtss_3d": {
                "control_contract": CONTROL_CONTRACT,
                "control_scope": {
                    "preserve": PRESERVE_SCOPE,
                    "ignore": IGNORE_SCOPE,
                },
                "entity_bindings": [
                    _binding_for_schema(binding) for binding in bindings
                ],
                "blocking_motion": blocking_motion,
                "geometry": geometry,
                "evidence": evidence,
            }
        },
        "annotation_meta": {
            "caption_model": caption_model,
            "visual_prompt_version": "visual_v1",
            "audio_prompt_version": "audio_v1",
            "identity_registry_version": identity.get("registry_version"),
            "created_at": now_iso(),
            "request_hashes": request_hashes,
            "notes": notes,
        },
    }
    return document, notes
