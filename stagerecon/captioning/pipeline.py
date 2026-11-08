"""End-to-end MTSS-3D captioning orchestration for one StageRecon clip run."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from .geometry import build_blocking_motion, package_geometry
from .identity import (
    bind_identities,
    bind_registry_references,
    keep_qwen_selected_references,
)
from .io import (
    PROMPT_ROOT,
    SCHEMA_ROOT,
    now_iso,
    read_json,
    relative_path,
    resolve_reference,
    sha256_file,
    sha256_text,
    write_json,
)
from .manifest import build_input_manifest
from .merge import merge_mtss3d
from .observations import sample_observations
from .omni_normalize import normalize_omni_proposal
from .prompt_builder import build_training_record
from .prompts import audio_request_text, load_prompt, omni_request_text, visual_request_text
from .providers.qwen import QwenProvider
from .review import write_review
from .validation import (
    validate_audio_proposal,
    validate_omni_proposal,
    validate_visual_proposal,
    write_validation_report,
)


def _manual_bindings(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    value = read_json(path, "manual identity bindings")
    if "bindings" in value:
        return {
            int(item["track_id"]): str(item["canonical_identity"])
            for item in value["bindings"]
        }
    return {int(key): str(identity) for key, identity in value.items()}


def _dry_visual_proposal(
    observations: dict[str, Any], identity: dict[str, Any]
) -> dict[str, Any]:
    references = []
    for binding in identity.get("bindings", []):
        references.append(
            {
                "ref_id": binding["ref_id"],
                "semantic_description": f"由{binding['ref_id']}标识的可见人物，外观待标注",
                "id_features": "身份外观细节待标注",
                "attributes": {
                    "clothing": None,
                    "hairstyle": None,
                    "accessories": None,
                },
                "confidence": 0.0,
            }
        )
    first_observation = observations["observations"][0]["observation_id"]
    return {
        "references": references,
        "shot": {
            "description_segments": [
                {
                    "observation_id": first_observation,
                    "description": "该镜头的具体动作和互动等待语义标注。",
                }
            ],
            "camera": {"shot_type": None, "angle": None, "movement": None},
            "references_in_shot": [item["ref_id"] for item in references],
            "confidence": 0.0,
        },
        "global": {
            "scene_description": "场景内容等待语义标注",
            "global_style": "视觉风格等待语义标注",
            "ambient_sound": None,
        },
    }


def _dry_omni_proposal(identity: dict[str, Any], timeline: dict[str, Any]) -> dict[str, Any]:
    references = [
        {
            "ref_id": binding["ref_id"],
            "semantic_description": f"Visible person identified as {binding['ref_id']}",
            "id_features": "Appearance details require annotation",
            "attributes": {
                "clothing": None,
                "hairstyle": None,
                "accessories": None,
            },
            "confidence": 0.0,
        }
        for binding in identity.get("bindings", [])
    ]
    return {
        "references": references,
        "shots": [
            {
                "shot_id": "SHOT_1",
                "start_sec": 0.0,
                "end_sec": float(timeline["duration_sec"]),
                "visual_description": "Visual action and camera description require annotation.",
                "camera": {"shot_type": None, "angle": None, "movement": None},
                "references_in_shot": [item["ref_id"] for item in references],
                "active_event_ids": [],
                "confidence": 0.0,
            }
        ],
        "events": [],
        "global": {
            "scene_description": "Scene description requires annotation",
            "global_style": "Visual style requires annotation",
            "global_audio": {"background_music": None, "ambient_sound": None},
        },
    }


def _request_hash(
    *, system_prompt: str, user_text: str, images: list[tuple[str, Path]], model: str,
    videos: list[tuple[str, Path]] | None = None,
) -> str:
    value = {
        "system_prompt": system_prompt,
        "user_text": user_text,
        "model": model,
        "images": [
            {"label": label, "path": str(path), "sha256": sha256_file(path)}
            for label, path in images
        ],
        "videos": [
            {"label": label, "path": str(path), "sha256": sha256_file(path)}
            for label, path in videos or []
        ],
    }
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _request_validated(
    *,
    provider: QwenProvider,
    system_prompt: str,
    user_text: str,
    images: list[tuple[str, Path]],
    videos: list[tuple[str, Path]] | None = None,
    schema_name: str,
    validator: Callable[[dict[str, Any]], list[str]],
    raw_dir: Path,
    stem: str,
    semantic_retries: int = 2,
    normalizer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    schema = read_json(SCHEMA_ROOT / schema_name, "proposal schema")
    metadata_records = []
    correction = ""
    for attempt in range(semantic_retries + 1):
        raw_proposal, metadata = provider.request_json(
            system_prompt=system_prompt,
            user_text=user_text + correction,
            images=images,
            schema=schema,
            schema_name=Path(schema_name).stem.replace(".", "_"),
            videos=videos,
        )
        proposal = normalizer(raw_proposal) if normalizer else raw_proposal
        errors = validator(proposal)
        metadata["semantic_validation_errors"] = errors
        metadata_records.append(metadata)
        write_json(raw_dir / f"{stem}_response_attempt_{attempt + 1}.json", raw_proposal)
        if normalizer:
            write_json(raw_dir / f"{stem}_normalized_attempt_{attempt + 1}.json", proposal)
        write_json(raw_dir / f"{stem}_metadata_attempt_{attempt + 1}.json", metadata)
        if not errors:
            return proposal, metadata_records
        correction = (
            "\n\n你上一次的JSON没有通过校验。请重新输出完整JSON，并修复以下错误：\n- "
            + "\n- ".join(errors)
        )
    raise RuntimeError(
        f"{stem} proposal remained invalid after {semantic_retries + 1} attempts: "
        + "; ".join(metadata_records[-1]["semantic_validation_errors"])
    )


def _finalize_outputs(
    *,
    output_dir: Path,
    final_path: Path,
    document: dict[str, Any],
    manifest: dict[str, Any],
    config: dict[str, Any],
    registry_path: Path | None,
    identity: dict[str, Any],
    request_hashes: dict[str, str],
    dry_run: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    write_json(final_path, document)
    report = write_validation_report(
        output_dir / "validation_report.json",
        document,
        base_dir=output_dir,
    )
    _, training_errors = build_training_record(
        document=document,
        mtss_path=final_path,
        output=output_dir / "training_record.json",
    )
    if training_errors:
        report["valid"] = False
        report["status"] = "failed"
        report["errors"].extend(
            f"training_record: {error}" for error in training_errors
        )
        report["error_count"] = len(report["errors"])
        write_json(output_dir / "validation_report.json", report)
    write_review(output_dir / "review.html", document, report)
    write_json(
        output_dir / "run_manifest.json",
        {
            "schema_version": "caption_run_v1",
            "sample_id": manifest["sample_id"],
            "status": document["status"],
            "validation_status": report["status"],
            "created_at": now_iso(),
            "dry_run": dry_run,
            "config": config,
            "registry": str(registry_path.resolve()) if registry_path else None,
            "identity_backend": identity.get("backend"),
            "request_hashes": request_hashes,
            "outputs": {
                "mtss_3d": "mtss_3d.json",
                "validation": "validation_report.json",
                "training_record": "training_record.json",
                "review": "review.html",
            },
        },
    )
    return document, report


def run_pipeline(
    *,
    clip_run: Path,
    previs_video: Path,
    output_dir: Path,
    registry_path: Path | None = None,
    config_path: Path | None = None,
    sample_id: str | None = None,
    sequence_id: str | None = None,
    target_video: Path | None = None,
    audio: Path | None = None,
    transcript_path: Path | None = None,
    omni_proposal_path: Path | None = None,
    visual_proposal_path: Path | None = None,
    audio_proposal_path: Path | None = None,
    manual_bindings_path: Path | None = None,
    identity_mode: str = "auto",
    identity_backend: str | None = None,
    identity_model_name: str = "buffalo_l",
    identity_model_root: Path | None = None,
    identity_device: str = "cpu",
    qwen_base_url: str | None = None,
    qwen_model: str | None = None,
    qwen_api_key: str | None = None,
    dry_run: bool = False,
    resume: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "mtss_3d.json"
    if resume and final_path.is_file():
        document = read_json(final_path, "existing MTSS-3D document")
        report = write_validation_report(
            output_dir / "validation_report.json",
            document,
            base_dir=output_dir,
        )
        return document, report

    default_config = Path(__file__).resolve().parents[2] / "configs/mtss3d/default.json"
    config = read_json(config_path or default_config, "captioning config")
    identity_config = config.get("identity", {})
    qwen_config = dict(config.get("qwen", {}))
    if qwen_base_url:
        qwen_config["base_url"] = qwen_base_url
    if qwen_model:
        qwen_config["model"] = qwen_model

    manifest, assets = build_input_manifest(
        clip_run=clip_run,
        previs_video=previs_video,
        output_dir=output_dir,
        sample_id=sample_id,
        sequence_id=sequence_id,
        target_video=target_video,
        audio=audio,
    )
    timeline = manifest["timeline"]
    observations = sample_observations(
        frames_path=Path(manifest["inputs"]["frames"]),
        frames_dir=clip_run.resolve() / "frames",
        output_dir=output_dir / "observations",
        caption_root=output_dir,
        sample_fps=float(config.get("sample_fps", 2.0)),
        max_observations=int(config.get("max_observations", 16)),
    )
    if identity_mode not in {"auto", "qwen_direct", "track"}:
        raise ValueError(f"Unknown identity mode: {identity_mode}")
    direct_identity = identity_mode == "qwen_direct"
    if identity_mode == "auto" and registry_path is not None:
        registry_value = read_json(registry_path, "character registry")
        direct_identity = any(
            character.get("gallery")
            for character in registry_value.get("characters", [])
        )
    if direct_identity:
        if registry_path is None:
            raise ValueError("qwen_direct identity mode requires --registry")
        identity = bind_registry_references(
            registry_path=registry_path.resolve(),
            output_dir=output_dir / "identity",
            caption_root=output_dir,
        )
    else:
        identity = bind_identities(
            clip_run=clip_run.resolve(),
            person_tracks_path=Path(manifest["inputs"]["person_tracks"]),
            registry_path=registry_path.resolve() if registry_path else None,
            output_dir=output_dir / "identity",
            caption_root=output_dir,
            backend=identity_backend or str(identity_config.get("backend", "histogram")),
            accept_threshold=float(identity_config.get("accept_threshold", 0.90)),
            margin_threshold=float(identity_config.get("margin_threshold", 0.08)),
            representative_crops=int(identity_config.get("representative_crops", 5)),
            min_frame_gap=int(identity_config.get("min_frame_gap", 8)),
            model_name=identity_model_name,
            model_root=identity_model_root,
            device=identity_device,
            manual_bindings=_manual_bindings(manual_bindings_path),
        )
    geometry = package_geometry(
        clip_run=clip_run.resolve(), output_dir=output_dir / "geometry",
        caption_root=output_dir,
    )
    blocking_motion = (
        []
        if direct_identity
        else build_blocking_motion(
            clip_run=clip_run.resolve(),
            bindings=identity["bindings"],
            fps=float(timeline["fps"]),
            frame_count=int(timeline["frame_count"]),
            caption_root=output_dir,
        )
    )

    raw_dir = output_dir / "raw"
    semantic_dir = output_dir / "semantic"
    raw_dir.mkdir(exist_ok=True)
    semantic_dir.mkdir(exist_ok=True)
    allowed_ref_ids = {item["ref_id"] for item in identity["bindings"]}
    observation_ids = {
        item["observation_id"] for item in observations["observations"]
    }
    request_hashes: dict[str, str] = {}
    provider: QwenProvider | None = None
    model_name = str(qwen_config.get("model", "")) or None
    use_omni = omni_proposal_path is not None or not any(
        (transcript_path, visual_proposal_path, audio_proposal_path)
    )

    if use_omni:
        omni_system = load_prompt("mtss_3d/omni_joint_v1.txt")
        omni_user = omni_request_text(
            bindings=identity,
            duration_sec=float(timeline["duration_sec"]),
            fps=float(timeline["fps"]),
            frame_count=int(timeline["frame_count"]),
        )
        omni_images = [
            (
                f"REFERENCE {binding['ref_id']}",
                resolve_reference(binding["appearance_image"], output_dir),
            )
            for binding in identity["bindings"]
        ]
        omni_videos = [
            (
                "TARGET_VIDEO_WITH_ORIGINAL_AUDIO",
                Path(manifest["inputs"]["target_video"]),
            )
        ]
        request_hashes["omni_joint"] = _request_hash(
            system_prompt=omni_system,
            user_text=omni_user,
            images=omni_images,
            videos=omni_videos,
            model=model_name or "dry-run",
        )
        write_json(
            raw_dir / "omni_joint_request.json",
            {
                "model": model_name,
                "system_prompt_version": "omni_joint_v1",
                "user_text": omni_user,
                "videos": [
                    {"label": label, "path": relative_path(path, output_dir)}
                    for label, path in omni_videos
                ],
                "images": [
                    {"label": label, "path": relative_path(path, output_dir)}
                    for label, path in omni_images
                ],
                "request_hash": request_hashes["omni_joint"],
            },
        )
        if omni_proposal_path:
            omni_proposal = normalize_omni_proposal(
                read_json(omni_proposal_path, "Omni joint proposal"),
                allowed_ref_ids=allowed_ref_ids,
            )
        elif dry_run:
            omni_proposal = _dry_omni_proposal(identity, timeline)
        else:
            key = qwen_api_key or os.environ.get(
                str(qwen_config.get("api_key_env", "DASHSCOPE_API_KEY")), ""
            )
            provider = QwenProvider(
                base_url=str(qwen_config["base_url"]),
                model=str(qwen_config.get("model", "")),
                api_key=key,
                timeout_sec=float(qwen_config.get("timeout_sec", 600)),
                max_retries=int(qwen_config.get("max_retries", 2)),
                structured_output=str(qwen_config.get("structured_output", "json_object")),
                reasoning_effort=(
                    str(qwen_config["reasoning_effort"])
                    if qwen_config.get("reasoning_effort") is not None
                    else None
                ),
                modalities=[str(value) for value in qwen_config.get("modalities", ["text"])],
                max_tokens=int(qwen_config.get("max_tokens", 8192)),
            )
            omni_proposal, _ = _request_validated(
                provider=provider,
                system_prompt=omni_system,
                user_text=omni_user,
                images=omni_images,
                videos=omni_videos,
                schema_name="qwen_omni_mtss_proposal_v1.schema.json",
                validator=lambda value: validate_omni_proposal(
                    value,
                    allowed_ref_ids=allowed_ref_ids,
                    duration_sec=float(timeline["duration_sec"]),
                    require_all_refs=not direct_identity,
                ),
                raw_dir=raw_dir,
                stem="omni_joint",
                normalizer=lambda value: normalize_omni_proposal(
                    value, allowed_ref_ids=allowed_ref_ids
                ),
            )
        omni_errors = validate_omni_proposal(
            omni_proposal,
            allowed_ref_ids=allowed_ref_ids,
            duration_sec=float(timeline["duration_sec"]),
            require_all_refs=not direct_identity,
        )
        if omni_errors:
            write_json(raw_dir / "omni_joint_validation_errors.json", {"errors": omni_errors})
            raise ValueError("Invalid Omni joint proposal: " + "; ".join(omni_errors))
        write_json(semantic_dir / "omni_joint_proposal.json", omni_proposal)
        if direct_identity:
            identity = keep_qwen_selected_references(
                identity=identity,
                proposal=omni_proposal,
                output_dir=output_dir / "identity",
                model_name=model_name,
            )
        document, _ = merge_mtss3d(
            sample_id=str(manifest["sample_id"]),
            sequence_id=manifest.get("sequence_id"),
            assets=assets,
            timeline=timeline,
            observations=observations,
            identity=identity,
            visual_proposal=omni_proposal,
            audio_proposal=None,
            omni_proposal=omni_proposal,
            geometry=geometry,
            blocking_motion=blocking_motion,
            caption_model=model_name,
            request_hashes=request_hashes,
            dry_run=dry_run,
        )
        return _finalize_outputs(
            output_dir=output_dir,
            final_path=final_path,
            document=document,
            manifest=manifest,
            config=config,
            registry_path=registry_path,
            identity=identity,
            request_hashes=request_hashes,
            dry_run=dry_run,
        )

    visual_system = load_prompt("mtss_3d/visual_v1.txt")
    visual_user = visual_request_text(
        observations=observations,
        bindings=identity,
        duration_sec=float(timeline["duration_sec"]),
    )
    visual_images = []
    for binding in identity["bindings"]:
        visual_images.append(
            (
                f"REFERENCE {binding['ref_id']}",
                resolve_reference(binding["appearance_image"], output_dir),
            )
        )
    for observation in observations["observations"]:
        visual_images.append(
            (
                f"{observation['observation_id']} t={observation['timestamp_sec']:.6f}s",
                resolve_reference(observation["image"], output_dir),
            )
        )
    request_hashes["visual"] = _request_hash(
        system_prompt=visual_system,
        user_text=visual_user,
        images=visual_images,
        model=model_name or "dry-run",
    )
    write_json(
        raw_dir / "visual_request.json",
        {
            "model": model_name,
            "system_prompt_version": "visual_v1",
            "user_text": visual_user,
            "images": [
                {"label": label, "path": relative_path(path, output_dir)}
                for label, path in visual_images
            ],
            "request_hash": request_hashes["visual"],
        },
    )

    if visual_proposal_path:
        visual_proposal = read_json(visual_proposal_path, "visual proposal")
    elif dry_run:
        visual_proposal = _dry_visual_proposal(observations, identity)
    else:
        key = qwen_api_key or os.environ.get(
            str(qwen_config.get("api_key_env", "QWEN_API_KEY")), ""
        )
        provider = QwenProvider(
            base_url=str(qwen_config["base_url"]),
            model=str(qwen_config.get("model", "")),
            api_key=key,
            timeout_sec=float(qwen_config.get("timeout_sec", 180)),
            max_retries=int(qwen_config.get("max_retries", 2)),
            structured_output=str(qwen_config.get("structured_output", "json_schema")),
        )
        visual_proposal, _ = _request_validated(
            provider=provider,
            system_prompt=visual_system,
            user_text=visual_user,
            images=visual_images,
            schema_name="qwen_visual_proposal_v1.schema.json",
            validator=lambda value: validate_visual_proposal(
                value, allowed_ref_ids=allowed_ref_ids,
                observation_ids=observation_ids,
            ),
            raw_dir=raw_dir,
            stem="visual",
        )
    visual_errors = validate_visual_proposal(
        visual_proposal,
        allowed_ref_ids=allowed_ref_ids,
        observation_ids=observation_ids,
    )
    if visual_errors:
        write_json(raw_dir / "visual_validation_errors.json", {"errors": visual_errors})
        raise ValueError("Invalid visual proposal: " + "; ".join(visual_errors))
    write_json(semantic_dir / "visual_proposal.json", visual_proposal)

    audio_proposal = None
    if audio_proposal_path:
        audio_proposal = read_json(audio_proposal_path, "audio proposal")
    elif transcript_path and not dry_run:
        transcript = read_json(transcript_path, "timestamped transcript")
        audio_system = load_prompt("mtss_3d/audio_v1.txt")
        audio_user = audio_request_text(
            transcript=transcript,
            bindings=identity,
            duration_sec=float(timeline["duration_sec"]),
        )
        if provider is None:
            key = qwen_api_key or os.environ.get(
                str(qwen_config.get("api_key_env", "QWEN_API_KEY")), ""
            )
            provider = QwenProvider(
                base_url=str(qwen_config["base_url"]),
                model=str(qwen_config.get("model", "")),
                api_key=key,
                timeout_sec=float(qwen_config.get("timeout_sec", 180)),
                max_retries=int(qwen_config.get("max_retries", 2)),
                structured_output=str(qwen_config.get("structured_output", "json_schema")),
            )
        request_hashes["audio"] = _request_hash(
            system_prompt=audio_system,
            user_text=audio_user,
            images=[],
            model=str(qwen_config.get("model", "")),
        )
        write_json(
            raw_dir / "audio_request.json",
            {
                "model": qwen_config.get("model"),
                "system_prompt_version": "audio_v1",
                "user_text": audio_user,
                "request_hash": request_hashes["audio"],
            },
        )
        audio_proposal, _ = _request_validated(
            provider=provider,
            system_prompt=audio_system,
            user_text=audio_user,
            images=[],
            schema_name="qwen_audio_proposal_v1.schema.json",
            validator=lambda value: validate_audio_proposal(
                value, allowed_ref_ids=allowed_ref_ids,
                duration_sec=float(timeline["duration_sec"]),
            ),
            raw_dir=raw_dir,
            stem="audio",
        )
    if audio_proposal is not None:
        audio_errors = validate_audio_proposal(
            audio_proposal,
            allowed_ref_ids=allowed_ref_ids,
            duration_sec=float(timeline["duration_sec"]),
        )
        if audio_errors:
            write_json(raw_dir / "audio_validation_errors.json", {"errors": audio_errors})
            raise ValueError("Invalid audio proposal: " + "; ".join(audio_errors))
        write_json(semantic_dir / "audio_proposal.json", audio_proposal)

    document, merge_notes = merge_mtss3d(
        sample_id=str(manifest["sample_id"]),
        sequence_id=manifest.get("sequence_id"),
        assets=assets,
        timeline=timeline,
        observations=observations,
        identity=identity,
        visual_proposal=visual_proposal,
        audio_proposal=audio_proposal,
        geometry=geometry,
        blocking_motion=blocking_motion,
        caption_model=model_name,
        request_hashes=request_hashes,
        dry_run=dry_run,
    )
    if assets["audio"] is not None and audio_proposal is None:
        document["annotation_meta"]["notes"].append(
            "Audio asset exists but no transcript/audio proposal was supplied; Event Stream is empty"
        )
        if not dry_run:
            document["status"] = "needs_review"
    return _finalize_outputs(
        output_dir=output_dir,
        final_path=final_path,
        document=document,
        manifest=manifest,
        config=config,
        registry_path=registry_path,
        identity=identity,
        request_hashes=request_hashes,
        dry_run=dry_run,
    )
