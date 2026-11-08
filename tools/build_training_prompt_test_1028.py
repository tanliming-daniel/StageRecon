#!/usr/bin/env python3
"""Build one schema-valid MTSS-3D training prompt test sample."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stagerecon.captioning.io import file_asset, sha256_text, write_json
from stagerecon.captioning.merge import merge_mtss3d
from stagerecon.captioning.prompt_builder import build_training_record
from stagerecon.captioning.validation import validate_mtss3d


OUT = ROOT / "outputs/training_prompt_test_1028"
TARGET_SOURCE = ROOT / "data/clips/1028_1791454112.mp4"
PREVIS = OUT / "assets/coarse_previs_test.mp4"
TARGET = OUT / "assets/target_friends_1028.mp4"
MONICA = OUT / "references/PERSON_MONICA.jpg"
CHANDLER = OUT / "references/PERSON_CHANDLER.jpg"


def main() -> int:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if not TARGET.is_file():
        shutil.copy2(TARGET_SOURCE, TARGET)

    assets = {
        "source_video": file_asset(TARGET, OUT),
        "previs_video": file_asset(PREVIS, OUT, role="coarse_3d_previs"),
        "target_video": file_asset(TARGET, OUT),
        "audio": file_asset(TARGET, OUT),
    }
    timeline = {
        "fps": 30.0,
        "frame_count": 239,
        "duration_sec": 8.011,
        "time_unit": "seconds",
        "interval_convention": "[start_sec,end_sec)",
    }
    observations = {
        "observations": [
            {"observation_id": "OBS_0001", "frame_index": 0, "timestamp_sec": 0.0, "image": "references/PERSON_MONICA.jpg", "reason": ["shot_start"]},
            {"observation_id": "OBS_0002", "frame_index": 90, "timestamp_sec": 3.0, "image": "references/PERSON_CHANDLER.jpg", "reason": ["speaker_change"]},
            {"observation_id": "OBS_0003", "frame_index": 156, "timestamp_sec": 5.2, "image": "references/PERSON_MONICA.jpg", "reason": ["interaction"]},
        ]
    }
    identity = {
        "registry_version": "friends_v1_test",
        "bindings": [
            {
                "ref_id": "PERSON_MONICA",
                "canonical_identity": "friends:monica",
                "identity_status": "manual",
                "source_track": {"sam3_track_id": 1, "first_frame": 0, "last_frame": 238},
                "appearance_image": "references/PERSON_MONICA.jpg",
                "anchor_origin": "manual",
                "anchor_timestamp_sec": 5.2,
                "match": None,
                "candidate_matches": [],
            },
            {
                "ref_id": "PERSON_CHANDLER",
                "canonical_identity": "friends:chandler",
                "identity_status": "manual",
                "source_track": {"sam3_track_id": 2, "first_frame": 0, "last_frame": 238},
                "appearance_image": "references/PERSON_CHANDLER.jpg",
                "anchor_origin": "manual",
                "anchor_timestamp_sec": 5.2,
                "match": None,
                "candidate_matches": [],
            },
        ],
    }
    visual = {
        "references": [
            {
                "ref_id": "PERSON_MONICA",
                "semantic_description": "An adult woman with long dark hair, a light cardigan, a pale top, a green shoulder bag, and travel documents.",
                "id_features": "Long dark wavy hair, a light cardigan, and a green shoulder bag.",
                "attributes": {"clothing": "Light cardigan, pale top, and khaki trousers", "hairstyle": "Long dark wavy hair", "accessories": "Green shoulder bag and travel documents"},
                "confidence": 0.98,
            },
            {
                "ref_id": "PERSON_CHANDLER",
                "semantic_description": "An adult man with short brown hair, a blue plaid shirt, and a white undershirt.",
                "id_features": "Short brown hair and a blue plaid shirt over a white undershirt.",
                "attributes": {"clothing": "Blue plaid shirt and white undershirt", "hairstyle": "Short brown hair", "accessories": None},
                "confidence": 0.98,
            },
        ],
        "shot": {
            "description_segments": [
                {"observation_id": "OBS_0001", "description": "In the apartment living room, PERSON_MONICA enters carrying a shoulder bag and travel documents while PERSON_CHANDLER rises near the sofa and approaches her."},
                {"observation_id": "OBS_0002", "description": "They move toward the front of the room as PERSON_CHANDLER talks to PERSON_MONICA with visible excitement."},
                {"observation_id": "OBS_0003", "description": "The camera settles into a medium two-shot. PERSON_MONICA sorts and displays the travel documents while PERSON_CHANDLER watches beside her and playfully sings."},
            ],
            "camera": {"shot_type": "medium_wide_to_medium_two_shot", "angle": "eye_level", "movement": "gentle_tracking_then_steady"},
            "references_in_shot": ["PERSON_MONICA", "PERSON_CHANDLER"],
            "confidence": 0.9,
        },
        "global": {
            "scene_description": "A warm, bright apartment living room with a sofa, coffee table, large windows, patterned curtains, television cabinet, refrigerator, and indoor plants.",
            "global_style": "Realistic live-action sitcom cinematography with warm interior lighting and natural performances.",
            "ambient_sound": "Indoor dialogue, light footsteps, clothing movement, and paper rustle.",
        },
    }
    audio = {
        "events": [
            {"type": "dialogue", "start_sec": 0.74, "end_sec": 1.62, "speaker": "PERSON_CHANDLER", "line": "Hey babe.", "description": "an affectionate, excited greeting", "confidence": 0.98},
            {"type": "dialogue", "start_sec": 1.88, "end_sec": 3.44, "speaker": "PERSON_CHANDLER", "line": "Aren't you excited? We're going on a honeymoon?", "description": "an eager, upbeat conversational delivery", "confidence": 0.98},
            {"type": "dialogue", "start_sec": 3.52, "end_sec": 4.42, "speaker": "PERSON_MONICA", "line": "Yeah, I am.", "description": "a brief, calm, affirmative response", "confidence": 0.96},
            {"type": "dialogue", "start_sec": 4.62, "end_sec": 7.84, "speaker": "PERSON_CHANDLER", "line": "Bermuda, Bahama, come on pretty mama.", "description": "a playful, lightly sung delivery", "confidence": 0.97},
        ],
        "global_audio": {"background_music": None, "ambient_sound": "Quiet indoor room tone with light footsteps, clothing movement, and paper rustle."},
    }
    blocking_motion = [
        {
            "motion_id": "MOTION_MONICA_ENTRY",
            "ref_id": "PERSON_MONICA",
            "source_track_id": 1,
            "time_range": {"start_frame": 0, "end_frame_exclusive": 150, "start_sec": 0.0, "end_sec": 5.0, "temporal_precision": "sampled_estimate"},
            "trajectory_ref": "assets/coarse_previs_test.mp4#PERSON_MONICA",
            "meaning": "从房间后部走向前方并在另一人物身旁停下。",
        },
        {
            "motion_id": "MOTION_CHANDLER_APPROACH",
            "ref_id": "PERSON_CHANDLER",
            "source_track_id": 2,
            "time_range": {"start_frame": 30, "end_frame_exclusive": 150, "start_sec": 1.0, "end_sec": 5.0, "temporal_precision": "sampled_estimate"},
            "trajectory_ref": "assets/coarse_previs_test.mp4#PERSON_CHANDLER",
            "meaning": "从座位附近起身并靠近另一人物。",
        },
    ]
    document, notes = merge_mtss3d(
        sample_id="friends_1028_training_prompt_test",
        sequence_id="friends_s08e09_test",
        assets=assets,
        timeline=timeline,
        observations=observations,
        identity=identity,
        visual_proposal=visual,
        audio_proposal=audio,
        geometry={"status": "missing", "producer": None, "coordinate_system": None, "files": []},
        blocking_motion=blocking_motion,
        caption_model="qwen3.8-omni-flash + MOSS-Transcribe-Diarize",
        request_hashes={
            "omni_official_prompt": sha256_text((ROOT / "outputs/qwen_video_probe/manifest_omni_official_prompt.json").read_text()),
            "moss_transcript": sha256_text((ROOT / "outputs/1028_1791454112/captions/moss_transcript.json").read_text()),
        },
    )
    document["annotation_meta"]["notes"].extend([
        "Formal training-prompt structure test; coarse_previs_test.mp4 is a synthetic stand-in, not Blender ground truth.",
        "Dialogue timestamps originate from MOSS; speaker-to-Person mapping was cross-checked with Qwen3.8-Omni joint audio-video analysis.",
    ])
    mtss_path = OUT / "mtss_3d.json"
    write_json(mtss_path, document)
    validation = validate_mtss3d(document, base_dir=OUT, check_assets=True, check_hashes=True)
    write_json(OUT / "validation_report.json", validation)
    record, record_errors = build_training_record(
        document=document, mtss_path=mtss_path, output=OUT / "training_record.json"
    )
    (OUT / "training_prompt.txt").write_text(record["prompt"] + "\n", encoding="utf-8")
    report = {
        "mtss_validation": validation,
        "training_record_errors": record_errors,
        "merge_notes": notes,
        "prompt_chars": len(record["prompt"]),
        "outputs": {
            "mtss": str(mtss_path),
            "training_record": str(OUT / "training_record.json"),
            "training_prompt": str(OUT / "training_prompt.txt"),
        },
    }
    write_json(OUT / "build_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if validation["valid"] and not record_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
