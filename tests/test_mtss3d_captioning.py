from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stagerecon.captioning.io import validate_with_schema, write_json
from stagerecon.captioning.identity import _person_ref_id, build_episode_registry
from stagerecon.captioning.merge import merge_mtss3d
from stagerecon.captioning.observations import sample_observations
from stagerecon.captioning.omni_normalize import normalize_omni_proposal
from stagerecon.captioning.prompt_builder import build_training_record
from stagerecon.captioning.providers.qwen import parse_json_response
from stagerecon.captioning.validation import (
    validate_mtss3d,
    validate_omni_proposal,
    validate_visual_proposal,
)


class QwenParsingTests(unittest.TestCase):
    def test_parse_json_response_accepts_fenced_json(self) -> None:
        value = parse_json_response('```json\n{"answer": 1}\n```')
        self.assertEqual(value, {"answer": 1})

    def test_parse_json_response_extracts_object_from_text(self) -> None:
        value = parse_json_response('result: {"answer": 2} done')
        self.assertEqual(value, {"answer": 2})

    def test_named_identity_becomes_stable_person_ref_id(self) -> None:
        self.assertEqual(_person_ref_id("friends:monica", 1), "PERSON_MONICA")
        self.assertEqual(_person_ref_id("friends:chandler", 2), "PERSON_CHANDLER")
        self.assertEqual(
            _person_ref_id("friends:S08E02:person_009", 3), "PERSON_009"
        )
        self.assertEqual(_person_ref_id(None, 3), "PERSON_3")

    def test_episode_registry_uses_generated_reference_bank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "S08E02"
            person = episode / "person_009"
            person.mkdir(parents=True)
            (episode / "manifest.json").write_text(
                json.dumps({"episode_id": "S08E02"}), encoding="utf-8"
            )
            (person / "reference_bank.png").write_bytes(b"image")
            output = root / "registry.json"
            registry = build_episode_registry(episode_dir=episode, output=output)
            character = registry["characters"][0]
            self.assertEqual(
                character["canonical_identity"], "friends:S08E02:person_009"
            )
            self.assertEqual(_person_ref_id(character["canonical_identity"], 1), "PERSON_009")
            self.assertEqual(
                (output.parent / character["gallery"][0]["path"]).resolve(),
                (person / "reference_bank.png").resolve(),
            )


class ObservationTests(unittest.TestCase):
    def test_sampling_keeps_exact_start_and_end_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames_dir = root / "frames"
            frames_dir.mkdir()
            frames = []
            for index in range(5):
                name = f"{index:06d}.jpg"
                (frames_dir / name).write_bytes(f"frame-{index}".encode())
                frames.append(
                    {
                        "frame_index": index,
                        "frame_name": name,
                        "timestamp_seconds": index / 2.0,
                    }
                )
            metadata_path = root / "frames.json"
            write_json(
                metadata_path,
                {"fps": 2.0, "frame_count": 5, "frames": frames},
            )
            result = sample_observations(
                frames_path=metadata_path,
                frames_dir=frames_dir,
                output_dir=root / "observations",
                caption_root=root,
                sample_fps=1.0,
                max_observations=16,
            )
            indices = [item["frame_index"] for item in result["observations"]]
            self.assertEqual(indices[0], 0)
            self.assertEqual(indices[-1], 4)
            self.assertEqual(
                result["observations"][-1]["timestamp_sec"], 2.0
            )


class ProposalValidationTests(unittest.TestCase):
    def test_omni_aliases_normalize_without_previs_context(self) -> None:
        raw = {
            "reference_bank": [
                {"ref_id": "PERSON_1", "appearance_image": "person.png"}
            ],
            "shots": [
                {
                    "shot_id": "SHOT_1", "start_sec": 0.0, "end_sec": 2.0,
                    "description": "PERSON_1 crosses the room.",
                    "shot_scale": "medium wide", "camera_motion": "static",
                    "active_event_ids": ["EVENT_1"],
                }
            ],
            "events": [
                {
                    "event_id": "EVENT_1", "type": "sound_effect",
                    "start_sec": 0.5, "end_sec": 1.0,
                    "speaker_ref_id": None, "text": None,
                    "speaker_evidence": None,
                }
            ],
            "global": {
                "setting": "An apartment.", "visual_style": "Live action.",
                "background_music": None, "ambient_sound": "Room tone.",
            },
        }
        proposal = normalize_omni_proposal(raw, allowed_ref_ids={"PERSON_1"})
        self.assertEqual(proposal["events"][0]["type"], "sfx")
        self.assertEqual(proposal["shots"][0]["camera"]["shot_type"], "medium wide")
        self.assertEqual(proposal["shots"][0]["references_in_shot"], ["PERSON_1"])
        self.assertEqual(
            validate_omni_proposal(
                proposal, allowed_ref_ids={"PERSON_1"}, duration_sec=2.0
            ),
            [],
        )

    def test_nested_annotations_normalize(self) -> None:
        raw = {
            "reference_bank": [{"ref_id": "PERSON_1"}],
            "annotations": {
                "shots": [
                    {
                        "shot_id": "SHOT_1", "start_sec": 0.0, "end_sec": 2.0,
                        "description": "PERSON_1 speaks in a room.",
                        "active_event_ids": ["EVENT_1"],
                    }
                ],
                "events": [
                    {
                        "event_id": "EVENT_1", "type": "dialogue",
                        "time_range": [0.2, 1.3],
                        "dialogue": {
                            "speaker_ref_id": "PERSON_1", "line": "Hello.",
                            "speaker_evidence": "Visible mouth movement.",
                        },
                    }
                ],
                "global": {
                    "scene_description": "A room.", "visual_style": "Live action.",
                    "background_music": None, "ambient_sound": None,
                },
            },
        }
        proposal = normalize_omni_proposal(raw, allowed_ref_ids={"PERSON_1"})
        self.assertEqual(proposal["events"][0]["line"], "Hello.")
        self.assertEqual(proposal["events"][0]["speaker"], "PERSON_1")
        self.assertEqual(
            validate_omni_proposal(
                proposal, allowed_ref_ids={"PERSON_1"}, duration_sec=2.0
            ),
            [],
        )

    def test_visual_proposal_rejects_unknown_reference(self) -> None:
        proposal = {
            "references": [
                {
                    "ref_id": "PERSON_2",
                    "semantic_description": "一个人物",
                    "id_features": "短发",
                    "attributes": {
                        "clothing": None,
                        "hairstyle": "短发",
                        "accessories": None,
                    },
                    "confidence": 0.8,
                }
            ],
            "shot": {
                "description_segments": [
                    {"observation_id": "OBS_0001", "description": "人物走入画面"}
                ],
                "camera": {
                    "shot_type": "medium",
                    "angle": "eye_level",
                    "movement": "static",
                },
                "references_in_shot": ["PERSON_2"],
                "confidence": 0.8,
            },
            "global": {
                "scene_description": "室内",
                "global_style": "写实",
                "ambient_sound": None,
            },
        }
        errors = validate_visual_proposal(
            proposal,
            allowed_ref_ids={"PERSON_1"},
            observation_ids={"OBS_0001"},
        )
        self.assertTrue(any("unknown IDs" in error for error in errors))

    def test_omni_proposal_rejects_unknown_speaker(self) -> None:
        proposal = {
            "references": [
                {
                    "ref_id": "PERSON_MONICA",
                    "semantic_description": "Monica is visible in the apartment.",
                    "id_features": "Dark hair and a blue top.",
                    "attributes": {
                        "clothing": "blue top",
                        "hairstyle": "dark shoulder-length hair",
                        "accessories": None,
                    },
                    "confidence": 0.9,
                }
            ],
            "shots": [
                {
                    "shot_id": "SHOT_1",
                    "start_sec": 0.0,
                    "end_sec": 2.0,
                    "visual_description": "Monica speaks in a medium shot.",
                    "camera": {"shot_type": "medium", "angle": "eye_level", "movement": "static"},
                    "references_in_shot": ["PERSON_MONICA"],
                    "active_event_ids": ["EVENT_1"],
                    "confidence": 0.9,
                }
            ],
            "events": [
                {
                    "event_id": "EVENT_1",
                    "type": "dialogue",
                    "start_sec": 0.2,
                    "end_sec": 1.2,
                    "speaker": "PERSON_UNKNOWN",
                    "line": "Hi.",
                    "description": None,
                    "speaker_evidence": {
                        "visible_on_screen": True,
                        "mouth_activity": True,
                        "voice_continuity": None,
                    },
                    "confidence": 0.7,
                }
            ],
            "global": {
                "scene_description": "An apartment interior.",
                "global_style": "Live-action sitcom.",
                "global_audio": {"background_music": None, "ambient_sound": "Room tone."},
            },
        }
        errors = validate_omni_proposal(
            proposal, allowed_ref_ids={"PERSON_MONICA"}, duration_sec=2.0
        )
        self.assertTrue(any("unknown speaker" in error for error in errors))


class MergeAndPromptTests(unittest.TestCase):
    def _document(self) -> dict:
        zeros = "0" * 64
        assets = {
            "source_video": {"path": "source.mp4", "sha256": zeros},
            "previs_video": {
                "path": "previs.mp4",
                "sha256": zeros,
                "role": "coarse_3d_previs",
            },
            "target_video": {"path": "target.mp4", "sha256": zeros},
            "audio": None,
        }
        timeline = {
            "fps": 24.0,
            "frame_count": 48,
            "duration_sec": 2.0,
            "time_unit": "seconds",
            "interval_convention": "[start_sec,end_sec)",
        }
        observations = {
            "observations": [
                {
                    "observation_id": "OBS_0001",
                    "frame_index": 0,
                    "timestamp_sec": 0.0,
                    "image": "observations/frames/000000.jpg",
                    "reason": ["shot_start"],
                }
            ]
        }
        identity = {
            "registry_version": "friends_v1",
            "bindings": [
                {
                    "ref_id": "PERSON_1",
                    "canonical_identity": None,
                    "identity_status": "anonymous",
                    "source_track": {
                        "sam3_track_id": 0,
                        "first_frame": 0,
                        "last_frame": 47,
                    },
                    "appearance_image": "identity/person_1.png",
                    "anchor_origin": "current_target_shot",
                    "anchor_timestamp_sec": 0.0,
                    "match": None,
                    "candidate_matches": [],
                }
            ],
        }
        visual = {
            "references": [
                {
                    "ref_id": "PERSON_1",
                    "semantic_description": "一名站在室内的人物",
                    "id_features": "深色上衣，短发",
                    "attributes": {
                        "clothing": "深色上衣",
                        "hairstyle": "短发",
                        "accessories": None,
                    },
                    "confidence": 0.8,
                }
            ],
            "shot": {
                "description_segments": [
                    {"observation_id": "OBS_0001", "description": "人物站在画面中央"}
                ],
                "camera": {
                    "shot_type": "medium",
                    "angle": "eye_level",
                    "movement": "static",
                },
                "references_in_shot": ["PERSON_1"],
                "confidence": 0.8,
            },
            "global": {
                "scene_description": "室内客厅",
                "global_style": "写实影视画面",
                "ambient_sound": None,
            },
        }
        document, notes = merge_mtss3d(
            sample_id="sample_1",
            sequence_id=None,
            assets=assets,
            timeline=timeline,
            observations=observations,
            identity=identity,
            visual_proposal=visual,
            audio_proposal=None,
            geometry={
                "status": "missing",
                "producer": None,
                "coordinate_system": None,
                "files": [],
            },
            blocking_motion=[],
            caption_model="test-model",
            request_hashes={"visual": "request-hash"},
        )
        self.assertEqual(notes, [])
        return document

    def test_merged_document_validates_and_keeps_geometry_separate(self) -> None:
        document = self._document()
        report = validate_mtss3d(
            document,
            base_dir=Path("."),
            check_assets=False,
            check_hashes=False,
        )
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(document["mtss"]["shots"][0]["shot_id"], "SHOT_1")
        self.assertIn("geometry", document["extensions"]["mtss_3d"])
        self.assertNotIn("geometry", document["mtss"]["shots"][0])

    def test_training_prompt_freezes_previs_contract(self) -> None:
        document = self._document()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mtss_path = root / "mtss_3d.json"
            mtss_path.write_text(json.dumps(document), encoding="utf-8")
            record, errors = build_training_record(
                document=document,
                mtss_path=mtss_path,
                output=root / "training_record.json",
            )
            self.assertEqual(errors, [])
            self.assertEqual(
                validate_with_schema(record, "training_record_v1.schema.json"), []
            )
            self.assertIn("subject_definitions:", record["prompt"])
            self.assertIn("<Subject 1>", record["prompt"])
            self.assertIn("<Video 1> is the coarse 3D previs", record["prompt"])
            self.assertIn(
                "<Video 1> (structural guidance only): partially_preserved",
                record["prompt"],
            )
            self.assertIn("subject root positions, movement paths", record["prompt"])
            self.assertIn("Do not preserve proxy geometry", record["prompt"])
            self.assertNotIn(
                "<Video 1> (composition, blocking, camera trajectory, and timing): fully_preserved",
                record["prompt"],
            )
            self.assertIn("detailed_description:", record["prompt"])
            self.assertIn("overall_soundscape:", record["prompt"])
            self.assertNotIn("<MTSS>", record["prompt"])
            self.assertNotIn("evidence_ids", record["prompt"])

    def test_omni_merge_keeps_multishot_events_and_speaker_evidence(self) -> None:
        zeros = "0" * 64
        assets = {
            "source_video": {"path": "source.mp4", "sha256": zeros},
            "previs_video": {"path": "previs.mp4", "sha256": zeros, "role": "coarse_3d_previs"},
            "target_video": {"path": "target.mp4", "sha256": zeros},
            "audio": {"path": "target.mp4", "sha256": zeros, "role": "original_audio_source"},
        }
        timeline = {
            "fps": 24.0, "frame_count": 96, "duration_sec": 4.0,
            "time_unit": "seconds", "interval_convention": "[start_sec,end_sec)",
        }
        identity = {
            "registry_version": "friends_v1",
            "bindings": [
                {
                    "ref_id": "PERSON_MONICA", "canonical_identity": "MONICA",
                    "identity_status": "matched",
                    "source_track": {"sam3_track_id": 0, "first_frame": 0, "last_frame": 95},
                    "appearance_image": "identity/monica.png",
                    "anchor_origin": "character_gallery", "anchor_timestamp_sec": 0.0,
                    "match": {
                        "method": "insightface", "score": 0.95,
                        "second_best_score": 0.4, "margin": 0.55,
                        "observation_count": 3, "model_version": "test",
                    },
                    "candidate_matches": [{"canonical_identity": "MONICA", "score": 0.95}],
                }
            ],
        }
        omni = {
            "references": [
                {
                    "ref_id": "PERSON_MONICA",
                    "semantic_description": "Monica stands in the apartment.",
                    "id_features": "Dark hair and a blue top.",
                    "attributes": {"clothing": "blue top", "hairstyle": "dark hair", "accessories": None},
                    "confidence": 0.95,
                }
            ],
            "shots": [
                {
                    "shot_id": "SHOT_1", "start_sec": 0.0, "end_sec": 2.0,
                    "visual_description": "Monica turns toward the doorway.",
                    "camera": {"shot_type": "medium", "angle": "eye_level", "movement": "static"},
                    "references_in_shot": ["PERSON_MONICA"], "active_event_ids": ["EVENT_1"], "confidence": 0.9,
                },
                {
                    "shot_id": "SHOT_2", "start_sec": 2.0, "end_sec": 4.0,
                    "visual_description": "A closer angle holds as Monica finishes speaking.",
                    "camera": {"shot_type": "close_up", "angle": "eye_level", "movement": "static"},
                    "references_in_shot": ["PERSON_MONICA"], "active_event_ids": ["EVENT_2"], "confidence": 0.9,
                },
            ],
            "events": [
                {
                    "event_id": "EVENT_1", "type": "dialogue",
                    "start_sec": 0.4, "end_sec": 1.5, "speaker": "PERSON_MONICA",
                    "line": "Hey, where are you going?", "description": None,
                    "speaker_evidence": {"visible_on_screen": True, "mouth_activity": True, "voice_continuity": True},
                    "confidence": 0.94,
                },
                {
                    "event_id": "EVENT_2", "type": "sfx",
                    "start_sec": 2.2, "end_sec": 2.5, "speaker": None, "line": None,
                    "description": "A door closes.",
                    "speaker_evidence": {"visible_on_screen": None, "mouth_activity": None, "voice_continuity": None},
                    "confidence": 0.8,
                },
            ],
            "global": {
                "scene_description": "A warmly lit apartment living room.",
                "global_style": "Live-action multi-camera sitcom.",
                "global_audio": {"background_music": None, "ambient_sound": "Interior room tone."},
            },
        }
        document, notes = merge_mtss3d(
            sample_id="omni_sample", sequence_id=None, assets=assets, timeline=timeline,
            observations={"observations": []}, identity=identity,
            visual_proposal=omni, audio_proposal=None, omni_proposal=omni,
            geometry={"status": "missing", "producer": None, "coordinate_system": None, "files": []},
            blocking_motion=[], caption_model="qwen3.8-omni-flash",
            request_hashes={"omni_joint": "hash"},
        )
        self.assertEqual(notes, [])
        self.assertEqual(len(document["mtss"]["shots"]), 2)
        self.assertEqual(document["mtss"]["events"][0]["content"]["speaker"], "PERSON_MONICA")
        self.assertTrue(document["mtss"]["events"][0]["content"]["speaker_evidence"]["mouth_activity"])
        report = validate_mtss3d(document, base_dir=Path("."), check_assets=False, check_hashes=False)
        self.assertTrue(report["valid"], report["errors"])


if __name__ == "__main__":
    unittest.main()
