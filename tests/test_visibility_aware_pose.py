"""Regressions for independent head and torso updates in close-ups."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from refine_actor_pose import apply_orientations


def sample(tid, shoulders=False, yaw=None):
    pose = None
    if yaw is not None:
        vectors = np.zeros((22, 3))
        vectors[0, 1] = yaw
        pose = {"rotvecs": vectors}
    return dict(
        track_id=tid,
        actor={"facing_world": [1, 0, 0]},
        fit=pose,
        shoulders=shoulders,
        source="whole" if pose else "missing",
    )


class VisibilityTests(unittest.TestCase):
    def test_head_only_keeps_previous_world_torso_of_same_track(self):
        rows = [
            {"frame_index": 0, "people": [sample(3, True, 0.0)]},
            {
                "frame_index": 1,
                "people": [sample(3, False, 1.0), sample(4, False, 1.0)],
            },
        ]
        cameras = {
            0: {"world_to_camera_rotation": np.eye(3)},
            1: {"world_to_camera_rotation": np.eye(3)},
        }
        apply_orientations(rows, cameras, {})
        current, other = [p["actor"]["proxy_orientation"] for p in rows[1]["people"]]
        self.assertEqual(current["body_reference_frame"], 0)
        self.assertEqual(current["head_source"], "whole")
        np.testing.assert_allclose(current["body_facing_world"], [0, 0, 1])
        self.assertFalse(
            np.allclose(
                np.array(current["head_basis_world"])[:, 0],
                current["body_facing_world"],
            )
        )
        self.assertEqual(other["body_source"], "actor_track")
        self.assertIsNone(other["body_reference_frame"])
        np.testing.assert_allclose(other["body_facing_world"], [1, 0, 0])

    def test_missing_head_expires_without_removing_actor(self):
        rows = [
            {"frame_index": i, "people": [sample(7, i == 0, 0.0 if i == 0 else None)]}
            for i in (0, 2, 5)
        ]
        apply_orientations(rows, {0: {"world_to_camera_rotation": np.eye(3)}}, {})
        states = [r["people"][0]["actor"]["proxy_orientation"] for r in rows]
        self.assertEqual(
            [p["head_source"] for p in states],
            ["whole", "previous_head", "actor_track"],
        )
        self.assertNotIn("head_basis_world", states[-1])
        self.assertEqual(states[-1]["body_reference_frame"], 0)
