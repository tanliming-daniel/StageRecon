from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from stagerecon.character_gallery import (
    CharacterCluster,
    FaceObservation,
    cluster_face_observations,
    infer_episode_id,
    select_representatives,
)


def observation(
    frame: int, descriptor: list[float], body: list[float], quality: float
) -> FaceObservation:
    return FaceObservation(
        frame_index=frame,
        timestamp_sec=float(frame),
        bbox_xywh=(10, 10, 64, 64),
        descriptor=np.asarray(descriptor, dtype=np.float32),
        body_descriptor=np.asarray(body, dtype=np.float32),
        quality=quality,
    )


class CharacterGalleryTests(unittest.TestCase):
    def test_episode_id_is_normalized(self) -> None:
        self.assertEqual(
            infer_episode_id(Path("friends.s8e5.720p.mkv")),
            "S08E05",
        )

    def test_observation_allows_source_image(self) -> None:
        item = FaceObservation(
            frame_index=42,
            timestamp_sec=1.75,
            bbox_xywh=(1, 2, 3, 4),
            descriptor=np.asarray([1.0, 0.0], dtype=np.float32),
            body_descriptor=np.asarray([1.0, 0.0], dtype=np.float32),
            quality=0.9,
            source_path=Path("Friends_S08E05_00000042.jpg"),
        )
        self.assertEqual(item.source_path.name, "Friends_S08E05_00000042.jpg")

    def test_co_visible_faces_never_merge(self) -> None:
        items = [
            observation(10, [1.0, 0.0], [1.0, 0.0], 0.9),
            observation(10, [0.99, 0.01], [1.0, 0.0], 0.8),
            observation(20, [1.0, 0.0], [1.0, 0.0], 0.7),
        ]
        clusters = cluster_face_observations(
            items, assign_threshold=0.7, merge_threshold=0.7
        )
        self.assertEqual(len(clusters), 2)
        self.assertEqual(sorted(len(cluster.members) for cluster in clusters), [1, 2])

    def test_representatives_prefer_dominant_outfit(self) -> None:
        cluster = CharacterCluster(
            [
                observation(1, [1.0, 0.0], [1.0, 0.0], 0.80),
                observation(10, [1.0, 0.0], [1.0, 0.0], 0.78),
                observation(20, [1.0, 0.0], [1.0, 0.0], 0.76),
                observation(30, [1.0, 0.0], [0.0, 1.0], 0.99),
            ]
        )
        selected = select_representatives(cluster, count=3, duration_sec=40)
        self.assertEqual(len(selected), 3)
        self.assertNotIn(30, [item.frame_index for item in selected])


if __name__ == "__main__":
    unittest.main()
