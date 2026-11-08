"""Utilities for building per-episode character reference candidates.

The gallery workflow intentionally separates cheap episode-wide face discovery
from expensive person segmentation.  Face observations are clustered first;
only the final representative frames are passed to SAM3.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


EPISODE_PATTERN = re.compile(r"(?i)(?:^|[^a-z0-9])s(\d{1,2})e(\d{1,2})(?:[^a-z0-9]|$)")


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_") or "unknown"


def infer_episode_id(path: Path) -> str:
    """Return a stable ``SxxExx`` ID when present, else a safe stem."""

    match = EPISODE_PATTERN.search(path.stem)
    if match:
        return f"S{int(match.group(1)):02d}E{int(match.group(2)):02d}"
    return safe_slug(path.stem)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-8 else value


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.dot(l2_normalize(first), l2_normalize(second)))


def observation_similarity(
    observation: "FaceObservation", cluster: "CharacterCluster"
) -> float:
    """Episode-local identity score: face first, stable outfit second."""

    face_score = cosine_similarity(observation.descriptor, cluster.prototype)
    outfit_score = cosine_similarity(observation.body_descriptor, cluster.body_prototype)
    return 0.84 * face_score + 0.16 * outfit_score


def bbox_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def deduplicate_boxes(
    boxes: list[tuple[int, int, int, int]], *, iou_threshold: float = 0.45
) -> list[tuple[int, int, int, int]]:
    kept: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        if all(bbox_iou(box, existing) < iou_threshold for existing in kept):
            kept.append(box)
    return kept


def expanded_body_box(
    face_bbox: tuple[int, int, int, int], image_shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    x, y, width, height = face_bbox
    image_height, image_width = image_shape[:2]
    x1 = max(0, int(round(x - 1.15 * width)))
    x2 = min(image_width, int(round(x + 2.15 * width)))
    y1 = max(0, int(round(y - 0.35 * height)))
    y2 = min(image_height, int(round(y + 6.75 * height)))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def handcrafted_face_descriptor(face_bgr: np.ndarray) -> np.ndarray:
    """Portable face descriptor used when no learned embedding is supplied.

    It combines contrast-normalized HOG and low-frequency DCT coefficients.
    This is intentionally a fallback; InsightFace/SFace embeddings are preferred
    for large or diverse datasets.
    """

    if face_bgr.size == 0:
        return np.zeros(1763 + 255, dtype=np.float32)
    grey = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    grey = cv2.resize(grey, (64, 64), interpolation=cv2.INTER_AREA)
    grey = cv2.equalizeHist(grey)
    hog = cv2.HOGDescriptor(
        (64, 64), (16, 16), (8, 8), (8, 8), 9
    ).compute(grey).reshape(-1).astype(np.float32)
    dct = cv2.dct(grey.astype(np.float32) / 255.0)[:16, :16].reshape(-1)[1:]
    return l2_normalize(np.concatenate((0.82 * l2_normalize(hog), 0.18 * l2_normalize(dct))))


def body_colour_descriptor(image_bgr: np.ndarray, face_bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = expanded_body_box(face_bbox, image_bgr.shape)
    body = image_bgr[y : y + height, x : x + width]
    if body.size == 0:
        return np.zeros(128, dtype=np.float32)
    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return l2_normalize(histogram.reshape(-1).astype(np.float32))


def face_quality(
    image_bgr: np.ndarray, face_bbox: tuple[int, int, int, int]
) -> float:
    x, y, width, height = face_bbox
    image_height, image_width = image_bgr.shape[:2]
    face = image_bgr[y : y + height, x : x + width]
    if face.size == 0:
        return -1.0
    area_ratio = (width * height) / max(1.0, image_width * image_height)
    grey = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    sharpness = min(1.0, math.log1p(cv2.Laplacian(grey, cv2.CV_64F).var()) / 7.5)
    size_score = min(1.0, math.sqrt(area_ratio * 28.0))
    aspect_score = max(0.0, 1.0 - abs(width / max(1.0, height) - 0.82))
    border = x <= 1 or y <= 1 or x + width >= image_width - 1 or y + height >= image_height - 1
    return 0.48 * size_score + 0.37 * sharpness + 0.15 * aspect_score - (0.25 if border else 0.0)


@dataclass
class FaceObservation:
    frame_index: int
    timestamp_sec: float
    bbox_xywh: tuple[int, int, int, int]
    descriptor: np.ndarray
    body_descriptor: np.ndarray
    quality: float
    source_path: Path | None = None
    image_shape: tuple[int, int] | None = None


@dataclass
class CharacterCluster:
    members: list[FaceObservation] = field(default_factory=list)

    @property
    def prototype(self) -> np.ndarray:
        return l2_normalize(np.mean([item.descriptor for item in self.members], axis=0))

    @property
    def body_prototype(self) -> np.ndarray:
        return l2_normalize(np.median([item.body_descriptor for item in self.members], axis=0))

    @property
    def frame_indices(self) -> set[int]:
        return {item.frame_index for item in self.members}


def _clusters_can_merge(first: CharacterCluster, second: CharacterCluster) -> bool:
    return first.frame_indices.isdisjoint(second.frame_indices)


def cluster_face_observations(
    observations: list[FaceObservation],
    *,
    assign_threshold: float = 0.76,
    merge_threshold: float = 0.82,
    ambiguity_margin: float = 0.025,
) -> list[CharacterCluster]:
    """Conservatively cluster face observations without merging co-visible faces."""

    clusters: list[CharacterCluster] = []
    for observation in sorted(observations, key=lambda item: (-item.quality, item.frame_index)):
        candidates = []
        for index, cluster in enumerate(clusters):
            if observation.frame_index in cluster.frame_indices:
                continue
            candidates.append((observation_similarity(observation, cluster), index))
        candidates.sort(reverse=True)
        best = candidates[0] if candidates else None
        second_score = candidates[1][0] if len(candidates) > 1 else -1.0
        if (
            best
            and best[0] >= assign_threshold
            and (best[0] - second_score >= ambiguity_margin or second_score < assign_threshold)
        ):
            clusters[best[1]].members.append(observation)
        else:
            clusters.append(CharacterCluster([observation]))

    while True:
        best_pair: tuple[float, int, int] | None = None
        for first_index in range(len(clusters)):
            for second_index in range(first_index + 1, len(clusters)):
                first, second = clusters[first_index], clusters[second_index]
                if not _clusters_can_merge(first, second):
                    continue
                face_score = cosine_similarity(first.prototype, second.prototype)
                outfit_score = cosine_similarity(first.body_prototype, second.body_prototype)
                score = 0.84 * face_score + 0.16 * outfit_score
                if score >= merge_threshold and (best_pair is None or score > best_pair[0]):
                    best_pair = (score, first_index, second_index)
        if best_pair is None:
            break
        _, first_index, second_index = best_pair
        clusters[first_index].members.extend(clusters[second_index].members)
        del clusters[second_index]

    return sorted(
        clusters,
        key=lambda cluster: (
            -len(cluster.members),
            -float(np.mean([item.quality for item in cluster.members])),
            min(item.frame_index for item in cluster.members),
        ),
    )


def select_representatives(
    cluster: CharacterCluster,
    *,
    count: int = 3,
    duration_sec: float = 1.0,
    minimum_gap_sec: float = 4.0,
) -> list[FaceObservation]:
    """Select sharp, temporally diverse images with the dominant outfit."""

    if count <= 0:
        return []
    body_prototype = cluster.body_prototype
    selected: list[FaceObservation] = []
    candidates = list(cluster.members)
    while candidates and len(selected) < count:
        scored = []
        for item in candidates:
            clothing_score = max(0.0, cosine_similarity(item.body_descriptor, body_prototype))
            if selected:
                nearest_gap = min(abs(item.timestamp_sec - other.timestamp_sec) for other in selected)
                temporal_score = min(1.0, nearest_gap / max(minimum_gap_sec, 1e-6))
            else:
                temporal_score = min(1.0, item.timestamp_sec / max(duration_sec * 0.2, 1.0))
            score = 0.62 * item.quality + 0.28 * clothing_score + 0.10 * temporal_score
            scored.append((score, -item.frame_index, item))
        chosen = max(scored, key=lambda value: (value[0], value[1]))[2]
        selected.append(chosen)
        candidates = [item for item in candidates if item.frame_index != chosen.frame_index]
    return sorted(selected, key=lambda item: item.timestamp_sec)


def square_rgba(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    padding_ratio: float = 0.08,
) -> np.ndarray:
    """Crop a person mask to a transparent square PNG canvas."""

    binary = mask > 0
    if not np.any(binary):
        raise ValueError("Person mask is empty")
    ys, xs = np.where(binary)
    image_height, image_width = image_bgr.shape[:2]
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    padding = int(round(max(x2 - x1, y2 - y1) * padding_ratio))
    x1, x2 = max(0, x1 - padding), min(image_width, x2 + padding)
    y1, y2 = max(0, y1 - padding), min(image_height, y2 + padding)
    colour = image_bgr[y1:y2, x1:x2]
    alpha = (binary[y1:y2, x1:x2].astype(np.uint8) * 255)
    rgba = np.dstack((colour, alpha))
    height, width = rgba.shape[:2]
    side = max(height, width)
    canvas = np.zeros((side, side, 4), dtype=np.uint8)
    x_offset = (side - width) // 2
    y_offset = (side - height) // 2
    canvas[y_offset : y_offset + height, x_offset : x_offset + width] = rgba
    return canvas
