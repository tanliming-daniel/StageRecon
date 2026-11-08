#!/usr/bin/env python3
"""Export three representative RGB crops per character from reconstruction images.

Images are grouped by the ``SxxExx`` token in names such as
``Friends_S08E05_00012827.jpg``. Faces are clustered within each episode, then
SAM3 finds person bounding boxes for the final representatives. Exported files
keep original RGB pixels and do not contain masks or alpha channels.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stagerecon.character_gallery import (
    FaceObservation,
    body_colour_descriptor,
    cluster_face_observations,
    deduplicate_boxes,
    expanded_body_box,
    face_quality,
    handcrafted_face_descriptor,
    infer_episode_id,
    cosine_similarity,
    select_representatives,
    square_rgba,
)


DEFAULT_IMAGES = (
    REPO_ROOT
    / "data/sparse_reconstruction_and_image_data/Friends-monica_apartment/images"
)
DEFAULT_OUTPUT = REPO_ROOT / "data/raw/friends_gallery"
DEFAULT_SAM3_ROOT = Path("/root/userdata/300workspace/sam3")
DEFAULT_SAM3_CHECKPOINT = Path("/root/userdata/300workspace/ckpt/sam3/sam3.pt")
DEFAULT_SAM3_PYTHON = Path("/root/userdata/300workspace/sm3r/bin/python")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SOURCE_FRAME_PATTERN = re.compile(r"_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--episode", action="append", default=[],
        help="Only process this episode ID, e.g. S08E05. Repeat as needed.",
    )
    parser.add_argument("--images-per-character", type=int, default=3)
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--max-characters", type=int, default=16)
    parser.add_argument(
        "--max-source-images", type=int, default=160,
        help="Uniformly sample at most this many reconstruction images per episode; 0 keeps all.",
    )
    parser.add_argument("--min-face-pixels", type=int, default=36)
    parser.add_argument(
        "--candidate-images-per-person", type=int, default=6,
        help="Maximum candidates sent to SAM3 before keeping the best three.",
    )
    parser.add_argument("--assign-threshold", type=float, default=0.84)
    parser.add_argument("--merge-threshold", type=float, default=0.90)
    parser.add_argument("--source-fps", type=float, default=24.0)
    parser.add_argument("--sam3-root", type=Path, default=DEFAULT_SAM3_ROOT)
    parser.add_argument("--sam3-checkpoint", type=Path, default=DEFAULT_SAM3_CHECKPOINT)
    parser.add_argument("--sam3-python", type=Path, default=DEFAULT_SAM3_PYTHON)
    parser.add_argument("--sam3-device", default="cuda")
    parser.add_argument("--sam3-threshold", type=float, default=0.45)
    parser.add_argument("--prompt", default="person")
    parser.add_argument(
        "--no-sam3", action="store_true",
        help="Use face-guided RGB crops instead of SAM3 person-bbox RGB crops.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class FaceDetector:
    def __init__(self, minimum_pixels: int) -> None:
        root = Path(cv2.data.haarcascades)
        self.frontal = cv2.CascadeClassifier(str(root / "haarcascade_frontalface_alt2.xml"))
        self.profile = cv2.CascadeClassifier(str(root / "haarcascade_profileface.xml"))
        self.eyes = cv2.CascadeClassifier(str(root / "haarcascade_eye_tree_eyeglasses.xml"))
        if self.frontal.empty() or self.profile.empty() or self.eyes.empty():
            raise RuntimeError(f"Could not load OpenCV face cascades from {root}")
        self.minimum_pixels = minimum_pixels

    def has_eye_pair(
        self, grey: np.ndarray, box: tuple[int, int, int, int]
    ) -> bool:
        x, y, width, height = box
        upper = grey[y : y + max(1, int(round(height * 0.70))), x : x + width]
        minimum = max(7, int(round(width * 0.09)))
        eyes = self.eyes.detectMultiScale(
            upper, scaleFactor=1.05, minNeighbors=3,
            minSize=(minimum, minimum),
        )
        centres = [
            (float(ex + ew * 0.5), float(ey + eh * 0.5))
            for ex, ey, ew, eh in eyes
            if 0.08 * width <= ew <= 0.42 * width
        ]
        for first_index, first in enumerate(centres):
            for second in centres[first_index + 1 :]:
                horizontal = abs(first[0] - second[0])
                vertical = abs(first[1] - second[1])
                if 0.18 * width <= horizontal <= 0.78 * width and vertical <= 0.22 * height:
                    return True
        return False

    def detect(self, image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        grey = cv2.equalizeHist(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY))
        kwargs = {
            "scaleFactor": 1.08, "minNeighbors": 5,
            "minSize": (self.minimum_pixels, self.minimum_pixels),
        }
        frontal_boxes = [
            tuple(map(int, item))
            for item in self.frontal.detectMultiScale(grey, **kwargs)
        ]
        frontal_boxes = [
            box
            for box in deduplicate_boxes(frontal_boxes)
            if self.has_eye_pair(grey, box)
        ]
        profile_boxes = [
            tuple(map(int, item))
            for item in self.profile.detectMultiScale(grey, **kwargs)
        ]
        flipped = cv2.flip(grey, 1)
        image_width = grey.shape[1]
        for fx, fy, fw, fh in self.profile.detectMultiScale(flipped, **kwargs):
            profile_boxes.append((image_width - int(fx) - int(fw), int(fy), int(fw), int(fh)))
        return deduplicate_boxes(frontal_boxes + profile_boxes)


class Sam3Segmenter:
    def __init__(
        self, *, python: Path, root: Path, checkpoint: Path,
        device: str, threshold: float, prompt: str
    ) -> None:
        python, root, checkpoint = python.resolve(), root.resolve(), checkpoint.resolve()
        if not python.is_file():
            raise FileNotFoundError(f"SAM3 Python not found: {python}")
        if not (root / "sam3").is_dir():
            raise FileNotFoundError(f"SAM3 package not found: {root}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")
        worker = Path(__file__).resolve().with_name("sam3_person_worker.py")
        self.process = subprocess.Popen(
            [
                str(python), str(worker), "--sam3-root", str(root),
                "--checkpoint", str(checkpoint), "--device", device,
                "--threshold", str(threshold), "--prompt", prompt,
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        ready = self._read_message()
        if ready.get("status") != "ready":
            raise RuntimeError(f"SAM3 worker did not become ready: {ready}")

    def _read_message(self) -> dict[str, Any]:
        assert self.process.stdout is not None
        prefix = "__STAGERECON_SAM3__"
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"SAM3 worker exited with {self.process.poll()}")
            if line.startswith(prefix):
                return json.loads(line[len(prefix) :])
            print(f"[sam3] {line.rstrip()}", flush=True)

    def cutout(
        self, image_path: Path, face_bbox: tuple[int, int, int, int], output_path: Path
    ) -> dict[str, Any]:
        assert self.process.stdin is not None
        request = {
            "image_path": str(image_path), "output_path": str(output_path),
            "face_bbox_xywh": list(face_bbox),
        }
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        result = self._read_message()
        return result

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps({"command": "close"}) + "\n")
        self.process.stdin.flush()
        try:
            self._read_message()
        finally:
            self.process.wait(timeout=30)


def source_frame(path: Path, fallback: int) -> int:
    match = SOURCE_FRAME_PATTERN.search(path.stem)
    return int(match.group(1)) if match else fallback


def group_episode_images(images_dir: Path, selected_ids: set[str]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in sorted(images_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        episode_id = infer_episode_id(path)
        if not episode_id.upper().startswith("S"):
            continue
        if selected_ids and episode_id.upper() not in selected_ids:
            continue
        grouped.setdefault(episode_id, []).append(path.resolve())
    for paths in grouped.values():
        paths.sort(key=lambda path: (source_frame(path, 0), path.name))
    return dict(sorted(grouped.items()))


def uniformly_sample_paths(paths: list[Path], maximum: int) -> list[Path]:
    if maximum <= 0 or len(paths) <= maximum:
        return paths
    indices = np.linspace(0, len(paths) - 1, num=maximum, dtype=np.int64)
    return [paths[int(index)] for index in sorted(set(indices.tolist()))]


def observation_from_box(
    image: np.ndarray, *, image_path: Path, image_index: int, source_fps: float,
    box: tuple[int, int, int, int],
) -> FaceObservation | None:
    x, y, width, height = box
    face = image[y : y + height, x : x + width]
    quality = face_quality(image, box)
    if face.size == 0 or quality < 0.05:
        return None
    frame_index = source_frame(image_path, image_index)
    return FaceObservation(
        frame_index=frame_index, timestamp_sec=frame_index / source_fps,
        bbox_xywh=box, descriptor=handcrafted_face_descriptor(face),
        body_descriptor=body_colour_descriptor(image, box), quality=quality,
        source_path=image_path, image_shape=image.shape[:2],
    )


def scan_episode_images(
    paths: list[Path], detector: FaceDetector, *, source_fps: float
) -> list[FaceObservation]:
    observations: list[FaceObservation] = []
    for image_index, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        scale = min(1.0, 1280.0 / max(image.shape[:2]))
        detection_image = (
            cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0 else image
        )
        boxes = detector.detect(detection_image)
        if scale < 1.0:
            boxes = [
                tuple(int(round(value / scale)) for value in box)
                for box in boxes
            ]
        for box in boxes:
            item = observation_from_box(
                image, image_path=path, image_index=image_index,
                source_fps=source_fps, box=box,
            )
            if item is not None:
                observations.append(item)
    return observations


def full_body_candidates(
    cluster: Any, *, maximum: int
) -> list[FaceObservation]:
    """Prefer frames with the most space available for a full body."""

    ranked = []
    for item in cluster.members:
        _, face_y, _, face_height = item.bbox_xywh
        face_height = max(1, face_height)
        distance_score = 1.0 - min(1.0, face_height / 260.0)
        image_height = item.image_shape[0] if item.image_shape else 1080
        below_face_ratio = (
            image_height - (face_y + face_height)
        ) / max(1.0, face_height)
        body_room_score = min(1.0, max(0.0, below_face_ratio / 7.5))
        score = (
            0.62 * body_room_score
            + 0.30 * distance_score
            + 0.08 * item.quality
        )
        ranked.append((score, -item.frame_index, item))
    ranked.sort(key=lambda value: (value[0], value[1]), reverse=True)
    return [item for _, _, item in ranked[:maximum]]


def dominant_outfit_candidates(
    candidates: list[dict[str, Any]], *, minimum: int
) -> list[dict[str, Any]]:
    """Keep the largest SAM3-foreground torso-colour group within one face cluster."""

    usable = [
        item for item in candidates
        if item["result"].get("torso_colour_histogram") is not None
    ]
    if not usable:
        return candidates
    groups: list[list[dict[str, Any]]] = []
    prototypes: list[np.ndarray] = []
    for item in sorted(usable, key=lambda value: value["observation"].quality, reverse=True):
        descriptor = np.asarray(
            item["result"]["torso_colour_histogram"], dtype=np.float32
        )
        scores = [cosine_similarity(descriptor, prototype) for prototype in prototypes]
        best_index = int(np.argmax(scores)) if scores else -1
        if best_index >= 0 and scores[best_index] >= 0.72:
            groups[best_index].append(item)
            prototypes[best_index] = np.mean(
                [
                    np.asarray(member["result"]["torso_colour_histogram"], dtype=np.float32)
                    for member in groups[best_index]
                ],
                axis=0,
            )
        else:
            groups.append([item])
            prototypes.append(descriptor)
    eligible = [group for group in groups if len(group) >= minimum]
    if not eligible:
        return []
    return max(
        eligible,
        key=lambda group: (
            len(group),
            float(np.mean([member["observation"].quality for member in group])),
        ),
    )


def choose_person_mask(
    predictions: list[dict[str, Any]], face_bbox: tuple[int, int, int, int]
) -> tuple[np.ndarray | None, float | None]:
    x, y, width, height = face_bbox
    centre_x, centre_y = x + 0.5 * width, y + 0.55 * height
    ranked = []
    for prediction in predictions:
        x1, y1, x2, y2 = prediction["bbox_xyxy"]
        mask = prediction["mask"]
        px = min(mask.shape[1] - 1, max(0, int(round(centre_x))))
        py = min(mask.shape[0] - 1, max(0, int(round(centre_y))))
        contains = x1 <= centre_x <= x2 and y1 <= centre_y <= y2
        hits_face = bool(mask[py, px])
        box_area = max(1.0, (x2 - x1) * (y2 - y1))
        specificity = min(1.0, (width * height * 20.0) / box_area)
        rank_score = (
            float(prediction["score"]) + 0.22 * specificity
            + 0.35 * hits_face + 0.15 * contains
        )
        if hits_face or contains:
            ranked.append((rank_score, prediction))
    if not ranked:
        return None, None
    prediction = max(ranked, key=lambda item: item[0])[1]
    return prediction["mask"].astype(np.uint8) * 255, float(prediction["score"])


def fallback_body_mask(image: np.ndarray, face_bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = expanded_body_box(face_bbox, image.shape)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y : y + height, x : x + width] = 255
    return mask


def rough_full_body_crop(
    image: np.ndarray, face_bbox: tuple[int, int, int, int]
) -> tuple[np.ndarray, tuple[int, int, int, int], dict[str, Any]]:
    """Return an ordinary RGB crop that conservatively keeps the full body.

    The face anchors the horizontal centre.  The crop extends from slightly
    above the head to the bottom of the source frame, because distant sitcom
    shots commonly contain the person's legs even when a face-based 6--7 head
    estimate would stop at the waist.  No mask or alpha channel is created.
    """

    x, y, width, height = face_bbox
    image_height, image_width = image.shape[:2]
    y1 = max(0, int(round(y - 0.55 * height)))
    y2 = image_height
    crop_height = max(1, y2 - y1)

    centre_x = x + 0.5 * width
    target_width = int(round(max(5.0 * width, 0.46 * crop_height)))
    target_width = min(image_width, max(1, target_width))
    x1 = int(round(centre_x - 0.5 * target_width))
    x1 = min(max(0, x1), image_width - target_width)
    x2 = x1 + target_width

    crop = image[y1:y2, x1:x2].copy()
    below_face_ratio = (image_height - (y + height)) / max(1.0, height)
    distance_score = 1.0 - min(1.0, height / 260.0)
    completeness = min(1.0, max(0.0, below_face_ratio / 8.0))
    result = {
        "ok": True,
        "score": None,
        "person_bbox_xyxy": [x1, y1, x2, y2],
        "person_to_face_height_ratio": crop_height / max(1.0, height),
        "below_face_height_ratio": below_face_ratio,
        "touches_top": y1 == 0,
        "touches_bottom": True,
        "touches_side": x1 == 0 or x2 == image_width,
        "full_body_likely": below_face_ratio >= 6.5 and distance_score >= 0.45,
        "completeness_score": 0.72 * completeness + 0.28 * distance_score,
    }
    return crop, (x1, y1, x2, y2), result


def composite_preview(image: np.ndarray, size: int = 320) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        colour = image[:, :, :3].astype(np.float32)
        image = (colour * alpha + 245.0 * (1.0 - alpha)).astype(np.uint8)
    height, width = image.shape[:2]
    scale = min(size / max(1, width), size / max(1, height))
    resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))
    canvas = np.full((size, size, 3), 245, dtype=np.uint8)
    y, x = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def write_contact_sheet(episode_dir: Path, characters: list[dict[str, Any]]) -> None:
    cell, label_height = 320, 42
    rows = max(1, len(characters))
    sheet = np.full((rows * (cell + label_height), 3 * cell, 3), 250, dtype=np.uint8)
    for row, character in enumerate(characters):
        cv2.putText(
            sheet, str(character.get("person_id", character.get("character_id"))),
            (10, row * (cell + label_height) + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (25, 25, 25), 2, cv2.LINE_AA,
        )
        for column, image_path in enumerate(character["images"][:3]):
            image = cv2.imread(str(episode_dir / image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            top = row * (cell + label_height) + label_height
            sheet[top : top + cell, column * cell : (column + 1) * cell] = (
                composite_preview(image, cell)
            )
    cv2.imwrite(str(episode_dir / "contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def episode_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "images_per_character": args.images_per_character,
        "min_observations": args.min_observations,
        "max_characters": args.max_characters,
        "max_source_images": args.max_source_images,
        "min_face_pixels": args.min_face_pixels,
        "candidate_images_per_person": args.candidate_images_per_person,
        "face_detector": "opencv_haar_frontal_eye_pair_plus_profile_v3",
        "assign_threshold": args.assign_threshold,
        "merge_threshold": args.merge_threshold,
        "source_fps": args.source_fps,
        "sam3_enabled": not args.no_sam3,
        "sam3_python": str(args.sam3_python.resolve()) if not args.no_sam3 else None,
        "sam3_threshold": args.sam3_threshold,
        "prompt": args.prompt,
        "selection_strategy": (
            "sam3_full_body_first_mask_width_rgb_v2" if not args.no_sam3
            else "face_guided_rgb_full_height_crop_v2"
        ),
    }


def source_signature(paths: list[Path]) -> dict[str, Any]:
    stats = [path.stat() for path in paths]
    return {
        "directory": str(paths[0].parent.resolve()), "count": len(paths),
        "total_size": sum(item.st_size for item in stats),
        "latest_mtime_ns": max(item.st_mtime_ns for item in stats),
        "first": paths[0].name, "last": paths[-1].name,
    }


def is_complete(episode_dir: Path, paths: list[Path], options: dict[str, Any]) -> bool:
    manifest_path = episode_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("source_images") != source_signature(paths) or manifest.get("options") != options:
        return False
    return all(
        (episode_dir / image).is_file()
        for character in manifest.get("characters", [])
        for image in character.get("images", [])
    )


def process_episode(
    episode_id: str, paths: list[Path], *, args: argparse.Namespace,
    detector: FaceDetector, segmenter: Sam3Segmenter | None,
) -> dict[str, Any]:
    episode_dir = args.output_root.resolve() / episode_id
    options = episode_options(args)
    if args.resume and not args.overwrite and is_complete(episode_dir, paths, options):
        print(f"[{episode_id}] complete; skipped", flush=True)
        return json.loads((episode_dir / "manifest.json").read_text())
    if episode_dir.exists() and any(episode_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Episode output is incomplete or uses different options: {episode_dir}. "
            "Use --overwrite to rebuild it."
        )

    started = time.perf_counter()
    scan_paths = uniformly_sample_paths(paths, args.max_source_images)
    observations = scan_episode_images(scan_paths, detector, source_fps=args.source_fps)
    clusters = cluster_face_observations(
        observations, assign_threshold=args.assign_threshold, merge_threshold=args.merge_threshold,
    )
    clusters = [cluster for cluster in clusters if len(cluster.members) >= args.min_observations]
    clusters = clusters[: args.max_characters]
    print(
        f"[{episode_id}] {len(scan_paths)}/{len(paths)} images, {len(observations)} faces -> "
        f"{len(clusters)} characters", flush=True,
    )

    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{episode_id}.", dir=args.output_root.resolve()))
    person_entries = []
    try:
        for cluster in clusters:
            representatives = full_body_candidates(
                cluster,
                maximum=min(len(cluster.members), args.candidate_images_per_person),
            )
            person_id = f"person_{len(person_entries) + 1:03d}"
            person_dir = staging / person_id
            person_dir.mkdir(parents=True)
            candidate_dir = person_dir / ".candidates"
            candidate_dir.mkdir()
            candidates = []
            for candidate_index, observation in enumerate(representatives, start=1):
                if observation.source_path is None:
                    continue
                backend = "body_bbox_fallback"
                candidate_path = candidate_dir / f"{candidate_index:03d}.png"
                result: dict[str, Any]
                if segmenter is not None:
                    result = segmenter.cutout(
                        observation.source_path, observation.bbox_xywh, candidate_path
                    )
                    if not result.get("ok"):
                        continue
                    backend = "sam3_person_bbox_rgb_crop"
                else:
                    image = cv2.imread(str(observation.source_path), cv2.IMREAD_COLOR)
                    if image is None:
                        continue
                    crop, _, result = rough_full_body_crop(image, observation.bbox_xywh)
                    if not cv2.imwrite(str(candidate_path), crop):
                        raise RuntimeError(f"Could not write {candidate_path}")
                    backend = "rgb_face_guided_full_height_crop"
                candidates.append(
                    {
                        "candidate_path": candidate_path,
                        "observation": observation,
                        "backend": backend,
                        "result": result,
                    }
                )
            candidates.sort(
                key=lambda item: (
                    bool(item["result"].get("full_body_likely", False)),
                    float(item["result"].get("person_to_face_height_ratio") or 0.0),
                    float(item["result"].get("below_face_height_ratio") or 0.0),
                    float(item["result"].get("completeness_score", 0.0)),
                    float(item["result"].get("score") or 0.0),
                    float(item["observation"].quality),
                ),
                reverse=True,
            )
            selected = candidates[: args.images_per_character]
            images, sources = [], []
            for rank, item in enumerate(selected, start=1):
                observation = item["observation"]
                result = item["result"]
                image_path = person_dir / f"{rank:02d}.png"
                item["candidate_path"].replace(image_path)
                relative = str(image_path.relative_to(staging))
                images.append(relative)
                sources.append(
                    {
                        "image": relative,
                        "source_image": str(observation.source_path),
                        "source_frame": observation.frame_index,
                        "timestamp_sec": round(observation.timestamp_sec, 3),
                        "face_bbox_xywh": list(observation.bbox_xywh),
                        "face_quality": round(float(observation.quality), 6),
                        "segmentation_backend": item["backend"],
                        "crop_policy": "full_source_height_sam_mask_width_rgb",
                        "segmentation_score": result.get("score"),
                        "person_bbox_xyxy": result.get("person_bbox_xyxy"),
                        "person_to_face_height_ratio": result.get("person_to_face_height_ratio"),
                        "below_face_height_ratio": result.get("below_face_height_ratio"),
                        "touches_top": result.get("touches_top"),
                        "touches_bottom": result.get("touches_bottom"),
                        "touches_side": result.get("touches_side"),
                        "full_body_likely": result.get("full_body_likely", False),
                        "completeness_score": result.get("completeness_score"),
                    }
                )
            shutil.rmtree(candidate_dir, ignore_errors=True)
            if len(images) == args.images_per_character:
                person_entries.append(
                    {
                        "person_id": person_id,
                        "observation_count": len(cluster.members),
                        "images": images, "sources": sources,
                    }
                )
            else:
                shutil.rmtree(person_dir, ignore_errors=True)
        manifest = {
            "schema_version": "episode_person_gallery_v3",
            "episode_id": episode_id, "source_images": source_signature(paths),
            "options": options, "face_observation_count": len(observations),
            "person_count": len(person_entries), "persons": person_entries,
            "character_count": len(person_entries), "characters": person_entries,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        write_contact_sheet(staging, person_entries)
        if episode_dir.exists():
            shutil.rmtree(episode_dir)
        staging.replace(episode_dir)
        print(f"[{episode_id}] wrote {len(person_entries)} people to {episode_dir}", flush=True)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    args = parse_args()
    if args.images_per_character <= 0:
        raise ValueError("--images-per-character must be positive")
    if args.source_fps <= 0:
        raise ValueError("--source-fps must be positive")
    images_dir = args.images_dir.resolve()
    if not images_dir.is_dir():
        raise NotADirectoryError(images_dir)
    selected_ids = {item.upper() for item in args.episode}
    episodes = group_episode_images(images_dir, selected_ids)
    if not episodes:
        raise ValueError(f"No matching episode images found in {images_dir}")

    detector = FaceDetector(args.min_face_pixels)
    segmenter = None
    if not args.no_sam3:
        print("Loading SAM3 once for final representative images...", flush=True)
        segmenter = Sam3Segmenter(
            python=args.sam3_python, root=args.sam3_root,
            checkpoint=args.sam3_checkpoint,
            device=args.sam3_device, threshold=args.sam3_threshold, prompt=args.prompt,
        )
    try:
        results = [
            process_episode(episode_id, paths, args=args, detector=detector, segmenter=segmenter)
            for episode_id, paths in episodes.items()
        ]
    finally:
        if segmenter is not None:
            segmenter.close()
    print(
        json.dumps(
            {
                "episodes": len(results),
                "characters": sum(item["character_count"] for item in results),
                "output_root": str(args.output_root.resolve()),
            }, indent=2,
        )
    )


if __name__ == "__main__":
    main()
