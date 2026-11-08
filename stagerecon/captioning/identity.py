"""Build Reference bindings from SAM3 tracks and a character gallery.

The default histogram backend is dependency-light and deliberately conservative.
For production identity decisions use ``insightface`` or manual bindings; low
confidence tracks always fall back to an anonymous Reference with a masked RGB
crop instead of forcing a cast identity.
"""

from __future__ import annotations

import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .io import read_json, relative_path, write_json


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class Embedder(Protocol):
    name: str

    def embed(self, path: Path) -> np.ndarray | None: ...


class HistogramEmbedder:
    """A portable proposal backend; not a substitute for face recognition."""

    name = "opencv_hs_histogram_v1"

    def embed(self, path: Path) -> np.ndarray | None:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        mask = None
        if image.ndim == 3 and image.shape[2] == 4:
            mask = image[:, :, 3]
            image = image[:, :, :3]
        if image.size == 0:
            return None
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], mask, [32, 32], [0, 180, 0, 256])
        vector = histogram.reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else None


class InsightFaceEmbedder:
    """Optional ArcFace-style backend loaded only when requested."""

    def __init__(
        self,
        *,
        model_name: str = "buffalo_l",
        model_root: Path | None = None,
        device: str = "cpu",
    ) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as error:
            raise RuntimeError(
                "The insightface backend requires insightface and onnxruntime. "
                "Install the optional identity dependencies or use --identity-backend histogram."
            ) from error
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        kwargs: dict[str, Any] = {"name": model_name, "providers": providers}
        if model_root:
            kwargs["root"] = str(model_root)
        self.app = FaceAnalysis(**kwargs)
        self.app.prepare(ctx_id=0 if device.startswith("cuda") else -1, det_size=(640, 640))
        self.name = f"insightface_{model_name}"

    def embed(self, path: Path) -> np.ndarray | None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            return None
        faces = self.app.get(image)
        if not faces:
            return None
        face = max(
            faces,
            key=lambda item: float(
                max(0.0, item.bbox[2] - item.bbox[0])
                * max(0.0, item.bbox[3] - item.bbox[1])
            ),
        )
        vector = np.asarray(face.normed_embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else None


def make_embedder(
    backend: str, *, model_name: str = "buffalo_l", model_root: Path | None = None,
    device: str = "cpu",
) -> Embedder:
    if backend == "histogram":
        return HistogramEmbedder()
    if backend == "insightface":
        return InsightFaceEmbedder(
            model_name=model_name, model_root=model_root, device=device
        )
    raise ValueError(f"Unknown identity backend: {backend}")


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.dot(first, second))


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def _person_ref_id(canonical_identity: str | None, fallback_index: int) -> str:
    if canonical_identity:
        name = canonical_identity.split(":")[-1]
        token = re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")
        if token:
            if token.startswith("PERSON_"):
                return token
            return f"PERSON_{token}"
    return f"PERSON_{fallback_index}"


def build_registry(
    *, gallery_root: Path, template_path: Path, output: Path
) -> dict[str, Any]:
    gallery_root = gallery_root.resolve()
    template = read_json(template_path, "character registry template")
    characters = []
    for character in template.get("characters", []):
        canonical = str(character["canonical_identity"])
        directory_name = canonical.split(":")[-1]
        directory = gallery_root / directory_name
        gallery = []
        if directory.is_dir():
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    gallery.append(
                        {
                            "path": relative_path(path, output.parent),
                            "season": None,
                            "view": None,
                            "quality": None,
                        }
                    )
        characters.append(
            {
                "canonical_identity": canonical,
                "display_name": str(character["display_name"]),
                "gallery": gallery,
            }
        )
    result = {
        "schema_version": "character_registry_v1",
        "registry_version": str(template.get("registry_version", "friends_v1")),
        "characters": characters,
    }
    write_json(output, result)
    return result


def build_episode_registry(
    *,
    episode_dir: Path,
    output: Path,
    reference_name: str = "reference_bank.png",
) -> dict[str, Any]:
    """Build a registry whose stable IDs mirror per-episode ``person_*`` folders.

    A directory such as ``S08E02/person_009/reference_bank.png`` becomes the
    canonical identity ``friends:S08E02:person_009`` and therefore the prompt
    reference ID ``PERSON_009``.  The IDs are intentionally episode-local.
    """
    episode_dir = episode_dir.resolve()
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"Episode gallery directory does not exist: {episode_dir}")
    output = output.resolve()
    episode_id = episode_dir.name
    manifest_path = episode_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path, "episode person gallery manifest")
        episode_id = str(manifest.get("episode_id") or episode_id)

    characters = []
    for directory in sorted(episode_dir.glob("person_*")):
        if not directory.is_dir():
            continue
        reference = directory / reference_name
        if not reference.is_file():
            continue
        person_id = directory.name.lower()
        characters.append(
            {
                "canonical_identity": f"friends:{episode_id}:{person_id}",
                "display_name": person_id,
                "gallery": [
                    {
                        "path": relative_path(reference, output.parent),
                        "season": episode_id,
                        "view": "generated_reference_bank",
                        "quality": None,
                    }
                ],
            }
        )
    if not characters:
        raise ValueError(
            f"No {reference_name} files found under episode gallery: {episode_dir}"
        )
    result = {
        "schema_version": "character_registry_v1",
        "registry_version": f"friends_{episode_id.lower()}_person_ids_v1",
        "episode_id": episode_id,
        "characters": characters,
    }
    write_json(output, result)
    return result


def bind_registry_references(
    *, registry_path: Path, output_dir: Path, caption_root: Path
) -> dict[str, Any]:
    """Expose a complete episode registry to Qwen without SAM3 matching."""
    registry = read_json(registry_path, "character registry")
    reference_dir = output_dir / "references"
    reference_dir.mkdir(parents=True, exist_ok=True)
    bindings = []
    for index, character in enumerate(registry.get("characters", []), start=1):
        canonical = str(character["canonical_identity"])
        gallery_path = None
        for gallery_item in character.get("gallery", []):
            candidate = Path(str(gallery_item["path"]))
            if not candidate.is_absolute():
                candidate = registry_path.parent / candidate
            candidate = candidate.resolve()
            if candidate.is_file():
                gallery_path = candidate
                break
        if gallery_path is None:
            continue
        ref_id = _person_ref_id(canonical, index)
        destination = reference_dir / (
            _safe_slug(canonical or ref_id) + gallery_path.suffix.lower()
        )
        if not destination.is_file():
            shutil.copy2(gallery_path, destination)
        bindings.append(
            {
                "ref_id": ref_id,
                "canonical_identity": canonical,
                "identity_status": "matched",
                "source_track": None,
                "appearance_image": relative_path(destination, caption_root),
                "anchor_origin": "character_gallery",
                "anchor_frame": 0,
                "anchor_timestamp_sec": 0.0,
                "representative_crops": [],
                "match": {
                    "method": "qwen_omni_direct_reference_candidate_v1",
                    "score": 0.0,
                    "second_best_score": None,
                    "margin": None,
                    "observation_count": 1,
                    "model_version": "pending_qwen_caption",
                },
                "candidate_matches": [
                    {"canonical_identity": canonical, "score": 0.0}
                ],
            }
        )
    if not bindings:
        raise ValueError(f"Registry contains no usable gallery images: {registry_path}")
    payload = {
        "registry_version": registry.get("registry_version"),
        "backend": "qwen_omni_direct_reference_bank_v1",
        "bindings": bindings,
    }
    write_json(output_dir / "bindings.json", payload)
    return payload


def keep_qwen_selected_references(
    *,
    identity: dict[str, Any],
    proposal: dict[str, Any],
    output_dir: Path,
    model_name: str | None,
) -> dict[str, Any]:
    """Keep only candidates that Qwen uses in shots or as speakers."""
    proposal_by_ref = {
        str(item["ref_id"]): item for item in proposal.get("references", [])
    }
    selected_ids = {
        str(ref_id)
        for shot in proposal.get("shots", [])
        for ref_id in shot.get("references_in_shot", [])
    }
    selected_ids.update(
        str(event["speaker"])
        for event in proposal.get("events", [])
        if event.get("speaker") is not None
    )
    selected = []
    for binding in identity.get("bindings", []):
        ref_id = str(binding["ref_id"])
        reference = proposal_by_ref.get(ref_id)
        if reference is None or ref_id not in selected_ids:
            continue
        confidence = float(reference.get("confidence", 0.0))
        binding["match"] = {
            "method": "qwen_omni_direct_reference_bank_v1",
            "score": confidence,
            "second_best_score": None,
            "margin": None,
            "observation_count": 1,
            "model_version": model_name or "qwen_omni",
        }
        binding["candidate_matches"] = [
            {
                "canonical_identity": str(binding["canonical_identity"]),
                "score": confidence,
            }
        ]
        selected.append(binding)
    if not selected:
        raise ValueError("Qwen caption selected no person from the supplied reference bank")
    identity["bindings"] = selected
    write_json(output_dir / "bindings.json", identity)
    return identity


def _resolve_mask(item: dict[str, Any], clip_run: Path, track_id: int) -> Path:
    supplied = Path(str(item.get("mask", "")))
    if supplied.is_file():
        return supplied.resolve()
    frame_stem = Path(str(item["frame_name"])).stem
    fallback = clip_run / "masks/instances" / f"{track_id:04d}" / f"{frame_stem}.png"
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(f"Mask does not exist for track {track_id}: {fallback}")


def _candidate_quality(
    image: np.ndarray, mask: np.ndarray, bbox: list[int], item: dict[str, Any]
) -> float:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width - 1, x2), min(height - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return -1.0
    crop = image[y1 : y2 + 1, x1 : x2 + 1]
    crop_mask = mask[y1 : y2 + 1, x1 : x2 + 1]
    visible_area = float(np.count_nonzero(crop_mask))
    area_ratio = visible_area / max(1.0, width * height)
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = min(1.0, math.log1p(cv2.Laplacian(grey, cv2.CV_64F).var()) / 7.0)
    segmentation_score = float(item.get("score", 0.5))
    touches_border = x1 == 0 or y1 == 0 or x2 >= width - 1 or y2 >= height - 1
    border_penalty = 0.25 if touches_border else 0.0
    return 0.45 * segmentation_score + 0.30 * min(1.0, math.sqrt(area_ratio * 10)) + 0.25 * sharpness - border_penalty


def extract_track_crops(
    *,
    clip_run: Path,
    person_tracks_path: Path,
    output_dir: Path,
    caption_root: Path,
    maximum: int = 5,
    min_frame_gap: int = 8,
) -> list[dict[str, Any]]:
    people = read_json(person_tracks_path, "person tracks")
    frames_dir = clip_run / "frames"
    fps = float(read_json(clip_run / "frames.json", "frames metadata")["fps"])
    results = []
    for track in sorted(people.get("tracks", []), key=lambda item: int(item["first_frame"])):
        track_id = int(track["track_id"])
        candidates = []
        for item in track.get("frames", []):
            frame_index = int(item["frame_index"])
            frame_path = frames_dir / str(item["frame_name"])
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            mask_path = _resolve_mask(item, clip_run, track_id)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape[:2] != image.shape[:2]:
                continue
            bbox = [int(round(value)) for value in item["bbox_xyxy"]]
            quality = _candidate_quality(image, mask, bbox, item)
            candidates.append((quality, frame_index, frame_path, mask_path, bbox))
        candidates.sort(key=lambda value: (-value[0], value[1]))
        selected = []
        for candidate in candidates:
            if all(abs(candidate[1] - existing[1]) >= min_frame_gap for existing in selected):
                selected.append(candidate)
            if len(selected) >= maximum:
                break
        if not selected and candidates:
            selected.append(candidates[0])

        track_dir = output_dir / f"track_{track_id:04d}"
        track_dir.mkdir(parents=True, exist_ok=True)
        crops = []
        for rank, (quality, frame_index, frame_path, mask_path, bbox) in enumerate(
            selected, start=1
        ):
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            height, width = image.shape[:2]
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            colour = image[y1 : y2 + 1, x1 : x2 + 1]
            alpha = mask[y1 : y2 + 1, x1 : x2 + 1]
            rgba = np.dstack((colour, alpha))
            crop_path = track_dir / f"crop_f{frame_index:06d}_q{rank}.png"
            if not cv2.imwrite(str(crop_path), rgba):
                raise RuntimeError(f"Could not write identity crop: {crop_path}")
            crops.append(
                {
                    "frame_index": frame_index,
                    "timestamp_sec": frame_index / fps,
                    "quality": quality,
                    "path": relative_path(crop_path, caption_root),
                    "absolute_path": str(crop_path.resolve()),
                }
            )
        results.append(
            {
                "track_id": track_id,
                "first_frame": int(track["first_frame"]),
                "last_frame": int(track["last_frame"]),
                "frame_count": int(track["frame_count"]),
                "crops": crops,
            }
        )
    write_json(output_dir / "track_observations.json", {"tracks": results})
    return results


@dataclass
class CharacterEmbeddings:
    canonical_identity: str
    display_name: str
    entries: list[tuple[Path, np.ndarray]]


def _load_gallery_embeddings(
    registry: dict[str, Any], registry_path: Path, embedder: Embedder
) -> list[CharacterEmbeddings]:
    characters = []
    for character in registry.get("characters", []):
        entries = []
        for gallery_item in character.get("gallery", []):
            path = Path(str(gallery_item["path"]))
            if not path.is_absolute():
                path = registry_path.parent / path
            path = path.resolve()
            if not path.is_file():
                continue
            embedding = embedder.embed(path)
            if embedding is not None:
                entries.append((path, embedding))
        characters.append(
            CharacterEmbeddings(
                canonical_identity=str(character["canonical_identity"]),
                display_name=str(character["display_name"]),
                entries=entries,
            )
        )
    return characters


def _track_character_score(
    crop_embeddings: list[np.ndarray], character: CharacterEmbeddings
) -> tuple[float, Path | None]:
    if not crop_embeddings or not character.entries:
        return -1.0, None
    crop_scores = []
    best_score = -1.0
    best_path = None
    for crop_embedding in crop_embeddings:
        scores = [
            (cosine_similarity(crop_embedding, gallery_embedding), gallery_path)
            for gallery_path, gallery_embedding in character.entries
        ]
        score, path = max(scores, key=lambda item: item[0])
        crop_scores.append(score)
        if score > best_score:
            best_score, best_path = score, path
    aggregate = float(np.median(np.asarray(crop_scores, dtype=np.float32)))
    return aggregate, best_path


def bind_identities(
    *,
    clip_run: Path,
    person_tracks_path: Path,
    registry_path: Path | None,
    output_dir: Path,
    caption_root: Path,
    backend: str = "histogram",
    accept_threshold: float = 0.90,
    margin_threshold: float = 0.08,
    representative_crops: int = 5,
    min_frame_gap: int = 8,
    model_name: str = "buffalo_l",
    model_root: Path | None = None,
    device: str = "cpu",
    manual_bindings: dict[int, str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks = extract_track_crops(
        clip_run=clip_run,
        person_tracks_path=person_tracks_path,
        output_dir=output_dir,
        caption_root=caption_root,
        maximum=representative_crops,
        min_frame_gap=min_frame_gap,
    )
    embedder = make_embedder(
        backend, model_name=model_name, model_root=model_root, device=device
    )
    registry = (
        read_json(registry_path, "character registry") if registry_path else {
            "registry_version": None,
            "characters": [],
        }
    )
    characters = (
        _load_gallery_embeddings(registry, registry_path, embedder)
        if registry_path
        else []
    )
    manual_bindings = manual_bindings or {}

    proposals: dict[int, dict[str, Any]] = {}
    for track in tracks:
        crop_embeddings = []
        for crop in track["crops"]:
            embedding = embedder.embed(Path(crop["absolute_path"]))
            if embedding is not None:
                crop_embeddings.append(embedding)
        candidates = []
        best_gallery_by_identity: dict[str, Path | None] = {}
        for character in characters:
            score, best_gallery = _track_character_score(crop_embeddings, character)
            if score >= -0.5:
                candidates.append(
                    {
                        "canonical_identity": character.canonical_identity,
                        "display_name": character.display_name,
                        "score": score,
                    }
                )
                best_gallery_by_identity[character.canonical_identity] = best_gallery
        candidates.sort(key=lambda item: (-float(item["score"]), item["canonical_identity"]))
        top = candidates[0] if candidates else None
        second_score = float(candidates[1]["score"]) if len(candidates) > 1 else None
        margin = float(top["score"]) - second_score if top and second_score is not None else None
        eligible = bool(
            top
            and float(top["score"]) >= accept_threshold
            and (margin is None or margin >= margin_threshold)
        )
        proposals[int(track["track_id"])] = {
            "top": top,
            "second_score": second_score,
            "margin": margin,
            "eligible": eligible,
            "candidates": candidates[:3],
            "best_gallery_by_identity": best_gallery_by_identity,
            "observation_count": len(crop_embeddings),
        }

    # Enforce one accepted automatic track per canonical identity. A duplicate
    # proposal becomes anonymous instead of being coerced to its second choice.
    automatic_winners: dict[str, int] = {}
    for track_id, proposal in proposals.items():
        top = proposal["top"]
        if not proposal["eligible"] or not top or track_id in manual_bindings:
            continue
        canonical = str(top["canonical_identity"])
        current = automatic_winners.get(canonical)
        if current is None or float(top["score"]) > float(proposals[current]["top"]["score"]):
            automatic_winners[canonical] = track_id

    reference_dir = output_dir / "references"
    reference_dir.mkdir(exist_ok=True)
    bindings = []
    used_manual: set[str] = set()
    for reference_index, track in enumerate(
        sorted(tracks, key=lambda item: (item["first_frame"], item["track_id"])), start=1
    ):
        track_id = int(track["track_id"])
        proposal = proposals[track_id]
        canonical: str | None = None
        status = "anonymous"
        match = None
        gallery_path = None
        if track_id in manual_bindings:
            canonical = str(manual_bindings[track_id])
            status = "manual"
            used_manual.add(canonical)
            gallery_path = proposal["best_gallery_by_identity"].get(canonical)
        elif proposal["top"]:
            proposed_canonical = str(proposal["top"]["canonical_identity"])
            if automatic_winners.get(proposed_canonical) == track_id:
                canonical = proposed_canonical
                status = "matched"
                gallery_path = proposal["best_gallery_by_identity"].get(canonical)
                match = {
                    "method": embedder.name,
                    "score": float(proposal["top"]["score"]),
                    "second_best_score": proposal["second_score"],
                    "margin": proposal["margin"],
                    "observation_count": max(1, int(proposal["observation_count"])),
                    "model_version": embedder.name,
                }
        ref_id = _person_ref_id(canonical, reference_index)

        if gallery_path and gallery_path.is_file():
            destination = reference_dir / (
                _safe_slug(canonical or ref_id) + gallery_path.suffix.lower()
            )
            if not destination.is_file():
                shutil.copy2(gallery_path, destination)
            appearance_image = relative_path(destination, caption_root)
            anchor_origin = "manual" if status == "manual" else "character_gallery"
        else:
            if not track["crops"]:
                raise RuntimeError(f"Track {track_id} has no usable appearance crop")
            appearance_image = str(track["crops"][0]["path"])
            anchor_origin = "manual" if status == "manual" else "current_target_shot"

        anchor = track["crops"][0]
        bindings.append(
            {
                "ref_id": ref_id,
                "canonical_identity": canonical,
                "identity_status": status,
                "source_track": {
                    "sam3_track_id": track_id,
                    "first_frame": int(track["first_frame"]),
                    "last_frame": int(track["last_frame"]),
                },
                "appearance_image": appearance_image,
                "anchor_origin": anchor_origin,
                "anchor_frame": int(anchor["frame_index"]),
                "anchor_timestamp_sec": float(anchor["timestamp_sec"]),
                "representative_crops": [crop["path"] for crop in track["crops"]],
                "match": match,
                "candidate_matches": [
                    {
                        "canonical_identity": str(candidate["canonical_identity"]),
                        "score": float(candidate["score"]),
                    }
                    for candidate in proposal["candidates"]
                ],
            }
        )
    payload = {
        "registry_version": registry.get("registry_version"),
        "backend": embedder.name,
        "accept_threshold": accept_threshold,
        "margin_threshold": margin_threshold,
        "bindings": bindings,
    }
    write_json(output_dir / "bindings.json", payload)
    return payload
