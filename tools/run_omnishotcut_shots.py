#!/usr/bin/env python3
"""Run OmniShotCut and retain shots supported by target-scene images.

OmniShotCut still scans the complete source video because it needs full temporal
context.  Only segments containing a numbered image from ``--image-dir`` are
written to ``target_scene_segments.csv`` for downstream clipping/reconstruction.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
from collections import Counter
from pathlib import Path

NUMBER_RE = re.compile(r"(\d+)$")
EPISODE_RE = re.compile(r"S(\d{1,2})[ ._-]*E(\d{1,2})", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--image-glob", default="*")
    parser.add_argument(
        "--episode",
        help=(
            "Episode key such as S08E09. Defaults to the key inferred from "
            "--video. This filter is mandatory when --image-dir contains "
            "reference images from multiple episodes."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument(
        "--allow-empty-targets",
        action="store_true",
        help=(
            "Run full-video shot detection even when the selected episode has "
            "no target-scene reference images. Target manifests are written "
            "with headers and zero rows; no clips should be cut from them."
        ),
    )
    return parser.parse_args()


def write_csv(
    path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None,
) -> None:
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer CSV columns for empty output: {path}")
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def episode_key(value: str) -> str | None:
    match = EPISODE_RE.search(value)
    if match is None:
        return None
    return f"S{int(match.group(1)):02d}E{int(match.group(2)):02d}"


def collect_target_images(
    image_dir: Path, image_glob: str, video: Path, explicit_episode: str | None,
    allow_empty: bool = False,
) -> tuple[str, list[tuple[int, Path]]]:
    candidates: list[tuple[str | None, int, Path]] = []
    for image_path in sorted(image_dir.glob(image_glob)):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        number_match = NUMBER_RE.search(image_path.stem)
        if number_match is None:
            continue
        candidates.append(
            (episode_key(image_path.stem), int(number_match.group(1)), image_path)
        )

    requested_episode = episode_key(explicit_episode or "")
    if explicit_episode is not None and requested_episode is None:
        raise ValueError(f"Invalid --episode value: {explicit_episode!r}")
    inferred_episode = episode_key(video.name) or episode_key(str(video.parent))
    selected_episode = requested_episode or inferred_episode

    available_episodes = sorted({key for key, _, _ in candidates if key is not None})
    if selected_episode is None:
        if len(available_episodes) != 1:
            raise RuntimeError(
                "Cannot infer one episode for target-scene filtering. "
                f"Found {available_episodes}; pass --episode explicitly."
            )
        selected_episode = available_episodes[0]

    if not candidates:
        if allow_empty:
            return selected_episode, []
        raise RuntimeError(
            f"No numbered target-scene images matching {image_glob!r} in {image_dir}"
        )

    selected = [
        (image_number, image_path)
        for key, image_number, image_path in candidates
        if key == selected_episode
    ]
    if not selected:
        if allow_empty:
            return selected_episode, []
        raise RuntimeError(
            f"No target-scene images for {selected_episode} in {image_dir}; "
            f"available episodes: {available_episodes}"
        )
    return selected_episode, selected


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_episode, target_images = collect_target_images(
        args.image_dir, args.image_glob, args.video, args.episode,
        allow_empty=args.allow_empty_targets,
    )
    print(
        f"target_scene episode={selected_episode} images={len(target_images)}",
        flush=True,
    )

    # Keep target-scene discovery usable without importing the optional ML stack.
    # The model dependencies are required only once inference actually starts.
    from omnishotcut.engine import load_model, single_video_inference
    from omnishotcut.label_correspondence import intra_int2string, inter_int2string

    print("loading_model", flush=True)
    model, model_args = load_model(str(args.checkpoint))
    print("running_inference", flush=True)
    ranges, intra, inter, video_frames, fps = single_video_inference(
        str(args.video), model, model_args, args.overlap
    )
    total_frames = len(video_frames)
    del video_frames
    print(f"inference_complete frames={total_frames} shots={len(ranges)} fps={fps}", flush=True)

    shot_rows: list[dict[str, object]] = []
    for index, ((start, end), intra_id, inter_id) in enumerate(
        zip(ranges, intra, inter)
    ):
        shot_rows.append(
            {
                "shot_index": index,
                "start_frame_zero_based": start,
                "end_frame_exclusive_zero_based": end,
                "start_image_number_one_based": start + 1,
                "end_image_number_one_based": end,
                "start_seconds": f"{start / fps:.6f}",
                "end_seconds": f"{end / fps:.6f}",
                "duration_seconds": f"{(end - start) / fps:.6f}",
                "frame_count": end - start,
                "intra_label_id": intra_id,
                "intra_label": intra_int2string.get(intra_id, str(intra_id)),
                "inter_label_id": inter_id,
                "inter_label": inter_int2string.get(inter_id, str(inter_id)),
                "is_clean_shot": "yes" if intra_id == 0 else "no",
            }
        )

    starts = [int(row["start_frame_zero_based"]) for row in shot_rows]
    image_rows: list[dict[str, object]] = []
    for image_number, image_path in target_images:
        frame = image_number - 1
        if not 0 <= frame < total_frames:
            raise RuntimeError(
                f"Target-scene image {image_path.name} maps outside the video "
                f"(frame={frame}, total_frames={total_frames})"
            )
        index = bisect.bisect_right(starts, frame) - 1
        if index < 0 or frame >= int(shot_rows[index]["end_frame_exclusive_zero_based"]):
            raise RuntimeError(f"No shot contains {image_path.name} at frame {frame}")
        shot = shot_rows[index]
        image_rows.append(
            {
                "image_name": image_path.name,
                "episode": selected_episode,
                "image_number_one_based": image_number,
                "video_frame_zero_based": frame,
                "video_timestamp_seconds": f"{frame / fps:.6f}",
                "shot_index": index,
                "shot_start_frame_zero_based": shot["start_frame_zero_based"],
                "shot_end_frame_exclusive_zero_based": shot["end_frame_exclusive_zero_based"],
                "frame_offset_in_shot": frame - int(shot["start_frame_zero_based"]),
                "intra_label": shot["intra_label"],
                "inter_label": shot["inter_label"],
                "is_clean_shot": shot["is_clean_shot"],
            }
        )

    selected = sorted({int(row["shot_index"]) for row in image_rows})
    reference_counts = Counter(int(row["shot_index"]) for row in image_rows)
    target_scene_rows = []
    for shot_index in selected:
        target_scene_rows.append(
            {
                **shot_rows[shot_index],
                "episode": selected_episode,
                "target_scene_reference_image_count": reference_counts[shot_index],
            }
        )
    # all_segments.csv is diagnostic only. Every other segment list is scoped to
    # the target scene so downstream jobs cannot accidentally process the rest
    # of the episode.
    write_csv(args.output_dir / "all_segments.csv", shot_rows)
    image_fields = [
        "image_name", "episode", "image_number_one_based",
        "video_frame_zero_based", "video_timestamp_seconds", "shot_index",
        "shot_start_frame_zero_based", "shot_end_frame_exclusive_zero_based",
        "frame_offset_in_shot", "intra_label", "inter_label", "is_clean_shot",
    ]
    target_fields = list(shot_rows[0]) + [
        "episode", "target_scene_reference_image_count",
    ]
    write_csv(
        args.output_dir / "image_to_segment.csv", image_rows,
        fieldnames=image_fields,
    )
    write_csv(
        args.output_dir / "target_scene_segments.csv", target_scene_rows,
        fieldnames=target_fields,
    )
    clean_target_rows = [
        row for row in target_scene_rows if row["is_clean_shot"] == "yes"
    ]
    write_csv(
        args.output_dir / "clean_shots.csv",
        clean_target_rows,
        fieldnames=target_fields,
    )
    target_payload = {
        "scope": "target_scene_only",
        "episode": selected_episode,
        "image_dir": str(args.image_dir),
        "image_glob": args.image_glob,
        "segment_indices": selected,
        "segments": target_scene_rows,
    }
    (args.output_dir / "target_scene_segments.json").write_text(
        json.dumps(target_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    payload = {
        "scope": "full_video_audit_only",
        "downstream_manifest": "target_scene_segments.json",
        "video": str(args.video),
        "checkpoint": str(args.checkpoint),
        "fps": fps,
        "total_frames": total_frames,
        "overlap": args.overlap,
        "target_scene_episode": selected_episode,
        "target_scene_image_dir": str(args.image_dir),
        "target_scene_image_glob": args.image_glob,
        "ranges": ranges,
        "intra_labels": [intra_int2string.get(x, str(x)) for x in intra],
        "inter_labels": [inter_int2string.get(x, str(x)) for x in inter],
    }
    (args.output_dir / "omnishotcut_default.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    image_labels = Counter(str(row["intra_label"]) for row in image_rows)
    selected_labels = Counter(str(shot_rows[i]["intra_label"]) for i in selected)
    summary = {
        "total_frames": total_frames,
        "fps": fps,
        "all_segments": len(shot_rows),
        "all_clean_segments": sum(
            row["is_clean_shot"] == "yes" for row in shot_rows
        ),
        "clean_segments": len(clean_target_rows),
        "target_scene_clean_segments": len(clean_target_rows),
        "mapped_images": len(image_rows),
        "target_scene_episode": selected_episode,
        "target_scene_images": len(target_images),
        "target_scene_segments": len(selected),
        "segments_containing_images": len(selected),
        "image_intra_labels": dict(image_labels),
        "selected_segment_intra_labels": dict(selected_labels),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
