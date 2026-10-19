#!/usr/bin/env python3
"""Cut OmniShotCut segments containing numbered reference images.

General shots and transition segments are written to separate folders so gradual
transitions are never silently treated as clean camera shots.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import subprocess
from collections import defaultdict
from fractions import Fraction
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--segments-csv", type=Path, required=True)
    parser.add_argument("--image-map-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def probe(video: Path) -> tuple[float, float]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,start_time",
            "-of", "json", str(video),
        ],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return float(Fraction(stream["avg_frame_rate"])), float(stream.get("start_time", 0.0))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(
    video: Path, output: Path, start_seconds: float, duration: float,
    stream_start: float, preset: str, crf: int, overwrite: bool,
) -> None:
    if output.exists() and not overwrite:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    # Input frame zero has a non-zero MKV PTS.  The 2 ms epsilon prevents ffmpeg
    # from skipping a frame when the requested PTS lies on a millisecond boundary.
    seek = max(0.0, start_seconds + stream_start - 0.002)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y" if overwrite else "-n",
            "-ss", f"{seek:.9f}", "-i", str(video),
            "-t", f"{duration:.9f}",
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart", "-avoid_negative_ts", "make_zero",
            str(output),
        ],
        check=True,
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fps, stream_start = probe(args.video)
    segments = read_csv(args.segments_csv)
    image_map = read_csv(args.image_map_csv)
    segments_by_index = {int(row["shot_index"]): row for row in segments}
    images_by_segment: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in image_map:
        images_by_segment[int(row["shot_index"])].append(row)

    selected_indices = sorted(images_by_segment)
    selected_rows: list[dict[str, object]] = []
    clip_by_segment: dict[int, tuple[str, int]] = {}
    jobs: list[tuple[Path, float, float]] = []
    clean_index = transition_index = 0

    for segment_index in selected_indices:
        segment = segments_by_index[segment_index]
        is_clean = segment["is_clean_shot"] == "yes"
        if is_clean:
            clean_index += 1
            output_index = clean_index
            relative_folder = "clean"
        else:
            transition_index += 1
            output_index = transition_index
            relative_folder = "transitions_review"

        start = int(segment["start_frame_zero_based"])
        end = int(segment["end_frame_exclusive_zero_based"])
        label = segment["intra_label"].lower()
        clip_name = (
            f"{label}_{output_index:04d}_global_{segment_index:04d}_"
            f"f{start + 1:08d}-f{end:08d}.mp4"
        )
        relative_path = str(Path(relative_folder) / clip_name)
        clip_by_segment[segment_index] = (relative_path, start)
        duration = (end - start) / fps
        selected_rows.append(
            {
                "clip_path": relative_path,
                "output_index": output_index,
                "shot_index": segment_index,
                "start_frame_zero_based": start,
                "end_frame_exclusive_zero_based": end,
                "start_image_number_one_based": start + 1,
                "end_image_number_one_based": end,
                "start_seconds": f"{start / fps:.6f}",
                "end_seconds": f"{end / fps:.6f}",
                "duration_seconds": f"{duration:.6f}",
                "intra_label": segment["intra_label"],
                "inter_label": segment["inter_label"],
                "is_clean_shot": segment["is_clean_shot"],
                "reference_image_count": len(images_by_segment[segment_index]),
            }
        )
        jobs.append((args.output_dir / relative_path, start / fps, duration))

    mapped_rows: list[dict[str, object]] = []
    for row in image_map:
        segment_index = int(row["shot_index"])
        relative_path, start = clip_by_segment[segment_index]
        frame = int(row["video_frame_zero_based"])
        mapped_rows.append(
            {
                "image_name": row["image_name"],
                "image_number_one_based": row["image_number_one_based"],
                "video_frame_zero_based": frame,
                "video_timestamp_seconds": row["video_timestamp_seconds"],
                "clip_path": relative_path,
                "shot_index": segment_index,
                "frame_offset_in_clip": frame - start,
                "timestamp_in_clip_seconds": f"{(frame - start) / fps:.6f}",
                "intra_label": row["intra_label"],
                "inter_label": row["inter_label"],
                "is_clean_shot": row["is_clean_shot"],
            }
        )

    write_csv(args.output_dir / "selected_segments.csv", selected_rows)
    write_csv(args.output_dir / "image_to_clip.csv", mapped_rows)

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                render, args.video, output, start, duration, stream_start,
                args.preset, args.crf, args.overwrite,
            )
            for output, start, duration in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            if completed % 25 == 0 or completed == len(futures):
                print(f"rendered={completed}/{len(futures)}", flush=True)

    print(f"clean_clips={clean_index}")
    print(f"transition_clips={transition_index}")
    print(f"mapped_images={len(mapped_rows)}")
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
