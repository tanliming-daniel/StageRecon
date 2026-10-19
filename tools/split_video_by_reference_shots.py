#!/usr/bin/env python3
"""Split a video into shots that contain numbered reference images.

Reference files produced by ffmpeg's ``%08d`` pattern are one-based, so image
number N maps to zero-based decoded video frame N - 1.  The script detects hard
cuts, keeps each shot containing at least one reference image, writes MP4 clips,
and emits shot/image mapping CSV files.
"""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import csv
import json
import re
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


NUMBER_RE = re.compile(r"(\d+)$")
PTS_RE = re.compile(r"pts_time:([0-9.]+)")
SCORE_RE = re.compile(r"lavfi.scene_score=([0-9.]+)")


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    start_time: float
    duration: float
    frame_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument(
        "--image-glob", default="*",
        help="Filename glob inside --image-dir, e.g. 'Friends_S08E09_*'",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--candidate-threshold", type=float, default=0.10)
    parser.add_argument("--min-cut-gap-frames", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def probe_video(video: Path) -> VideoInfo:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,start_time:format=duration",
            "-of", "json", str(video),
        ],
        check=True, capture_output=True, text=True,
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    fps = float(Fraction(stream["avg_frame_rate"]))
    start_time = float(stream.get("start_time", 0.0))
    duration = float(data["format"]["duration"])
    return VideoInfo(fps, start_time, duration, round(duration * fps))


def collect_images(image_dir: Path, image_glob: str) -> list[tuple[int, Path]]:
    images: list[tuple[int, Path]] = []
    for path in image_dir.glob(image_glob):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        match = NUMBER_RE.search(path.stem)
        if match:
            images.append((int(match.group(1)), path))
    images.sort()
    if not images:
        raise RuntimeError(f"No numbered images matching {image_glob!r} found in {image_dir}")
    return images


def detect_cuts(
    video: Path, info: VideoInfo, score_file: Path, candidate_threshold: float,
    threshold: float, min_gap: int,
) -> list[tuple[int, float]]:
    score_file.unlink(missing_ok=True)
    filter_graph = (
        f"select=gt(scene\,{candidate_threshold}),"
        f"metadata=print:file={score_file}"
    )
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
        "-map", "0:v:0", "-vf", filter_graph, "-an", "-sn",
        "-f", "null", "-",
    ])

    candidates: list[tuple[int, float]] = []
    pts_time: float | None = None
    for line in score_file.read_text(encoding="utf-8").splitlines():
        pts_match = PTS_RE.search(line)
        if pts_match:
            pts_time = float(pts_match.group(1))
        score_match = SCORE_RE.search(line)
        if score_match and pts_time is not None:
            score = float(score_match.group(1))
            if score >= threshold:
                frame = round((pts_time - info.start_time) * info.fps)
                if 0 < frame < info.frame_count:
                    candidates.append((frame, score))

    cuts: list[tuple[int, float]] = []
    for candidate in candidates:
        if cuts and candidate[0] - cuts[-1][0] < min_gap:
            if candidate[1] > cuts[-1][1]:
                cuts[-1] = candidate
        else:
            cuts.append(candidate)
    return cuts


def timestamp(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def render_clip(
    video: Path, output: Path, start: float, duration: float,
    stream_start_time: float, preset: str, crf: int, overwrite: bool,
) -> None:
    if output.exists() and not overwrite:
        return
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y" if overwrite else "-n",
        # Frame zero in this MKV starts at a non-zero PTS.  Add that offset so
        # logical frame N lands at clip frame N-start_frame after seeking.
        # Seek a tiny epsilon before the exact PTS.  Some MKV timestamps are
        # millisecond-quantized; seeking to equality can otherwise skip the
        # requested first frame at those boundaries.
        "-ss", f"{max(0.0, start + stream_start_time - 0.002):.9f}",
        "-i", str(video),
        "-t", f"{duration:.9f}",
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart", "-avoid_negative_ts", "make_zero",
        str(output),
    ]
    run(command)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = args.output_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    info = probe_video(args.video)
    images = collect_images(args.image_dir, args.image_glob)
    cuts = detect_cuts(
        args.video, info, args.output_dir / "scene_scores.txt",
        args.candidate_threshold, args.threshold, args.min_cut_gap_frames,
    )
    boundaries = [0] + [frame for frame, _ in cuts] + [info.frame_count]
    scores_by_frame = {frame: score for frame, score in cuts}

    image_records: list[tuple[int, Path, int]] = []
    for image_number, image_path in images:
        frame = image_number - 1
        if not 0 <= frame < info.frame_count:
            raise RuntimeError(f"{image_path.name} maps outside the video")
        shot_index = bisect.bisect_right(boundaries, frame) - 1
        image_records.append((frame, image_path, shot_index))

    selected_shots = sorted({record[2] for record in image_records})
    clip_by_shot: dict[int, str] = {}
    shot_rows: list[dict[str, object]] = []
    render_jobs: list[tuple[Path, float, float]] = []
    for output_index, shot_index in enumerate(selected_shots, start=1):
        start_frame = boundaries[shot_index]
        end_frame = boundaries[shot_index + 1]
        start_seconds = start_frame / info.fps
        end_seconds = end_frame / info.fps
        clip_name = (
            f"shot_{output_index:04d}_global_{shot_index:04d}_"
            f"f{start_frame + 1:08d}-f{end_frame:08d}.mp4"
        )
        clip_by_shot[shot_index] = clip_name
        references = [r for r in image_records if r[2] == shot_index]
        shot_rows.append({
            "clip_name": clip_name,
            "output_shot_index": output_index,
            "global_shot_index": shot_index,
            "start_frame_zero_based": start_frame,
            "end_frame_exclusive_zero_based": end_frame,
            "start_image_number_one_based": start_frame + 1,
            "end_image_number_one_based": end_frame,
            "start_seconds": f"{start_seconds:.6f}",
            "end_seconds": f"{end_seconds:.6f}",
            "duration_seconds": f"{end_seconds - start_seconds:.6f}",
            "start_timestamp": timestamp(start_seconds),
            "end_timestamp": timestamp(end_seconds),
            "cut_score_at_start": (
                f"{scores_by_frame[start_frame]:.6f}" if start_frame in scores_by_frame else ""
            ),
            "reference_image_count": len(references),
        })
        render_jobs.append((clips_dir / clip_name, start_seconds, end_seconds - start_seconds))

    image_rows: list[dict[str, object]] = []
    for frame, image_path, shot_index in image_records:
        start_frame = boundaries[shot_index]
        image_rows.append({
            "image_name": image_path.name,
            "image_number_one_based": frame + 1,
            "video_frame_zero_based": frame,
            "video_timestamp_seconds": f"{frame / info.fps:.6f}",
            "clip_name": clip_by_shot[shot_index],
            "global_shot_index": shot_index,
            "frame_offset_in_clip": frame - start_frame,
            "timestamp_in_clip_seconds": f"{(frame - start_frame) / info.fps:.6f}",
        })

    write_csv(args.output_dir / "shots.csv", shot_rows)
    write_csv(args.output_dir / "image_to_clip.csv", image_rows)

    if not args.manifest_only:
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    render_clip, args.video, output, start, duration,
                    info.start_time, args.preset, args.crf, args.overwrite,
                )
                for output, start, duration in render_jobs
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
                completed += 1
                if completed % 20 == 0 or completed == len(futures):
                    print(f"rendered={completed}/{len(futures)}", flush=True)

    print(f"fps={info.fps:.12g}")
    print(f"detected_shots={len(boundaries) - 1}")
    print(f"selected_shots={len(selected_shots)}")
    print(f"mapped_images={len(image_records)}")
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
