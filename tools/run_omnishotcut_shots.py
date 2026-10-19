#!/usr/bin/env python3
"""Run OmniShotCut and save frame-accurate ranges plus image mappings."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
from collections import Counter
from pathlib import Path

from omnishotcut.engine import load_model, single_video_inference
from omnishotcut.label_correspondence import intra_int2string, inter_int2string


NUMBER_RE = re.compile(r"(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--image-glob", default="*")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overlap", type=int, default=10)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    for image_path in sorted(args.image_dir.glob(args.image_glob)):
        match = NUMBER_RE.search(image_path.stem)
        if not match:
            continue
        image_number = int(match.group(1))
        frame = image_number - 1
        index = bisect.bisect_right(starts, frame) - 1
        if index < 0 or frame >= int(shot_rows[index]["end_frame_exclusive_zero_based"]):
            raise RuntimeError(f"No shot contains {image_path.name} at frame {frame}")
        shot = shot_rows[index]
        image_rows.append(
            {
                "image_name": image_path.name,
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

    write_csv(args.output_dir / "all_segments.csv", shot_rows)
    write_csv(args.output_dir / "image_to_segment.csv", image_rows)
    write_csv(
        args.output_dir / "clean_shots.csv",
        [row for row in shot_rows if row["is_clean_shot"] == "yes"],
    )
    payload = {
        "video": str(args.video),
        "checkpoint": str(args.checkpoint),
        "fps": fps,
        "total_frames": total_frames,
        "overlap": args.overlap,
        "ranges": ranges,
        "intra_labels": [intra_int2string.get(x, str(x)) for x in intra],
        "inter_labels": [inter_int2string.get(x, str(x)) for x in inter],
    }
    (args.output_dir / "omnishotcut_default.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    selected = {int(row["shot_index"]) for row in image_rows}
    image_labels = Counter(str(row["intra_label"]) for row in image_rows)
    selected_labels = Counter(str(shot_rows[i]["intra_label"]) for i in selected)
    summary = {
        "total_frames": total_frames,
        "fps": fps,
        "all_segments": len(shot_rows),
        "clean_segments": sum(row["is_clean_shot"] == "yes" for row in shot_rows),
        "mapped_images": len(image_rows),
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
