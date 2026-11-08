#!/usr/bin/env python3
from __future__ import annotations

import collections
import json
from pathlib import Path

from PIL import Image


ROOT = Path(
    "/root/userdata/300workspace/StageRecon/data/raw/"
    "friends_gallery_sam_width_preview"
)


def main() -> None:
    directories = sorted(path for path in ROOT.glob("S*/person_*") if path.is_dir())
    errors: list[str] = []
    sizes: collections.Counter[tuple[int, int]] = collections.Counter()
    reference_counts: collections.Counter[int] = collections.Counter()
    valid_pngs = 0
    metadata_files = 0
    final_prompt_records = 0

    for directory in directories:
        relative = directory.relative_to(ROOT).as_posix()
        image_path = directory / "reference_bank.png"
        metadata_path = directory / "reference_bank.json"

        if not image_path.is_file():
            errors.append(f"{relative}: missing png")
        else:
            try:
                with Image.open(image_path) as image:
                    sizes[image.size] += 1
                    image.verify()
                valid_pngs += 1
            except Exception as exc:
                errors.append(f"{relative}: bad png {exc}")

        if not metadata_path.is_file():
            errors.append(f"{relative}: missing json")
            continue

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata_files += 1
            reference_counts[metadata.get("source_image_count")] += 1
            if metadata.get("requested_size") != [1920, 1088]:
                errors.append(
                    f"{relative}: requested_size={metadata.get('requested_size')}"
                )
            if metadata.get("generated_size") != [1920, 1088]:
                errors.append(
                    f"{relative}: generated_size={metadata.get('generated_size')}"
                )
            prompt = metadata.get("prompt", "")
            required = (
                "《老友记》",
                "真实演员",
                "正面脸部超近特写",
                "完整全身背面",
            )
            if all(fragment in prompt for fragment in required):
                final_prompt_records += 1
            else:
                errors.append(f"{relative}: old prompt")
        except Exception as exc:
            errors.append(f"{relative}: bad json {exc}")

    print(
        json.dumps(
            {
                "person_dirs": len(directories),
                "valid_pngs": valid_pngs,
                "metadata_files": metadata_files,
                "sizes": {str(key): value for key, value in sizes.items()},
                "source_image_counts": dict(reference_counts),
                "final_prompt_records": final_prompt_records,
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
