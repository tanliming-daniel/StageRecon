#!/usr/bin/env python3
"""Render proposal-specific masked image views from COLMAP tracks.

Each proposal is defined by a set of 3D point ids produced by
``build_semantic_pointcloud.py``. For every source frame, this script:

1. finds the 2D COLMAP observations belonging to that proposal;
2. intersects the proposal region with its semantic segmentation mask;
3. writes a transparent crop, a white-background crop, and a binary mask;
4. records the source frame, bbox, track count, and mask statistics.

The resulting crops are intended as inputs to image-to-3D asset APIs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
SITCOMS3D_ROOT = REPO_ROOT / "sitcoms3D"
if str(SITCOMS3D_ROOT) not in sys.path:
    sys.path.insert(0, str(SITCOMS3D_ROOT))

from sitcoms3D.utils.colmap_utils import read_model  # noqa: E402


def normalize_name(name: str) -> str:
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


def resolve_path(base: Path, value: str | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def image_mask_path(root: Path, source: str, image_name: str) -> Path:
    return root / source / f"{Path(image_name).stem}.png"


def load_mask(path: Path, expected_shape: Tuple[int, int]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing segmentation mask: {path}")
    mask = np.asarray(Image.open(path))
    if mask.ndim != 2 or tuple(mask.shape) != expected_shape:
        raise ValueError(
            f"Expected {expected_shape} single-channel mask at {path}, got {mask.shape}"
        )
    return mask


def load_proposals(
    proposal_dir: Path,
    manifest_path: Path | None,
    selected_categories: Sequence[str] | None,
) -> Tuple[dict, List[dict]]:
    manifest_path = manifest_path or proposal_dir / "proposals.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing proposal manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    category_filter = (
        {normalize_name(item) for item in selected_categories}
        if selected_categories
        else None
    )
    proposals = [
        proposal
        for proposal in manifest.get("proposals", [])
        if category_filter is None
        or normalize_name(proposal["category"]) in category_filter
    ]
    if not proposals:
        raise ValueError(f"No proposals selected from {manifest_path}")
    return manifest, proposals


def load_point_ids(proposal_dir: Path, proposal: Mapping[str, object]) -> np.ndarray:
    relative_path = Path(str(proposal["point3d_ids"]))
    path = relative_path if relative_path.is_absolute() else proposal_dir / relative_path
    point_ids = np.asarray(np.load(path), dtype=np.int64)
    if point_ids.ndim != 1 or len(point_ids) == 0:
        raise ValueError(f"Proposal has no point3D ids: {path}")
    return point_ids


def category_mask(
    stuff: np.ndarray,
    thing: np.ndarray,
    category_labels: Sequence[Mapping[str, object]],
) -> np.ndarray:
    result = np.zeros(stuff.shape, dtype=bool)
    for item in category_labels:
        source = str(item["source"])
        label_id = int(item["label_id"])
        if source == "thing":
            result |= thing == label_id
        elif source == "stuff":
            # Stuff is background only where no thing instance occludes it.
            result |= (thing == 0) & (stuff == label_id)
        else:
            raise ValueError(f"Unknown segmentation source: {source}")
    return result


def draw_track_region(
    shape: Tuple[int, int],
    xy: np.ndarray,
    radius: int,
    bbox_padding: float,
) -> np.ndarray:
    height, width = shape
    in_bounds = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )
    xy = xy[in_bounds]
    if len(xy) == 0:
        return np.zeros(shape, dtype=bool)

    x_min, y_min = np.floor(xy.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(xy.max(axis=0)).astype(int)
    box_width = max(1, x_max - x_min + 1)
    box_height = max(1, y_max - y_min + 1)
    padding = int(round(max(box_width, box_height) * bbox_padding))
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(width - 1, x_max + padding)
    y_max = min(height - 1, y_max + padding)

    region = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(region)
    draw.rectangle((x_min, y_min, x_max, y_max), fill=255)
    if radius > 0:
        # The expanded bbox is the primary region. Circles make sparse
        # observations useful when a proposal is only partially tracked.
        for x, y in np.rint(xy).astype(int):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    return np.asarray(region, dtype=bool)


def connected_components(mask: np.ndarray) -> List[np.ndarray]:
    """Return flat pixel indices for 8-connected components."""
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components: List[np.ndarray] = []
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        pixels: List[int] = []
        while stack:
            current_y, current_x = stack.pop()
            pixels.append(current_y * width + current_x)
            for delta_y in (-1, 0, 1):
                for delta_x in (-1, 0, 1):
                    if delta_y == 0 and delta_x == 0:
                        continue
                    neighbor_y = current_y + delta_y
                    neighbor_x = current_x + delta_x
                    if (
                        0 <= neighbor_y < height
                        and 0 <= neighbor_x < width
                        and mask[neighbor_y, neighbor_x]
                        and not visited[neighbor_y, neighbor_x]
                    ):
                        visited[neighbor_y, neighbor_x] = True
                        stack.append((neighbor_y, neighbor_x))
        components.append(np.asarray(pixels, dtype=np.int64))
    return components


def choose_component_mask(
    semantic_mask: np.ndarray,
    track_region: np.ndarray,
    xy: np.ndarray,
    min_area: int,
) -> np.ndarray:
    del track_region
    height, width = semantic_mask.shape
    track_xy = np.rint(xy).astype(np.int64)
    in_bounds = (
        (track_xy[:, 0] >= 0)
        & (track_xy[:, 0] < width)
        & (track_xy[:, 1] >= 0)
        & (track_xy[:, 1] < height)
    )
    track_flat = set((track_xy[in_bounds, 1] * width + track_xy[in_bounds, 0]).tolist())
    if not track_flat:
        return np.zeros(semantic_mask.shape, dtype=bool)

    result = np.zeros(semantic_mask.shape, dtype=bool)
    for component in connected_components(semantic_mask):
        if track_flat.intersection(component.tolist()):
            result[component // width, component % width] = True
    if int(result.sum()) < min_area:
        return np.zeros(semantic_mask.shape, dtype=bool)
    return result


def padded_bbox(
    mask: np.ndarray,
    padding: float,
    minimum_size: int,
) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot compute bbox for an empty mask")
    height, width = mask.shape
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    box_width = max(1, x_max - x_min + 1)
    box_height = max(1, y_max - y_min + 1)
    pad_x = max(minimum_size, int(round(box_width * padding)))
    pad_y = max(minimum_size, int(round(box_height * padding)))
    return (
        max(0, x_min - pad_x),
        max(0, y_min - pad_y),
        min(width, x_max + pad_x + 1),
        min(height, y_max + pad_y + 1),
    )


def rgba_crop(
    rgb: Image.Image,
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Image.Image:
    x_min, y_min, x_max, y_max = bbox
    rgb_array = np.asarray(rgb.convert("RGB"), dtype=np.uint8)
    alpha = (mask.astype(np.uint8) * 255)[y_min:y_max, x_min:x_max]
    crop = rgb_array[y_min:y_max, x_min:x_max]
    rgba = np.dstack([crop, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def write_view(
    rgb: Image.Image,
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    crop = rgba_crop(rgb, mask, bbox)
    alpha = np.asarray(crop.getchannel("A"))
    crop.save(output_dir / f"{stem}_rgba.png")

    white = Image.new("RGBA", crop.size, (255, 255, 255, 255))
    white.alpha_composite(crop)
    white.convert("RGB").save(output_dir / f"{stem}_white.jpg", quality=95)

    black = Image.new("RGBA", crop.size, (0, 0, 0, 255))
    black.alpha_composite(crop)
    black.convert("RGB").save(output_dir / f"{stem}_black.jpg", quality=95)

    Image.fromarray(alpha, mode="L").save(output_dir / f"{stem}_mask.png")
    return {
        "rgba": str((output_dir / f"{stem}_rgba.png").name),
        "white": str((output_dir / f"{stem}_white.jpg").name),
        "black": str((output_dir / f"{stem}_black.jpg").name),
        "mask": str((output_dir / f"{stem}_mask.png").name),
    }


def proposal_view(
    environment_dir: Path,
    proposal_dir: Path,
    proposal: Mapping[str, object],
    manifest: Mapping[str, object],
    images: Mapping[int, object],
    max_views: int,
    min_track_points: int,
    min_mask_area: int,
    track_radius: int,
    bbox_padding: float,
    bbox_min_padding: int,
    min_track_overlap: float,
    max_person_fraction: float,
    max_images: int | None,
    output_dir: Path,
) -> dict:
    point_ids = load_point_ids(proposal_dir, proposal)
    category = normalize_name(str(proposal["category"]))
    category_labels = manifest["category_labels"][category]
    mask_root = environment_dir / "segmentations"
    image_root = environment_dir / "images"
    candidates: List[dict] = []

    sorted_images = sorted(images.values(), key=lambda image: image.name)
    if max_images is not None:
        sorted_images = sorted_images[:max_images]

    for image in sorted_images:
        source_path = image_root / image.name
        stuff_path = image_mask_path(mask_root, "stuff", image.name)
        thing_path = image_mask_path(mask_root, "thing", image.name)
        if not source_path.exists() or not stuff_path.exists() or not thing_path.exists():
            continue

        rgb = Image.open(source_path).convert("RGB")
        width, height = rgb.size
        stuff = load_mask(stuff_path, (height, width))
        thing = load_mask(thing_path, (height, width))
        point3d_ids = np.asarray(image.point3D_ids, dtype=np.int64)
        selected = np.isin(point3d_ids, point_ids)
        if int(selected.sum()) < min_track_points:
            continue
        xy = np.asarray(image.xys, dtype=np.float64)[selected]
        track_region = draw_track_region(
            (height, width), xy, radius=track_radius, bbox_padding=bbox_padding
        )
        semantic = category_mask(stuff, thing, category_labels)
        track_xy = np.rint(xy).astype(np.int64)
        in_bounds = (
            (track_xy[:, 0] >= 0)
            & (track_xy[:, 0] < width)
            & (track_xy[:, 1] >= 0)
            & (track_xy[:, 1] < height)
        )
        track_overlap = float(
            semantic[track_xy[in_bounds, 1], track_xy[in_bounds, 0]].mean()
        ) if np.any(in_bounds) else 0.0
        if track_overlap < min_track_overlap:
            continue
        proposal_mask = choose_component_mask(
            semantic, track_region, xy, min_area=min_mask_area
        )
        mask_area = int(proposal_mask.sum())
        if mask_area < min_mask_area:
            continue
        bbox = padded_bbox(proposal_mask, bbox_padding, bbox_min_padding)
        x_min, y_min, x_max, y_max = bbox
        person_fraction = float(
            np.count_nonzero(thing[y_min:y_max, x_min:x_max] == 1)
            / max(1, (x_max - x_min) * (y_max - y_min))
        )
        if person_fraction > max_person_fraction:
            continue
        candidates.append(
            {
                "image": image,
                "rgb": rgb,
                "mask": proposal_mask,
                "bbox": bbox,
                "track_count": int(selected.sum()),
                "mask_area": mask_area,
                "track_overlap": track_overlap,
                "person_fraction": person_fraction,
                "coverage": float(mask_area / (width * height)),
            }
        )

    candidates.sort(
        key=lambda item: (item["mask_area"] * item["track_overlap"], item["track_count"]),
        reverse=True,
    )
    if max_views > 0:
        candidates = candidates[:max_views]
    proposal_output = output_dir / str(proposal["id"])
    proposal_output.mkdir(parents=True, exist_ok=True)
    views: List[dict] = []
    for view_index, candidate in enumerate(candidates):
        image = candidate["image"]
        stem = f"view_{view_index:03d}"
        files = write_view(
            candidate["rgb"],
            candidate["mask"],
            candidate["bbox"],
            proposal_output,
            stem,
        )
        x_min, y_min, x_max, y_max = candidate["bbox"]
        views.append(
            {
                "view_id": stem,
                "source_image": image.name,
                "source_image_path": str((image_root / image.name).resolve()),
                "files": files,
                "bbox_xyxy": [x_min, y_min, x_max, y_max],
                "crop_size": [x_max - x_min, y_max - y_min],
                "track_count": candidate["track_count"],
                "track_overlap": candidate["track_overlap"],
                "person_fraction_in_crop": candidate["person_fraction"],
                "mask_area": candidate["mask_area"],
                "mask_fraction_of_frame": candidate["coverage"],
            }
        )

    return {
        "proposal_id": str(proposal["id"]),
        "category": category,
        "point_count": int(proposal["point_count"]),
        "selected_view_count": len(views),
        "views": views,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate proposal-specific masked RGB views for image-to-3D APIs."
    )
    parser.add_argument("--environment-dir", type=Path, required=True)
    parser.add_argument(
        "--proposal-dir",
        type=Path,
        default=None,
        help="Semantic proposal directory. Defaults to <environment-dir>/semantic_proposals.",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--colmap-dir", type=str, default=None)
    parser.add_argument("--colmap-format", choices=["", ".bin", ".txt"], default="")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <environment-dir>/proposal_views.",
    )
    parser.add_argument(
        "--categories",
        type=lambda value: [item.strip() for item in value.split(",") if item.strip()],
        default=None,
        help="Optional comma-separated category filter.",
    )
    parser.add_argument(
        "--max-views",
        type=int,
        default=12,
        help="Top views per proposal. 0 keeps all valid views.",
    )
    parser.add_argument("--min-track-points", type=int, default=3)
    parser.add_argument("--min-mask-area", type=int, default=100)
    parser.add_argument(
        "--track-radius",
        type=int,
        default=12,
        help="Radius in pixels used to connect sparse 3D track observations.",
    )
    parser.add_argument(
        "--bbox-padding",
        type=float,
        default=0.08,
        help="Relative padding around projected tracks and final object crop.",
    )
    parser.add_argument("--bbox-min-padding", type=int, default=8)
    parser.add_argument("--min-track-overlap", type=float, default=0.6)
    parser.add_argument("--max-person-fraction", type=float, default=0.35)
    parser.add_argument("--max-images", type=int, default=None)
    return parser


def resolve_colmap_dir(environment_dir: Path, value: str | None) -> Path:
    if value is None:
        return environment_dir / "colmap"
    path = Path(value)
    return path if path.is_absolute() else environment_dir / path


def main() -> None:
    args = make_parser().parse_args()
    if args.max_views < 0:
        raise ValueError("--max-views must be >= 0")
    if args.min_track_points < 1 or args.min_mask_area < 1:
        raise ValueError("min-track-points and min-mask-area must be positive")
    if args.track_radius < 0 or args.bbox_padding < 0 or args.bbox_min_padding < 0:
        raise ValueError("track radius and bbox padding must be non-negative")
    if not 0.0 <= args.min_track_overlap <= 1.0:
        raise ValueError("min-track-overlap must be in [0, 1]")
    if not 0.0 <= args.max_person_fraction <= 1.0:
        raise ValueError("max-person-fraction must be in [0, 1]")

    environment_dir = args.environment_dir.resolve()
    proposal_dir = resolve_path(
        environment_dir,
        str(args.proposal_dir) if args.proposal_dir else None,
        environment_dir / "semantic_proposals",
    ).resolve()
    output_dir = resolve_path(
        environment_dir,
        str(args.output_dir) if args.output_dir else None,
        environment_dir / "proposal_views",
    ).resolve()
    manifest, proposals = load_proposals(
        proposal_dir, args.manifest.resolve() if args.manifest else None, args.categories
    )
    _, images, _ = read_model(
        str(resolve_colmap_dir(environment_dir, args.colmap_dir)),
        ext=args.colmap_format,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    proposal_manifests = []
    for index, proposal in enumerate(proposals, start=1):
        result = proposal_view(
            environment_dir=environment_dir,
            proposal_dir=proposal_dir,
            proposal=proposal,
            manifest=manifest,
            images=images,
            max_views=args.max_views,
            min_track_points=args.min_track_points,
            min_mask_area=args.min_mask_area,
            track_radius=args.track_radius,
            bbox_padding=args.bbox_padding,
            bbox_min_padding=args.bbox_min_padding,
            min_track_overlap=args.min_track_overlap,
            max_person_fraction=args.max_person_fraction,
            max_images=args.max_images,
            output_dir=output_dir,
        )
        proposal_manifests.append(result)
        print(
            f"[INFO] {index}/{len(proposals)} {result['proposal_id']}: "
            f"{result['selected_view_count']} views"
        )

    output_manifest = {
        "schema_version": 1,
        "environment_dir": str(environment_dir),
        "proposal_dir": str(proposal_dir),
        "output_dir": str(output_dir),
        "parameters": {
            "max_views": args.max_views,
            "min_track_points": args.min_track_points,
            "min_mask_area": args.min_mask_area,
            "track_radius": args.track_radius,
            "bbox_padding": args.bbox_padding,
            "bbox_min_padding": args.bbox_min_padding,
            "max_images": args.max_images,
            "min_track_overlap": args.min_track_overlap,
            "max_person_fraction": args.max_person_fraction,
        },
        "proposals": proposal_manifests,
    }
    (output_dir / "views.json").write_text(
        json.dumps(output_manifest, indent=2, ensure_ascii=True) + "\n"
    )
    print(json.dumps(
        {
            "proposal_count": len(proposal_manifests),
            "view_count": sum(
                item["selected_view_count"] for item in proposal_manifests
            ),
            "output_dir": str(output_dir),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
