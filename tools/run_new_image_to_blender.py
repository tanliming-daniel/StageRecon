#!/usr/bin/env python3
"""Run the Friends scene localization-to-camera-info and preview pipeline.

The runner stages one new image, retrieves registered scene frames by coarse
appearance, extracts DISK features, localizes against existing COLMAP tracks,
exports importable camera information, and produces a native GPU viewport
preview from the Blender scene geometry.

It never adds the query image to COLMAP or modifies the source Blender file.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SITCOMS3D_ROOT = REPO_ROOT / "sitcoms3D"
if str(SITCOMS3D_ROOT) not in sys.path:
    sys.path.insert(0, str(SITCOMS3D_ROOT))

from sitcoms3D.utils.colmap_utils import read_model  # noqa: E402


DEFAULT_ENVIRONMENT = (
    REPO_ROOT
    / "sitcoms3D/data/sparse_reconstruction_and_nerf_data/Friends-monica_apartment"
)
DEFAULT_BLENDER = Path("/root/userdata/300workspace/zdj/blender/blender")
DISK_DETECT = SITCOMS3D_ROOT / "external/disk/detect.py"
DEFAULT_DISK_CHECKPOINT = REPO_ROOT / "outputs/Friends-monica_apartment/new_image_localization/models/depth-save.pth"
LOCALIZER = REPO_ROOT / "tools/localize_query_from_disk.py"
CAMERA_EXPORTER = REPO_ROOT / "tools/export_predicted_camera_info.py"
CAMERA_IMPORTER = REPO_ROOT / "tools/import_predicted_camera.py"
VIEWPORT_CAPTURE = REPO_ROOT / "tools/capture_viewport_preview.py"
GLOBAL_RETRIEVER = REPO_ROOT / "tools/retrieve_disk_global.py"
SAM3_PERSON_MASKER = REPO_ROOT / "tools/extract_person_mask_sam3.py"
DEFAULT_GLOBAL_INDEX = (
    REPO_ROOT / "outputs/Friends-monica_apartment/models/disk_global_retrieval_index.npz"
)
DEFAULT_SAM3_ROOT = Path("/root/userdata/300workspace/sam3")
DEFAULT_SAM3_CHECKPOINT = Path("/root/userdata/300workspace/ckpt/sam3/sam3.pt")
DEFAULT_BLENDER_TEMP = REPO_ROOT / ".cache/blender/tmp"
DEFAULT_BLENDER_RUNTIME = REPO_ROOT / ".cache/blender/xdg-runtime"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-image", type=Path, required=True)
    parser.add_argument("--blender-input", type=Path, required=True)
    parser.add_argument("--environment-dir", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--camera-name", default="PredictedCamera_query_0001")
    parser.add_argument("--blender-executable", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--viewport-display", default=":88")
    parser.add_argument("--disk-checkpoint", type=Path, default=DEFAULT_DISK_CHECKPOINT)
    parser.add_argument("--point-cloud-object", default="pointcloud")
    parser.add_argument("--query-mask", type=Path)
    parser.add_argument(
        "--person-mask",
        choices=("auto", "off"),
        default="auto",
        help="Generate a SAM3 dynamic-person mask unless --query-mask is supplied (default: auto).",
    )
    parser.add_argument("--sam3-root", type=Path, default=DEFAULT_SAM3_ROOT)
    parser.add_argument("--sam3-checkpoint", type=Path, default=DEFAULT_SAM3_CHECKPOINT)
    parser.add_argument("--sam3-confidence-threshold", type=float, default=0.35)
    parser.add_argument("--sam3-dilate-pixels", type=int, default=12)
    parser.add_argument("--disable-reference-person-mask", action="store_true")
    parser.add_argument("--focal-pixels", default="")
    parser.add_argument(
        "--focal-multipliers",
        default="0.8,0.9,1.0,1.1,1.2,1.35,1.5,1.7,1.9,2.1",
    )
    parser.add_argument("--exclude-reference", action="append", default=[])
    parser.add_argument("--retrieval-pool", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-features", type=int, default=2048)
    parser.add_argument("--ratio", type=float, default=0.90)
    parser.add_argument(
        "--global-index",
        type=Path,
        default=DEFAULT_GLOBAL_INDEX,
        help="Cached pooled-DISK retrieval index used only if coarse localization fails.",
    )
    parser.add_argument("--global-top-k", type=int, default=200)
    parser.add_argument("--global-ratio", type=float, default=0.95)
    parser.add_argument("--reprojection-error", type=float, default=8.0)
    preview_group = parser.add_mutually_exclusive_group()
    preview_group.add_argument(
        "--preview",
        dest="preview",
        action="store_true",
        default=True,
        help="Emit the Solid+Texture+Studio predicted-camera PNG (default).",
    )
    preview_group.add_argument(
        "--no-preview",
        dest="preview",
        action="store_false",
        help="Skip the predicted-camera preview PNG.",
    )
    parser.add_argument("--preview-width", type=int, default=960)
    parser.add_argument("--preview-height", type=int, default=540)
    parser.add_argument(
        "--hide-collection",
        action="append",
        default=["actor1", "camera"],
        help="Temporarily hide a collection only for the exported preview.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def blender_environment() -> dict[str, str]:
    DEFAULT_BLENDER_TEMP.mkdir(parents=True, exist_ok=True)
    DEFAULT_BLENDER_RUNTIME.mkdir(parents=True, exist_ok=True)
    DEFAULT_BLENDER_RUNTIME.chmod(0o700)
    environment = os.environ.copy()
    environment.update(
        {
            "TMPDIR": str(DEFAULT_BLENDER_TEMP),
            "TEMP": str(DEFAULT_BLENDER_TEMP),
            "TMP": str(DEFAULT_BLENDER_TEMP),
            "XDG_RUNTIME_DIR": str(DEFAULT_BLENDER_RUNTIME),
        }
    )
    return environment


def run(
    command: list[str],
    cwd: Path | None = None,
    *,
    environment: dict[str, str] | None = None,
) -> None:
    print("$", shlex.join(command), flush=True)
    subprocess.run(
        command,
        check=True,
        cwd=str(cwd) if cwd else None,
        env=environment,
    )


def try_run(command: list[str], cwd: Path | None = None) -> bool:
    """Run a recoverable stage, returning false so the caller can use a fallback."""
    print("$", shlex.join(command), flush=True)
    completed = subprocess.run(command, check=False, cwd=str(cwd) if cwd else None)
    if completed.returncode:
        print(f"Recoverable stage failed with exit code {completed.returncode}", flush=True)
        return False
    return True


def measure(stage: str, durations: dict[str, float], action):
    """Run one pipeline stage and retain wall-clock time in its manifest."""
    started = time.perf_counter()
    result = action()
    durations[stage] = round(time.perf_counter() - started, 3)
    return result


from pose_quality import assess_pose_quality


def stage_query_image(source: Path, destination: Path, mask: Path | None) -> tuple[Path, Path]:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode query image: {source}")
    if mask is not None:
        mask_image = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        if mask_image is None:
            raise ValueError(f"OpenCV could not decode query mask: {mask}")
        if mask_image.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"Query mask shape {mask_image.shape[:2]} does not match image shape {image.shape[:2]}"
            )
    destination.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    if not suffix:
        raise ValueError(f"Query image must have a file extension: {source}")
    staged = destination / f"query{suffix}"
    shutil.copy2(source, staged)
    image_list = destination / "query_images.txt"
    image_list.write_text(f"{staged.name}\n")
    return staged, image_list


def disk_input_size(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read staged query image: {image_path}")
    height, width = image.shape[:2]
    return ((height + 15) // 16 * 16, (width + 15) // 16 * 16)


def compact_visual_features(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    small = cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA)
    color = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    edge_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    edge_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    structure = np.stack((edge_x, edge_y), axis=-1)
    return color, structure


def build_retrieval_candidates(
    query_image: Path,
    query_mask: Path | None,
    environment: Path,
    output_path: Path,
    count: int,
) -> None:
    if count < 2:
        raise ValueError("--retrieval-pool must be at least 2")
    query = cv2.imread(str(query_image), cv2.IMREAD_COLOR)
    if query is None:
        raise ValueError(f"Could not read staged query image: {query_image}")
    query_color, query_structure = compact_visual_features(query)
    query_static: np.ndarray | None = None
    if query_mask is not None:
        mask = cv2.imread(str(query_mask), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read query mask: {query_mask}")
        if mask.shape != query.shape[:2]:
            raise ValueError(
                f"Query mask shape {mask.shape} does not match image shape {query.shape[:2]}"
            )
        query_static = cv2.resize(mask, (160, 90), interpolation=cv2.INTER_NEAREST) == 0
        # Near-full-frame actor shots are not useful for pixel-layout appearance
        # retrieval. The later global DISK stage still uses all remaining static
        # descriptors, but the coarse stage falls back to an unmasked score here.
        if int(query_static.sum()) < 576:
            query_static = None

    _, registered_images, _ = read_model(str(environment / "colmap"), ext="")
    image_dir = environment / "images"
    scores: list[tuple[float, float, float, str]] = []
    for image in registered_images.values():
        name = Path(image.name).name
        reference_path = image_dir / name
        if not reference_path.is_file():
            continue
        reference = cv2.imread(str(reference_path), cv2.IMREAD_REDUCED_COLOR_4)
        if reference is None:
            continue
        color, structure = compact_visual_features(reference)
        if query_static is None:
            color_error = float(np.mean(np.abs(query_color - color)))
            structure_error = float(np.mean(np.abs(query_structure - structure)))
        else:
            color_error = float(np.mean(np.abs(query_color[query_static] - color[query_static])))
            structure_error = float(
                np.mean(np.abs(query_structure[query_static] - structure[query_static]))
            )
        score = color_error + 0.60 * structure_error
        scores.append((score, color_error, structure_error, name))

    if len(scores) < 2:
        raise RuntimeError(f"Only {len(scores)} registered images were readable from {image_dir}")
    scores.sort(key=lambda item: item[0])
    selected = scores[:count]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["rank\tscore\tcolour\tstructure\timage"]
    lines.extend(
        f"{rank}\t{score:.8f}\t{color:.8f}\t{structure:.8f}\t{name}"
        for rank, (score, color, structure, name) in enumerate(selected, start=1)
    )
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Retrieved {len(selected)} candidates: {output_path}")


def preview_expression(
    camera_info_path: Path,
    output_path: Path,
    width: int,
    height: int,
    hide_collections: list[str],
) -> str:
    script = f"""import bpy, json
from mathutils import Matrix

info = json.loads(open({str(camera_info_path)!r}).read())
name = info['camera_name']
camera = bpy.data.objects.get(name)
if camera is None:
    data = bpy.data.cameras.new(name)
    collection = bpy.data.collections.get('PredictedCameras')
    if collection is None:
        collection = bpy.data.collections.new('PredictedCameras')
        bpy.context.scene.collection.children.link(collection)
    camera = bpy.data.objects.new(name, data)
    collection.objects.link(camera)
if camera.type != 'CAMERA':
    raise RuntimeError('Target object is not a camera: ' + name)
settings = info['blender']
camera.data.lens = float(settings['lens_mm'])
camera.data.sensor_width = float(settings['sensor_width_mm'])
camera.data.sensor_fit = settings.get('sensor_fit', 'HORIZONTAL')
camera.data.clip_start = float(settings.get('clip_start', 0.01))
camera.data.clip_end = float(settings.get('clip_end', 1000.0))
camera.matrix_world = Matrix(settings['matrix_world'])
scene = bpy.context.scene
scene.camera = camera
visibility = []
for collection in bpy.data.collections:
    if collection.name in {hide_collections!r}:
        visibility.append((collection, collection.hide_viewport, collection.hide_render))
        collection.hide_viewport = True
        collection.hide_render = True
area = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
region = next(region for region in area.regions if region.type == 'WINDOW')
space = area.spaces.active
old_shading = {{
    'type': space.shading.type,
    'light': space.shading.light,
    'color_type': space.shading.color_type,
    'show_shadows': space.shading.show_shadows,
    'show_cavity': space.shading.show_cavity,
    'show_specular_highlight': space.shading.show_specular_highlight,
}}
space.shading.type = 'SOLID'
space.shading.light = 'FLAT'
space.shading.color_type = 'TEXTURE'
space.shading.show_shadows = False
space.shading.show_cavity = False
space.shading.show_specular_highlight = False
scene.render.resolution_x = {width}
scene.render.resolution_y = {height}
scene.render.resolution_percentage = 100
try:
    with bpy.context.temp_override(
        window=bpy.context.window,
        area=area,
        region=region,
        scene=scene,
        space_data=space,
    ):
        bpy.ops.render.opengl(write_still=False, view_context=False)
    source = bpy.data.images.get('Render Result')
    if source is None or source.size[0] == 0 or source.size[1] == 0:
        raise RuntimeError('Viewport preview produced an empty Render Result')
    output = bpy.data.images.new('__camera_only_preview', source.size[0], source.size[1], alpha=True)
    output.pixels.foreach_set(source.pixels[:])
    output.filepath_raw = {str(output_path)!r}
    output.file_format = 'PNG'
    output.save()
    bpy.data.images.remove(output)
finally:
    for collection, hide_viewport, hide_render in visibility:
        collection.hide_viewport = hide_viewport
        collection.hide_render = hide_render
    for key, value in old_shading.items():
        setattr(space.shading, key, value)
print('preview_saved=' + {str(output_path)!r})
bpy.ops.wm.quit_blender()"""
    return f"exec({script!r})"


def main() -> None:
    args = parse_args()
    pipeline_started = time.perf_counter()
    stage_seconds: dict[str, float] = {}
    query_image = require_file(args.query_image, "Query image")
    blender_input = require_file(args.blender_input, "Blender input")
    environment = args.environment_dir.resolve()
    require_file(environment / "colmap/images.bin", "COLMAP images.bin")
    require_file(environment / "colmap/points3D.bin", "COLMAP points3D.bin")
    require_file(environment / "h5/descriptors.h5", "Reference DISK descriptors")
    require_file(DISK_DETECT, "DISK detector")
    disk_checkpoint = require_file(args.disk_checkpoint, "DISK checkpoint")
    require_file(LOCALIZER, "Localization script")
    require_file(CAMERA_EXPORTER, "Blender camera exporter")
    require_file(CAMERA_IMPORTER, "Blender camera importer")
    require_file(VIEWPORT_CAPTURE, "Viewport capture script")
    require_file(GLOBAL_RETRIEVER, "Global DISK retrieval script")
    require_file(args.global_index, "Global DISK retrieval index")
    require_file(args.blender_executable, "Blender executable")
    if args.query_mask is None and args.person_mask == "auto":
        require_file(SAM3_PERSON_MASKER, "SAM3 person-mask script")
        require_file(args.sam3_checkpoint, "SAM3 checkpoint")
        if not (args.sam3_root / "sam3").is_dir():
            raise FileNotFoundError(f"SAM3 source package does not exist: {args.sam3_root}")

    run_name = f"{query_image.stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = (args.output_dir or (REPO_ROOT / "outputs/Friends-monica_apartment/new_image_runs" / run_name)).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --overwrite or choose another path.")
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_info_path = output_dir / "camera_info.json"
    camera_import_path = output_dir / "import_predicted_camera.py"

    supplied_mask = args.query_mask.resolve() if args.query_mask else None
    staged_image, image_list = measure(
        "stage_query_image",
        stage_seconds,
        lambda: stage_query_image(
            query_image,
            output_dir / "query_images",
            supplied_mask,
        ),
    )
    active_mask = supplied_mask
    mask_source = "user" if supplied_mask is not None else None
    person_mask_metadata: Path | None = None
    if active_mask is None and args.person_mask == "auto":
        active_mask = output_dir / "person_mask_sam3.png"
        person_mask_metadata = output_dir / "person_mask_sam3.json"
        measure(
            "sam3_person_mask",
            stage_seconds,
            lambda: run(
                [
                    str(args.python_executable),
                    str(SAM3_PERSON_MASKER),
                    "--image",
                    str(staged_image),
                    "--output",
                    str(active_mask),
                    "--metadata",
                    str(person_mask_metadata),
                    "--overlay",
                    str(output_dir / "person_mask_sam3_overlay.jpg"),
                    "--checkpoint",
                    str(args.sam3_checkpoint),
                    "--sam3-root",
                    str(args.sam3_root),
                    "--confidence-threshold",
                    str(args.sam3_confidence_threshold),
                    "--dilate-pixels",
                    str(args.sam3_dilate_pixels),
                ]
            ),
        )
        mask_source = "sam3"
    candidates = output_dir / "retrieval_candidates.tsv"
    disk_height, disk_width = disk_input_size(staged_image)
    measure(
        "coarse_appearance_retrieval",
        stage_seconds,
        lambda: build_retrieval_candidates(
            staged_image, active_mask, environment, candidates, args.retrieval_pool
        ),
    )

    query_h5 = output_dir / "query_h5"
    measure(
        "disk_feature_extraction",
        stage_seconds,
        lambda: run(
            [
            str(args.python_executable),
            str(DISK_DETECT),
            "--model_path",
            str(disk_checkpoint),
            "--image-extension",
            staged_image.suffix.lstrip("."),
            "--height",
            str(disk_height),
            "--width",
            str(disk_width),
            "--n",
            str(args.max_features),
            "--image_list_path",
            str(image_list),
            str(query_h5),
            str(staged_image.parent),
            ],
            cwd=DISK_DETECT.parent,
        ),
    )

    def localize_command(candidate_list: Path, top_k: int, ratio: float, output: Path) -> list[str]:
        command = [
            str(args.python_executable),
            str(LOCALIZER),
            "--query-h5",
            str(query_h5),
            "--query-key",
            "query",
            "--query-image",
            str(staged_image),
            "--candidate-list",
            str(candidate_list),
            "--environment-dir",
            str(environment),
            "--top-k",
            str(top_k),
            "--ratio",
            str(ratio),
            "--reprojection-error",
            str(args.reprojection_error),
            "--focal-multipliers",
            args.focal_multipliers,
            "--output",
            str(output),
        ]
        if active_mask is not None:
            command.extend(("--query-mask", str(active_mask)))
        if args.disable_reference_person_mask:
            command.append("--disable-reference-person-mask")
        if args.focal_pixels.strip():
            command.extend(("--focal-pixels", args.focal_pixels))
        for image_name in args.exclude_reference:
            command.extend(("--exclude-reference", image_name))
        return command

    pose_path = output_dir / "localization_colmap.json"
    coarse_pose_path = output_dir / "localization_colmap_coarse.json"
    localization_strategy = "coarse_appearance"
    candidate_path_used = candidates
    selected_pose_source = coarse_pose_path
    coarse_ok = measure(
        "coarse_localization",
        stage_seconds,
        lambda: try_run(
            localize_command(candidates, args.top_k, args.ratio, coarse_pose_path)
        ),
    )
    coarse_quality = None
    if coarse_ok and coarse_pose_path.is_file():
        coarse_quality = assess_pose_quality(json.loads(coarse_pose_path.read_text()))
    if coarse_quality is not None and coarse_quality["status"] == "accepted":
        shutil.copy2(coarse_pose_path, pose_path)
    else:
        if coarse_quality is not None:
            print(
                "Coarse PnP completed but failed the pose-quality gate; "
                "running global DISK fallback.",
                flush=True,
            )
        global_candidates = output_dir / "disk_global_candidates.tsv"
        measure(
            "global_disk_retrieval",
            stage_seconds,
            lambda: run(
                [
                str(args.python_executable),
                str(GLOBAL_RETRIEVER),
                "--query-h5",
                str(query_h5),
                "--query-key",
                "query",
                "--query-mask",
                str(active_mask),
                "--index",
                str(args.global_index),
                "--environment-dir",
                str(environment),
                "--top-k",
                str(args.global_top_k),
                "--output",
                str(global_candidates),
                ] if active_mask is not None else [
                    str(args.python_executable),
                    str(GLOBAL_RETRIEVER),
                    "--query-h5",
                    str(query_h5),
                    "--query-key",
                    "query",
                    "--index",
                    str(args.global_index),
                    "--environment-dir",
                    str(environment),
                    "--top-k",
                    str(args.global_top_k),
                    "--output",
                    str(global_candidates),
                ]
            ),
        )
        global_pose_path = output_dir / "localization_colmap_global.json"
        measure(
            "global_disk_localization",
            stage_seconds,
            lambda: run(
                localize_command(
                    global_candidates,
                    args.global_top_k,
                    args.global_ratio,
                    global_pose_path,
                )
            ),
        )
        candidates_to_rank = [global_pose_path]
        if coarse_ok and coarse_pose_path.is_file():
            candidates_to_rank.append(coarse_pose_path)

        def pose_rank(path: Path) -> tuple[int, int, float]:
            pose = json.loads(path.read_text())
            quality = assess_pose_quality(pose)
            status_rank = {"rejected": 0, "review": 1, "accepted": 2}[quality["status"]]
            return (
                status_rank,
                int(quality["pnp_inlier_count"]),
                -float(quality["pnp_rmse_pixels"]),
            )

        selected_pose = max(candidates_to_rank, key=pose_rank)
        shutil.copy2(selected_pose, pose_path)
        selected_pose_source = selected_pose
        localization_strategy = "global_disk_fallback"
        candidate_path_used = global_candidates

    measure(
        "blender_camera_export",
        stage_seconds,
        lambda: run(
            [
            str(args.blender_executable),
            "--background",
            str(blender_input),
            "--python",
            str(CAMERA_EXPORTER),
            "--",
            "--localization",
            str(pose_path),
            "--point-cloud-object",
            args.point_cloud_object,
            "--camera-name",
            args.camera_name,
            "--output",
            str(camera_info_path),
            ],
            environment=blender_environment(),
        ),
    )
    shutil.copy2(CAMERA_IMPORTER, camera_import_path)

    preview_path = output_dir / "predicted_camera_preview.png" if args.preview else None
    if preview_path is not None:
        preview_command = [
            "env",
            f"DISPLAY={args.viewport_display}",
            "GHOST_BACKEND=x11",
            "__GLX_VENDOR_LIBRARY_NAME=nvidia",
            str(args.blender_executable),
            str(blender_input),
            "--python",
            str(VIEWPORT_CAPTURE),
            "--",
            "--camera-info",
            str(camera_info_path),
            "--output",
            str(preview_path),
            "--metadata",
            str(output_dir / "viewport_capture.json"),
            "--camera-name",
            args.camera_name,
        ]
        for collection_name in args.hide_collection:
            preview_command.extend(("--hide-collection", collection_name))
        measure(
            "blender_viewport_preview",
            stage_seconds,
            lambda: run(preview_command, environment=blender_environment()),
        )
        if not preview_path.is_file():
            raise RuntimeError(f"Blender did not create the preview image: {preview_path}")

    pose = json.loads(pose_path.read_text())
    quality = assess_pose_quality(pose)
    manifest = {
        "query_image": str(query_image),
        "staged_query_image": str(staged_image),
        "environment_dir": str(environment),
        "retrieval_candidates": str(candidate_path_used),
        "coarse_retrieval_candidates": str(candidates),
        "localization_strategy": localization_strategy,
        "selected_pose_source": str(selected_pose_source),
        "query_h5": str(query_h5),
        "dynamic_mask": str(active_mask) if active_mask is not None else None,
        "dynamic_mask_source": mask_source,
        "dynamic_mask_metadata": str(person_mask_metadata) if person_mask_metadata else None,
        "localization": str(pose_path),
        "blender_input": str(blender_input),
        "camera_info": str(camera_info_path),
        "camera_import_script": str(camera_import_path),
        "preview": str(preview_path) if preview_path else None,
        "camera_name": args.camera_name,
        "focal_pixels": pose["pnp"]["focal"],
        "camera_center_colmap": pose["camera_center"],
        "pnp_inlier_count": pose["pnp"]["inlier_count"],
        "pnp_rmse_pixels": pose["pnp"]["inlier_rmse_pixels"],
        "pose_quality": quality,
        "stage_durations_seconds": stage_seconds,
        "total_duration_seconds": round(time.perf_counter() - pipeline_started, 3),
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
