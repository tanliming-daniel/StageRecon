#!/usr/bin/env python3
"""Export visible Blender meshes for the CPU camera-preview renderer.

The script is read-only with respect to the opened blend file.  It produces a
temporary compressed mesh cache with per-triangle material colours.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hide-collection", action="append", default=[])
    parser.add_argument("--texture-size", type=int, default=128)
    return parser.parse_args(argv)


def collection_objects(collection) -> set[str]:
    names = {obj.name for obj in collection.objects}
    for child in collection.children:
        names.update(collection_objects(child))
    return names


def image_average_colour(image) -> tuple[float, float, float] | None:
    if image is None or not image.has_data or image.size[0] == 0 or image.size[1] == 0:
        return None
    width, height = image.size
    channels = image.channels
    pixels = image.pixels
    samples: list[tuple[float, float, float]] = []
    for row in range(8):
        for column in range(8):
            x = int((column + 0.5) * width / 8)
            y = int((row + 0.5) * height / 8)
            index = (min(y, height - 1) * width + min(x, width - 1)) * channels
            samples.append((pixels[index], pixels[index + 1], pixels[index + 2]))
    return tuple(float(np.clip(np.mean(samples, axis=0)[index], 0.0, 1.0)) for index in range(3))


def material_colour(mesh, polygon_index: int, cache: dict[str, tuple[float, float, float]]) -> tuple[float, float, float]:
    slot_index = mesh.polygons[polygon_index].material_index
    if slot_index >= len(mesh.materials) or mesh.materials[slot_index] is None:
        return (0.62, 0.62, 0.62)
    material = mesh.materials[slot_index]
    key = material.name_full
    if key in cache:
        return cache[key]
    colour = tuple(float(min(1.0, max(0.0, component))) for component in material.diffuse_color[:3])
    if material.use_nodes:
        image_nodes = [node for node in material.node_tree.nodes if node.type == "TEX_IMAGE"]
        for node in image_nodes:
            sampled = image_average_colour(node.image)
            if sampled is not None:
                colour = sampled
                break
    cache[key] = colour
    return colour


def main() -> None:
    args = parse_args()
    excluded: set[str] = set()
    for name in args.hide_collection:
        collection = bpy.data.collections.get(name)
        if collection is not None:
            excluded.update(collection_objects(collection))

    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    colours: list[tuple[float, float, float]] = []
    material_cache: dict[str, tuple[float, float, float]] = {}
    object_count = 0
    offset = 0

    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.name in excluded or obj.hide_render:
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            if not mesh.loop_triangles:
                continue
            coords = np.empty((len(mesh.vertices), 3), dtype=np.float32)
            mesh.vertices.foreach_get("co", coords.ravel())
            world = np.asarray(evaluated.matrix_world, dtype=np.float32)
            coords = coords @ world[:3, :3].T + world[:3, 3]
            local_faces = np.asarray(
                [triangle.vertices[:] for triangle in mesh.loop_triangles], dtype=np.int32
            )
            vertices.append(coords)
            triangles.append(local_faces + offset)
            colours.extend(
                material_colour(mesh, triangle.polygon_index, material_cache)
                for triangle in mesh.loop_triangles
            )
            offset += len(coords)
            object_count += 1
        finally:
            evaluated.to_mesh_clear()

    if not vertices:
        raise RuntimeError("No visible mesh geometry was available for preview export")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        vertices=np.concatenate(vertices, axis=0),
        triangles=np.concatenate(triangles, axis=0),
        colours=np.asarray(colours, dtype=np.float32),
    )
    print(
        f"preview_mesh={output} objects={object_count} vertices={offset} triangles={len(colours)}"
    )


if __name__ == "__main__":
    main()
