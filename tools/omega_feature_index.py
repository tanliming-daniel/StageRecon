"""Versioned, atomic feature-to-Omega-world indexes without pickle."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

SCHEMA_VERSION = 1
EXTRACTOR_VERSION = "sift1400_disk4096_rgb_768x432_static_erode5_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(reconstruction, images, checkpoint):
    reconstruction, images = Path(reconstruction), Path(images)
    with np.load(reconstruction / "predictions.npz", allow_pickle=False) as data:
        names = list(map(str, data["names"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "coordinate_system": "omega_ba_world",
        "prediction_sha256": sha256(reconstruction / "predictions.npz"),
        "checkpoint_sha256": sha256(checkpoint),
        "reference_image_sha256": {name: sha256(images / name) for name in names},
    }


def save_index(path, reference_sets, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"metadata": np.asarray(json.dumps(metadata, sort_keys=True))}
    for backend, refs in reference_sets.items():
        arrays[backend + "_names"] = np.asarray([row[0] for row in refs])
        arrays[backend + "_offsets"] = np.r_[0, np.cumsum([len(row[1]) for row in refs])]
        for field, column in (("pixels", 1), ("descriptors", 2), ("xyz", 3)):
            arrays[backend + "_" + field] = np.concatenate([row[column] for row in refs])
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_index(path, expected):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata != expected:
            raise ValueError("Omega feature index input signature differs; rebuild the index")
        sets = {}
        for backend in ("sift", "disk"):
            names = data[backend + "_names"]
            offsets = data[backend + "_offsets"]
            xy = data[backend + "_pixels"]
            descriptors = data[backend + "_descriptors"]
            xyz = data[backend + "_xyz"]
            count = len(xy)
            if (len(offsets) != len(names) + 1 or offsets.dtype.kind not in "iu"
                    or offsets[0] != 0 or offsets[-1] != count
                    or np.any(np.diff(offsets) < 16)
                    or xy.shape != (count, 2) or xyz.shape != (count, 3)
                    or descriptors.shape != (count, 128)
                    or not all(np.isfinite(a).all() for a in (xy, descriptors, xyz))
                    or len(set(map(str, names))) != len(names)):
                raise ValueError(f"Invalid {backend} feature index arrays")
            sets[backend] = [
                (str(name), xy[a:b].copy(), descriptors[a:b].copy(), xyz[a:b].copy())
                for name, a, b in zip(names, offsets[:-1], offsets[1:])
            ]
        return sets, metadata
