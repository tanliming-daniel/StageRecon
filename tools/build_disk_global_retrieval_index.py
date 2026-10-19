#!/usr/bin/env python3
"""Build a compact global-retrieval index from registered DISK descriptors."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def pooled_descriptor(descriptors: np.ndarray) -> np.ndarray:
    vector = descriptors.astype(np.float32, copy=False).mean(axis=0)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def main() -> None:
    args = parse_args()
    names: list[str] = []
    vectors: list[np.ndarray] = []
    with h5py.File(args.descriptors, "r") as descriptors_h5:
        keys = sorted(descriptors_h5.keys())
        for index, name in enumerate(keys, start=1):
            vectors.append(pooled_descriptor(np.asarray(descriptors_h5[name])))
            names.append(name)
            if index % 250 == 0 or index == len(keys):
                print(f"Indexed {index}/{len(keys)} frames", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, names=np.asarray(names), vectors=np.stack(vectors))
    print(args.output)


if __name__ == "__main__":
    main()
