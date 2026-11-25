"""Build a reusable DISK/SIFT feature index in the checked Omega world."""
import argparse
import json
from pathlib import Path
import sys
import time

from omega_feature_index import signature, save_index, load_index


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reconstruction", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--disk-root", type=Path, default=Path(__file__).resolve().parents[1] / "sitcoms3D/external/disk")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.disk_root.resolve()))
    from omega_clip_localizer import OmegaPnPLocalizer
    started = time.perf_counter()
    metadata = signature(args.reconstruction, args.images, args.checkpoint)
    if args.output.exists():
        sets, _ = load_index(args.output, metadata)
        reused = True
    else:
        model = OmegaPnPLocalizer(args.reconstruction, args.images, disk_checkpoint=args.checkpoint)
        sets = model.reference_sets
        if metadata != signature(args.reconstruction, args.images, args.checkpoint):
            raise RuntimeError("Source files changed while building the Omega feature index")
        save_index(args.output, sets, metadata)
        sets, _ = load_index(args.output, metadata)
        reused = False
    report = {
        "path": str(args.output.resolve()),
        "coordinate_system": "omega_ba_world",
        "reused": reused,
        "seconds": time.perf_counter() - started,
        "backends": {key: {"reference_images": len(refs), "features_with_3d": sum(len(row[1]) for row in refs)}
                     for key, refs in sets.items()},
        "inputs": metadata,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "inputs"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
