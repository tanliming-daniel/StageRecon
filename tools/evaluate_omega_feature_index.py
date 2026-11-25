"""Compare baseline and indexed localization on identical SAM3-masked clips."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_clip_render_data_batch import read_line, send


def evaluate_cached(args, cases):
    """Repeat localization with deterministic RANSAC and previously computed masks."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "sitcoms3D/external/disk"))
    from omega_clip_localizer import OmegaPnPLocalizer
    models = {
        "sift": OmegaPnPLocalizer(args.reconstruction, args.images),
        "indexed": OmegaPnPLocalizer(args.reconstruction, args.images,
                                    disk_checkpoint=args.checkpoint, feature_index=args.index),
    }
    records = []
    for case in cases:
        relative = case["clip"]
        clip = Path(relative)
        output = args.output / "clips" / clip.parts[0] / clip.stem
        metadata = json.loads((output / "frames.json").read_text())
        record = {"clip": relative, "split": case.get("split"), "endpoints": {}}
        for backend, model in models.items():
            endpoints = []
            for index, label in ((0, "anchor"), (int(metadata["frame_count"])-1, "tail")):
                try:
                    pose = model.endpoint_for_clip(
                        output / "frames" / f"{index:06d}.jpg",
                        output / "masks/union" / f"{index:06d}.png",
                        output / f"cached_{backend}_{label}.json",
                        str(args.clips_root / clip), index, int(metadata["frame_count"]), "estimated", [])
                    endpoints.append({"passed": True, "quality": pose["quality"]})
                except RuntimeError as error:
                    endpoints.append({"passed": False, "error": str(error)})
            record["endpoints"][backend] = endpoints
            record[backend + "_pnp_passed"] = all(e["passed"] for e in endpoints)
        records.append(record)
        summary = {
            "tested": len(records), "planned": len(cases),
            "metric": "Both endpoints pass PnP; visual usability is evaluated separately",
            "ransac_seed": 52,
            "sift_pnp_passed": sum(r["sift_pnp_passed"] for r in records),
            "indexed_pnp_passed": sum(r["indexed_pnp_passed"] for r in records),
            "recovered_numerically": [r["clip"] for r in records if r["indexed_pnp_passed"] and not r["sift_pnp_passed"]],
            "regressions": [r["clip"] for r in records if r["sift_pnp_passed"] and not r["indexed_pnp_passed"]],
        }
        dest = args.output / "cached_report.json"
        temporary = dest.with_suffix(".tmp")
        temporary.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
        temporary.replace(dest)
        print(json.dumps(summary), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("cases", "clips-root", "reconstruction", "images", "index", "checkpoint", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--reuse-observations", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    cases = json.loads(args.cases.read_text())
    if args.reuse_observations:
        evaluate_cached(args, cases)
        return
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    runtime = args.output / "runtime"
    runtime.mkdir(exist_ok=True)
    env["TMPDIR"] = str(runtime.resolve())
    command = [
        "/root/userdata/300workspace/sm3r/bin/python", "-u", str(root / "tools/clip_data_stage1_worker.py"),
        "--environment-dir", str(root / "sitcoms3D/data/sparse_reconstruction_and_nerf_data/Friends-monica_apartment"),
        "--anchor-source", "estimated", "--sam3-root", "/root/userdata/300workspace/sam3",
        "--sam3-checkpoint", "/root/userdata/300workspace/ckpt/sam3/sam3.pt",
        "--omega-reconstruction", str(args.reconstruction.resolve()),
        "--omega-images", str(args.images.resolve()),
        "--omega-feature-index", str(args.index.resolve()),
        "--disk-checkpoint", str(args.checkpoint.resolve()), "--keep-failed-output",
    ]
    report_path = args.output / "report.json"
    records = json.loads(report_path.read_text())["records"] if report_path.exists() else []
    finished = {r["clip"] for r in records}
    started = time.perf_counter()
    with (args.output / "worker.log").open("a") as log:
        worker = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=log, text=True, bufsize=1, env=env, cwd=root)
        try:
            if read_line(worker, "pilot").get("type") != "ready":
                raise RuntimeError("Localization worker did not initialize")
            from omega_clip_localizer import OmegaPnPLocalizer
            baseline = OmegaPnPLocalizer(args.reconstruction, args.images)
            for case in cases:
                relative = case if isinstance(case, str) else case["clip"]
                if relative in finished:
                    continue
                clip = Path(relative)
                output = args.output / "clips" / clip.parts[0] / clip.stem
                job = {"clip_key": relative, "video": str((args.clips_root / clip).resolve()),
                       "output_dir": str(output.resolve())}
                send(worker, job)
                result = read_line(worker, "pilot")
                if result.get("clip_key") != relative:
                    raise RuntimeError("Worker returned a different clip")
                metadata = json.loads((output / "frames.json").read_text())
                failures = []
                for index, label in ((0, "anchor"), (int(metadata["frame_count"]) - 1, "tail_anchor")):
                    try:
                        baseline.endpoint_for_clip(
                            output / "frames" / f"{index:06d}.jpg",
                            output / "masks/union" / f"{index:06d}.png",
                            output / ("baseline_" + label + ".json"),
                            job["video"], index, int(metadata["frame_count"]), "estimated", [])
                    except RuntimeError as error:
                        failures.append(str(error))
                records.append({"clip": relative, "split": case.get("split", "unknown"),
                                "baseline_passed": not failures, "baseline_errors": failures,
                                "indexed_passed": result.get("status") == "ok", "indexed_result": result})
                summary = {
                    "tested": len(records), "planned": len(cases),
                    "baseline_passed": sum(r["baseline_passed"] for r in records),
                    "indexed_passed": sum(r["indexed_passed"] for r in records),
                    "recovered": [r["clip"] for r in records if r["indexed_passed"] and not r["baseline_passed"]],
                    "regressions": [r["clip"] for r in records if r["baseline_passed"] and not r["indexed_passed"]],
                }
                temporary = report_path.with_suffix(".tmp")
                temporary.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
                temporary.replace(report_path)
                print(json.dumps({**summary, "elapsed_seconds": time.perf_counter() - started}), flush=True)
        finally:
            if worker.stdin:
                worker.stdin.close()
            try:
                worker.wait(timeout=15)
            except subprocess.TimeoutExpired:
                worker.terminate()
                worker.wait()


if __name__ == "__main__":
    main()
