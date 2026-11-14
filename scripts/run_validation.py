"""Run the full challenge and two independent real-video sequences.

First run scripts/download_validation_samples.py. The default run processes the
entire challenge, the entire bus clip (150 frames), and a disclosed first-300-
frame highway excerpt. --only-real avoids repeating a completed challenge run.
Output video, JSONL, metrics and summary stay in scripts/validation_runs/.
Recognition accuracy requires separate manual annotations and evaluate.py;
synthetic geometry belongs in separate calibrated tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def validation_plan(samples_dir: Path, real_frames: int, only_real: bool = False) -> list[dict]:
    if real_frames < 1:
        raise ValueError("real_frames must be positive")
    runs = []
    if not only_real:
        runs.append({"name": "challenge_full", "input": PROJECT_ROOT / "data/challenge_video.mp4",
                     "limit": None, "provenance": None})
    for name in ("bus", "highway"):
        runs.append({"name": name, "input": samples_dir / f"{name}.mp4",
                     "limit": real_frames, "provenance": samples_dir / f"{name}.provenance.json"})
    return runs


def run_validation(config: Path, samples_dir: Path, output_dir: Path,
                   real_frames: int = 300, only_real: bool = False, pipeline=None,
                   source_counter=None) -> dict:
    if pipeline is None:
        from src.cv3dbox.pipeline import run_pipeline
        pipeline = run_pipeline
    plan = validation_plan(samples_dir, real_frames, only_real)
    if not only_real:
        if source_counter is None:
            from scripts.verify_video import source_decode_count
            source_counter = source_decode_count
        import yaml
        settings = yaml.safe_load(config.read_text(encoding="utf-8"))
        if settings.get("input", {}).get("max_frames") is not None:
            raise ValueError("Full challenge requires input.max_frames=null; pipeline None uses the configured limit")
    for run in plan:
        if not run["input"].is_file():
            raise FileNotFoundError(f"Missing {run['input']}; run download_validation_samples.py first")
        if run["provenance"] is not None and not run["provenance"].is_file():
            raise FileNotFoundError(f"Missing sample provenance: {run['provenance']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"runs": [], "scope": "Runtime/output validation; no unannotated accuracy claims",
              "challenge_included": not only_real, "requested_real_frame_limit": real_frames}
    for run in plan:
        capture = cv2.VideoCapture(str(run["input"]))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot open {run['input']}")
            source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if source_frames < 1:
                raise ValueError(f"Cannot determine frame count: {run['input']}")
        finally:
            capture.release()
        # Uncapped challenge decode must not stop at an estimated container count.
        limit = None if run["limit"] is None else min(source_frames, run["limit"])
        independent_count = None
        verification_seconds = None
        if limit is None:
            verification_started = time.perf_counter()
            independent_count = source_counter(run["input"])
            verification_seconds = time.perf_counter() - verification_started
            if not isinstance(independent_count, int) or independent_count < 1:
                raise ValueError("Independent source decode returned an invalid frame count")
        output = output_dir / f"validation_{run['name']}.mp4"
        started = time.perf_counter()
        metrics = pipeline(config_path=config, input_video=run["input"], output_video=output,
                           max_frames=limit, snapshot_path=output.with_suffix(".jpg"))
        wall_seconds = time.perf_counter() - started
        processed = metrics["frames_processed"]
        if limit is not None and processed != limit:
            raise ValueError(f"{run['name']}: processed {metrics['frames_processed']} of {limit} frames")
        if independent_count is not None and processed != independent_count:
            raise ValueError(f"{run['name']}: processed {processed}; independent FFmpeg source "
                             f"decode counted {independent_count} readable frames")
        metrics.update({"validation_name": run["name"], "source_frame_range_inclusive": [0, processed - 1],
                        "excerpt": limit is not None and limit < source_frames,
                        "requested_frames": limit, "decoded_frames": processed,
                        "metadata_source_frame_count": source_frames,
                        "decoded_count_matches_metadata": processed == source_frames,
                        "independent_source_decoded_frames": independent_count,
                        "decoded_count_matches_independent_source": processed == independent_count
                                                                     if independent_count is not None else None,
                        "source_verification_seconds": verification_seconds,
                        "source_verification": "scripts.verify_video.source_decode_count; "
                                               "FFmpeg passthrough decode, not container frame metadata"
                                               if independent_count is not None else None,
                        "frame_count_note": "Uncapped sequential pipeline decode count equals an independent "
                                            "FFmpeg passthrough source decode count. Container metadata "
                                            "may disagree; source verification is outside pipeline timing."
                                            if limit is None else "Explicit excerpt cap, or entire sample if shorter.",
                        "source_provenance": str(run["provenance"].resolve()) if run["provenance"] else None,
                        "prediction_path": str(output.with_suffix(".predictions.jsonl").resolve()),
                        "rendered_video_path": str(output.resolve()),
                        "pipeline_call_wall_seconds": wall_seconds,
                        "wall_timing_scope": "Entire pipeline call including calibration, model "
                                             "initialization and finalization; excludes downloads and preflight.",
                        "annotation_status": "No ground-truth accuracy computed in this runner"})
        report["runs"].append(metrics)
        # Preserve completed runs if a later run fails.
        (output_dir / "validation_report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"{run['name']}: decoded {processed}; metadata {source_frames}; "
              f"cap {limit}; {metrics['detector_backend']}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    parser.add_argument("--samples-dir", type=Path, default=PROJECT_ROOT / "scripts/validation_samples")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "scripts/validation_runs")
    parser.add_argument("--real-frames", type=int, default=300)
    parser.add_argument("--only-real", action="store_true")
    args = parser.parse_args()
    try:
        run_validation(args.config, args.samples_dir, args.output_dir, args.real_frames, args.only_real)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Validation report: {args.output_dir / 'validation_report.json'}")


if __name__ == "__main__":
    main()
