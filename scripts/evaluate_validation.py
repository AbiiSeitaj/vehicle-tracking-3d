"""Evaluate completed challenge/bus/highway outputs against independent labels.

Run after main inference finishes:
python scripts/evaluate_validation.py
Defaults: outputs/challenge_3d_boxes.predictions.jsonl and
outputs/validation/validation_bus.predictions.jsonl, validation_highway.predictions.jsonl.
Evaluation reports default to scripts/validation_runs/evaluation/. Use
--output-dir to place reports elsewhere. Each run uses its own annotation
coordinate size and identities; unrelated sequences are never concatenated.
No pipeline inference or model initialization is performed by this tool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def evaluate_outputs(challenge: Path, real_dir: Path, annotations_dir: Path, output_dir: Path) -> dict:
    runs = [("challenge", challenge),
            ("bus", real_dir / "validation_bus.predictions.jsonl"),
            ("highway", real_dir / "validation_highway.predictions.jsonl")]
    for name, predictions in runs:
        if not predictions.is_file():
            raise FileNotFoundError(f"{name}: missing {predictions}")
        metrics = predictions.with_name(predictions.name.removesuffix(".predictions.jsonl") + ".metrics.json")
        if not metrics.is_file():
            raise FileNotFoundError(f"{name}: missing completion metrics {metrics}")
        if not (annotations_dir / f"{name}_manual_annotations.json").is_file():
            raise FileNotFoundError(f"{name}: missing manual annotations")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"runs": [], "scope": "Per-sequence sparse manual annotation evaluation; "
               "do not average unrelated tracking identities or claim dense benchmark accuracy"}
    for name, predictions in runs:
        output = output_dir / f"{name}.evaluation.json"
        command = [sys.executable, str(PROJECT_ROOT / "scripts/evaluate.py"),
                   "--predictions", str(predictions),
                   "--annotations", str(annotations_dir / f"{name}_manual_annotations.json"),
                   "--output", str(output), "--ignore-labels"]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise ValueError(f"{name}: evaluation failed: {result.stderr.strip()}")
        report = json.loads(output.read_text(encoding="utf-8"))
        summary["runs"].append({"name": name, "report_path": str(output.resolve()),
                                "scope": report["scope"], "detections": report["detections"],
                                "tracks": report["tracks"], "identity": report["identity"],
                                "limitations": report["limitations"],
                                "annotation_provenance": report["annotation_provenance"]})
        print(result.stdout.strip(), flush=True)
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--challenge-predictions", type=Path,
                        default=PROJECT_ROOT / "outputs/challenge_3d_boxes.predictions.jsonl")
    parser.add_argument("--real-output-dir", type=Path, default=PROJECT_ROOT / "outputs/validation")
    parser.add_argument("--annotations-dir", type=Path, default=PROJECT_ROOT / "scripts/validation_samples")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "scripts/validation_runs/evaluation")
    args = parser.parse_args()
    try:
        evaluate_outputs(args.challenge_predictions, args.real_output_dir, args.annotations_dir, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
