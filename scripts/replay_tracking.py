"""Replay saved detections to compare tracker settings without rerunning YOLO.

Timing from this tool is not end-to-end pipeline performance. Replayed outputs
have no cuboids; rerun run_demo.py to render and measure a changed configuration.
"""
import argparse
import json
from pathlib import Path
import sys
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cv3dbox.config import load_config
from src.cv3dbox.detectors import Detection
from src.cv3dbox.tracker import MultiObjectTracker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tracker = MultiObjectTracker(load_config(args.config)["tracker"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as output:
        for line in source:
            record = json.loads(line)
            detections = [Detection(tuple(d["bbox"]), d["score"], d["label"]) for d in record["detections"]]
            record["tracks"] = [asdict(track) for track in tracker.update(detections)]
            output.write(json.dumps(record) + "\n")
