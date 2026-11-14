"""Summarize cuboid coverage and motion-sensitive stability diagnostics."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np


def summarize(path):
    instances = valid = 0
    reasons = Counter()
    history = {}
    accelerations = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            index = record["frame_index"]
            for track in record["tracks"]:
                instances += 1
                if not track.get("valid", False):
                    reasons[track.get("diagnostics", {}).get("rejection_reason", "unspecified")] += 1
                    continue
                valid += 1
                box = np.asarray(track["bbox"], float)
                normalized = (np.asarray(track["corners"], float) - box[:2]) / (box[2:] - box[:2])
                samples = history.get(track["track_id"], [])
                if len(samples) == 2 and samples[0][0] == index - 2 and samples[1][0] == index - 1:
                    acceleration = normalized - 2 * samples[1][1] + samples[0][1]
                    accelerations.append(float(np.sqrt(np.mean(acceleration ** 2))))
                history[track["track_id"]] = (samples + [(index, normalized)])[-2:]
    return {"track_frame_instances": instances, "valid_cuboid_frame_instances": valid,
        "valid_cuboid_fraction": valid / instances if instances else None,
        "invalid_reasons": dict(reasons), "stability_samples": len(accelerations),
        "median_bbox_normalized_corner_second_difference": float(np.median(accelerations)) if accelerations else None,
        "p95_bbox_normalized_corner_second_difference": float(np.percentile(accelerations, 95)) if accelerations else None,
        "stability_interpretation": "RMS second finite difference of corners normalized within each tracked bbox, on consecutive valid frames of the same ID. Translation/box-scale motion is factored out. Turning, occlusion and localization changes still affect this proxy; not independent jitter or 3D accuracy.",
        "coverage_interpretation": "Valid means pipeline camera/prior fit acceptance. Coverage is not correctness or recall."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summarize(args.predictions), indent=2), encoding="utf-8")
    print(f"Diagnostics written to {args.output}")
