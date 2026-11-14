"""Render manual ground-truth overlays on exact sequential source frames.

No predictions are read. Bounds, coordinates and optional source hashes are
checked; completeness and identity correctness still require human review.
Example: python scripts/review_annotations.py --video data/input.mp4
--annotations scripts/validation_samples/manual_annotations.json
--output-dir scripts/validation_runs/challenge_annotation_review
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

try:
    from scripts.download_validation_samples import sha256
    from scripts.evaluate import evaluate, _reject_json_constant
    from scripts.export_annotation_frames import export_frames
except ModuleNotFoundError:
    from download_validation_samples import sha256
    from evaluate import evaluate, _reject_json_constant
    from export_annotation_frames import export_frames


def review_annotations(video: Path, annotations_path: Path, output_dir: Path) -> dict:
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"),
                             parse_constant=_reject_json_constant)
    evaluate([], annotations)
    size = annotations.get("image_size")
    if not isinstance(size, list) or len(size) != 2 or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in size):
        raise ValueError("Annotation image_size must be [positive integer width, height]")
    width, height = size
    for frame in annotations["frames"]:
        for obj in frame["objects"]:
            x1, y1, x2, y2 = obj["bbox"]
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Frame {frame['frame_index']}: annotation bbox exceeds image bounds")
    source_hash = sha256(video)
    expected_hash = annotations.get("source_video_sha256")
    if expected_hash is not None and source_hash != expected_hash:
        raise ValueError("Annotation source_video_sha256 does not match the input video")
    manifest = export_frames(video, [f["frame_index"] for f in annotations["frames"]],
                             output_dir / "source_frames", width=width)
    if manifest["image_size"] != size:
        raise ValueError("Annotation image_size does not match resized sequential source pixels")
    objects_by_frame = {f["frame_index"]: f["objects"] for f in annotations["frames"]}
    rendered = []
    for entry in manifest["frames"]:
        frame = cv2.imread(entry["path"])
        if frame is None:
            raise ValueError(f"Cannot read exported source frame {entry['path']}")
        objects = objects_by_frame[entry["frame_index"]]
        for obj in objects:
            x1, y1, x2, y2 = [round(v) for v in obj["bbox"]]
            cv2.rectangle(frame, (x1, y1), (min(x2, width - 1), min(y2, height - 1)),
                          (40, 230, 40), 2)
            label = f"{obj['id']} : {obj.get('label', 'vehicle')}"
            cv2.putText(frame, label, (min(x1, max(0, width - 400)), max(18, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (40, 230, 40), 1, cv2.LINE_AA)
        cv2.putText(frame, f"MANUAL GT / sequential frame {entry['frame_index']} / {len(objects)} objects",
                    (8, height - 12), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2, cv2.LINE_AA)
        path = output_dir / f"manual_gt_{entry['frame_index']:06d}.jpg"
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 98]):
            raise ValueError(f"Cannot write {path}")
        rendered.append({"frame_index": entry["frame_index"], "objects": len(objects),
                         "source_image_sha256": entry["sha256"],
                         "overlay_path": str(path.resolve()), "overlay_sha256": sha256(path)})
    report = {"annotations_sha256": sha256(annotations_path), "source_video_sha256": source_hash,
              "image_size": size, "source_hash_verified": expected_hash is not None,
              "checks": ["Schema and duplicate identities", "BBox bounds", "Sequential frame indices",
                         "Resized source dimensions", "Input and overlay SHA-256 hashes"],
              "limitations": "Structural checks do not prove exhaustive manual coverage, correct "
                             "identities, accurate boxes or3D geometry. Review overlays visually.",
              "frames": rendered}
    (output_dir / "annotation_review.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = review_annotations(args.video, args.annotations, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Rendered {len(report['frames'])} manual-only overlays at {report['image_size']}")


if __name__ == "__main__":
    main()
