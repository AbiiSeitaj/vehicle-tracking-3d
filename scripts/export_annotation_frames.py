"""Extract exact indexed source frames for independent manual annotation.

Uses the pipeline's aspect-preserving INTER_AREA resize to --width (1280).
Frames are decoded sequentially to avoid keyframe-seek ambiguity. This exports
source pixels only, without model predictions. The manifest records input and
image hashes plus coordinate size. Example:
python scripts/export_annotation_frames.py --video data/input.mp4 \
    --frames 0 60 120 240 360 480 600 720 --output-dir scripts/validation_samples/challenge_frames
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

try:
    from scripts.download_validation_samples import sha256
except ModuleNotFoundError:
    from download_validation_samples import sha256


def export_frames(video: Path, indices: list[int], output_dir: Path, width: int = 1280) -> dict:
    if width < 1 or not indices or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices):
        raise ValueError("Use a positive width and nonnegative integer frame indices")
    wanted = set(indices)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    exported = []
    image_size = None
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open {video}")
        index = 0
        while wanted:
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Frames beyond readable sequence: {sorted(wanted)}")
            if index in wanted:
                if frame.shape[1] != width:
                    height = round(frame.shape[0] * width / frame.shape[1])
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                size = [frame.shape[1], frame.shape[0]]
                if image_size is not None and image_size != size:
                    raise ValueError("Frame coordinate size changed")
                image_size = size
                path = output_dir / f"frame_{index:06d}.jpg"
                if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 98]):
                    raise ValueError(f"Cannot write {path}")
                exported.append({"frame_index": index, "path": str(path.resolve()), "sha256": sha256(path)})
                wanted.remove(index)
            index += 1
    finally:
        capture.release()
    report = {"source_video": str(video.resolve()), "source_video_sha256": sha256(video),
              "image_size": image_size, "method": "Sequential source decode; no model overlays",
              "resize": "Aspect-preserving OpenCV INTER_AREA", "opencv_version": cv2.__version__,
              "frames": exported}
    (output_dir / "frame_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()
    try:
        report = export_frames(args.video, args.frames, args.output_dir, args.width)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Exported {len(report['frames'])} source frames at {report['image_size']}")


if __name__ == "__main__":
    main()
