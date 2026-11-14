"""Verify a generated video by sequentially decoding every output frame."""
import argparse
import json
from pathlib import Path
import subprocess
import cv2
import imageio_ffmpeg


def verify(path, expected_frames=None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot decode {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    minimum_std = float("inf")
    shape = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if shape is not None and frame.shape != shape:
                raise ValueError("Video dimensions changed during decoding")
            shape = frame.shape
            minimum_std = min(minimum_std, float(frame.std()))
            count += 1
    finally:
        cap.release()
    if not count or minimum_std < 1:
        raise ValueError("Empty or blank output video")
    if expected_frames is not None and count != expected_frames:
        raise ValueError(f"Decoded {count}, expected {expected_frames} frames")
    return {"video": Path(path).name, "decoded_frames": count, "fps": fps,
            "size": [shape[1], shape[0]], "minimum_frame_std": minimum_std}


def source_decode_count(path):
    process = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostats",
        "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough", "-progress", "pipe:1",
        "-f", "null", "-"], capture_output=True, text=True, check=True)
    progress = {}
    for line in process.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            progress[key] = value.strip()
    if progress.get("progress") != "end":
        raise ValueError("Independent source decoder did not reach completion.")
    return int(progress["frame"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--source", type=Path, help="Independent FFmpeg source decode count.")
    parser.add_argument("--metrics", type=Path, help="Augment metrics only after verified source/output count agreement.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.video, args.expected_frames)
    if args.source:
        result["source_ffmpeg_decoded_frames"] = source_decode_count(args.source)
        result["all_source_frames_present"] = result["decoded_frames"] == result["source_ffmpeg_decoded_frames"]
        if not result["all_source_frames_present"]:
            raise ValueError("Source/output frame counts differ.")
        if args.metrics:
            metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
            if metrics["frames_processed"] != result["decoded_frames"]:
                raise ValueError("Metrics/output frame counts differ.")
            metrics.update({"complete_sequence": True, "source_decoded_frames": result["source_ffmpeg_decoded_frames"],
                "completion_verification": "Output fully decoded; matches independent FFmpeg passthrough source decode count",
                "metadata_count_matches_processed": metrics["source_frames"] == result["decoded_frames"]})
            args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    elif args.metrics:
        raise ValueError("--metrics requires --source.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
