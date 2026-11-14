from __future__ import annotations

import argparse
from pathlib import Path

from src.cv3dbox.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run calibration-free 3D vehicle wireframe tracking.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML configuration file.")
    parser.add_argument("--input", default=None, help="Input video path.")
    parser.add_argument("--output", default=None, help="Output annotated video path.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for quick demos.")
    parser.add_argument("--snapshot", default="outputs/sample_frame.jpg", help="Optional annotated snapshot path.")
    parser.add_argument("--show", action="store_true", help="Show live processed frames; Q or Escape stops.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = project_root / args.config
    metrics = run_pipeline(
        config_path=config_path,
        input_video=args.input,
        output_video=args.output,
        max_frames=args.max_frames,
        snapshot_path=args.snapshot,
        show=args.show,
    )
    print("Pipeline complete")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
