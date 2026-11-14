from __future__ import annotations

import json
import hashlib
import platform
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .config import load_config, resolve_project_path
from .detectors import AutoVehicleDetector
from .geometry import CuboidProjector, GeometryEstimator
from .renderer import Renderer
from .tracker import MultiObjectTracker


def _resize(frame, output_width):
    if not output_width or frame.shape[1] == output_width:
        return frame
    height = round(frame.shape[0] * output_width / frame.shape[1])
    return cv2.resize(frame, (output_width, height), interpolation=cv2.INTER_AREA)


def _encoder_size(shape):
    height, width = shape[:2]
    return width + width % 2, height + height % 2


def _encode_frame(frame):
    # Chroma-subsampled writers truncate odd dimensions; pad without moving coordinates.
    height, width = frame.shape[:2]
    if height % 2 or width % 2:
        return cv2.copyMakeBorder(frame, 0, height % 2, 0, width % 2,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return frame


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def run_pipeline(config_path, input_video=None, output_video=None, max_frames=None,
                 snapshot_path=None, show=False):
    root = Path(config_path).resolve().parents[1]
    config = load_config(config_path)
    cv2.setNumThreads(int(config.get("runtime", {}).get("opencv_threads", 2)))
    input_cfg = config["input"]
    video = Path(input_video) if input_video else resolve_project_path(root, input_cfg["video_path"])
    output = Path(output_video) if output_video else resolve_project_path(root, input_cfg["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if video.resolve() == output.resolve():
        raise ValueError("Input and output must be different files.")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive.")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = max_frames if max_frames is not None else input_cfg.get("max_frames")
    detector_config = dict(config["detector"])
    detector_config["yolo_model"] = str(resolve_project_path(root, detector_config.get("yolo_model", "models/yolov8n.pt")))
    initialization_start = time.perf_counter()
    try:
        detector = AutoVehicleDetector(detector_config)
        tracker = MultiObjectTracker(config["tracker"], fps)
    except Exception:
        cap.release()
        raise
    initialization_seconds = time.perf_counter() - initialization_start
    estimator = GeometryEstimator(config["geometry"])
    calibration_start = time.perf_counter()
    first_shape = None
    for sample_index in range(int(config["geometry"].get("line_sample_frames", 30))):
        ok, sample = cap.read()
        if not ok:
            break
        sample = _resize(sample, input_cfg.get("output_width"))
        first_shape = sample.shape
        estimator.add_sample(sample)
        if sample_index % 7 == 0 and hasattr(estimator, "add_detection"):
            for detection in detector.detect(sample, sample_index):
                if detection.score >= .3:
                    estimator.add_detection(detection.bbox, detection.label)
    if first_shape is None:
        cap.release()
        raise ValueError("Video contains no readable frames.")
    geometry = estimator.estimate(first_shape)
    calibration_seconds = time.perf_counter() - calibration_start
    cap.release()
    cap = cv2.VideoCapture(str(video))
    renderer = Renderer(config["render"])
    projector = CuboidProjector(geometry, config["geometry"])
    writer = _make_writer(output, first_shape, fps)
    prediction_path = output.with_suffix(".predictions.jsonl")
    frame_count = total_detections = max_tracks = valid_cuboids = 0
    latencies = []
    fit_errors = []
    seen_ids = set()
    snapshot_written = False
    read_terminated = False
    started = time.perf_counter()
    try:
        with prediction_path.open("w", encoding="utf-8") as predictions, tqdm(
                total=limit or source_frames, desc=video.name, unit="frame") as progress:
            while limit is None or frame_count < limit:
                frame_start = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    read_terminated = True
                    break
                frame = _resize(frame, input_cfg.get("output_width"))
                detections = detector.detect(frame, frame_count)
                tracks = tracker.update(detections, frame)
                record = {"frame_index": frame_count, "source_timestamp_seconds": cap.get(cv2.CAP_PROP_POS_MSEC) / 1000,
                          "detections": [_jsonable(d) for d in detections], "tracks": []}
                for track in tracks:
                    fit = projector.fit(track.bbox, label=track.label, track_id=track.track_id)
                    data = _jsonable(fit.to_dict() if hasattr(fit, "to_dict") else fit)
                    points = data.get("corners", [])
                    is_valid = bool(data.get("valid", True)) and len(points) == 8
                    if is_valid:
                        renderer.draw_track(frame, track, points)
                        valid_cuboids += 1
                        fit_errors.append(float(data.get("diagnostics", {}).get("normalized_fit_error", 0)))
                    else:
                        renderer.draw_uncertain_track(frame, track)
                    record["tracks"].append({**_jsonable(track), **data})
                    seen_ids.add(track.track_id)
                renderer.draw_scene(frame, geometry, frame_count, detector.backend_name, len(tracks))
                predictions.write(json.dumps(record, allow_nan=False) + "\n")
                writer.write(_encode_frame(frame))
                if snapshot_path and tracks and frame_count >= 60 and not snapshot_written:
                    snapshot = Path(snapshot_path)
                    snapshot.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(snapshot), frame):
                        raise RuntimeError("Snapshot write failed.")
                    snapshot_written = True
                total_detections += len(detections)
                max_tracks = max(max_tracks, len(tracks))
                latencies.append(time.perf_counter() - frame_start)
                frame_count += 1
                progress.update(1)
                if show:
                    cv2.imshow("Monocular vehicle cuboids", frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
    finally:
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()
    elapsed = time.perf_counter() - started
    if snapshot_path and frame_count and not snapshot_written:
        snapshot = Path(snapshot_path)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(snapshot), frame):
            raise RuntimeError("Snapshot write failed.")
    metrics = {
        "input_video": video.name, "output_video": output.name,
        "frames_processed": frame_count, "source_frames": source_frames,
        "complete_sequence": frame_count == source_frames or (limit is None and read_terminated),
        "completion_definition": "All readable sequential frames when no limit; cross-check source decode count with verify_video.py",
        "source_frame_count_metadata": source_frames,
        "metadata_count_matches_processed": frame_count == source_frames,
        "read_terminated": read_terminated, "source_fps": fps,
        "processing_fps": frame_count / max(elapsed, 1e-9),
        "latency_ms_median": float(np.median(latencies) * 1000) if latencies else None,
        "latency_ms_p95": float(np.percentile(latencies, 95) * 1000) if latencies else None,
        "calibration_seconds": calibration_seconds, "initialization_seconds": initialization_seconds,
        "timing_scope": "decode+resize+detect+track+fit+render+write JSONL/video; excludes calibration/model initialization; includes UI if enabled",
        "total_detections": total_detections, "max_active_tracks": max_tracks,
        "unique_track_ids": len(seen_ids), "valid_cuboid_frame_instances": valid_cuboids,
        "median_normalized_fit_error": float(np.median(fit_errors)) if fit_errors else None,
        "fit_error_note": "Envelope fit to detected 2D box; not independent 3D accuracy.",
        "detector_backend": detector.backend_name, "tracker": "Ultralytics ByteTrack",
        "geometry": _jsonable(geometry.to_dict() if hasattr(geometry, "to_dict") else geometry),
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "opencv": cv2.__version__},
        "output_size": [first_shape[1], first_shape[0]],
        "encoded_video_size": list(_encoder_size(first_shape)),
        "video_padding": "Black bottom/right padding for odd dimensions; prediction coordinates unchanged",
        "configuration": config,
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(Path(detector_config["yolo_model"]).read_bytes()).hexdigest()
            if Path(detector_config["yolo_model"]).is_file() else None,
    }
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    return metrics


def _make_writer(path, shape, fps):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, _encoder_size(shape))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {path}")
    return writer
