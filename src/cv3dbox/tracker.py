from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .detectors import Detection


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    score: float
    label: str = "vehicle"


def box_iou(a, b) -> float:
    lo = np.maximum(a[:2], b[:2])
    hi = np.minimum(a[2:], b[2:])
    intersection = float(np.prod(np.maximum(0, hi - lo)))
    areas = [float(np.prod(np.maximum(0, np.array(box[2:]) - box[:2]))) for box in (a, b)]
    return intersection / max(areas[0] + areas[1] - intersection, 1e-9)


class MultiObjectTracker:
    """Official ByteTrack; lost tracks are retained but not rendered."""

    def __init__(self, config: dict, fps: float = 30):
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.engine = BYTETracker(SimpleNamespace(
            track_high_thresh=float(config.get("track_high_thresh", 0.30)),
            track_low_thresh=float(config.get("track_low_thresh", 0.10)),
            new_track_thresh=float(config.get("new_track_thresh", 0.35)),
            track_buffer=int(config.get("track_buffer", 30)),
            match_thresh=float(config.get("match_thresh", 0.8)),
            fuse_score=True,
        ))
        self.smoothing = float(config.get("smoothing", 0.20))
        self.previous = {}
        self.names = ["car", "truck", "bus", "motorcycle", "vehicle"]

    def update(self, detections: list[Detection], frame=None) -> list[Track]:
        from ultralytics.engine.results import Boxes

        array = np.asarray([[*det.bbox, det.score,
            self.names.index(det.label) if det.label in self.names else 4]
            for det in detections], dtype=np.float32).reshape(-1, 6)
        height, width = frame.shape[:2] if frame is not None else (720, 1280)
        active = self.engine.update(Boxes(array, (height, width)).numpy(), frame)
        tracks = []
        current = {}
        for row in active:
            track_id = int(row[4])
            box = tuple(float(v) for v in row[:4])
            if track_id in self.previous:
                box = tuple(self.smoothing * old + (1 - self.smoothing) * new
                            for old, new in zip(self.previous[track_id], box))
            current[track_id] = box
            tracks.append(Track(track_id, box, float(row[5]), self.names[int(row[6])]))
        self.previous = current
        return tracks
