from __future__ import annotations

import cv2
import numpy as np

from .geometry import SceneGeometry
from .tracker import Track


EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def _color(value: list[int] | tuple[int, int, int]) -> tuple[int, int, int]:
    return int(value[0]), int(value[1]), int(value[2])


class Renderer:
    """Draw tracks, projective wireframes, and diagnostic geometry."""

    def __init__(self, config: dict):
        self.config = config
        self.front_color = _color(config.get("box_color", [0, 255, 255]))
        self.rear_color = _color(config.get("rear_color", [0, 170, 255]))
        self.text_color = _color(config.get("text_color", [255, 255, 255]))
        self.thickness = int(config.get("thickness", 3))

    def draw_wireframe(self, frame: np.ndarray, points: list[tuple[int, int]]) -> None:
        points = [tuple(int(round(v)) for v in point) for point in points]
        for index, (a, b) in enumerate(EDGES):
            color = self.front_color if index < 4 or index >= 8 else self.rear_color
            cv2.line(frame, points[a], points[b], color, self.thickness, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, max(2, self.thickness), self.front_color, -1, cv2.LINE_AA)

    def draw_track(self, frame: np.ndarray, track: Track, points: list[tuple[int, int]]) -> None:
        if self.config.get("draw_2d_box", True):
            x1, y1, x2, y2 = [int(round(v)) for v in track.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 180, 255), 1, cv2.LINE_AA)

        self.draw_wireframe(frame, points)

        if self.config.get("draw_labels", True):
            x1, y1, _, _ = [int(round(v)) for v in track.bbox]
            label = f"ID {track.track_id} {track.label}"
            cv2.putText(
                frame,
                label,
                (max(0, x1), max(88, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                self.text_color,
                2,
                cv2.LINE_AA,
            )

    def draw_uncertain_track(self, frame: np.ndarray, track: Track) -> None:
        x1, y1, x2, y2 = [int(round(v)) for v in track.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 90, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"ID {track.track_id} 3D uncertain", (max(0, x1), max(88, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (60, 90, 255), 1, cv2.LINE_AA)

    def draw_scene(self, frame: np.ndarray, geometry: SceneGeometry, frame_index: int, backend: str, active_tracks: int) -> None:
        if self.config.get("draw_vanishing_point", True):
            vp = tuple(int(round(v)) for v in geometry.vanishing_point)
            cv2.circle(frame, vp, 7, (255, 80, 80), -1, cv2.LINE_AA)
            line = geometry.horizon_line
            if line is not None and abs(line[1]) > 1e-9:
                y0 = int(round(-line[2] / line[1]))
                y1 = int(round(-(line[0] * frame.shape[1] + line[2]) / line[1]))
            else:
                y0 = y1 = int(round(geometry.horizon_y))
            cv2.line(frame, (0, y0), (frame.shape[1], y1), (255, 80, 80), 1, cv2.LINE_AA)

        text = f"Frame {frame_index} | {backend} + ByteTrack | Vehicles {active_tracks}"
        cv2.rectangle(frame, (12, 12), (min(frame.shape[1] - 12, 680), 70), (0, 0, 0), -1)
        cv2.putText(frame, text, (22, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58, self.text_color, 1, cv2.LINE_AA)
        cv2.putText(frame, "Camera-consistent cuboids | scene estimate + dimension priors", (22, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.46, self.text_color, 1, cv2.LINE_AA)
