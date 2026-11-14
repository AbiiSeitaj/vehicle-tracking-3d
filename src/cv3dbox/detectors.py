from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import cv2
import numpy as np


VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "van"}


@dataclass(frozen=True)
class Detection:
    """A single image-space object detection."""

    bbox: tuple[float, float, float, float]
    score: float
    label: str = "vehicle"


class Detector(Protocol):
    def detect(self, frame: np.ndarray, frame_index: int) -> list[Detection]:
        """Return detections for the current frame."""


def _fraction_polygon(points: Iterable[Iterable[float]], width: int, height: int) -> np.ndarray:
    absolute = [[int(x * width), int(y * height)] for x, y in points]
    return np.array(absolute, dtype=np.int32)


def _clip_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        float(np.clip(x1, 0, width - 1)),
        float(np.clip(y1, 0, height - 1)),
        float(np.clip(x2, 0, width - 1)),
        float(np.clip(y2, 0, height - 1)),
    )


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def non_max_suppression(detections: list[Detection], threshold: float = 0.35) -> list[Detection]:
    """Suppress heavily overlapping detections while retaining the highest score."""
    remaining = sorted(detections, key=lambda det: det.score, reverse=True)
    kept: list[Detection] = []
    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        remaining = [det for det in remaining if _iou(current.bbox, det.bbox) < threshold]
    return kept


class MotionVehicleDetector:
    """Static-camera foreground detector for near moving vehicles.

    The detector is deliberately calibration-free. It models the background
    from the video stream, constrains detections to a road polygon, cleans the
    foreground mask, and filters connected components by size and aspect ratio.
    """

    def __init__(self, config: dict):
        self.config = config
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=260,
            varThreshold=34,
            detectShadows=True,
        )
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 11))
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 9))

    def detect(self, frame: np.ndarray, frame_index: int) -> list[Detection]:
        height, width = frame.shape[:2]
        area = width * height
        min_area = self.config.get("min_area_ratio", 0.00045) * area
        max_area = self.config.get("max_area_ratio", 0.18) * area
        min_aspect = self.config.get("min_aspect_ratio", 0.45)
        max_aspect = self.config.get("max_aspect_ratio", 4.50)
        expand = self.config.get("bbox_expand", 0.08)

        fg = self.bg.apply(frame, learningRate=float(self.config.get("background_learning_rate", 0.002)))
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        roi_mask = np.zeros((height, width), dtype=np.uint8)
        roi_poly = _fraction_polygon(self.config.get("road_roi", []), width, height)
        if len(roi_poly) >= 3:
            cv2.fillPoly(roi_mask, [roi_poly], 255)
            fg = cv2.bitwise_and(fg, roi_mask)

        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel_open, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel_close, iterations=2)
        fg = cv2.dilate(fg, self.kernel_dilate, iterations=1)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            box_area = w * h
            if box_area < min_area or box_area > max_area:
                continue
            aspect = w / max(1, h)
            if aspect < min_aspect or aspect > max_aspect:
                continue
            if h < 24 or w < 28:
                continue
            if (x <= 3 or x + w >= width - 3) and w > width * 0.12:
                continue

            dx = w * expand
            dy = h * expand
            box = _clip_box((x - dx, y - dy, x + w + dx, y + h + dy), width, height)
            score = min(0.99, 0.45 + box_area / max(min_area * 12.0, 1.0))
            detections.append(Detection(bbox=box, score=float(score), label="vehicle"))

        return non_max_suppression(detections)


class UltralyticsVehicleDetector:
    """Optional YOLO detector used when the ultralytics package is installed."""

    def __init__(self, model_name: str, confidence: float, config: dict | None = None):
        from ultralytics import YOLO
        import torch

        self.config = config or {}
        torch.set_num_threads(int(self.config.get("cpu_threads", 4)))
        self.model = YOLO(model_name)
        self.confidence = confidence

    def detect(self, frame: np.ndarray, frame_index: int) -> list[Detection]:
        result = self.model.predict(frame, conf=self.confidence, verbose=False, classes=[2, 3, 5, 7],
            imgsz=int(self.config.get("image_size", 640)), device=self.config.get("device", "cpu"))[0]
        detections: list[Detection] = []
        names = result.names
        for box in result.boxes:
            cls_name = str(names[int(box.cls[0])]).lower()
            if cls_name not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            if y2 - y1 < float(self.config.get("min_height_px", 18)):
                continue
            detections.append(
                Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    score=float(box.conf[0]),
                    label=cls_name,
                )
            )
        return non_max_suppression(detections)


class AutoVehicleDetector:
    """Choose YOLO when available, otherwise fall back to OpenCV motion."""

    def __init__(self, config: dict):
        backend = str(config.get("backend", "yolo")).lower()
        self.motion = MotionVehicleDetector(config)
        self.primary: Detector = self.motion
        self.backend_name = "opencv-motion"

        if backend in {"auto", "yolo", "ultralytics"}:
            try:
                self.primary = UltralyticsVehicleDetector(
                    model_name=config.get("yolo_model", "yolov8n.pt"),
                    confidence=float(config.get("confidence", 0.28)),
                    config=config,
                )
                self.backend_name = "ultralytics-yolo"
            except Exception as exc:
                raise RuntimeError("YOLO could not load. Install requirements and run scripts/download_model.py. Use backend: motion only for the explicit baseline.") from exc
        elif backend != "motion":
            raise ValueError(f"Unsupported detector backend: {backend}")

    def detect(self, frame: np.ndarray, frame_index: int) -> list[Detection]:
        return self.primary.detect(frame, frame_index)
