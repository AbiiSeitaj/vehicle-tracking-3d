from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations
from time import perf_counter
from typing import Any

import cv2
import numpy as np


Point = tuple[int, int]


@dataclass(frozen=True)
class SceneGeometry:
    """Pinhole camera with R mapping world (lateral, up, road) to camera XYZ.

    Camera coordinates are right/down/forward. World ground is Y=0 and
    camera center is (0, camera_height_m, 0). Confidence measures road VP
    line support, not metric calibration. The first four fields retain the
    original constructor contract.
    """

    vanishing_point: tuple[float, float]
    horizon_y: float
    confidence: float
    source: str
    image_size: tuple[int, int] | None = None
    K: np.ndarray | None = field(default=None, repr=False, compare=False)
    R: np.ndarray | None = field(default=None, repr=False, compare=False)
    ground_normal: np.ndarray | None = field(default=None, repr=False, compare=False)
    camera_height_m: float = 2.5
    horizon_line: tuple[float, float, float] | None = None
    secondary_vanishing_point: tuple[float, float] | None = None
    calibration_mode: str = "legacy_uncalibrated"
    assumptions: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible calibration, conventions and evidence."""
        return {
            "vanishing_point": list(self.vanishing_point),
            "horizon_y": float(self.horizon_y),
            "confidence": float(self.confidence),
            "source": self.source,
            "image_size": None if self.image_size is None else list(self.image_size),
            "K": None if self.K is None else np.asarray(self.K).tolist(),
            "R": None if self.R is None else np.asarray(self.R).tolist(),
            "calibration_mode": self.calibration_mode,
            "horizon_line": None if self.horizon_line is None else list(self.horizon_line),
            "secondary_vanishing_point": (
                None if self.secondary_vanishing_point is None
                else list(self.secondary_vanishing_point)
            ),
            "ground_plane": {
                "world_equation": [0.0, 1.0, 0.0, 0.0],
                "camera_normal": (
                    None if self.ground_normal is None else np.asarray(self.ground_normal).tolist()
                ),
                "camera_offset_m": float(self.camera_height_m),
                "camera_center_world_m": [0.0, float(self.camera_height_m), 0.0],
            },
            "coordinate_system": {
                "world": "X lateral (up cross road), Y up, Z road; meters; right-handed",
                "camera": "X right, Y down, Z forward",
                "rotation": "camera = R @ (world - camera_center_world)",
            },
            "assumptions": dict(self.assumptions),
            "diagnostics": dict(self.diagnostics),
        }


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-10:
        raise ValueError("Degenerate camera direction or ground plane")
    return vector / norm


def _line_from_segment(segment: np.ndarray) -> tuple[float, float, float] | None:
    x1, y1, x2, y2 = np.asarray(segment, dtype=float)
    length = np.hypot(x2 - x1, y2 - y1)
    if not np.isfinite(length) or length < 20:
        return None
    return (y1 - y2) / length, (x2 - x1) / length, (x1 * y2 - x2 * y1) / length


def _intersection(line_a: tuple, line_b: tuple) -> tuple[float, float] | None:
    point = np.cross(line_a, line_b)
    if abs(point[2]) < 1e-7:
        return None
    return float(point[0] / point[2]), float(point[1] / point[2])


def _camera_geometry(
    width: int, height: int, vp: tuple[float, float], horizon_prior_y: float,
    config: dict, confidence: float, source: str, diagnostics: dict,
    secondary: tuple[float, float] | None = None,
) -> SceneGeometry:
    center = np.array([width / 2.0, height / 2.0])
    fov = float(config.get("horizontal_fov_deg", config.get("fov_deg", 60.0)))
    if not 5.0 < fov < 175.0:
        raise ValueError("horizontal_fov_deg must be between 5 and 175 degrees")
    focal = width / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
    focal_source = "horizontal_fov_prior"
    if "focal_length_px" in config:
        focal = float(config["focal_length_px"])
        focal_source = "configured_focal_length"
    if secondary is not None:
        focal_squared = -float(np.dot(np.asarray(vp) - center, np.asarray(secondary) - center))
        if focal_squared > 0:
            focal = float(np.sqrt(focal_squared))
            focal_source = "orthogonal_ground_vps_assumption"
        else:
            secondary = None
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError("focal_length_px must be finite and positive")
    K = np.array([[focal, 0.0, center[0]], [0.0, focal, center[1]], [0.0, 0.0, 1.0]])
    road = _unit(np.linalg.solve(K, [*vp, 1.0]))
    if secondary is not None:
        transverse = _unit(np.linalg.solve(K, [*secondary, 1.0]))
        up = _unit(np.cross(road, transverse))
        horizon_source = "two_ground_vps_assumption"
    else:
        # Closest normal to the horizontal-horizon prior containing the
        # road ray. One VP cannot independently recover roll/focal length.
        prior_up = -_unit(K.T @ np.array([0.0, 1.0, -horizon_prior_y]))
        up = _unit(prior_up - np.dot(prior_up, road) * road)
        horizon_source = "horizontal_horizon_prior_conditioned_on_road_vp"
    if up[1] > 0:
        up = -up
    right = _unit(np.cross(up, road))
    R = np.column_stack((right, up, road))
    horizon = np.linalg.solve(K.T, up)
    if abs(horizon[1]) < 1e-9:
        raise ValueError("Ground horizon is vertical; unsupported camera pose")
    horizon /= horizon[1]
    horizon_y = -float(horizon[0] * center[0] + horizon[2])
    camera_height = float(config.get("camera_height_m", 2.5))
    if not np.isfinite(camera_height) or camera_height <= 0:
        raise ValueError("camera_height_m must be finite and positive")
    mode = ("two_vp_assumed_orthogonal" if secondary is not None else
            "one_vp_with_priors" if confidence > 0 else "prior_only")
    assumptions = {
        "principal_point": "image_center", "square_pixels": True, "skew": 0.0,
        "lens_distortion": "ignored", "ground": "single_flat_plane",
        "road_direction": "dominant_supported_line_family" if confidence > 0 else "fallback_prior",
        "focal_length_source": focal_source, "horizontal_fov_prior_deg": fov,
        "horizon_source": horizon_source, "horizon_prior_y_px": float(horizon_prior_y),
        "camera_height_source": "configured_prior" if "camera_height_m" in config else "default_prior",
        "metric_scale": "assumed_camera_height_and_class_dimensions",
        "orthogonal_ground_directions": secondary is not None,
        "pose_quality": "conditional_on_scene_priors",
    }
    diagnostics = dict(diagnostics)
    diagnostics["confidence_meaning"] = "road_vp_line_support_only"
    diagnostics["horizon_prior_adjustment_px"] = horizon_y - horizon_prior_y
    return SceneGeometry(
        vanishing_point=tuple(float(v) for v in vp), horizon_y=horizon_y,
        confidence=float(confidence), source=source, image_size=(width, height),
        K=K, R=R, ground_normal=up, camera_height_m=camera_height,
        horizon_line=tuple(float(v) for v in horizon),
        secondary_vanishing_point=secondary, calibration_mode=mode,
        assumptions=assumptions, diagnostics=diagnostics,
    )


class GeometryEstimator:
    """RANSAC line support with explicit uncalibrated-camera priors."""

    def __init__(self, config: dict):
        self.config = config
        self.samples: list[np.ndarray] = []
        self.detections: list[tuple[np.ndarray, str, Any]] = []

    def add_sample(self, frame: np.ndarray) -> None:
        if len(self.samples) < int(self.config.get("line_sample_frames", 45)):
            self.samples.append(frame.copy())

    def add_detection(self, bbox: tuple, label: str = "car", track_id=None) -> None:
        """Collect semantic horizon evidence before estimate, in sample pixels.

        The height relation neglects footprint extent/perspective, suspension,
        and class-size variation. It is a prior-informed estimate, not measured
        camera calibration. Truncated boxes are filtered once image size is known.
        """
        box = np.asarray(bbox, dtype=float)
        if box.shape != (4,) or not np.all(np.isfinite(box)) or np.any(box[2:] <= box[:2]):
            return
        label = str(label).lower()
        priors = self.config.get("dimension_priors", self.config.get("class_dimensions", {}))
        if label not in CuboidProjector.DIMENSION_PRIORS and label not in priors:
            return
        if label == "vehicle":
            return
        if len(self.detections) < int(self.config.get("max_semantic_samples", 300)):
            self.detections.append((box.copy(), label, track_id))

    def add_detections(self, detections) -> None:
        """Accept detection objects or dictionaries with bbox and label fields."""
        for detection in detections:
            if isinstance(detection, dict):
                self.add_detection(detection["bbox"], detection.get("label", "car"), detection.get("track_id"))
            else:
                self.add_detection(detection.bbox, detection.label, getattr(detection, "track_id", None))

    def _semantic_horizon(self, width: int, height: int) -> tuple[float | None, dict]:
        camera_height = float(self.config.get("camera_height_m", 2.5))
        priors = self.config.get("dimension_priors", self.config.get("class_dimensions", {}))
        values: dict[Any, list[float]] = {}
        accepted = 0
        for index, (box, label, track_id) in enumerate(self.detections):
            if (box[0] <= 1 or box[1] <= 1 or box[2] >= width - 2 or box[3] >= height - 2
                    or box[3] - box[1] < float(self.config.get("semantic_min_height_px", 20))):
                continue
            dimensions = priors.get(label, CuboidProjector.DIMENSION_PRIORS.get(label))
            class_height = float(dimensions["height"] if isinstance(dimensions, dict) else dimensions[2])
            if not np.isfinite(class_height) or class_height <= 0:
                continue
            estimate = float(box[3] - camera_height * (box[3] - box[1]) / class_height)
            if not 0.05 * height <= estimate <= 0.65 * height:
                continue
            # Repeated observations of one known track receive one vote.
            key = (label, track_id) if track_id is not None else ("sample", index)
            values.setdefault(key, []).append(estimate)
            accepted += 1
        estimates = np.array([np.median(v) for v in values.values()])
        diagnostics = {
            "sample_count": len(self.detections), "accepted_box_count": accepted,
            "vote_count": len(estimates), "formula": "bbox_bottom - camera_height_m * bbox_height / class_height_m",
            "interpretation": "approximate_horizon_using_assumed_heights_and_negligible_footprint_depth",
            "limitations": ["ignores_footprint_extent", "class_heights_and_camera_height_are_priors",
                            "detections_without_track_ids_may_be_correlated"],
        }
        if len(estimates) < int(self.config.get("min_semantic_samples", 3)):
            diagnostics["accepted"] = False
            return None, diagnostics
        median = float(np.median(estimates))
        mad = float(np.median(np.abs(estimates - median)))
        inliers = estimates[np.abs(estimates - median) <= max(15.0, 3.0 * 1.4826 * mad)]
        median = float(np.median(inliers))
        diagnostics.update({"accepted": True, "horizon_y_px": median, "mad_px": mad,
                            "inlier_vote_count": len(inliers)})
        return median, diagnostics

    def _segments(self, width: int, height: int) -> np.ndarray:
        segments = []
        roi = self.config.get("line_roi_y", [0.40, 0.95])
        for frame in self.samples:
            if frame.shape[:2] != (height, width):
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 160)
            edges[:int(height * roi[0])] = 0
            edges[int(height * roi[1]):] = 0
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180, int(self.config.get("hough_threshold", 60)),
                minLineLength=int(self.config.get("min_line_length_px", 60)), maxLineGap=18,
            )
            if lines is not None:
                segments.extend(lines.reshape(-1, 4))
        return np.asarray(segments, dtype=float).reshape(-1, 4)

    def _unique_segments(self, segments: np.ndarray) -> np.ndarray:
        # Static-frame repeats and opposite edges of a painted line are not
        # independent evidence for camera calibration.
        unique, lines = [], []
        lengths = np.linalg.norm(segments[:, 2:] - segments[:, :2], axis=1)
        for index in np.argsort(-lengths):
            segment = segments[index]
            line = _line_from_segment(segment)
            if line is None:
                continue
            line = np.asarray(line)
            if line[0] < 0 or (abs(line[0]) < 1e-8 and line[1] < 0):
                line = -line
            if any(np.dot(line[:2], old[:2]) > np.cos(np.deg2rad(1.0))
                   and abs(line[2] - old[2]) < 6.0 for old in lines):
                continue
            unique.append(segment)
            lines.append(line)
            if len(unique) >= int(self.config.get("max_line_segments", 300)):
                break
        return np.asarray(unique, dtype=float).reshape(-1, 4)

    def _ransac(
        self, segments: np.ndarray, width: int, height: int, *, secondary: bool = False,
        road_vp: tuple[float, float] | None = None,
    ) -> tuple[tuple[float, float] | None, np.ndarray, dict]:
        empty = np.zeros(len(segments), dtype=bool)
        minimum = int(self.config.get("min_vp_lines", 4))
        if len(segments) < minimum:
            return None, empty, {"inlier_count": 0}
        lines = np.asarray([_line_from_segment(s) for s in segments])
        mids = (segments[:, :2] + segments[:, 2:]) / 2
        lengths = np.minimum(np.linalg.norm(segments[:, 2:] - segments[:, :2], axis=1), width / 3)
        threshold = np.sin(np.deg2rad(float(self.config.get("vp_inlier_angle_deg", 2.0))))
        trials = int(self.config.get("ransac_trials", 1200))
        if len(segments) * (len(segments) - 1) // 2 <= trials:
            pairs = list(combinations(range(len(segments)), 2))
        else:
            rng = np.random.default_rng(int(self.config.get("ransac_seed", 0)))
            pairs = [rng.choice(len(segments), 2, replace=False) for _ in range(trials)]

        def allowed(point: np.ndarray) -> bool:
            if not np.all(np.isfinite(point)):
                return False
            if not secondary:
                return (-0.5 * width <= point[0] <= 1.5 * width
                        and -0.5 * height <= point[1] <= 0.75 * height)
            if abs(point[0]) > 20 * width or abs(point[1]) > 10 * height:
                return False
            center = np.array([width / 2, height / 2])
            f2 = -np.dot(np.asarray(road_vp) - center, point - center)
            if f2 <= 0:
                return False
            fov = np.rad2deg(2 * np.arctan(width / (2 * np.sqrt(f2))))
            return 20 <= fov <= 130

        def support(point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            distance = np.maximum(np.linalg.norm(point - mids, axis=1), 20)
            errors = np.abs(lines[:, :2] @ point + lines[:, 2]) / distance
            return errors <= threshold, errors

        best_point, best_score = None, -1.0
        for a, b in pairs:
            point = _intersection(lines[a], lines[b])
            if point is None or not allowed(np.asarray(point)):
                continue
            point = np.asarray(point)
            inliers, errors = support(point)
            if np.count_nonzero(inliers) < minimum:
                continue
            normals = lines[inliers, :2]
            if np.linalg.svd(normals, compute_uv=False)[-1] < 0.08:
                continue
            score = float(np.sum(lengths[inliers] * (1 - (errors[inliers] / threshold) ** 2)))
            if score > best_score:
                best_point, best_score = point, score
        if best_point is None:
            return None, empty, {"inlier_count": 0}
        for _ in range(4):
            inliers, _ = support(best_point)
            weights = np.sqrt(lengths[inliers]) / np.maximum(np.linalg.norm(best_point - mids[inliers], axis=1), 20)
            refined = np.linalg.lstsq(
                lines[inliers, :2] * weights[:, None], -lines[inliers, 2] * weights, rcond=None,
            )[0]
            if not allowed(refined):
                break
            refined_inliers, _ = support(refined)
            if np.count_nonzero(refined_inliers) < minimum:
                break
            best_point = refined
        inliers, errors = support(best_point)
        diagnostics = {
            "inlier_count": int(np.count_nonzero(inliers)),
            "line_count": len(segments), "inlier_fraction": float(np.mean(inliers)),
            "median_inlier_angle_deg": float(np.rad2deg(np.arcsin(np.clip(np.median(errors[inliers]), 0, 1)))),
            "weighted_support": float(np.sum(lengths[inliers]) / np.sum(lengths)),
        }
        return tuple(float(v) for v in best_point), inliers, diagnostics

    def estimate(self, frame_shape: tuple[int, ...]) -> SceneGeometry:
        height, width = frame_shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive")
        fallback_vp = self.config.get("fallback_vanishing_point", [0.5, 0.36])
        fallback = (float(fallback_vp[0]) * width, float(fallback_vp[1]) * height)
        horizon_prior = float(self.config.get("fallback_horizon_y", 0.36)) * height
        semantic_horizon, semantic_diagnostics = self._semantic_horizon(width, height)
        segments = self._unique_segments(self._segments(width, height))
        vp, inliers, road_diagnostics = self._ransac(segments, width, height)
        secondary, secondary_diagnostics = None, {"inlier_count": 0}
        if vp is not None and self.config.get("estimate_orthogonal_vp", True):
            secondary, _, secondary_diagnostics = self._ransac(
                segments[~inliers], width, height, secondary=True, road_vp=vp,
            )
        confidence = 0.0
        if vp is not None:
            confidence = min(0.9, road_diagnostics["weighted_support"]
                             * min(1.0, road_diagnostics["inlier_count"] / 12))
        diagnostics = {
            "sample_count": len(self.samples), "unique_line_count": len(segments),
            "road_vp_evidence": road_diagnostics,
            "secondary_vp_evidence": secondary_diagnostics,
            "secondary_vp_interpretation": "orthogonality_and_ground_membership_are_assumed",
            "limitations": ["dominant_lines_may_be_buildings_or_vehicles", "no_metric_calibration_from_one_vp"],
            "semantic_horizon_evidence": semantic_diagnostics,
        }
        if secondary is not None:
            pair_line = np.cross([*vp, 1.0], [*secondary, 1.0])
            roll = np.rad2deg(np.arctan2(abs(pair_line[0]), abs(pair_line[1])))
            diagnostics["line_pair_roll_deg"] = float(roll)
            support = secondary_diagnostics.get("weighted_support", 0.0)
            if support < float(self.config.get("min_secondary_vp_support", 0.15)):
                diagnostics["line_pair_rejected"] = "insufficient_secondary_line_support"
                secondary = None
            elif roll > float(self.config.get("max_horizon_roll_deg", 15.0)):
                diagnostics["line_pair_rejected"] = "exceeds_camera_roll_prior"
                secondary = None
        semantic_pose = False
        if semantic_horizon is not None:
            horizon_prior = semantic_horizon
            tolerance = float(self.config.get("semantic_horizon_tolerance_px", 60.0))
            diagnostics["observed_line_vanishing_point"] = None if vp is None else list(vp)
            diagnostics["observed_line_vp_confidence"] = confidence
            if secondary is not None:
                pair_horizon = np.cross([*vp, 1.0], [*secondary, 1.0])
                pair_y = (float("inf") if abs(pair_horizon[1]) < 1e-9 else
                          -(pair_horizon[0] * width / 2 + pair_horizon[2]) / pair_horizon[1])
                diagnostics["line_pair_horizon_y_px"] = float(pair_y) if np.isfinite(pair_y) else None
                if abs(pair_y - semantic_horizon) > tolerance:
                    diagnostics["line_pair_rejected"] = "incompatible_with_semantic_horizon_prior"
                    secondary = None
            if secondary is None:
                incompatible = vp is None or abs(vp[1] - semantic_horizon) > tolerance
                diagnostics["road_vp_ground_membership_rejected"] = incompatible
                diagnostics["road_direction_vertical_source"] = "semantic_horizon_with_zero_roll_prior"
                vp = (fallback[0] if incompatible else vp[0], semantic_horizon)
                confidence = 0.0
                semantic_pose = True
        scene = _camera_geometry(
            width, height, fallback if vp is None else vp, horizon_prior, self.config,
            confidence, "semantics_with_fov_prior" if semantic_pose else
            "fallback" if vp is None else "ransac-line-support", diagnostics, secondary,
        )
        if semantic_pose:
            scene = replace(scene, calibration_mode="semantics_with_fov_prior", assumptions={
                **scene.assumptions, "horizon_source": "semantic_vehicle_height_prior",
                "road_direction": "semantic_horizon_and_fallback_road_direction_prior"
                if diagnostics["road_vp_ground_membership_rejected"] else "line_vp_horizontal_coordinate_with_semantic_horizon_prior",
                "roll": "zero_prior", "semantic_horizon_is_metric_calibration": False,
            })
        return scene


@dataclass(frozen=True)
class CuboidFit:
    """Conditional pose: dimensions L/W/H, yaw radians, bbox-edge RMS pixels."""

    corners: list[tuple[float, float]]
    fit_error: float
    position: tuple[float, float, float]
    yaw: float
    dimensions: tuple[float, float, float]
    valid: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "corners": [list(p) for p in self.corners],
            "fit_error": float(self.fit_error) if np.isfinite(self.fit_error) else None,
            "position": list(self.position), "yaw": float(self.yaw),
            "dimensions": list(self.dimensions), "valid": self.valid,
            "diagnostics": dict(self.diagnostics),
        }


def _bounded_least_squares(
    fun, initial: np.ndarray, lower: np.ndarray, upper: np.ndarray, iterations: int,
) -> np.ndarray:
    """Small damped Gauss-Newton fallback for installations without SciPy."""
    state = np.clip(initial, lower, upper)
    damping = 1e-3
    for _ in range(iterations):
        residual = fun(state)
        steps = 1e-5 * np.maximum(np.abs(state), 1.0)
        jacobian = np.column_stack([
            (fun(state + np.eye(len(state))[j] * steps[j]) - residual) / steps[j]
            for j in range(len(state))
        ])
        try:
            delta = np.linalg.solve(jacobian.T @ jacobian + damping * np.eye(len(state)),
                                    -jacobian.T @ residual)
        except np.linalg.LinAlgError:
            break
        candidate = np.clip(state + delta, lower, upper)
        candidate_residual = fun(candidate)
        if np.dot(candidate_residual, candidate_residual) < np.dot(residual, residual):
            change = np.linalg.norm(candidate - state)
            state = candidate
            damping = max(1e-8, damping / 3)
            if change < 1e-6:
                break
        else:
            damping = min(1e6, damping * 10)
    return state


class CuboidProjector:
    """Fit ground cuboids, projecting all corners through the same K and R.

    Position is the footprint center in meters. Yaw is radians from world
    +Z toward +X. Each face has two bottom then two top corners in cyclic
    order, followed by its matching far face, preserving renderer edges.
    Track smoothing requires reusing this projector across frames.
    """

    DIMENSION_PRIORS = {
        "car": (4.5, 1.8, 1.5), "vehicle": (4.5, 1.8, 1.5),
        "truck": (12.0, 2.5, 3.8), "bus": (12.0, 2.5, 3.2),
        "van": (5.5, 2.0, 2.5), "motorcycle": (2.2, 0.8, 1.4),
    }

    def __init__(self, geometry: SceneGeometry, config: dict):
        self.config = config
        if geometry.K is None or geometry.R is None:
            size = geometry.image_size or config.get("image_size")
            if size is None:
                width = int(config.get("image_width", max(2 * geometry.vanishing_point[0], 640)))
                size = (width, int(config.get("image_height", width * 9 / 16)))
            geometry = _camera_geometry(
                int(size[0]), int(size[1]), geometry.vanishing_point, geometry.horizon_y,
                config, geometry.confidence, geometry.source,
                {"legacy_camera": "image_size_inferred_from_vp_and_16_9_prior"
                 if geometry.image_size is None and "image_size" not in config
                 and "image_width" not in config else "configured_image_size"},
            )
        self.geometry = geometry
        self.K = np.asarray(geometry.K, dtype=float).copy()
        self.R = np.asarray(geometry.R, dtype=float).copy()
        if (self.K.shape != (3, 3) or self.R.shape != (3, 3)
                or not np.all(np.isfinite(self.K)) or not np.all(np.isfinite(self.R))
                or not np.allclose(self.R.T @ self.R, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(self.R), 1.0, atol=1e-6)):
            raise ValueError("K and R must define a finite camera with a proper rotation")
        self.camera_center = np.array([0.0, geometry.camera_height_m, 0.0])
        self._states: dict[Any, tuple[str, np.ndarray]] = {}
        self._track_fit_counts: dict[Any, int] = {}
        self._state_valid: dict[Any, bool] = {}

    @staticmethod
    def world_corners(position: tuple, yaw: float, dimensions: tuple) -> np.ndarray:
        length, width, height = dimensions
        local = np.array([
            [-width / 2, 0, -length / 2], [width / 2, 0, -length / 2],
            [width / 2, height, -length / 2], [-width / 2, height, -length / 2],
            [-width / 2, 0, length / 2], [width / 2, 0, length / 2],
            [width / 2, height, length / 2], [-width / 2, height, length / 2],
        ])
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return local @ rotation.T + np.asarray(position)

    def project(self, points: np.ndarray) -> np.ndarray:
        """Project world points; never mirror or clamp behind-camera points."""
        camera = (np.asarray(points, dtype=float) - self.camera_center) @ self.R.T
        if not np.all(np.isfinite(camera)) or np.any(camera[:, 2] <= float(self.config.get("near_clip_m", 0.1))):
            raise ValueError("Cuboid has nonfinite corners or corners behind the camera")
        pixels = camera @ self.K.T
        return pixels[:, :2] / pixels[:, 2:3]

    @staticmethod
    def _envelope(corners: np.ndarray) -> np.ndarray:
        return np.r_[np.min(corners, axis=0), np.max(corners, axis=0)]

    def fit(self, bbox: tuple[float, float, float, float], label: str = "car", track_id=None) -> CuboidFit:
        started = perf_counter()
        box = np.asarray(bbox, dtype=float)
        if box.shape != (4,) or not np.all(np.isfinite(box)) or np.any(box[2:] <= box[:2]):
            raise ValueError("bbox must contain finite x1,y1,x2,y2 with positive extent")
        label = str(label).lower()
        priors = self.config.get("dimension_priors", self.config.get("class_dimensions", {}))
        dimensions = priors.get(label, self.DIMENSION_PRIORS.get(label, self.DIMENSION_PRIORS["car"]))
        if isinstance(dimensions, dict):
            dimensions = [dimensions[key] for key in ("length", "width", "height")]
        dimensions = tuple(float(v) for v in dimensions)
        if len(dimensions) != 3 or not np.all(np.isfinite(dimensions)) or min(dimensions) <= 0:
            raise ValueError("Class dimensions must be positive length,width,height in meters")
        turning = label in ("truck", "bus", "van")
        limit = np.deg2rad(float(self.config.get("truck_yaw_limit_deg" if turning else "yaw_limit_deg", 85.0)))
        if not 0 < limit <= np.pi / 2:
            raise ValueError("Yaw limit must be in (0, 90] degrees")
        max_depth = float(self.config.get("max_depth_m", 500.0))
        near = float(self.config.get("near_clip_m", 0.1))
        if near <= 0 or max_depth <= max(near, 1.0):
            raise ValueError("Depth limits must be positive and increasing")
        image_size = self.geometry.image_size
        truncated = np.zeros(4, dtype=bool)
        if image_size is not None:
            image_width, image_height = image_size
            margin = float(self.config.get("truncation_margin_px", 1.0))
            truncated = np.array([box[0] <= margin, box[1] <= margin,
                                  box[2] >= image_width - 1 - margin,
                                  box[3] >= image_height - 1 - margin])

        def compare_envelope(points: np.ndarray) -> np.ndarray:
            envelope = self._envelope(points)
            # A clipped detection constrains only the visible extent. The
            # actual projected corners always remain uncropped.
            envelope[:2] = np.where(truncated[:2], np.maximum(envelope[:2], box[:2]), envelope[:2])
            envelope[2:] = np.where(truncated[2:], np.minimum(envelope[2:], box[2:]), envelope[2:])
            return envelope

        normal = self.R[:, 1]
        bottom_ray = np.linalg.solve(self.K, [(box[0] + box[2]) / 2, box[3], 1.0])
        denominator = np.dot(normal, bottom_ray)
        diagnostics = {
            "pose_quality": "single_view_prior_constrained", "dimensions_order": "length,width,height_m",
            "yaw_unit": "radians", "yaw_limit_deg": float(np.rad2deg(limit)),
            "dimension_source": "class_prior", "rigid_cuboid": True,
            "limitations": ["bbox_does_not_identify_unique_yaw_or_dimensions",
                            "articulation_not_observable_from_bbox"],
            "smoothed": False,
            "truncation": dict(zip(("left", "top", "right", "bottom"), map(bool, truncated))),
            "bbox_comparison": "edge_censored" if np.any(truncated) else "full_envelope",
        }
        if denominator >= -1e-8:
            return CuboidFit([], float("inf"), (0.0, 0.0, 0.0), 0.0, dimensions, False,
                             {**diagnostics, "rejection_reason": "bbox_bottom_on_or_above_ground_horizon"})
        ground_camera = -self.geometry.camera_height_m / denominator * bottom_ray
        ground_world = self.R.T @ ground_camera + self.camera_center
        initial_depth = max(1.0, ground_world[2] + dimensions[0] / 2)
        scale = np.array([box[2] - box[0], box[3] - box[1]] * 2)
        lower = np.array([-max_depth * 3, np.log(max(near, 0.2)), -limit])
        upper = np.array([max_depth * 3, np.log(max_depth), limit])
        yaw_weight = float(self.config.get("yaw_prior_weight", 0.005 if turning else 0.02))

        def corners_for(state: np.ndarray) -> np.ndarray:
            return self.world_corners((state[0], 0.0, np.exp(state[1])), state[2], dimensions)

        evaluations = 0

        def residual(state: np.ndarray) -> np.ndarray:
            nonlocal evaluations
            evaluations += 1
            camera = (corners_for(state) - self.camera_center) @ self.R.T
            # Keep optimization gradients outside the feasible region; the
            # final projection rejects every nonpositive-depth solution.
            pixels = camera @ self.K.T
            points = pixels[:, :2] / np.maximum(camera[:, 2:3], near)
            envelope_error = (compare_envelope(points) - box) / scale
            behind = np.maximum(near - camera[:, 2], 0) * 10
            return np.r_[envelope_error, behind, yaw_weight * state[2] / limit]

        solver = str(self.config.get("fit_solver", "auto"))
        if solver not in ("auto", "scipy", "numpy"):
            raise ValueError("fit_solver must be auto, scipy, or numpy")
        least_squares = None
        if solver != "numpy":
            try:
                from scipy.optimize import least_squares
            except ImportError:
                if solver == "scipy":
                    raise
        diagnostics["solver"] = "scipy" if least_squares is not None else "numpy"
        iterations = max(1, int(self.config.get("fit_max_nfev", 10)))
        warm_iterations = max(1, int(self.config.get("warm_start_max_nfev", 10)))
        start_count = max(1, int(self.config.get("truck_fit_starts" if turning else "fit_starts", 5 if turning else 3)))
        starts = [np.array([ground_world[0], np.log(initial_depth), yaw])
                  for yaw in np.linspace(-limit * 0.95, limit * 0.95, start_count)]
        previous = self._states.get(track_id) if track_id is not None else None
        if previous is not None and previous[0] != label:
            previous = None
        count = self._track_fit_counts.get(track_id, 0) + 1 if track_id is not None else 1
        recovery_interval = max(1, int(self.config.get("global_refit_interval", 30)))
        periodic_recovery = previous is not None and count % recovery_interval == 0
        threshold = float(self.config.get("max_fit_error_ratio", 0.25))
        best, best_score = None, float("inf")
        attempted, attempted_score = None, float("inf")

        def remember(state: np.ndarray, valid_state: bool) -> None:
            if track_id is None:
                return
            self._states[track_id] = (label, state.copy())
            self._track_fit_counts[track_id] = count
            self._state_valid[track_id] = valid_state
            while len(self._states) > max(0, int(self.config.get("max_track_states", 256))):
                expired = next(iter(self._states))
                del self._states[expired]
                self._track_fit_counts.pop(expired, None)
                self._state_valid.pop(expired, None)

        def try_start(start: np.ndarray, budget: int) -> None:
            nonlocal best, best_score, attempted, attempted_score
            start = np.clip(start, lower, upper)
            if least_squares is not None:
                solution = least_squares(residual, start, bounds=(lower, upper), max_nfev=budget,
                                         loss="soft_l1", f_scale=0.1)
                state = solution.x
            else:
                state = _bounded_least_squares(residual, start, lower, upper, budget)
            values = residual(state)
            score = float(np.dot(values, values))
            if np.isfinite(score) and score < attempted_score:
                attempted, attempted_score = state, score
            try:
                self.project(corners_for(state))
            except ValueError:
                return
            if score < best_score:
                best, best_score = state, score

        if previous is not None:
            try_start(previous[1], warm_iterations)
        warm_error = (float("inf") if best is None else
                      float(np.sqrt(np.mean(((compare_envelope(self.project(corners_for(best))) - box) / scale) ** 2))))
        global_search = (previous is None or periodic_recovery
                         or (self.config.get("recover_on_rejection", False) and warm_error > threshold))
        if global_search:
            for start in starts:
                try_start(start, iterations)
        diagnostics["search_mode"] = "global_recovery" if previous is not None and global_search else "global_initial" if global_search else "warm_start"
        diagnostics["residual_evaluations"] = evaluations
        if best is None:
            if attempted is not None:
                remember(attempted, False)
            return CuboidFit([], float("inf"), (0.0, 0.0, 0.0), 0.0, dimensions, False,
                             {**diagnostics, "rejection_reason": "no_solution_in_front_of_camera"})
        points = self.project(corners_for(best))
        normalized_error = float(np.sqrt(np.mean(((compare_envelope(points) - box) / scale) ** 2)))
        valid = normalized_error <= threshold
        if valid and previous is not None and self._state_valid.get(track_id, False):
            alpha = float(np.clip(self.config.get("track_smoothing", 0.35), 0, 1))
            smoothed = alpha * best + (1 - alpha) * previous[1]
            try:
                smoothed_points = self.project(corners_for(smoothed))
                smoothed_error = np.sqrt(np.mean(((compare_envelope(smoothed_points) - box) / scale) ** 2))
                if smoothed_error <= threshold:
                    best, points = smoothed, smoothed_points
                    diagnostics["smoothed"] = True
            except ValueError:
                pass
        remember(best, valid)
        fit_error = float(np.sqrt(np.mean((compare_envelope(points) - box) ** 2)))
        diagnostics["normalized_fit_error"] = float(np.sqrt(np.mean(((compare_envelope(points) - box) / scale) ** 2)))
        diagnostics["projected_envelope"] = self._envelope(points).tolist()
        diagnostics["yaw_at_limit"] = bool(abs(best[2]) >= limit * 0.99)
        if self.config.get("fit_timing", False):
            diagnostics["fit_time_ms"] = (perf_counter() - started) * 1000
        if not valid:
            diagnostics["rejection_reason"] = "bbox_envelope_inconsistent_with_camera_and_class_prior"
        return CuboidFit([tuple(float(v) for v in point) for point in points], fit_error,
                         (float(best[0]), 0.0, float(np.exp(best[1]))), float(best[2]), dimensions, valid, diagnostics)

    def wireframe(self, bbox: tuple[float, float, float, float]) -> list[Point]:
        """Legacy integer corners; return no wireframe for an infeasible fit."""
        result = self.fit(bbox)
        return [(int(round(x)), int(round(y))) for x, y in result.corners] if result.valid else []
