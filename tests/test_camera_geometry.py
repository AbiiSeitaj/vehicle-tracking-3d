from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from src.cv3dbox.geometry import (
    CuboidFit,
    CuboidProjector,
    GeometryEstimator,
    SceneGeometry,
)


def known_camera(*, yaw_deg=0.0, roll_deg=0.0, pitch_deg=8.0, focal=900.0):
    """Analytic right-handed world-to-camera rotation, independent of estimator."""
    yaw, roll, pitch = np.deg2rad([yaw_deg, roll_deg, pitch_deg])
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    rotation = rz @ rx @ ry @ np.diag([-1.0, -1.0, 1.0])
    K = np.array([[focal, 0, 640], [0, focal, 360], [0, 0, 1.0]])
    vp = K @ rotation[:, 2]
    vp /= vp[2]
    horizon = np.linalg.solve(K.T, rotation[:, 1])
    horizon /= horizon[1]
    return SceneGeometry(
        tuple(vp[:2]), -(horizon[0] * 640 + horizon[2]), 1.0, "known-camera",
        image_size=(1280, 720), K=K, R=rotation, ground_normal=rotation[:, 1],
        camera_height_m=2.5, horizon_line=tuple(horizon), calibration_mode="known_camera",
    )


def bbox_of(corners):
    return tuple(np.r_[np.min(corners, axis=0), np.max(corners, axis=0)])


def vanishing_point_of_edges(corners, edges):
    a, b = edges
    homogeneous = np.c_[corners, np.ones(8)]
    line_a = np.cross(homogeneous[a[0]], homogeneous[a[1]])
    line_b = np.cross(homogeneous[b[0]], homogeneous[b[1]])
    vp = np.cross(line_a, line_b)
    return vp / np.linalg.norm(vp)


@pytest.mark.parametrize("yaw_deg", [0.0, 27.0, -65.0])
def test_eight_corners_use_one_pinhole_camera_and_parallel_edge_vps(yaw_deg):
    scene = known_camera(yaw_deg=19, roll_deg=3)
    projector = CuboidProjector(scene, {})
    yaw = np.deg2rad(yaw_deg)
    world = projector.world_corners((1.5, 0, 35), yaw, (12, 2.5, 3.8))
    points = projector.project(world)
    camera = (world - np.array([0, 2.5, 0])) @ scene.R.T
    independent = (scene.K @ camera.T).T
    np.testing.assert_allclose(points, independent[:, :2] / independent[:, 2:3])
    assert points.shape == (8, 2)
    np.testing.assert_allclose(world[[0, 1, 4, 5], 1], 0)
    assert np.all(camera[:, 2] > 0)
    directions = [
        (np.array([np.sin(yaw), 0, np.cos(yaw)]), [(0, 4), (3, 7)]),
        (np.array([np.cos(yaw), 0, -np.sin(yaw)]), [(0, 1), (7, 6)]),
        (np.array([0, 1, 0]), [(0, 3), (5, 6)]),
    ]
    for direction, edges in directions:
        observed = vanishing_point_of_edges(points, edges)
        expected = scene.K @ scene.R @ direction
        expected /= np.linalg.norm(expected)
        assert abs(np.dot(observed, expected)) == pytest.approx(1, abs=1e-10)


@pytest.mark.parametrize("solver", ["scipy", "numpy"])
@pytest.mark.parametrize("label,yaw_deg", [("car", 0), ("car", 22), ("truck", 65), ("truck", -78)])
def test_bbox_fit_recovers_projected_envelope_and_consistent_pose(solver, label, yaw_deg):
    if solver == "scipy":
        pytest.importorskip("scipy.optimize")
    projector = CuboidProjector(known_camera(), {"fit_solver": solver, "yaw_prior_weight": 0.0})
    dimensions = projector.DIMENSION_PRIORS[label]
    world = projector.world_corners((-1, 0, 30), np.deg2rad(yaw_deg), dimensions)
    bbox = bbox_of(projector.project(world))
    result = projector.fit(bbox, label=label)
    assert isinstance(result, CuboidFit)
    assert result.valid
    assert result.fit_error < 0.5
    assert result.dimensions == dimensions
    assert result.position[1] == 0
    assert len(result.corners) == 8
    assert isinstance(result.corners[0][0], float)
    reconstructed = projector.project(projector.world_corners(result.position, result.yaw, result.dimensions))
    np.testing.assert_allclose(result.corners, reconstructed, atol=1e-9)
    np.testing.assert_allclose(bbox_of(reconstructed), bbox, atol=1)
    if label == "truck":
        assert abs(result.yaw) > np.deg2rad(35)
        assert dimensions == (12.0, 2.5, 3.8)
    json.dumps(result.to_dict(), allow_nan=False)


def line_family(vp, count, seed):
    rng = np.random.default_rng(seed)
    anchors = np.c_[rng.uniform(30, 1250, count), rng.uniform(400, 700, count)]
    vp = np.asarray(vp)
    return np.c_[vp + 0.5 * (anchors - vp), anchors]


def test_ransac_support_rejects_outliers_and_duplicate_frame_evidence(monkeypatch):
    rng = np.random.default_rng(8)
    target = np.array([675.0, 240.0])
    family = line_family(target, 20, 3)
    family += rng.normal(0, 0.3, family.shape)
    outliers = rng.uniform([0, 300, 0, 300], [1280, 720, 1280, 720], (15, 4))
    estimator = GeometryEstimator({"estimate_orthogonal_vp": False})
    monkeypatch.setattr(estimator, "_segments", lambda *_: np.vstack([family, outliers]))
    scene = estimator.estimate((720, 1280, 3))
    assert scene.source == "ransac-line-support"
    np.testing.assert_allclose(scene.vanishing_point, target, atol=8)
    assert scene.diagnostics["road_vp_evidence"]["inlier_count"] >= 16
    assert 0 < scene.confidence <= 0.9
    assert scene.calibration_mode == "one_vp_with_priors"
    monkeypatch.setattr(estimator, "_segments", lambda *_: np.tile(np.vstack([family, outliers]), (20, 1)))
    repeated = estimator.estimate((720, 1280, 3))
    assert repeated.confidence == pytest.approx(scene.confidence)
    assert repeated.diagnostics["unique_line_count"] == scene.diagnostics["unique_line_count"]
    np.testing.assert_allclose(repeated.vanishing_point, scene.vanishing_point)


def test_two_supported_ground_vps_recover_focal_and_horizon(monkeypatch):
    truth = known_camera(yaw_deg=27, roll_deg=2, focal=900)
    road = truth.K @ truth.R[:, 2]
    road /= road[2]
    transverse = truth.K @ truth.R[:, 0]
    transverse /= transverse[2]
    segments = np.vstack([line_family(road[:2], 26, 1), line_family(transverse[:2], 16, 2)])
    estimator = GeometryEstimator({"vp_inlier_angle_deg": 0.5})
    monkeypatch.setattr(estimator, "_segments", lambda *_: segments)
    result = estimator.estimate((720, 1280, 3))
    assert result.calibration_mode == "two_vp_assumed_orthogonal"
    np.testing.assert_allclose(result.vanishing_point, road[:2], atol=1e-6)
    np.testing.assert_allclose(result.secondary_vanishing_point, transverse[:2], atol=1e-5)
    assert result.K[0, 0] == pytest.approx(900, abs=1e-5)
    np.testing.assert_allclose(result.R, truth.R, atol=1e-7)
    np.testing.assert_allclose(result.horizon_line, truth.horizon_line, atol=1e-6)
    assert result.assumptions["orthogonal_ground_directions"] is True
    assert "assumed" in result.diagnostics["secondary_vp_interpretation"]


def test_prior_only_is_not_reported_as_vp_evidence_and_is_serializable():
    scene = GeometryEstimator({"horizontal_fov_deg": 70, "camera_height_m": 2.8}).estimate((720, 1280, 3))
    assert scene.confidence == 0
    assert scene.source == "fallback"
    assert scene.calibration_mode == "prior_only"
    assert scene.K[0, 0] == pytest.approx(640 / np.tan(np.deg2rad(35)))
    assert scene.K[0, 0] == scene.K[1, 1]
    assert scene.assumptions["focal_length_source"] == "horizontal_fov_prior"
    assert scene.assumptions["horizon_prior_y_px"] == pytest.approx(259.2)
    np.testing.assert_allclose(scene.R.T @ scene.R, np.eye(3), atol=1e-12)
    assert np.linalg.det(scene.R) == pytest.approx(1)
    road = scene.K @ scene.R[:, 2]
    np.testing.assert_allclose(road[:2] / road[2], scene.vanishing_point)
    assert np.dot(scene.horizon_line, [*scene.vanishing_point, 1]) == pytest.approx(0, abs=1e-9)
    plane = scene.to_dict()["ground_plane"]
    camera_ground = (np.array([1.0, 0, 20]) - [0, 2.8, 0]) @ scene.R.T
    assert np.dot(plane["camera_normal"], camera_ground) + plane["camera_offset_m"] == pytest.approx(0)
    json.dumps(scene.to_dict(), allow_nan=False)


def test_one_vp_conditions_horizon_prior_without_claiming_observed_horizon(monkeypatch):
    vp = (770, 205)
    estimator = GeometryEstimator({"estimate_orthogonal_vp": False, "fallback_horizon_y": 0.36})
    monkeypatch.setattr(estimator, "_segments", lambda *_: line_family(vp, 20, 7))
    scene = estimator.estimate((720, 1280, 3))
    assert scene.assumptions["horizon_source"] == "horizontal_horizon_prior_conditioned_on_road_vp"
    assert scene.assumptions["horizon_prior_y_px"] == pytest.approx(259.2)
    assert abs(scene.diagnostics["horizon_prior_adjustment_px"]) > 1
    assert np.dot(scene.horizon_line, [*vp, 1]) == pytest.approx(0, abs=1e-8)


def test_parallel_segments_do_not_create_finite_vp(monkeypatch):
    estimator = GeometryEstimator({})
    segments = np.array([[100, y, 1000, y] for y in range(300, 650, 30)], dtype=float)
    monkeypatch.setattr(estimator, "_segments", lambda *_: segments)
    scene = estimator.estimate((720, 1280, 3))
    assert scene.confidence == 0
    assert scene.calibration_mode == "prior_only"


def test_real_line_extraction_and_sample_limit():
    frame = np.zeros((720, 1280, 3), np.uint8)
    for segment in line_family((640, 230), 12, 5):
        cv2.line(frame, tuple(segment[:2].astype(int)), tuple(segment[2:].astype(int)), (255, 255, 255), 2)
    estimator = GeometryEstimator({"line_sample_frames": 2, "estimate_orthogonal_vp": False})
    for _ in range(4):
        estimator.add_sample(frame)
    assert len(estimator.samples) == 2
    frame[:] = 0
    assert np.any(estimator.samples[0])
    scene = estimator.estimate((720, 1280, 3))
    assert scene.confidence > 0
    np.testing.assert_allclose(scene.vanishing_point, [640, 230], atol=10)


def test_behind_camera_and_impossible_ground_contact_rejected():
    projector = CuboidProjector(known_camera(), {})
    with pytest.raises(ValueError, match="behind"):
        projector.project(projector.world_corners((0, 0, 1), 0, (12, 2.5, 3.8)))
    result = projector.fit((100, 20, 200, 100), label="truck", track_id=9)
    assert not result.valid
    assert result.corners == []
    assert 9 not in projector._states
    assert projector.wireframe((100, 20, 200, 100)) == []
    assert result.to_dict()["fit_error"] is None
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("bbox", [(2, 2, 1, 3), (1, 1, 3, 1), (0, 0, np.nan, 2)])
def test_malformed_boxes_raise(bbox):
    with pytest.raises(ValueError, match="bbox"):
        CuboidProjector(known_camera(), {}).fit(bbox)


def test_track_smoothing_operates_on_pose_then_reprojects_and_is_per_track():
    projector = CuboidProjector(known_camera(), {"track_smoothing": 0.25, "yaw_prior_weight": 0})
    dims = (12, 2.5, 3.8)
    first_box = bbox_of(projector.project(projector.world_corners((-1, 0, 32), 0.7, dims)))
    next_box = bbox_of(projector.project(projector.world_corners((-0.8, 0, 31.5), 0.72, dims)))
    first = projector.fit(first_box, "truck", track_id=7)
    other = projector.fit(next_box, "truck", track_id=8)
    smoothed = projector.fit(next_box, "truck", track_id=7)
    assert first.valid and other.valid and smoothed.valid
    assert not other.diagnostics["smoothed"]
    assert smoothed.diagnostics["smoothed"]
    assert np.linalg.norm(np.asarray(smoothed.position) - first.position) < np.linalg.norm(np.asarray(other.position) - first.position)
    np.testing.assert_allclose(smoothed.corners, projector.project(
        projector.world_corners(smoothed.position, smoothed.yaw, smoothed.dimensions)))
    error = np.sqrt(np.mean((np.asarray(bbox_of(np.asarray(smoothed.corners))) - next_box) ** 2))
    assert smoothed.fit_error == pytest.approx(error)


def test_configured_class_dimensions_and_legacy_four_field_constructor():
    scene = SceneGeometry((640, 250), 250, 0.5, "legacy")
    projector = CuboidProjector(scene, {
        "image_size": [1280, 720], "class_dimensions": {"truck": {"length": 13, "width": 2.6, "height": 4}},
    })
    bbox = bbox_of(projector.project(projector.world_corners((1, 0, 35), 0.9, (13, 2.6, 4))))
    result = projector.fit(bbox, "truck")
    assert result.valid
    assert result.dimensions == (13, 2.6, 4)
    assert projector.geometry.image_size == (1280, 720)
    assert projector.geometry.diagnostics["legacy_camera"] == "configured_image_size"


def test_semantic_height_horizon_rejects_building_line_ground_membership(monkeypatch):
    estimator = GeometryEstimator({"horizontal_fov_deg": 60})
    for bottom, pixel_height, label in [(218, 37, "car"), (273, 180, "truck"), (230, 45, "car")]:
        estimator.add_detection((400, bottom - pixel_height, 550, bottom), label)
    # Unknown classes, generic vehicles, and truncated detections do not vote.
    estimator.add_detection((0, 100, 500, 300), "truck")
    estimator.add_detection((400, 150, 550, 250), "person")
    estimator.add_detection((400, 150, 550, 250), "vehicle")
    building = (300, 272)
    center = np.array([640, 360])
    secondary_x = 2200.0
    secondary_y = center[1] + (-900 ** 2 - (building[0] - center[0]) * (secondary_x - center[0])) / (building[1] - center[1])
    lines = np.vstack([line_family(building, 24, 2), line_family((secondary_x, secondary_y), 12, 3)])
    monkeypatch.setattr(estimator, "_segments", lambda *_: lines)
    result = estimator.estimate((720, 1280, 3))
    expected = np.median([218 - 2.5 * 37 / 1.5, 273 - 2.5 * 180 / 3.8, 230 - 2.5 * 45 / 1.5])
    assert result.horizon_y == pytest.approx(expected)
    assert result.source == "semantics_with_fov_prior"
    assert result.calibration_mode == "semantics_with_fov_prior"
    assert result.confidence == 0
    assert result.secondary_vanishing_point is None
    assert result.assumptions["horizon_source"] == "semantic_vehicle_height_prior"
    assert result.assumptions["semantic_horizon_is_metric_calibration"] is False
    assert result.assumptions["focal_length_source"] == "horizontal_fov_prior"
    assert result.diagnostics["road_vp_ground_membership_rejected"]
    assert result.diagnostics["semantic_horizon_evidence"]["accepted_box_count"] == 3
    assert result.diagnostics["observed_line_vanishing_point"] is not None
    json.dumps(result.to_dict(), allow_nan=False)


def test_semantic_horizon_robust_median_and_per_track_votes(monkeypatch):
    estimator = GeometryEstimator({})
    for track_id, horizon in [(1, 155), (2, 160), (3, 150), (4, 300)]:
        bottom = horizon + 2.5 * 40 / 1.5
        estimator.add_detections([{"bbox": (400, bottom - 40, 500, bottom), "label": "car", "track_id": track_id}])
    for _ in range(20):
        estimator.add_detection((400, 330, 500, 370), "car", track_id=4)
    estimator.add_detection((0, 0, float("nan"), 10), "car")
    monkeypatch.setattr(estimator, "_segments", lambda *_: np.empty((0, 4)))
    result = estimator.estimate((720, 1280, 3))
    assert result.horizon_y == pytest.approx(155)
    evidence = result.diagnostics["semantic_horizon_evidence"]
    assert evidence["vote_count"] == 4
    assert evidence["inlier_vote_count"] == 3
    assert "footprint" in evidence["interpretation"]


def test_too_few_semantic_samples_retain_explicit_fallback(monkeypatch):
    estimator = GeometryEstimator({})
    estimator.add_detection((400, 180, 500, 220), "car")
    monkeypatch.setattr(estimator, "_segments", lambda *_: np.empty((0, 4)))
    result = estimator.estimate((720, 1280, 3))
    assert result.calibration_mode == "prior_only"
    assert not result.diagnostics["semantic_horizon_evidence"]["accepted"]


@pytest.mark.parametrize("position,edge", [((17, 0, 24), "left"), ((-17, 0, 24), "right")])
def test_truncated_envelope_keeps_uncropped_pinhole_corners(position, edge):
    scene = GeometryEstimator({"fallback_horizon_y": 0.22}).estimate((720, 1280, 3))
    projector = CuboidProjector(scene, {"fit_max_nfev": 25, "yaw_prior_weight": 0.0})
    original = projector.project(projector.world_corners(position, np.deg2rad(70), (12, 2.5, 3.8)))
    original_box = np.asarray(bbox_of(original))
    clipped_box = np.clip(original_box, [0, 0, 0, 0], [1279, 719, 1279, 719])
    result = projector.fit(clipped_box, "truck")
    assert result.valid
    assert result.diagnostics["truncation"][edge]
    assert result.diagnostics["bbox_comparison"] == "edge_censored"
    assert result.fit_error < 1
    points = np.asarray(result.corners)
    assert np.any(points[:, 0] < 0) if edge == "left" else np.any(points[:, 0] > 1279)
    np.testing.assert_allclose(points, projector.project(projector.world_corners(result.position, result.yaw, result.dimensions)))


def test_warm_start_budget_and_periodic_recovery(monkeypatch):
    optimize = pytest.importorskip("scipy.optimize")
    original = optimize.least_squares
    budgets = []

    def record(*args, **kwargs):
        budgets.append(kwargs["max_nfev"])
        return original(*args, **kwargs)

    monkeypatch.setattr(optimize, "least_squares", record)
    projector = CuboidProjector(known_camera(), {"warm_start_max_nfev": 8, "global_refit_interval": 3})
    bbox = bbox_of(projector.project(projector.world_corners((0, 0, 30), 0.1, (4.5, 1.8, 1.5))))
    projector.fit(bbox, "car", 1)
    assert budgets == [10, 10, 10]
    budgets.clear()
    second = projector.fit(bbox, "car", 1)
    assert budgets == [8]
    assert second.diagnostics["search_mode"] == "warm_start"
    budgets.clear()
    third = projector.fit(bbox, "car", 1)
    assert budgets == [8, 10, 10, 10]
    assert third.diagnostics["search_mode"] == "global_recovery"


def test_rejected_fit_is_cached_without_global_retry_every_frame():
    projector = CuboidProjector(known_camera(), {"max_fit_error_ratio": 1e-8, "fit_timing": True})
    bbox = (400, 300, 440, 600)
    first = projector.fit(bbox, "car", 11)
    assert not first.valid
    assert 11 in projector._states
    second = projector.fit(bbox, "car", 11)
    assert not second.valid
    assert second.diagnostics["search_mode"] == "warm_start"
    assert projector._track_fit_counts[11] == 2
    assert second.diagnostics["fit_time_ms"] > 0
    assert not second.diagnostics["smoothed"]


def test_cars_can_turn_beyond_previous_road_yaw_limit():
    projector = CuboidProjector(known_camera(), {"yaw_prior_weight": 0, "fit_max_nfev": 25})
    bbox = bbox_of(projector.project(projector.world_corners((0, 0, 30), np.deg2rad(65), (4.5, 1.8, 1.5))))
    result = projector.fit(bbox, "car")
    assert result.valid
    assert result.fit_error < 1
    assert abs(result.yaw) > np.deg2rad(35)
    assert result.diagnostics["yaw_limit_deg"] == pytest.approx(85)
