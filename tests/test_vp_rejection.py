import numpy as np
from src.cv3dbox.geometry import GeometryEstimator


def estimator_with_pair(monkeypatch, second_support):
    estimator = GeometryEstimator({})
    segments = np.array([[10, 600, 200, 400], [500, 500, 700, 300], [900, 600, 1100, 700]])
    monkeypatch.setattr(estimator, "_segments", lambda *args: segments)
    monkeypatch.setattr(estimator, "_unique_segments", lambda values: values)
    results = iter([
        ((698, 718), np.array([True, True, False]), {"weighted_support": .7, "inlier_count": 12}),
        ((17, 19), np.array([True]), {"weighted_support": second_support, "inlier_count": 5}),
    ])
    monkeypatch.setattr(estimator, "_ransac", lambda *args, **kwargs: next(results))
    return estimator


def test_weak_secondary_family_cannot_define_ground_horizon(monkeypatch):
    geometry = estimator_with_pair(monkeypatch, .09).estimate((1047, 1280, 3))
    assert geometry.secondary_vanishing_point is None
    assert geometry.diagnostics["line_pair_rejected"] == "insufficient_secondary_line_support"


def test_strong_but_implausibly_rolled_pair_uses_explicit_prior(monkeypatch):
    geometry = estimator_with_pair(monkeypatch, .9).estimate((1047, 1280, 3))
    assert geometry.secondary_vanishing_point is None
    assert geometry.diagnostics["line_pair_rejected"] == "exceeds_camera_roll_prior"
