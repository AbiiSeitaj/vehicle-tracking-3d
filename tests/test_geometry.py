from __future__ import annotations

from src.cv3dbox.geometry import CuboidProjector, SceneGeometry


def test_wireframe_preserves_integer_corner_api() -> None:
    geometry = SceneGeometry(vanishing_point=(640.0, 250.0), horizon_y=250.0, confidence=0.5, source="test")
    projector = CuboidProjector(geometry, {"image_size": [1280, 720]})
    bbox = (420.0, 330.0, 600.0, 470.0)
    points = projector.wireframe(bbox)
    fit = projector.fit(bbox)

    assert len(points) == 8
    assert fit.valid
    assert all(isinstance(value, int) for point in points for value in point)
    assert points == [(round(x), round(y)) for x, y in fit.corners]
