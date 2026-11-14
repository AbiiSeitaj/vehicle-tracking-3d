import numpy as np

from src.cv3dbox.detectors import Detection
from src.cv3dbox.tracker import MultiObjectTracker


def test_identity_survives_short_occlusion_without_rendering_ghosts():
    tracker = MultiObjectTracker({"track_buffer": 10})
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    first = tracker.update([Detection((50, 60, 140, 130), .9, "car")], frame)
    assert len(first) == 1
    identity = first[0].track_id
    tracker.update([Detection((52, 60, 142, 130), .9, "car")], frame)
    assert tracker.update([], frame) == []
    assert tracker.update([], frame) == []
    restored = tracker.update([Detection((55, 60, 145, 130), .9, "car")], frame)
    assert len(restored) == 1 and restored[0].track_id == identity


def test_low_confidence_match_preserves_existing_identity():
    tracker = MultiObjectTracker({})
    first = tracker.update([Detection((50, 60, 140, 130), .9, "car")])
    matched = tracker.update([Detection((52, 60, 142, 130), .2, "car")])
    assert len(matched) == 1 and matched[0].track_id == first[0].track_id


def test_expired_track_does_not_reuse_identity():
    tracker = MultiObjectTracker({"track_buffer": 2})
    first = tracker.update([Detection((50, 60, 140, 130), .9, "car")])
    for _ in range(5):
        assert tracker.update([]) == []
    tracker.update([Detection((50, 60, 140, 130), .9, "car")])
    restored = tracker.update([Detection((50, 60, 140, 130), .9, "car")])
    assert len(restored) == 1 and restored[0].track_id != first[0].track_id
