import cv2
import numpy as np

from src.cv3dbox.pipeline import _encode_frame, _encoder_size, _make_writer


def test_odd_size_padding_preserves_image_coordinates():
    frame = np.arange(7 * 9 * 3, dtype=np.uint8).reshape(7, 9, 3)
    padded = _encode_frame(frame)
    assert _encoder_size(frame.shape) == (10, 8)
    assert padded.shape == (8, 10, 3)
    np.testing.assert_array_equal(padded[:7, :9], frame)
    assert not padded[7].any()
    assert not padded[:, 9].any()


def test_even_frame_needs_no_copy():
    frame = np.zeros((8, 10, 3), dtype=np.uint8)
    assert _encode_frame(frame) is frame
    assert _encoder_size(frame.shape) == (10, 8)


def test_writer_retains_padded_dimensions(tmp_path):
    frame = np.full((31, 33, 3), 160, dtype=np.uint8)
    path = tmp_path / "odd_dimensions.mp4"
    writer = _make_writer(path, frame.shape, 15)
    try:
        writer.write(_encode_frame(frame))
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(path))
    try:
        ok, decoded = capture.read()
        assert ok
        assert decoded.shape == (32, 34, 3)
        assert not capture.read()[0]
    finally:
        capture.release()
