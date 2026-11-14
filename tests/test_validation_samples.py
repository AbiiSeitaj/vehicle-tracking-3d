from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.download_validation_samples import download, convert_y4m
from scripts.run_validation import run_validation, validation_plan
from scripts.export_annotation_frames import export_frames


class Response(io.BytesIO):
    status = 200

    def geturl(self):
        return "https://example.org/resolved.y4m"


def test_download_verifies_digest_and_reuses_verified_cache(tmp_path):
    data = b"YUV4MPEG2 test\n"
    checksum = hashlib.sha256(data).hexdigest()
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        return Response(data)

    output = tmp_path / "test.y4m"
    info = download("https://example.org/test.y4m", output, expected_bytes=len(data),
                    expected_sha256=checksum, opener=opener)
    assert info["resolved_url"].endswith("resolved.y4m")
    assert output.read_bytes() == data
    cached = download("https://example.org/test.y4m", output, expected_bytes=len(data),
                      expected_sha256=checksum, opener=opener)
    assert cached["reused_verified_file"]
    assert len(calls) == 1


@pytest.mark.parametrize("options,message", [
    ({"expected_bytes": 999}, "expected 999"),
    ({"expected_sha256": "0" * 64}, "SHA-256"),
    ({"max_bytes": 3}, "download limit"),
])
def test_failed_download_preserves_existing_file_and_removes_partial(tmp_path, options, message):
    output = tmp_path / "test.y4m"
    output.write_bytes(b"existing")
    with pytest.raises(ValueError, match=message):
        download("https://example.org/test.y4m", output,
                 opener=lambda *a, **kw: Response(b"new contents"), **options)
    assert output.read_bytes() == b"existing"
    assert not list(tmp_path.glob("*.part"))


def small_video(path: Path, count: int):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    for index in range(count):
        writer.write(np.full((48, 64, 3), index * 30, dtype=np.uint8))
    writer.release()


def test_full_challenge_plan_and_explicit_real_excerpt():
    plan = validation_plan(Path("samples"), 300)
    assert [run["name"] for run in plan] == ["challenge_full", "bus", "highway"]
    assert plan[0]["limit"] is None
    assert [run["limit"] for run in plan[1:]] == [300, 300]


def test_runner_reports_requested_and_actual_source_frame_ranges(tmp_path):
    for name, count in [("bus", 3), ("highway", 5)]:
        small_video(tmp_path / f"{name}.mp4", count)
        (tmp_path / f"{name}.provenance.json").write_text("{}")
    calls = []

    def pipeline(**kwargs):
        calls.append(kwargs)
        return {"frames_processed": kwargs["max_frames"], "detector_backend": "test"}

    report = run_validation(Path("config"), tmp_path, tmp_path / "runs", 4, True, pipeline)
    assert [call["max_frames"] for call in calls] == [3, 4]
    assert report["runs"][0]["excerpt"] is False
    assert report["runs"][1]["excerpt"] is True
    assert report["runs"][1]["source_frame_range_inclusive"] == [0, 3]
    assert json.loads((tmp_path / "runs/validation_report.json").read_text())["runs"] == report["runs"]


def test_y4m_conversion_preserves_frame_count_and_dimensions(tmp_path):
    source = tmp_path / "sample.y4m"
    # This is a codec/transport test fixture, not a recognition benchmark.
    with source.open("wb") as stream:
        stream.write(b"YUV4MPEG2 W64 H48 F10:1 Ip A1:1 C420jpeg\n")
        for value in (50, 90, 130):
            stream.write(b"FRAME\n" + bytes([value]) * (64 * 48) + bytes([128]) * (64 * 48 // 2))
    result = convert_y4m(source, tmp_path / "sample.mp4", 3)
    assert (result["frames"], result["width"], result["height"], result["fps"]) == (3, 64, 48, 10)
    assert len(result["sha256"]) == 64
    with pytest.raises(ValueError, match="decoded 3"):
        convert_y4m(source, tmp_path / "bad.mp4", 4)
    assert not (tmp_path / "bad.mp4").exists()
    assert not list(tmp_path.glob("*.partial.mp4"))


def test_annotation_extraction_reads_sequentially_and_never_seeks(tmp_path, monkeypatch):
    class SequentialCapture:
        index = 0

        def isOpened(self):
            return True

        def read(self):
            if self.index == 8:
                return False, None
            result = np.full((48, 64, 3), self.index * 20, dtype=np.uint8)
            self.index += 1
            return True, result

        def set(self, *args):
            pytest.fail("Frame-index seeking is invalid for variable-timestamp sources")

        def release(self):
            pass

    video = tmp_path / "source.mp4"
    video.write_bytes(b"source fixture")
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: SequentialCapture())
    report = export_frames(video, [7, 0, 3], tmp_path / "frames", width=1280)
    assert report["image_size"] == [1280, 960]
    assert [entry["frame_index"] for entry in report["frames"]] == [0, 3, 7]
    for entry in report["frames"]:
        image = cv2.imread(entry["path"])
        assert image.mean() == pytest.approx(entry["frame_index"] * 20, abs=1)


def test_manual_annotation_files_have_disclosed_selection_and_coordinates():
    from scripts.evaluate import evaluate

    root = Path(__file__).resolve().parents[1] / "scripts/validation_samples"
    expected = {"bus": (10, [1280, 1047]),
                "highway": (10, [1280, 1047]), "highway_extended": (10, [1280, 1047])}
    for name, (count, image_size) in expected.items():
        annotations = json.loads((root / f"{name}_manual_annotations.json").read_text())
        assert len(annotations["frames"]) == count
        assert annotations["image_size"] == image_size
        assert annotations["selection"]["min_height_px"] == 35
        assert annotations["exhaustive_visible_selected_objects"] is True
        assert annotations["annotation_provenance"]["limitations"]
        evaluate([], annotations)
        width, height = image_size
        for record in annotations["frames"]:
            for obj in record["objects"]:
                x1, y1, x2, y2 = obj["bbox"]
                assert 0 <= x1 < x2 <= width
                assert 0 <= y1 < y2 <= height
                assert y2 - y1 >= 35


def test_challenge_is_uncapped_and_metadata_mismatch_is_disclosed(tmp_path, monkeypatch):
    import scripts.run_validation as runner

    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    small_video(tmp_path / "data/challenge_video.mp4", 5)
    for name in ("bus", "highway"):
        small_video(tmp_path / f"{name}.mp4", 3)
        (tmp_path / f"{name}.provenance.json").write_text("{}")
    config = tmp_path / "config.yaml"
    config.write_text("input:\n  max_frames: null\n")
    calls = []

    def pipeline(**kwargs):
        calls.append(kwargs)
        return {"frames_processed": 4 if kwargs["max_frames"] is None else kwargs["max_frames"],
                "complete_sequence": kwargs["max_frames"] is not None,
                "detector_backend": "test"}

    report = runner.run_validation(config, tmp_path, tmp_path / "runs", pipeline=pipeline,
                                   source_counter=lambda path: 4)
    assert calls[0]["max_frames"] is None
    challenge = report["runs"][0]
    assert challenge["source_frame_range_inclusive"] == [0, 3]
    assert challenge["metadata_source_frame_count"] == 5
    assert challenge["decoded_frames"] == 4
    assert challenge["decoded_count_matches_metadata"] is False
    assert challenge["independent_source_decoded_frames"] == 4
    assert challenge["decoded_count_matches_independent_source"] is True
    assert challenge["complete_sequence"] is False
    assert challenge["excerpt"] is False
    with pytest.raises(ValueError, match="independent FFmpeg source decode counted 5"):
        runner.run_validation(config, tmp_path, tmp_path / "mismatch", pipeline=pipeline,
                              source_counter=lambda path: 5)


def test_manual_review_checks_coordinates_hashes_and_renders_gt_only(tmp_path):
    from scripts.review_annotations import review_annotations
    from scripts.download_validation_samples import sha256

    video = tmp_path / "source.mp4"
    small_video(video, 3)
    data = {"image_size": [64, 48], "source_video_sha256": sha256(video), "frames": [
        {"frame_index": 0, "objects": [{"id": "car", "label": "vehicle", "bbox": [10, 10, 30, 46]}]},
        {"frame_index": 2, "objects": []}]}
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps(data))
    output = tmp_path / "review"
    report = review_annotations(video, annotations, output)
    assert report["source_hash_verified"] is True
    assert [f["frame_index"] for f in report["frames"]] == [0, 2]
    assert [f["objects"] for f in report["frames"]] == [1, 0]
    overlay = cv2.imread(report["frames"][0]["overlay_path"])
    assert overlay.shape[:2] == (48, 64)
    assert overlay[25, 10, 1] > 120  # GT rectangle appears in the source coordinate system.
    assert report["annotations_sha256"] == sha256(annotations)
    assert report["frames"][0]["overlay_sha256"] == sha256(Path(report["frames"][0]["overlay_path"]))
    assert report["frames"][0]["source_image_sha256"] != report["frames"][0]["overlay_sha256"]
    data["source_video_sha256"] = "0" * 64
    annotations.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="does not match the input video"):
        review_annotations(video, annotations, tmp_path / "wrong_hash")
    del data["source_video_sha256"]
    data["image_size"] = [64, 49]
    annotations.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="resized sequential source pixels"):
        review_annotations(video, annotations, tmp_path / "wrong_size")
    data["image_size"] = [64, 48]
    data["frames"][0]["objects"][0]["bbox"] = [10, 10, 65, 46]
    annotations.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="exceeds image bounds"):
        review_annotations(video, annotations, tmp_path / "wrong_box")


def test_later_highway_gt_has_observed_identity_evidence_and_separate_scope():
    root = Path(__file__).resolve().parents[1] / "scripts/validation_samples"
    early = json.loads((root / "highway_manual_annotations.json").read_text())
    later = json.loads((root / "highway_extended_manual_annotations.json").read_text())
    assert max(f["frame_index"] for f in early["frames"]) < 300
    assert min(f["frame_index"] for f in later["frames"]) > 300
    observations = [obj for frame in later["frames"] for obj in frame["objects"]]
    assert len(observations) == 10
    assert sum(obj["id"] == "overtaking_car_1" for obj in observations) == 8
    assert any(not frame["objects"] for frame in later["frames"])
    assert "not a random" in later["annotation_provenance"]["limitations"][0]
