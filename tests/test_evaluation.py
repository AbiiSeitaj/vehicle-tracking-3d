from __future__ import annotations

import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.evaluate import evaluate, iou_matrix, load_predictions, match_frame


def obj(identity="a", bbox=(0, 0, 50, 50), label="car", **extra):
    return {"id": identity, "bbox": list(bbox), "label": label, **extra}


def track(identity=1, bbox=(0, 0, 50, 50), label="car", **extra):
    return {"track_id": identity, "bbox": list(bbox), "label": label, **extra}


def detection(bbox=(0, 0, 50, 50), label="car"):
    return {"bbox": list(bbox), "label": label, "score": 0.9}


def frame(index, tracks=(), detections=()):
    return {"frame_index": index, "tracks": list(tracks), "detections": list(detections)}


def gt(*objects_by_frame, exhaustive=True):
    return {"exhaustive_objects": exhaustive,
            "frames": [{"frame_index": i, "objects": list(objects)}
                       for i, objects in enumerate(objects_by_frame)]}


def test_perfect_spatial_and_identity_matching():
    report = evaluate([frame(i, [track()], [detection()]) for i in range(3)],
                      gt([obj()], [obj()], [obj()]))
    assert report["detections"] == {"tp": 3, "fp": 0, "fn": 0, "precision": 1, "recall": 1}
    assert report["tracks"]["tp"] == 3
    assert report["identity"]["idf1"] == 1
    assert report["identity"]["id_switches"] == 0


def test_duplicates_false_positives_missing_objects_and_negative_frame():
    report = evaluate([frame(0, [track(1), track(2)], [detection(), detection()]),
                       frame(1, [track(1)], [detection()]), frame(2)],
                      gt([obj()], [], [obj()]))
    for kind in ("detections", "tracks"):
        assert report[kind]["tp"] == 1
        assert report[kind]["fp"] == 2
        assert report[kind]["fn"] == 1
        assert report[kind]["precision"] == pytest.approx(1 / 3)
        assert report[kind]["recall"] == 0.5


def test_class_agnostic_default_and_explicit_label_matching():
    predictions = [frame(0, [track(label="truck")], [detection(label="truck")])]
    annotations = gt([obj(label="car")])
    assert evaluate(predictions, annotations)["identity"]["idf1"] == 1
    exact = evaluate(predictions, annotations, ignore_labels=False)
    assert exact["tracks"]["tp"] == 0
    assert exact["identity"]["idf1"] == 0


def test_iou_exact_boundary_continuous_coordinates():
    # Overlap is 2500 and union is 5000: exactly 0.5 without +1 convention.
    predictions = [frame(0, [track(bbox=(0, 0, 100, 50))])]
    assert iou_matrix([obj()], predictions[0]["tracks"])[0, 0] == 0.5
    assert evaluate(predictions, gt([obj()]))["tracks"]["tp"] == 1
    assert evaluate(predictions, gt([obj()]), iou_threshold=0.50001)["tracks"]["tp"] == 0


def test_hungarian_prioritizes_count_over_total_iou():
    overlaps = np.array([[0.99, 0.5], [0.5, 0.49]])
    assert match_frame(overlaps, overlaps >= 0.5) == [(0, 1), (1, 0)]


def test_global_idf1_penalizes_fragmentation_despite_perfect_boxes():
    predictions = [frame(i, [track(1 if i < 2 else 2)]) for i in range(4)]
    report = evaluate(predictions, gt(*[[obj()] for _ in range(4)]))
    assert report["tracks"]["recall"] == 1
    assert report["identity"]["id_switches"] == 1
    assert report["identity"]["idtp"] == 2
    assert report["identity"]["idfp"] == report["identity"]["idfn"] == 2
    assert report["identity"]["idf1"] == 0.5


def test_global_identity_uses_all_eligible_pairs_not_local_assignment():
    # Two identical GT boxes create valid ambiguous correspondences. Locally,
    # tracker ordering reverses on frame 1; globally both identities stay valid.
    annotations = gt([obj("a"), obj("b")], [obj("a"), obj("b")])
    predictions = [frame(0, [track(10), track(20)]), frame(1, [track(20), track(10)])]
    report = evaluate(predictions, annotations)
    assert report["identity"]["idtp"] == 4
    assert report["identity"]["idf1"] == 1


def test_global_hungarian_beats_greedy_identity_pair_assignment():
    # Eligible counts [[3,2],[2,0]]; greedy takes 3, global optimum takes 4.
    records, truth = [], []
    cases = [("a", 10)] * 3 + [("a", 20)] * 2 + [("b", 10)] * 2
    for index, (identity, predicted_id) in enumerate(cases):
        records.append(frame(index, [track(predicted_id)]))
        truth.append({"frame_index": index, "objects": [obj(identity)]})
    report = evaluate(records, {"frames": truth, "exhaustive_objects": True})
    assert report["identity"]["idtp"] == 4
    assert report["identity"]["idf1"] == pytest.approx(4 / 7)


def test_switch_after_missed_detection_and_sparse_annotations():
    records = [frame(0, [track(1)]), frame(1), frame(4, [track(2)])]
    truth = {"frames": [{"frame_index": i, "objects": [obj()]} for i in (0, 1, 4)]}
    report = evaluate(records[::-1], truth)
    assert report["identity"]["id_switches"] == 1
    assert report["identity"]["switch_events"][0]["previous_frame_index"] == 0
    assert any("Sparse" in warning for warning in report["limitations"])


def test_unannotated_frames_excluded_but_missing_prediction_frames_scored():
    report = evaluate([frame(0, [track()]), frame(100, [track(999)])], gt([obj()], [obj()]))
    assert report["tracks"]["fn"] == 1
    assert report["tracks"]["fp"] == 0
    assert report["identity"]["idf1"] == pytest.approx(2 / 3)
    assert report["scope"]["excluded_prediction_frames"] == 1
    assert report["scope"]["missing_prediction_frames"] == [1]


def test_partial_object_annotations_are_explicitly_conditional():
    report = evaluate([frame(0, [track()])], gt([obj()], exhaustive=False))
    assert not report["scope"]["metrics_valid_for_selected_objects"]
    assert any("unlabeled true objects" in warning for warning in report["limitations"])


def test_height_cutoff_boundary_stationary_and_truncated_boxes():
    annotations = gt([obj("a", (0, 0, 50, 35)), obj("b", (100, 0, 150, 34)),
                      obj("c", (-10, 50, 25, 100))])
    predictions = [frame(0, [track(1, (0, 0, 50, 35)), track(2, (100, 0, 150, 34)),
                                   track(3, (-10, 50, 25, 100))])]
    report = evaluate(predictions, annotations)
    assert report["tracks"]["tp"] == 2
    assert report["tracks"]["fp"] == 0
    assert report["scope"]["excluded_by_selection"]["objects"] == 1
    assert report["scope"]["excluded_by_selection"]["tracks"] == 1


@pytest.mark.parametrize("records,annotations,expected", [
    ([frame(0)], gt([]), (None, None, None)),
    ([frame(0)], gt([obj()]), (None, 0, 0)),
    ([frame(0, [track()])], gt([]), (0, None, 0)),
])
def test_empty_denominators_are_null(records, annotations, expected):
    report = evaluate(records, annotations)
    assert (report["tracks"]["precision"], report["tracks"]["recall"],
            report["identity"]["idf1"]) == expected
    json.dumps(report, allow_nan=False)


def test_numeric_and_string_ids_remain_distinct():
    report = evaluate([frame(0, [track(1), track("1", (100, 0, 150, 50))])],
                      gt([obj(1), obj("1", (100, 0, 150, 50))]))
    assert report["identity"]["idtp"] == 2


def test_cuboids_do_not_claim_accuracy_without_ground_truth():
    report = evaluate([frame(0, [track(corners=[[0, 0]] * 8, fit_error=0.02)])], gt([obj()]))
    assert report["cuboids"]["corner_accuracy"] is None
    assert report["cuboids"]["mean_reported_fit_error"] == 0.02


def test_corner_reprojection_and_missing_corner_coverage():
    annotations = gt([obj(corners=[[0, 0]] * 8)], [obj(corners=[[0, 0]] * 8)])
    report = evaluate([frame(0, [track(corners=[[3, 4]] * 8)]), frame(1, [track(corners=[])])],
                      annotations, compare_corners=True)
    metric = report["cuboids"]["corner_accuracy"]
    assert metric["mean_error_px"] == metric["rmse_px"] == 5
    assert metric["compared_points"] == 8
    assert metric["matched_objects_missing_prediction_corners"] == 1


@pytest.mark.parametrize("mutation,message", [
    (lambda p: p.append(p[0]), "duplicate frame_index"),
    (lambda p: p[0]["tracks"].append(track()), "duplicate track_id"),
    (lambda p: p[0].update(frame_index=True), "frame_index"),
    (lambda p: p[0]["tracks"][0].update(bbox=[0, 0, 0, 50]), "positive width"),
    (lambda p: p[0]["tracks"][0].update(bbox=[0, 0, float("nan"), 50]), "finite number"),
    (lambda p: p[0]["tracks"][0].update(corners=[[0, 0]] * 7), "eight"),
    (lambda p: p[0]["tracks"][0].update(fit_error=-1), "nonnegative"),
])
def test_invalid_prediction_records_are_rejected(mutation, message):
    records = [frame(0, [track()])]
    mutation(records)
    with pytest.raises(ValueError, match=message):
        evaluate(records, gt([obj()]))


def test_invalid_gt_duplicate_identity_and_empty_annotation_set():
    with pytest.raises(ValueError, match="duplicate id"):
        evaluate([], gt([obj(), obj()]))
    with pytest.raises(ValueError, match="at least one"):
        evaluate([], {"frames": []})


def test_jsonl_line_records_and_array_batches_and_parse_location(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(frame(0)) + "\n\n" + json.dumps([frame(1), frame(2)]) + "\n")
    assert [record["frame_index"] for record in load_predictions(path)] == [0, 1, 2]
    path.write_text(json.dumps(frame(0)) + "\n{" )
    with pytest.raises(ValueError, match="line 2"):
        load_predictions(path)


def test_cli_evaluates_real_schema_and_checks_coordinate_size(tmp_path):
    predictions = tmp_path / "demo.predictions.jsonl"
    annotations = tmp_path / "annotations.json"
    output = tmp_path / "evaluation.json"
    predictions.write_text(json.dumps(frame(0, [track(label="truck")], [detection()])) + "\n")
    annotations.write_text(json.dumps({**gt([obj()]), "image_size": [1280, 720]}))
    (tmp_path / "demo.metrics.json").write_text(json.dumps({"output_size": [1280, 720]}))
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/evaluate.py"),
               "--predictions", str(predictions), "--annotations", str(annotations), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["identity"]["idf1"] == 1
    assert report["scope"]["coordinate_size_verified"]
    assert len(report["inputs"]["annotations_sha256"]) == 64
    (tmp_path / "demo.metrics.json").write_text(json.dumps({"output_size": [640, 360]}))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "does not match" in result.stderr
    (tmp_path / "demo.metrics.json").write_text(json.dumps({"output_size": [1280, 720], "frames_processed": 2}))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "incomplete or stale" in result.stderr


def test_assignment_matches_brute_force_small_adversarial_cases():
    rng = np.random.default_rng(7)
    for shape in [(2, 3), (3, 2), (3, 3)]:
        for _ in range(10):
            overlaps = rng.random(shape)
            eligible = overlaps >= 0.5
            matches = match_frame(overlaps, eligible)
            actual = (len(matches), sum(overlaps[r, c] for r, c in matches))
            candidates = []
            for count in range(min(shape) + 1):
                for rows in itertools.combinations(range(shape[0]), count):
                    for cols in itertools.permutations(range(shape[1]), count):
                        if all(eligible[r, c] for r, c in zip(rows, cols)):
                            candidates.append((count, sum(overlaps[r, c] for r, c in zip(rows, cols))))
            assert actual == pytest.approx(max(candidates))


def test_nonfinite_json_in_unscored_metadata_is_rejected(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"frame_index":0,"detections":[],"tracks":[],"diagnostics":{"value":Infinity}}\n')
    with pytest.raises(ValueError, match="Nonfinite JSON"):
        load_predictions(path)


def test_exhaustive_visible_selection_alias_and_conflicts():
    annotations = gt([obj()])
    del annotations["exhaustive_objects"]
    annotations["exhaustive_visible_selected_objects"] = True
    assert evaluate([], annotations)["scope"]["metrics_valid_for_selected_objects"]
    annotations["exhaustive_objects"] = False
    with pytest.raises(ValueError, match="Conflicting"):
        evaluate([], annotations)


def test_batch_evaluation_keeps_sequence_identities_separate(tmp_path):
    from scripts.evaluate_validation import evaluate_outputs

    real_dir, annotation_dir = tmp_path / "real", tmp_path / "annotations"
    real_dir.mkdir()
    annotation_dir.mkdir()
    challenge = tmp_path / "challenge.predictions.jsonl"
    for name, predictions, size in [
        ("challenge", challenge, [1280, 720]),
        ("bus", real_dir / "validation_bus.predictions.jsonl", [1280, 1047]),
        ("highway", real_dir / "validation_highway.predictions.jsonl", [1280, 1047]),
    ]:
        predictions.write_text(json.dumps(frame(0, [track()], [detection()])) + "\n")
        predictions.with_name(predictions.name.removesuffix(".predictions.jsonl") + ".metrics.json").write_text(
            json.dumps({"output_size": size, "frames_processed": 1}))
        (annotation_dir / f"{name}_manual_annotations.json").write_text(json.dumps({**gt([obj()]), "image_size": size}))
    result = evaluate_outputs(challenge, real_dir, annotation_dir, tmp_path / "evaluation")
    assert [entry["name"] for entry in result["runs"]] == ["challenge", "bus", "highway"]
    assert all(entry["identity"]["idtp"] == 1 for entry in result["runs"])
    assert (tmp_path / "evaluation/evaluation_summary.json").is_file()
