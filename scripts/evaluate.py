"""Evaluate one sequence in pipeline JSONL against manual frame annotations.

Annotations: {"frames": [{"frame_index": 0, "objects": [
    {"id": "car-1", "bbox": [10, 20, 60, 80], "label": "vehicle"}]}]}.
Boxes are continuous pixel coordinates [x1, y1, x2, y2], without a +1 area
correction. Frame indices and image coordinates must agree with pipeline output
(including any resizing). Matching is class-agnostic by default; --match-labels
enables exact label matching. The default minimum bbox height is 35 pixels.

Only explicitly annotated frames are scored. An empty objects list means a
negative frame, not an unannotated frame. Missing prediction records count as
empty predictions. Partial object annotations bias precision and IDF1 because
unlabeled objects can be counted as false positives. Recall applies only to
labeled objects. Sparse annotations cannot reveal switches between samples.
Set annotations.exhaustive_objects=true or --assume-exhaustive only after
checking that every relevant object in each annotated frame was labeled.
Optional annotations.selection.min_height_px (or --min-height-px) applies the
same image-space height cutoff to GT and predictions, before matching. Optional
selection.labels restricts both sides to the listed classes. Exhaustiveness is
relative to this selection; distant/smaller objects are outside evaluation.

Dependencies: numpy, scipy. Run from any directory with:
python scripts/evaluate.py --predictions predictions.jsonl \
    --annotations annotations.json --output evaluation.json

IDF1 reference: https://arxiv.org/abs/1609.01775
Reference implementation (all eligible pairs, then global assignment):
https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/identity.py
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{context}: expected a finite number")
    return float(value)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant {value} is not permitted")


def _identity(value: Any, context: str) -> tuple[type, Any]:
    if isinstance(value, bool) or not isinstance(value, (int, str)) or value == "":
        raise ValueError(f"{context}: expected an integer or nonempty string identity")
    return type(value), value


def _corners(value: Any, context: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 8:
        raise ValueError(f"{context}: expected eight [x, y] corners")
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{context}: expected eight [x, y] corners")
        for coordinate in point:
            _finite(coordinate, context)
    return np.asarray(value, dtype=float)


def _objects(value: Any, context: str, id_key: str | None = None) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError(f"{context}: expected an array")
    seen = set()
    for obj in value:
        if not isinstance(obj, dict):
            raise ValueError(f"{context}: expected object records")
        bbox = obj.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"{context}.bbox: expected [x1, y1, x2, y2]")
        x1, y1, x2, y2 = [_finite(v, f"{context}.bbox") for v in bbox]
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"{context}.bbox: box must have positive width and height")
        if not isinstance(obj.get("label"), str) or not obj["label"]:
            raise ValueError(f"{context}.label: expected a nonempty string")
        if id_key:
            identity = _identity(obj.get(id_key), f"{context}.{id_key}")
            if identity in seen:
                raise ValueError(f"{context}: duplicate {id_key} in one frame")
            seen.add(identity)
        if obj.get("corners") not in (None, []):
            _corners(obj["corners"], f"{context}.corners")
        if obj.get("fit_error") is not None:
            if _finite(obj["fit_error"], f"{context}.fit_error") < 0:
                raise ValueError(f"{context}.fit_error: expected a nonnegative value")
        if "valid" in obj and not isinstance(obj["valid"], bool):
            raise ValueError(f"{context}.valid: expected a boolean")
        if id_key is None:
            _finite(obj.get("score"), f"{context}.score")
    return value


def _frames(records: Any, prediction: bool) -> dict[int, dict]:
    if not isinstance(records, list):
        raise ValueError("frames: expected an array")
    indexed = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("frame: expected an object")
        index = record.get("frame_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("frame_index: expected a nonnegative integer")
        if index in indexed:
            raise ValueError(f"duplicate frame_index {index}")
        if prediction:
            _objects(record.get("detections"), f"frame {index}.detections")
            _objects(record.get("tracks"), f"frame {index}.tracks", "track_id")
        else:
            _objects(record.get("objects"), f"frame {index}.objects", "id")
        indexed[index] = record
    return indexed


def load_predictions(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, parse_constant=_reject_json_constant)
            except ValueError as exc:
                raise ValueError(f"{path}: line {line_number}: invalid JSON: {exc}") from exc
            if isinstance(record, dict):
                records.append(record)
            elif isinstance(record, list):
                records.extend(record)
            else:
                raise ValueError(f"{path}: line {line_number}: expected a frame or frame array")
    _frames(records, True)
    return records


def iou_matrix(objects: list[dict], predictions: list[dict]) -> np.ndarray:
    a = np.asarray([o["bbox"] for o in objects], dtype=float).reshape(-1, 4)
    b = np.asarray([o["bbox"] for o in predictions], dtype=float).reshape(-1, 4)
    low = np.maximum(a[:, None, :2], b[None, :, :2])
    high = np.minimum(a[:, None, 2:], b[None, :, 2:])
    area = np.prod(np.maximum(high - low, 0), axis=2)
    union = np.prod(a[:, 2:] - a[:, :2], axis=1)[:, None]
    union = union + np.prod(b[:, 2:] - b[:, :2], axis=1)[None, :] - area
    return area / union


def eligible_pairs(objects: list[dict], predictions: list[dict], threshold: float,
                   ignore_labels: bool) -> tuple[np.ndarray, np.ndarray]:
    overlaps = iou_matrix(objects, predictions)
    eligible = overlaps >= threshold
    if not ignore_labels:
        labels = np.asarray([[a["label"] == b["label"] for b in predictions]
                             for a in objects], dtype=bool).reshape(overlaps.shape)
        eligible &= labels
    return overlaps, eligible


def match_frame(overlaps: np.ndarray, eligible: np.ndarray) -> list[tuple[int, int]]:
    if not overlaps.size:
        return []
    # The bonus makes maximum cardinality primary and summed IoU secondary.
    bonus = min(overlaps.shape) + 1
    weights = np.where(eligible, bonus + overlaps, 0)
    rows, cols = linear_sum_assignment(weights, maximize=True)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if eligible[r, c]]


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _counts(tp: int, fp: int, fn: int) -> dict:
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn)}


def evaluate(records: list[dict], annotations: dict, *, iou_threshold: float = 0.5,
             ignore_labels: bool = True, assume_exhaustive: bool = False,
             compare_corners: bool = False, min_height_px: float | None = None,
             labels: list[str] | None = None) -> dict:
    """Return observed metrics and their scope; never extrapolate missing labels.

    Optional --compare-corners requires verified identical semantic corner order
    in prediction and annotation, and eight image-space [x,y] corners on GT
    objects. This measures matched 2D reprojection only, not metric 3D accuracy.
    """
    _finite(iou_threshold, "iou_threshold")
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    if not isinstance(annotations, dict):
        raise ValueError("annotations: expected an object containing frames")
    if "exhaustive_objects" in annotations and not isinstance(annotations["exhaustive_objects"], bool):
        raise ValueError("exhaustive_objects: expected a boolean")
    visible_exhaustive = annotations.get("exhaustive_visible_selected_objects")
    if visible_exhaustive is not None:
        if not isinstance(visible_exhaustive, bool):
            raise ValueError("exhaustive_visible_selected_objects: expected a boolean")
        if "exhaustive_objects" in annotations and annotations["exhaustive_objects"] != visible_exhaustive:
            raise ValueError("Conflicting annotation exhaustiveness flags")
    predicted = _frames(records, True)
    truth = _frames(annotations.get("frames"), False)
    if not truth:
        raise ValueError("annotations: at least one explicitly annotated frame is required")
    exhaustive = assume_exhaustive or annotations.get("exhaustive_objects", visible_exhaustive or False)
    selection = annotations.get("selection", {})
    if not isinstance(selection, dict):
        raise ValueError("selection: expected an object")
    height_cutoff = selection.get("min_height_px", 35) if min_height_px is None else min_height_px
    _finite(height_cutoff, "min_height_px")
    if height_cutoff < 0:
        raise ValueError("min_height_px must be nonnegative")
    selected_labels = selection.get("labels") if labels is None else labels
    if selected_labels is not None and (not isinstance(selected_labels, list) or
                                       not selected_labels or
                                       any(not isinstance(v, str) or not v for v in selected_labels)):
        raise ValueError("selection.labels: expected a nonempty array of nonempty strings")

    def selected(obj: dict) -> bool:
        return (obj["bbox"][3] - obj["bbox"][1] >= height_cutoff and
                (selected_labels is None or obj["label"] in selected_labels))

    indices = sorted(truth)
    totals = {"detections": [0, 0, 0], "tracks": [0, 0, 0]}
    gt_counts, track_counts, potential = Counter(), Counter(), Counter()
    last_matched = {}
    switch_events, frame_reports = [], []
    corner_errors, fit_errors = [], []
    with_corners = without_corners = matched_with_gt_corners = missing_matched_corners = 0
    excluded = Counter()
    cuboid_validity = Counter()
    for index in indices:
        objects = [obj for obj in truth[index]["objects"] if selected(obj)]
        excluded["objects"] += len(truth[index]["objects"]) - len(objects)
        prediction = predicted.get(index, {"detections": [], "tracks": []})
        frame_report = {"frame_index": index, "prediction_record_present": index in predicted}
        for kind in ("detections", "tracks"):
            items = [item for item in prediction[kind] if selected(item)]
            excluded[kind] += len(prediction[kind]) - len(items)
            overlaps, eligible = eligible_pairs(objects, items, iou_threshold, ignore_labels)
            matches = match_frame(overlaps, eligible)
            counts = [len(matches), len(items) - len(matches), len(objects) - len(matches)]
            totals[kind] = [a + b for a, b in zip(totals[kind], counts)]
            frame_report[kind] = _counts(*counts)
            if kind == "detections":
                continue
            for obj in objects:
                gt_counts[_identity(obj["id"], "id")] += 1
            for item in items:
                track_counts[_identity(item["track_id"], "track_id")] += 1
                if not item.get("corners"):
                    without_corners += 1
                else:
                    with_corners += 1
                if item.get("fit_error") is not None:
                    fit_errors.append(item["fit_error"])
                cuboid_validity["unspecified" if "valid" not in item else
                                "valid" if item["valid"] else "invalid"] += 1
            # Global identity counts use ALL eligible pairs, not local matches.
            for row, col in zip(*np.nonzero(eligible)):
                potential[(_identity(objects[row]["id"], "id"),
                           _identity(items[col]["track_id"], "track_id"))] += 1
            for row, col in matches:
                obj, item = objects[row], items[col]
                gt_id = _identity(obj["id"], "id")
                track_id = _identity(item["track_id"], "track_id")
                previous = last_matched.get(gt_id)
                if previous and previous[0] != track_id:
                    switch_events.append({"frame_index": index, "id": obj["id"],
                                          "previous_frame_index": previous[1],
                                          "previous_track_id": previous[0][1],
                                          "track_id": item["track_id"]})
                last_matched[gt_id] = track_id, index
                if compare_corners and obj.get("corners"):
                    matched_with_gt_corners += 1
                    if not item.get("corners"):
                        missing_matched_corners += 1
                    else:
                        errors = np.linalg.norm(_corners(obj["corners"], "GT corners") -
                                                _corners(item["corners"], "track corners"), axis=1)
                        corner_errors.extend(errors.tolist())
        frame_reports.append(frame_report)

    gt_ids, track_ids = list(gt_counts), list(track_counts)
    weights = np.asarray([[potential[g, p] for p in track_ids] for g in gt_ids],
                         dtype=np.int64).reshape(len(gt_ids), len(track_ids))
    rows, cols = linear_sum_assignment(weights, maximize=True)
    idtp = int(weights[rows, cols].sum())
    gt_total, track_total = sum(gt_counts.values()), sum(track_counts.values())
    idfn, idfp = gt_total - idtp, track_total - idtp
    identity = {"idtp": idtp, "idfp": idfp, "idfn": idfn,
                "id_precision": _ratio(idtp, track_total),
                "id_recall": _ratio(idtp, gt_total),
                "idf1": _ratio(2 * idtp, gt_total + track_total),
                "global_assignment": [
                    {"id": gt_ids[r][1], "track_id": track_ids[c][1],
                     "matched_observations": int(weights[r, c])}
                    for r, c in zip(rows, cols) if weights[r, c] > 0],
                "id_switches": len(switch_events), "switch_events": switch_events}
    limitations = [
        "Metrics cover only explicitly annotated frames; all other frames are excluded.",
        "Missing prediction frames are scored as empty predictions, not silently skipped.",
        "Recall measures labeled objects only; no accuracy is inferred for unannotated objects.",
        "Observed ID switches compare each GT identity's current spatial match with its last "
        "matched track, including gaps. Frame matching maximizes count then IoU, with no "
        "continuity preference; this is not the full MOTChallenge/CLEAR matching protocol.",
        "IDs must remain unique across one sequence. Evaluate unrelated sequences separately.",
        "Corners and fit_error without geometric ground truth cannot establish 3D accuracy.",
        "Undefined ratios are null; scores are fractions in [0,1], not percentages.",
    ]
    if not exhaustive:
        limitations.append("Object annotations are not declared exhaustive: unmatched predictions "
                           "may be unlabeled true objects. Precision, false-positive counts and "
                           "IDF1 are conditional diagnostics, not valid full-scene estimates.")
    if any(b != a + 1 for a, b in zip(indices, indices[1:])):
        limitations.append("Sparse frame annotations can hide intervening ID switches and "
                           "fragmentation; IDF1 is computed only over sampled observations.")
    if ignore_labels:
        limitations.append("Labels were ignored explicitly; classification accuracy is not scored.")
    if height_cutoff:
        limitations.append("Object selection uses bbox height >= the cutoff on each side independently. "
                           "Distant/smaller boxes are excluded. Height is a size proxy, not measured "
                           "distance; localization errors across the cutoff can change eligibility.")
    cuboids = {"track_observations_with_corners": with_corners,
               "track_observations_without_corners": without_corners,
               "fit_error_count": len(fit_errors),
               "pipeline_reported_validity": {key: cuboid_validity[key]
                                               for key in ("valid", "invalid", "unspecified")},
               "mean_reported_fit_error": float(np.mean(fit_errors)) if fit_errors else None,
               "fit_error_interpretation": "Pipeline self-reported diagnostic; not ground-truth error.",
               "corner_accuracy": None}
    if compare_corners:
        cuboids["corner_accuracy"] = {
            "matched_objects_with_gt_corners": matched_with_gt_corners,
            "matched_objects_missing_prediction_corners": missing_matched_corners,
            "compared_points": len(corner_errors),
            "mean_error_px": float(np.mean(corner_errors)) if corner_errors else None,
            "rmse_px": float(np.sqrt(np.mean(np.square(corner_errors)))) if corner_errors else None,
            "interpretation": "Matched image-space corners only; caller asserted semantic "
                              "corner correspondence. Not metric 3D error; unmatched objects "
                              "are excluded and coverage is reported above.",
        }
    return {"schema_version": 1,
            "protocol": {"iou_threshold": iou_threshold, "bbox_format": "xyxy",
                         "label_matching": "ignored" if ignore_labels else "exact",
                         "frame_matching": "Hungarian: maximize match count, then summed IoU",
                         "identity_matching": "Hungarian: maximize all eligible identity-pair observations",
                         "selection": {"min_height_px": height_cutoff, "labels": selected_labels},
                         "sources": ["https://arxiv.org/abs/1609.01775",
                                     "https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/identity.py"]},
            "scope": {"annotated_frames": len(truth), "prediction_frames": len(predicted),
                      "excluded_prediction_frames": len(set(predicted) - set(truth)),
                      "missing_prediction_frames": sorted(set(truth) - set(predicted)),
                      "exhaustive_objects": exhaustive,
                      "exhaustive_visible_selected_objects": exhaustive,
                      "metrics_valid_for_selected_objects": exhaustive,
                      "excluded_by_selection": dict(excluded),
                      "ground_truth_observations": gt_total},
            "detections": _counts(*totals["detections"]),
            "tracks": _counts(*totals["tracks"]), "identity": identity,
            "cuboids": cuboids, "limitations": limitations, "frames": frame_reports,
            "annotation_provenance": annotations.get("annotation_provenance"),
            "annotation_source_video_sha256": annotations.get("source_video_sha256"),
            "annotation_tracking_scope": annotations.get("tracking_scope"),
            "annotation_selection_description": selection.get("definition")}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report_input_path(path: Path, project_root: Path | None = None) -> str:
    root = project_root if project_root is not None else Path(__file__).resolve().parents[1]
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    matching = parser.add_mutually_exclusive_group()
    matching.add_argument("--ignore-labels", dest="ignore_labels", action="store_true", default=True)
    matching.add_argument("--match-labels", dest="ignore_labels", action="store_false")
    parser.add_argument("--assume-exhaustive", action="store_true")
    parser.add_argument("--compare-corners", action="store_true",
                        help="Assert identical semantic corner order and compute matched 2D errors")
    parser.add_argument("--min-height-px", type=float, default=None,
                        help="Override annotation selection; filter GT and prediction box heights")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Override annotation selection; filter both sides to these labels")
    parser.add_argument("--prediction-metrics", type=Path,
                        help="Pipeline metrics JSON for checking annotation image_size against output_size")
    args = parser.parse_args()
    if args.output.resolve() in (args.predictions.resolve(), args.annotations.resolve()):
        parser.error("output must not overwrite predictions or annotations")
    try:
        records = load_predictions(args.predictions)
        annotations = json.loads(args.annotations.read_text(encoding="utf-8-sig"),
                                 parse_constant=_reject_json_constant)
        metrics_path = args.prediction_metrics
        if metrics_path is None and args.predictions.name.endswith(".predictions.jsonl"):
            candidate = args.predictions.with_name(args.predictions.name.removesuffix(".predictions.jsonl") +
                                                   ".metrics.json")
            if candidate.exists():
                metrics_path = candidate
        image_size = None
        if metrics_path is not None:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig"),
                                 parse_constant=_reject_json_constant)
            image_size = metrics.get("output_size")
            if "frames_processed" in metrics and metrics["frames_processed"] != len(records):
                raise ValueError("Prediction frame count differs from pipeline metrics; outputs may be incomplete or stale")
            if isinstance(annotations, dict) and annotations.get("image_size") is not None:
                if image_size != annotations["image_size"]:
                    raise ValueError("Annotation image_size does not match pipeline output_size")
        report = evaluate(records, annotations, iou_threshold=args.iou_threshold,
                          ignore_labels=args.ignore_labels, assume_exhaustive=args.assume_exhaustive,
                          compare_corners=args.compare_corners, min_height_px=args.min_height_px,
                          labels=args.labels)
        report["scope"]["prediction_image_size"] = image_size
        report["scope"]["annotation_image_size"] = annotations.get("image_size")
        report["scope"]["coordinate_size_verified"] = (image_size is not None and
                                                          image_size == annotations.get("image_size"))
        report["inputs"] = {"predictions": _report_input_path(args.predictions),
                            "predictions_sha256": _sha256(args.predictions),
                            "annotations": _report_input_path(args.annotations),
                            "annotations_sha256": _sha256(args.annotations)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Evaluation written to {args.output}")


if __name__ == "__main__":
    main()
