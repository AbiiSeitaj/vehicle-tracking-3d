from scripts.evaluate import _report_input_path


def test_input_path_inside_project_is_relative(tmp_path):
    root = tmp_path / "project"
    path = root / "outputs" / "run.predictions.jsonl"
    assert _report_input_path(path, root) == "outputs/run.predictions.jsonl"


def test_external_input_path_does_not_expose_parent_directories(tmp_path):
    root = tmp_path / "project"
    path = tmp_path / "private_directory" / "run.predictions.jsonl"
    assert _report_input_path(path, root) == "run.predictions.jsonl"
