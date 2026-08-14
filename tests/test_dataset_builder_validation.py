import json

import pytest

from conftest import locked_annotations, manifest, write_signal_csv
from dataset_builder import (
    AnnotationLockedError,
    empty_annotations,
    load_annotations,
    read_json,
    save_annotations,
    save_annotations_override,
    save_dataset_session,
    session_paths,
    unlock_annotations,
    validate_annotation_rows,
)


@pytest.mark.parametrize(
    "rows, expected",
    [
        ([{"rep_number": 1, "start_time": "x", "end_time": 1.0}], "non-numeric"),
        ([{"rep_number": 1, "start_time": 1.0, "end_time": 1.0}], "start before"),
        ([{"rep_number": 1, "start_time": 2.0, "end_time": 1.0}], "start before"),
        ([{"rep_number": 1, "start_time": -0.1, "end_time": 1.0}], "outside"),
        ([{"rep_number": 1, "start_time": 0.0, "end_time": 10.1}], "outside"),
        ([{}], "start before"),
    ],
)
def test_annotation_validation_rejects_bad_rows(rows, expected):
    errors, _ = validate_annotation_rows(rows, max_time=10.0)

    assert any(expected in error for error in errors)


def test_verified_rep_overlap_is_rejected():
    errors, _ = validate_annotation_rows(
        [
            {"rep_number": 1, "start_time": 0.0, "end_time": 2.0},
            {"rep_number": 2, "start_time": 1.9, "end_time": 3.0},
        ],
        max_time=10.0,
    )

    assert any("overlaps" in error for error in errors)


def test_false_interval_overlap_is_rejected():
    errors, _ = validate_annotation_rows(
        [
            {"start_time": 0.0, "end_time": 2.0},
            {"start_time": 1.9, "end_time": 3.0},
        ],
        max_time=10.0,
        require_rep_numbers=False,
    )

    assert any("overlaps" in error for error in errors)


def test_intervals_touching_at_endpoint_are_allowed():
    errors, normalized = validate_annotation_rows(
        [
            {"rep_number": 1, "start_time": 0.0, "end_time": 2.0},
            {"rep_number": 2, "start_time": 2.0, "end_time": 3.0},
        ],
        max_time=10.0,
    )

    assert errors == []
    assert normalized[1]["start_time"] == normalized[0]["end_time"]


def test_missing_annotation_file_returns_safe_empty_structure(isolated_dataset_dir):
    annotations = load_annotations("missing-session")

    assert annotations["annotation_status"] == "unreviewed"
    assert annotations["actual_reps"] is None
    assert annotations["verified_rep_intervals"] == []


def test_locked_annotations_reject_normal_save_and_allow_explicit_unlock(isolated_dataset_dir, tmp_path):
    recording = write_signal_csv(tmp_path / "source.csv")
    session = manifest("locked-session", recording)
    save_dataset_session(session, locked_annotations())
    _, _, annotation_file = session_paths("locked-session")
    before = annotation_file.read_text(encoding="utf-8")

    with pytest.raises(AnnotationLockedError):
        save_annotations("locked-session", empty_annotations())

    assert annotation_file.read_text(encoding="utf-8") == before

    unlock_annotations("locked-session", reason="test unlock")
    save_annotations("locked-session", empty_annotations())

    saved = json.loads(annotation_file.read_text(encoding="utf-8"))
    assert saved["annotation_status"] == "unreviewed"


def test_explicit_override_can_update_locked_annotation(isolated_dataset_dir, tmp_path):
    recording = write_signal_csv(tmp_path / "source.csv")
    session = manifest("override-session", recording)
    save_dataset_session(session, locked_annotations())

    save_annotations_override("override-session", empty_annotations(), reason="test override")

    assert load_annotations("override-session")["annotation_status"] == "unreviewed"


def test_rep_count_concepts_round_trip_distinctly(isolated_dataset_dir, tmp_path):
    recording = write_signal_csv(tmp_path / "source.csv")
    session = manifest("counts-session", recording, planned_reps=8)
    session["legacy_detector_intervals"] = [
        {"start_time": index, "end_time": index + 0.4}
        for index in range(5)
    ]
    session["hybrid_detector_intervals"] = [
        {"start_time": index, "end_time": index + 0.4}
        for index in range(7)
    ]
    session["hybrid_candidate_count"] = 11
    annotations = locked_annotations(
        actual_reps=6,
        intervals=[
            {"rep_number": index + 1, "start_time": index, "end_time": index + 0.4}
            for index in range(6)
        ],
    )

    manifest_file, _ = save_dataset_session(session, annotations)
    loaded_manifest = read_json(manifest_file)
    loaded_annotations = load_annotations("counts-session")

    assert loaded_manifest["planned_reps"] == 8
    assert loaded_annotations["actual_reps"] == 6
    assert len(loaded_manifest["legacy_detector_intervals"]) == 5
    assert len(loaded_manifest["hybrid_detector_intervals"]) == 7
    assert loaded_manifest["hybrid_candidate_count"] == 11


def test_actual_reps_zero_is_distinct_from_unset(isolated_dataset_dir, tmp_path):
    recording = write_signal_csv(tmp_path / "source.csv")
    session = manifest("zero-session", recording)
    save_dataset_session(session, locked_annotations(actual_reps=0, intervals=[]))

    assert load_annotations("zero-session")["actual_reps"] == 0
