import csv

import pytest

from conftest import locked_annotations, manifest, write_signal_csv
from dataset_builder import save_dataset_session
from summarize_boundary_dataset import load_candidate_rows, summarize_dataset


def write_candidate_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["session_id", "participant_id", "exercise", "side", "weight", "human_label"]

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def test_zero_eligible_sessions_and_missing_candidate_csv_do_not_crash(isolated_dataset_dir, tmp_path):
    summary = summarize_dataset(tmp_path / "missing.csv")

    assert summary["candidate_csv_exists"] is False
    assert summary["total_candidates"] == 0
    assert summary["sessions"] == []


def test_missing_optional_annotation_fields_do_not_crash_summary(isolated_dataset_dir, tmp_path):
    recording = write_signal_csv(tmp_path / "source.csv")
    session = manifest("minimal-annotation", recording)
    save_dataset_session(session, {
        "schema_version": "boundary_annotations_v1",
        "annotation_status": "locked",
        "actual_reps": 2,
        "verified_rep_intervals": [
            {"rep_number": 1, "start_time": 0.0, "end_time": 0.4},
            {"rep_number": 2, "start_time": 0.5, "end_time": 0.9},
        ],
        "excluded_false_intervals": [],
    })
    candidate_csv = write_candidate_csv(tmp_path / "candidates.csv", [])

    summary = summarize_dataset(candidate_csv)

    assert summary["total_verified_reps"] == 2
    assert summary["skipped_reasons"] == {}


def test_summary_counts_by_session_participant_status_and_label(isolated_dataset_dir, tmp_path):
    rec1 = write_signal_csv(tmp_path / "one.csv")
    rec2 = write_signal_csv(tmp_path / "two.csv")
    session_one = manifest("s1", rec1, participant_id="p1")
    session_two = manifest("s2", rec2, participant_id="p2")
    save_dataset_session(session_one, locked_annotations())
    save_dataset_session(session_two)
    candidate_csv = write_candidate_csv(
        tmp_path / "candidates.csv",
        [
            {"session_id": "s1", "participant_id": "p1", "exercise": "curl", "side": "right", "weight": "10", "human_label": "true_boundary"},
            {"session_id": "s1", "participant_id": "p1", "exercise": "curl", "side": "right", "weight": "10", "human_label": "false_boundary"},
        ],
    )

    summary = summarize_dataset(candidate_csv)

    assert len(summary["sessions"]) == 2
    assert len(summary["participants"]) == 2
    assert summary["labels"]["true_boundary"] == 1
    assert summary["labels"]["false_boundary"] == 1
    assert summary["reviewed_statuses"]["locked"] == 1
    assert summary["reviewed_statuses"]["unreviewed"] == 1
    assert summary["skipped_reasons"]["annotation_status_unreviewed"] == 1


def test_malformed_candidate_csv_required_fields_fail_clearly(tmp_path):
    malformed = tmp_path / "malformed.csv"
    malformed.write_text("session_id,human_label\ns1,true_boundary\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_candidate_rows(malformed)
