import csv

import pytest

from build_boundary_dataset import build_dataset, sha256_file, write_rows
from conftest import locked_annotations, manifest, write_signal_csv
from dataset_builder import eligibility_for_candidate_export, save_dataset_session, session_paths


def read_csv_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_missing_annotations_are_skipped(isolated_dataset_dir, tmp_path):
    recording = write_signal_csv(tmp_path / "source.csv")
    session = manifest("missing-annotations", recording)
    session_dir, manifest_file, _ = session_paths(session["session_id"])
    session_dir.mkdir(parents=True)
    manifest_file.write_text("{}", encoding="utf-8")

    eligible, reason, _ = eligibility_for_candidate_export(session)

    assert eligible is False
    assert reason == "missing_annotations"


def test_unreviewed_annotations_are_skipped(isolated_dataset_dir, tmp_path):
    recording = write_signal_csv(tmp_path / "source.csv")
    session = manifest("unreviewed", recording)
    save_dataset_session(session)

    eligible, reason, _ = eligibility_for_candidate_export(session)

    assert eligible is False
    assert reason == "annotation_status_unreviewed"


def test_reviewed_but_incomplete_annotations_are_skipped(isolated_dataset_dir, tmp_path):
    recording = write_signal_csv(tmp_path / "source.csv")
    session = manifest("reviewed-incomplete", recording)
    save_dataset_session(session, {
        "schema_version": "boundary_annotations_v1",
        "annotation_status": "reviewed",
        "actual_reps": None,
        "verified_rep_intervals": [],
        "excluded_false_intervals": [],
    })

    eligible, reason, _ = eligibility_for_candidate_export(session)

    assert eligible is False
    assert reason == "annotation_status_reviewed"


def test_locked_valid_annotations_are_eligible(isolated_dataset_dir, tmp_path):
    recording = write_signal_csv(tmp_path / "source.csv")
    session = manifest("locked-valid", recording)
    save_dataset_session(session, locked_annotations())

    eligible, reason, _ = eligibility_for_candidate_export(session)

    assert eligible is True
    assert reason == "eligible"


def test_locked_actual_rep_conflict_is_skipped(isolated_dataset_dir, tmp_path):
    recording = write_signal_csv(tmp_path / "source.csv")
    session = manifest("locked-conflict", recording)
    save_dataset_session(session, locked_annotations(actual_reps=1))

    eligible, reason, _ = eligibility_for_candidate_export(session)

    assert eligible is False
    assert "actual_reps must match" in reason


def test_malformed_manifest_required_fields_fail_clearly(isolated_dataset_dir):
    eligible, reason, _ = eligibility_for_candidate_export({"session_id": "bad"})

    assert eligible is False
    assert reason == "missing_manifest_field_participant_id"


def test_mixed_dataset_skips_ineligible_sessions(isolated_dataset_dir, tmp_path):
    good_recording = write_signal_csv(tmp_path / "good.csv")
    bad_recording = write_signal_csv(tmp_path / "bad.csv")
    good = manifest("good", good_recording)
    bad = manifest("bad", bad_recording)
    save_dataset_session(good, locked_annotations())
    save_dataset_session(bad)

    rows, skipped = build_dataset([good, bad], tmp_path / "out.csv", tolerance=0.25)

    assert all(row["session_id"] == "good" for row in rows)
    assert skipped == [{"session_id": "bad", "reason": "annotation_status_unreviewed"}]
    assert len(read_csv_rows(tmp_path / "out.csv")) == len(rows)


def test_output_may_not_target_source_csv_and_source_hash_is_unchanged(isolated_dataset_dir, tmp_path):
    recording = write_signal_csv(tmp_path / "source.csv")
    session = manifest("protected-source", recording)
    save_dataset_session(session, locked_annotations())
    before_hash = sha256_file(recording)

    with pytest.raises(ValueError, match="may not overwrite source CSV"):
        build_dataset([session], recording, tolerance=0.25)

    assert sha256_file(recording) == before_hash


def test_output_may_not_target_calibration_csv(isolated_dataset_dir, tmp_path):
    recording = write_signal_csv(tmp_path / "source.csv")
    calibration = write_signal_csv(tmp_path / "calibration.csv")
    session = manifest("protected-calibration", recording)
    session["calibration_csv"] = str(calibration)
    save_dataset_session(session, locked_annotations())

    with pytest.raises(ValueError, match="may not overwrite source CSV"):
        build_dataset([session], calibration, tolerance=0.25)


def test_write_rows_is_atomic_enough_to_leave_complete_csv(tmp_path):
    output = tmp_path / "rows.csv"

    write_rows([
        {
            "session_id": "s1",
            "recording_filename": "r.csv",
            "participant_id": "p1",
            "exercise": "curl",
            "side": "right",
            "weight": "10",
            "candidate_timestamp": 1.0,
            "valley_depth": 1,
            "normalized_valley_depth": 1,
            "rebound_strength": 1,
            "valley_duration": 1,
            "adjacent_contraction_center_gap": 1,
            "left_segment_duration": 1,
            "right_segment_duration": 1,
            "plateau_support": 1,
            "high_activation_area": 1,
            "local_cycle_duration_estimate": 1,
            "candidate_score": 1,
            "hybrid_status": "accepted",
            "human_label": "true_boundary",
            "matched_human_boundary_timestamp": 1.0,
            "matching_error": 0.0,
            "matched_boundary_index": 1,
            "annotation_confidence": "high",
        }
    ], output)

    rows = read_csv_rows(output)
    assert rows[0]["human_label"] == "true_boundary"


def test_replay_session_references_source_without_copying(isolated_dataset_dir, tmp_path):
    recording = write_signal_csv(tmp_path / "replay.csv")
    session = manifest("replay", recording)
    save_dataset_session(session)
    copied_csvs = list(isolated_dataset_dir.glob("**/*.csv"))

    assert copied_csvs == []
