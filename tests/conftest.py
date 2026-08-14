import csv
import sys
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
PYTHON_DIR = BASE_DIR / "python"

for path in (APP_DIR, PYTHON_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture
def isolated_dataset_dir(tmp_path, monkeypatch):
    import dataset_builder

    sessions_dir = tmp_path / "datasets" / "sessions"
    monkeypatch.setattr(dataset_builder, "DATASETS_DIR", sessions_dir)
    return sessions_dir


def write_signal_csv(path, values=None):
    values = values or [0, 10, 50, 100, 50, 10, 0, 20, 80, 20, 0]
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_ms", "emg_value"])

        for index, value in enumerate(values):
            writer.writerow([index * 100, value])

    return path


def manifest(session_id, recording_csv, participant_id="p1", planned_reps=8):
    return {
        "schema_version": "boundary_dataset_session_v1",
        "session_id": session_id,
        "participant_id": participant_id,
        "recording_csv": str(recording_csv),
        "calibration_csv": "",
        "exercise_metadata": {
            "exercise": "curl",
            "muscle": "bicep",
            "side": "right",
            "weight": "10",
        },
        "placement_id": "place-1",
        "planned_reps": planned_reps,
        "cadence": {"seconds_up": 1, "hold_seconds": 0, "seconds_down": 1, "bottom_rest_seconds": 1},
        "cue_timestamps": [],
        "legacy_detector_intervals": [],
        "hybrid_detector_intervals": [],
        "hybrid_candidate_count": 0,
        "recording_started_at": "synthetic",
        "created_at": "2026-01-01T00:00:00",
        "notes": "",
    }


def locked_annotations(actual_reps=2, intervals=None):
    if intervals is None:
        intervals = [
            {"rep_number": 1, "start_time": 0.0, "end_time": 0.4, "confidence": "high", "note": ""},
            {"rep_number": 2, "start_time": 0.5, "end_time": 0.9, "confidence": "high", "note": ""},
        ]
    return {
        "schema_version": "boundary_annotations_v1",
        "annotation_status": "locked",
        "actual_reps": actual_reps,
        "verified_rep_intervals": intervals,
        "excluded_false_intervals": [],
        "confidence": 0.9,
        "notes": "",
        "last_modified": "2026-01-01T00:00:00",
    }
