import csv
import json
from pathlib import Path

import streamlit_app


def write_replay(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_ms", "emg_value"])
        for index, value in enumerate(values):
            writer.writerow([index * 100, value])
    return path


def write_metadata(csv_file, exercise, side, weight, planned_reps="6", muscle="bicep"):
    metadata = {
        "exercise_name": exercise,
        "muscle": muscle,
        "side": side,
        "weight": weight,
        "expected_reps": planned_reps,
        "calibration_csv": f"calibration_{side}.csv",
    }
    metadata_file = csv_file.with_name(f"{csv_file.stem}_metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as file:
        json.dump(metadata, file)
    return metadata


def test_replay_metadata_replaces_prior_ready_selection_and_freezes_for_set(tmp_path, monkeypatch):
    replay_a = write_replay(tmp_path / "replay_a.csv", [10, 20, 30])
    replay_b = write_replay(tmp_path / "replay_b.csv", [200, 240, 280, 320])
    metadata_a = write_metadata(replay_a, "exercise_a", "right", "3")
    metadata_b = write_metadata(replay_b, "exercise_b", "left", "5")
    state = {}

    assert streamlit_app.sync_replay_metadata_to_state(state, replay_a, ready=True)
    assert state["workout_exercise"] == "exercise_a"
    assert state["workout_side"] == "right"
    assert state["workout_weight"] == "3"

    assert streamlit_app.sync_replay_metadata_to_state(state, replay_b, ready=True)
    assert state["workout_exercise"] == "exercise_b"
    assert state["workout_side"] == "left"
    assert state["workout_weight"] == "5"
    assert state["workout_calibration_csv"] == "calibration_left.csv"

    state["workout_notes"] = "manual note for B"
    state["workout_participant_id"] = "p1"
    state["workout_comparison_target"] = ""
    monkeypatch.setattr(streamlit_app.st, "session_state", state)
    monkeypatch.setattr(streamlit_app, "DATA_DIR", tmp_path)

    spec_b = streamlit_app.build_workout_session_spec("Replay CSV", None, replay_b, False)

    assert spec_b.metadata["exercise_name"] == "exercise_b"
    assert spec_b.metadata["side"] == "left"
    assert spec_b.metadata["weight"] == "5"
    assert spec_b.metadata["source_replay_filename"] == "replay_b.csv"
    assert spec_b.output_file.name.startswith("exercise_b_")

    state["workout_exercise"] = "mutated_after_start"
    state["workout_side"] = "right"
    assert spec_b.metadata["exercise_name"] == "exercise_b"
    assert spec_b.metadata["side"] == "left"

    identity = streamlit_app.completed_set_identity(spec_b.metadata, spec_b.output_file)
    assert identity == {
        "source_replay_filename": "replay_b.csv",
        "output_recording_filename": spec_b.output_file.name,
        "exercise": "exercise_b",
        "side": "left",
        "weight": "5",
    }

    previous_a = tmp_path / "previous_a.csv"
    previous_b = tmp_path / "previous_b.csv"
    write_replay(previous_a, [1, 2])
    write_replay(previous_b, [3, 4])
    write_metadata(previous_a, metadata_a["exercise_name"], metadata_a["side"], metadata_a["weight"])
    write_metadata(previous_b, metadata_b["exercise_name"], metadata_b["side"], metadata_b["weight"])
    monkeypatch.setattr(streamlit_app, "valid_workout_recordings", lambda: [previous_a, previous_b])

    assert streamlit_app.compatible_previous_recording(spec_b.output_file, spec_b.metadata) == previous_b

    captured = {}

    def fake_create_session_manifest(**kwargs):
        captured.update(kwargs)
        return {"session_id": kwargs["session_id"], "exercise_metadata": kwargs["exercise_metadata"]}

    monkeypatch.setattr(streamlit_app, "create_session_manifest", fake_create_session_manifest)
    manifest = streamlit_app.build_completed_set_dataset_manifest(
        spec_b.output_file,
        spec_b.metadata,
        {"legacy": {"summary": {}}, "hybrid": {"summary": {}}},
    )

    assert manifest["exercise_metadata"] == {
        "exercise": "exercise_b",
        "muscle": "bicep",
        "side": "left",
        "weight": "5",
        "body_position": "",
        "grip": "",
        "placement_id": "",
    }
    assert captured["calibration_csv"].name == "calibration_left.csv"

    assert not streamlit_app.sync_replay_metadata_to_state(state, replay_a, ready=False)
    assert state["workout_exercise"] == "mutated_after_start"

    streamlit_app.clear_replay_metadata_source_marker(state)
    assert "workout_replay_metadata_source" not in state
