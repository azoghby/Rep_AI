import csv
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "repai_matplotlib"))

import matplotlib.pyplot as plt

from calibration_utils import calculate_calibration, load_csv_signal, normalize_values
from detect_reps import (
    SMOOTHING_WINDOW,
    detect_reps,
    detect_reps_hybrid,
    detector_threshold_values,
    moving_average,
    read_signal,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets" / "sessions"
SCHEMA_VERSION = "boundary_dataset_session_v1"
ANNOTATION_SCHEMA_VERSION = "boundary_annotations_v1"


class AnnotationLockedError(RuntimeError):
    pass


class DatasetEligibilityError(ValueError):
    pass


def safe_slug(value, fallback):
    cleaned = "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(value).strip()
    ).strip("_")
    return cleaned or fallback


def relative_or_absolute(path):
    path = Path(path)

    try:
        return str(path.resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(path.resolve())


def resolve_repo_path(path_text):
    path = Path(path_text)
    return path if path.is_absolute() else BASE_DIR / path


def timestamp_now():
    return datetime.now().isoformat(timespec="seconds")


def new_session_id(participant_id, exercise, created_at=None):
    created_at = created_at or datetime.now()
    timestamp = created_at.strftime("%Y%m%d_%H%M%S")
    participant = safe_slug(participant_id, "participant")
    exercise_slug = safe_slug(exercise, "exercise")
    return f"{timestamp}_{participant}_{exercise_slug}"


def planned_cue_schedule(planned_reps, cadence):
    cues = []
    elapsed = 0.0
    phases = [
        ("lift", float(cadence.get("seconds_up") or 0)),
        ("hold", float(cadence.get("hold_seconds") or 0)),
        ("lower", float(cadence.get("seconds_down") or 0)),
        ("rest", float(cadence.get("bottom_rest_seconds") or 0)),
    ]

    for rep_number in range(1, int(planned_reps) + 1):
        for phase, duration in phases:
            if duration <= 0:
                continue

            start_time = elapsed
            elapsed += duration
            cues.append({
                "rep": rep_number,
                "phase": phase,
                "start_time": round(start_time, 3),
                "end_time": round(elapsed, 3),
                "duration": round(duration, 3),
                "verified_boundary": False,
            })

    return cues


def planned_total_duration(planned_reps, cadence):
    return sum(cue["duration"] for cue in planned_cue_schedule(planned_reps, cadence))


def detector_thresholds(csv_file):
    times, values = read_signal(csv_file)
    smoothed_values = moving_average(values, SMOOTHING_WINDOW)
    thresholds = detector_threshold_values(times, smoothed_values, values)
    start_threshold = thresholds["start_threshold"]
    end_threshold = thresholds["end_threshold"]
    return times, values, smoothed_values, start_threshold, end_threshold


def rep_interval(rep):
    return {
        "start_time": round(float(rep["start_time"]), 4),
        "end_time": round(float(rep["end_time"]), 4),
        "peak_time": round(float(rep.get("peak_time", 0)), 4),
        "peak_value": round(float(rep.get("peak_value", 0)), 4),
    }


def detector_outputs(csv_file):
    times, values, smoothed_values, start_threshold, end_threshold = detector_thresholds(csv_file)
    legacy_reps = detect_reps(times, values, smoothed_values, start_threshold, end_threshold)
    hybrid_reps, hybrid_diagnostics = detect_reps_hybrid(
        times,
        values,
        smoothed_values,
        start_threshold,
        end_threshold,
    )
    return {
        "legacy_detector_intervals": [rep_interval(rep) for rep in legacy_reps],
        "hybrid_detector_intervals": [rep_interval(rep) for rep in hybrid_reps],
        "hybrid_candidate_count": len(hybrid_diagnostics.get("candidate_valleys", [])),
    }


def empty_annotations():
    return {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "annotation_status": "unreviewed",
        "actual_reps": None,
        "verified_rep_intervals": [],
        "excluded_false_intervals": [],
        "confidence": None,
        "notes": "",
        "last_modified": timestamp_now(),
    }


def session_paths(session_id):
    session_dir = DATASETS_DIR / session_id
    return session_dir, session_dir / "manifest.json", session_dir / "annotations.json"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def create_session_manifest(
    session_id,
    participant_id,
    recording_csv,
    calibration_csv,
    exercise_metadata,
    planned_reps,
    cadence,
    cue_timestamps,
    recording_started_at,
    notes,
):
    outputs = detector_outputs(recording_csv)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "participant_id": participant_id,
        "recording_csv": relative_or_absolute(recording_csv),
        "calibration_csv": relative_or_absolute(calibration_csv) if calibration_csv else "",
        "exercise_metadata": exercise_metadata,
        "placement_id": exercise_metadata.get("placement_id", ""),
        "planned_reps": int(planned_reps),
        "cadence": cadence,
        "cue_timestamps": cue_timestamps,
        "legacy_detector_intervals": outputs["legacy_detector_intervals"],
        "hybrid_detector_intervals": outputs["hybrid_detector_intervals"],
        "hybrid_candidate_count": outputs["hybrid_candidate_count"],
        "recording_started_at": recording_started_at,
        "created_at": timestamp_now(),
        "notes": notes,
    }
    return manifest


def save_dataset_session(manifest, annotations=None):
    annotations = annotations or empty_annotations()
    session_dir, manifest_file, annotation_file = session_paths(manifest["session_id"])
    session_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest_file, manifest)
    write_json(annotation_file, annotations)
    return manifest_file, annotation_file


def list_dataset_sessions():
    if not DATASETS_DIR.exists():
        return []

    sessions = []
    for manifest_file in sorted(DATASETS_DIR.glob("*/manifest.json"), reverse=True):
        try:
            sessions.append(read_json(manifest_file))
        except (OSError, json.JSONDecodeError):
            continue

    return sessions


def load_annotations(session_id):
    _, _, annotation_file = session_paths(session_id)

    if not annotation_file.exists():
        return empty_annotations()

    return read_json(annotation_file)


def save_annotations(session_id, annotations):
    existing = load_annotations(session_id)

    if existing.get("annotation_status") == "locked":
        raise AnnotationLockedError(
            f"Annotation for session {session_id} is locked and cannot be overwritten."
        )

    return save_annotations_override(session_id, annotations)


def save_annotations_override(session_id, annotations, reason="manual override"):
    if not reason:
        raise ValueError("An explicit override reason is required.")

    annotations = dict(annotations)
    annotations["last_modified"] = timestamp_now()
    _, _, annotation_file = session_paths(session_id)
    write_json(annotation_file, annotations)
    return annotation_file


def unlock_annotations(session_id, reason):
    if not reason:
        raise ValueError("An explicit unlock reason is required.")

    annotations = load_annotations(session_id)
    annotations["annotation_status"] = "reviewed"
    annotations["unlock_reason"] = reason
    return save_annotations_override(session_id, annotations, reason=reason)


def recording_duration(csv_file):
    times, _ = read_signal(csv_file)
    return max(times) if times else 0


def validate_annotation_rows(rows, max_time, require_rep_numbers=True):
    errors = []
    normalized = []

    for index, row in enumerate(rows, start=1):
        if require_rep_numbers:
            try:
                rep_number = int(row.get("rep_number"))
            except (TypeError, ValueError):
                errors.append(f"Rep {index} must have rep number {index}.")
                rep_number = index

            if rep_number != index:
                errors.append(f"Rep numbers must be sequential; row {index} must be rep {index}.")
        else:
            rep_number = row.get("rep_number", index)

        try:
            start_time = float(row.get("start_time", 0))
            end_time = float(row.get("end_time", 0))
        except (TypeError, ValueError):
            errors.append(f"Rep {index} has a non-numeric start or end time.")
            continue

        if start_time >= end_time:
            errors.append(f"Rep {index} must start before it ends.")

        if start_time < 0 or end_time > max_time:
            errors.append(f"Rep {index} falls outside the recording duration.")

        normalized.append({
            "rep_number": rep_number,
            "start_time": start_time,
            "end_time": end_time,
            "confidence": row.get("confidence", ""),
            "note": row.get("note", ""),
        })

    for index in range(1, len(normalized)):
        previous = normalized[index - 1]
        current = normalized[index]

        if current["start_time"] < previous["end_time"]:
            errors.append(
                f"Rep {current['rep_number']} overlaps rep {previous['rep_number']}."
            )

    return errors, normalized


def validate_annotations_payload(annotations, max_time):
    errors, normalized_reps = validate_annotation_rows(
        annotations.get("verified_rep_intervals", []),
        max_time,
    )
    false_errors, normalized_false = validate_annotation_rows(
        annotations.get("excluded_false_intervals", []),
        max_time,
        require_rep_numbers=False,
    )
    errors.extend(error.replace("Rep", "False interval") for error in false_errors)

    actual_reps = annotations.get("actual_reps")
    if actual_reps is not None:
        try:
            actual_reps = int(actual_reps)
        except (TypeError, ValueError):
            errors.append("actual_reps must be an integer when set.")
        else:
            if actual_reps != len(normalized_reps):
                errors.append(
                    "actual_reps must match the number of verified rep intervals "
                    "for training-ready locked annotations."
                )

    return errors, normalized_reps, normalized_false


def annotation_file_exists(session_id):
    _, _, annotation_file = session_paths(session_id)
    return annotation_file.exists()


def eligibility_for_candidate_export(manifest):
    for field in ("session_id", "participant_id", "recording_csv", "exercise_metadata"):
        if field not in manifest or manifest.get(field) in ("", None):
            return False, f"missing_manifest_field_{field}", None

    session_id = manifest["session_id"]

    if not annotation_file_exists(session_id):
        return False, "missing_annotations", None

    annotations = load_annotations(session_id)

    if annotations.get("annotation_status") != "locked":
        return False, f"annotation_status_{annotations.get('annotation_status', 'missing')}", annotations

    recording_csv = resolve_repo_path(manifest.get("recording_csv", ""))
    if not recording_csv.is_file():
        return False, "missing_recording_csv", annotations

    max_time = recording_duration(recording_csv)
    errors, normalized_reps, normalized_false = validate_annotations_payload(annotations, max_time)

    if errors:
        return False, "invalid_annotations: " + "; ".join(errors), annotations

    if not normalized_reps:
        return False, "no_verified_rep_intervals", annotations

    return True, "eligible", annotations


def calibration_for_manifest(manifest):
    calibration_csv = manifest.get("calibration_csv")

    if not calibration_csv:
        return None

    calibration_path = resolve_repo_path(calibration_csv)

    if not calibration_path.exists():
        return None

    _, calibration_values = load_csv_signal(calibration_path)
    calibration = calculate_calibration(calibration_values)
    calibration["source_csv"] = calibration_csv
    return calibration


def annotation_figure(manifest, annotations):
    csv_file = resolve_repo_path(manifest["recording_csv"])
    times, values, smoothed_values, _, _ = detector_thresholds(csv_file)
    calibration = calibration_for_manifest(manifest)
    normalized = normalize_values(values, calibration) if calibration else None

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, smoothed_values, linewidth=1.8, label="Smoothed EMG")

    if normalized:
        normalized_smoothed = moving_average(normalized, SMOOTHING_WINDOW)
        ax.plot(times, normalized_smoothed, linewidth=1.2, alpha=0.65, label="Calibration-normalized")

    for cue in manifest.get("cue_timestamps", []):
        ax.axvspan(cue["start_time"], cue["end_time"], alpha=0.04)
        ax.axvline(cue["start_time"], color="gray", linestyle=":", alpha=0.25)

    for interval in manifest.get("legacy_detector_intervals", []):
        ax.axvspan(interval["start_time"], interval["end_time"], color="green", alpha=0.06)

    for interval in manifest.get("hybrid_detector_intervals", []):
        ax.axvline(interval["start_time"], color="purple", linestyle="--", alpha=0.35)
        ax.axvline(interval["end_time"], color="purple", linestyle=":", alpha=0.35)

    for interval in annotations.get("verified_rep_intervals", []):
        ax.axvspan(interval["start_time"], interval["end_time"], color="gold", alpha=0.18)
        ax.text(
            interval["start_time"],
            max(smoothed_values),
            str(interval.get("rep_number", "")),
            va="top",
            fontsize=9,
        )

    for interval in annotations.get("excluded_false_intervals", []):
        ax.axvspan(interval["start_time"], interval["end_time"], color="red", alpha=0.12)

    ax.set_title(f"Dataset Annotation: {manifest['session_id']}")
    ax.set_xlabel("Time from recording start (seconds)")
    ax.set_ylabel("Signal")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig
