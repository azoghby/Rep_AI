import json
import re
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
GRAPHS_DIR = BASE_DIR / "graphs"
SUMMARIES_DIR = BASE_DIR / "summaries"
PYTHON_DIR = BASE_DIR / "python"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from recording_metadata import load_metadata  # noqa: E402
from calibration_utils import calculate_calibration, load_csv_signal  # noqa: E402


def read_text(path):
    if not path.exists():
        return ""

    return path.read_text(encoding="utf-8").strip()


def load_current_calibration():
    calibration_files = calibration_recordings()

    if calibration_files:
        return load_calibration_from_csv_name(calibration_files[0].name)

    calibration_file = SUMMARIES_DIR / "latest_calibration.json"

    if not calibration_file.exists():
        return None, {}

    with open(calibration_file, "r", encoding="utf-8") as file:
        calibration = json.load(file)

    source_csv = calibration.get("source_csv", "")
    source_metadata = {}

    if source_csv:
        source_metadata = load_metadata(DATA_DIR / source_csv)

    return calibration, source_metadata


def normalized_metadata_value(value):
    return str(value or "").strip().lower()


def metadata_setup_id(metadata):
    return normalized_metadata_value(
        metadata.get("calibration_setup_id")
        or metadata.get("sensor_setup_id")
        or metadata.get("setup_id")
    )


def calibration_matches(metadata, muscle, side, setup_id=""):
    required_muscle = normalized_metadata_value(muscle)
    required_side = normalized_metadata_value(side)
    required_setup_id = normalized_metadata_value(setup_id)
    calibration_setup_id = metadata_setup_id(metadata)

    if not required_muscle or not required_side:
        return False

    if calibration_setup_id != required_setup_id:
        return False

    return (
        normalized_metadata_value(metadata.get("muscle")) == required_muscle
        and normalized_metadata_value(metadata.get("side")) == required_side
    )


def compatible_calibration_recordings(muscle, side, setup_id=""):
    return [
        csv_file
        for csv_file in calibration_recordings()
        if calibration_matches(load_metadata(csv_file), muscle, side, setup_id)
    ]


def load_compatible_current_calibration(muscle, side, setup_id=""):
    for csv_file in compatible_calibration_recordings(muscle, side, setup_id):
        calibration, metadata = load_calibration_from_csv_name(csv_file.name)

        if calibration is not None:
            return calibration, metadata

    return None, {}


def metadata_marks_excluded(metadata):
    structured_marker_text = " ".join(
        str(metadata.get(key, ""))
        for key in ("test_type", "status", "include", "excluded")
    ).lower()
    notes = str(metadata.get("notes", "")).lower()

    if "diagnostic" in structured_marker_text or "diagnostic" in notes:
        return True

    return any(marker in structured_marker_text for marker in ("exclude", "excluded"))


def is_valid_workout_recording(csv_file):
    if csv_file.parent != DATA_DIR:
        return False

    metadata = load_metadata(csv_file)

    if not metadata:
        return False

    if metadata.get("data_type") != "real":
        return False

    if metadata.get("test_type") != "workout_set":
        return False

    if metadata_marks_excluded(metadata):
        return False

    return True


def valid_workout_recordings():
    if not DATA_DIR.exists():
        return []

    return sorted(
        [csv_file for csv_file in DATA_DIR.glob("*.csv") if is_valid_workout_recording(csv_file)],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def calibration_recordings():
    if not DATA_DIR.exists():
        return []

    calibration_files = []

    for csv_file in DATA_DIR.glob("*.csv"):
        metadata = load_metadata(csv_file)

        if metadata.get("test_type") == "calibration":
            calibration_files.append(csv_file)

    return sorted(calibration_files, key=lambda path: path.stat().st_mtime, reverse=True)


def load_calibration_from_csv_name(csv_name):
    if not csv_name:
        return None, {}

    csv_file = DATA_DIR / csv_name

    if not csv_file.exists():
        return None, {}

    _, values = load_csv_signal(csv_file)
    calibration = calculate_calibration(values)
    calibration["source_csv"] = csv_file.name
    return calibration, load_metadata(csv_file)


def summary_path_for_csv(csv_file):
    return SUMMARIES_DIR / f"{csv_file.stem}_summary.txt"


def graph_path_for_csv(csv_file):
    return GRAPHS_DIR / f"{csv_file.stem}_reps.png"


def output_is_current(source_file, output_file):
    return output_file.exists() and output_file.stat().st_mtime >= source_file.stat().st_mtime


def current_summary_path_for_csv(csv_file):
    summary_file = summary_path_for_csv(csv_file)
    return summary_file if output_is_current(csv_file, summary_file) else None


def current_graph_path_for_csv(csv_file):
    graph_file = graph_path_for_csv(csv_file)
    return graph_file if output_is_current(csv_file, graph_file) else None


def summary_value(summary_text, label):
    pattern = re.compile(rf"^{re.escape(label)}:\s*(.+)$", re.MULTILINE)
    match = pattern.search(summary_text)
    return match.group(1).strip() if match else ""


def user_insights(summary_text):
    marker = "\nUser Insights\n"

    if marker not in summary_text:
        return []

    return [
        line.strip()
        for line in summary_text.split(marker, 1)[1].splitlines()
        if line.strip()
    ]


def recording_summary(csv_file):
    summary_text = read_text(summary_path_for_csv(csv_file))

    return {
        "summary_text": summary_text,
        "detected_reps": summary_value(summary_text, "Total reps"),
        "average_normalized_rep_activation": summary_value(
            summary_text,
            "Average normalized rep activation",
        ),
        "peak_normalized_activation": summary_value(
            summary_text,
            "Peak normalized activation",
        ),
        "activation_trend": summary_value(
            summary_text,
            "Activation change from first half to second half",
        ),
        "average_rep_duration": summary_value(summary_text, "Average rep duration"),
        "user_insights": user_insights(summary_text),
    }


def appears_derived_or_trimmed(metadata):
    notes = str(metadata.get("notes", "")).lower()
    return any(marker in notes for marker in ("derived", "trimmed", "trim "))


def appears_diagnostic(csv_file, metadata):
    marker_text = " ".join([
        str(csv_file),
        str(metadata.get("test_type", "")),
        str(metadata.get("notes", "")),
    ]).lower()
    return "diagnostic" in marker_text


def comparison_blocks(comparison_text):
    sections = {}
    current = None
    lines = []

    for line in comparison_text.splitlines():
        if line in {"Set 1", "Set 2", "Full-Set Comparison", "Normalized Comparison", "User Insights"}:
            if current is not None:
                sections[current] = "\n".join(lines).strip()
            current = line
            lines = []
        elif current is not None:
            lines.append(line)

    if current is not None:
        sections[current] = "\n".join(lines).strip()

    return sections


def indented_value(block, label):
    pattern = re.compile(rf"^\s*{re.escape(label)}:\s*(.+)$", re.MULTILINE)
    match = pattern.search(block)
    return match.group(1).strip() if match else ""


def latest_comparison():
    comparison_text = read_text(SUMMARIES_DIR / "latest_set_comparison.txt")

    if not comparison_text:
        return None

    sections = comparison_blocks(comparison_text)
    set_1 = sections.get("Set 1", "")
    set_2 = sections.get("Set 2", "")
    full_set = sections.get("Full-Set Comparison", "")
    insights = sections.get("User Insights", "")

    return {
        "set_1": {
            "filename": indented_value(set_1, "File name"),
            "rep_count": indented_value(set_1, "Rep count"),
            "average_normalized_activation": indented_value(
                set_1,
                "Average normalized rep activation",
            ),
            "peak_normalized_activation": indented_value(
                set_1,
                "Peak normalized rep activation",
            ),
        },
        "set_2": {
            "filename": indented_value(set_2, "File name"),
            "rep_count": indented_value(set_2, "Rep count"),
            "average_normalized_activation": indented_value(
                set_2,
                "Average normalized rep activation",
            ),
            "peak_normalized_activation": indented_value(
                set_2,
                "Peak normalized rep activation",
            ),
        },
        "activation_trend_comparison": indented_value(
            full_set,
            "Activation trend comparison",
        ),
        "user_insights": [line for line in insights.splitlines() if line.strip()],
    }


def comparison_mentions_file(comparison, csv_file):
    if comparison is None:
        return False

    filename = csv_file.name
    return filename in (
        comparison["set_1"]["filename"],
        comparison["set_2"]["filename"],
    )
