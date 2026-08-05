import math
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
PYTHON_DIR = BASE_DIR / "python"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from compare_latest_sets import analyze_set  # noqa: E402
from detect_reps import read_signal  # noqa: E402


FLAT_SIGNAL_RANGE = 2.0
SMALL_SIGNAL_RANGE = 20.0
NEAR_MAX_FRACTION = 0.01
NEAR_MAX_MIN_WINDOW = 2.0
SATURATION_SAMPLE_PERCENT = 3.0
SATURATION_ACTIVE_SAMPLE_PERCENT = 8.0


def number_from_text(value):
    if value in ("", None):
        return None

    try:
        return float(str(value).strip().rstrip("%s"))
    except ValueError:
        return None


def percent_difference(value_a, value_b):
    lower = min(abs(value_a), abs(value_b))

    if lower == 0:
        return 0

    return abs(value_a - value_b) / lower * 100


def rep_consistency(set_analysis):
    durations = [
        rep["end_time"] - rep["start_time"]
        for rep in set_analysis.get("reps", [])
    ]

    if len(durations) < 2:
        return "Not enough reps"

    mean = sum(durations) / len(durations)

    if mean == 0:
        return "Not available"

    variance = sum((duration - mean) ** 2 for duration in durations) / len(durations)
    coefficient = math.sqrt(variance) / mean * 100
    return f"{coefficient:.1f}% duration CV"


def near_max_stats(values, active_values=None):
    if not values:
        return {
            "maximum_observed_signal": 0,
            "near_max_sample_percent": 0,
            "near_max_active_sample_percent": None,
        }

    maximum = max(values)
    minimum = min(values)
    window = max(NEAR_MAX_MIN_WINDOW, (maximum - minimum) * NEAR_MAX_FRACTION)
    near_max_count = sum(1 for value in values if value >= maximum - window)
    active_percent = None

    if active_values:
        active_near_max_count = sum(
            1 for value in active_values if value >= maximum - window
        )
        active_percent = active_near_max_count / len(active_values) * 100

    return {
        "maximum_observed_signal": maximum,
        "near_max_sample_percent": near_max_count / len(values) * 100,
        "near_max_active_sample_percent": active_percent,
    }


def recording_quality(
    csv_file,
    metadata,
    summary,
    graph_path,
    calibration=None,
    calibration_metadata=None,
    required_side=None,
    expected_reps=None,
):
    warnings = []
    info = []

    if not csv_file or not csv_file.exists():
        return {"warnings": ["Recording is missing."], "info": [], "saturation": None}

    try:
        _, values = read_signal(csv_file)
        set_analysis = analyze_set(csv_file, calibration)
    except Exception as error:  # pragma: no cover - UI guardrail
        return {
            "warnings": [f"Recording could not be analyzed: {error}"],
            "info": [],
            "saturation": None,
        }

    signal_range = max(values) - min(values) if values else 0

    if signal_range <= FLAT_SIGNAL_RANGE:
        warnings.append("Flat or nearly flat signal heuristic triggered.")
    elif signal_range <= SMALL_SIGNAL_RANGE:
        warnings.append("Small signal-range heuristic triggered.")

    expected = expected_reps if expected_reps not in ("", None) else metadata.get("expected_reps", "")
    detected = summary.get("detected_reps") or str(set_analysis["rep_count"])

    if expected not in ("", None) and str(expected).strip() != str(detected).strip():
        warnings.append("Expected and detected reps differ.")

    calibration_side = (calibration_metadata or {}).get("side", "")

    if calibration is None:
        warnings.append("Missing side-specific calibration.")
    elif required_side and calibration_side and calibration_side != required_side:
        warnings.append(
            f"Calibration side is {calibration_side}; this slot expects {required_side}."
        )

    if not graph_path.exists():
        warnings.append("Rep graph is missing or stale.")
    else:
        summary_path = graph_path.parent.parent / "summaries" / f"{csv_file.stem}_summary.txt"

        if summary_path.exists() and graph_path.stat().st_mtime < summary_path.stat().st_mtime:
            warnings.append("Rep graph appears older than the summary.")

    diagnostic_text = " ".join([
        str(csv_file),
        str(metadata.get("test_type", "")),
        str(metadata.get("notes", "")),
    ]).lower()
    exclusion_text = " ".join([
        str(metadata.get("test_type", "")),
        str(metadata.get("status", "")),
        str(metadata.get("include", "")),
        str(metadata.get("excluded", "")),
    ]).lower()

    if "diagnostic" in diagnostic_text or "exclude" in exclusion_text:
        warnings.append("Recording is diagnostic or excluded.")

    active_values = [
        value
        for rep in set_analysis.get("reps", [])
        for value in rep.get("values", [])
    ]
    saturation = near_max_stats(values, active_values)

    if (
        saturation["near_max_sample_percent"] >= SATURATION_SAMPLE_PERCENT
        or (
            saturation["near_max_active_sample_percent"] is not None
            and saturation["near_max_active_sample_percent"] >= SATURATION_ACTIVE_SAMPLE_PERCENT
        )
    ):
        warnings.append("Possible clipping/saturation heuristic triggered.")

    info.append(f"Signal range: {signal_range:.1f}")
    return {
        "warnings": warnings,
        "info": info,
        "saturation": saturation,
        "analysis": set_analysis,
        "rep_consistency": rep_consistency(set_analysis),
    }
