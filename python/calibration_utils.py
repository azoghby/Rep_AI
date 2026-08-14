import csv
import json
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
SUMMARIES_DIR = BASE_DIR / "summaries"
LATEST_CALIBRATION_FILE = SUMMARIES_DIR / "latest_calibration.json"

LOW_PERCENTILE = 10
HIGH_PERCENTILE = 10
MIN_SIGNAL_RANGE = 1.0


def signal_column(fieldnames):
    for column_name in ("signal_value", "emg_value"):
        if column_name in fieldnames:
            return column_name

    raise ValueError("CSV must contain either a signal_value or emg_value column.")


def load_csv_signal(csv_file):
    times = []
    values = []

    with open(csv_file, "r", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(f"{csv_file.name} does not have a header row.")

        value_key = signal_column(reader.fieldnames)

        for row in reader:
            times.append(float(row["time_ms"]) / 1000)
            values.append(float(row[value_key]))

    if not values:
        raise ValueError(f"{csv_file.name} does not contain any signal rows.")

    return times, values


def average(values):
    if not values:
        return 0

    return sum(values) / len(values)


def low_percentile_average(values, percentile=LOW_PERCENTILE):
    sorted_values = sorted(values)
    count = max(1, int(len(sorted_values) * (percentile / 100)))
    return average(sorted_values[:count])


def high_percentile_average(values, percentile=HIGH_PERCENTILE):
    sorted_values = sorted(values)
    count = max(1, int(len(sorted_values) * (percentile / 100)))
    return average(sorted_values[-count:])


def calculate_calibration(values):
    baseline = low_percentile_average(values)
    max_flex = high_percentile_average(values)
    signal_range = max_flex - baseline

    return {
        "baseline": baseline,
        "max_flex": max_flex,
        "signal_range": signal_range,
        "low_percentile": LOW_PERCENTILE,
        "high_percentile": HIGH_PERCENTILE,
        "usable": signal_range >= MIN_SIGNAL_RANGE,
    }


def normalize_values(values, calibration):
    signal_range = calibration.get("signal_range", 0)

    if signal_range < MIN_SIGNAL_RANGE:
        return [0 for _ in values]

    baseline = calibration["baseline"]
    return [max(0, (value - baseline) / signal_range * 100) for value in values]


def save_latest_calibration(calibration):
    SUMMARIES_DIR.mkdir(exist_ok=True)

    with open(LATEST_CALIBRATION_FILE, "w", encoding="utf-8") as file:
        json.dump(calibration, file, indent=2)
        file.write("\n")

    return LATEST_CALIBRATION_FILE


def calibration_summary_lines(calibration):
    return [
        "Latest Calibration",
        f"Source CSV: {calibration['source_csv']}",
        f"Generated at: {calibration['generated_at']}",
        f"Baseline average: {calibration['baseline']:.1f}",
        f"Max flex value: {calibration['max_flex']:.1f}",
        f"Signal range: {calibration['signal_range']:.1f}",
        f"Low percentile: {calibration['low_percentile']}",
        f"High percentile: {calibration['high_percentile']}",
        f"Usable calibration: {'yes' if calibration['usable'] else 'no'}",
    ]


def load_latest_calibration():
    if not LATEST_CALIBRATION_FILE.exists():
        return None

    with open(LATEST_CALIBRATION_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def normalized_metadata_value(value):
    return str(value or "").strip().lower()


def metadata_setup_id(metadata):
    return normalized_metadata_value(
        metadata.get("calibration_setup_id")
        or metadata.get("sensor_setup_id")
        or metadata.get("setup_id")
    )


def calibration_marked_questionable(metadata):
    if metadata.get("calibration_questionable") is True:
        return True

    status = normalized_metadata_value(metadata.get("calibration_status"))
    return "questionable" in status or "stale" in status


def calibration_metadata_matches(recording_metadata, calibration_metadata):
    required_muscle = normalized_metadata_value(recording_metadata.get("muscle"))
    required_side = normalized_metadata_value(recording_metadata.get("side"))
    required_setup_id = metadata_setup_id(recording_metadata)
    calibration_setup_id = metadata_setup_id(calibration_metadata)

    if not required_muscle or not required_side:
        return True

    if calibration_setup_id != required_setup_id:
        return False

    return (
        normalized_metadata_value(calibration_metadata.get("muscle")) == required_muscle
        and normalized_metadata_value(calibration_metadata.get("side")) == required_side
    )


def metadata_is_calibration(metadata):
    return metadata.get("test_type") == "calibration"


def calibration_recordings():
    from recording_metadata import load_metadata

    data_dir = BASE_DIR / "data"

    if not data_dir.exists():
        return []

    return sorted(
        [
            csv_file
            for csv_file in data_dir.glob("*.csv")
            if metadata_is_calibration(load_metadata(csv_file))
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def compatible_calibration_recordings(recording_metadata):
    from recording_metadata import load_metadata

    return [
        csv_file
        for csv_file in calibration_recordings()
        if calibration_metadata_matches(recording_metadata, load_metadata(csv_file))
    ]


def calibration_csv_path(csv_name):
    csv_path = Path(csv_name)

    if csv_path.is_absolute():
        return csv_path

    return BASE_DIR / "data" / csv_name


def load_calibration_from_csv(csv_name):
    if not csv_name:
        return None, {}

    from recording_metadata import load_metadata

    csv_file = calibration_csv_path(csv_name)

    if not csv_file.exists():
        return None, {}

    _, values = load_csv_signal(csv_file)
    calibration = calculate_calibration(values)
    calibration["source_csv"] = csv_file.name
    return calibration, load_metadata(csv_file)


def load_newest_compatible_calibration(recording_metadata):
    for csv_file in compatible_calibration_recordings(recording_metadata):
        calibration, metadata = load_calibration_from_csv(csv_file.name)

        if calibration is not None:
            return calibration, metadata

    return None, {}


def build_latest_calibration_for_csv(csv_file, generated_at=None):
    _, values = load_csv_signal(csv_file)
    calibration = calculate_calibration(values)
    calibration["source_csv"] = Path(csv_file).name
    calibration["generated_at"] = (
        generated_at if generated_at is not None else datetime.now().isoformat(timespec="seconds")
    )
    return calibration


def latest_calibration_with_metadata():
    csv_files = calibration_recordings()

    if csv_files:
        calibration, metadata = load_calibration_from_csv(csv_files[0].name)

        if calibration is not None:
            calibration["generated_at"] = datetime.now().isoformat(timespec="seconds")
            return calibration, metadata

    calibration = load_latest_calibration()

    if calibration is None:
        return None, {}

    source_csv = calibration.get("source_csv", "")
    _, metadata = load_calibration_from_csv(source_csv) if source_csv else (None, {})
    return calibration, metadata


def load_calibration_for_recording(csv_file, metadata=None):
    from recording_metadata import load_metadata

    csv_file = Path(csv_file)
    metadata = metadata if metadata is not None else load_metadata(csv_file)

    if calibration_marked_questionable(metadata):
        return None

    calibration_csv = metadata.get("calibration_csv", "")

    if calibration_csv:
        calibration, calibration_metadata = load_calibration_from_csv(calibration_csv)

        if calibration is None:
            return None

        if not calibration_metadata_matches(metadata, calibration_metadata):
            return None

        return calibration

    calibration, calibration_metadata = load_newest_compatible_calibration(metadata)

    if calibration is None:
        return None

    if not calibration_metadata_matches(metadata, calibration_metadata):
        return None

    return calibration
