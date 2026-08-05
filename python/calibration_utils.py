import csv
import json
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


def load_latest_calibration():
    if not LATEST_CALIBRATION_FILE.exists():
        return None

    with open(LATEST_CALIBRATION_FILE, "r", encoding="utf-8") as file:
        return json.load(file)
