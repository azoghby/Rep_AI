import csv
import time
from datetime import datetime
from pathlib import Path

import serial

from recording_metadata import save_metadata
from serial_port import find_arduino_port


BAUD_RATE = 115200
SEGMENT_SECONDS = 5
CALIBRATION_STEPS = [
    ("relaxed", "Relax your muscle"),
    ("flexed", "Flex your muscle"),
    ("relaxed", "Relax your muscle"),
    ("flexed", "Flex your muscle"),
    ("relaxed", "Relax your muscle"),
]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def parse_emg_line(raw_line):
    time_text, value_text = raw_line.split(",")
    return int(time_text), int(value_text)


def record_segment(ser, writer, phase, instruction):
    print(f"{instruction} for {SEGMENT_SECONDS} seconds...")
    segment_values = []
    end_time = time.monotonic() + SEGMENT_SECONDS

    while time.monotonic() < end_time:
        raw_line = ser.readline().decode("utf-8", errors="ignore").strip()

        try:
            time_ms, emg_value = parse_emg_line(raw_line)
            writer.writerow([time_ms, emg_value, phase])
            segment_values.append(emg_value)
            print(time_ms, emg_value, phase)
        except ValueError:
            pass

    return segment_values


def average(values):
    if not values:
        return 0

    return sum(values) / len(values)


def main():
    muscle = input("Muscle: ").strip()
    side = input("Side (right/left): ").strip()
    notes = input("Notes (blank if none): ").strip()

    timestamp = datetime.now()
    timestamp_text = timestamp.strftime("%Y%m%d_%H%M%S")
    filename_parts = ["calibration"]

    if muscle:
        filename_parts.append(muscle)

    if side:
        filename_parts.append(side)

    output_file = DATA_DIR / f"{'_'.join(filename_parts)}_{timestamp_text}.csv"
    metadata = {
        "csv_filename": output_file.name,
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "exercise_name": "calibration",
        "muscle": muscle,
        "side": side,
        "weight": "",
        "expected_reps": "",
        "data_type": "real",
        "test_type": "calibration",
        "notes": notes,
    }

    port = find_arduino_port()

    with serial.Serial(port, BAUD_RATE, timeout=2) as ser:
        metadata_file = save_metadata(output_file, metadata)
        time.sleep(2)
        ser.reset_input_buffer()

        print(f"Recording calibration to: {output_file.resolve()}")
        print(f"Saved metadata to: {metadata_file.resolve()}")
        print("Calibration sequence: relaxed, flexed, relaxed, flexed, relaxed.")

        relaxed_values = []
        flexed_values = []

        with open(output_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["time_ms", "emg_value", "phase"])

            for phase, instruction in CALIBRATION_STEPS:
                values = record_segment(ser, writer, phase, instruction)

                if phase == "relaxed":
                    relaxed_values.extend(values)
                else:
                    flexed_values.extend(values)

        baseline_average = average(relaxed_values)
        max_flex_value = max(flexed_values) if flexed_values else 0
        signal_range = max_flex_value - baseline_average

        print()
        print("Calibration Stats")
        print(f"Baseline average: {baseline_average:.1f}")
        print(f"Max flex value: {max_flex_value:.1f}")
        print(f"Signal range: {signal_range:.1f}")


if __name__ == "__main__":
    main()
