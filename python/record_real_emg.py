import csv
import time
from datetime import datetime
from pathlib import Path

import serial

from recording_metadata import save_metadata
from serial_port import find_arduino_port


BAUD_RATE = 115200

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def prompt_metadata():
    exercise_name = input("Exercise name: ").strip()
    muscle = input("Muscle: ").strip()
    side = input("Side (right/left): ").strip()
    weight = input("Weight (blank if none): ").strip()
    expected_reps = input("Expected reps (blank if unknown): ").strip()
    test_type = input("Test type (flex_test/workout_set): ").strip()
    notes = input("Notes (blank if none): ").strip()

    if exercise_name == "":
        exercise_name = "real_emg_test"

    if test_type == "":
        test_type = "workout_set"

    return {
        "exercise_name": exercise_name,
        "muscle": muscle,
        "side": side,
        "weight": weight,
        "expected_reps": expected_reps,
        "data_type": "real",
        "test_type": test_type,
        "notes": notes,
    }


def parse_emg_line(raw_line):
    time_text, value_text = raw_line.split(",")
    return int(time_text), int(value_text)


def main():
    metadata = prompt_metadata()
    timestamp = datetime.now()
    timestamp_text = timestamp.strftime("%Y%m%d_%H%M%S")
    output_file = DATA_DIR / f"{metadata['exercise_name']}_{timestamp_text}.csv"
    metadata["csv_filename"] = output_file.name
    metadata["timestamp"] = timestamp.isoformat(timespec="seconds")

    port = find_arduino_port()

    with serial.Serial(port, BAUD_RATE, timeout=2) as ser:
        metadata_file = save_metadata(output_file, metadata)
        time.sleep(2)
        ser.reset_input_buffer()

        print(f"Recording real EMG to: {output_file.resolve()}")
        print(f"Saved metadata to: {metadata_file.resolve()}")
        print("Press Ctrl+C to stop recording.")

        with open(output_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["time_ms", "emg_value"])

            try:
                while True:
                    raw_line = ser.readline().decode("utf-8", errors="ignore").strip()

                    try:
                        time_ms, emg_value = parse_emg_line(raw_line)
                        writer.writerow([time_ms, emg_value])
                        print(time_ms, emg_value)
                    except ValueError:
                        pass

            except KeyboardInterrupt:
                print("\nRecording stopped.")


if __name__ == "__main__":
    main()
