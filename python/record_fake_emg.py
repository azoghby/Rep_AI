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

exercise_name = input("Exercise name: ").strip()
muscle = input("Muscle: ").strip()
side = input("Side (right/left): ").strip()
weight = input("Weight (blank if none): ").strip()
expected_reps = input("Expected reps (blank if unknown): ").strip()
data_type = input("Data type [fake]: ").strip()
notes = input("Notes (blank if none): ").strip()

if exercise_name == "":
    exercise_name = "fake_emg_test"

if data_type == "":
    data_type = "fake"

timestamp = datetime.now()
timestamp_text = timestamp.strftime("%Y%m%d_%H%M%S")
output_file = DATA_DIR / f"{exercise_name}_{timestamp_text}.csv"
metadata = {
    "csv_filename": output_file.name,
    "timestamp": timestamp.isoformat(timespec="seconds"),
    "exercise_name": exercise_name,
    "muscle": muscle,
    "side": side,
    "weight": weight,
    "expected_reps": expected_reps,
    "data_type": data_type,
    "notes": notes,
}

PORT = find_arduino_port()
with serial.Serial(PORT, BAUD_RATE, timeout=2) as ser:
    metadata_file = save_metadata(output_file, metadata)
    time.sleep(2)
    ser.reset_input_buffer()
    print("Starting fake set from rep 1...")
    ser.write(b"S\n")
    ser.flush()
    time.sleep(0.1)

    print(f"Recording to: {output_file}")
    print(f"Saved metadata to: {metadata_file}")
    print("Press Ctrl+C to stop recording.")

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time_ms", "signal_value"])

        try:
            while True:
                raw_line = ser.readline().decode("utf-8", errors="ignore").strip()

                try:
                    time_text, value_text = raw_line.split(",")
                    time_ms = int(time_text)
                    signal_value = int(value_text)

                    writer.writerow([time_ms, signal_value])
                    print(time_ms, signal_value)

                except ValueError:
                    pass

        except KeyboardInterrupt:
            print("\nRecording stopped.")
