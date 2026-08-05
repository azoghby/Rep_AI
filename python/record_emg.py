import serial
import csv
from datetime import datetime

from serial_port import find_arduino_port


PORT = find_arduino_port()
BAUD_RATE = 115200

exercise_name = input("Exercise name: ")
output_file = f"../data/{exercise_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

ser = serial.Serial(PORT, BAUD_RATE)

print(f"Recording data to {output_file}")
print("Press Ctrl+C to stop.")

with open(output_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["time_ms", "emg_value"])

    try:
        while True:
            raw_line = ser.readline().decode("utf-8").strip()

            try:
                time_text, value_text = raw_line.split(",")
                time_ms = int(time_text)
                emg_value = int(value_text)

                writer.writerow([time_ms, emg_value])
                print(time_ms, emg_value)

            except ValueError:
                pass

    except KeyboardInterrupt:
        print("Recording stopped.")
