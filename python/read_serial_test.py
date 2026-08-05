import time
import serial

from serial_port import find_arduino_port


PORT = find_arduino_port()
BAUD_RATE = 115200

with serial.Serial(PORT, BAUD_RATE, timeout=2) as ser:
    time.sleep(2)

    for _ in range(500):
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        print(line)
