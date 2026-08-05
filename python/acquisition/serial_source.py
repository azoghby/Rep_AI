import time

import serial
from serial.tools import list_ports

from .base import SignalReading


BAUD_RATE = 115200


def available_serial_ports():
    ports = []

    for port in list_ports.comports():
        is_likely_arduino = any(
            marker in f"{port.device} {port.description}".lower()
            for marker in ("usbmodem", "arduino", "wchusbserial", "usbserial")
        )
        ports.append({
            "device": port.device,
            "description": port.description,
            "likely_arduino": is_likely_arduino,
        })

    return ports


class SerialSignalSource:
    def __init__(self, port, baud_rate=BAUD_RATE, timeout=2):
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self._serial = None

    def connect(self):
        self._serial = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)
        time.sleep(2)
        self._serial.reset_input_buffer()

    def disconnect(self):
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
        self._serial = None

    def read(self):
        if self._serial is None:
            raise RuntimeError("Serial source is not connected.")

        raw_line = self._serial.readline().decode("utf-8", errors="ignore").strip()
        value = self._parse_signal_value(raw_line)

        if value is None:
            return None

        return SignalReading(host_time_ms=int(time.time() * 1000), signal_value=value)

    @staticmethod
    def _parse_signal_value(raw_line):
        if not raw_line:
            return None

        parts = [part.strip() for part in raw_line.split(",") if part.strip()]
        candidate = parts[-1] if parts else raw_line.strip()

        try:
            return float(candidate)
        except ValueError:
            return None
