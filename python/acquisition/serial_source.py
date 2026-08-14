import time

import serial
from serial.tools import list_ports

from .base import SignalReading


BAUD_RATE = 115200
ARDUINO_MILLIS_ROLLOVER = 2**32


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
        self._previous_device_raw_ms = None
        self._previous_device_time_ms = None
        self._device_rollover_offset_ms = 0

    def connect(self):
        self._serial = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)
        time.sleep(2)
        self._serial.reset_input_buffer()
        self._previous_device_raw_ms = None
        self._previous_device_time_ms = None
        self._device_rollover_offset_ms = 0

    def disconnect(self):
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
        self._serial = None

    def read(self):
        if self._serial is None:
            raise RuntimeError("Serial source is not connected.")

        raw_line = self._serial.readline().decode("utf-8", errors="ignore").strip()
        return self._reading_from_line(raw_line)

    def read_many(self, max_samples=250, max_duration_seconds=0.05):
        if self._serial is None:
            raise RuntimeError("Serial source is not connected.")

        readings = []
        deadline = time.monotonic() + max(0, max_duration_seconds)

        while len(readings) < max_samples:
            if readings and self._serial.in_waiting <= 0:
                break
            if readings and time.monotonic() >= deadline:
                break

            reading = self.read()

            if reading is not None:
                readings.append(reading)
                continue

            if self._serial.in_waiting <= 0:
                break

        return readings

    @staticmethod
    def _parse_serial_line(raw_line):
        if not raw_line:
            return None, None

        parts = [part.strip() for part in raw_line.split(",") if part.strip()]
        candidate = parts[-1] if parts else raw_line.strip()

        try:
            value = float(candidate)
        except ValueError:
            return None, None

        if len(parts) >= 2:
            try:
                return int(float(parts[0])), value
            except ValueError:
                return None, value

        return None, value

    @staticmethod
    def _parse_signal_value(raw_line):
        _, value = SerialSignalSource._parse_serial_line(raw_line)
        return value

    def _adjust_device_time(self, raw_device_time_ms):
        if raw_device_time_ms is None:
            return None

        if self._previous_device_raw_ms is not None:
            if raw_device_time_ms < self._previous_device_raw_ms:
                drop = self._previous_device_raw_ms - raw_device_time_ms

                if drop > ARDUINO_MILLIS_ROLLOVER / 2:
                    self._device_rollover_offset_ms += ARDUINO_MILLIS_ROLLOVER

        adjusted = raw_device_time_ms + self._device_rollover_offset_ms

        if (
            self._previous_device_time_ms is not None
            and adjusted < self._previous_device_time_ms
        ):
            return None

        self._previous_device_raw_ms = raw_device_time_ms
        self._previous_device_time_ms = adjusted
        return adjusted

    def _reading_from_line(self, raw_line):
        raw_device_time_ms, value = self._parse_serial_line(raw_line)

        if value is None:
            return None

        return SignalReading(
            host_time_ms=int(time.time() * 1000),
            signal_value=value,
            raw_sample=raw_line,
            device_time_ms=self._adjust_device_time(raw_device_time_ms),
        )
