import csv
import time
from pathlib import Path

from .base import SignalReading


class ReplaySignalSource:
    def __init__(self, csv_file, realtime=False, loop=False):
        self.csv_file = Path(csv_file)
        self.realtime = realtime
        self.loop = loop
        self._rows = []
        self._index = 0
        self._started_at_ms = None
        self._source_started_at_ms = 0

    def connect(self):
        self._rows = self._load_rows()
        self._index = 0
        self._started_at_ms = self._now_ms()
        self._source_started_at_ms = self._rows[0][0] if self._rows else 0

    def disconnect(self):
        self._rows = []
        self._index = 0
        self._started_at_ms = None

    def read(self):
        if not self._rows:
            return None

        if self._index >= len(self._rows):
            if not self.loop:
                return None
            self._index = 0
            self._started_at_ms = self._now_ms()
            self._source_started_at_ms = self._rows[0][0]

        source_time_ms, value, raw_sample = self._rows[self._index]
        elapsed_ms = source_time_ms - self._source_started_at_ms
        host_time_ms = int(self._started_at_ms + elapsed_ms)

        if self.realtime:
            delay_ms = host_time_ms - self._now_ms()
            if delay_ms > 0:
                time.sleep(delay_ms / 1000)

        self._index += 1
        return SignalReading(
            host_time_ms=host_time_ms,
            signal_value=value,
            raw_sample=raw_sample,
            device_time_ms=int(source_time_ms),
        )

    def read_many(self, max_samples=250, max_duration_seconds=0.05):
        if self.realtime:
            reading = self.read()
            return [reading] if reading is not None else []

        readings = []
        deadline = time.monotonic() + max(0, max_duration_seconds)

        while len(readings) < max_samples:
            if readings and time.monotonic() >= deadline:
                break

            reading = self.read()

            if reading is None:
                break

            readings.append(reading)

        return readings

    def _load_rows(self):
        with open(self.csv_file, "r", newline="") as file:
            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                raise ValueError(f"{self.csv_file.name} does not have a header row.")

            value_key = self._signal_column(reader.fieldnames)
            rows = []

            for row in reader:
                try:
                    rows.append((float(row["time_ms"]), float(row[value_key]), row.get(value_key, "")))
                except (KeyError, TypeError, ValueError):
                    continue

        if not rows:
            raise ValueError(f"{self.csv_file.name} does not contain replayable signal rows.")

        return rows

    @staticmethod
    def _signal_column(fieldnames):
        for column_name in ("signal_value", "emg_value"):
            if column_name in fieldnames:
                return column_name

        raise ValueError("CSV must contain either a signal_value or emg_value column.")

    @staticmethod
    def _now_ms():
        return int(time.time() * 1000)
