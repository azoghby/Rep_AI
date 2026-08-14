import csv

from acquisition.serial_source import SerialSignalSource
from acquisition.replay_source import ReplaySignalSource
from acquisition.base import SignalReading
from set_lifecycle import SetLifecycle, SetLifecycleConfig, SetState, write_recording_atomic


class FakeSerial:
    def __init__(self, lines):
        self.lines = [line.encode("utf-8") for line in lines]
        self.is_open = True

    @property
    def in_waiting(self):
        return len(self.lines)

    def readline(self):
        if not self.lines:
            return b""

        return self.lines.pop(0)

    def close(self):
        self.is_open = False


def lifecycle(**overrides):
    flow = SetLifecycle(
        config=SetLifecycleConfig(
            smoothing_window=1,
            baseline_sample_count=5,
            backlog_drift_warning_seconds=0.1,
            **overrides,
        )
    )
    flow.start_countdown()
    flow.start_recording(calibration=None)
    return flow


def test_serial_read_many_drains_pending_device_samples(monkeypatch):
    lines = [f"{index * 10},{100 + index}\n" for index in range(100)]
    source = SerialSignalSource("fake")
    source._serial = FakeSerial(lines)
    monkeypatch.setattr("acquisition.serial_source.time.time", lambda: 1000.0)

    readings = source.read_many(max_samples=100)

    assert len(readings) == 100
    assert [reading.device_time_ms for reading in readings] == list(range(0, 1000, 10))
    assert readings[0].signal_value == 100
    assert readings[-1].signal_value == 199


def test_serial_read_many_respects_batch_limit(monkeypatch):
    lines = [f"{index * 10},{index}\n" for index in range(100)]
    source = SerialSignalSource("fake")
    source._serial = FakeSerial(lines)
    monkeypatch.setattr("acquisition.serial_source.time.time", lambda: 1000.0)

    readings = source.read_many(max_samples=25)

    assert len(readings) == 25
    assert source._serial.in_waiting == 75


def test_serial_parser_handles_malformed_and_value_only_lines():
    assert SerialSignalSource._parse_serial_line("") == (None, None)
    assert SerialSignalSource._parse_serial_line("not-a-number") == (None, None)
    assert SerialSignalSource._parse_serial_line("123,not-a-number") == (None, None)
    assert SerialSignalSource._parse_serial_line("456,789") == (456, 789)
    assert SerialSignalSource._parse_serial_line("321") == (None, 321)


def test_lifecycle_uses_device_elapsed_for_recording_time_when_host_is_slow():
    flow = lifecycle()

    for index in range(100):
        flow.observe(
            SignalReading(
                host_time_ms=index * 200,
                device_time_ms=index * 10,
                signal_value=100 + index,
                raw_sample=f"{index * 10},{100 + index}",
            )
        )

    assert flow.state != SetState.ANALYZING
    assert len(flow.samples) == 100
    assert flow.samples[-1].time_ms == 990
    assert flow.samples[-1].host_elapsed_ms == 19800
    assert flow.samples[-1].timing_drift_ms == 18810
    assert "Host receipt timing and device timing differ" in flow.timing_drift_warning


def test_normal_device_stream_preserves_device_elapsed_time():
    flow = lifecycle()

    for index, device_time_ms in enumerate([100, 110, 120, 130]):
        flow.observe(
            SignalReading(
                host_time_ms=1_000 + index * 10,
                device_time_ms=device_time_ms,
                signal_value=100 + index,
                raw_sample=f"{device_time_ms},{100 + index}",
            )
        )

    assert [sample.time_ms for sample in flow.samples] == [0, 10, 20, 30]
    assert flow.startup_discarded_samples == 0
    assert flow.device_time_discontinuities == []


def test_stale_startup_device_sample_is_discarded_before_origin():
    flow = lifecycle()

    for index, device_time_ms in enumerate([90, 1_395_100, 1_395_110, 1_395_120]):
        flow.observe(
            SignalReading(
                host_time_ms=10_000,
                device_time_ms=device_time_ms,
                signal_value=100 + index,
                raw_sample=f"{device_time_ms},{100 + index}",
            )
        )

    assert [sample.device_time_ms for sample in flow.samples] == [
        1_395_100,
        1_395_110,
        1_395_120,
    ]
    assert [sample.time_ms for sample in flow.samples] == [0, 10, 20]
    assert flow.samples[-1].time_ms < 1_000
    assert flow.startup_discarded_samples == 1
    assert flow.device_time_discontinuities[0]["type"] == "stale_startup_sample_discarded"


def test_moderate_device_time_gap_is_not_classified_as_stale_startup():
    flow = lifecycle()

    flow.observe(SignalReading(host_time_ms=0, device_time_ms=100, signal_value=100))
    flow.observe(SignalReading(host_time_ms=600, device_time_ms=700, signal_value=101))

    assert len(flow.samples) == 2
    assert [sample.time_ms for sample in flow.samples] == [0, 600]
    assert flow.startup_discarded_samples == 0


def test_large_nonstartup_gap_is_diagnosed_but_preserved():
    flow = lifecycle()

    flow.observe(SignalReading(host_time_ms=0, device_time_ms=100, signal_value=100))
    flow.observe(SignalReading(host_time_ms=6_500, device_time_ms=6_600, signal_value=101))

    assert len(flow.samples) == 2
    assert [sample.time_ms for sample in flow.samples] == [0, 6_500]
    assert flow.startup_discarded_samples == 0
    assert flow.device_time_discontinuities[0]["type"] == "device_time_gap"


def test_arduino_millis_rollover_remains_monotonic(monkeypatch):
    lines = [
        "4294967290,100\n",
        "4,101\n",
        "14,102\n",
    ]
    source = SerialSignalSource("fake")
    source._serial = FakeSerial(lines)
    monkeypatch.setattr("acquisition.serial_source.time.time", lambda: 1000.0)

    readings = source.read_many(max_samples=3)

    assert [reading.device_time_ms for reading in readings] == [
        4_294_967_290,
        4_294_967_300,
        4_294_967_310,
    ]


def test_replay_acquisition_keeps_recorded_timeline(tmp_path, monkeypatch):
    csv_file = tmp_path / "replay.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_ms", "emg_value"])
        writer.writerow([90, 100])
        writer.writerow([1_395_100, 101])
        writer.writerow([1_395_110, 102])

    monkeypatch.setattr("acquisition.replay_source.time.time", lambda: 1000.0)
    source = ReplaySignalSource(csv_file, realtime=False)
    source.connect()

    readings = source.read_many(max_samples=3)

    assert [reading.device_time_ms for reading in readings] == [90, 1_395_100, 1_395_110]
    assert [reading.host_time_ms for reading in readings] == [1_000_000, 2_395_010, 2_395_020]


def test_recording_writer_preserves_device_and_host_timing_columns(tmp_path):
    flow = lifecycle()
    flow.observe(
        SignalReading(
            host_time_ms=1_000,
            device_time_ms=50_000,
            signal_value=123,
            raw_sample="50000,123",
        )
    )
    flow.observe(
        SignalReading(
            host_time_ms=1_200,
            device_time_ms=50_010,
            signal_value=130,
            raw_sample="50010,130",
        )
    )

    output_file = tmp_path / "set.csv"
    metadata_file = tmp_path / "set_metadata.json"
    write_recording_atomic(output_file, metadata_file, {"session_id": "s1"}, flow.samples)

    with open(output_file, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows[0]["time_ms"] == "0.000"
    assert rows[0]["device_time_ms"] == "50000"
    assert rows[0]["host_elapsed_ms"] == "0.000"
    assert rows[1]["time_ms"] == "10.000"
    assert rows[1]["host_elapsed_ms"] == "200.000"
    assert rows[1]["timing_drift_ms"] == "190.000"
