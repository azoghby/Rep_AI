import csv
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class SetState(str, Enum):
    READY = "ready"
    COUNTDOWN = "countdown"
    RECORDING = "recording"
    POSSIBLE_END = "possible_end"
    ANALYZING = "analyzing"
    RESULTS = "results"
    CANCELLED = "cancelled"
    ERROR = "error"


VALID_TRANSITIONS = {
    SetState.READY: {SetState.COUNTDOWN, SetState.CANCELLED, SetState.ERROR},
    SetState.COUNTDOWN: {SetState.RECORDING, SetState.CANCELLED, SetState.ERROR},
    SetState.RECORDING: {SetState.POSSIBLE_END, SetState.ANALYZING, SetState.CANCELLED, SetState.ERROR},
    SetState.POSSIBLE_END: {SetState.RECORDING, SetState.ANALYZING, SetState.CANCELLED, SetState.ERROR},
    SetState.ANALYZING: {SetState.RESULTS, SetState.ERROR},
    SetState.RESULTS: {SetState.READY},
    SetState.CANCELLED: {SetState.READY},
    SetState.ERROR: {SetState.READY},
}


@dataclass(frozen=True)
class SetLifecycleConfig:
    smoothing_window: int = 15
    baseline_sample_count: int = 30
    minimum_active_duration_seconds: float = 0.75
    minimum_recording_duration_seconds: float = 10.0
    auto_stop_min_active_episodes: int = 3
    auto_stop_min_active_duration_seconds: float = 8.0
    auto_stop_min_elapsed_after_meaningful_activity_seconds: float = 6.0
    substantial_activity_duration_seconds: float = 0.50
    min_substantial_activity_gap_seconds: float = 0.75
    inactivity_duration_seconds: float = 6.0
    end_grace_period_seconds: float = 0.0
    post_activity_grace_period_seconds: float = 1.25
    activity_threshold_fraction: float = 0.12
    inactive_threshold_fraction: float = 0.075
    fallback_activity_threshold: float = 18.0
    fallback_inactive_threshold: float = 12.0
    backlog_drift_warning_seconds: float = 1.0
    startup_stale_device_gap_ms: int = 5_000
    startup_stale_host_window_ms: int = 1_000


@dataclass(frozen=True)
class SetSessionSpec:
    source_type: str
    output_file: Path
    metadata_file: Path
    metadata: dict
    port: str | None = None
    replay_csv: Path | None = None
    replay_realtime: bool = False

    @property
    def source_label(self):
        if self.source_type == "Replay CSV" and self.replay_csv is not None:
            return self.replay_csv.name

        return self.port or self.source_type


@dataclass(frozen=True)
class SetSample:
    time_ms: float
    host_time_ms: int
    emg_value: float
    smoothed_value: float
    active: bool
    raw_sample: str = ""
    device_time_ms: int | None = None
    host_elapsed_ms: float | None = None
    timing_drift_ms: float | None = None


@dataclass
class EndOfSetInference:
    config: SetLifecycleConfig = field(default_factory=SetLifecycleConfig)
    calibration: dict | None = None
    values: list[float] = field(default_factory=list)
    baseline: float | None = None
    threshold: float | None = None
    inactive_threshold: float | None = None
    first_host_time_ms: int | None = None
    activity_seen_at: float | None = None
    meaningful_activity_started_at: float | None = None
    last_elapsed_seconds: float | None = None
    active_duration_seconds: float = 0.0
    inactive_since_seconds: float | None = None
    active: bool = False
    active_started_at: float | None = None
    last_active_seconds: float | None = None
    substantial_activity_episodes: int = 0
    last_substantial_activity_ended_at: float | None = None
    auto_stop_armed_at: float | None = None
    auto_stop_arm_reason: str = ""

    def observe(self, reading):
        if self.first_host_time_ms is None:
            self.first_host_time_ms = reading.host_time_ms

        time_seconds = (reading.host_time_ms - self.first_host_time_ms) / 1000
        self.values.append(float(reading.signal_value))
        smoothed = self._smoothed_value()
        self._update_baseline_and_threshold()
        active = self._activity_state(smoothed)

        if active:
            if self.activity_seen_at is None:
                self.activity_seen_at = time_seconds
            if self.last_elapsed_seconds is not None:
                self.active_duration_seconds += max(0.0, time_seconds - self.last_elapsed_seconds)
            if (
                self.meaningful_activity_started_at is None
                and self.active_duration_seconds >= self.config.minimum_active_duration_seconds
            ):
                self.meaningful_activity_started_at = time_seconds
            if not self.active:
                self.active_started_at = time_seconds
            self.last_active_seconds = time_seconds
            self.inactive_since_seconds = None
        else:
            if self.active:
                self._complete_active_episode(time_seconds)

            if self.auto_stop_armed and self._post_activity_grace_elapsed(time_seconds):
                if self.inactive_since_seconds is None:
                    self.inactive_since_seconds = time_seconds
            else:
                self.inactive_since_seconds = None

        self.active = active

        if self.auto_stop_armed_at is None and self._auto_stop_should_arm(time_seconds):
            self.auto_stop_armed_at = time_seconds
            self.auto_stop_arm_reason = self._auto_stop_arm_reason(time_seconds)

        self.last_elapsed_seconds = time_seconds

        return {
            "elapsed_seconds": time_seconds,
            "smoothed_value": smoothed,
            "active": active,
            "baseline": self.baseline,
            "threshold": self.threshold,
            "inactive_threshold": self.inactive_threshold,
            "activity_seen": self.activity_seen,
            "has_meaningful_activity": self.has_meaningful_activity,
            "workout_set_established": self.workout_set_established(time_seconds),
            "auto_stop_armed": self.auto_stop_armed,
            "auto_stop_arm_reason": self.auto_stop_arm_reason,
            "substantial_activity_episodes": self.current_substantial_activity_episodes(time_seconds),
            "active_duration_seconds": self.active_duration_seconds,
            "inactivity_elapsed_seconds": self.inactivity_elapsed_seconds(time_seconds),
            "should_auto_finish": self.should_auto_finish(time_seconds),
        }

    @property
    def activity_seen(self):
        return self.activity_seen_at is not None

    @property
    def has_meaningful_activity(self):
        return self.meaningful_activity_started_at is not None

    def inactivity_elapsed_seconds(self, elapsed_seconds):
        if self.inactive_since_seconds is None:
            return 0.0

        return max(0.0, elapsed_seconds - self.inactive_since_seconds)

    def inactivity_remaining_seconds(self, elapsed_seconds):
        return max(
            0.0,
            self.config.inactivity_duration_seconds - self.inactivity_elapsed_seconds(elapsed_seconds),
        )

    def should_auto_finish(self, elapsed_seconds):
        if not self.auto_stop_armed:
            return False
        if self.inactive_since_seconds is None:
            return False

        required = self.config.inactivity_duration_seconds + self.config.end_grace_period_seconds
        return self.inactivity_elapsed_seconds(elapsed_seconds) >= required

    @property
    def auto_stop_armed(self):
        return self.auto_stop_armed_at is not None

    def current_substantial_activity_episodes(self, elapsed_seconds):
        episodes = self.substantial_activity_episodes

        if (
            self.active
            and self.active_started_at is not None
            and elapsed_seconds - self.active_started_at >= self.config.substantial_activity_duration_seconds
            and self._episode_gap_allows_counting(self.active_started_at)
        ):
            episodes += 1

        return episodes

    def workout_set_established(self, elapsed_seconds):
        if not self.has_meaningful_activity:
            return False
        if self.meaningful_activity_started_at is None:
            return False
        if (
            elapsed_seconds - self.meaningful_activity_started_at
            < self.config.auto_stop_min_elapsed_after_meaningful_activity_seconds
        ):
            return False

        return (
            self.current_substantial_activity_episodes(elapsed_seconds)
            >= self.config.auto_stop_min_active_episodes
            or self.active_duration_seconds >= self.config.auto_stop_min_active_duration_seconds
        )

    def _auto_stop_should_arm(self, elapsed_seconds):
        if elapsed_seconds < self.config.minimum_recording_duration_seconds:
            return False
        if not self.workout_set_established(elapsed_seconds):
            return False

        return True

    def _auto_stop_arm_reason(self, elapsed_seconds):
        if (
            self.current_substantial_activity_episodes(elapsed_seconds)
            >= self.config.auto_stop_min_active_episodes
        ):
            return "substantial_activity_episodes"

        if self.active_duration_seconds >= self.config.auto_stop_min_active_duration_seconds:
            return "active_duration"

        return ""

    def _post_activity_grace_elapsed(self, elapsed_seconds):
        if self.last_active_seconds is None:
            return False

        return (
            elapsed_seconds - self.last_active_seconds
            >= self.config.post_activity_grace_period_seconds
        )

    def _complete_active_episode(self, elapsed_seconds):
        if self.active_started_at is None:
            return

        duration = elapsed_seconds - self.active_started_at

        if (
            duration >= self.config.substantial_activity_duration_seconds
            and self._episode_gap_allows_counting(self.active_started_at)
        ):
            self.substantial_activity_episodes += 1

        if duration >= self.config.substantial_activity_duration_seconds:
            self.last_substantial_activity_ended_at = elapsed_seconds

        self.active_started_at = None

    def _episode_gap_allows_counting(self, active_started_at):
        if self.last_substantial_activity_ended_at is None:
            return True

        return (
            active_started_at - self.last_substantial_activity_ended_at
            >= self.config.min_substantial_activity_gap_seconds
        )

    def _activity_state(self, smoothed):
        if self.threshold is None:
            return False

        if self.active:
            leave_threshold = (
                self.inactive_threshold
                if self.inactive_threshold is not None
                else self.threshold
            )
            return smoothed > leave_threshold

        return smoothed >= self.threshold

    def _smoothed_value(self):
        window_size = max(1, self.config.smoothing_window)
        window = self.values[-window_size:]
        return sum(window) / len(window)

    def _update_baseline_and_threshold(self):
        if self.calibration and self.calibration.get("usable"):
            baseline = float(self.calibration.get("baseline", 0))
            signal_range = float(self.calibration.get("signal_range", 0))
            self.baseline = baseline
            self.threshold = baseline + signal_range * self.config.activity_threshold_fraction
            self.inactive_threshold = baseline + signal_range * self.config.inactive_threshold_fraction
            return

        if len(self.values) < self.config.baseline_sample_count:
            self.baseline = sum(self.values) / len(self.values)
            self.threshold = self.baseline + self.config.fallback_activity_threshold
            self.inactive_threshold = self.baseline + self.config.fallback_inactive_threshold
            return

        baseline_values = sorted(self.values)[: max(1, int(len(self.values) * 0.1))]
        self.baseline = sum(baseline_values) / len(baseline_values)
        observed_range = max(self.values) - self.baseline
        adaptive_threshold = observed_range * self.config.activity_threshold_fraction
        adaptive_inactive_threshold = observed_range * self.config.inactive_threshold_fraction
        self.threshold = self.baseline + max(self.config.fallback_activity_threshold, adaptive_threshold)
        self.inactive_threshold = self.baseline + max(
            self.config.fallback_inactive_threshold,
            adaptive_inactive_threshold,
        )


@dataclass
class SetLifecycle:
    state: SetState = SetState.READY
    config: SetLifecycleConfig = field(default_factory=SetLifecycleConfig)
    inference: EndOfSetInference | None = None
    samples: list[SetSample] = field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None
    output_file: Path | None = None
    metadata_file: Path | None = None
    analysis_key: str | None = None
    analysis_result: dict | None = None
    write_result: dict | None = None
    active_spec: SetSessionSpec | None = None
    source_object_id: int | None = None
    calibration: dict | None = None
    first_device_time_ms: int | None = None
    previous_sample_device_time_ms: int | None = None
    latest_timing_drift_ms: float | None = None
    timing_drift_warning: str = ""
    device_time_discontinuities: list[dict] = field(default_factory=list)
    startup_discarded_samples: int = 0
    error_message: str = ""
    cancelled_reason: str = ""

    def transition(self, new_state):
        new_state = SetState(new_state)
        if new_state not in VALID_TRANSITIONS[self.state]:
            raise ValueError(f"Invalid transition from {self.state.value} to {new_state.value}.")
        self.state = new_state

    def start_countdown(self, now=None):
        self.started_at = None
        self.completed_at = None
        self.error_message = ""
        self.cancelled_reason = ""
        self.transition(SetState.COUNTDOWN)
        return now if now is not None else time.monotonic()

    def begin_session(self, spec, now=None):
        self.active_spec = spec
        self.output_file = spec.output_file
        self.metadata_file = spec.metadata_file
        return self.start_countdown(now=now)

    def start_recording(self, calibration=None, now=None):
        self.samples = []
        self.calibration = calibration
        self.inference = EndOfSetInference(config=self.config, calibration=calibration)
        self.first_device_time_ms = None
        self.previous_sample_device_time_ms = None
        self.latest_timing_drift_ms = None
        self.timing_drift_warning = ""
        self.device_time_discontinuities = []
        self.startup_discarded_samples = 0
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.transition(SetState.RECORDING)
        return now if now is not None else time.monotonic()

    def remember_source_object(self, source):
        self.source_object_id = id(source)

    def observe(self, reading):
        if self.state not in (SetState.RECORDING, SetState.POSSIBLE_END):
            raise ValueError(f"Cannot observe samples while {self.state.value}.")
        if self.inference is None:
            raise ValueError("Recording inference has not been initialized.")

        if self._discard_stale_startup_sample_if_needed(reading):
            self.inference = EndOfSetInference(config=self.config, calibration=self.calibration)
            self.first_device_time_ms = None
            self.previous_sample_device_time_ms = None

        return self._observe_recording_sample(reading)

    def _discard_stale_startup_sample_if_needed(self, reading):
        device_time_ms = getattr(reading, "device_time_ms", None)

        if len(self.samples) != 1 or device_time_ms is None:
            return False

        first_sample = self.samples[0]
        first_device_time_ms = first_sample.device_time_ms

        if first_device_time_ms is None:
            return False

        device_gap_ms = device_time_ms - first_device_time_ms
        host_gap_ms = reading.host_time_ms - first_sample.host_time_ms

        if (
            device_gap_ms <= self.config.startup_stale_device_gap_ms
            or host_gap_ms > self.config.startup_stale_host_window_ms
        ):
            return False

        self.device_time_discontinuities.append({
            "type": "stale_startup_sample_discarded",
            "discarded_device_time_ms": first_device_time_ms,
            "next_device_time_ms": device_time_ms,
            "device_gap_ms": device_gap_ms,
            "host_gap_ms": host_gap_ms,
        })
        self.startup_discarded_samples += 1
        self.samples = []
        self.timing_drift_warning = (
            "Discarded a stale startup device timestamp before establishing recording time."
        )
        return True

    def _observe_recording_sample(self, reading):
        status = self.inference.observe(reading)
        host_elapsed_ms = status["elapsed_seconds"] * 1000
        device_time_ms = getattr(reading, "device_time_ms", None)
        sample_time_ms = host_elapsed_ms
        timing_drift_ms = None

        if device_time_ms is not None:
            if (
                self.previous_sample_device_time_ms is not None
                and device_time_ms - self.previous_sample_device_time_ms
                > self.config.startup_stale_device_gap_ms
            ):
                self.device_time_discontinuities.append({
                    "type": "device_time_gap",
                    "previous_device_time_ms": self.previous_sample_device_time_ms,
                    "device_time_ms": device_time_ms,
                    "device_gap_ms": device_time_ms - self.previous_sample_device_time_ms,
                    "host_elapsed_ms": host_elapsed_ms,
                })

            if self.first_device_time_ms is None:
                self.first_device_time_ms = device_time_ms

            device_elapsed_ms = max(0, device_time_ms - self.first_device_time_ms)
            sample_time_ms = device_elapsed_ms
            timing_drift_ms = host_elapsed_ms - device_elapsed_ms
            self.latest_timing_drift_ms = timing_drift_ms

            if abs(timing_drift_ms) / 1000 >= self.config.backlog_drift_warning_seconds:
                self.timing_drift_warning = (
                    "Host receipt timing and device timing differ by "
                    f"{timing_drift_ms / 1000:.2f}s."
                )
            self.previous_sample_device_time_ms = device_time_ms

        status["sample_elapsed_seconds"] = sample_time_ms / 1000
        status["device_time_ms"] = device_time_ms
        status["host_elapsed_seconds"] = status["elapsed_seconds"]
        status["timing_drift_seconds"] = (
            timing_drift_ms / 1000 if timing_drift_ms is not None else None
        )
        status["timing_drift_warning"] = self.timing_drift_warning
        sample = SetSample(
            time_ms=sample_time_ms,
            host_time_ms=reading.host_time_ms,
            emg_value=float(reading.signal_value),
            smoothed_value=status["smoothed_value"],
            active=status["active"],
            raw_sample=getattr(reading, "raw_sample", ""),
            device_time_ms=device_time_ms,
            host_elapsed_ms=host_elapsed_ms,
            timing_drift_ms=timing_drift_ms,
        )
        self.samples.append(sample)

        if status["should_auto_finish"]:
            self.finish()
        elif self.inference.inactive_since_seconds is not None and self.state == SetState.RECORDING:
            self.transition(SetState.POSSIBLE_END)
        elif status["active"] and self.state == SetState.POSSIBLE_END:
            self.transition(SetState.RECORDING)

        return status

    def acquisition_diagnostics(self):
        return {
            "startup_discarded_samples": self.startup_discarded_samples,
            "device_time_discontinuities": list(self.device_time_discontinuities),
            "timing_drift_warning": self.timing_drift_warning,
        }

    def finish(self):
        if self.state not in (SetState.RECORDING, SetState.POSSIBLE_END):
            raise ValueError(f"Cannot finish while {self.state.value}.")
        self.completed_at = datetime.now().isoformat(timespec="seconds")
        self.transition(SetState.ANALYZING)

    def cancel(self, reason="User cancelled set."):
        if self.state not in (SetState.READY, SetState.COUNTDOWN, SetState.RECORDING, SetState.POSSIBLE_END):
            raise ValueError(f"Cannot cancel while {self.state.value}.")
        self.cancelled_reason = reason
        self.transition(SetState.CANCELLED)

    def fail(self, message):
        self.error_message = str(message)
        self.transition(SetState.ERROR)

    def mark_results(self, analysis_result):
        if self.state != SetState.ANALYZING:
            raise ValueError(f"Cannot show results while {self.state.value}.")
        self.analysis_result = analysis_result
        self.transition(SetState.RESULTS)

    def reset(self):
        if self.state not in (SetState.RESULTS, SetState.CANCELLED, SetState.ERROR):
            raise ValueError(f"Cannot reset while {self.state.value}.")
        self.state = SetState.READY
        self.inference = None
        self.samples = []
        self.started_at = None
        self.completed_at = None
        self.output_file = None
        self.metadata_file = None
        self.analysis_key = None
        self.analysis_result = None
        self.write_result = None
        self.active_spec = None
        self.source_object_id = None
        self.calibration = None
        self.first_device_time_ms = None
        self.previous_sample_device_time_ms = None
        self.latest_timing_drift_ms = None
        self.timing_drift_warning = ""
        self.device_time_discontinuities = []
        self.startup_discarded_samples = 0
        self.error_message = ""
        self.cancelled_reason = ""


def write_recording_atomic(output_file, metadata_file, metadata, samples):
    if output_file.exists():
        raise FileExistsError(f"{output_file} already exists.")
    if metadata_file.exists():
        raise FileExistsError(f"{metadata_file} already exists.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    csv_temp = None
    metadata_temp = None

    try:
        with tempfile.NamedTemporaryFile(
            "w",
            newline="",
            delete=False,
            dir=output_file.parent,
            prefix=f".{output_file.stem}_",
            suffix=".tmp",
        ) as file:
            csv_temp = Path(file.name)
            writer = csv.writer(file)
            writer.writerow([
                "time_ms",
                "host_time_ms",
                "device_time_ms",
                "host_elapsed_ms",
                "timing_drift_ms",
                "emg_value",
                "raw_serial_sample",
            ])
            for sample in samples:
                writer.writerow([
                    f"{sample.time_ms:.3f}",
                    int(sample.host_time_ms),
                    "" if sample.device_time_ms is None else int(sample.device_time_ms),
                    "" if sample.host_elapsed_ms is None else f"{sample.host_elapsed_ms:.3f}",
                    "" if sample.timing_drift_ms is None else f"{sample.timing_drift_ms:.3f}",
                    sample.emg_value,
                    sample.raw_sample,
                ])

        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=metadata_file.parent,
            prefix=f".{metadata_file.stem}_",
            suffix=".tmp",
        ) as file:
            metadata_temp = Path(file.name)
            import json

            json.dump(metadata, file, indent=2)
            file.write("\n")

        os.replace(csv_temp, output_file)
        os.replace(metadata_temp, metadata_file)
    finally:
        for temp_file in (csv_temp, metadata_temp):
            if temp_file and temp_file.exists():
                temp_file.unlink()

    return output_file, metadata_file


def run_once_for_completed_set(lifecycle, analysis_key, callback):
    if lifecycle.analysis_key == analysis_key and lifecycle.analysis_result is not None:
        return lifecycle.analysis_result

    result = callback()
    lifecycle.analysis_key = analysis_key
    lifecycle.analysis_result = result
    return result
