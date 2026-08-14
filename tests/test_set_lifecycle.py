import csv
from pathlib import Path

import pytest

from acquisition.base import SignalReading
from acquisition.replay_source import ReplaySignalSource
from set_lifecycle import (
    SetLifecycle,
    SetLifecycleConfig,
    SetSessionSpec,
    SetState,
    run_once_for_completed_set,
    write_recording_atomic,
)


def reading(seconds, value):
    return SignalReading(host_time_ms=int(seconds * 1000), signal_value=value)


def lifecycle(**overrides):
    config_values = {
        "smoothing_window": 1,
        "baseline_sample_count": 5,
        "minimum_active_duration_seconds": 0.75,
        "minimum_recording_duration_seconds": 10.0,
        "auto_stop_min_active_episodes": 3,
        "auto_stop_min_active_duration_seconds": 8.0,
        "auto_stop_min_elapsed_after_meaningful_activity_seconds": 6.0,
        "substantial_activity_duration_seconds": 0.5,
        "min_substantial_activity_gap_seconds": 0.75,
        "inactivity_duration_seconds": 6.0,
        "end_grace_period_seconds": 0.0,
        "post_activity_grace_period_seconds": 0.5,
        "activity_threshold_fraction": 0.12,
        "inactive_threshold_fraction": 0.075,
        "fallback_activity_threshold": 18.0,
        "fallback_inactive_threshold": 12.0,
    }
    config_values.update(overrides)
    config = SetLifecycleConfig(**config_values)
    flow = SetLifecycle(config=config)
    flow.start_countdown()
    flow.start_recording(calibration={"baseline": 100, "signal_range": 1000, "usable": True})
    return flow


def feed(flow, values, step=0.25):
    start = 0.0
    if flow.samples:
        start = flow.samples[-1].host_elapsed_ms / 1000 + step

    for index, value in enumerate(values):
        flow.observe(reading(start + index * step, value))
        if flow.state == SetState.ANALYZING:
            break
    return flow


def rest_samples(seconds, step=0.25):
    return [100] * int(seconds / step)


def active_samples(seconds, step=0.25):
    return [500] * int(seconds / step)


def real_episode():
    return active_samples(0.75) + rest_samples(1.0)


def established_three_episode_set():
    return rest_samples(1.0) + real_episode() + real_episode() + real_episode()


def test_default_auto_stop_policy_is_conservative_for_live_sets():
    config = SetLifecycleConfig()

    assert config.minimum_recording_duration_seconds == 10.0
    assert config.auto_stop_min_active_episodes == 3
    assert config.auto_stop_min_active_duration_seconds == 8.0
    assert config.auto_stop_min_elapsed_after_meaningful_activity_seconds == 6.0
    assert config.inactivity_duration_seconds == 6.0


def test_no_auto_stop_before_meaningful_activity():
    flow = lifecycle()
    feed(flow, rest_samples(15.0))

    assert flow.state == SetState.RECORDING
    assert not flow.inference.activity_seen
    assert not flow.inference.has_meaningful_activity
    assert not flow.inference.workout_set_established(15.0)
    assert not flow.inference.auto_stop_armed


def test_baseline_noise_for_longer_than_minimum_recording_never_arms_auto_stop():
    flow = lifecycle()
    feed(flow, [100, 102, 99, 101, 98, 100, 103, 97] * 8)

    assert flow.state == SetState.RECORDING
    assert not flow.inference.activity_seen
    assert not flow.inference.has_meaningful_activity
    assert not flow.inference.auto_stop_armed


def test_one_substantial_contraction_does_not_arm_auto_stop():
    flow = lifecycle()
    feed(flow, rest_samples(1.0) + active_samples(1.0) + rest_samples(12.0))

    assert flow.state == SetState.RECORDING
    assert flow.inference.activity_seen
    assert flow.inference.has_meaningful_activity
    assert flow.inference.substantial_activity_episodes == 1
    assert not flow.inference.workout_set_established(14.0)
    assert not flow.inference.auto_stop_armed


def test_two_setup_like_movements_do_not_arm_auto_stop():
    flow = lifecycle()
    values = rest_samples(1.0) + [500] + rest_samples(1.0) + [500] + rest_samples(12.0)
    feed(flow, values)

    assert flow.state == SetState.RECORDING
    assert flow.inference.activity_seen
    assert not flow.inference.has_meaningful_activity
    assert flow.inference.substantial_activity_episodes == 0
    assert not flow.inference.auto_stop_armed


def test_three_real_substantial_episodes_arm_auto_stop():
    flow = lifecycle()
    feed(flow, established_three_episode_set() + rest_samples(6.0))

    assert flow.state in (SetState.RECORDING, SetState.POSSIBLE_END)
    assert flow.inference.substantial_activity_episodes == 3
    assert flow.inference.workout_set_established(10.0)
    assert flow.inference.auto_stop_armed
    assert flow.inference.auto_stop_arm_reason == "substantial_activity_episodes"


def test_fragmented_contraction_is_not_counted_as_several_episodes():
    flow = lifecycle()
    values = (
        rest_samples(1.0)
        + active_samples(0.75)
        + rest_samples(0.25)
        + active_samples(0.75)
        + rest_samples(0.25)
        + active_samples(0.75)
        + rest_samples(12.0)
    )
    feed(flow, values)

    assert flow.state == SetState.RECORDING
    assert flow.inference.substantial_activity_episodes == 1
    assert not flow.inference.auto_stop_armed


def test_continuous_meaningful_activity_for_required_duration_arms_auto_stop():
    flow = lifecycle()
    feed(flow, rest_samples(1.0) + active_samples(10.0))

    assert flow.state == SetState.RECORDING
    assert flow.inference.auto_stop_armed
    assert flow.inference.auto_stop_arm_reason == "active_duration"


def test_once_armed_brief_inactivity_does_not_finish():
    flow = lifecycle()
    feed(flow, established_three_episode_set() + rest_samples(4.0))

    assert flow.inference.auto_stop_armed
    assert flow.state in (SetState.RECORDING, SetState.POSSIBLE_END)
    assert flow.state != SetState.ANALYZING


def test_armed_inactivity_reaching_threshold_finishes_set():
    flow = lifecycle()
    feed(flow, established_three_episode_set() + rest_samples(11.0))

    assert flow.state == SetState.ANALYZING


def test_activity_resuming_from_possible_end_resets_inactivity_timer():
    flow = lifecycle()
    feed(flow, established_three_episode_set() + rest_samples(5.0))
    assert flow.state == SetState.POSSIBLE_END
    first_inactive_since = flow.inference.inactive_since_seconds

    feed(flow, active_samples(1.0))
    assert flow.state == SetState.RECORDING
    assert flow.inference.inactive_since_seconds is None

    feed(flow, rest_samples(1.5))
    assert flow.state == SetState.POSSIBLE_END
    assert flow.inference.inactive_since_seconds > first_inactive_since
    assert flow.inference.inactivity_elapsed_seconds(
        flow.inference.last_elapsed_seconds
    ) < flow.config.inactivity_duration_seconds


def test_no_automatic_finish_before_minimum_recording_duration():
    flow = lifecycle(
        minimum_recording_duration_seconds=20.0,
        auto_stop_min_elapsed_after_meaningful_activity_seconds=1.0,
    )
    feed(flow, established_three_episode_set() + rest_samples(8.0))

    assert flow.state == SetState.RECORDING
    assert flow.inference.workout_set_established(flow.inference.last_elapsed_seconds)
    assert not flow.inference.auto_stop_armed


def test_manual_finish_works_before_auto_stop():
    flow = lifecycle()
    flow.observe(reading(0, 100))
    flow.finish()

    assert flow.state == SetState.ANALYZING


def test_manual_finish_works_after_auto_stop_is_armed():
    flow = lifecycle()
    feed(flow, established_three_episode_set() + rest_samples(6.0))

    assert flow.inference.auto_stop_armed

    flow.finish()

    assert flow.state == SetState.ANALYZING


def test_short_one_rep_set_can_be_manually_finished_without_auto_stop():
    flow = lifecycle()
    feed(flow, rest_samples(1.0) + active_samples(1.0) + rest_samples(2.0))

    assert not flow.inference.auto_stop_armed
    flow.finish()

    assert flow.state == SetState.ANALYZING


def test_short_two_rep_set_can_be_manually_finished_without_auto_stop():
    flow = lifecycle()
    feed(flow, rest_samples(1.0) + real_episode() + real_episode() + rest_samples(2.0))

    assert flow.inference.substantial_activity_episodes == 2
    assert not flow.inference.auto_stop_armed
    flow.finish()

    assert flow.state == SetState.ANALYZING


def test_cancel_does_not_run_analysis():
    flow = lifecycle()
    calls = {"count": 0}

    flow.cancel()
    if flow.state == SetState.ANALYZING:
        run_once_for_completed_set(flow, "set-1", lambda: calls.__setitem__("count", calls["count"] + 1))

    assert flow.state == SetState.CANCELLED
    assert calls["count"] == 0


def test_cancel_works_after_auto_stop_is_armed():
    flow = lifecycle()
    feed(flow, established_three_episode_set() + rest_samples(6.0))

    assert flow.inference.auto_stop_armed

    flow.cancel()

    assert flow.state == SetState.CANCELLED


def test_analysis_is_triggered_exactly_once():
    flow = lifecycle()
    calls = {"count": 0}

    def analyze():
        calls["count"] += 1
        return {"ok": True}

    first = run_once_for_completed_set(flow, "set-1", analyze)
    second = run_once_for_completed_set(flow, "set-1", analyze)

    assert first == {"ok": True}
    assert second == {"ok": True}
    assert calls["count"] == 1


def test_lifecycle_rerun_preserves_auto_stop_arming_evidence():
    flow = lifecycle()
    feed(flow, established_three_episode_set() + rest_samples(6.0))
    same_session_state_object = flow

    assert same_session_state_object.inference.substantial_activity_episodes == 3
    assert same_session_state_object.inference.auto_stop_armed
    assert same_session_state_object.inference.activity_seen
    assert same_session_state_object.inference.has_meaningful_activity


def test_no_duplicate_recording_written_across_reruns(tmp_path):
    flow = lifecycle()
    flow.observe(reading(0, 100))
    output_file = tmp_path / "set.csv"
    metadata_file = tmp_path / "set_metadata.json"

    write_recording_atomic(output_file, metadata_file, {"session_id": "s1"}, flow.samples)
    with pytest.raises(FileExistsError):
        write_recording_atomic(output_file, metadata_file, {"session_id": "s1"}, flow.samples)

    with open(output_file, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["host_time_ms"] == "0"


def test_continuous_tension_reps_do_not_look_ended():
    flow = lifecycle()
    feed(flow, rest_samples(1.0) + [500, 650, 480, 700, 520, 680, 500, 650, 520, 680] * 4)

    assert flow.state == SetState.RECORDING
    assert flow.inference.has_meaningful_activity
    assert flow.inference.auto_stop_armed


def test_continuous_tension_recording_does_not_prematurely_stop_after_arming():
    flow = lifecycle()
    feed(flow, rest_samples(1.0) + active_samples(12.0))

    assert flow.state == SetState.RECORDING
    assert flow.inference.auto_stop_armed


def test_slow_top_pause_sequence_does_not_end_during_long_rep():
    flow = lifecycle()
    values = rest_samples(1.0)
    for _ in range(3):
        values.extend(active_samples(0.75))
        values.extend([185] * 5)
        values.extend(active_samples(0.75))
        values.extend(rest_samples(1.0))
    feed(flow, values + active_samples(3.0))

    assert flow.state == SetState.RECORDING
    assert flow.inference.auto_stop_armed
    assert flow.inference.active


def test_noisy_values_between_active_and_inactive_threshold_do_not_chatter():
    flow = lifecycle()
    statuses = []

    for index, value in enumerate([100] * 4 + [500, 210, 190, 180, 178, 176, 500]):
        statuses.append(flow.observe(reading(index * 0.25, value)))

    assert statuses[4]["active"]
    assert all(status["active"] for status in statuses[5:])
    assert flow.inference.active


def test_baseline_only_noise_does_not_create_workout_set():
    flow = lifecycle()
    feed(flow, [100, 101, 99, 102, 98, 100, 101, 99] * 4)

    assert flow.state == SetState.RECORDING
    assert not flow.inference.has_meaningful_activity


def test_missing_calibration_uses_safe_fallback_behavior():
    config = SetLifecycleConfig(
        smoothing_window=1,
        baseline_sample_count=5,
        minimum_active_duration_seconds=0.5,
        minimum_recording_duration_seconds=1.0,
        auto_stop_min_active_episodes=2,
        auto_stop_min_active_duration_seconds=0.75,
        auto_stop_min_elapsed_after_meaningful_activity_seconds=0.5,
        substantial_activity_duration_seconds=0.5,
        min_substantial_activity_gap_seconds=0.25,
        inactivity_duration_seconds=1.0,
        end_grace_period_seconds=0.0,
        post_activity_grace_period_seconds=0.25,
        fallback_activity_threshold=18.0,
        fallback_inactive_threshold=12.0,
    )
    flow = SetLifecycle(config=config)
    flow.start_countdown()
    flow.start_recording(calibration=None)
    feed(flow, [100] * 5 + [150] * 4 + [100] * 2 + [150] * 4 + [100] * 10)

    assert flow.inference.threshold is not None
    assert flow.state == SetState.ANALYZING


def test_state_transitions_reject_invalid_transitions():
    flow = SetLifecycle()

    with pytest.raises(ValueError):
        flow.transition(SetState.RESULTS)


def write_replay_csv(path, values, step_ms):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_ms", "emg_value"])
        for index, value in enumerate(values):
            writer.writerow([index * step_ms, value])
    return path


def spec_for_replay(replay_csv, output_file):
    metadata = {
        "session_id": output_file.stem,
        "source_type": "Replay CSV",
        "source_replay_csv": str(replay_csv),
        "source_replay_filename": replay_csv.name,
        "csv_filename": output_file.name,
    }
    return SetSessionSpec(
        source_type="Replay CSV",
        replay_csv=replay_csv,
        output_file=output_file,
        metadata_file=output_file.with_name(f"{output_file.stem}_metadata.json"),
        metadata=metadata,
    )


def collect_replay_set(flow, source):
    source.connect()
    flow.start_recording(calibration=None)
    flow.remember_source_object(source)

    try:
        while True:
            replay_reading = source.read()
            if replay_reading is None:
                break
            flow.observe(replay_reading)
    finally:
        source.disconnect()

    if flow.state in (SetState.RECORDING, SetState.POSSIBLE_END):
        flow.finish()


def test_replay_selection_is_fresh_per_set_and_frozen_while_active(tmp_path):
    replay_a = write_replay_csv(tmp_path / "replay_a.csv", [10, 20, 30, 40], 100)
    replay_b = write_replay_csv(tmp_path / "replay_b.csv", [200, 220, 240, 260, 280, 300], 250)
    flow = SetLifecycle(config=SetLifecycleConfig(smoothing_window=1))

    spec_a = spec_for_replay(replay_a, tmp_path / "generated_a.csv")
    flow.begin_session(spec_a)
    source_a = ReplaySignalSource(flow.active_spec.replay_csv)
    collect_replay_set(flow, source_a)
    source_a_id = flow.source_object_id
    write_recording_atomic(
        flow.output_file,
        flow.metadata_file,
        flow.active_spec.metadata,
        flow.samples,
    )
    analysis_outputs = []
    result_a = run_once_for_completed_set(
        flow,
        str(flow.output_file),
        lambda: analysis_outputs.append(flow.output_file) or {
            "csv_file": flow.output_file,
            "metadata": flow.active_spec.metadata,
        },
    )

    assert [sample.emg_value for sample in flow.samples] == [10, 20, 30, 40]
    assert flow.active_spec.metadata["source_replay_filename"] == "replay_a.csv"
    assert result_a["metadata"]["source_replay_filename"] == "replay_a.csv"

    with pytest.raises(FileExistsError):
        write_recording_atomic(
            flow.output_file,
            flow.metadata_file,
            flow.active_spec.metadata,
            flow.samples,
        )

    flow.mark_results(result_a)
    flow.reset()
    assert flow.source_object_id is None
    assert flow.active_spec is None

    spec_b = spec_for_replay(replay_b, tmp_path / "generated_b.csv")
    flow.begin_session(spec_b)
    selected_replay_after_active_start = replay_a
    assert flow.active_spec.replay_csv == replay_b
    assert flow.active_spec.replay_csv != selected_replay_after_active_start

    source_b = ReplaySignalSource(flow.active_spec.replay_csv)
    collect_replay_set(flow, source_b)
    write_recording_atomic(
        flow.output_file,
        flow.metadata_file,
        flow.active_spec.metadata,
        flow.samples,
    )
    result_b = run_once_for_completed_set(
        flow,
        str(flow.output_file),
        lambda: analysis_outputs.append(flow.output_file) or {
            "csv_file": flow.output_file,
            "metadata": flow.active_spec.metadata,
        },
    )

    assert source_a_id != flow.source_object_id
    assert [sample.emg_value for sample in flow.samples] == [200, 220, 240, 260, 280, 300]
    assert flow.samples[-1].time_ms == 1250
    assert result_b["csv_file"].name == "generated_b.csv"
    assert analysis_outputs == [tmp_path / "generated_a.csv", tmp_path / "generated_b.csv"]
    assert result_b["metadata"]["source_replay_filename"] == "replay_b.csv"
