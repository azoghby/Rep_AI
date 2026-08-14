from detect_reps import (
    BASELINE_PERCENTILE,
    END_THRESHOLD_FRACTION,
    START_THRESHOLD_FRACTION,
    detect_reps,
    detect_reps_hybrid,
    detector_threshold_values,
    low_percentile_average,
    moving_average,
)


STEP_SECONDS = 0.1
SMOOTHING_WINDOW = 15


def repeated(value, count):
    return [value] * count


def sample_times(values):
    return [index * STEP_SECONDS for index in range(len(values))]


def old_full_recording_thresholds(smoothed_values):
    baseline = low_percentile_average(smoothed_values, BASELINE_PERCENTILE)
    max_signal = max(smoothed_values)
    signal_range = max_signal - baseline
    return {
        "baseline": baseline,
        "start_threshold": baseline + START_THRESHOLD_FRACTION * signal_range,
        "end_threshold": baseline + END_THRESHOLD_FRACTION * signal_range,
    }


def detected_intervals(values, thresholds=None):
    times = sample_times(values)
    smoothed_values = moving_average(values, SMOOTHING_WINDOW)
    thresholds = thresholds or detector_threshold_values(times, smoothed_values, values)
    legacy_reps = detect_reps(
        times,
        values,
        smoothed_values,
        thresholds["start_threshold"],
        thresholds["end_threshold"],
    )
    hybrid_reps, _ = detect_reps_hybrid(
        times,
        values,
        smoothed_values,
        thresholds["start_threshold"],
        thresholds["end_threshold"],
    )
    return {
        "legacy": [(round(rep["start_time"], 1), round(rep["end_time"], 1)) for rep in legacy_reps],
        "hybrid": [(round(rep["start_time"], 1), round(rep["end_time"], 1)) for rep in hybrid_reps],
        "thresholds": thresholds,
    }


def synthetic_six_rep_signal(trailing_rest_samples=0):
    values = repeated(100, 10)
    for peak in (520, 560, 430, 410, 450, 540):
        values.extend(repeated(160, 4))
        values.extend(repeated(peak, 12))
        values.extend(repeated(150, 4))
        values.extend(repeated(100, 16))
    values.extend(repeated(30, trailing_rest_samples))
    return values


def test_trailing_quiet_does_not_materially_change_six_rep_segmentation():
    short_signal = synthetic_six_rep_signal()
    long_tail_signal = synthetic_six_rep_signal(trailing_rest_samples=200)

    short_result = detected_intervals(short_signal)
    long_tail_result = detected_intervals(long_tail_signal)

    assert len(short_result["legacy"]) == 6
    assert len(short_result["hybrid"]) == 6
    assert long_tail_result["legacy"] == short_result["legacy"]
    assert long_tail_result["hybrid"] == short_result["hybrid"]


def test_synthetic_trailing_quiet_reproduces_old_full_reference_collapse():
    long_tail_signal = synthetic_six_rep_signal(trailing_rest_samples=200)
    times = sample_times(long_tail_signal)
    smoothed_values = moving_average(long_tail_signal, SMOOTHING_WINDOW)
    old_thresholds = old_full_recording_thresholds(smoothed_values)
    old_result = detected_intervals(long_tail_signal, thresholds=old_thresholds)
    new_result = detected_intervals(long_tail_signal)

    assert len(old_result["legacy"]) == 1
    assert len(new_result["legacy"]) == 6
    assert len(new_result["hybrid"]) == 6
    assert new_result["thresholds"]["reference_end_time"] < times[-1] - 5


def test_no_activity_recording_keeps_zero_reps_and_zero_signal_range():
    values = repeated(42, 120)
    result = detected_intervals(values)

    assert result["thresholds"]["signal_range"] == 0
    assert result["legacy"] == []
    assert result["hybrid"] == []


def test_short_one_rep_set_with_trailing_rest_keeps_one_rep():
    values = (
        repeated(80, 10)
        + repeated(450, 14)
        + repeated(80, 10)
        + repeated(30, 80)
    )
    result = detected_intervals(values)

    assert len(result["legacy"]) == 1
    assert len(result["hybrid"]) == 1


def test_recording_starting_slightly_active_keeps_later_rep_segmentation():
    values = repeated(100, 8) + repeated(30, 30)
    for _ in range(6):
        values.extend(repeated(400, 12))
        values.extend(repeated(30, 15))

    result = detected_intervals(values)

    assert len(result["legacy"]) == 6
    assert len(result["hybrid"]) == 6
