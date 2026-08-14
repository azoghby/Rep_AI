from detect_reps import (
    SMOOTHING_WINDOW,
    detect_reps,
    detect_reps_hybrid,
    detector_threshold_values,
    moving_average,
)


def repeated(value, count):
    return [value] * count


def sustained_tension_hammer_like_signal():
    values = repeated(20, 20)
    peaks = [310, 325, 300, 315, 305, 320]
    valleys = [90, 82, 88, 84, 86]

    for index, peak in enumerate(peaks):
        values.extend(repeated(120, 5))
        values.extend(repeated(peak, 12))
        values.extend(repeated(210, 5))

        if index < len(valleys):
            values.extend(repeated(valleys[index], 18))

    values.extend(repeated(85, 80))
    values.extend(repeated(20, 20))
    return values


def detector_results(values, step=0.1):
    times = [index * step for index in range(len(values))]
    smoothed_values = moving_average(values, SMOOTHING_WINDOW)
    thresholds = detector_threshold_values(times, smoothed_values, values)
    legacy_reps = detect_reps(
        times,
        values,
        smoothed_values,
        thresholds["start_threshold"],
        thresholds["end_threshold"],
    )
    hybrid_reps, diagnostics = detect_reps_hybrid(
        times,
        values,
        smoothed_values,
        thresholds["start_threshold"],
        thresholds["end_threshold"],
    )
    return legacy_reps, hybrid_reps, diagnostics


def test_hybrid_splits_moderate_scale_sustained_tension_cycles_above_legacy_end_threshold():
    legacy_reps, hybrid_reps, diagnostics = detector_results(
        sustained_tension_hammer_like_signal()
    )

    assert len(legacy_reps) == 1
    assert len(hybrid_reps) == 6
    assert sorted(round(candidate["time"], 1) for candidate in diagnostics["accepted_valleys"]) == [
        5.2,
        9.2,
        13.2,
        17.2,
        21.2,
    ]
