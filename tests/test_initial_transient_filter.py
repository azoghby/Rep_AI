from detect_reps import detect_reps, detect_reps_hybrid


START_THRESHOLD = 10
END_THRESHOLD = 5


def repeated(value, count):
    return [value] * count


def detect(values, step=0.1):
    times = [index * step for index in range(len(values))]
    return detect_reps(times, values, values, START_THRESHOLD, END_THRESHOLD)


def detect_hybrid(values, step=0.1):
    times = [index * step for index in range(len(values))]
    reps, _ = detect_reps_hybrid(times, values, values, START_THRESHOLD, END_THRESHOLD)
    return reps


def intervals(reps):
    return [(round(rep["start_time"], 2), round(rep["end_time"], 2)) for rep in reps]


def test_short_low_evidence_preset_transient_is_rejected_before_legitimate_reps():
    values = (
        repeated(0, 5)
        + repeated(20, 5)
        + repeated(0, 8)
        + repeated(100, 12)
        + repeated(0, 8)
        + repeated(90, 12)
        + repeated(0, 8)
    )

    legacy_reps = detect(values)
    hybrid_reps = detect_hybrid(values)

    assert intervals(legacy_reps) == [(1.8, 3.0), (3.8, 5.0)]
    assert intervals(hybrid_reps) == [(1.8, 3.0), (3.8, 5.0)]


def test_legitimate_first_rep_beginning_soon_after_recording_starts_is_retained():
    values = (
        repeated(0, 1)
        + repeated(90, 12)
        + repeated(0, 8)
        + repeated(100, 12)
        + repeated(0, 8)
    )

    legacy_reps = detect(values)
    hybrid_reps = detect_hybrid(values)

    assert intervals(legacy_reps) == [(0.1, 1.3), (2.1, 3.3)]
    assert intervals(hybrid_reps) == [(0.1, 1.3), (2.1, 3.3)]


def test_no_preset_transient_behavior_is_unchanged():
    values = (
        repeated(0, 8)
        + repeated(100, 12)
        + repeated(0, 8)
        + repeated(90, 12)
        + repeated(0, 8)
    )

    legacy_reps = detect(values)
    hybrid_reps = detect_hybrid(values)

    assert intervals(legacy_reps) == [(0.8, 2.0), (2.8, 4.0)]
    assert intervals(hybrid_reps) == [(0.8, 2.0), (2.8, 4.0)]


def test_one_rep_set_keeps_its_only_rep_without_later_comparison_data():
    values = repeated(0, 5) + repeated(90, 12) + repeated(0, 8)

    legacy_reps = detect(values)
    hybrid_reps = detect_hybrid(values)

    assert intervals(legacy_reps) == [(0.5, 1.7)]
    assert intervals(hybrid_reps) == [(0.5, 1.7)]


def test_two_rep_set_keeps_first_rep_when_evidence_is_comparable():
    values = (
        repeated(0, 5)
        + repeated(80, 12)
        + repeated(0, 8)
        + repeated(100, 12)
        + repeated(0, 8)
    )

    legacy_reps = detect(values)
    hybrid_reps = detect_hybrid(values)

    assert intervals(legacy_reps) == [(0.5, 1.7), (2.5, 3.7)]
    assert intervals(hybrid_reps) == [(0.5, 1.7), (2.5, 3.7)]


def test_two_region_set_prefers_retaining_first_when_later_evidence_is_weak():
    values = (
        repeated(0, 5)
        + repeated(20, 5)
        + repeated(0, 8)
        + repeated(25, 5)
        + repeated(0, 8)
    )

    legacy_reps = detect(values)
    hybrid_reps = detect_hybrid(values)

    assert intervals(legacy_reps) == [(0.5, 1.0), (1.8, 2.3)]
    assert intervals(hybrid_reps) == [(0.5, 1.0), (1.8, 2.3)]
