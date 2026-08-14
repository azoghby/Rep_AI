import csv
import json

import pytest

from activation_comparisons import (
    analyze_recording,
    calibration_quality,
    compare_sets,
    group_sets,
    main,
    rep_metric,
    to_plain_data,
)


def write_recording(path, reps, baseline=100, step_ms=100):
    values = [baseline] * 10

    for peak in reps:
        values.extend([
            baseline,
            baseline + (peak - baseline) * 0.25,
            baseline + (peak - baseline) * 0.7,
            peak,
            peak,
            baseline + (peak - baseline) * 0.7,
            baseline + (peak - baseline) * 0.25,
            baseline,
        ])
        values.extend([baseline] * 18)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_ms", "emg_value"])

        for index, value in enumerate(values):
            writer.writerow([index * step_ms, round(value, 2)])

    return path


def write_metadata(csv_file, **overrides):
    metadata = {
        "exercise_name": "seated_supinated_curl",
        "muscle": "bicep",
        "side": "right",
        "weight": "15",
        "expected_reps": "3",
        "participant_id": "p1",
        "data_type": "real",
        "test_type": "workout_set",
        "csv_filename": csv_file.name,
    }
    metadata.update(overrides)
    metadata_file = csv_file.with_name(f"{csv_file.stem}_metadata.json")
    metadata_file.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


def calibration(source="calibration_right.csv"):
    return {
        "baseline": 100,
        "max_flex": 500,
        "signal_range": 400,
        "usable": True,
        "source_csv": source,
    }


def quality_calibration():
    return {
        "baseline": 100,
        "max_flex": 200,
        "signal_range": 100,
        "usable": True,
        "source_csv": "quality_calibration.csv",
    }


def test_analyze_recording_returns_per_rep_and_set_activation_metrics(tmp_path):
    csv_file = write_recording(tmp_path / "set_a.csv", [300, 400, 500])
    write_metadata(csv_file)

    metrics = analyze_recording(csv_file, calibration=calibration())

    assert metrics.rep_count == 3
    assert metrics.active_time > 0
    assert metrics.normalized_mean_activation is not None
    assert metrics.normalized_peak_activation == pytest.approx(100)
    assert metrics.integrated_normalized_emg is not None
    assert metrics.mean_rep_duration > 0
    assert metrics.rep_duration_cv_percent >= 0
    assert metrics.normalized_activation_slope_per_rep > 0
    assert len(metrics.reps) == 3
    assert metrics.reps[0].normalized_mean_activation is not None
    assert metrics.active_time_definition.startswith("Sum of detected rep interval")
    assert metrics.integrated_emg_units == "signal_value_seconds"
    assert metrics.integrated_normalized_emg_units == "percent_activation_seconds"


def test_rep_integration_uses_sample_timestamps_for_irregular_sampling():
    rep = {
        "start_time": 0.0,
        "end_time": 2.0,
        "peak_value": 300,
        "values": [100, 200, 300],
    }

    metrics = rep_metric(1, rep, sample_times=[0.0, 0.5, 2.0])

    assert metrics.duration == 2.0
    assert metrics.integrated_emg == pytest.approx(450.0)
    assert metrics.mean_activation == pytest.approx(225.0)


def test_normalized_activation_above_100_is_preserved_not_clamped():
    rep = {
        "start_time": 0.0,
        "end_time": 1.0,
        "peak_value": 300,
        "values": [256, 300],
    }

    metrics = rep_metric(1, rep, sample_times=[0.0, 1.0], calibration=quality_calibration())

    assert metrics.normalized_mean_activation == pytest.approx(178)
    assert metrics.normalized_peak_activation == pytest.approx(200)


def test_specific_above_100_values_remain_156_and_200():
    calibration_values = quality_calibration()

    rep_156 = rep_metric(
        1,
        {"start_time": 0.0, "end_time": 1.0, "peak_value": 256, "values": [256]},
        sample_times=[0.0],
        calibration=calibration_values,
    )
    rep_200 = rep_metric(
        2,
        {"start_time": 0.0, "end_time": 1.0, "peak_value": 300, "values": [300]},
        sample_times=[0.0],
        calibration=calibration_values,
    )

    assert rep_156.normalized_peak_activation == pytest.approx(156)
    assert rep_200.normalized_peak_activation == pytest.approx(200)


def test_fraction_above_calibration_reference_is_calculated_from_samples():
    reps = [
        {"values": [100, 210, 220]},
        {"values": [150, 180, 240]},
    ]

    quality = calibration_quality(
        [100, 150, 180, 210, 220, 240],
        reps,
        quality_calibration(),
    )

    assert quality.fraction_samples_above_100 == pytest.approx(0.5)
    assert quality.fraction_rep_samples_above_100 == pytest.approx(0.5)
    assert quality.calibration_reference_exceeded


def test_severe_exceedance_warns_more_strongly_than_mild_exceedance():
    mild = calibration_quality(
        [100, 120, 130, 140, 150, 256],
        [{"values": [100, 120, 130, 140, 150, 256]}],
        quality_calibration(),
    )
    severe = calibration_quality(
        [100, 481],
        [{"values": [481]}],
        quality_calibration(),
    )

    assert mild.normalized_workout_peak == pytest.approx(156)
    assert mild.severity == "mild_exceedance"
    assert mild.normalization_usable_for_cross_session
    assert severe.normalized_workout_peak == pytest.approx(381)
    assert severe.severity == "severe_exceedance"
    assert not severe.normalization_usable_for_cross_session


def test_zero_reps_have_defined_duration_variability_and_trend_metrics(tmp_path):
    csv_file = write_recording(tmp_path / "flat.csv", [])
    write_metadata(csv_file)

    metrics = analyze_recording(csv_file, calibration=calibration())

    assert metrics.rep_count == 0
    assert metrics.active_time == 0
    assert metrics.mean_rep_duration == 0
    assert metrics.rep_duration_stddev == 0
    assert metrics.rep_duration_cv_percent == 0
    assert metrics.early_mean_activation is None
    assert metrics.late_mean_activation is None
    assert metrics.activation_slope_per_rep is None


def test_one_rep_has_no_late_split_or_activation_slope(tmp_path):
    csv_file = write_recording(tmp_path / "one_rep.csv", [300])
    write_metadata(csv_file, expected_reps="1")

    metrics = analyze_recording(csv_file, calibration=calibration())

    assert metrics.rep_count == 1
    assert metrics.rep_duration_stddev == 0
    assert metrics.rep_duration_cv_percent == 0
    assert metrics.early_mean_activation is not None
    assert metrics.late_mean_activation is None
    assert metrics.activation_slope_per_rep is None


def test_compare_sets_allows_normalized_same_calibration_context(tmp_path):
    csv_a = write_recording(tmp_path / "set_a.csv", [300, 350, 400])
    csv_b = write_recording(tmp_path / "set_b.csv", [350, 425, 500])
    write_metadata(csv_a)
    write_metadata(csv_b)
    shared_calibration = calibration()

    set_a = analyze_recording(csv_a, calibration=shared_calibration)
    set_b = analyze_recording(csv_b, calibration=shared_calibration)
    comparison = compare_sets(set_a, set_b)

    assert comparison.metadata_compatible
    assert comparison.normalized_comparison_available
    assert not comparison.exploratory
    assert comparison.normalized_differences["normalized_mean_activation"]["delta"] > 0


def test_calibration_mismatch_suppresses_unjustified_normalized_percentage_comparison(tmp_path):
    csv_a = write_recording(tmp_path / "set_a.csv", [300, 350, 400])
    csv_b = write_recording(tmp_path / "set_b.csv", [350, 425, 500])
    write_metadata(csv_a)
    write_metadata(csv_b)

    set_a = analyze_recording(csv_a, calibration=calibration("cal_a.csv"))
    set_b = analyze_recording(csv_b, calibration=calibration("cal_b.csv"))
    comparison = compare_sets(set_a, set_b)

    assert not comparison.normalized_comparison_available
    assert comparison.normalized_differences == {}
    assert "same calibration context" in " ".join(comparison.warnings)


def test_severe_calibration_quality_suppresses_normalized_cross_session_comparison(tmp_path):
    csv_a = write_recording(tmp_path / "set_a.csv", [500])
    csv_b = write_recording(tmp_path / "set_b.csv", [500])
    write_metadata(csv_a)
    write_metadata(csv_b)
    low_reference = quality_calibration()

    set_a = analyze_recording(csv_a, calibration=low_reference)
    set_b = analyze_recording(csv_b, calibration=low_reference)
    comparison = compare_sets(set_a, set_b)

    assert set_a.calibration_quality.severity == "severe_exceedance"
    assert not comparison.normalized_comparison_available
    assert comparison.normalized_differences == {}


def test_left_right_comparison_is_available_but_exploratory_with_side_calibrations(tmp_path):
    right_csv = write_recording(tmp_path / "right.csv", [300, 350, 400])
    left_csv = write_recording(tmp_path / "left.csv", [320, 370, 420])
    write_metadata(right_csv, side="right")
    write_metadata(left_csv, side="left")

    right = analyze_recording(
        right_csv,
        calibration=calibration("cal_right.csv"),
        calibration_metadata={"side": "right"},
    )
    left = analyze_recording(
        left_csv,
        calibration=calibration("cal_left.csv"),
        calibration_metadata={"side": "left"},
    )
    comparison = compare_sets(right, left)

    assert comparison.normalized_comparison_available
    assert comparison.exploratory
    assert "exploratory" in " ".join(comparison.warnings)


def test_side_specific_calibration_mismatch_blocks_left_right_normalized_comparison(tmp_path):
    right_csv = write_recording(tmp_path / "right.csv", [300, 350, 400])
    left_csv = write_recording(tmp_path / "left.csv", [320, 370, 420])
    write_metadata(right_csv, side="right")
    write_metadata(left_csv, side="left")

    right = analyze_recording(
        right_csv,
        calibration=calibration("cal_right.csv"),
        calibration_metadata={"side": "left"},
    )
    left = analyze_recording(
        left_csv,
        calibration=calibration("cal_left.csv"),
        calibration_metadata={"side": "left"},
    )
    comparison = compare_sets(right, left)

    assert comparison.exploratory
    assert not comparison.normalized_comparison_available
    assert "calibration side does not match" in " ".join(comparison.warnings)


def test_weight_exercise_and_participant_mismatches_are_reported(tmp_path):
    csv_a = write_recording(tmp_path / "set_a.csv", [300, 350, 400])
    csv_b = write_recording(tmp_path / "set_b.csv", [350, 425, 500])
    write_metadata(csv_a, participant_id="p1", exercise_name="curl", weight="10")
    write_metadata(csv_b, participant_id="p2", exercise_name="hammer", weight="15")

    set_a = analyze_recording(csv_a, calibration=calibration())
    set_b = analyze_recording(csv_b, calibration=calibration())
    comparison = compare_sets(set_a, set_b)
    warning_text = " ".join(comparison.warnings)

    assert not comparison.metadata_compatible
    assert not comparison.normalized_comparison_available
    assert comparison.normalized_differences == {}
    assert "participant_id" in warning_text
    assert "exercise_name" in warning_text
    assert "weight" in warning_text


def test_missing_calibration_reports_unavailable_normalized_metrics(tmp_path):
    csv_file = write_recording(tmp_path / "set.csv", [300])
    write_metadata(csv_file)

    metrics = analyze_recording(csv_file, calibration=None)

    assert not metrics.calibration_usable
    assert metrics.normalized_mean_activation is None
    assert metrics.calibration_quality.severity == "missing"
    assert "No calibration provided" in " ".join(metrics.warnings)


def test_group_sets_uses_exercise_weight_and_side_with_missing_metadata_report(tmp_path):
    csv_a = write_recording(tmp_path / "set_a.csv", [300, 350])
    csv_b = write_recording(tmp_path / "set_b.csv", [350, 400])
    write_metadata(csv_a)
    write_metadata(csv_b, weight="")

    metrics = [
        analyze_recording(csv_a, calibration=calibration()),
        analyze_recording(csv_b, calibration=calibration()),
    ]
    grouped = group_sets(metrics)

    assert ("seated_supinated_curl", "15", "right") in grouped["groups"]
    assert grouped["missing_metadata"][str(csv_b)] == ["weight"]
    assert ("seated_supinated_curl", "", "right") not in grouped["groups"]


def test_json_serialization_and_output_file(tmp_path, capsys):
    csv_file = write_recording(tmp_path / "set.csv", [300])
    calibration_csv = write_recording(tmp_path / "calibration.csv", [200])
    write_metadata(csv_file)
    write_metadata(calibration_csv, exercise_name="calibration", test_type="calibration")
    output_json = tmp_path / "analysis.json"

    main([
        "--calibration-csv",
        str(calibration_csv),
        "--output-json",
        str(output_json),
        str(csv_file),
    ])

    captured = capsys.readouterr()
    assert "\"recordings\"" in captured.out
    assert output_json.exists()
    assert json.loads(output_json.read_text(encoding="utf-8"))["recordings"]


def test_to_plain_data_serializes_calibration_quality(tmp_path):
    csv_file = write_recording(tmp_path / "set.csv", [300])
    write_metadata(csv_file)

    payload = to_plain_data(analyze_recording(csv_file, calibration=calibration()))

    assert payload["calibration_quality"]["calibration_source"] == "calibration_right.csv"
