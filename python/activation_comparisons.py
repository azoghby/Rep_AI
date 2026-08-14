import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from calibration_utils import (
    MIN_SIGNAL_RANGE,
    calculate_calibration,
    load_csv_signal,
    normalize_values,
)
from detect_reps import (
    BASELINE_PERCENTILE,
    END_THRESHOLD_FRACTION,
    SMOOTHING_WINDOW,
    START_THRESHOLD_FRACTION,
    detect_reps,
    detect_reps_hybrid,
    low_percentile_average,
    moving_average,
    read_signal,
)
from recording_metadata import load_metadata


GROUPING_FIELDS = ("exercise_name", "weight", "side")
STRICT_COMPATIBILITY_FIELDS = (
    "participant_id",
    "muscle",
    "side",
    "exercise_name",
    "weight",
)
OPTIONAL_COMPATIBILITY_FIELDS = (
    "source_type",
    "test_type",
    "set_end_rule",
    "comparison_target",
)
EPSILON = 1e-9
MODERATE_EXCEEDANCE_RATIO = 2.0
SEVERE_EXCEEDANCE_RATIO = 3.0
MODERATE_EXCEEDANCE_FRACTION = 0.25
SEVERE_EXCEEDANCE_FRACTION = 0.50


@dataclass
class CalibrationQuality:
    calibration_source: str | None
    calibration_baseline: float | None
    calibration_max_flex: float | None
    calibration_range: float | None
    raw_workout_peak: float
    normalized_workout_peak: float | None
    fraction_samples_above_100: float | None
    fraction_rep_samples_above_100: float | None
    calibration_reference_exceeded: bool
    maximum_exceedance_ratio: float | None
    normalization_usable_for_cross_session: bool
    severity: str
    warnings: list[str]
    reasons: list[str]


@dataclass
class RepMetrics:
    rep_number: int
    start_time: float
    end_time: float
    duration: float
    mean_activation: float
    median_activation: float
    peak_activation: float
    integrated_emg: float
    integrated_emg_units: str
    normalized_mean_activation: float | None = None
    normalized_median_activation: float | None = None
    normalized_peak_activation: float | None = None
    integrated_normalized_emg: float | None = None
    integrated_normalized_emg_units: str | None = None


@dataclass
class SetMetrics:
    csv_file: str
    metadata: dict
    calibration_source: str | None
    calibration_usable: bool
    calibration_signature: dict | None
    calibration_metadata: dict
    calibration_quality: CalibrationQuality
    warnings: list[str]
    detector_method: str
    raw_min_signal: float
    raw_max_signal: float
    recording_duration: float
    rep_count: int
    active_time: float
    active_time_definition: str
    set_duration: float
    mean_activation: float
    median_activation: float
    peak_activation: float
    integrated_emg: float
    integrated_emg_units: str
    mean_rep_duration: float
    median_rep_duration: float
    rep_duration_stddev: float
    rep_duration_cv_percent: float
    early_mean_activation: float | None
    late_mean_activation: float | None
    early_late_activation_delta: float | None
    activation_slope_per_rep: float | None
    normalized_mean_activation: float | None
    normalized_median_activation: float | None
    normalized_peak_activation: float | None
    integrated_normalized_emg: float | None
    integrated_normalized_emg_units: str | None
    normalized_early_mean_activation: float | None
    normalized_late_mean_activation: float | None
    normalized_early_late_activation_delta: float | None
    normalized_activation_slope_per_rep: float | None
    reps: list[RepMetrics]


@dataclass
class SetComparison:
    set_a: str
    set_b: str
    metadata_compatible: bool
    normalized_comparison_available: bool
    exploratory: bool
    warnings: list[str]
    raw_differences: dict
    normalized_differences: dict


def average(values):
    return sum(values) / len(values) if values else 0


def median(values):
    return statistics.median(values) if values else 0


def stddev(values):
    if len(values) < 2:
        return 0

    return statistics.pstdev(values)


def coefficient_of_variation(values):
    mean_value = average(values)

    if not values or mean_value == 0:
        return 0

    return stddev(values) / mean_value * 100


def linear_slope(values):
    if len(values) < 2:
        return None

    x_values = list(range(1, len(values) + 1))
    x_mean = average(x_values)
    y_mean = average(values)
    denominator = sum((x - x_mean) ** 2 for x in x_values)

    if denominator == 0:
        return None

    return sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values)) / denominator


def trapezoid_integral(sample_times, values, duration):
    if duration <= 0 or not values:
        return 0

    if len(sample_times) != len(values) or len(values) == 1:
        return average(values) * duration

    integral = 0

    for index in range(1, len(values)):
        elapsed = sample_times[index] - sample_times[index - 1]

        if elapsed > 0:
            integral += (values[index - 1] + values[index]) / 2 * elapsed

    return integral


def time_weighted_mean(sample_times, values, duration):
    if duration <= 0 or not values:
        return 0

    return trapezoid_integral(sample_times, values, duration) / duration


def percent_difference(value_a, value_b):
    if value_a is None or value_b is None:
        return None

    lower = min(abs(value_a), abs(value_b))

    if lower <= EPSILON:
        return 0

    return abs(value_a - value_b) / lower * 100


def fraction_above(values, threshold):
    if not values:
        return 0

    return sum(1 for value in values if value > threshold) / len(values)


def calibration_is_usable(calibration):
    return bool(
        calibration
        and calibration.get("usable", True)
        and float(calibration.get("signal_range", 0)) >= MIN_SIGNAL_RANGE
    )


def calibration_signature(calibration):
    if not calibration:
        return None

    return {
        "source_csv": calibration.get("source_csv", ""),
        "baseline": round(float(calibration.get("baseline", 0)), 6),
        "signal_range": round(float(calibration.get("signal_range", 0)), 6),
    }


def calibration_from_csv(csv_file):
    _, values = load_csv_signal(csv_file)
    calibration_values = calculate_calibration(values)
    calibration_values["source_csv"] = Path(csv_file).name
    return calibration_values


def calibration_quality(values, reps, calibration):
    raw_peak = max(values) if values else 0

    if not calibration:
        return CalibrationQuality(
            calibration_source=None,
            calibration_baseline=None,
            calibration_max_flex=None,
            calibration_range=None,
            raw_workout_peak=raw_peak,
            normalized_workout_peak=None,
            fraction_samples_above_100=None,
            fraction_rep_samples_above_100=None,
            calibration_reference_exceeded=False,
            maximum_exceedance_ratio=None,
            normalization_usable_for_cross_session=False,
            severity="missing",
            warnings=["No calibration provided; normalized metrics are unavailable."],
            reasons=["missing_calibration"],
        )

    baseline = float(calibration.get("baseline", 0))
    max_flex = float(calibration.get("max_flex", 0))
    signal_range = float(calibration.get("signal_range", 0))
    source = calibration.get("source_csv")
    warnings = []
    reasons = []

    if not calibration_is_usable(calibration):
        return CalibrationQuality(
            calibration_source=source,
            calibration_baseline=baseline,
            calibration_max_flex=max_flex,
            calibration_range=signal_range,
            raw_workout_peak=raw_peak,
            normalized_workout_peak=None,
            fraction_samples_above_100=None,
            fraction_rep_samples_above_100=None,
            calibration_reference_exceeded=False,
            maximum_exceedance_ratio=None,
            normalization_usable_for_cross_session=False,
            severity="unusable",
            warnings=["Calibration range is too small for normalized metrics."],
            reasons=["unusable_calibration_range"],
        )

    normalized_values = normalize_values(values, calibration)
    normalized_rep_values = [
        normalized_value
        for rep in reps
        for normalized_value in normalize_values(rep.get("values", []), calibration)
    ]
    fraction_samples = fraction_above(normalized_values, 100)
    fraction_rep_samples = fraction_above(normalized_rep_values, 100)
    normalized_peak = max(normalized_values) if normalized_values else None
    maximum_exceedance_ratio = normalized_peak / 100 if normalized_peak is not None else None
    calibration_reference_exceeded = bool(
        normalized_peak is not None and normalized_peak > 100
    )

    severity = "ok"

    if calibration_reference_exceeded:
        severity = "mild_exceedance"
        warnings.append(
            "Workout EMG exceeded the calibration max-flex reference; values above 100% are preserved."
        )
        reasons.append("calibration_reference_exceeded")

    if (
        maximum_exceedance_ratio is not None
        and maximum_exceedance_ratio >= MODERATE_EXCEEDANCE_RATIO
    ) or max(fraction_samples, fraction_rep_samples) >= MODERATE_EXCEEDANCE_FRACTION:
        severity = "moderate_exceedance"
        warnings.append(
            "Calibration exceedance is moderate; cross-session normalized amplitude comparisons need caution."
        )
        reasons.append("moderate_calibration_exceedance")

    if (
        maximum_exceedance_ratio is not None
        and maximum_exceedance_ratio >= SEVERE_EXCEEDANCE_RATIO
    ) or max(fraction_samples, fraction_rep_samples) >= SEVERE_EXCEEDANCE_FRACTION:
        severity = "severe_exceedance"
        warnings.append(
            "Calibration exceedance is severe; recalibration is recommended before cross-session normalized comparisons."
        )
        reasons.append("severe_calibration_exceedance")

    usable_for_cross_session = severity not in {"severe_exceedance"}

    if severity == "ok":
        warnings.append("Calibration reference was not exceeded.")

    return CalibrationQuality(
        calibration_source=source,
        calibration_baseline=baseline,
        calibration_max_flex=max_flex,
        calibration_range=signal_range,
        raw_workout_peak=raw_peak,
        normalized_workout_peak=normalized_peak,
        fraction_samples_above_100=fraction_samples,
        fraction_rep_samples_above_100=fraction_rep_samples,
        calibration_reference_exceeded=calibration_reference_exceeded,
        maximum_exceedance_ratio=maximum_exceedance_ratio,
        normalization_usable_for_cross_session=usable_for_cross_session,
        severity=severity,
        warnings=warnings,
        reasons=reasons,
    )


def rep_metric(rep_number, rep, sample_times=None, calibration=None):
    duration = max(0, rep["end_time"] - rep["start_time"])
    values = rep.get("values", [])
    sample_times = sample_times or []
    normalized_values = normalize_values(values, calibration) if calibration else []
    integrated_emg = trapezoid_integral(sample_times, values, duration)
    integrated_normalized_emg = (
        trapezoid_integral(sample_times, normalized_values, duration)
        if calibration else None
    )

    return RepMetrics(
        rep_number=rep_number,
        start_time=rep["start_time"],
        end_time=rep["end_time"],
        duration=duration,
        mean_activation=time_weighted_mean(sample_times, values, duration),
        median_activation=median(values),
        peak_activation=rep.get("peak_value", max(values) if values else 0),
        integrated_emg=integrated_emg,
        integrated_emg_units="signal_value_seconds",
        normalized_mean_activation=(
            time_weighted_mean(sample_times, normalized_values, duration)
            if calibration else None
        ),
        normalized_median_activation=median(normalized_values) if calibration else None,
        normalized_peak_activation=max(normalized_values) if normalized_values else None,
        integrated_normalized_emg=integrated_normalized_emg,
        integrated_normalized_emg_units="percent_activation_seconds" if calibration else None,
    )


def split_average(values):
    if not values:
        return None, None, None

    split_index = max(1, len(values) // 2)
    early = average(values[:split_index])
    late = average(values[split_index:]) if values[split_index:] else None
    delta = None if late is None else late - early
    return early, late, delta


def detect_recording_reps(csv_file, detector_method="legacy"):
    times, values = read_signal(csv_file)
    smoothed_values = moving_average(values, SMOOTHING_WINDOW)
    baseline = low_percentile_average(smoothed_values, BASELINE_PERCENTILE)
    max_signal = max(smoothed_values)
    signal_range = max_signal - baseline
    start_threshold = baseline + START_THRESHOLD_FRACTION * signal_range
    end_threshold = baseline + END_THRESHOLD_FRACTION * signal_range
    if detector_method == "legacy":
        reps = detect_reps(times, values, smoothed_values, start_threshold, end_threshold)
    elif detector_method == "hybrid":
        reps, _ = detect_reps_hybrid(
            times,
            values,
            smoothed_values,
            start_threshold,
            end_threshold,
        )
    else:
        raise ValueError(f"Unknown detector method: {detector_method}")

    return times, values, reps


def rep_sample_times(times, rep):
    return [
        time
        for time in times
        if rep["start_time"] <= time <= rep["end_time"]
    ]


def analyze_recording(
    csv_file,
    calibration=None,
    calibration_metadata=None,
    detector_method="legacy",
):
    csv_file = Path(csv_file)
    metadata = load_metadata(csv_file)
    usable_calibration = calibration if calibration_is_usable(calibration) else None
    warnings = []

    times, values, reps = detect_recording_reps(csv_file, detector_method=detector_method)
    rep_metrics = [
        rep_metric(index, rep, rep_sample_times(times, rep), usable_calibration)
        for index, rep in enumerate(reps, start=1)
    ]
    durations = [rep.duration for rep in rep_metrics]
    rep_means = [rep.mean_activation for rep in rep_metrics]
    normalized_rep_means = [
        rep.normalized_mean_activation
        for rep in rep_metrics
        if rep.normalized_mean_activation is not None
    ]
    early, late, delta = split_average(rep_means)
    normalized_early, normalized_late, normalized_delta = split_average(normalized_rep_means)
    active_time = sum(durations)
    integrated_emg = sum(rep.integrated_emg for rep in rep_metrics)
    integrated_normalized_emg = (
        sum(
            rep.integrated_normalized_emg
            for rep in rep_metrics
            if rep.integrated_normalized_emg is not None
        )
        if usable_calibration else None
    )

    if times:
        recording_duration = times[-1] - times[0]
    else:
        recording_duration = 0

    if reps:
        set_duration = reps[-1]["end_time"] - reps[0]["start_time"]
    elif times:
        set_duration = recording_duration
    else:
        set_duration = 0

    quality = calibration_quality(values, reps, calibration)
    warnings.extend(quality.warnings)

    return SetMetrics(
        csv_file=str(csv_file),
        metadata=metadata,
        calibration_source=(calibration or {}).get("source_csv") if calibration else None,
        calibration_usable=usable_calibration is not None,
        calibration_signature=calibration_signature(usable_calibration),
        calibration_metadata=calibration_metadata or {},
        calibration_quality=quality,
        warnings=warnings,
        detector_method=detector_method,
        raw_min_signal=min(values) if values else 0,
        raw_max_signal=max(values) if values else 0,
        recording_duration=recording_duration,
        rep_count=len(rep_metrics),
        active_time=active_time,
        active_time_definition="Sum of detected rep interval durations from detector start/end timestamps.",
        set_duration=set_duration,
        mean_activation=integrated_emg / active_time if active_time > 0 else 0,
        median_activation=median(rep_means),
        peak_activation=max([rep.peak_activation for rep in rep_metrics], default=0),
        integrated_emg=integrated_emg,
        integrated_emg_units="signal_value_seconds",
        mean_rep_duration=average(durations),
        median_rep_duration=median(durations),
        rep_duration_stddev=stddev(durations),
        rep_duration_cv_percent=coefficient_of_variation(durations),
        early_mean_activation=early,
        late_mean_activation=late,
        early_late_activation_delta=delta,
        activation_slope_per_rep=linear_slope(rep_means),
        normalized_mean_activation=(
            integrated_normalized_emg / active_time
            if usable_calibration and active_time > 0 else None
        ),
        normalized_median_activation=median(normalized_rep_means) if usable_calibration else None,
        normalized_peak_activation=max(
            [
                rep.normalized_peak_activation
                for rep in rep_metrics
                if rep.normalized_peak_activation is not None
            ],
            default=None,
        ) if usable_calibration else None,
        integrated_normalized_emg=integrated_normalized_emg,
        integrated_normalized_emg_units="percent_activation_seconds" if usable_calibration else None,
        normalized_early_mean_activation=normalized_early,
        normalized_late_mean_activation=normalized_late,
        normalized_early_late_activation_delta=normalized_delta,
        normalized_activation_slope_per_rep=linear_slope(normalized_rep_means),
        reps=rep_metrics,
    )


def metadata_value(metrics, key):
    return str(metrics.metadata.get(key, "")).strip()


def compatible_metadata(set_a, set_b):
    warnings = []

    for field in STRICT_COMPATIBILITY_FIELDS:
        value_a = metadata_value(set_a, field)
        value_b = metadata_value(set_b, field)

        if not value_a or not value_b:
            warnings.append(f"Missing metadata field for compatibility check: {field}.")
        elif value_a != value_b:
            warnings.append(f"Metadata differs for {field}: {value_a} vs {value_b}.")

    for field in OPTIONAL_COMPATIBILITY_FIELDS:
        value_a = metadata_value(set_a, field)
        value_b = metadata_value(set_b, field)

        if value_a and value_b and value_a != value_b:
            warnings.append(f"Recording protocol differs for {field}: {value_a} vs {value_b}.")

    return not warnings, warnings


def normalized_comparison_status(set_a, set_b):
    warnings = []
    side_a = metadata_value(set_a, "side")
    side_b = metadata_value(set_b, "side")
    exploratory = False

    if not set_a.calibration_usable or not set_b.calibration_usable:
        return False, exploratory, ["Usable calibration is required for normalized comparisons."]

    if (
        not set_a.calibration_quality.normalization_usable_for_cross_session
        or not set_b.calibration_quality.normalization_usable_for_cross_session
    ):
        return False, exploratory, [
            "Calibration quality does not support cross-session normalized comparison."
        ]

    if side_a and side_b and side_a != side_b:
        exploratory = True
        calibration_side_a = str(set_a.calibration_metadata.get("side", "")).strip()
        calibration_side_b = str(set_b.calibration_metadata.get("side", "")).strip()

        if calibration_side_a and calibration_side_a != side_a:
            warnings.append("Set A calibration side does not match set metadata side.")
        if calibration_side_b and calibration_side_b != side_b:
            warnings.append("Set B calibration side does not match set metadata side.")

        warnings.append(
            "Left/right comparison is exploratory because sequential single-sensor placement can vary."
        )
        return not any("does not match" in warning for warning in warnings), exploratory, warnings

    if set_a.calibration_signature != set_b.calibration_signature:
        return False, exploratory, [
            "Same-side normalized comparison requires the same calibration context."
        ]

    return True, exploratory, warnings


def metric_difference(set_a, set_b, metric_name):
    value_a = getattr(set_a, metric_name)
    value_b = getattr(set_b, metric_name)
    return {
        "set_a": value_a,
        "set_b": value_b,
        "delta": None if value_a is None or value_b is None else value_b - value_a,
        "percent_difference": percent_difference(value_a, value_b),
    }


def compare_sets(set_a, set_b):
    metadata_ok, metadata_warnings = compatible_metadata(set_a, set_b)
    normalized_ok, exploratory, calibration_warnings = normalized_comparison_status(set_a, set_b)
    warnings = metadata_warnings + calibration_warnings

    if normalized_ok and metadata_warnings and not exploratory:
        normalized_ok = False
        warnings.append(
            "Normalized amplitude comparison suppressed because metadata/protocol compatibility warnings exist."
        )

    raw_metrics = [
        "mean_activation",
        "median_activation",
        "peak_activation",
        "integrated_emg",
        "active_time",
        "mean_rep_duration",
        "rep_duration_cv_percent",
        "early_late_activation_delta",
        "activation_slope_per_rep",
    ]
    normalized_metrics = [
        "normalized_mean_activation",
        "normalized_median_activation",
        "normalized_peak_activation",
        "integrated_normalized_emg",
        "normalized_early_late_activation_delta",
        "normalized_activation_slope_per_rep",
    ]

    return SetComparison(
        set_a=set_a.csv_file,
        set_b=set_b.csv_file,
        metadata_compatible=metadata_ok,
        normalized_comparison_available=normalized_ok,
        exploratory=exploratory,
        warnings=warnings,
        raw_differences={
            metric: metric_difference(set_a, set_b, metric)
            for metric in raw_metrics
        },
        normalized_differences={
            metric: metric_difference(set_a, set_b, metric)
            for metric in normalized_metrics
        } if normalized_ok else {},
    )


def grouping_key(metrics):
    values = tuple(metadata_value(metrics, field) for field in GROUPING_FIELDS)
    missing = [
        field
        for field, value in zip(GROUPING_FIELDS, values)
        if not value
    ]
    return values, missing


def group_sets(metrics_list):
    groups = {}
    missing = {}

    for metrics in metrics_list:
        key, missing_fields = grouping_key(metrics)

        if missing_fields:
            missing[metrics.csv_file] = missing_fields
            continue

        groups.setdefault(key, []).append(metrics)

    return {"groups": groups, "missing_metadata": missing}


def to_plain_data(value):
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]

    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)

    if isinstance(value, dict):
        return {
            str(key): to_plain_data(item)
            for key, item in value.items()
        }

    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None

    return value


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyze EMG activation metrics for one or more recordings."
    )
    parser.add_argument("csv_files", nargs="+", type=Path)
    parser.add_argument(
        "--calibration-csv",
        type=Path,
        help="Optional calibration CSV used for normalized activation metrics.",
    )
    parser.add_argument(
        "--detector-method",
        choices=("legacy", "hybrid"),
        default="legacy",
        help="Rep detector to use for reported per-rep metrics.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Include a comparison for exactly two supplied recordings.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write the structured JSON payload.",
    )
    args = parser.parse_args(argv)
    calibration = calibration_from_csv(args.calibration_csv) if args.calibration_csv else None
    calibration_metadata = load_metadata(args.calibration_csv) if args.calibration_csv else {}
    analyses = [
        analyze_recording(
            csv_file,
            calibration=calibration,
            calibration_metadata=calibration_metadata,
            detector_method=args.detector_method,
        )
        for csv_file in args.csv_files
    ]
    payload = {"recordings": [to_plain_data(analysis) for analysis in analyses]}

    if args.compare:
        if len(analyses) != 2:
            parser.error("--compare requires exactly two CSV files.")

        payload["comparison"] = to_plain_data(compare_sets(analyses[0], analyses[1]))

    output_text = json.dumps(payload, indent=2)

    if args.output_json:
        args.output_json.write_text(f"{output_text}\n", encoding="utf-8")

    print(output_text)


if __name__ == "__main__":
    main()
