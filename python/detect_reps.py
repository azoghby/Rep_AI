import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

from calibration_utils import load_calibration_for_recording, normalize_values
from recording_metadata import load_metadata, metadata_lines


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
GRAPHS_DIR = BASE_DIR / "graphs"
SUMMARIES_DIR = BASE_DIR / "summaries"

SMOOTHING_WINDOW = 15
BASELINE_PERCENTILE = 10
START_THRESHOLD_FRACTION = 0.18
END_THRESHOLD_FRACTION = 0.085
MIN_REP_DURATION_SECONDS = 0.35
MIN_GAP_SECONDS = 0.25
CONSISTENT_FATIGUE_THRESHOLD = 5

HYBRID_MIN_SEGMENT_DURATION_SECONDS = 0.75
HYBRID_MIN_CENTER_GAP_SECONDS = 0.90
HYBRID_MIN_VALLEY_DURATION_SECONDS = 0.12
HYBRID_MIN_SPLITTABLE_REGION_SECONDS = 5.0
HYBRID_MIN_ABSOLUTE_DROP = 300
HYBRID_MIN_LOCAL_RANGE_DROP_FRACTION = 0.25
HYBRID_MIN_ADJACENT_PEAK_DROP_FRACTION = 0.30
HYBRID_STRONG_RELATIVE_DROP_FRACTION = 0.80
HYBRID_STRONG_LOCAL_DROP_FRACTION = 0.65
HYBRID_STRONG_RELATIVE_VALLEY_DURATION_SECONDS = 1.0
HYBRID_VALLEY_CLUSTER_SECONDS = 0.30
HYBRID_PLATEAU_FRACTION = 0.05
HYBRID_MIN_REP_PEAK_EXCESS_FRACTION = 0.08
HYBRID_MIN_REP_AREA_FRACTION_SECONDS = 0.06
INITIAL_TRANSIENT_DURATION_MEDIAN_FRACTION = 0.55
INITIAL_TRANSIENT_AREA_MEDIAN_FRACTION = 0.35
INITIAL_TRANSIENT_PEAK_EXCESS_MEDIAN_FRACTION = 0.55
HYBRID_INITIAL_LULL_SECONDS = 0.25
HYBRID_INITIAL_SUBSTANTIAL_RISE_FRACTION = 0.50
HYBRID_SHORT_FRAGMENT_MEDIAN_FRACTION = 0.55
HYBRID_FRAGMENT_PLATEAU_MEDIAN_FRACTION = 0.45
HYBRID_FRAGMENT_AREA_MEDIAN_FRACTION = 0.35
HYBRID_MIN_CYCLE_PERIOD_SECONDS = 2.8
HYBRID_MAX_CYCLE_PERIOD_SECONDS = 10.0
HYBRID_PERIOD_SAMPLE_SECONDS = 0.05


def newest_csv_file(folder):
    csv_files = list(folder.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No CSV files found in the data folder.")

    return max(csv_files, key=lambda path: path.stat().st_mtime)


def signal_column(fieldnames):
    for column_name in ("signal_value", "emg_value"):
        if column_name in fieldnames:
            return column_name

    raise ValueError("CSV must contain either a signal_value or emg_value column.")


def read_signal(csv_file):
    times = []
    values = []

    with open(csv_file, "r", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(f"{csv_file.name} does not have a header row.")

        value_key = signal_column(reader.fieldnames)

        for row in reader:
            times.append(float(row["time_ms"]) / 1000)
            values.append(float(row[value_key]))

    if not values:
        raise ValueError(f"{csv_file.name} does not contain any signal rows.")

    return times, values


def moving_average(values, window_size):
    if window_size <= 1 or len(values) <= 1:
        return values[:]

    window_size = min(window_size, len(values))
    half_window = window_size // 2
    smoothed = []

    for index in range(len(values)):
        start_index = max(0, index - half_window)
        end_index = min(len(values), index + half_window + 1)
        window = values[start_index:end_index]
        smoothed.append(sum(window) / len(window))

    return smoothed


def low_percentile_average(values, percentile):
    sorted_values = sorted(values)
    count = max(1, int(len(sorted_values) * (percentile / 100)))
    low_values = sorted_values[:count]
    return sum(low_values) / len(low_values)


def detector_threshold_values(times, smoothed_values, raw_values=None):
    if not smoothed_values:
        raise ValueError("Cannot calculate detector thresholds without signal values.")

    full_thresholds = threshold_values_for_reference(times, smoothed_values, smoothed_values)
    reference_values = threshold_reference_values(times, smoothed_values)
    active_thresholds = threshold_values_for_reference(times, smoothed_values, reference_values)

    if len(reference_values) == len(smoothed_values) or raw_values is None:
        return active_thresholds

    raw_values = raw_values or smoothed_values
    full_reps, _ = detect_reps_hybrid(
        times,
        raw_values,
        smoothed_values,
        full_thresholds["start_threshold"],
        full_thresholds["end_threshold"],
    )
    active_reps, _ = detect_reps_hybrid(
        times,
        raw_values,
        smoothed_values,
        active_thresholds["start_threshold"],
        active_thresholds["end_threshold"],
    )
    full_legacy_reps = detect_reps(
        times,
        raw_values,
        smoothed_values,
        full_thresholds["start_threshold"],
        full_thresholds["end_threshold"],
    )
    active_legacy_reps = detect_reps(
        times,
        raw_values,
        smoothed_values,
        active_thresholds["start_threshold"],
        active_thresholds["end_threshold"],
    )

    if (
        len(active_reps) <= len(full_reps)
        and len(active_legacy_reps) <= len(full_legacy_reps)
    ):
        return full_thresholds

    full_first_start = full_reps[0]["start_time"] if full_reps else None
    active_first_start = active_reps[0]["start_time"] if active_reps else None

    if (
        full_first_start is not None
        and active_first_start is not None
        and active_first_start < full_first_start - HYBRID_INITIAL_LULL_SECONDS
    ):
        return full_thresholds

    return active_thresholds


def threshold_values_for_reference(times, smoothed_values, reference_values):
    baseline = low_percentile_average(reference_values, BASELINE_PERCENTILE)
    max_signal = max(reference_values)
    signal_range = max_signal - baseline
    start_threshold = baseline + START_THRESHOLD_FRACTION * signal_range
    end_threshold = baseline + END_THRESHOLD_FRACTION * signal_range

    return {
        "baseline": baseline,
        "max_signal": max_signal,
        "signal_range": signal_range,
        "start_threshold": start_threshold,
        "end_threshold": end_threshold,
        "reference_sample_count": len(reference_values),
        "reference_end_time": times[len(reference_values) - 1] if times else None,
    }


def threshold_reference_values(times, smoothed_values):
    """Use the active prefix for threshold estimation, excluding trailing quiet.

    The detector still runs over the full signal. This only prevents appended
    post-set inactivity from lowering the low-percentile baseline and merging
    already-finished reps.
    """
    if not smoothed_values:
        return []

    baseline = low_percentile_average(smoothed_values, BASELINE_PERCENTILE)
    max_signal = max(smoothed_values)
    signal_range = max_signal - baseline

    if signal_range <= 0:
        return smoothed_values

    preliminary_end_threshold = baseline + END_THRESHOLD_FRACTION * signal_range
    last_active_index = None

    for index in range(len(smoothed_values) - 1, -1, -1):
        if smoothed_values[index] > preliminary_end_threshold:
            last_active_index = index
            break

    if last_active_index is None:
        return smoothed_values

    return smoothed_values[:last_active_index + 1]


def detect_reps(times, raw_values, smoothed_values, start_threshold, end_threshold):
    reps = []
    in_rep = False
    current_rep = None
    cooldown_until = None
    previous_smooth_value = None

    for time, raw_value, smooth_value in zip(times, raw_values, smoothed_values):
        cooldown_active = cooldown_until is not None and time < cooldown_until
        crossed_start_threshold = (
            previous_smooth_value is not None
            and previous_smooth_value <= start_threshold
            and smooth_value > start_threshold
        )
        recording_starts_above_threshold = (
            previous_smooth_value is None
            and cooldown_until is None
            and smooth_value >= start_threshold
        )

        # After an accepted rep, ignore any start-threshold crossing that occurs
        # during cooldown. A new rep must be armed by a fresh upward crossing
        # after cooldown has elapsed, not by a signal that is already high.
        if (
            not in_rep
            and not cooldown_active
            and (recording_starts_above_threshold or crossed_start_threshold)
        ):
            in_rep = True
            current_rep = {
                "start_time": time,
                "end_time": None,
                "peak_value": raw_value,
                "peak_time": time,
                "values": [],
            }

        if not in_rep:
            previous_smooth_value = smooth_value
            continue

        current_rep["values"].append(raw_value)

        if raw_value > current_rep["peak_value"]:
            current_rep["peak_value"] = raw_value
            current_rep["peak_time"] = time

        if smooth_value <= end_threshold:
            current_rep["end_time"] = time
            duration = current_rep["end_time"] - current_rep["start_time"]

            if duration >= MIN_REP_DURATION_SECONDS:
                reps.append(current_rep)
                cooldown_until = current_rep["end_time"] + MIN_GAP_SECONDS

            in_rep = False
            current_rep = None

        previous_smooth_value = smooth_value

    if in_rep and current_rep is not None:
        current_rep["end_time"] = times[-1]
        duration = current_rep["end_time"] - current_rep["start_time"]

        if duration >= MIN_REP_DURATION_SECONDS:
            reps.append(current_rep)

    return reject_initial_transient_reps(
        times,
        smoothed_values,
        reps,
        start_threshold,
    )


def index_at_or_after(times, target_time):
    for index, time in enumerate(times):
        if time >= target_time:
            return index

    return len(times) - 1


def legacy_active_regions(times, raw_values, smoothed_values, start_threshold, end_threshold):
    return detect_reps(
        times,
        raw_values,
        smoothed_values,
        start_threshold,
        end_threshold,
    )


def rep_from_indices(times, raw_values, start_index, end_index):
    values = raw_values[start_index:end_index + 1]
    peak_offset, peak_value = max(
        enumerate(values),
        key=lambda item: item[1],
    )
    peak_index = start_index + peak_offset

    return {
        "start_time": times[start_index],
        "end_time": times[end_index],
        "peak_value": peak_value,
        "peak_time": times[peak_index],
        "values": values,
    }


def rep_index_bounds(times, rep):
    return (
        index_at_or_after(times, rep["start_time"]),
        index_at_or_after(times, rep["end_time"]),
    )


def rep_contraction_evidence(times, smoothed_values, rep, start_threshold):
    start_index, end_index = rep_index_bounds(times, rep)
    quality = rep_signal_quality(
        times,
        smoothed_values,
        start_index,
        end_index,
        start_threshold,
        max(1, max(smoothed_values) - start_threshold),
    )
    evidence = segment_contraction_evidence(
        times,
        smoothed_values,
        start_index,
        end_index,
        start_threshold,
    )
    evidence["peak_excess"] = quality["peak_excess"]
    evidence["area_above_start"] = quality["area_above_start"]
    return evidence


def weak_initial_transient(first_evidence, later_evidence):
    if not later_evidence:
        return False

    typical_duration = median([evidence["duration"] for evidence in later_evidence])
    typical_area = median([evidence["area_above_start"] for evidence in later_evidence])
    typical_peak_excess = median([evidence["peak_excess"] for evidence in later_evidence])

    return (
        typical_duration > 0
        and typical_area > 0
        and typical_peak_excess > 0
        and first_evidence["duration"] < typical_duration * INITIAL_TRANSIENT_DURATION_MEDIAN_FRACTION
        and first_evidence["area_above_start"] < typical_area * INITIAL_TRANSIENT_AREA_MEDIAN_FRACTION
        and first_evidence["peak_excess"] < typical_peak_excess * INITIAL_TRANSIENT_PEAK_EXCESS_MEDIAN_FRACTION
    )


def reject_initial_transient_reps(times, smoothed_values, reps, start_threshold):
    if len(reps) < 2:
        return reps

    evidence = [
        rep_contraction_evidence(
            times,
            smoothed_values,
            rep,
            start_threshold,
        )
        for rep in reps
    ]

    if weak_initial_transient(evidence[0], evidence[1:]):
        return reps[1:]

    return reps


def rep_signal_quality(times, smoothed_values, start_index, end_index, start_threshold, active_range):
    segment_values = smoothed_values[start_index:end_index + 1]
    duration = times[end_index] - times[start_index]
    peak = max(segment_values)
    peak_excess = peak - start_threshold

    if len(segment_values) > 1:
        sample_span = times[end_index] - times[start_index]
        sample_step = sample_span / (len(segment_values) - 1)
    else:
        sample_step = 0

    area_above_start = sum(
        max(0, value - start_threshold)
        for value in segment_values
    ) * sample_step
    min_peak_excess = active_range * HYBRID_MIN_REP_PEAK_EXCESS_FRACTION
    min_area = active_range * HYBRID_MIN_REP_AREA_FRACTION_SECONDS
    reasons = []

    if peak_excess < min_peak_excess and area_above_start < min_area:
        reasons.append("low-amplitude rebound without enough area above activity threshold")

    return {
        "start_time": times[start_index],
        "end_time": times[end_index],
        "duration": duration,
        "smoothed_peak": peak,
        "peak_excess": peak_excess,
        "area_above_start": area_above_start,
        "min_peak_excess": min_peak_excess,
        "min_area": min_area,
        "accepted": not reasons,
        "reasons": reasons,
    }


def local_minima_indices(smoothed_values, start_index, end_index):
    candidates = []

    for index in range(start_index + 1, end_index):
        if (
            smoothed_values[index] <= smoothed_values[index - 1]
            and smoothed_values[index] < smoothed_values[index + 1]
        ):
            candidates.append(index)

    return candidates


def cluster_valleys(times, smoothed_values, candidate_indices):
    clusters = []

    for index in candidate_indices:
        if not clusters or times[index] - times[clusters[-1][-1]] > HYBRID_VALLEY_CLUSTER_SECONDS:
            clusters.append([index])
        else:
            clusters[-1].append(index)

    return [
        min(cluster, key=lambda candidate: smoothed_values[candidate])
        for cluster in clusters
    ]


def plateau_center_time(times, smoothed_values, start_index, end_index, peak_value, local_range):
    plateau_window = max(10, local_range * HYBRID_PLATEAU_FRACTION)
    plateau_indices = [
        index
        for index in range(start_index, end_index + 1)
        if smoothed_values[index] >= peak_value - plateau_window
    ]

    if not plateau_indices:
        peak_index = max(
            range(start_index, end_index + 1),
            key=lambda index: smoothed_values[index],
        )
        return times[peak_index]

    return average([times[index] for index in plateau_indices])


def valley_duration(times, smoothed_values, start_index, end_index, valley_index, low_threshold):
    left_index = valley_index
    right_index = valley_index

    while left_index > start_index and smoothed_values[left_index - 1] <= low_threshold:
        left_index -= 1

    while right_index < end_index and smoothed_values[right_index + 1] <= low_threshold:
        right_index += 1

    return times[right_index] - times[left_index]


def median(values):
    if not values:
        return 0

    sorted_values = sorted(values)
    middle = len(sorted_values) // 2

    if len(sorted_values) % 2:
        return sorted_values[middle]

    return (sorted_values[middle - 1] + sorted_values[middle]) / 2


def sustained_duration(times, indices):
    longest = 0
    run_start = None
    previous_index = None

    for index in indices:
        if run_start is None or index != previous_index + 1:
            if run_start is not None:
                longest = max(longest, times[previous_index] - times[run_start])

            run_start = index

        previous_index = index

    if run_start is not None:
        longest = max(longest, times[previous_index] - times[run_start])

    return longest


def segment_contraction_evidence(times, smoothed_values, start_index, end_index, start_threshold):
    segment_values = smoothed_values[start_index:end_index + 1]
    duration = times[end_index] - times[start_index]
    peak = max(segment_values)
    high_threshold = start_threshold + (peak - start_threshold) * 0.65

    if len(segment_values) > 1:
        sample_step = duration / (len(segment_values) - 1)
    else:
        sample_step = 0

    high_indices = [
        index
        for index in range(start_index, end_index + 1)
        if smoothed_values[index] >= high_threshold
    ]
    area_above_high = sum(
        max(0, value - high_threshold)
        for value in segment_values
    ) * sample_step

    return {
        "start_time": times[start_index],
        "end_time": times[end_index],
        "duration": duration,
        "peak": peak,
        "high_threshold": high_threshold,
        "plateau_duration": sustained_duration(times, high_indices),
        "area_above_high": area_above_high,
    }


def weak_short_fragment(evidence, typical_duration, typical_plateau, typical_area):
    short_duration = (
        typical_duration > 0
        and evidence["duration"] < typical_duration * HYBRID_SHORT_FRAGMENT_MEDIAN_FRACTION
    )
    weak_plateau = (
        typical_plateau > 0
        and evidence["plateau_duration"] < typical_plateau * HYBRID_FRAGMENT_PLATEAU_MEDIAN_FRACTION
    )
    weak_area = (
        typical_area > 0
        and evidence["area_above_high"] < typical_area * HYBRID_FRAGMENT_AREA_MEDIAN_FRACTION
    )

    return short_duration and weak_plateau and weak_area


def reject_split_candidate(diagnostics, split_time, reason):
    remaining_accepted = []

    for candidate in diagnostics["accepted_valleys"]:
        if abs(candidate["time"] - split_time) < 0.001:
            candidate["accepted"] = False
            candidate["reasons"].append(reason)
        else:
            remaining_accepted.append(candidate)

    diagnostics["accepted_valleys"] = remaining_accepted


def split_candidate_score(diagnostics, split_time):
    for candidate in diagnostics["accepted_valleys"]:
        if abs(candidate["time"] - split_time) < 0.001:
            return candidate["score"]

    return 0


def reject_fragment_splits(times, smoothed_values, segment_indices, start_threshold, diagnostics):
    if len(segment_indices) < 4:
        return segment_indices

    refined_segments = segment_indices[:]

    while len(refined_segments) >= 3:
        evidence = [
            segment_contraction_evidence(
                times,
                smoothed_values,
                segment_start,
                segment_end,
                start_threshold,
            )
            for segment_start, segment_end in refined_segments
        ]
        durations = [segment["duration"] for segment in evidence]
        typical_duration = median(durations)
        supported = [
            segment
            for segment in evidence
            if segment["duration"] >= typical_duration * HYBRID_SHORT_FRAGMENT_MEDIAN_FRACTION
        ]
        typical_plateau = median([segment["plateau_duration"] for segment in supported])
        typical_area = median([segment["area_above_high"] for segment in supported])
        merge_index = None

        for index, segment in enumerate(evidence):
            if weak_short_fragment(segment, typical_duration, typical_plateau, typical_area):
                merge_index = index
                break

        if merge_index is None:
            break

        if merge_index == 0:
            removed_split_time = times[refined_segments[0][1]]
            refined_segments[1] = (refined_segments[0][0], refined_segments[1][1])
            del refined_segments[0]
        elif merge_index == len(refined_segments) - 1:
            removed_split_time = times[refined_segments[merge_index][0]]
            refined_segments[merge_index - 1] = (
                refined_segments[merge_index - 1][0],
                refined_segments[merge_index][1],
            )
            del refined_segments[merge_index]
        else:
            left_split_time = times[refined_segments[merge_index][0]]
            right_split_time = times[refined_segments[merge_index][1]]

            if (
                split_candidate_score(diagnostics, right_split_time)
                <= split_candidate_score(diagnostics, left_split_time)
            ):
                removed_split_time = right_split_time
                refined_segments[merge_index] = (
                    refined_segments[merge_index][0],
                    refined_segments[merge_index + 1][1],
                )
                del refined_segments[merge_index + 1]
            else:
                removed_split_time = left_split_time
                refined_segments[merge_index - 1] = (
                    refined_segments[merge_index - 1][0],
                    refined_segments[merge_index][1],
                )
                del refined_segments[merge_index]

        reject_split_candidate(
            diagnostics,
            removed_split_time,
            "split creates a cadence-relative short fragment without independent plateau support",
        )

    return refined_segments


def autocorrelation_cycle_duration(times, smoothed_values, start_index, end_index):
    duration = times[end_index] - times[start_index]

    if duration < HYBRID_MIN_CYCLE_PERIOD_SECONDS * 2:
        return 0

    sampled_values = []
    sample_time = times[start_index]
    source_index = start_index

    while sample_time <= times[end_index]:
        while source_index + 1 <= end_index and times[source_index + 1] < sample_time:
            source_index += 1

        sampled_values.append(smoothed_values[source_index])
        sample_time += HYBRID_PERIOD_SAMPLE_SECONDS

    if len(sampled_values) < 10:
        return 0

    mean_value = average(sampled_values)
    centered = [value - mean_value for value in sampled_values]
    denominator = sum(value * value for value in centered)

    if denominator <= 0:
        return 0

    min_lag = int(HYBRID_MIN_CYCLE_PERIOD_SECONDS / HYBRID_PERIOD_SAMPLE_SECONDS)
    max_lag = min(
        int(HYBRID_MAX_CYCLE_PERIOD_SECONDS / HYBRID_PERIOD_SAMPLE_SECONDS),
        len(centered) // 2,
    )
    correlations = []

    for lag in range(min_lag, max_lag):
        numerator = sum(
            centered[index] * centered[index + lag]
            for index in range(len(centered) - lag)
        )
        correlations.append((
            numerator / denominator,
            lag * HYBRID_PERIOD_SAMPLE_SECONDS,
        ))

    local_peaks = [
        correlations[index]
        for index in range(1, len(correlations) - 1)
        if (
            correlations[index][0] > correlations[index - 1][0]
            and correlations[index][0] >= correlations[index + 1][0]
        )
    ]

    if not local_peaks:
        return 0

    best_correlation = max(peak[0] for peak in local_peaks)

    if best_correlation <= 0:
        return 0

    competitive_peaks = [
        peak
        for peak in local_peaks
        if peak[0] >= best_correlation * 0.8
    ]

    return min(competitive_peaks, key=lambda peak: peak[1])[1]


def estimate_cycle_duration(times, smoothed_values, start_index, end_index, candidate_nodes):
    autocorrelation_period = autocorrelation_cycle_duration(
        times,
        smoothed_values,
        start_index,
        end_index,
    )

    if autocorrelation_period:
        return autocorrelation_period

    spacings = [
        times[candidate_nodes[index + 1]] - times[candidate_nodes[index]]
        for index in range(len(candidate_nodes) - 1)
    ]
    supported_spacings = [
        spacing
        for spacing in spacings
        if spacing >= HYBRID_MIN_CYCLE_PERIOD_SECONDS
    ]

    if supported_spacings:
        return median(supported_spacings)

    return median(spacings)


def global_segment_score(times, smoothed_values, start_index, end_index, start_threshold, cycle_duration):
    duration = times[end_index] - times[start_index]
    ratio = duration / cycle_duration if cycle_duration else 1
    duration_penalty = (
        math.log(max(ratio, 0.05)) / 0.55
    ) ** 2 * 2.5

    if ratio < 0.5:
        duration_penalty += (0.5 - ratio) * 14

    if ratio > 1.75:
        duration_penalty += (ratio - 1.75) * 6

    evidence = segment_contraction_evidence(
        times,
        smoothed_values,
        start_index,
        end_index,
        start_threshold,
    )
    plateau_score = min(
        1.5,
        evidence["plateau_duration"] / max(0.1, cycle_duration) * 2.2,
    )
    area_score = min(1.0, evidence["area_above_high"] / 350)

    return 2.0 + plateau_score + area_score - duration_penalty


def global_boundary_score(candidate):
    return (
        min(4.0, math.log1p(candidate["score"]) / 2.2)
        + candidate["normalized_adjacent_drop"] * 1.2
        + min(0.8, candidate["valley_duration"] * 2)
        - 3.8
    )


def select_global_cycle_segments(times, smoothed_values, segment_indices, start_threshold, diagnostics):
    if len(segment_indices) < 4:
        return segment_indices

    candidate_nodes = sorted(
        {segment_start for segment_start, _ in segment_indices}
        | {segment_end for _, segment_end in segment_indices}
    )
    candidates_by_index = {
        candidate["index"]: candidate
        for candidate in diagnostics["accepted_valleys"]
        if candidate["index"] in candidate_nodes
    }

    if not candidates_by_index:
        return segment_indices

    cycle_duration = estimate_cycle_duration(
        times,
        smoothed_values,
        candidate_nodes[0],
        candidate_nodes[-1],
        candidate_nodes,
    )
    diagnostics["global_cycle_selection"].append({
        "start_time": times[candidate_nodes[0]],
        "end_time": times[candidate_nodes[-1]],
        "estimated_cycle_duration": cycle_duration,
        "candidate_boundaries": [
            times[node]
            for node in candidate_nodes[1:-1]
        ],
    })

    best_scores = [-1e9] * len(candidate_nodes)
    previous_nodes = [None] * len(candidate_nodes)
    best_scores[0] = 0

    for end_position in range(1, len(candidate_nodes)):
        for start_position in range(end_position):
            segment_start = candidate_nodes[start_position]
            segment_end = candidate_nodes[end_position]
            duration = times[segment_end] - times[segment_start]

            if duration < MIN_REP_DURATION_SECONDS:
                continue

            score = (
                best_scores[start_position]
                + global_segment_score(
                    times,
                    smoothed_values,
                    segment_start,
                    segment_end,
                    start_threshold,
                    cycle_duration,
                )
            )

            if end_position < len(candidate_nodes) - 1:
                score += global_boundary_score(candidates_by_index[segment_end])

            if score > best_scores[end_position]:
                best_scores[end_position] = score
                previous_nodes[end_position] = start_position

    selected_positions = []
    position = len(candidate_nodes) - 1

    while position is not None:
        selected_positions.append(position)
        position = previous_nodes[position]

    selected_positions.reverse()
    selected_nodes = [candidate_nodes[position] for position in selected_positions]
    selected_boundaries = set(selected_nodes[1:-1])
    selected_segments = [
        (selected_nodes[index], selected_nodes[index + 1])
        for index in range(len(selected_nodes) - 1)
    ]

    for candidate in diagnostics["accepted_valleys"]:
        if candidate["index"] not in candidates_by_index:
            continue

        if candidate["index"] in selected_boundaries:
            continue

        candidate["accepted"] = False
        candidate["reasons"].append("not selected by global contraction-cycle sequence")

    diagnostics["accepted_valleys"] = [
        candidate
        for candidate in diagnostics["accepted_valleys"]
        if candidate["accepted"]
    ]
    diagnostics["global_cycle_selection"][-1]["selected_boundaries"] = [
        times[node]
        for node in selected_boundaries
    ]

    return selected_segments


def adjusted_hybrid_region_start(times, smoothed_values, start_index, end_index, start_threshold):
    if start_index != 0 or smoothed_values[start_index] < start_threshold:
        return start_index, None

    region_peak = max(smoothed_values[start_index:end_index + 1])
    substantial_level = start_threshold + (
        region_peak - start_threshold
    ) * HYBRID_INITIAL_SUBSTANTIAL_RISE_FRACTION
    first_substantial_index = None

    for index in range(start_index + 1, end_index + 1):
        if smoothed_values[index] >= substantial_level:
            first_substantial_index = index
            break

    if first_substantial_index is None:
        return start_index, None

    initial_peak = max(smoothed_values[start_index:first_substantial_index])

    if initial_peak >= substantial_level:
        return start_index, None

    low_indices = [
        index
        for index in range(start_index, first_substantial_index + 1)
        if smoothed_values[index] < start_threshold
    ]

    if not low_indices:
        low_duration = times[first_substantial_index] - times[start_index]

        if low_duration < HYBRID_INITIAL_LULL_SECONDS:
            return start_index, None

        return first_substantial_index, {
            "from_time": times[start_index],
            "to_time": times[first_substantial_index],
            "below_start_duration": 0,
            "initial_peak": initial_peak,
            "substantial_level": substantial_level,
            "reason": "recording opened below a substantial contraction level before first supported rise",
        }

    last_low_index = low_indices[-1]

    crossing_index = first_substantial_index

    for index in range(last_low_index + 1, first_substantial_index + 1):
        if (
            smoothed_values[index - 1] <= start_threshold
            and smoothed_values[index] > start_threshold
        ):
            crossing_index = index
            break

    below_duration = times[crossing_index] - times[last_low_index]

    if below_duration < HYBRID_INITIAL_LULL_SECONDS:
        preceding_lull = times[first_substantial_index] - times[start_index]

        if preceding_lull < HYBRID_INITIAL_LULL_SECONDS:
            return start_index, None

    return crossing_index, {
        "from_time": times[start_index],
        "to_time": times[crossing_index],
        "below_start_duration": below_duration,
        "initial_peak": initial_peak,
        "substantial_level": substantial_level,
        "reason": "recording opened just above start threshold before first substantial contraction rise",
    }


def evaluate_split_candidate(times, smoothed_values, start_index, end_index, valley_index, start_threshold):
    segment_duration = times[end_index] - times[start_index]
    left_duration = times[valley_index] - times[start_index]
    right_duration = times[end_index] - times[valley_index]
    local_max = max(smoothed_values[start_index:end_index + 1])
    local_min = min(smoothed_values[start_index:end_index + 1])
    local_range = max(1, local_max - local_min)
    valley_value = smoothed_values[valley_index]
    left_peak = max(smoothed_values[start_index:valley_index + 1])
    right_peak = max(smoothed_values[valley_index:end_index + 1])
    adjacent_peak = min(left_peak, right_peak)
    drop_from_left = left_peak - valley_value
    rebound_height = right_peak - valley_value
    adjacent_drop = min(drop_from_left, rebound_height)
    normalized_local_drop = adjacent_drop / local_range
    normalized_adjacent_drop = adjacent_drop / max(1, adjacent_peak - local_min)
    low_threshold = valley_value + adjacent_drop * 0.25
    low_duration = valley_duration(
        times,
        smoothed_values,
        start_index,
        end_index,
        valley_index,
        low_threshold,
    )
    left_center = plateau_center_time(
        times,
        smoothed_values,
        start_index,
        valley_index,
        left_peak,
        local_range,
    )
    right_center = plateau_center_time(
        times,
        smoothed_values,
        valley_index,
        end_index,
        right_peak,
        local_range,
    )
    center_gap = right_center - left_center
    min_drop = max(
        HYBRID_MIN_ABSOLUTE_DROP,
        local_range * HYBRID_MIN_LOCAL_RANGE_DROP_FRACTION,
    )
    strong_relative_drop = (
        normalized_adjacent_drop >= HYBRID_STRONG_RELATIVE_DROP_FRACTION
        and normalized_local_drop >= HYBRID_STRONG_LOCAL_DROP_FRACTION
        and low_duration >= HYBRID_STRONG_RELATIVE_VALLEY_DURATION_SECONDS
    )
    reasons = []

    if segment_duration < HYBRID_MIN_SPLITTABLE_REGION_SECONDS:
        reasons.append("segment too short to split")

    if left_duration < HYBRID_MIN_SEGMENT_DURATION_SECONDS:
        reasons.append("left segment too short")

    if right_duration < HYBRID_MIN_SEGMENT_DURATION_SECONDS:
        reasons.append("right segment too short")

    if left_peak < start_threshold or right_peak < start_threshold:
        reasons.append("adjacent contraction peak below start threshold")

    if adjacent_drop < min_drop and not strong_relative_drop:
        reasons.append("valley drop too small")

    if normalized_adjacent_drop < HYBRID_MIN_ADJACENT_PEAK_DROP_FRACTION:
        reasons.append("valley is not deep relative to adjacent peaks")

    if low_duration < HYBRID_MIN_VALLEY_DURATION_SECONDS:
        reasons.append("valley duration too short")

    if center_gap < HYBRID_MIN_CENTER_GAP_SECONDS:
        reasons.append("adjacent contraction centers too close")

    return {
        "index": valley_index,
        "time": times[valley_index],
        "value": valley_value,
        "left_peak": left_peak,
        "right_peak": right_peak,
        "drop_from_left": drop_from_left,
        "rebound_height": rebound_height,
        "adjacent_drop": adjacent_drop,
        "normalized_local_drop": normalized_local_drop,
        "normalized_adjacent_drop": normalized_adjacent_drop,
        "valley_duration": low_duration,
        "left_center": left_center,
        "right_center": right_center,
        "center_gap": center_gap,
        "accepted": not reasons,
        "reasons": reasons,
        "score": adjacent_drop * normalized_adjacent_drop * max(0.1, center_gap),
    }


def best_split_candidate(times, smoothed_values, start_index, end_index, start_threshold):
    valley_indices = cluster_valleys(
        times,
        smoothed_values,
        local_minima_indices(smoothed_values, start_index, end_index),
    )
    candidates = [
        evaluate_split_candidate(
            times,
            smoothed_values,
            start_index,
            end_index,
            valley_index,
            start_threshold,
        )
        for valley_index in valley_indices
    ]
    accepted = [candidate for candidate in candidates if candidate["accepted"]]

    if not accepted:
        return None, candidates

    selected = max(accepted, key=lambda candidate: candidate["score"])

    for candidate in accepted:
        if candidate is not selected:
            candidate["accepted"] = False
            candidate["reasons"].append("lower-scoring candidate in same region")

    return selected, candidates


def split_active_region(times, smoothed_values, start_index, end_index, start_threshold, diagnostics):
    split, candidates = best_split_candidate(
        times,
        smoothed_values,
        start_index,
        end_index,
        start_threshold,
    )
    diagnostics["candidate_valleys"].extend(candidates)

    if split is None:
        return [(start_index, end_index)]

    diagnostics["accepted_valleys"].append(split)
    split_index = split["index"]

    return (
        split_active_region(
            times,
            smoothed_values,
            start_index,
            split_index,
            start_threshold,
            diagnostics,
        )
        + split_active_region(
            times,
            smoothed_values,
            split_index,
            end_index,
            start_threshold,
            diagnostics,
        )
    )


def detect_reps_hybrid(times, raw_values, smoothed_values, start_threshold, end_threshold):
    broad_regions = legacy_active_regions(
        times,
        raw_values,
        smoothed_values,
        start_threshold,
        end_threshold,
    )
    diagnostics = {
        "broad_regions": [],
        "candidate_valleys": [],
        "accepted_valleys": [],
        "rep_quality": [],
        "rejected_reps": [],
        "final_boundaries": [],
        "boundary_adjustments": [],
        "global_cycle_selection": [],
    }
    segment_indices = []

    for region in broad_regions:
        start_index = index_at_or_after(times, region["start_time"])
        end_index = index_at_or_after(times, region["end_time"])
        adjusted_start_index, boundary_adjustment = adjusted_hybrid_region_start(
            times,
            smoothed_values,
            start_index,
            end_index,
            start_threshold,
        )

        if boundary_adjustment is not None:
            diagnostics["boundary_adjustments"].append(boundary_adjustment)
            start_index = adjusted_start_index

        diagnostics["broad_regions"].append({
            "start_time": region["start_time"],
            "end_time": region["end_time"],
            "peak_time": region["peak_time"],
            "peak_value": region["peak_value"],
        })
        region_segments = split_active_region(
            times,
            smoothed_values,
            start_index,
            end_index,
            start_threshold,
            diagnostics,
        )
        segment_indices.extend(
            select_global_cycle_segments(
                times,
                smoothed_values,
                region_segments,
                start_threshold,
                diagnostics,
            )
        )

    active_range = max(1, max(smoothed_values) - end_threshold)
    reps = []

    for start_index, end_index in segment_indices:
        if times[end_index] - times[start_index] < MIN_REP_DURATION_SECONDS:
            continue

        quality = rep_signal_quality(
            times,
            smoothed_values,
            start_index,
            end_index,
            start_threshold,
            active_range,
        )
        diagnostics["rep_quality"].append(quality)

        if quality["accepted"]:
            reps.append(rep_from_indices(times, raw_values, start_index, end_index))
        else:
            diagnostics["rejected_reps"].append(quality)

    diagnostics["final_boundaries"] = [
        {
            "start_time": rep["start_time"],
            "end_time": rep["end_time"],
            "peak_time": rep["peak_time"],
            "peak_value": rep["peak_value"],
        }
        for rep in reps
    ]

    return reps, diagnostics


def average(values):
    if not values:
        return 0

    return sum(values) / len(values)


def rep_duration(rep):
    return rep["end_time"] - rep["start_time"]


def rep_average_activation(rep):
    return average(rep["values"])


def average_rep_peak(reps):
    return average([rep["peak_value"] for rep in reps])


def apply_normalization_to_reps(reps, calibration):
    if calibration is None:
        return

    for rep in reps:
        normalized_values = normalize_values(rep["values"], calibration)
        rep["normalized_values"] = normalized_values
        rep["normalized_average_activation"] = average(normalized_values)
        rep["normalized_peak_activation"] = max(normalized_values) if normalized_values else 0


def build_set_summary(values, reps, calibration=None):
    total_reps = len(reps)
    rep_durations = [rep_duration(rep) for rep in reps]
    rep_activations = [rep_average_activation(rep) for rep in reps]
    normalized_values = normalize_values(values, calibration) if calibration else []
    normalized_rep_averages = [
        rep.get("normalized_average_activation", 0)
        for rep in reps
        if "normalized_average_activation" in rep
    ]
    strongest_rep = None

    if reps:
        set_duration = reps[-1]["end_time"] - reps[0]["start_time"]
        strongest_rep_index, strongest_rep_value = max(
            enumerate([rep["peak_value"] for rep in reps], start=1),
            key=lambda item: item[1],
        )
        strongest_rep = {
            "number": strongest_rep_index,
            "peak_value": strongest_rep_value,
        }
    else:
        set_duration = 0

    split_index = total_reps // 2
    first_half_reps = reps[:split_index]
    second_half_reps = reps[split_index:]
    first_half_average_peak = average_rep_peak(first_half_reps)
    second_half_average_peak = average_rep_peak(second_half_reps)

    if first_half_average_peak > 0 and second_half_reps:
        peak_drop_percent = (
            (first_half_average_peak - second_half_average_peak)
            / first_half_average_peak
            * 100
        )
    else:
        peak_drop_percent = 0

    return {
        "total_reps": total_reps,
        "set_duration": set_duration,
        "overall_average_signal": average(values),
        "overall_peak_signal": max(values) if values else 0,
        "average_rep_duration": average(rep_durations),
        "average_peak_across_reps": average_rep_peak(reps),
        "average_activation_across_reps": average(rep_activations),
        "first_half_average_rep_peak": first_half_average_peak,
        "second_half_average_rep_peak": second_half_average_peak,
        "peak_drop_percent": peak_drop_percent,
        "possible_fatigue": peak_drop_percent >= 15,
        "strongest_rep": strongest_rep,
        "has_calibration": calibration is not None,
        "normalized_average_activation": average(normalized_values),
        "normalized_peak_activation": max(normalized_values) if normalized_values else 0,
        "normalized_average_rep_activation": average(normalized_rep_averages),
    }


def summary_lines(summary):
    fatigue_text = "yes" if summary["possible_fatigue"] else "no"
    activation_change = activation_change_phrase(summary["peak_drop_percent"])

    lines = [
        "Set Summary",
        f"Total reps: {summary['total_reps']}",
        f"Set duration: {summary['set_duration']:.2f}s",
        f"Overall average signal: {summary['overall_average_signal']:.1f}",
        f"Overall peak signal: {summary['overall_peak_signal']:.1f}",
        f"Average rep duration: {summary['average_rep_duration']:.2f}s",
        f"Average rep peak: {summary['average_peak_across_reps']:.1f}",
        f"Average rep activation: {summary['average_activation_across_reps']:.1f}",
        f"First-half average rep peak: {summary['first_half_average_rep_peak']:.1f}",
        f"Second-half average rep peak: {summary['second_half_average_rep_peak']:.1f}",
        "Activation change from first half to second half: "
        f"{activation_change}",
        f"Signal may indicate greater fatigue: {fatigue_text}",
    ]

    if summary["has_calibration"]:
        lines.extend([
            "Normalized Stats",
            "Full-recording normalized average: "
            f"{summary['normalized_average_activation']:.1f}%",
            "Average normalized rep activation: "
            f"{summary['normalized_average_rep_activation']:.1f}%",
            "Peak normalized activation: "
            f"{summary['normalized_peak_activation']:.1f}%",
        ])

    return lines


def activation_change_phrase(change_percent):
    if abs(change_percent) <= CONSISTENT_FATIGUE_THRESHOLD:
        return "changed very little"

    if change_percent > 0:
        return f"decreased by {change_percent:.1f}%"

    return f"increased by {abs(change_percent):.1f}%"


def activation_trend_line(peak_drop_percent):
    change_phrase = activation_change_phrase(peak_drop_percent)
    return (
        f"Your activation {change_phrase} from the first half "
        "to the second half."
    )


def fatigue_status_line(peak_drop_percent):
    if peak_drop_percent >= 15:
        return "Activation decreased across the set, so the signal may indicate greater fatigue."

    if peak_drop_percent > CONSISTENT_FATIGUE_THRESHOLD:
        return "Activation decreased across the set, but not enough to flag a strong fatigue signal."

    if peak_drop_percent < -CONSISTENT_FATIGUE_THRESHOLD:
        return "Activation increased across the set, so this does not suggest greater fatigue."

    return "Activation changed very little across the set, so fatigue was not flagged."


def user_insight_lines(summary):
    insights = ["User Insights"]
    total_reps = summary["total_reps"]
    strongest_rep = summary["strongest_rep"]
    peak_drop_percent = summary["peak_drop_percent"]

    insights.append(f"You completed {total_reps} reps.")

    if strongest_rep is not None:
        insights.append(f"Your strongest rep was rep {strongest_rep['number']}.")
    else:
        insights.append("No strongest rep was found because no reps were detected.")

    insights.append(activation_trend_line(peak_drop_percent))
    insights.append(fatigue_status_line(peak_drop_percent))

    if summary["has_calibration"]:
        insights.append(
            "Using calibration, your active reps averaged "
            f"{summary['normalized_average_rep_activation']:.1f}% activation."
        )
        insights.append(
            "The full recording averaged "
            f"{summary['normalized_average_activation']:.1f}% activation "
            "including rest time."
        )
        insights.append(
            "Peak activation reached "
            f"{summary['normalized_peak_activation']:.1f}%."
        )

        if summary["normalized_peak_activation"] > 110:
            insights.append(
                "Peak activation exceeded the calibration max, so this session may need recalibration."
            )
    else:
        insights.append("No calibration file was found, so normalized activation was skipped.")

    return insights


def print_stats(
    csv_file,
    metadata,
    baseline,
    max_signal,
    start_threshold,
    end_threshold,
    reps,
    summary,
    calibration,
    detector_method="legacy",
    diagnostics=None,
):
    print(f"Analyzing file: {csv_file.name}")
    print(f"Detector method: {detector_method}")

    for line in metadata_lines(metadata):
        print(line)

    if metadata:
        print()

    print(f"Baseline: {baseline:.1f}")
    print(f"Max signal: {max_signal:.1f}")
    print(f"Start threshold: {start_threshold:.1f}")
    print(f"End threshold: {end_threshold:.1f}")
    if calibration:
        print("Calibration loaded:")
        print(f"  Source CSV: {calibration.get('source_csv', 'unknown')}")
        print(f"  Baseline: {calibration['baseline']:.1f}")
        print(f"  Max flex: {calibration['max_flex']:.1f}")
        print(f"  Signal range: {calibration['signal_range']:.1f}")
    else:
        print("Calibration loaded: no")
    print(f"Detected rep count: {len(reps)}")
    if diagnostics:
        accepted_count = len(diagnostics.get("accepted_valleys", []))
        candidate_count = len(diagnostics.get("candidate_valleys", []))
        print(f"Hybrid candidate valleys: {candidate_count}")
        print(f"Hybrid accepted split valleys: {accepted_count}")
    print()

    for index, rep in enumerate(reps, start=1):
        duration = rep_duration(rep)
        average_activation = rep_average_activation(rep)

        print(f"Rep {index}")
        print(f"  Start time: {rep['start_time']:.2f}s")
        print(f"  End time: {rep['end_time']:.2f}s")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Peak value: {rep['peak_value']:.1f}")
        print(f"  Average value: {average_activation:.1f}")

        if "normalized_average_activation" in rep:
            print(
                "  Normalized average rep activation: "
                f"{rep['normalized_average_activation']:.1f}%"
            )
            print(
                "  Normalized peak rep activation: "
                f"{rep['normalized_peak_activation']:.1f}%"
            )

        print()

    for line in summary_lines(summary):
        print(line)
    print()

    for line in user_insight_lines(summary):
        print(line)
    print()

    if diagnostics:
        print("Hybrid Diagnostics")
        for candidate in diagnostics.get("candidate_valleys", []):
            status = "accepted" if candidate["accepted"] else "rejected"
            reasons = ", ".join(candidate["reasons"]) if candidate["reasons"] else "none"
            print(
                f"  {status} valley at {candidate['time']:.2f}s "
                f"value={candidate['value']:.1f} "
                f"left_peak={candidate['left_peak']:.1f} "
                f"right_peak={candidate['right_peak']:.1f} "
                f"drop={candidate['adjacent_drop']:.1f} "
                f"rebound={candidate['rebound_height']:.1f} "
                f"duration={candidate['valley_duration']:.2f}s "
                f"center_gap={candidate['center_gap']:.2f}s "
                f"reasons={reasons}"
            )
        for rejected_rep in diagnostics.get("rejected_reps", []):
            reasons = ", ".join(rejected_rep["reasons"])
            print(
                f"  rejected rep interval {rejected_rep['start_time']:.2f}-"
                f"{rejected_rep['end_time']:.2f}s "
                f"smoothed_peak={rejected_rep['smoothed_peak']:.1f} "
                f"peak_excess={rejected_rep['peak_excess']:.1f} "
                f"area_above_start={rejected_rep['area_above_start']:.1f} "
                f"reasons={reasons}"
            )
        print()


def save_summary(csv_file, metadata, summary, summary_label="summary"):
    SUMMARIES_DIR.mkdir(exist_ok=True)
    summary_file = SUMMARIES_DIR / f"{csv_file.stem}_{summary_label}.txt"

    with open(summary_file, "w", newline="") as file:
        file.write(f"Source file: {csv_file.name}\n")

        metadata_text = metadata_lines(metadata)

        if metadata_text:
            file.write("\n".join(metadata_text))
            file.write("\n\n")

        file.write("\n".join(summary_lines(summary)))
        file.write("\n\n")
        file.write("\n".join(user_insight_lines(summary)))
        file.write("\n")

    print(f"Saved summary to: {summary_file.resolve()}")


def plot_reps(
    csv_file,
    times,
    raw_values,
    smoothed_values,
    start_threshold,
    end_threshold,
    reps,
    show_plot=True,
    diagnostics=None,
    graph_label="reps",
):
    GRAPHS_DIR.mkdir(exist_ok=True)
    output_graph = GRAPHS_DIR / f"{csv_file.stem}_{graph_label}.png"

    plt.figure(figsize=(12, 6))
    plt.plot(times, raw_values, color="gray", alpha=0.35, label="Raw signal")
    plt.plot(times, smoothed_values, color="blue", linewidth=2, label="Smoothed signal")

    plt.axhline(
        start_threshold,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label="Start threshold",
    )
    plt.axhline(
        end_threshold,
        color="red",
        linestyle=":",
        linewidth=1.5,
        label="End threshold",
    )

    for index, rep in enumerate(reps, start=1):
        plt.axvline(rep["start_time"], color="green", linestyle="--", alpha=0.45)
        plt.axvline(rep["end_time"], color="red", linestyle=":", alpha=0.45)
        plt.scatter(rep["peak_time"], rep["peak_value"], color="black", zorder=3)
        plt.text(
            rep["peak_time"],
            rep["peak_value"],
            f" Rep {index}",
            fontsize=9,
            va="bottom",
        )

    if diagnostics:
        for region in diagnostics.get("broad_regions", []):
            plt.axvspan(
                region["start_time"],
                region["end_time"],
                color="green",
                alpha=0.05,
            )

        rejected_candidates = [
            candidate
            for candidate in diagnostics.get("candidate_valleys", [])
            if not candidate["accepted"]
        ]
        accepted_candidates = diagnostics.get("accepted_valleys", [])

        if rejected_candidates:
            plt.scatter(
                [candidate["time"] for candidate in rejected_candidates],
                [candidate["value"] for candidate in rejected_candidates],
                marker="x",
                color="orange",
                zorder=4,
                label="Rejected split valley",
            )

        if accepted_candidates:
            plt.scatter(
                [candidate["time"] for candidate in accepted_candidates],
                [candidate["value"] for candidate in accepted_candidates],
                marker="v",
                color="purple",
                s=70,
                zorder=5,
                label="Accepted split valley",
            )
            for candidate in accepted_candidates:
                plt.axvline(candidate["time"], color="purple", linestyle="-.", alpha=0.55)

                plt.scatter(
                    [candidate["left_center"], candidate["right_center"]],
                    [candidate["left_peak"], candidate["right_peak"]],
                    marker="o",
                    color="black",
                    s=35,
                    zorder=5,
                    label=None,
                )

    plt.title(f"Detected Reps: {csv_file.name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    plt.savefig(output_graph, dpi=150)
    print(f"Saved graph to: {output_graph.resolve()}")
    if show_plot:
        plt.show()
    else:
        plt.close()

    return output_graph


def analyze_csv_file(csv_file, show_plot=True, method="legacy"):
    csv_file = Path(csv_file)
    metadata = load_metadata(csv_file)
    calibration = load_calibration_for_recording(csv_file, metadata)
    times, values = read_signal(csv_file)
    smoothed_values = moving_average(values, SMOOTHING_WINDOW)

    thresholds = detector_threshold_values(times, smoothed_values, values)
    baseline = thresholds["baseline"]
    max_signal = thresholds["max_signal"]
    start_threshold = thresholds["start_threshold"]
    end_threshold = thresholds["end_threshold"]

    if method == "legacy":
        reps = detect_reps(
            times,
            values,
            smoothed_values,
            start_threshold,
            end_threshold,
        )
        diagnostics = None
    elif method == "hybrid":
        reps, diagnostics = detect_reps_hybrid(
            times,
            values,
            smoothed_values,
            start_threshold,
            end_threshold,
        )
    else:
        raise ValueError(f"Unknown detector method: {method}")

    apply_normalization_to_reps(reps, calibration)
    summary = build_set_summary(values, reps, calibration)

    print_stats(
        csv_file,
        metadata,
        baseline,
        max_signal,
        start_threshold,
        end_threshold,
        reps,
        summary,
        calibration,
        detector_method=method,
        diagnostics=diagnostics,
    )
    output_label = "summary" if method == "legacy" else f"{method}_summary"
    graph_label = "reps" if method == "legacy" else f"{method}_reps"
    save_summary(csv_file, metadata, summary, summary_label=output_label)
    graph_file = plot_reps(
        csv_file,
        times,
        values,
        smoothed_values,
        start_threshold,
        end_threshold,
        reps,
        show_plot=show_plot,
        diagnostics=diagnostics,
        graph_label=graph_label,
    )

    return {
        "metadata": metadata,
        "reps": reps,
        "summary": summary,
        "summary_file": SUMMARIES_DIR / f"{csv_file.stem}_{output_label}.txt",
        "graph_file": graph_file,
        "diagnostics": diagnostics,
        "method": method,
    }


def main():
    latest_file = newest_csv_file(DATA_DIR)
    analyze_csv_file(latest_file)


if __name__ == "__main__":
    main()
