import matplotlib.pyplot as plt

from calibration_utils import load_latest_calibration, normalize_values
from detect_reps import (
    BASELINE_PERCENTILE,
    DATA_DIR,
    END_THRESHOLD_FRACTION,
    GRAPHS_DIR,
    SMOOTHING_WINDOW,
    START_THRESHOLD_FRACTION,
    SUMMARIES_DIR,
    activation_change_phrase,
    average,
    average_rep_peak,
    detect_reps,
    low_percentile_average,
    moving_average,
    read_signal,
    rep_average_activation,
    rep_duration,
)
from recording_metadata import load_metadata, metadata_lines


COMPARISON_SUMMARY_FILE = SUMMARIES_DIR / "latest_set_comparison.txt"
COMPARISON_GRAPH_FILE = GRAPHS_DIR / "latest_set_comparison.png"
MEANINGFUL_PERCENT_DIFFERENCE = 5
MEANINGFUL_FATIGUE_POINT_DIFFERENCE = 5
CONSISTENT_FATIGUE_THRESHOLD = 5


def real_workout_csv_files(folder):
    csv_files = []

    for csv_file in folder.glob("*.csv"):
        metadata = load_metadata(csv_file)

        if (
            metadata.get("data_type") == "real"
            and metadata.get("test_type") == "workout_set"
        ):
            csv_files.append(csv_file)

    if len(csv_files) < 2:
        raise FileNotFoundError(
            "Need at least two real workout-set CSV files in data/ to compare sets."
        )

    return sorted(csv_files, key=lambda path: path.stat().st_mtime)


def metadata_value(metadata, key):
    return str(metadata.get(key, "")).strip()


def compatible_workout_sets(csv_file_a, csv_file_b):
    metadata_a = load_metadata(csv_file_a)
    metadata_b = load_metadata(csv_file_b)

    if metadata_value(metadata_a, "muscle") != metadata_value(metadata_b, "muscle"):
        return False

    if metadata_value(metadata_a, "side") != metadata_value(metadata_b, "side"):
        return False

    weight_a = metadata_value(metadata_a, "weight")
    weight_b = metadata_value(metadata_b, "weight")

    return not weight_a or not weight_b or weight_a == weight_b


def automatic_comparison_csv_files(folder):
    csv_files = real_workout_csv_files(folder)
    newest_first = list(reversed(csv_files))

    for newer_index, newer_file in enumerate(newest_first):
        for older_file in newest_first[newer_index + 1:]:
            if compatible_workout_sets(older_file, newer_file):
                return [older_file, newer_file]

    return csv_files[-2:]


def percent_difference(higher_value, lower_value):
    if lower_value == 0:
        return 0

    return (higher_value - lower_value) / lower_value * 100


def percentage_point_difference(value_a, value_b):
    return abs(value_a - value_b)


def relative_times(times):
    if not times:
        return []

    start_time = times[0]
    return [time - start_time for time in times]


def fatigue_drop_percent(reps):
    total_reps = len(reps)
    split_index = total_reps // 2
    first_half_reps = reps[:split_index]
    second_half_reps = reps[split_index:]
    first_half_average_peak = average_rep_peak(first_half_reps)
    second_half_average_peak = average_rep_peak(second_half_reps)

    if first_half_average_peak > 0 and second_half_reps:
        return (
            (first_half_average_peak - second_half_average_peak)
            / first_half_average_peak
            * 100
        )

    return 0


def rep_metrics(reps, calibration=None):
    metrics = {
        "average_rep_peak": average_rep_peak(reps),
        "average_rep_activation": average(
            [rep_average_activation(rep) for rep in reps]
        ),
        "fatigue_drop_percent": fatigue_drop_percent(reps),
        "highest_peak_activation": max(
            [rep["peak_value"] for rep in reps],
            default=0,
        ),
    }

    if calibration:
        normalized_rep_averages = []
        normalized_rep_peaks = []

        for rep in reps:
            normalized_values = normalize_values(rep["values"], calibration)
            normalized_rep_averages.append(average(normalized_values))
            normalized_rep_peaks.append(max(normalized_values) if normalized_values else 0)

        metrics.update({
            "normalized_average_activation": average(normalized_rep_averages),
            "normalized_peak_activation": max(normalized_rep_peaks, default=0),
        })

    return metrics


def analyze_set(csv_file, calibration=None):
    metadata = load_metadata(csv_file)
    times, values = read_signal(csv_file)
    smoothed_values = moving_average(values, SMOOTHING_WINDOW)

    baseline = low_percentile_average(smoothed_values, BASELINE_PERCENTILE)
    max_signal = max(smoothed_values)
    signal_range = max_signal - baseline
    start_threshold = baseline + START_THRESHOLD_FRACTION * signal_range
    end_threshold = baseline + END_THRESHOLD_FRACTION * signal_range

    reps = detect_reps(
        times,
        values,
        smoothed_values,
        start_threshold,
        end_threshold,
    )
    total_reps = len(reps)
    full_rep_metrics = rep_metrics(reps, calibration)

    if reps:
        set_duration = reps[-1]["end_time"] - reps[0]["start_time"]
    else:
        set_duration = 0

    return {
        "file": csv_file,
        "metadata": metadata,
        "times": times,
        "relative_times": relative_times(times),
        "values": values,
        "smoothed_values": smoothed_values,
        "reps": reps,
        "rep_count": total_reps,
        "set_duration": set_duration,
        "overall_average_signal": average(values),
        "overall_peak_signal": max(values) if values else 0,
        "average_rep_peak": full_rep_metrics["average_rep_peak"],
        "average_rep_activation": full_rep_metrics["average_rep_activation"],
        "average_rep_duration": average([rep_duration(rep) for rep in reps]),
        "fatigue_drop_percent": full_rep_metrics["fatigue_drop_percent"],
        "highest_peak_activation": full_rep_metrics["highest_peak_activation"],
        "has_calibration": calibration is not None,
        "normalized_average_activation": full_rep_metrics.get(
            "normalized_average_activation",
            0,
        ),
        "normalized_peak_activation": full_rep_metrics.get(
            "normalized_peak_activation",
            0,
        ),
    }


def higher_set_label(set_a, set_b, metric_name):
    value_a = set_a[metric_name]
    value_b = set_b[metric_name]

    if value_a > value_b:
        return "Set 1"

    if value_b > value_a:
        return "Set 2"

    return "Tie"


def fatigue_comparison_label(set_1, set_2):
    fatigue_1 = set_1["fatigue_drop_percent"]
    fatigue_2 = set_2["fatigue_drop_percent"]
    meaningful_1 = fatigue_1 > CONSISTENT_FATIGUE_THRESHOLD
    meaningful_2 = fatigue_2 > CONSISTENT_FATIGUE_THRESHOLD

    if not meaningful_1 and not meaningful_2:
        return "Tie"

    if meaningful_1 and not meaningful_2:
        return "Set 1"

    if meaningful_2 and not meaningful_1:
        return "Set 2"

    if fatigue_1 > fatigue_2:
        return "Set 1"

    if fatigue_2 > fatigue_1:
        return "Set 2"

    return "Tie"


def activation_trend_direction(change_percent):
    if abs(change_percent) <= CONSISTENT_FATIGUE_THRESHOLD:
        return "consistent"

    if change_percent > 0:
        return "decreased"

    return "increased"


def trend_winner_label(change_1, change_2):
    if abs(change_1) > abs(change_2):
        return "Set 1"

    if abs(change_2) > abs(change_1):
        return "Set 2"

    return "Tie"


def opposite_trend_line(direction_1, direction_2):
    return (
        "Set 1 activation "
        f"{direction_1}, while Set 2 activation {direction_2}."
    )


def one_consistent_trend_line(change_1, change_2, direction_1, direction_2):
    if direction_1 == "consistent":
        return (
            "Set 1 activation stayed relatively consistent, while "
            f"Set 2 activation {direction_2} by {abs(change_2):.1f}%."
        )

    return (
        f"Set 1 activation {direction_1} by {abs(change_1):.1f}%, while "
        "Set 2 activation stayed relatively consistent."
    )


def same_direction_trend_line(change_1, change_2, direction):
    difference = percentage_point_difference(abs(change_1), abs(change_2))
    winner = trend_winner_label(change_1, change_2)

    if direction == "decreased":
        if winner == "Tie":
            return "Both sets decreased in activation by the same amount."

        if difference >= MEANINGFUL_FATIGUE_POINT_DIFFERENCE:
            return f"Both sets decreased in activation. {winner} showed the larger decrease across the set."

        return f"Both sets decreased in activation. {winner} had a slightly larger decrease."

    if winner == "Tie":
        return "Both sets increased in activation by the same amount."

    if difference >= MEANINGFUL_FATIGUE_POINT_DIFFERENCE:
        return f"Both sets increased in activation. {winner} had the larger increase."

    return f"Both sets increased in activation. {winner} had a slightly larger increase."


def activation_trend_comparison_line(change_1, change_2):
    direction_1 = activation_trend_direction(change_1)
    direction_2 = activation_trend_direction(change_2)

    if direction_1 == "consistent" and direction_2 == "consistent":
        return "Both sets stayed relatively consistent from the first half to the second half."

    if direction_1 == "consistent" or direction_2 == "consistent":
        return one_consistent_trend_line(
            change_1,
            change_2,
            direction_1,
            direction_2,
        )

    if direction_1 != direction_2:
        return opposite_trend_line(direction_1, direction_2)

    return same_direction_trend_line(change_1, change_2, direction_1)


def comparison_summary(set_1, set_2):
    activation_1 = set_1["average_rep_activation"]
    activation_2 = set_2["average_rep_activation"]
    peak_1 = set_1["highest_peak_activation"]
    peak_2 = set_2["highest_peak_activation"]
    activation_change_1 = set_1["fatigue_drop_percent"]
    activation_change_2 = set_2["fatigue_drop_percent"]

    if activation_1 >= activation_2:
        higher_activation_set = "Set 1"
        activation_percent_difference = percent_difference(activation_1, activation_2)
    else:
        higher_activation_set = "Set 2"
        activation_percent_difference = percent_difference(activation_2, activation_1)

    comparison = {
        "higher_average_activation_set": (
            "Tie" if activation_1 == activation_2 else higher_activation_set
        ),
        "activation_percent_difference": activation_percent_difference,
        "higher_peak_activation_set": higher_set_label(
            set_1,
            set_2,
            "highest_peak_activation",
        ),
        "more_fatigue_drop_set": fatigue_comparison_label(set_1, set_2),
        "activation_trend_comparison": activation_trend_comparison_line(
            activation_change_1,
            activation_change_2,
        ),
        "peak_activation_percent_difference": percent_difference(
            max(peak_1, peak_2),
            min(peak_1, peak_2),
        ),
        "fatigue_drop_percentage_point_difference": percentage_point_difference(
            activation_change_1,
            activation_change_2,
        ),
    }

    if set_1["has_calibration"] and set_2["has_calibration"]:
        normalized_activation_1 = set_1["normalized_average_activation"]
        normalized_activation_2 = set_2["normalized_average_activation"]
        normalized_peak_1 = set_1["normalized_peak_activation"]
        normalized_peak_2 = set_2["normalized_peak_activation"]

        if normalized_activation_1 > normalized_activation_2:
            normalized_activation_winner = "Set 1"
        elif normalized_activation_2 > normalized_activation_1:
            normalized_activation_winner = "Set 2"
        else:
            normalized_activation_winner = "Tie"

        if normalized_peak_1 > normalized_peak_2:
            normalized_peak_winner = "Set 1"
        elif normalized_peak_2 > normalized_peak_1:
            normalized_peak_winner = "Set 2"
        else:
            normalized_peak_winner = "Tie"

        comparison.update({
            "has_calibration": True,
            "higher_normalized_activation_set": normalized_activation_winner,
            "normalized_activation_percent_difference": percent_difference(
                max(normalized_activation_1, normalized_activation_2),
                min(normalized_activation_1, normalized_activation_2),
            ),
            "higher_normalized_peak_set": normalized_peak_winner,
            "normalized_peak_percent_difference": percent_difference(
                max(normalized_peak_1, normalized_peak_2),
                min(normalized_peak_1, normalized_peak_2),
            ),
        })
    else:
        comparison["has_calibration"] = False

    return comparison


def matched_rep_comparison(set_1, set_2, calibration=None):
    shared_rep_count = min(set_1["rep_count"], set_2["rep_count"])

    if set_1["rep_count"] == set_2["rep_count"]:
        return None

    set_1_metrics = rep_metrics(set_1["reps"][:shared_rep_count], calibration)
    set_2_metrics = rep_metrics(set_2["reps"][:shared_rep_count], calibration)

    return {
        "shared_rep_count": shared_rep_count,
        "set_1": set_1_metrics,
        "set_2": set_2_metrics,
        "average_rep_activation_percent_difference": percent_difference(
            max(
                set_1_metrics["average_rep_activation"],
                set_2_metrics["average_rep_activation"],
            ),
            min(
                set_1_metrics["average_rep_activation"],
                set_2_metrics["average_rep_activation"],
            ),
        ),
        "average_rep_peak_percent_difference": percent_difference(
            max(set_1_metrics["average_rep_peak"], set_2_metrics["average_rep_peak"]),
            min(set_1_metrics["average_rep_peak"], set_2_metrics["average_rep_peak"]),
        ),
        "fatigue_drop_percentage_point_difference": percentage_point_difference(
            set_1_metrics["fatigue_drop_percent"],
            set_2_metrics["fatigue_drop_percent"],
        ),
    }


def set_lines(label, set_summary):
    lines = [
        label,
        f"File name: {set_summary['file'].name}",
    ]
    metadata = set_summary["metadata"]
    set_metadata_lines = metadata_lines(metadata)

    if set_metadata_lines:
        lines.extend([f"  {line}" for line in set_metadata_lines[1:]])

    lines.extend([
        f"Rep count: {set_summary['rep_count']}",
        f"Set duration: {set_summary['set_duration']:.2f}s",
        f"Overall average signal: {set_summary['overall_average_signal']:.1f}",
        f"Overall peak signal: {set_summary['overall_peak_signal']:.1f}",
        f"Average rep peak: {set_summary['average_rep_peak']:.1f}",
        f"Average rep activation: {set_summary['average_rep_activation']:.1f}",
        "Activation change from first half to second half: "
        f"{activation_change_phrase(set_summary['fatigue_drop_percent'])}",
        f"Highest peak activation: {set_summary['highest_peak_activation']:.1f}",
    ])

    if set_summary["has_calibration"]:
        lines.extend([
            "Average normalized rep activation: "
            f"{set_summary['normalized_average_activation']:.1f}%",
            "Peak normalized rep activation: "
            f"{set_summary['normalized_peak_activation']:.1f}%",
        ])

    return lines


def comparison_lines(comparison):
    lines = [
        "Full-Set Comparison",
        "Higher average rep activation: "
        f"{comparison['higher_average_activation_set']}",
        "Percent difference in average rep activation: "
        f"{comparison['activation_percent_difference']:.1f}%",
        f"Higher peak activation: {comparison['higher_peak_activation_set']}",
        "Percent difference in peak activation: "
        f"{comparison['peak_activation_percent_difference']:.1f}%",
        "Activation trend comparison: "
        f"{comparison['activation_trend_comparison']}",
        "Activation-change difference: "
        f"{comparison['fatigue_drop_percentage_point_difference']:.1f} "
        "percentage points",
    ]

    if comparison["has_calibration"]:
        lines.extend([
            "Normalized Comparison",
            "Higher average normalized rep activation: "
            f"{comparison['higher_normalized_activation_set']}",
            "Percent difference in average normalized rep activation: "
            f"{comparison['normalized_activation_percent_difference']:.1f}%",
            "Higher peak normalized rep activation: "
            f"{comparison['higher_normalized_peak_set']}",
            "Percent difference in peak normalized rep activation: "
            f"{comparison['normalized_peak_percent_difference']:.1f}%",
        ])

    return lines


def matched_comparison_lines(matched_comparison):
    if matched_comparison is None:
        return []

    set_1 = matched_comparison["set_1"]
    set_2 = matched_comparison["set_2"]

    return [
        "Matched Rep Comparison",
        "Shared rep count: "
        f"{matched_comparison['shared_rep_count']}",
        f"Set 1 matched average rep activation: "
        f"{set_1['average_rep_activation']:.1f}",
        f"Set 2 matched average rep activation: "
        f"{set_2['average_rep_activation']:.1f}",
        "Matched average rep activation percent difference: "
        f"{matched_comparison['average_rep_activation_percent_difference']:.1f}%",
        f"Set 1 matched average rep peak: {set_1['average_rep_peak']:.1f}",
        f"Set 2 matched average rep peak: {set_2['average_rep_peak']:.1f}",
        "Matched average rep peak percent difference: "
        f"{matched_comparison['average_rep_peak_percent_difference']:.1f}%",
        "Set 1 matched activation change: "
        f"{activation_change_phrase(set_1['fatigue_drop_percent'])}",
        "Set 2 matched activation change: "
        f"{activation_change_phrase(set_2['fatigue_drop_percent'])}",
        "Matched activation-change difference: "
        f"{matched_comparison['fatigue_drop_percentage_point_difference']:.1f} "
        "percentage points",
    ]


def activation_insight(prefix, winner, difference):
    if difference < MEANINGFUL_PERCENT_DIFFERENCE or winner == "Tie":
        return f"{prefix}average activation was essentially the same between sets."

    other_set = "Set 1" if winner == "Set 2" else "Set 2"
    return (
        f"{prefix}{winner} had noticeably higher average activation than "
        f"{other_set}."
    )


def peak_insight(winner, difference):
    if difference < MEANINGFUL_PERCENT_DIFFERENCE or winner == "Tie":
        return "Peak activation was similar between sets."

    return f"{winner} reached a noticeably higher peak activation."


def fatigue_insight(comparison):
    return comparison["activation_trend_comparison"]


def matched_activation_winner(matched_set_1, matched_set_2):
    activation_1 = matched_set_1["average_rep_activation"]
    activation_2 = matched_set_2["average_rep_activation"]

    if activation_1 > activation_2:
        return "Set 1"

    if activation_2 > activation_1:
        return "Set 2"

    return "Tie"


def user_insight_lines(set_1, set_2, comparison, matched_comparison):
    insights = ["User Insights"]

    if comparison["has_calibration"]:
        insights.append(
            activation_insight(
                "Using calibration and active reps, ",
                comparison["higher_normalized_activation_set"],
                comparison["normalized_activation_percent_difference"],
            )
        )
    else:
        activation_winner = comparison["higher_average_activation_set"]
        activation_difference = comparison["activation_percent_difference"]

        insights.append(
            activation_insight(
                "Across the full sets, ",
                activation_winner,
                activation_difference,
            )
        )

    insights.append(fatigue_insight(comparison))

    if set_1["rep_count"] == set_2["rep_count"]:
        insights.append("Both sets had the same rep count.")
    elif set_1["rep_count"] > set_2["rep_count"]:
        insights.append("Set 1 had more reps than Set 2.")
    else:
        insights.append("Set 2 had more reps than Set 1.")

    if matched_comparison is not None:
        shared_rep_count = matched_comparison["shared_rep_count"]
        insights.append(
            "Because the sets had different rep counts, the matched comparison "
            f"used the first {shared_rep_count} reps from each set."
        )

        matched_set_1 = matched_comparison["set_1"]
        matched_set_2 = matched_comparison["set_2"]
        matched_activation_difference = matched_comparison[
            "average_rep_activation_percent_difference"
        ]
        matched_winner = matched_activation_winner(matched_set_1, matched_set_2)

        insights.append(
            activation_insight(
                "For the matched reps, ",
                matched_winner,
                matched_activation_difference,
            )
        )

    insights.append(
        peak_insight(
            comparison.get("higher_normalized_peak_set", comparison["higher_peak_activation_set"]),
            comparison.get("normalized_peak_percent_difference", comparison["peak_activation_percent_difference"]),
        )
    )

    return insights


def report_lines(set_1, set_2, comparison, matched_comparison):
    lines = ["Latest Set Comparison", ""]
    lines.extend(set_lines("Set 1", set_1))
    lines.append("")
    lines.extend(set_lines("Set 2", set_2))
    lines.append("")
    lines.extend(comparison_lines(comparison))
    matched_lines = matched_comparison_lines(matched_comparison)

    if matched_lines:
        lines.append("")
        lines.extend(matched_lines)

    lines.append("")
    lines.extend(user_insight_lines(set_1, set_2, comparison, matched_comparison))
    return lines


def save_comparison(lines):
    SUMMARIES_DIR.mkdir(exist_ok=True)

    with open(COMPARISON_SUMMARY_FILE, "w", newline="") as file:
        file.write("\n".join(lines))
        file.write("\n")

    print(f"Saved comparison to: {COMPARISON_SUMMARY_FILE.resolve()}")


def save_graph(set_1, set_2):
    GRAPHS_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(
        set_1["relative_times"],
        set_1["smoothed_values"],
        linewidth=2,
        label=f"Set 1: {set_1['file'].name}",
    )
    plt.plot(
        set_2["relative_times"],
        set_2["smoothed_values"],
        linewidth=2,
        label=f"Set 2: {set_2['file'].name}",
    )

    plt.title("Latest Set Comparison")
    plt.xlabel("Time from start (seconds)")
    plt.ylabel("Smoothed Signal Value")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(COMPARISON_GRAPH_FILE, dpi=150)
    plt.close()

    print(f"Saved graph to: {COMPARISON_GRAPH_FILE.resolve()}")


def main():
    set_files = automatic_comparison_csv_files(DATA_DIR)
    calibration = load_latest_calibration()
    set_1 = analyze_set(set_files[0], calibration)
    set_2 = analyze_set(set_files[1], calibration)
    comparison = comparison_summary(set_1, set_2)
    matched_comparison = matched_rep_comparison(set_1, set_2, calibration)
    lines = report_lines(set_1, set_2, comparison, matched_comparison)

    for line in lines:
        print(line)

    save_comparison(lines)
    save_graph(set_1, set_2)


if __name__ == "__main__":
    main()
