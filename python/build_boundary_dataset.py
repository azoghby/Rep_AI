import argparse
import csv
import hashlib
import os
import sys
import tempfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from dataset_builder import (  # noqa: E402
    eligibility_for_candidate_export,
    list_dataset_sessions,
    resolve_repo_path,
)
from detect_reps import (  # noqa: E402
    SMOOTHING_WINDOW,
    detect_reps_hybrid,
    detector_threshold_values,
    estimate_cycle_duration,
    index_at_or_after,
    local_minima_indices,
    moving_average,
    read_signal,
    segment_contraction_evidence,
)


DEFAULT_OUTPUT = BASE_DIR / "datasets" / "boundary_candidates.csv"


def human_boundaries(verified_intervals):
    """Return human transition regions for candidate valleys.

    A hybrid candidate_valley is the local EMG minimum inside a broad active
    detector region being evaluated as a possible split between two adjacent
    contraction lobes. The human target is therefore the transition region
    between consecutive verified reps. If annotations leave a relaxation gap,
    any candidate valley inside that gap is geometrically compatible with the
    human boundary. If reps touch under continuous tension, the region
    collapses to the shared endpoint.
    """
    boundaries = []

    for index in range(len(verified_intervals) - 1):
        previous_interval = verified_intervals[index]
        next_interval = verified_intervals[index + 1]
        previous_end = float(previous_interval["end_time"])
        next_start = float(next_interval["start_time"])
        timestamp = (previous_end + next_start) / 2
        boundaries.append({
            "boundary_index": index + 1,
            "timestamp": timestamp,
            "start_time": previous_end,
            "end_time": next_start,
            "previous_rep_number": previous_interval.get("rep_number", index + 1),
            "next_rep_number": next_interval.get("rep_number", index + 2),
            "confidence": boundary_confidence(previous_interval, next_interval),
        })

    return boundaries


def boundary_confidence(previous_interval, next_interval):
    previous_confidence = previous_interval.get("confidence", "")
    next_confidence = next_interval.get("confidence", "")

    if previous_confidence and previous_confidence == next_confidence:
        return previous_confidence

    return previous_confidence or next_confidence or ""


def candidate_boundary_error(candidate_time, boundary):
    start_time = float(boundary.get("start_time", boundary["timestamp"]))
    end_time = float(boundary.get("end_time", boundary["timestamp"]))

    if start_time > end_time:
        start_time, end_time = end_time, start_time

    if candidate_time < start_time:
        return start_time - candidate_time
    if candidate_time > end_time:
        return candidate_time - end_time

    return 0.0


def match_candidate_boundaries(candidate_times, boundaries, tolerance):
    """Deterministically match candidates to human boundaries one-to-one.

    Candidate-boundary pairs are sorted by distance to the annotated transition
    region first, then by distance to the midpoint, candidate time, boundary
    time, and original indices. This greedily assigns the closest available
    pair while preventing duplicate positives for the same human boundary.
    """
    candidate_pairs = []

    for candidate_index, candidate_time in enumerate(candidate_times):
        for boundary_index, boundary in enumerate(boundaries):
            error = candidate_boundary_error(candidate_time, boundary)
            midpoint_error = abs(candidate_time - boundary["timestamp"])

            if error <= tolerance:
                candidate_pairs.append((
                    error,
                    midpoint_error,
                    candidate_time,
                    boundary["timestamp"],
                    candidate_index,
                    boundary_index,
                ))

    candidate_pairs.sort()
    matched_candidates = set()
    matched_boundaries = set()
    matches = {}

    for error, _, _, _, candidate_index, boundary_index in candidate_pairs:
        if candidate_index in matched_candidates or boundary_index in matched_boundaries:
            continue

        matched_candidates.add(candidate_index)
        matched_boundaries.add(boundary_index)
        boundary = boundaries[boundary_index]
        matches[candidate_index] = {
            "boundary": boundary,
            "error": error,
        }

    return matches


def timestamp_alignment_warning(candidate_times, boundaries, tolerance):
    if not candidate_times or not boundaries:
        return ""

    matches = match_candidate_boundaries(candidate_times, boundaries, tolerance)

    if matches:
        return ""

    boundary_times = [boundary["timestamp"] for boundary in boundaries]
    max_boundary_time = max(boundary_times)
    max_candidate_time = max(candidate_times)

    if max_boundary_time <= 0 or max_candidate_time <= 0:
        return ""

    ratio = max_candidate_time / max_boundary_time

    if ratio > 100 or ratio < 0.01:
        return (
            "candidate and human boundary timestamps may use different units "
            f"(max_candidate={max_candidate_time:.3f}, max_boundary={max_boundary_time:.3f})"
        )

    return ""


def thresholds(csv_file):
    times, values = read_signal(csv_file)
    smoothed_values = moving_average(values, SMOOTHING_WINDOW)
    detector_thresholds = detector_threshold_values(times, smoothed_values, values)
    start_threshold = detector_thresholds["start_threshold"]
    end_threshold = detector_thresholds["end_threshold"]
    return times, values, smoothed_values, start_threshold, end_threshold


def broad_region_for_candidate(times, diagnostics, candidate):
    candidate_time = candidate["time"]

    for region in diagnostics.get("broad_regions", []):
        if region["start_time"] <= candidate_time <= region["end_time"]:
            return (
                index_at_or_after(times, region["start_time"]),
                index_at_or_after(times, region["end_time"]),
            )

    return 0, len(times) - 1


def feature_extensions(times, smoothed_values, start_threshold, diagnostics, candidate):
    start_index, end_index = broad_region_for_candidate(times, diagnostics, candidate)
    candidate_index = candidate["index"]
    left_start = start_index
    left_end = max(start_index, candidate_index)
    right_start = min(end_index, candidate_index)
    right_end = end_index
    left_evidence = segment_contraction_evidence(
        times,
        smoothed_values,
        left_start,
        left_end,
        start_threshold,
    )
    right_evidence = segment_contraction_evidence(
        times,
        smoothed_values,
        right_start,
        right_end,
        start_threshold,
    )
    candidate_nodes = local_minima_indices(smoothed_values, start_index, end_index)
    cycle_duration = estimate_cycle_duration(
        times,
        smoothed_values,
        start_index,
        end_index,
        candidate_nodes,
    ) if len(candidate_nodes) >= 2 else 0

    return {
        "left_segment_duration": times[left_end] - times[left_start],
        "right_segment_duration": times[right_end] - times[right_start],
        "plateau_support": left_evidence["plateau_duration"] + right_evidence["plateau_duration"],
        "high_activation_area": left_evidence["area_above_high"] + right_evidence["area_above_high"],
        "local_cycle_duration_estimate": cycle_duration,
    }


def candidate_rows_from_diagnostics(manifest, annotations, diagnostics, times, smoothed_values, start_threshold, tolerance):
    boundaries = human_boundaries(annotations.get("verified_rep_intervals", []))
    candidates = diagnostics.get("candidate_valleys", [])
    matches = match_candidate_boundaries(
        [candidate["time"] for candidate in candidates],
        boundaries,
        tolerance,
    )
    metadata = manifest.get("exercise_metadata", {})
    rows = []

    for candidate_index, candidate in enumerate(candidates):
        match = matches.get(candidate_index)
        label = "true_boundary" if match is not None else "false_boundary"
        extras = feature_extensions(times, smoothed_values, start_threshold, diagnostics, candidate)
        matched_boundary = match["boundary"] if match else None
        rows.append({
            "session_id": manifest["session_id"],
            "recording_filename": Path(manifest["recording_csv"]).name,
            "participant_id": manifest.get("participant_id", ""),
            "exercise": metadata.get("exercise", ""),
            "side": metadata.get("side", ""),
            "weight": metadata.get("weight", ""),
            "candidate_timestamp": candidate["time"],
            "valley_depth": candidate["adjacent_drop"],
            "normalized_valley_depth": candidate["normalized_adjacent_drop"],
            "rebound_strength": candidate["rebound_height"],
            "valley_duration": candidate["valley_duration"],
            "adjacent_contraction_center_gap": candidate["center_gap"],
            "left_segment_duration": extras["left_segment_duration"],
            "right_segment_duration": extras["right_segment_duration"],
            "plateau_support": extras["plateau_support"],
            "high_activation_area": extras["high_activation_area"],
            "local_cycle_duration_estimate": extras["local_cycle_duration_estimate"],
            "candidate_score": candidate["score"],
            "hybrid_status": "accepted" if candidate["accepted"] else "rejected",
            "human_label": label,
            "matched_human_boundary_timestamp": matched_boundary["timestamp"] if matched_boundary else "",
            "matching_error": match["error"] if match else "",
            "matched_boundary_index": matched_boundary["boundary_index"] if matched_boundary else "",
            "annotation_confidence": (
                matched_boundary.get("confidence")
                or annotations.get("confidence")
                or ""
            ) if matched_boundary else "",
        })

    return rows


def candidate_rows_for_session(manifest, tolerance, annotations=None):
    recording_csv = resolve_repo_path(manifest["recording_csv"])

    if annotations is None:
        eligible, reason, annotations = eligibility_for_candidate_export(manifest)

        if not eligible:
            raise ValueError(f"Session {manifest['session_id']} is not eligible: {reason}")

    times, values, smoothed_values, start_threshold, end_threshold = thresholds(recording_csv)
    _, diagnostics = detect_reps_hybrid(
        times,
        values,
        smoothed_values,
        start_threshold,
        end_threshold,
    )
    boundaries = human_boundaries(annotations.get("verified_rep_intervals", []))
    candidates = diagnostics.get("candidate_valleys", [])
    warning = timestamp_alignment_warning(
        [candidate["time"] for candidate in candidates],
        boundaries,
        tolerance,
    )

    if warning:
        print(f"Warning for {manifest['session_id']}: {warning}")

    return candidate_rows_from_diagnostics(
        manifest,
        annotations,
        diagnostics,
        times,
        smoothed_values,
        start_threshold,
        tolerance,
    )


def fieldnames():
    return [
        "session_id",
        "recording_filename",
        "participant_id",
        "exercise",
        "side",
        "weight",
        "candidate_timestamp",
        "valley_depth",
        "normalized_valley_depth",
        "rebound_strength",
        "valley_duration",
        "adjacent_contraction_center_gap",
        "left_segment_duration",
        "right_segment_duration",
        "plateau_support",
        "high_activation_area",
        "local_cycle_duration_estimate",
        "candidate_score",
        "hybrid_status",
        "human_label",
        "matched_human_boundary_timestamp",
        "matching_error",
        "matched_boundary_index",
        "annotation_confidence",
    ]


def resolved_source_csv_paths(manifests):
    paths = set()

    for manifest in manifests:
        for key in ("recording_csv", "calibration_csv"):
            value = manifest.get(key)

            if value:
                paths.add(resolve_repo_path(value).resolve())

    return paths


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def source_hashes(paths):
    return {
        path: sha256_file(path)
        for path in paths
        if path.exists()
    }


def ensure_output_is_not_source(output_file, source_paths):
    output_path = output_file.resolve()

    if output_path in source_paths:
        raise ValueError(f"Output path may not overwrite source CSV: {output_path}")


def verify_source_hashes_unchanged(before_hashes):
    after_hashes = source_hashes(before_hashes)

    if after_hashes != before_hashes:
        changed = [
            str(path)
            for path, before_hash in before_hashes.items()
            if after_hashes.get(path) != before_hash
        ]
        raise RuntimeError("Source CSV hashes changed during export: " + ", ".join(changed))


def write_rows(rows, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_file.name}.",
        suffix=".tmp",
        dir=output_file.parent,
        text=True,
    )

    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames())
            writer.writeheader()
            writer.writerows(rows)

        Path(temp_name).replace(output_file)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def build_dataset(manifests, output_file, tolerance):
    source_paths = resolved_source_csv_paths(manifests)
    ensure_output_is_not_source(output_file, source_paths)
    before_hashes = source_hashes(source_paths)
    rows = []
    skipped = []

    for manifest in manifests:
        eligible, reason, annotations = eligibility_for_candidate_export(manifest)

        if not eligible:
            skipped.append({
                "session_id": manifest.get("session_id", ""),
                "reason": reason,
            })
            continue

        rows.extend(candidate_rows_for_session(manifest, tolerance, annotations=annotations))

    write_rows(rows, output_file)
    verify_source_hashes_unchanged(before_hashes)
    return rows, skipped


def write_skipped_report(skipped, output_file):
    skipped_file = output_file.with_suffix(output_file.suffix + ".skipped.csv")

    with open(skipped_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["session_id", "reason"])
        writer.writeheader()
        writer.writerows(skipped)

    return skipped_file


def main():
    parser = argparse.ArgumentParser(description="Build a candidate-boundary dataset CSV.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.25,
        help="Seconds allowed between a candidate valley and a human boundary.",
    )
    args = parser.parse_args()

    output_file = Path(args.output)
    rows, skipped = build_dataset(list_dataset_sessions(), output_file, args.tolerance)
    print(f"Saved {len(rows)} candidate rows to {output_file.resolve()}")

    if skipped:
        skipped_file = write_skipped_report(skipped, output_file)
        print(f"Skipped {len(skipped)} ineligible sessions. Reasons written to {skipped_file.resolve()}")
        for item in skipped:
            print(f"  {item['session_id']}: {item['reason']}")


if __name__ == "__main__":
    main()
