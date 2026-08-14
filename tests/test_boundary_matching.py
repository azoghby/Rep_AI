import pytest

from build_boundary_dataset import (
    candidate_rows_from_diagnostics,
    human_boundaries,
    match_candidate_boundaries,
    timestamp_alignment_warning,
)


def candidate(index, time, accepted=True):
    return {
        "index": index,
        "time": time,
        "value": 10,
        "adjacent_drop": 100,
        "normalized_adjacent_drop": 0.5,
        "rebound_height": 120,
        "valley_duration": 0.2,
        "center_gap": 2.0,
        "score": 100,
        "accepted": accepted,
    }


def diagnostics(candidates):
    return {
        "broad_regions": [{"start_time": 0.0, "end_time": 10.0}],
        "candidate_valleys": candidates,
    }


def annotations(intervals):
    return {
        "annotation_status": "locked",
        "actual_reps": len(intervals),
        "verified_rep_intervals": intervals,
        "excluded_false_intervals": [],
        "confidence": 0.8,
    }


def interval(rep_number, start, end, confidence="medium"):
    return {
        "rep_number": rep_number,
        "start_time": start,
        "end_time": end,
        "confidence": confidence,
        "note": "",
    }


def rows_for(intervals, candidates, tolerance=0.25):
    times = [float(index) for index in range(11)]
    smoothed = [0, 80, 10, 90, 20, 100, 20, 85, 15, 70, 0]
    manifest = {
        "session_id": "synthetic-session",
        "recording_csv": "synthetic.csv",
        "participant_id": "p1",
        "exercise_metadata": {"exercise": "curl", "side": "right", "weight": "10"},
    }
    return candidate_rows_from_diagnostics(
        manifest,
        annotations(intervals),
        diagnostics(candidates),
        times,
        smoothed,
        start_threshold=30,
        tolerance=tolerance,
    )


@pytest.mark.parametrize(
    "case_intervals, case_candidates, expected_labels",
    [
        (
            [interval(1, 0.0, 1.0), interval(2, 3.0, 4.0)],
            [candidate(2, 2.0)],
            ["true_boundary"],
        ),
        (
            [interval(1, 0.0, 2.0), interval(2, 2.0, 4.0)],
            [candidate(2, 2.0)],
            ["true_boundary"],
        ),
        (
            [interval(1, 0.0, 4.0)],
            [candidate(2, 2.0)],
            ["false_boundary"],
        ),
        (
            [interval(1, 0.0, 3.0), interval(2, 3.0, 6.0)],
            [candidate(2, 1.5), candidate(3, 3.0)],
            ["false_boundary", "true_boundary"],
        ),
        (
            [interval(1, 0.0, 5.0), interval(2, 5.0, 10.0)],
            [candidate(5, 5.0), candidate(7, 7.0)],
            ["true_boundary", "false_boundary"],
        ),
    ],
)
def test_synthetic_movement_cases_label_human_boundaries(case_intervals, case_candidates, expected_labels):
    rows = rows_for(case_intervals, case_candidates)

    assert [row["human_label"] for row in rows] == expected_labels


def test_human_boundaries_use_inter_rep_transition_midpoint_for_relaxation_gap():
    boundaries = human_boundaries([interval(1, 0.0, 1.0), interval(2, 3.0, 4.0)])

    assert boundaries[0]["timestamp"] == 2.0
    assert boundaries[0]["start_time"] == 1.0
    assert boundaries[0]["end_time"] == 3.0


def test_candidate_inside_inter_rep_transition_gap_is_true_boundary():
    rows = rows_for(
        [interval(1, 0.0, 1.0), interval(2, 3.0, 4.0)],
        [candidate(2, 2.85)],
    )

    assert [row["human_label"] for row in rows] == ["true_boundary"]
    assert rows[0]["matching_error"] == 0.0


def test_n_reps_produce_n_minus_one_human_internal_boundaries():
    intervals = [
        interval(1, 0.0, 1.0),
        interval(2, 2.0, 3.0),
        interval(3, 4.0, 5.0),
        interval(4, 6.0, 7.0),
    ]

    boundaries = human_boundaries(intervals)

    assert len(boundaries) == 3
    assert [boundary["boundary_index"] for boundary in boundaries] == [1, 2, 3]


def test_candidate_inside_transition_gap_matches_only_one_boundary():
    boundaries = human_boundaries([
        interval(1, 0.0, 1.0),
        interval(2, 3.0, 4.0),
    ])

    matches = match_candidate_boundaries([1.25, 2.75], boundaries, tolerance=0.25)

    assert len(matches) == 1
    assert next(iter(matches.values()))["error"] == 0.0


def test_candidate_outside_transition_gap_and_tolerance_remains_false():
    rows = rows_for(
        [interval(1, 0.0, 1.0), interval(2, 3.0, 4.0)],
        [candidate(2, 3.26)],
    )

    assert [row["human_label"] for row in rows] == ["false_boundary"]


def test_one_to_one_matching_keeps_closest_duplicate_candidate():
    boundaries = [{"boundary_index": 1, "timestamp": 5.0}]
    matches = match_candidate_boundaries([4.9, 5.01], boundaries, tolerance=0.25)

    assert set(matches) == {1}
    assert matches[1]["error"] == pytest.approx(0.01)


def test_tie_matching_is_deterministic_by_candidate_time_then_index():
    boundaries = [{"boundary_index": 1, "timestamp": 5.0}]
    matches = match_candidate_boundaries([5.1, 4.9], boundaries, tolerance=0.25)

    assert set(matches) == {1}


def test_unmatched_candidates_do_not_steal_from_closer_candidate():
    boundaries = [{"boundary_index": 1, "timestamp": 5.0}]
    matches = match_candidate_boundaries([5.24, 5.01], boundaries, tolerance=0.25)

    assert set(matches) == {1}


def test_tolerance_edge_is_inclusive_and_outside_is_unmatched():
    boundaries = [{"boundary_index": 1, "timestamp": 5.0, "start_time": 5.0, "end_time": 5.0}]

    edge_matches = match_candidate_boundaries([5.25], boundaries, tolerance=0.25)
    outside_matches = match_candidate_boundaries([5.251], boundaries, tolerance=0.25)

    assert set(edge_matches) == {0}
    assert outside_matches == {}


def test_timestamp_unit_mismatch_warning_flags_all_negative_candidate_scale():
    boundaries = human_boundaries([interval(1, 0.0, 1.0), interval(2, 3.0, 4.0)])

    warning = timestamp_alignment_warning([2000.0], boundaries, tolerance=0.25)

    assert "different units" in warning


def test_exported_match_metadata_is_present_for_true_only():
    rows = rows_for(
        [interval(1, 0.0, 2.0), interval(2, 2.0, 4.0)],
        [candidate(2, 2.0), candidate(4, 4.0, accepted=False)],
    )

    assert rows[0]["matched_human_boundary_timestamp"] == 2.0
    assert rows[0]["matching_error"] == 0.0
    assert rows[1]["matched_human_boundary_timestamp"] == ""
    assert rows[1]["human_label"] == "false_boundary"
    assert rows[1]["hybrid_status"] == "rejected"
