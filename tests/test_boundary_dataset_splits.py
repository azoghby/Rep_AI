import pytest

from boundary_dataset_splits import split_candidate_rows, validate_no_leakage


def row(session_id, participant_id, index):
    return {
        "session_id": session_id,
        "participant_id": participant_id,
        "candidate_timestamp": index,
    }


def test_session_level_split_keeps_session_rows_together_and_is_deterministic():
    rows = [
        row("s1", "p1", 1),
        row("s1", "p1", 2),
        row("s2", "p1", 3),
        row("s3", "p2", 4),
        row("s3", "p2", 5),
    ]

    first = split_candidate_rows(rows, mode="session", seed=7)
    second = split_candidate_rows(rows, mode="session", seed=7)

    assert first == second
    train_sessions = {item["session_id"] for item in first["train"]}
    test_sessions = {item["session_id"] for item in first["test"]}
    assert train_sessions.isdisjoint(test_sessions)
    assert len(first["train"]) + len(first["test"]) == len(rows)


def test_participant_level_split_keeps_all_participant_sessions_together():
    rows = [
        row("s1", "p1", 1),
        row("s2", "p1", 2),
        row("s3", "p2", 3),
        row("s4", "p3", 4),
    ]

    partitions = split_candidate_rows(rows, mode="participant", seed=3)
    train_participants = {item["participant_id"] for item in partitions["train"]}
    test_participants = {item["participant_id"] for item in partitions["test"]}

    assert train_participants.isdisjoint(test_participants)


def test_participant_split_requires_at_least_two_participants():
    rows = [row("s1", "p1", 1), row("s2", "p1", 2)]

    with pytest.raises(ValueError, match="at least two usable participants"):
        split_candidate_rows(rows, mode="participant", seed=1)


def test_no_direct_row_split_mode_is_supported():
    with pytest.raises(ValueError, match="Split mode"):
        split_candidate_rows([row("s1", "p1", 1), row("s2", "p2", 2)], mode="row")


def test_validate_no_leakage_detects_group_overlap():
    partitions = {
        "train": [row("s1", "p1", 1)],
        "test": [row("s1", "p1", 2)],
    }

    with pytest.raises(ValueError, match="Split leakage"):
        validate_no_leakage(partitions, group_key="session_id")


def test_missing_group_field_fails_clearly():
    with pytest.raises(ValueError, match="missing required split group"):
        split_candidate_rows([{"participant_id": "p1"}], mode="session")
