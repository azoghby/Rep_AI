import random
from collections import defaultdict


def group_rows(rows, group_key):
    groups = defaultdict(list)

    for row in rows:
        group = row.get(group_key)

        if not group:
            raise ValueError(f"Candidate row is missing required split group: {group_key}")

        groups[group].append(row)

    return groups


def split_candidate_rows(rows, mode="session", test_fraction=0.2, seed=0):
    if mode not in ("session", "participant"):
        raise ValueError("Split mode must be 'session' or 'participant'.")

    group_key = "session_id" if mode == "session" else "participant_id"
    groups = group_rows(rows, group_key)

    if mode == "participant" and len(groups) < 2:
        raise ValueError("Participant-level split requires at least two usable participants.")

    if len(groups) < 2:
        raise ValueError(f"{mode.title()}-level split requires at least two usable groups.")

    group_ids = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    test_count = max(1, min(len(group_ids) - 1, round(len(group_ids) * test_fraction)))
    test_groups = set(group_ids[:test_count])
    train_groups = set(group_ids[test_count:])

    partitions = {
        "train": [
            row
            for group_id in sorted(train_groups)
            for row in groups[group_id]
        ],
        "test": [
            row
            for group_id in sorted(test_groups)
            for row in groups[group_id]
        ],
    }
    validate_no_leakage(partitions, group_key)
    return partitions


def validate_no_leakage(partitions, group_key="session_id"):
    group_sets = {}

    for partition_name, rows in partitions.items():
        group_sets[partition_name] = {
            row.get(group_key)
            for row in rows
            if row.get(group_key)
        }

    partition_names = sorted(group_sets)

    for index, left_name in enumerate(partition_names):
        for right_name in partition_names[index + 1:]:
            overlap = group_sets[left_name] & group_sets[right_name]

            if overlap:
                raise ValueError(
                    f"Split leakage detected between {left_name} and {right_name}: "
                    + ", ".join(sorted(overlap))
                )

    return True
