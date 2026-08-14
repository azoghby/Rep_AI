import json


METADATA_FIELDS = [
    "exercise_name",
    "muscle",
    "side",
    "weight",
    "expected_reps",
    "data_type",
    "test_type",
    "notes",
]


def metadata_path_for_csv(csv_file):
    return csv_file.with_name(f"{csv_file.stem}_metadata.json")


def load_metadata(csv_file):
    metadata_file = metadata_path_for_csv(csv_file)

    if not metadata_file.exists():
        return {}

    with open(metadata_file, "r", encoding="utf-8") as file:
        return json.load(file)


def save_metadata(csv_file, metadata):
    metadata_file = metadata_path_for_csv(csv_file)

    with open(metadata_file, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    return metadata_file


def metadata_lines(metadata):
    if not metadata:
        return []

    labels = {
        "csv_filename": "CSV filename",
        "timestamp": "Timestamp",
        "exercise_name": "Exercise",
        "muscle": "Muscle",
        "side": "Side",
        "weight": "Weight",
        "expected_reps": "Expected reps",
        "calibration_setup_id": "Calibration setup ID",
        "data_type": "Data type",
        "test_type": "Test type",
        "notes": "Notes",
    }

    lines = ["Metadata"]

    for key, label in labels.items():
        value = metadata.get(key, "")

        if value != "":
            lines.append(f"{label}: {value}")

    return lines
