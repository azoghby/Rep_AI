from datetime import datetime
from pathlib import Path

from recording_metadata import load_metadata, metadata_path_for_csv, save_metadata


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

ALLOWED_DATA_TYPES = ("fake", "real")
ALLOWED_TEST_TYPES = ("calibration", "flex_test", "workout_set")


def ask(prompt):
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("\nMetadata editing cancelled. No changes saved.")


def yes_no(prompt, default="n"):
    default = default.lower()
    suffix = " [y/N]: " if default == "n" else " [Y/n]: "

    while True:
        answer = ask(prompt + suffix).strip().lower()

        if answer == "":
            return default == "y"

        if answer in ("y", "yes"):
            return True

        if answer in ("n", "no"):
            return False

        print("Please enter y or n.")


def prompt_with_default(prompt, default=""):
    if default:
        answer = ask(f"{prompt} [{default}]: ").strip()
    else:
        answer = ask(f"{prompt}: ").strip()

    return default if answer == "" else answer


def prompt_choice(prompt, allowed_values, default=""):
    allowed_text = "/".join(allowed_values)

    while True:
        value = prompt_with_default(f"{prompt} ({allowed_text})", default).strip().lower()

        if value in allowed_values:
            return value

        print(f"Please enter one of: {allowed_text}")


def valid_weight(value):
    if value == "":
        return True

    if value.upper() == "N/A":
        return True

    try:
        float(value)
        return True
    except ValueError:
        return False


def prompt_weight(default=""):
    while True:
        value = prompt_with_default("Weight (blank, numeric, or N/A)", default).strip()

        if valid_weight(value):
            return "N/A" if value.upper() == "N/A" else value

        print("Weight must be blank, numeric, or N/A. It cannot be values like 'fake'.")


def show_metadata(csv_file, metadata):
    print(f"{csv_file.name}")

    if not metadata:
        print("  Metadata: missing")
        return

    for key in (
        "exercise_name",
        "muscle",
        "side",
        "weight",
        "expected_reps",
        "data_type",
        "test_type",
        "notes",
    ):
        print(f"  {key}: {metadata.get(key, '')}")


def csv_files():
    files = sorted(DATA_DIR.glob("*.csv"), key=lambda path: path.name)

    if not files:
        raise FileNotFoundError("No CSV files found in data/.")

    return files


def list_csv_files(files):
    print("CSV files in data/:")

    for index, csv_file in enumerate(files, start=1):
        metadata_file = metadata_path_for_csv(csv_file)
        metadata = load_metadata(csv_file)
        status = "has metadata" if metadata_file.exists() else "missing metadata"
        print(f"{index}. {csv_file.name} ({status})")

        if metadata:
            print(
                "   "
                f"exercise={metadata.get('exercise_name', '')}, "
                f"muscle={metadata.get('muscle', '')}, "
                f"side={metadata.get('side', '')}, "
                f"weight={metadata.get('weight', '')}, "
                f"data_type={metadata.get('data_type', '')}, "
                f"test_type={metadata.get('test_type', '')}"
            )


def choose_csv_file(files):
    while True:
        choice = ask("File number to edit: ").strip()

        try:
            index = int(choice) - 1
        except ValueError:
            print("Please enter a number.")
            continue

        if 0 <= index < len(files):
            return files[index]

        print("Invalid file number.")


def prompt_metadata(csv_file, existing_metadata=None):
    existing_metadata = existing_metadata or {}
    exercise_name = prompt_with_default(
        "Exercise name",
        existing_metadata.get("exercise_name", csv_file.stem),
    )
    muscle = prompt_with_default("Muscle", existing_metadata.get("muscle", ""))
    side = prompt_with_default("Side (right/left)", existing_metadata.get("side", ""))
    weight = prompt_weight(existing_metadata.get("weight", ""))
    expected_reps = prompt_with_default(
        "Expected reps (blank if unknown)",
        existing_metadata.get("expected_reps", ""),
    )
    data_type = prompt_choice(
        "Data type",
        ALLOWED_DATA_TYPES,
        existing_metadata.get("data_type", "fake"),
    )
    test_type = prompt_choice(
        "Test type",
        ALLOWED_TEST_TYPES,
        existing_metadata.get("test_type", ""),
    )
    notes = prompt_with_default("Notes", existing_metadata.get("notes", ""))

    metadata = {
        "csv_filename": csv_file.name,
        "generated_at": existing_metadata.get(
            "generated_at",
            datetime.now().isoformat(timespec="seconds"),
        ),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "exercise_name": exercise_name,
        "muscle": muscle,
        "side": side,
        "weight": weight,
        "expected_reps": expected_reps,
        "data_type": data_type,
        "test_type": test_type,
        "notes": notes,
    }

    if "timestamp" in existing_metadata:
        metadata["timestamp"] = existing_metadata["timestamp"]

    return metadata


def create_missing_metadata(files):
    for csv_file in files:
        metadata_file = metadata_path_for_csv(csv_file)

        if metadata_file.exists():
            continue

        if not yes_no(f"Add metadata for {csv_file.name}?", default="y"):
            continue

        print(f"Entering metadata for {csv_file.name}")
        metadata = prompt_metadata(csv_file)
        saved_path = save_metadata(csv_file, metadata)
        print(f"Saved metadata to: {saved_path.resolve()}")
        print()


def edit_existing_metadata(files):
    list_csv_files(files)
    print()
    csv_file = choose_csv_file(files)
    existing_metadata = load_metadata(csv_file)

    print()
    print("Current metadata:")
    show_metadata(csv_file, existing_metadata)
    print()

    if existing_metadata and not yes_no("Edit this metadata?", default="y"):
        return

    if not existing_metadata:
        print("No existing metadata found. Creating new metadata.")

    metadata = prompt_metadata(csv_file, existing_metadata)
    saved_path = save_metadata(csv_file, metadata)
    print(f"Saved metadata to: {saved_path.resolve()}")


def main():
    files = csv_files()
    list_csv_files(files)
    print()
    print("Modes:")
    print("1. Create metadata for missing files")
    print("2. Edit metadata for a selected file")
    mode = ask("Choose mode [1]: ").strip()

    if mode == "":
        mode = "1"

    if mode == "1":
        create_missing_metadata(files)
    elif mode == "2":
        edit_existing_metadata(files)
    else:
        raise ValueError("Invalid mode. Choose 1 or 2.")


if __name__ == "__main__":
    main()
