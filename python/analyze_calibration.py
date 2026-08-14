from datetime import datetime
from pathlib import Path

from calibration_utils import (
    build_latest_calibration_for_csv,
    calibration_summary_lines,
    save_latest_calibration,
)
from recording_metadata import load_metadata


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SUMMARIES_DIR = BASE_DIR / "summaries"
CALIBRATION_SUMMARY_FILE = SUMMARIES_DIR / "latest_calibration_summary.txt"


def calibration_csvs_from_metadata():
    calibration_files = []

    for csv_file in DATA_DIR.glob("*.csv"):
        metadata = load_metadata(csv_file)

        if metadata.get("test_type") == "calibration":
            calibration_files.append(csv_file)

    return sorted(calibration_files, key=lambda path: path.stat().st_mtime)


def choose_csv_file():
    csv_files = sorted(DATA_DIR.glob("*.csv"), key=lambda path: path.stat().st_mtime)

    if not csv_files:
        raise FileNotFoundError("No CSV files found in data/.")

    print("No metadata-marked calibration CSV found.")
    print("Choose a CSV to use as calibration:")

    for index, csv_file in enumerate(csv_files, start=1):
        print(f"{index}. {csv_file.name}")

    choice = input("CSV number: ").strip()
    selected_index = int(choice) - 1

    if selected_index < 0 or selected_index >= len(csv_files):
        raise ValueError("Invalid CSV selection.")

    return csv_files[selected_index]


def latest_calibration_csv():
    calibration_files = calibration_csvs_from_metadata()

    if calibration_files:
        return calibration_files[-1]

    return choose_csv_file()


def main():
    csv_file = latest_calibration_csv()
    calibration = build_latest_calibration_for_csv(
        csv_file,
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )

    calibration_file = save_latest_calibration(calibration)
    SUMMARIES_DIR.mkdir(exist_ok=True)
    CALIBRATION_SUMMARY_FILE.write_text(
        "\n".join(calibration_summary_lines(calibration)) + "\n",
        encoding="utf-8",
    )

    for line in calibration_summary_lines(calibration):
        print(line)

    print(f"Saved calibration JSON to: {calibration_file.resolve()}")
    print(f"Saved calibration summary to: {CALIBRATION_SUMMARY_FILE.resolve()}")


if __name__ == "__main__":
    main()
