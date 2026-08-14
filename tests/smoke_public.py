import os
import sys
import tempfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
PYTHON_DIR = BASE_DIR / "python"
APP_DIR = BASE_DIR / "app"
MPLCONFIGDIR = Path(tempfile.gettempdir()) / "repai-public-matplotlib"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from acquisition import ReplaySignalSource  # noqa: E402
from detect_reps import analyze_csv_file  # noqa: E402


SYNTHETIC_CSV = BASE_DIR / "examples" / "synthetic" / "synthetic_bicep_curl_6_reps.csv"
EXPECTED_REPS = 6


def assert_replay_loads():
    source = ReplaySignalSource(SYNTHETIC_CSV)
    source.connect()
    try:
        readings = []
        while True:
            reading = source.read()
            if reading is None:
                break
            readings.append(reading)
    finally:
        source.disconnect()

    if len(readings) < 20:
        raise AssertionError(f"Expected at least 20 replay readings, got {len(readings)}")


def assert_analysis_smoke():
    legacy = analyze_csv_file(SYNTHETIC_CSV, show_plot=False, method="legacy")
    hybrid = analyze_csv_file(SYNTHETIC_CSV, show_plot=False, method="hybrid")

    if legacy["summary"]["total_reps"] != EXPECTED_REPS:
        raise AssertionError(f"Legacy detector found {legacy['summary']['total_reps']} reps")

    if hybrid["summary"]["total_reps"] != EXPECTED_REPS:
        raise AssertionError(f"Hybrid detector found {hybrid['summary']['total_reps']} reps")


def main():
    assert_replay_loads()
    assert_analysis_smoke()
    print("Public smoke test passed: synthetic replay loaded and both detectors found 6 reps.")


if __name__ == "__main__":
    main()
