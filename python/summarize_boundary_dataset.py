import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from dataset_builder import DATASETS_DIR, eligibility_for_candidate_export, list_dataset_sessions, load_annotations  # noqa: E402


DEFAULT_CANDIDATES = BASE_DIR / "datasets" / "boundary_candidates.csv"
REQUIRED_CANDIDATE_COLUMNS = {
    "session_id",
    "participant_id",
    "exercise",
    "side",
    "weight",
    "human_label",
}


def load_candidate_rows(candidate_csv):
    if not candidate_csv.exists():
        return [], False

    with open(candidate_csv, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_CANDIDATE_COLUMNS - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                "Candidate CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )

        return list(reader), True


def print_counter(title, counter):
    print(title)

    if not counter:
        print("  none")
        return

    for key, count in sorted(counter.items(), key=lambda item: str(item[0])):
        print(f"  {key or 'unknown'}: {count}")


def summarize_dataset(candidate_csv):
    sessions = list_dataset_sessions()
    annotations_by_session = {
        session["session_id"]: load_annotations(session["session_id"])
        for session in sessions
    }
    rows, candidate_csv_exists = load_candidate_rows(Path(candidate_csv))
    participants = {session.get("participant_id", "") for session in sessions if session.get("participant_id")}
    recordings = {session.get("recording_csv", "") for session in sessions if session.get("recording_csv")}
    total_verified_reps = sum(
        len(annotation.get("verified_rep_intervals", []))
        for annotation in annotations_by_session.values()
    )
    labels = Counter(row.get("human_label", "") for row in rows)
    total_candidates = sum(labels.values())
    reviewed_statuses = Counter(
        annotation.get("annotation_status", "unreviewed")
        for annotation in annotations_by_session.values()
    )
    skipped_reasons = Counter()

    for session in sessions:
        eligible, reason, _ = eligibility_for_candidate_export(session)

        if not eligible:
            skipped_reasons[reason] += 1

    return {
        "sessions": sessions,
        "rows": rows,
        "candidate_csv_exists": candidate_csv_exists,
        "participants": participants,
        "recordings": recordings,
        "total_verified_reps": total_verified_reps,
        "labels": labels,
        "total_candidates": total_candidates,
        "reviewed_statuses": reviewed_statuses,
        "skipped_reasons": skipped_reasons,
    }


def print_summary(summary, candidate_csv):
    rows = summary["rows"]
    labels = summary["labels"]
    total_candidates = summary["total_candidates"]

    print("Boundary Dataset Summary")
    print(f"Dataset sessions directory: {DATASETS_DIR.resolve()}")
    print(f"Candidate CSV: {Path(candidate_csv).resolve()}")

    if not summary["candidate_csv_exists"]:
        print("Candidate CSV status: missing")

    print(f"Number of sessions: {len(summary['sessions'])}")
    print(f"Number of participants: {len(summary['participants'])}")
    print(f"Number of recordings: {len(summary['recordings'])}")
    print(f"Total verified reps: {summary['total_verified_reps']}")
    print(f"True candidates: {labels.get('true_boundary', 0)}")
    print(f"False candidates: {labels.get('false_boundary', 0)}")

    if total_candidates:
        true_fraction = labels.get("true_boundary", 0) / total_candidates * 100
        false_fraction = labels.get("false_boundary", 0) / total_candidates * 100
        print(f"Class balance: true {true_fraction:.1f}% / false {false_fraction:.1f}%")
    else:
        print("Class balance: no candidate rows")

    print_counter("Counts by exercise", Counter(row.get("exercise", "") for row in rows))
    print_counter("Counts by side", Counter(row.get("side", "") for row in rows))
    print_counter("Counts by weight", Counter(row.get("weight", "") for row in rows))
    print_counter("Counts by session", Counter(row.get("session_id", "") for row in rows))
    print_counter("Reviewed versus unreviewed annotations", summary["reviewed_statuses"])
    print_counter("Skipped sessions by reason", summary["skipped_reasons"])


def main():
    parser = argparse.ArgumentParser(description="Summarize the boundary candidate dataset.")
    parser.add_argument(
        "--candidate-csv",
        default=str(DEFAULT_CANDIDATES),
        help="Candidate dataset CSV from build_boundary_dataset.py.",
    )
    args = parser.parse_args()

    summary = summarize_dataset(Path(args.candidate_csv))
    print_summary(summary, Path(args.candidate_csv))


if __name__ == "__main__":
    main()
