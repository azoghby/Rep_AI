import argparse
from pathlib import Path

from boundary_classifier import load_candidate_rows, run_experiment, write_json_report


DEFAULT_CANDIDATES = Path(__file__).resolve().parent.parent / "datasets" / "boundary_candidates.csv"


def print_metric(name, value):
    if isinstance(value, float):
        print(f"{name}: {value:.3f}")
    else:
        print(f"{name}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a candidate-boundary classifier experiment from a labeled candidate CSV."
    )
    parser.add_argument(
        "--candidate-csv",
        default=str(DEFAULT_CANDIDATES),
        help="Labeled candidate dataset CSV from build_boundary_dataset.py; raw recordings are not accepted.",
    )
    parser.add_argument(
        "--model",
        choices=["logistic", "hybrid-rule", "gradient_boosting"],
        default="logistic",
        help="Model or baseline to evaluate.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["session", "participant"],
        default="session",
        help="Grouped split mode. Row-level splitting is intentionally unsupported.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Probability threshold for classifying a candidate as a true boundary.",
    )
    parser.add_argument(
        "--class-weight",
        choices=["balanced"],
        default=None,
        help="Optional sklearn LogisticRegression class_weight. Applies to --model logistic.",
    )
    parser.add_argument(
        "--report-output",
        default="",
        help="Optional JSON report path. Prefer an untracked artifacts/ path.",
    )
    args = parser.parse_args()

    rows = load_candidate_rows(Path(args.candidate_csv))
    result = run_experiment(
        rows,
        model_name=args.model,
        split_mode=args.split_mode,
        test_fraction=args.test_fraction,
        seed=args.seed,
        decision_threshold=args.decision_threshold,
        class_weight=args.class_weight,
    )
    metrics = result["candidate_metrics"]

    print(f"Model: {result['model']}")
    print(f"Split: {result['split_mode']} ({result['split_group_key']})")
    print(f"Decision threshold: {result['decision_threshold']:.3f}")
    print(f"Train candidates: {result['train_candidate_count']}")
    print(f"Test candidates: {result['test_candidate_count']}")
    print(f"Train sessions: {', '.join(result['train_session_ids'])}")
    print(f"Test sessions: {', '.join(result['test_session_ids'])}")

    for name in (
        "precision",
        "recall",
        "f1",
        "accuracy",
        "pr_auc",
        "roc_auc",
        "false_splits",
        "missed_boundaries",
    ):
        print_metric(name, metrics[name])

    session_aggregate = result["session_metrics"]["classifier_aggregate"]
    print_metric(
        "mean_absolute_boundary_count_error",
        session_aggregate["mean_absolute_boundary_count_error"],
    )
    print_metric("exact_count_accuracy", session_aggregate["exact_count_accuracy"])

    if args.report_output:
        output_path = write_json_report(result, args.report_output)
        print(f"Report: {output_path.resolve()}")


if __name__ == "__main__":
    main()
