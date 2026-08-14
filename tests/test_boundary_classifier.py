import csv
import json

import pytest

from boundary_classifier import (
    FEATURE_COLUMNS,
    FORBIDDEN_FEATURE_COLUMNS,
    GradientBoostingBoundaryClassifier,
    LogisticBoundaryClassifier,
    assert_feature_label_separation,
    evaluate_predictions,
    feature_matrix,
    hybrid_rule_predictions,
    load_candidate_rows,
    row_keys,
    run_experiment,
    session_metrics,
    threshold_predictions,
    train_classifier,
    write_json_report,
)


def candidate_row(session_id, participant_id, index, label, hybrid_status=None, scale=1.0):
    positive = label == "true_boundary"
    row = {
        "session_id": session_id,
        "recording_filename": f"{session_id}.csv",
        "participant_id": participant_id,
        "exercise": "curl",
        "side": "right",
        "weight": "10",
        "candidate_timestamp": float(index),
        "valley_depth": scale * (8.0 if positive else 1.0),
        "normalized_valley_depth": 0.9 if positive else 0.1,
        "rebound_strength": scale * (8.0 if positive else 1.0),
        "valley_duration": 0.3 if positive else 0.1,
        "adjacent_contraction_center_gap": 2.0 if positive else 0.5,
        "left_segment_duration": 1.0,
        "right_segment_duration": 1.0,
        "plateau_support": 1.2 if positive else 0.1,
        "high_activation_area": scale * (10.0 if positive else 1.0),
        "local_cycle_duration_estimate": 2.0 if positive else 0.5,
        "candidate_score": scale * (9.0 if positive else 0.5),
        "hybrid_status": hybrid_status or ("accepted" if positive else "rejected"),
        "human_label": label,
        "matched_human_boundary_timestamp": float(index) if positive else "",
        "matching_error": 0.0 if positive else "",
        "matched_boundary_index": index if positive else "",
        "annotation_confidence": "high" if positive else "",
    }
    return {key: str(value) for key, value in row.items()}


def synthetic_rows():
    return [
        candidate_row("s1", "p1", 1, "true_boundary"),
        candidate_row("s1", "p1", 2, "false_boundary"),
        candidate_row("s2", "p1", 1, "true_boundary"),
        candidate_row("s2", "p1", 2, "false_boundary"),
        candidate_row("s3", "p2", 1, "true_boundary"),
        candidate_row("s3", "p2", 2, "false_boundary"),
        candidate_row("s4", "p2", 1, "true_boundary"),
        candidate_row("s4", "p2", 2, "false_boundary"),
    ]


def test_sklearn_logistic_pipeline_trains_successfully():
    model = LogisticBoundaryClassifier(random_state=42).fit(synthetic_rows())

    assert model.pipeline.named_steps["scaler"].mean_ is not None
    assert set(model.coefficients()) == set(FEATURE_COLUMNS)
    assert model.predict(synthetic_rows()[:2]) == [1, 0]


def test_sklearn_gradient_boosting_trains_successfully():
    model = GradientBoostingBoundaryClassifier(random_state=42).fit(synthetic_rows())

    assert set(model.feature_importance()) == set(FEATURE_COLUMNS)
    assert model.predict(synthetic_rows()[:2]) == [1, 0]


def test_scaler_is_fit_only_on_training_rows():
    train_rows = synthetic_rows()[:6]
    test_rows = [
        candidate_row("s4", "p2", 100, "true_boundary", scale=1000.0),
        candidate_row("s4", "p2", 200, "false_boundary", scale=1000.0),
    ]
    model = LogisticBoundaryClassifier(random_state=42).fit(train_rows)
    first_feature = FEATURE_COLUMNS[0]
    train_first_feature_mean = sum(float(row[first_feature]) for row in train_rows) / len(train_rows)

    assert model.pipeline.named_steps["scaler"].mean_[0] == pytest.approx(train_first_feature_mean)
    assert model.pipeline.named_steps["scaler"].mean_[0] != pytest.approx(
        sum(float(row[first_feature]) for row in train_rows + test_rows) / (len(train_rows) + len(test_rows))
    )


def test_target_leakage_fields_are_excluded():
    assert "human_label" in FORBIDDEN_FEATURE_COLUMNS
    assert "matched_human_boundary_timestamp" in FORBIDDEN_FEATURE_COLUMNS
    assert "matching_error" in FORBIDDEN_FEATURE_COLUMNS
    assert "annotation_confidence" in FORBIDDEN_FEATURE_COLUMNS
    assert "session_id" in FORBIDDEN_FEATURE_COLUMNS
    assert "participant_id" in FORBIDDEN_FEATURE_COLUMNS
    assert "candidate_timestamp" in FORBIDDEN_FEATURE_COLUMNS

    with pytest.raises(ValueError, match="labels or metadata"):
        assert_feature_label_separation(["candidate_score", "matching_error"])


def test_candidate_timestamp_is_row_identity_not_predictive_feature():
    rows = [candidate_row("s1", "p1", 12345, "true_boundary")]
    matrix = feature_matrix(rows)

    assert "candidate_timestamp" not in FEATURE_COLUMNS
    assert len(matrix[0]) == len(FEATURE_COLUMNS)
    assert float(rows[0]["candidate_timestamp"]) not in matrix[0]

    with pytest.raises(ValueError, match="labels or metadata"):
        feature_matrix(rows, ["candidate_timestamp"])


def test_unknown_new_columns_cannot_silently_become_features():
    rows = [candidate_row("s1", "p1", 1, "true_boundary")]
    rows[0]["future_magic_column"] = "999"

    assert "future_magic_column" not in FEATURE_COLUMNS
    assert len(feature_matrix(rows)[0]) == len(FEATURE_COLUMNS)

    with pytest.raises(ValueError, match="explicitly whitelisted"):
        feature_matrix(rows, ["candidate_score", "future_magic_column"])


def test_hybrid_status_is_excluded_by_default():
    assert "hybrid_status" in FORBIDDEN_FEATURE_COLUMNS
    assert "hybrid_status" not in FEATURE_COLUMNS

    with pytest.raises(ValueError, match="labels or metadata"):
        feature_matrix(synthetic_rows(), ["candidate_score", "hybrid_status"])


def test_decision_threshold_changes_predicted_labels():
    probabilities = [0.2, 0.55, 0.8]

    assert threshold_predictions(probabilities, 0.5) == [0, 1, 1]
    assert threshold_predictions(probabilities, 0.7) == [0, 0, 1]


def test_threshold_is_recorded_in_report():
    result = run_experiment(
        synthetic_rows(),
        model_name="logistic",
        split_mode="session",
        seed=4,
        decision_threshold=0.7,
    )

    assert result["decision_threshold"] == 0.7


def test_model_and_hybrid_baseline_use_same_test_rows():
    result = run_experiment(synthetic_rows(), model_name="logistic", split_mode="session", seed=4)

    assert result["test_row_keys"] == result["hybrid_test_row_keys"]
    assert len(result["predictions"]) == len(result["hybrid_predictions"]) == result["test_candidate_count"]


def test_duplicate_candidate_timestamps_keep_distinct_report_row_keys():
    rows = [
        candidate_row("s1", "p1", 1, "true_boundary"),
        candidate_row("s1", "p1", 1, "false_boundary"),
    ]

    keys = row_keys(rows)

    assert keys[0]["candidate_timestamp"] == keys[1]["candidate_timestamp"]
    assert keys[0]["test_row_index"] != keys[1]["test_row_index"]


def test_candidate_metrics_report_confusion_accuracy_and_auc():
    rows = [
        candidate_row("s1", "p1", 1, "true_boundary"),
        candidate_row("s1", "p1", 2, "false_boundary"),
        candidate_row("s2", "p1", 1, "true_boundary"),
        candidate_row("s2", "p1", 2, "false_boundary"),
    ]
    metrics = evaluate_predictions(rows, [1, 1, 0, 0], probabilities=[0.9, 0.8, 0.4, 0.3])

    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"] == {
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_positive": 1,
    }
    assert metrics["pr_auc"] is not None
    assert metrics["roc_auc"] is not None


def test_session_level_count_metrics_are_correct():
    rows = [
        candidate_row("s1", "p1", 1, "true_boundary"),
        candidate_row("s1", "p1", 2, "false_boundary"),
        candidate_row("s2", "p1", 1, "true_boundary"),
        candidate_row("s2", "p1", 2, "false_boundary"),
    ]
    metrics = session_metrics(
        rows,
        classifier_predictions=[1, 1, 0, 0],
        hybrid_predictions=[1, 0, 1, 1],
    )

    assert metrics["per_session"] == [
        {
            "session_id": "s1",
            "human_boundary_count": 1,
            "classifier_predicted_boundary_count": 2,
            "hybrid_accepted_boundary_count": 1,
            "classifier_false_positives": 1,
            "classifier_misses": 0,
            "hybrid_false_positives": 0,
            "hybrid_misses": 0,
            "classifier_signed_count_error": 1,
            "hybrid_signed_count_error": 0,
            "classifier_absolute_count_error": 1,
            "hybrid_absolute_count_error": 0,
        },
        {
            "session_id": "s2",
            "human_boundary_count": 1,
            "classifier_predicted_boundary_count": 0,
            "hybrid_accepted_boundary_count": 2,
            "classifier_false_positives": 0,
            "classifier_misses": 1,
            "hybrid_false_positives": 1,
            "hybrid_misses": 0,
            "classifier_signed_count_error": -1,
            "hybrid_signed_count_error": 1,
            "classifier_absolute_count_error": 1,
            "hybrid_absolute_count_error": 1,
        },
    ]
    assert metrics["classifier_aggregate"]["mean_absolute_boundary_count_error"] == pytest.approx(1.0)
    assert metrics["classifier_aggregate"]["exact_count_accuracy"] == pytest.approx(0.0)
    assert metrics["classifier_aggregate"]["oversplit_session_count"] == 1
    assert metrics["classifier_aggregate"]["undersplit_session_count"] == 1


def test_json_report_serializes_cleanly(tmp_path):
    result = run_experiment(synthetic_rows(), model_name="gradient_boosting", split_mode="session", seed=4)
    output = write_json_report(result, tmp_path / "report.json")

    with open(output, "r", encoding="utf-8") as file:
        loaded = json.load(file)

    assert loaded["model"] == "gradient_boosting"
    assert loaded["feature_names"] == FEATURE_COLUMNS


def test_one_class_auc_case_is_handled():
    rows = [
        candidate_row("s1", "p1", 1, "true_boundary"),
        candidate_row("s1", "p1", 2, "true_boundary"),
    ]
    metrics = evaluate_predictions(rows, [1, 0], probabilities=[0.9, 0.1])

    assert metrics["pr_auc"] is None
    assert metrics["roc_auc"] is None


def test_session_split_has_zero_session_leakage():
    result = run_experiment(synthetic_rows(), model_name="logistic", split_mode="session", seed=4)

    assert set(result["train_session_ids"]).isdisjoint(result["test_session_ids"])


def test_participant_split_has_zero_participant_leakage():
    result = run_experiment(synthetic_rows(), model_name="hybrid-rule", split_mode="participant", seed=1)

    assert set(result["train_participant_ids"]).isdisjoint(result["test_participant_ids"])


def test_participant_split_fails_clearly_with_one_participant():
    rows = [
        candidate_row("s1", "p1", 1, "true_boundary"),
        candidate_row("s2", "p1", 2, "false_boundary"),
    ]

    with pytest.raises(ValueError, match="at least two usable participants"):
        run_experiment(rows, model_name="hybrid-rule", split_mode="participant", seed=1)


def test_fixed_seed_is_deterministic():
    first = run_experiment(synthetic_rows(), model_name="logistic", split_mode="session", seed=4)
    second = run_experiment(synthetic_rows(), model_name="logistic", split_mode="session", seed=4)

    assert first["test_session_ids"] == second["test_session_ids"]
    assert first["predictions"] == second["predictions"]
    assert first["probabilities"] == pytest.approx(second["probabilities"])


def test_hybrid_rule_baseline_and_metrics_report_boundary_errors():
    rows = [
        candidate_row("s1", "p1", 1, "true_boundary", hybrid_status="accepted"),
        candidate_row("s1", "p1", 2, "false_boundary", hybrid_status="accepted"),
        candidate_row("s2", "p1", 1, "true_boundary", hybrid_status="rejected"),
        candidate_row("s2", "p1", 2, "false_boundary", hybrid_status="rejected"),
    ]
    predictions = hybrid_rule_predictions(rows)
    metrics = evaluate_predictions(rows, predictions)

    assert predictions == [1, 1, 0, 0]
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["false_splits"] == 1
    assert metrics["missed_boundaries"] == 1


def test_gradient_boosting_train_classifier_entrypoint():
    model = train_classifier("gradient_boosting", synthetic_rows(), seed=42)

    assert isinstance(model, GradientBoostingBoundaryClassifier)


def test_missing_numeric_values_fail_clearly():
    rows = synthetic_rows()
    rows[0]["candidate_score"] = ""

    with pytest.raises(ValueError, match="missing numeric feature"):
        feature_matrix(rows)


def test_one_class_training_data_fails_clearly():
    rows = [
        candidate_row("s1", "p1", 1, "true_boundary"),
        candidate_row("s2", "p1", 2, "true_boundary"),
    ]

    with pytest.raises(ValueError, match="requires both true and false"):
        LogisticBoundaryClassifier().fit(rows)


def test_candidate_csv_loader_round_trips_rows(tmp_path):
    rows = synthetic_rows()[:2]
    path = tmp_path / "candidates.csv"

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    assert load_candidate_rows(path) == rows
