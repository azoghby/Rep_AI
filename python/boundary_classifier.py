import csv
import json
import math
import statistics
from importlib import metadata
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from boundary_dataset_splits import split_candidate_rows, validate_no_leakage


LABEL_COLUMN = "human_label"
POSITIVE_LABEL = "true_boundary"
NEGATIVE_LABEL = "false_boundary"
HYBRID_STATUS_COLUMN = "hybrid_status"
SESSION_COLUMN = "session_id"
PARTICIPANT_COLUMN = "participant_id"

FEATURE_COLUMNS = [
    "valley_depth",
    "normalized_valley_depth",
    "rebound_strength",
    "valley_duration",
    "adjacent_contraction_center_gap",
    "left_segment_duration",
    "right_segment_duration",
    "plateau_support",
    "high_activation_area",
    "local_cycle_duration_estimate",
    "candidate_score",
]

FORBIDDEN_FEATURE_COLUMNS = [
    LABEL_COLUMN,
    "matched_human_boundary_timestamp",
    "matching_error",
    "matched_boundary_index",
    "annotation_confidence",
    SESSION_COLUMN,
    PARTICIPANT_COLUMN,
    "recording_filename",
    "recording_csv",
    "source_filename",
    "source_path",
    "exercise",
    "side",
    "weight",
    "candidate_timestamp",
    HYBRID_STATUS_COLUMN,
]

EXPECTED_CANDIDATE_COLUMNS = [
    SESSION_COLUMN,
    "recording_filename",
    PARTICIPANT_COLUMN,
    "exercise",
    "side",
    "weight",
    "candidate_timestamp",
    *FEATURE_COLUMNS,
    HYBRID_STATUS_COLUMN,
    LABEL_COLUMN,
    "matched_human_boundary_timestamp",
    "matching_error",
    "matched_boundary_index",
    "annotation_confidence",
]


def load_candidate_rows(candidate_csv):
    with open(candidate_csv, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def label_to_int(label):
    if label == POSITIVE_LABEL:
        return 1
    if label == NEGATIVE_LABEL:
        return 0
    raise ValueError(f"Unsupported boundary label: {label!r}")


def numeric_value(row, column):
    value = row.get(column)

    if value in ("", None):
        raise ValueError(f"Candidate row is missing numeric feature: {column}")

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Candidate feature {column} must be numeric: {value!r}") from exc


def assert_feature_label_separation(feature_columns=None):
    columns = list(feature_columns or FEATURE_COLUMNS)
    overlap = set(columns) & set(FORBIDDEN_FEATURE_COLUMNS)

    if overlap:
        raise ValueError("Feature columns may not include labels or metadata: " + ", ".join(sorted(overlap)))

    unknown = set(columns) - set(FEATURE_COLUMNS)

    if unknown:
        raise ValueError("Feature columns must be explicitly whitelisted: " + ", ".join(sorted(unknown)))

    return True


def validate_candidate_schema(rows, feature_columns=None):
    columns = set(feature_columns or FEATURE_COLUMNS)
    assert_feature_label_separation(columns)

    if not rows:
        raise ValueError("At least one candidate row is required.")

    present_columns = set(rows[0])
    missing_required = set([LABEL_COLUMN, SESSION_COLUMN, PARTICIPANT_COLUMN, HYBRID_STATUS_COLUMN]) - present_columns
    missing_features = columns - present_columns

    if missing_required:
        raise ValueError("Candidate rows are missing required columns: " + ", ".join(sorted(missing_required)))
    if missing_features:
        raise ValueError("Candidate rows are missing feature columns: " + ", ".join(sorted(missing_features)))

    return {
        "expected_columns": list(EXPECTED_CANDIDATE_COLUMNS),
        "feature_columns": list(feature_columns or FEATURE_COLUMNS),
        "forbidden_feature_columns": list(FORBIDDEN_FEATURE_COLUMNS),
        "extra_columns": sorted(present_columns - set(EXPECTED_CANDIDATE_COLUMNS)),
    }


def feature_matrix(rows, feature_columns=None):
    columns = list(feature_columns or FEATURE_COLUMNS)
    assert_feature_label_separation(columns)
    return [
        [numeric_value(row, column) for column in columns]
        for row in rows
    ]


def labels(rows):
    return [label_to_int(row.get(LABEL_COLUMN)) for row in rows]


def normalization_stats(x_rows):
    columns = list(zip(*x_rows))
    means = [sum(column) / len(column) for column in columns]
    scales = []

    for mean, column in zip(means, columns):
        variance = sum((value - mean) ** 2 for value in column) / len(column)
        scale = math.sqrt(variance)
        scales.append(scale if scale > 0 else 1.0)

    return means, scales


def normalize_matrix(x_rows, means, scales):
    return [
        [
            (value - mean) / scale
            for value, mean, scale in zip(row, means, scales)
        ]
        for row in x_rows
    ]


class LogisticBoundaryClassifier:
    def __init__(self, class_weight=None, random_state=0, max_iter=1000):
        self.class_weight = class_weight
        self.random_state = random_state
        self.max_iter = max_iter
        self.feature_columns = list(FEATURE_COLUMNS)
        self.pipeline = None

    def fit(self, rows, feature_columns=None):
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)
        validate_training_rows(rows, self.feature_columns)
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight=self.class_weight,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )),
        ])
        self.pipeline.fit(feature_matrix(rows, self.feature_columns), labels(rows))
        return self

    def predict_proba(self, rows):
        if self.pipeline is None:
            raise ValueError("Model has not been fitted.")
        return [
            float(probability)
            for probability in self.pipeline.predict_proba(feature_matrix(rows, self.feature_columns))[:, 1]
        ]

    def predict(self, rows, threshold=0.5):
        return threshold_predictions(self.predict_proba(rows), threshold)

    def coefficients(self):
        if self.pipeline is None:
            raise ValueError("Model has not been fitted.")
        classifier = self.pipeline.named_steps["classifier"]
        return dict(zip(self.feature_columns, [float(value) for value in classifier.coef_[0]]))


class GradientBoostingBoundaryClassifier:
    def __init__(self, random_state=0):
        self.random_state = random_state
        self.feature_columns = list(FEATURE_COLUMNS)
        self.model = None

    def fit(self, rows, feature_columns=None):
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)
        validate_training_rows(rows, self.feature_columns)
        self.model = GradientBoostingClassifier(random_state=self.random_state)
        self.model.fit(feature_matrix(rows, self.feature_columns), labels(rows))
        return self

    def predict_proba(self, rows):
        if self.model is None:
            raise ValueError("Model has not been fitted.")
        return [
            float(probability)
            for probability in self.model.predict_proba(feature_matrix(rows, self.feature_columns))[:, 1]
        ]

    def predict(self, rows, threshold=0.5):
        return threshold_predictions(self.predict_proba(rows), threshold)

    def feature_importance(self):
        if self.model is None:
            raise ValueError("Model has not been fitted.")
        return dict(zip(self.feature_columns, [float(value) for value in self.model.feature_importances_]))


def validate_training_rows(rows, feature_columns=None):
    validate_candidate_schema(rows, feature_columns)
    y_values = labels(rows)

    if len(set(y_values)) < 2:
        raise ValueError("Classifier training requires both true and false boundary labels.")

    return True


def threshold_predictions(probabilities, threshold):
    return [1 if probability >= threshold else 0 for probability in probabilities]


def hybrid_rule_predictions(rows):
    return [
        1 if row.get(HYBRID_STATUS_COLUMN) == "accepted" else 0
        for row in rows
    ]


def evaluate_predictions(rows, predictions, probabilities=None):
    if len(rows) != len(predictions):
        raise ValueError("Prediction count must match candidate row count.")
    if probabilities is not None and len(rows) != len(probabilities):
        raise ValueError("Probability count must match candidate row count.")

    y_true = labels(rows)
    true_positive = sum(1 for truth, prediction in zip(y_true, predictions) if truth == 1 and prediction == 1)
    false_positive = sum(1 for truth, prediction in zip(y_true, predictions) if truth == 0 and prediction == 1)
    false_negative = sum(1 for truth, prediction in zip(y_true, predictions) if truth == 1 and prediction == 0)
    true_negative = sum(1 for truth, prediction in zip(y_true, predictions) if truth == 0 and prediction == 0)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (true_positive + true_negative) / len(rows) if rows else 0.0
    aucs = auc_metrics(y_true, probabilities)
    count_errors = per_session_rep_count_errors(rows, predictions)

    return {
        "candidate_count": len(rows),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "confusion_matrix": {
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
        "pr_auc": aucs["pr_auc"],
        "roc_auc": aucs["roc_auc"],
        "false_splits": false_positive,
        "missed_boundaries": false_negative,
        "per_session_rep_count_error": count_errors,
        "mean_absolute_rep_count_error": (
            sum(item["absolute_error"] for item in count_errors) / len(count_errors)
            if count_errors else 0.0
        ),
    }


def auc_metrics(y_true, probabilities):
    if probabilities is None or len(set(y_true)) < 2:
        return {"pr_auc": None, "roc_auc": None}

    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def session_metrics(rows, classifier_predictions, hybrid_predictions):
    if len(rows) != len(classifier_predictions) or len(rows) != len(hybrid_predictions):
        raise ValueError("Prediction counts must match candidate row count.")

    sessions = {}

    for row, classifier_prediction, hybrid_prediction in zip(rows, classifier_predictions, hybrid_predictions):
        session_id = row.get(SESSION_COLUMN)

        if not session_id:
            raise ValueError("Candidate row is missing session_id.")

        truth = label_to_int(row.get(LABEL_COLUMN))
        metrics = sessions.setdefault(session_id, {
            "session_id": session_id,
            "human_boundary_count": 0,
            "classifier_predicted_boundary_count": 0,
            "hybrid_accepted_boundary_count": 0,
            "classifier_false_positives": 0,
            "classifier_misses": 0,
            "hybrid_false_positives": 0,
            "hybrid_misses": 0,
        })

        if truth == 1:
            metrics["human_boundary_count"] += 1
        if classifier_prediction == 1:
            metrics["classifier_predicted_boundary_count"] += 1
        if hybrid_prediction == 1:
            metrics["hybrid_accepted_boundary_count"] += 1
        if truth == 0 and classifier_prediction == 1:
            metrics["classifier_false_positives"] += 1
        if truth == 1 and classifier_prediction == 0:
            metrics["classifier_misses"] += 1
        if truth == 0 and hybrid_prediction == 1:
            metrics["hybrid_false_positives"] += 1
        if truth == 1 and hybrid_prediction == 0:
            metrics["hybrid_misses"] += 1

    per_session = []

    for session_id in sorted(sessions):
        metrics = sessions[session_id]
        classifier_error = metrics["classifier_predicted_boundary_count"] - metrics["human_boundary_count"]
        hybrid_error = metrics["hybrid_accepted_boundary_count"] - metrics["human_boundary_count"]
        metrics["classifier_signed_count_error"] = classifier_error
        metrics["hybrid_signed_count_error"] = hybrid_error
        metrics["classifier_absolute_count_error"] = abs(classifier_error)
        metrics["hybrid_absolute_count_error"] = abs(hybrid_error)
        per_session.append(metrics)

    return {
        "per_session": per_session,
        "classifier_aggregate": aggregate_session_count_metrics(per_session, "classifier"),
        "hybrid_aggregate": aggregate_session_count_metrics(per_session, "hybrid"),
    }


def per_session_rep_count_errors(rows, predictions):
    hybrid_predictions = [0] * len(rows)
    metrics = session_metrics(rows, predictions, hybrid_predictions)

    return [
        {
            "session_id": session["session_id"],
            "true_boundaries": session["human_boundary_count"],
            "predicted_boundaries": session["classifier_predicted_boundary_count"],
            "signed_error": session["classifier_signed_count_error"],
            "absolute_error": session["classifier_absolute_count_error"],
        }
        for session in metrics["per_session"]
    ]


def aggregate_session_count_metrics(per_session, prefix):
    absolute_key = f"{prefix}_absolute_count_error"
    signed_key = f"{prefix}_signed_count_error"
    errors = [session[absolute_key] for session in per_session]
    signed_errors = [session[signed_key] for session in per_session]

    return {
        "mean_absolute_boundary_count_error": sum(errors) / len(errors) if errors else 0.0,
        "median_absolute_boundary_count_error": statistics.median(errors) if errors else 0.0,
        "exact_count_accuracy": (
            sum(1 for error in errors if error == 0) / len(errors)
            if errors else 0.0
        ),
        "oversplit_session_count": sum(1 for error in signed_errors if error > 0),
        "undersplit_session_count": sum(1 for error in signed_errors if error < 0),
    }


def class_balance(rows):
    y_values = labels(rows)
    positives = sum(y_values)
    negatives = len(y_values) - positives
    return {
        "candidate_count": len(rows),
        "true_boundary": positives,
        "false_boundary": negatives,
        "positive_fraction": positives / len(rows) if rows else 0.0,
    }


def group_values(rows, group_key):
    return sorted({row[group_key] for row in rows if row.get(group_key)})


def package_versions():
    versions = {}

    for package_name in ("scikit-learn", "numpy", "scipy"):
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None

    return versions


def train_classifier(model_name, train_rows, seed=0, class_weight=None):
    if model_name == "logistic":
        return LogisticBoundaryClassifier(class_weight=class_weight, random_state=seed).fit(train_rows)
    if model_name == "gradient_boosting":
        return GradientBoostingBoundaryClassifier(random_state=seed).fit(train_rows)
    raise ValueError(f"Unsupported boundary classifier model: {model_name}")


def run_experiment(
    rows,
    model_name="logistic",
    split_mode="session",
    test_fraction=0.2,
    seed=0,
    decision_threshold=0.5,
    class_weight=None,
):
    schema = validate_candidate_schema(rows)
    partitions = split_candidate_rows(
        rows,
        mode=split_mode,
        test_fraction=test_fraction,
        seed=seed,
    )
    group_key = SESSION_COLUMN if split_mode == "session" else PARTICIPANT_COLUMN
    validate_no_leakage(partitions, group_key=group_key)
    train_rows = partitions["train"]
    test_rows = partitions["test"]
    hybrid_predictions = hybrid_rule_predictions(test_rows)
    model_details = {}

    if model_name == "hybrid-rule":
        probabilities = [float(value) for value in hybrid_predictions]
        predictions = list(hybrid_predictions)
    else:
        model = train_classifier(model_name, train_rows, seed=seed, class_weight=class_weight)
        probabilities = model.predict_proba(test_rows)
        predictions = threshold_predictions(probabilities, decision_threshold)

        if model_name == "logistic":
            model_details["coefficients"] = model.coefficients()
        elif model_name == "gradient_boosting":
            model_details["feature_importance"] = model.feature_importance()

    classifier_metrics = evaluate_predictions(test_rows, predictions, probabilities)
    hybrid_metrics = evaluate_predictions(test_rows, hybrid_predictions, [float(value) for value in hybrid_predictions])
    per_session_metrics = session_metrics(test_rows, predictions, hybrid_predictions)
    train_sessions = group_values(train_rows, SESSION_COLUMN)
    test_sessions = group_values(test_rows, SESSION_COLUMN)
    train_participants = group_values(train_rows, PARTICIPANT_COLUMN)
    test_participants = group_values(test_rows, PARTICIPANT_COLUMN)

    result = {
        "model": model_name,
        "feature_names": list(FEATURE_COLUMNS),
        "forbidden_feature_columns": list(FORBIDDEN_FEATURE_COLUMNS),
        "ignored_extra_columns": schema["extra_columns"],
        "decision_threshold": decision_threshold,
        "seed": seed,
        "split_mode": split_mode,
        "split_group_key": group_key,
        "test_fraction": test_fraction,
        "train_session_ids": train_sessions,
        "test_session_ids": test_sessions,
        "train_participant_ids": train_participants,
        "test_participant_ids": test_participants,
        "train_candidate_count": len(train_rows),
        "test_candidate_count": len(test_rows),
        "train_class_balance": class_balance(train_rows),
        "test_class_balance": class_balance(test_rows),
        "candidate_metrics": classifier_metrics,
        "metrics": classifier_metrics,
        "hybrid_baseline_metrics": hybrid_metrics,
        "session_metrics": per_session_metrics,
        "package_versions": package_versions(),
        "model_details": model_details,
        "test_row_keys": row_keys(test_rows),
        "hybrid_test_row_keys": row_keys(test_rows),
        "probabilities": probabilities,
        "predictions": predictions,
        "hybrid_predictions": hybrid_predictions,
    }
    assert result["test_row_keys"] == result["hybrid_test_row_keys"]
    return result


def row_keys(rows):
    return [
        {
            "test_row_index": index,
            "session_id": row.get(SESSION_COLUMN),
            "candidate_timestamp": row.get("candidate_timestamp"),
        }
        for index, row in enumerate(rows)
    ]


def write_json_report(result, report_output):
    output_path = Path(report_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, sort_keys=True)
        file.write("\n")

    return output_path
