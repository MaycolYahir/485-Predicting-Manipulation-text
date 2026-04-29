from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    from .load_data import load_dataset
    from .utils import LABEL_COLUMN, RANDOM_STATE, TEST_SIZE, TEXT_COLUMN, ensure_dir, split_for_modeling
except ImportError:
    from load_data import load_dataset
    from utils import LABEL_COLUMN, RANDOM_STATE, TEST_SIZE, TEXT_COLUMN, ensure_dir, split_for_modeling


def validate_dataframe(df: pd.DataFrame) -> str:
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Missing required label column: {LABEL_COLUMN}")

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Missing required text column: {TEXT_COLUMN}")

    return TEXT_COLUMN


def _is_binary_problem(y_true: pd.Series) -> bool:
    return sorted(y_true.astype(str).unique().tolist()) == ["manipulative", "non_manipulative"]


def _confusion_matrix_payload(y_true: pd.Series, y_pred: list[str]) -> dict:
    labels = sorted(set(y_true.astype(str).tolist()) | set(map(str, y_pred)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "labels": labels,
        "matrix": matrix.tolist(),
    }


def _classification_report_table(y_true: pd.Series, y_pred: list[str]) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for label, scores in report.items():
        if not isinstance(scores, dict) or label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        rows.append(
            {
                "class": label,
                "precision": float(scores["precision"]),
                "recall": float(scores["recall"]),
                "f1_score": float(scores["f1-score"]),
                "support": int(scores["support"]),
            }
        )

    return pd.DataFrame(rows)


def run_majority_baseline(
    df: pd.DataFrame,
    stage_name: str = "stage1_binary",
    rare_label_min_count: int | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    text_column = validate_dataframe(df)

    train_df, test_df, split_metadata = split_for_modeling(
        df,
        label_column=LABEL_COLUMN,
        text_column=text_column,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        rare_label_min_count=rare_label_min_count,
    )

    y_train = train_df[LABEL_COLUMN]
    y_test = test_df[LABEL_COLUMN]

    majority_class = str(y_train.value_counts().idxmax())
    predicted_labels = [majority_class] * len(test_df)
    report_df = _classification_report_table(y_test, predicted_labels)

    metrics = {
        "model": "majority_class_baseline",
        "stage": stage_name,
        "data_source": "MentalManip with exact text deduplication before split",
        "target_column": LABEL_COLUMN,
        "text_column": text_column,
        "random_seed": RANDOM_STATE,
        "test_size_fraction": TEST_SIZE,
        **split_metadata,
        "number_of_classes": int(pd.concat([y_train, y_test]).nunique()),
        "majority_class": majority_class,
        "majority_class_training_count": int(y_train.value_counts().loc[majority_class]),
        "majority_class_training_proportion": float((y_train == majority_class).mean()),
        "accuracy": float(accuracy_score(y_test, predicted_labels)),
        "macro_f1": float(f1_score(y_test, predicted_labels, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_test, predicted_labels, average="weighted", zero_division=0)
        ),
        "label_distribution_full_data": {
            str(label): int(count)
            for label, count in pd.concat([y_train, y_test]).value_counts().items()
        },
        "confusion_matrix": _confusion_matrix_payload(y_test, predicted_labels),
        "class_metrics": report_df.set_index("class").to_dict(orient="index"),
    }

    if _is_binary_problem(y_test):
        metrics["precision"] = float(
            precision_score(y_test, predicted_labels, pos_label="manipulative", zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(y_test, predicted_labels, pos_label="manipulative", zero_division=0)
        )
        metrics["f1"] = float(
            f1_score(y_test, predicted_labels, pos_label="manipulative", zero_division=0)
        )

    predictions = pd.DataFrame(
        {
            "true_label": y_test.to_list(),
            "predicted_label": predicted_labels,
        }
    )

    return metrics, predictions, report_df


def save_metrics(
    metrics: dict,
    output_path: str | Path = Path("results/metrics/majority_baseline.json"),
) -> Path:
    path = Path(output_path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def save_predictions(
    predictions: pd.DataFrame,
    output_path: str | Path = Path("results/predictions/majority_baseline_predictions.csv"),
) -> Path:
    path = Path(output_path)
    ensure_dir(path.parent)
    predictions.to_csv(path, index=False)
    return path


def print_summary(metrics: dict) -> None:
    print("Majority Baseline Summary")
    print(f"Stage: {metrics['stage']}")
    print(f"Total examples: {metrics['total_examples_after_cleaning']}")
    print(f"Train size / test size: {metrics['train_size']} / {metrics['test_size']}")
    print(f"Number of classes: {metrics['number_of_classes']}")
    print(f"Majority class: {metrics['majority_class']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    if "precision" in metrics:
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")


def main() -> None:
    df = load_dataset(task="binary")
    metrics, predictions, report_df = run_majority_baseline(df, stage_name="stage1_binary")

    metrics_path = save_metrics(metrics)
    predictions_path = save_predictions(predictions)

    print_summary(metrics)
    print("\nClass-wise metrics:")
    print(report_df.to_string(index=False))
    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved predictions to: {predictions_path}")


if __name__ == "__main__":
    main()
