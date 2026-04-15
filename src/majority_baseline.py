from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from .load_data import load_dataset
from .utils import ensure_dir




LABEL_COLUMN = "manipulation_type"
TEXT_COLUMN = "text"


def validate_dataframe(df: pd.DataFrame) -> str:

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Missing required label column: {LABEL_COLUMN}")

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Missing required text column: {TEXT_COLUMN}")

    return TEXT_COLUMN


def run_majority_baseline(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:

    text_column = validate_dataframe(df)

    clean_df = df.dropna(subset=[LABEL_COLUMN]).copy()

    clean_df[LABEL_COLUMN] = clean_df[LABEL_COLUMN].astype(str)

    label_counts = clean_df[LABEL_COLUMN].value_counts()

    if label_counts.empty:

        raise ValueError("No examples remain after dropping missing labels.")

    train_df, test_df = train_test_split(
        clean_df,
        test_size=0.2,
        random_state=42,
        stratify=clean_df[LABEL_COLUMN],
    )

    y_train = train_df[LABEL_COLUMN]
    y_test = test_df[LABEL_COLUMN]

    train_counts = y_train.value_counts()

    majority_class = str(train_counts.idxmax())

    majority_class_count = int(train_counts.loc[majority_class])

    majority_class_proportion = majority_class_count / len(train_df)

    predicted_labels = [majority_class] * len(test_df)

    metrics = {
        "model": "majority_class_baseline",
        "data_source": "load_dataset()",
        "target_column": LABEL_COLUMN,
        "text_column": text_column,
        "random_seed": 42,
        "test_size_fraction": 0.2,
        "total_examples": int(len(clean_df)),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "number_of_classes": int(clean_df[LABEL_COLUMN].nunique()),
        "majority_class": majority_class,
        "majority_class_training_count": majority_class_count,
        "majority_class_training_proportion": float(majority_class_proportion),
        "accuracy": float(accuracy_score(y_test, predicted_labels)),
        "macro_f1": float(
            f1_score(y_test, predicted_labels, average="macro", zero_division=0)
        ),
        "label_distribution_full_data": {
            str(label): int(count) for label, count in label_counts.items()
        },
    }

    prediction_data = {
        "true_label": y_test.to_list(),
        "predicted_label": predicted_labels,
    }

    if "conversation_id" in test_df.columns:
        prediction_data = {
            "conversation_id": test_df["conversation_id"].to_list(),
            **prediction_data,
        }

    predictions = pd.DataFrame(prediction_data)

    return metrics, predictions


def save_metrics(
    metrics: dict,
    output_path: str | Path = Path("results/metrics/majority_baseline.json"),
) -> Path:
    path = Path(output_path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def save_predictions(predictions: pd.DataFrame, output_path: str | Path = Path("results/predictions/majority_baseline_predictions.csv"),) -> Path:
    path = Path(output_path)
    ensure_dir(path.parent)
    predictions.to_csv(path, index=False)

    return path


def progress_report_paragraph(metrics: dict) -> str:
    
    return (
        "For the initial processing baseline, we used an 80/20 stratified "
        "train/test split with a fixed random seed. The majority-class baseline "
        f"always predicted the most frequent training label, "
        f"{metrics['majority_class']!r}. On the test set, this baseline reached "
        f"{metrics['accuracy']:.4f} accuracy and {metrics['macro_f1']:.4f} macro F1. "
        "These results provide the minimum benchmark for later text-only models "
        "such as TF-IDF with Naive Bayes and Logistic Regression."
    )


def print_summary(metrics: dict) -> None:
    print("Majority Baseline Summary")
    print(f"Total examples: {metrics['total_examples']}")
    print(f"Train size / test size: {metrics['train_size']} / {metrics['test_size']}")
    print(f"Number of classes: {metrics['number_of_classes']}")
    print(f"Majority class: {metrics['majority_class']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")


def main() -> None:
    df = load_dataset()
    metrics, predictions = run_majority_baseline(df)

    metrics_path = save_metrics(metrics)
    predictions_path = save_predictions(predictions)

    print_summary(metrics)
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved predictions to: {predictions_path}")
    print()
    print(progress_report_paragraph(metrics))


if __name__ == "__main__":
    main()
