"""Compare a list of manual 0/1 predictions against the ground-truth labels
for the first 100 examples of MentalManip (binary task).
 
Convention: 1 = manipulative, 0 = non_manipulative.
The order of predictions corresponds to the row order returned by
load_dataset(task="binary"), which preserves the HuggingFace dataset order.
"""
 
from __future__ import annotations
 
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
    from .utils import LABEL_COLUMN, TEXT_COLUMN
except ImportError:
    from load_data import load_dataset
    from utils import LABEL_COLUMN, TEXT_COLUMN
 
 
MANUAL_PREDICTIONS = [
    0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
    0, 1, 1, 0, 0, 1, 1, 1, 0, 1,
    1, 1, 1, 0, 1, 0, 1, 1, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
    1, 1, 0, 0, 1, 1, 1, 0, 0, 1,
    0, 0, 1, 1, 0, 0, 1, 0, 0, 0,
    0, 0, 1, 1, 0, 0, 1, 1, 1, 1,
    1, 0, 1, 1, 0, 0, 1, 0, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 1, 1, 1, 0, 0, 1,
]
 
 
PREDICTION_TO_LABEL = {0: "non_manipulative", 1: "manipulative"}
 
 
def main() -> None:
    n = len(MANUAL_PREDICTIONS)
    df = load_dataset(task="binary").head(n).reset_index(drop=True)
    if len(df) < n:
        raise ValueError(f"Dataset has only {len(df)} rows, expected at least {n}.")
 
    predicted_labels = [PREDICTION_TO_LABEL[p] for p in MANUAL_PREDICTIONS]
    true_labels = df[LABEL_COLUMN].tolist()
 
    accuracy = accuracy_score(true_labels, predicted_labels)
    macro_f1 = f1_score(true_labels, predicted_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    precision_manip = precision_score(
        true_labels, predicted_labels, pos_label="manipulative", zero_division=0
    )
    recall_manip = recall_score(
        true_labels, predicted_labels, pos_label="manipulative", zero_division=0
    )
    f1_manip = f1_score(
        true_labels, predicted_labels, pos_label="manipulative", zero_division=0
    )
 
    print(f"Manual predictions on first {n} examples")
    print(f"Accuracy:                   {accuracy:.4f}")
    print(f"Macro F1:                   {macro_f1:.4f}")
    print(f"Weighted F1:                {weighted_f1:.4f}")
    print(f"Precision (manipulative):   {precision_manip:.4f}")
    print(f"Recall    (manipulative):   {recall_manip:.4f}")
    print(f"F1        (manipulative):   {f1_manip:.4f}")
 
    print("\nClassification report:")
    print(classification_report(true_labels, predicted_labels, zero_division=0))
 
    labels = ["manipulative", "non_manipulative"]
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    print("Confusion matrix (rows = true, cols = predicted):")
    print(f"  labels: {labels}")
    print(cm)
 
    # True class distribution in the first n examples (so you can compare to
    # the class distribution of your manual predictions)
    print("\nTrue label distribution in first 100:")
    print(pd.Series(true_labels).value_counts().to_string())
    print("\nManual prediction distribution:")
    print(pd.Series(predicted_labels).value_counts().to_string())
 
    # Disagreements with a snippet of text for quick eyeballing
    comparison = pd.DataFrame(
        {
            "row": range(n),
            "true": true_labels,
            "predicted": predicted_labels,
            "text_snippet": df[TEXT_COLUMN]
            .str.replace(r"\s+", " ", regex=True)
            .str.slice(0, 180),
        }
    )
    disagreements = comparison[comparison["true"] != comparison["predicted"]]
    print(f"\n{len(disagreements)} disagreements:")
    if not disagreements.empty:
        with pd.option_context("display.max_colwidth", 200, "display.width", 220):
            print(disagreements.to_string(index=False))
 
if __name__ == "__main__":
    main()