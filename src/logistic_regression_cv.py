"""5-fold stratified cross-validation for the Stage-1 LR baseline.

Runs twice: once on the full deduplicated data, once with the majority class
downsampled to the size of the minority class. Reuses the model, cleaning, and
helpers from logistic_regression.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

try:
    from .logistic_regression import (
        PROJECT_ROOT,
        build_model,
        classification_report_table,
        clean_dialogue_text,
        save_confusion_matrix_plot,
        validate_dataframe,
        _confusion_matrix_payload,
        _is_binary_problem,
    )
    from .load_data import load_dataset
    from .utils import (
        LABEL_COLUMN,
        RANDOM_STATE,
        ensure_dir,
        prepare_modeling_dataframe,
    )
except ImportError:
    from logistic_regression import (
        PROJECT_ROOT,
        build_model,
        classification_report_table,
        clean_dialogue_text,
        save_confusion_matrix_plot,
        validate_dataframe,
        _confusion_matrix_payload,
        _is_binary_problem,
    )
    from load_data import load_dataset
    from utils import (
        LABEL_COLUMN,
        RANDOM_STATE,
        ensure_dir,
        prepare_modeling_dataframe,
    )


def downsample_majority_class(
    df: pd.DataFrame,
    label_column: str = LABEL_COLUMN,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Random undersampling: match every class to the smallest class size."""
    label_counts = df[label_column].value_counts()
    min_count = int(label_counts.min())
    sampled_groups = [
        df[df[label_column] == label].sample(n=min_count, random_state=random_state)
        for label in label_counts.index
    ]
    return (
        pd.concat(sampled_groups, ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


def run_logistic_regression_cv(
    df: pd.DataFrame,
    stage_name: str = "stage1_binary",
    n_splits: int = 5,
    balance: bool = False,
    debug: bool = True,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    text_column = validate_dataframe(df)

    df = df.copy()
    df[text_column] = df[text_column].map(clean_dialogue_text)

    clean_df, original_count, duplicate_text_rows = prepare_modeling_dataframe(
        df,
        label_column=LABEL_COLUMN,
        text_column=text_column,
    )

    pre_balance_count = int(len(clean_df))
    if balance:
        clean_df = downsample_majority_class(clean_df)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    fold_rows: list[dict] = []
    oof_true: list[str] = []
    oof_pred: list[str] = []

    if debug:
        print(
            f"\nRunning {n_splits}-fold stratified CV "
            f"(balance={balance}, n_examples={len(clean_df)})"
        )

    for fold_index, (train_idx, test_idx) in enumerate(
        skf.split(clean_df, clean_df[LABEL_COLUMN]), start=1
    ):
        train_df = clean_df.iloc[train_idx]
        test_df = clean_df.iloc[test_idx]

        model = build_model()
        model.fit(train_df[text_column], train_df[LABEL_COLUMN])
        predicted = model.predict(test_df[text_column]).tolist()
        y_test = test_df[LABEL_COLUMN]

        fold_row = {
            "fold": fold_index,
            "train_size": int(len(train_df)),
            "test_size": int(len(test_df)),
            "accuracy": float(accuracy_score(y_test, predicted)),
            "macro_f1": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
            "weighted_f1": float(
                f1_score(y_test, predicted, average="weighted", zero_division=0)
            ),
        }
        if _is_binary_problem(y_test):
            fold_row["precision_manipulative"] = float(
                precision_score(y_test, predicted, pos_label="manipulative", zero_division=0)
            )
            fold_row["recall_manipulative"] = float(
                recall_score(y_test, predicted, pos_label="manipulative", zero_division=0)
            )
            fold_row["f1_manipulative"] = float(
                f1_score(y_test, predicted, pos_label="manipulative", zero_division=0)
            )
        fold_rows.append(fold_row)

        oof_true.extend(y_test.tolist())
        oof_pred.extend(predicted)

        if debug:
            print(
                f"  fold {fold_index}: "
                f"acc={fold_row['accuracy']:.4f} "
                f"macro_f1={fold_row['macro_f1']:.4f}"
            )

    fold_df = pd.DataFrame(fold_rows)

    metric_columns = [
        c for c in fold_df.columns if c not in {"fold", "train_size", "test_size"}
    ]
    summary: dict[str, float] = {}
    for column in metric_columns:
        summary[f"{column}_mean"] = float(fold_df[column].mean())
        summary[f"{column}_std"] = float(fold_df[column].std(ddof=1))

    oof_predictions = pd.DataFrame({"true_label": oof_true, "predicted_label": oof_pred})
    oof_report_df = classification_report_table(pd.Series(oof_true), oof_pred)
    oof_confusion = _confusion_matrix_payload(pd.Series(oof_true), oof_pred)

    metrics = {
        "model": "text_only_logistic_regression_cv",
        "stage": stage_name,
        "data_source": "MentalManip with exact text deduplication before split",
        "text_preprocessing": "tagged_person_speaker_prefixes",
        "balance_strategy": "undersample_majority_to_minority" if balance else None,
        "cross_validation": {
            "n_splits": n_splits,
            "stratified": True,
            "shuffle": True,
            "random_state": RANDOM_STATE,
        },
        "total_examples_before_deduplication": int(original_count),
        "duplicate_text_rows_removed_before_split": int(duplicate_text_rows),
        "examples_after_dedup_before_balance": pre_balance_count,
        "examples_used_for_cv": int(len(clean_df)),
        "fold_metrics": fold_rows,
        "summary": summary,
        "oof_class_metrics": oof_report_df.set_index("class").to_dict(orient="index"),
        "oof_confusion_matrix": oof_confusion,
    }

    return metrics, oof_predictions, fold_df


def save_cv_metrics(metrics: dict, suffix: str = "") -> Path:
    path = (
        PROJECT_ROOT
        / "results"
        / "metrics"
        / f"text_only_logistic_regression_cv{suffix}.json"
    )
    ensure_dir(path.parent)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def save_cv_fold_table(fold_df: pd.DataFrame, suffix: str = "") -> Path:
    path = (
        PROJECT_ROOT
        / "results"
        / "metrics"
        / f"text_only_logistic_regression_cv_folds{suffix}.csv"
    )
    ensure_dir(path.parent)
    fold_df.to_csv(path, index=False)
    return path


def save_cv_predictions(predictions: pd.DataFrame, suffix: str = "") -> Path:
    path = (
        PROJECT_ROOT
        / "results"
        / "predictions"
        / f"text_only_logistic_regression_cv_oof_predictions{suffix}.csv"
    )
    ensure_dir(path.parent)
    predictions.to_csv(path, index=False)
    return path


def print_cv_summary(metrics: dict) -> None:
    summary = metrics["summary"]
    print(
        f"\n{metrics['model']} | "
        f"balance_strategy={metrics['balance_strategy']}"
    )
    print(
        f"Examples used: {metrics['examples_used_for_cv']}  "
        f"(folds: {metrics['cross_validation']['n_splits']})"
    )
    print(f"Accuracy:    {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")
    print(f"Macro F1:    {summary['macro_f1_mean']:.4f} +/- {summary['macro_f1_std']:.4f}")
    print(f"Weighted F1: {summary['weighted_f1_mean']:.4f} +/- {summary['weighted_f1_std']:.4f}")
    if "precision_manipulative_mean" in summary:
        print(
            f"Precision (manipulative): "
            f"{summary['precision_manipulative_mean']:.4f} +/- {summary['precision_manipulative_std']:.4f}"
        )
        print(
            f"Recall    (manipulative): "
            f"{summary['recall_manipulative_mean']:.4f} +/- {summary['recall_manipulative_std']:.4f}"
        )
        print(
            f"F1        (manipulative): "
            f"{summary['f1_manipulative_mean']:.4f} +/- {summary['f1_manipulative_std']:.4f}"
        )


def main() -> None:
    df = load_dataset(task="binary")

    print("=" * 70)
    print("Run 1: stratified 5-fold CV, no data rebalancing")
    print("=" * 70)
    metrics_unb, oof_unb, folds_unb = run_logistic_regression_cv(
        df, stage_name="stage1_binary", n_splits=5, balance=False
    )
    save_cv_metrics(metrics_unb, suffix="")
    save_cv_fold_table(folds_unb, suffix="")
    save_cv_predictions(oof_unb, suffix="")
    cm_path_unb = save_confusion_matrix_plot(
        metrics_unb["oof_confusion_matrix"],
        output_path=PROJECT_ROOT
        / "figures"
        / "stage1_binary_logistic_regression_cv_confusion_matrix.png",
        title="Stage 1 LR 5-fold CV (no balancing): out-of-fold confusion matrix",
    )
    print_cv_summary(metrics_unb)
    print(f"Saved confusion matrix plot to: {cm_path_unb}")

    print("\n" + "=" * 70)
    print("Run 2: stratified 5-fold CV, majority class downsampled to minority size")
    print("=" * 70)
    metrics_bal, oof_bal, folds_bal = run_logistic_regression_cv(
        df, stage_name="stage1_binary", n_splits=5, balance=True
    )
    save_cv_metrics(metrics_bal, suffix="_balanced")
    save_cv_fold_table(folds_bal, suffix="_balanced")
    save_cv_predictions(oof_bal, suffix="_balanced")
    cm_path_bal = save_confusion_matrix_plot(
        metrics_bal["oof_confusion_matrix"],
        output_path=PROJECT_ROOT
        / "figures"
        / "stage1_binary_logistic_regression_cv_balanced_confusion_matrix.png",
        title="Stage 1 LR 5-fold CV (balanced): out-of-fold confusion matrix",
    )
    print_cv_summary(metrics_bal)
    print(f"Saved confusion matrix plot to: {cm_path_bal}")


if __name__ == "__main__":
    main()