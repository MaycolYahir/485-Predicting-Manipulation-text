from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIG_DIR = PROJECT_ROOT / ".matplotlib"
XDG_CACHE_DIR = PROJECT_ROOT / ".cache"
MPLCONFIG_DIR.mkdir(exist_ok=True)
XDG_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

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


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=5000,
                    stop_words="english",
                ),
            ),
            # fit_prior=False gives uniform class priors. For stage 1 this is
            # equivalent to [0.5, 0.5], and it still works for multi-class stage 2.
            ("classifier", MultinomialNB(fit_prior=False)),
        ]
    )


def _is_binary_problem(y_true: pd.Series) -> bool:
    return sorted(y_true.astype(str).unique().tolist()) == ["manipulative", "non_manipulative"]


def _confusion_matrix_payload(y_true: pd.Series, y_pred: list[str]) -> dict:
    labels = sorted(set(y_true.astype(str).tolist()) | set(map(str, y_pred)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "labels": labels,
        "matrix": matrix.tolist(),
        "manageable_for_display": len(labels) <= 20,
    }


def classification_report_table(y_true: pd.Series, y_pred: list[str]) -> pd.DataFrame:
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

    report_df = pd.DataFrame(rows)
    if not report_df.empty:
        report_df = report_df.sort_values(["support", "class"], ascending=[False, True]).reset_index(
            drop=True
        )
    return report_df


def print_dataset_debug(df: pd.DataFrame, text_column: str) -> None:
    print("\nDebug checks")
    print(f"DataFrame columns: {list(df.columns)}")
    print("\nSample text examples:")
    for _, row in df[[LABEL_COLUMN, text_column]].head(3).iterrows():
        print("---")
        print(f"label: {row[LABEL_COLUMN]}")
        print(str(row[text_column])[:700])


def print_top_model_features(model: Pipeline, top_n: int = 8) -> None:
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()

    print("\nTop TF-IDF/NB features by class:")
    for class_label, feature_log_probs in zip(classifier.classes_, classifier.feature_log_prob_):
        top_indices = feature_log_probs.argsort()[-top_n:][::-1]
        top_terms = ", ".join(feature_names[index] for index in top_indices)
        print(f"{class_label}: {top_terms}")


def save_confusion_matrix_plot(
    confusion_payload: dict,
    output_path: str | Path,
    title: str,
) -> Path:
    labels = confusion_payload["labels"]
    matrix = confusion_payload["matrix"]

    fig_width = min(max(8, len(labels) * 0.6), 20)
    fig_height = min(max(6, len(labels) * 0.45), 18)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, str(value), ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    path = Path(output_path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def run_multinomial_nb(
    df: pd.DataFrame,
    stage_name: str = "stage1_binary",
    debug: bool = True,
    rare_label_min_count: int | None = None,
    return_model: bool = False,
) -> tuple[dict, pd.DataFrame, pd.DataFrame] | tuple[dict, pd.DataFrame, pd.DataFrame, Pipeline]:
    text_column = validate_dataframe(df)
    if debug:
        print_dataset_debug(df, text_column)

    train_df, test_df, split_metadata = split_for_modeling(
        df,
        label_column=LABEL_COLUMN,
        text_column=text_column,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        rare_label_min_count=rare_label_min_count,
    )

    model = build_model()
    x_train = train_df[text_column]
    y_train = train_df[LABEL_COLUMN]
    x_test = test_df[text_column]
    y_test = test_df[LABEL_COLUMN]

    model.fit(x_train, y_train)

    if debug:
        print_top_model_features(model)

    predicted_labels = model.predict(x_test)
    report_df = classification_report_table(y_test, predicted_labels.tolist())
    confusion_payload = _confusion_matrix_payload(y_test, predicted_labels.tolist())

    metrics = {
        "model": "text_only_multinomial_nb",
        "stage": stage_name,
        "data_source": "MentalManip with exact text deduplication before split",
        "target_column": LABEL_COLUMN,
        "text_column": text_column,
        "random_seed": RANDOM_STATE,
        "test_size_fraction": TEST_SIZE,
        **split_metadata,
        "number_of_classes": int(pd.concat([y_train, y_test]).nunique()),
        "features": {
            "type": "tfidf",
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.95,
            "max_features": 5000,
            "stop_words": "english",
        },
        "classifier": "MultinomialNB",
        "accuracy": float(accuracy_score(y_test, predicted_labels)),
        "macro_f1": float(f1_score(y_test, predicted_labels, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_test, predicted_labels, average="weighted", zero_division=0)
        ),
        "class_metrics": report_df.set_index("class").to_dict(orient="index"),
        "confusion_matrix": confusion_payload,
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
            "predicted_label": predicted_labels.tolist(),
        }
    )

    if return_model:
        return metrics, predictions, report_df, model

    return metrics, predictions, report_df


def save_metrics(
    metrics: dict,
    output_path: str | Path = Path("results/metrics/text_only_multinomial_nb.json"),
) -> Path:
    path = Path(output_path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def save_predictions(
    predictions: pd.DataFrame,
    output_path: str | Path = Path("results/predictions/text_only_multinomial_nb_predictions.csv"),
) -> Path:
    path = Path(output_path)
    ensure_dir(path.parent)
    predictions.to_csv(path, index=False)
    return path


def save_classification_report(
    report_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    ensure_dir(path.parent)
    report_df.to_csv(path, index=False)
    return path


def save_baseline_comparison_chart(
    nb_metrics: dict,
    majority_metrics_path: str | Path,
    output_path: str | Path,
    bert_metrics: dict | None = None,
) -> Path:
    majority_path = Path(majority_metrics_path)
    if not majority_path.exists():
        raise FileNotFoundError(f"Missing majority baseline metrics: {majority_path}")

    majority_metrics = json.loads(majority_path.read_text(encoding="utf-8"))

    comparison_rows = [
        {
            "model": "Majority baseline",
            "accuracy": majority_metrics["accuracy"],
            "macro_f1": majority_metrics["macro_f1"],
        },
        {
            "model": "TF-IDF + MultinomialNB",
            "accuracy": nb_metrics["accuracy"],
            "macro_f1": nb_metrics["macro_f1"],
        },
    ]
    if bert_metrics is not None:
        transformer_name = str(bert_metrics.get("model_name", "BERT")).lower()
        transformer_label = (
            "DistilBERT" if "distilbert" in transformer_name else "BERT classifier"
        )
        comparison_rows.append(
            {
                "model": transformer_label,
                "accuracy": bert_metrics["accuracy"],
                "macro_f1": bert_metrics["macro_f1"],
            }
        )

    comparison = pd.DataFrame(comparison_rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    comparison.set_index("model")[["accuracy", "macro_f1"]].plot(
        kind="bar",
        ax=ax,
        color=["#4C78A8", "#59A14F"],
        edgecolor="black",
    )
    ax.set_title(f"Model Comparison: {nb_metrics['stage']}")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)

    fig.tight_layout()

    path = Path(output_path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def print_summary(metrics: dict) -> None:
    print("TF-IDF + Multinomial Naive Bayes Summary")
    print(f"Stage: {metrics['stage']}")
    print(f"Total examples: {metrics['total_examples_after_cleaning']}")
    print(f"Train size / test size: {metrics['train_size']} / {metrics['test_size']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    if "precision" in metrics:
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")


def main() -> None:
    df = load_dataset(task="binary")
    metrics, predictions, report_df = run_multinomial_nb(df, stage_name="stage1_binary")

    metrics_path = save_metrics(metrics)
    predictions_path = save_predictions(predictions)
    report_path = save_classification_report(
        report_df,
        output_path=PROJECT_ROOT / "results" / "metrics" / "stage1_binary_text_only_multinomial_nb_report.csv",
    )
    confusion_path = save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        output_path=PROJECT_ROOT / "figures" / "stage1_binary_text_only_multinomial_nb_confusion_matrix.png",
        title="Stage 1 Binary Confusion Matrix",
    )

    print_summary(metrics)
    print("\nClass-wise metrics:")
    print(report_df.to_string(index=False))
    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved class report to: {report_path}")
    print(f"Saved confusion matrix plot to: {confusion_path}")


if __name__ == "__main__":
    main()
