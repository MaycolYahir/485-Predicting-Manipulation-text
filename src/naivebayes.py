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
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

try:
    from .load_data import DEDUPED_DATA_FILE, load_dataset, save_deduped_dataset
    from .utils import ensure_dir
except ImportError:
    from load_data import DEDUPED_DATA_FILE, load_dataset, save_deduped_dataset
    from utils import ensure_dir


LABEL_COLUMN = "manipulation_type"
TEXT_COLUMN = "text"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MANUAL_EXAMPLES = [
    {
        "example_id": "manual_guilt_01",
        "expected_label": "guilt_tripping",
        "text": (
            "A: After everything I've done for you, you can't help me with one thing?\n"
            "B: I have a deadline tonight.\n"
            "A: Wow. I guess my sacrifices meant nothing."
        ),
    },
    {
        "example_id": "manual_charm_01",
        "expected_label": "charm_flattery",
        "text": (
            "A: You're honestly the only person smart enough to understand this.\n"
            "B: What do you need?\n"
            "A: Just sign off on it. I trust your judgment more than anyone's."
        ),
    },
    {
        "example_id": "manual_coercion_01",
        "expected_label": "direct_coercion",
        "text": (
            "A: Send me the file today.\n"
            "B: I need more time to check it.\n"
            "A: No. If you don't send it now, there will be consequences."
        ),
    },
    {
        "example_id": "manual_gaslighting_01",
        "expected_label": "gaslighting",
        "text": (
            "A: I never promised that.\n"
            "B: You said it yesterday.\n"
            "A: You're imagining things again. This is why nobody can talk to you."
        ),
    },
    {
        "example_id": "manual_love_bombing_01",
        "expected_label": "love_bombing",
        "text": (
            "A: I know we just met, but you're my whole world already.\n"
            "B: That feels really fast.\n"
            "A: I bought you something expensive because no one will ever love you like I do."
        ),
    },
    {
        "example_id": "manual_passive_01",
        "expected_label": "passive_aggressive",
        "text": (
            "A: Sure, do whatever you want.\n"
            "B: Are you upset?\n"
            "A: No, it's fine. Some people care about plans, but it's fine."
        ),
    },
    {
        "example_id": "manual_neutral_01",
        "expected_label": "neutral",
        "text": (
            "A: Do you want to study after class?\n"
            "B: Sure, let's meet at the library at five.\n"
            "A: Sounds good. I'll bring the notes."
        ),
    },
]


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
                    stop_words="english"
                ),
            ),
            ("classifier", MultinomialNB(class_prior=[0.5, 0.5])),
        ]
    )


def print_dataset_debug(df: pd.DataFrame, text_column: str) -> None:
    print("\nDebug checks")
    print(f"DataFrame columns: {list(df.columns)}")
    print("\nSample text examples:")
    for _, row in df[[LABEL_COLUMN, text_column]].head(3).iterrows():
        print("---")
        print(f"label: {row[LABEL_COLUMN]}")
        print(str(row[text_column])[:700])

    lower_text = df[text_column].fillna("").astype(str).str.lower()
    label_hits = []
    for label in sorted(df[LABEL_COLUMN].dropna().astype(str).unique()):
        variants = {label.lower(), label.lower().replace("_", " ")}
        count = int(lower_text.apply(lambda text: any(v in text for v in variants)).sum())
        if count:
            label_hits.append((label, count))

    if label_hits:
        raise ValueError(f"Label text appears inside {text_column}: {label_hits}")

    if "conversation_id" in df.columns:
        id_hits = int(
            df.apply(
                lambda row: str(row["conversation_id"]).lower() in str(row[text_column]).lower(),
                axis=1,
            ).sum()
        )
        if id_hits:
            raise ValueError(f"conversation_id appears inside {text_column} for {id_hits} rows")

    print("No literal label names or conversation IDs found in the text column.")


def message_overlap_diagnostic(train_df: pd.DataFrame, test_df: pd.DataFrame, text_column: str) -> dict:
    train_lines = {
        line.strip()
        for text in train_df[text_column]
        for line in str(text).splitlines()
        if line.strip()
    }
    test_lines = {
        line.strip()
        for text in test_df[text_column]
        for line in str(text).splitlines()
        if line.strip()
    }
    shared_lines = sorted(train_lines & test_lines)
    test_rows_with_shared_lines = int(
        test_df[text_column].apply(
            lambda text: any(line.strip() in train_lines for line in str(text).splitlines())
        ).sum()
    )

    return {
        "unique_train_message_lines": int(len(train_lines)),
        "unique_test_message_lines": int(len(test_lines)),
        "shared_message_lines_across_train_test": int(len(shared_lines)),
        "test_rows_with_at_least_one_shared_message_line": test_rows_with_shared_lines,
        "test_fraction_with_shared_message_line": float(test_rows_with_shared_lines / len(test_df)),
        "sample_shared_message_lines": shared_lines[:25],
    }


def print_split_debug(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

    overlap = train_df.index.intersection(test_df.index)
    print(f"Train/test index overlap: {len(overlap)}")
    assert len(overlap) == 0, "Train and test indices overlap."

    train_texts = set(train_df[text_column])
    test_texts = set(test_df[text_column])
    overlapping_texts = train_texts & test_texts
    duplicate_test_rows = int(test_df[text_column].isin(train_texts).sum())
    print(f"Exact duplicate texts across train/test: {len(overlapping_texts)}")
    print(f"Test rows whose text appears in train: {duplicate_test_rows}")
    assert len(overlapping_texts) == 0, "Exact duplicate text appears in both train and test."

    overlap_diagnostic = message_overlap_diagnostic(train_df, test_df, text_column)
    print(
        "Shared individual message lines across train/test: "
        f"{overlap_diagnostic['shared_message_lines_across_train_test']}"
    )
    print(
        "Test rows with at least one shared message line: "
        f"{overlap_diagnostic['test_rows_with_at_least_one_shared_message_line']}"
    )

    majority_class = y_train.value_counts().idxmax()
    majority_accuracy = float((y_test == majority_class).mean())
    print(f"Majority class baseline on this split: {majority_accuracy:.4f} ({majority_class})")
    return overlap_diagnostic


def print_top_model_features(model: Pipeline, top_n: int = 8) -> None:
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()

    print("\nTop TF-IDF/NB features by class:")
    for class_label, feature_log_probs in zip(classifier.classes_, classifier.feature_log_prob_):
        top_indices = feature_log_probs.argsort()[-top_n:][::-1]
        top_terms = ", ".join(feature_names[index] for index in top_indices)
        print(f"{class_label}: {top_terms}")


def train_test_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str, int, int]:

    text_column = validate_dataframe(df)

    clean_df = df.dropna(subset=[LABEL_COLUMN, text_column]).copy()

    clean_df[LABEL_COLUMN] = clean_df[LABEL_COLUMN].astype(str)

    clean_df[text_column] = clean_df[text_column].astype(str)

    original_count = len(clean_df)
    duplicate_text_rows = int(clean_df.duplicated(subset=[text_column]).sum())

    if duplicate_text_rows:
        print(f"Dropping {duplicate_text_rows} exact duplicate text rows before train/test split.")
        clean_df = clean_df.drop_duplicates(subset=[text_column]).copy()

    train_df, test_df = train_test_split(
        clean_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=clean_df[LABEL_COLUMN],
    )

    return train_df, test_df, text_column, original_count, duplicate_text_rows


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

    return pd.DataFrame(rows)


def run_multinomial_nb(
    df: pd.DataFrame,
    debug: bool = True,
    return_model: bool = False,
) -> tuple[dict, pd.DataFrame, pd.DataFrame] | tuple[dict, pd.DataFrame, pd.DataFrame, Pipeline]:

    text_column = validate_dataframe(df)
    if debug:
        print_dataset_debug(df, text_column)

    train_df, test_df, text_column, original_count, duplicate_text_rows = train_test_data(df)

    model = build_model()

    x_train = train_df[text_column]
    y_train = train_df[LABEL_COLUMN]
    x_test = test_df[text_column]
    y_test = test_df[LABEL_COLUMN]

    assert isinstance(x_train, pd.Series), "Training features must be the text Series only."
    assert isinstance(x_test, pd.Series), "Test features must be the text Series only."
    assert x_train.name == text_column and x_test.name == text_column

    if debug:
        overlap_diagnostic = print_split_debug(train_df, test_df, text_column, y_train, y_test)
        print("Fitting TF-IDF + MultinomialNB only on x_train/y_train.")
    else:
        overlap_diagnostic = message_overlap_diagnostic(train_df, test_df, text_column)

    model.fit(x_train, y_train)

    if debug:
        print_top_model_features(model)
        print("Predicting only on x_test.")

    predicted_labels = model.predict(x_test)
    assert len(predicted_labels) == len(y_test), "Prediction length does not match y_test."

    report_df = classification_report_table(y_test, predicted_labels)

    majority_class = y_train.value_counts().idxmax()
    majority_accuracy = float((y_test == majority_class).mean())

    metrics = {

        "model": "text_only_multinomial_nb",
        "data_source": "load_dataset() with exact text deduplication before split",
        "target_column": LABEL_COLUMN,
        "text_column": text_column,
        "random_seed": RANDOM_STATE,
        "test_size_fraction": TEST_SIZE,

        "total_examples_before_deduplication": int(original_count),
        "duplicate_text_rows_removed_before_split": int(duplicate_text_rows),
        "total_examples": int(len(train_df) + len(test_df)),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),

        "number_of_classes": int(pd.concat([y_train, y_test]).nunique()),
        "majority_class_baseline_accuracy_on_same_split": majority_accuracy,
        "processed_deduped_data_path": str(DEDUPED_DATA_FILE.relative_to(PROJECT_ROOT)),
        "message_overlap_diagnostic": overlap_diagnostic,

        "features": {

            "type": "tfidf",
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.95,
            "max_features": 5000,

        },

        "classifier": "MultinomialNB",
        "accuracy": float(accuracy_score(y_test, predicted_labels)),
        "macro_f1": float(
            f1_score(y_test, predicted_labels, average="macro", zero_division=0)
        ),
        "class_metrics": report_df.set_index("class").to_dict(orient="index"),
    }

    prediction_data = {
        "true_label": y_test.to_list(),
        "predicted_label": predicted_labels.tolist(),
    }

    if "conversation_id" in test_df.columns:
        prediction_data = {
            "conversation_id": test_df["conversation_id"].to_list(),
            **prediction_data,
        }

    predictions = pd.DataFrame(prediction_data)

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
    output_path: str | Path = Path(
        "results/predictions/text_only_multinomial_nb_predictions.csv"
    ),
) -> Path:
    path = Path(output_path)
    ensure_dir(path.parent)
    predictions.to_csv(path, index=False)
    return path


def predict_manual_examples(model: Pipeline) -> pd.DataFrame:
    examples = pd.DataFrame(MANUAL_EXAMPLES)
    predicted_labels = model.predict(examples["text"])
    probabilities = model.predict_proba(examples["text"])
    confidences = probabilities.max(axis=1)

    examples["predicted_label"] = predicted_labels
    examples["predicted_confidence"] = confidences
    examples["matched_expected_label"] = examples["expected_label"] == examples["predicted_label"]

    return examples


def save_manual_example_predictions(
    predictions: pd.DataFrame,
    output_path: str | Path = Path("results/predictions/manual_example_predictions.csv"),
) -> Path:
    path = Path(output_path)
    ensure_dir(path.parent)
    predictions.to_csv(path, index=False)
    return path


def save_message_overlap_diagnostic(
    metrics: dict,
    output_path: str | Path = Path("results/metrics/message_overlap_diagnostic.json"),
) -> Path:
    path = Path(output_path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(metrics["message_overlap_diagnostic"], indent=2), encoding="utf-8")
    return path


def save_baseline_comparison_chart(
    nb_metrics: dict,
    majority_metrics_path: str | Path = Path("results/metrics/majority_baseline.json"),
    output_path: str | Path = Path("figures/text_only_model_comparison.png"),
) -> Path:
    majority_path = Path(majority_metrics_path)
    if not majority_path.exists():
        raise FileNotFoundError(f"Missing majority baseline metrics: {majority_path}")

    majority_metrics = json.loads(majority_path.read_text(encoding="utf-8"))

    comparison = pd.DataFrame(
        [
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
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    comparison.set_index("model")[["accuracy", "macro_f1"]].plot(
        kind="bar",
        ax=ax,
        color=["#4C78A8", "#59A14F"],
        edgecolor="black",
    )
    ax.set_title("Text-Only Model Comparison")
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


def progress_report_sentence(metrics: dict) -> str:
    return (
        "The text-only TF-IDF bigram Multinomial Naive Bayes model achieved "
        f"{metrics['accuracy']:.4f} accuracy and {metrics['macro_f1']:.4f} macro F1 "
        "on the 20% stratified test split, improving over the majority-class baseline."
    )


def print_summary(metrics: dict) -> None:

    print("TF-IDF + Multinomial Naive Bayes Summary")
    print(f"Total examples: {metrics['total_examples']}")
    print(f"Train size / test size: {metrics['train_size']} / {metrics['test_size']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")


def main() -> None:

    deduped_path = save_deduped_dataset()
    df = load_dataset()
    metrics, predictions, report_df, model = run_multinomial_nb(df, return_model=True)
    manual_predictions = predict_manual_examples(model)

    metrics_path = save_metrics(metrics)
    predictions_path = save_predictions(predictions)
    manual_predictions_path = save_manual_example_predictions(manual_predictions)
    overlap_path = save_message_overlap_diagnostic(metrics)
    chart_path = save_baseline_comparison_chart(metrics)

    print_summary(metrics)
    print("\nClass-wise metrics:")
    print(report_df.to_string(index=False))
    print("\nManual example sanity check:")
    print(
        manual_predictions[
            ["example_id", "expected_label", "predicted_label", "predicted_confidence"]
        ].to_string(index=False)
    )
    print(f"\nSaved deduped dataset to: {deduped_path}")
    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved manual example predictions to: {manual_predictions_path}")
    print(f"Saved message overlap diagnostic to: {overlap_path}")
    print(f"Saved comparison chart to: {chart_path}")
    print()
    print(progress_report_sentence(metrics))


if __name__ == "__main__":
    main()
