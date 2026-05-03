from __future__ import annotations
 
import json
import os
import re
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
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
 
try:
    from .load_data import load_dataset
    from .utils import LABEL_COLUMN, RANDOM_STATE, TEST_SIZE, TEXT_COLUMN, ensure_dir, split_for_modeling
except ImportError:
    from load_data import load_dataset
    from utils import LABEL_COLUMN, RANDOM_STATE, TEST_SIZE, TEXT_COLUMN, ensure_dir, split_for_modeling
 
 
# Replace "Person1: " / "Person2: " with sentinel tokens __p1__ / __p2__
# (case-insensitive). Lets TF-IDF learn speaker-asymmetric features and lets
# the hand-crafted feature transformer split text by speaker.
DIALOGUE_SPEAKER_PATTERN = re.compile(r"Person(\d+)\s*:\s*", flags=re.IGNORECASE)
 
 
def clean_dialogue_text(text: str) -> str:
    return DIALOGUE_SPEAKER_PATTERN.sub(r"__p\1__ ", str(text)).strip()
 
 
# Patterns used by DialogueStatsTransformer.
SPEAKER_TAG_PATTERN = re.compile(r"__p(\d+)__")
WORD_PATTERN = re.compile(r"\b\w+\b")
SECOND_PERSON_PATTERN = re.compile(r"\b(?:you|your|yours|yourself)\b", re.IGNORECASE)
FIRST_PERSON_PATTERN = re.compile(r"\b(?:i|me|my|mine|myself)\b", re.IGNORECASE)
MODAL_PATTERN = re.compile(
    r"\b(?:should|must|need\s+to|have\s+to|ought\s+to|got\s+to|gotta)\b",
    re.IGNORECASE,
)
 
 
class DialogueStatsTransformer(BaseEstimator, TransformerMixin):
    """Hand-crafted numeric features extracted from speaker-tagged dialogue.
 
    Targets manipulation-relevant signals BoW cannot see directly:
      - structure: dialogue length, turn count, average turn length
      - punctuation: question / exclamation marks
      - lexicon:    count of imperative-ish modals (should, must, have to ...)
      - asymmetry:  who talks more (p1_word_share), and how lopsided the
                    "you" / "I" usage is across speakers (absolute differences,
                    so direction of which speaker is P1 vs P2 doesn't matter)
    """
 
    FEATURE_NAMES = [
        "n_words",
        "n_turns",
        "avg_words_per_turn",
        "question_marks",
        "exclamation_marks",
        "modal_count",
        "p1_word_share",
        "you_count_asymmetry",
        "i_count_asymmetry",
        "p1_question_marks",
        "p2_question_marks",
    ]
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X):
        return np.array([self._features_for(text) for text in X], dtype=float)
 
    def get_feature_names_out(self, input_features=None):
        return np.array(self.FEATURE_NAMES)
 
    @staticmethod
    def _extract_segments(text: str) -> list[tuple[str, str]]:
        # SPEAKER_TAG_PATTERN.split with one capturing group returns
        # [pre_text, speaker_id_1, content_1, speaker_id_2, content_2, ...]
        parts = SPEAKER_TAG_PATTERN.split(text)
        segments: list[tuple[str, str]] = []
        for index in range(1, len(parts) - 1, 2):
            speaker_id = parts[index]
            content = parts[index + 1].strip()
            if content:
                segments.append((speaker_id, content))
        return segments
 
    @classmethod
    def _features_for(cls, text: str) -> list[float]:
        text = str(text)
        segments = cls._extract_segments(text)
        n_turns = max(len(segments), 1)
 
        p1_text = " ".join(content for speaker, content in segments if speaker == "1")
        p2_text = " ".join(content for speaker, content in segments if speaker == "2")
        full_text = (p1_text + " " + p2_text).strip()
 
        p1_words = WORD_PATTERN.findall(p1_text)
        p2_words = WORD_PATTERN.findall(p2_text)
        n_words = max(len(p1_words) + len(p2_words), 1)
 
        p1_you = len(SECOND_PERSON_PATTERN.findall(p1_text))
        p2_you = len(SECOND_PERSON_PATTERN.findall(p2_text))
        p1_i = len(FIRST_PERSON_PATTERN.findall(p1_text))
        p2_i = len(FIRST_PERSON_PATTERN.findall(p2_text))
 
        return [
            float(n_words),
            float(n_turns),
            float(n_words / n_turns),
            float(full_text.count("?")),
            float(full_text.count("!")),
            float(len(MODAL_PATTERN.findall(full_text))),
            float(len(p1_words) / n_words),
            float(abs(p1_you - p2_you)),
            float(abs(p1_i - p2_i)),
            float(p1_text.count("?")),
            float(p2_text.count("?")),
        ]
 
 
def validate_dataframe(df: pd.DataFrame) -> str:
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Missing required label column: {LABEL_COLUMN}")
 
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Missing required text column: {TEXT_COLUMN}")
 
    return TEXT_COLUMN
 
 
def build_model() -> Pipeline:
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=10000,
        sublinear_tf=True,
    )
    # Character n-grams catch stylistic patterns (suffixes, intensifiers,
    # contractions) that word tokens miss.
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        max_features=20000,
        sublinear_tf=True,
    )
    # StandardScaler so the dense numeric features sit on a comparable scale
    # to TF-IDF before they go into FeatureUnion.
    stats_pipeline = Pipeline(
        steps=[
            ("extract", DialogueStatsTransformer()),
            ("scale", StandardScaler()),
        ]
    )
    features = FeatureUnion(
        transformer_list=[
            ("word", word_vectorizer),
            ("char", char_vectorizer),
            ("stats", stats_pipeline),
        ]
    )
 
    return Pipeline(
        steps=[
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=1000,
                    C=4.0,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
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
    print("\nSample text examples (after speaker-prefix tagging):")
    for _, row in df[[LABEL_COLUMN, text_column]].head(3).iterrows():
        print("---")
        print(f"label: {row[LABEL_COLUMN]}")
        print(str(row[text_column])[:700])
 
 
def print_top_model_features(model: Pipeline, top_n: int = 12) -> None:
    features_step = model.named_steps["features"]
    classifier = model.named_steps["classifier"]
    feature_names = features_step.get_feature_names_out()
    coefficients = classifier.coef_
 
    print("\nTop combined features by class:")
    if coefficients.shape[0] == 1:
        # Binary LR: coef_ has shape (1, n_features). Positive weights push toward
        # classes_[1], negative weights push toward classes_[0].
        weights = coefficients[0]
        top_positive_indices = weights.argsort()[-top_n:][::-1]
        top_negative_indices = weights.argsort()[:top_n]
        positive_terms = ", ".join(feature_names[i] for i in top_positive_indices)
        negative_terms = ", ".join(feature_names[i] for i in top_negative_indices)
        print(f"{classifier.classes_[1]}: {positive_terms}")
        print(f"{classifier.classes_[0]}: {negative_terms}")
    else:
        for class_label, weights in zip(classifier.classes_, coefficients):
            top_indices = weights.argsort()[-top_n:][::-1]
            top_terms = ", ".join(feature_names[index] for index in top_indices)
            print(f"{class_label}: {top_terms}")
 
    # Always surface where the hand-crafted features land in the ranking.
    stats_prefix = "stats__"
    stats_indices = [i for i, name in enumerate(feature_names) if name.startswith(stats_prefix)]
    if stats_indices and coefficients.shape[0] == 1:
        stats_weights = [(feature_names[i], coefficients[0][i]) for i in stats_indices]
        stats_weights.sort(key=lambda pair: abs(pair[1]), reverse=True)
        print("\nHand-crafted feature weights (sorted by |coef|):")
        for name, weight in stats_weights:
            print(f"  {name.replace(stats_prefix, ''):28s} {weight:+.4f}")
 
 
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
 
 
def run_logistic_regression(
    df: pd.DataFrame,
    stage_name: str = "stage1_binary",
    debug: bool = True,
    rare_label_min_count: int | None = None,
    return_model: bool = False,
) -> tuple[dict, pd.DataFrame, pd.DataFrame] | tuple[dict, pd.DataFrame, pd.DataFrame, Pipeline]:
    text_column = validate_dataframe(df)
 
    df = df.copy()
    df[text_column] = df[text_column].map(clean_dialogue_text)
 
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
 
    classifier = model.named_steps["classifier"]
    metrics = {
        "model": "text_only_logistic_regression",
        "stage": stage_name,
        "data_source": "MentalManip with exact text deduplication before split",
        "text_preprocessing": "tagged_person_speaker_prefixes",
        "target_column": LABEL_COLUMN,
        "text_column": text_column,
        "random_seed": RANDOM_STATE,
        "test_size_fraction": TEST_SIZE,
        **split_metadata,
        "number_of_classes": int(pd.concat([y_train, y_test]).nunique()),
        "features": {
            "type": "feature_union(word_tfidf + char_tfidf + dialogue_stats)",
            "word_tfidf": {
                "ngram_range": [1, 2],
                "min_df": 2,
                "max_df": 0.95,
                "max_features": 10000,
                "sublinear_tf": True,
            },
            "char_tfidf": {
                "analyzer": "char_wb",
                "ngram_range": [3, 5],
                "min_df": 2,
                "max_df": 0.95,
                "max_features": 20000,
                "sublinear_tf": True,
            },
            "dialogue_stats_features": list(DialogueStatsTransformer.FEATURE_NAMES),
            "speaker_tagging": True,
        },
        "classifier": "LogisticRegression",
        "classifier_params": {
            "solver": classifier.solver,
            "max_iter": classifier.max_iter,
            "C": classifier.C,
            "class_weight": classifier.class_weight,
            "random_state": classifier.random_state,
        },
        "n_iter": [int(value) for value in classifier.n_iter_.tolist()],
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
    output_path: str | Path = Path("results/metrics/text_only_logistic_regression.json"),
) -> Path:
    path = Path(output_path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path
 
 
def save_predictions(
    predictions: pd.DataFrame,
    output_path: str | Path = Path("results/predictions/text_only_logistic_regression_predictions.csv"),
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
 
 
def save_nb_comparison_chart(
    lr_metrics: dict,
    nb_metrics_path: str | Path,
    output_path: str | Path,
) -> Path:
    nb_path = Path(nb_metrics_path)
    if not nb_path.exists():
        raise FileNotFoundError(f"Missing Naive Bayes metrics: {nb_path}")
 
    nb_metrics = json.loads(nb_path.read_text(encoding="utf-8"))
 
    comparison = pd.DataFrame(
        [
            {
                "model": "TF-IDF + MultinomialNB",
                "accuracy": nb_metrics["accuracy"],
                "macro_f1": nb_metrics["macro_f1"],
            },
            {
                "model": "TF-IDF + LogisticRegression",
                "accuracy": lr_metrics["accuracy"],
                "macro_f1": lr_metrics["macro_f1"],
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
    ax.set_title(f"Model Comparison: {lr_metrics['stage']}")
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
    print("TF-IDF + Logistic Regression Summary")
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
    metrics, predictions, report_df = run_logistic_regression(df, stage_name="stage1_binary")
 
    metrics_path = save_metrics(metrics)
    predictions_path = save_predictions(predictions)
    report_path = save_classification_report(
        report_df,
        output_path=PROJECT_ROOT / "results" / "metrics" / "stage1_binary_text_only_logistic_regression_report.csv",
    )
    confusion_path = save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        output_path=PROJECT_ROOT / "figures" / "stage1_binary_text_only_logistic_regression_confusion_matrix.png",
        title="Stage 1 Binary Confusion Matrix (Logistic Regression)",
    )
    comparison_path = save_nb_comparison_chart(
        metrics,
        nb_metrics_path=PROJECT_ROOT / "results" / "metrics" / "text_only_multinomial_nb.json",
        output_path=PROJECT_ROOT / "figures" / "stage1_binary_logistic_regression_vs_naive_bayes.png",
    )
 
    print_summary(metrics)
    print("\nClass-wise metrics:")
    print(report_df.to_string(index=False))
    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved class report to: {report_path}")
    print(f"Saved confusion matrix plot to: {confusion_path}")
    print(f"Saved LR vs NB comparison chart to: {comparison_path}")
 
 
if __name__ == "__main__":
    main()