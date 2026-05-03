from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIG_DIR = PROJECT_ROOT / ".matplotlib"
XDG_CACHE_DIR = PROJECT_ROOT / ".cache"
MPLCONFIG_DIR.mkdir(exist_ok=True)
XDG_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
except ImportError as exc:
    raise ImportError(
        "bert.py requires torch and transformers. Install them before running this module."
    ) from exc

try:
    from .load_data import DEFAULT_CONFIG, load_dataset
    from .utils import LABEL_COLUMN, RANDOM_STATE, TEST_SIZE, TEXT_COLUMN, ensure_dir, split_for_modeling
except ImportError:
    from load_data import DEFAULT_CONFIG, load_dataset
    from utils import LABEL_COLUMN, RANDOM_STATE, TEST_SIZE, TEXT_COLUMN, ensure_dir, split_for_modeling


DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_TASK = "binary"
DEFAULT_MAX_LENGTH = 128
DEFAULT_NUM_TRAIN_EPOCHS = 1
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_TRAIN_BATCH_SIZE = 16
DEFAULT_EVAL_BATCH_SIZE = 16
DEFAULT_RARE_LABEL_MIN_COUNT: int | None = None
DEFAULT_USE_CLASS_WEIGHTS = False


@dataclass
class LabelEncoderBundle:
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]


def detect_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TextClassificationDataset(Dataset):
    """Minimal torch Dataset wrapper for tokenized text classification data."""

    def __init__(self, encodings: dict[str, list[int]], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(value[index])
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index])
        return item


class WeightedLossTrainer(Trainer):
    """Trainer variant that applies inverse-frequency class weights."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def validate_dataframe(df: pd.DataFrame) -> str:

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Missing required label column: {LABEL_COLUMN}")

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Missing required text column: {TEXT_COLUMN}")

    return TEXT_COLUMN


def build_label_encoder(labels: pd.Series) -> LabelEncoderBundle:

    unique_labels = sorted(labels.astype(str).unique().tolist())
    label_to_id = {label: index for index, label in enumerate(unique_labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    return LabelEncoderBundle(label_to_id=label_to_id, id_to_label=id_to_label)


def encode_labels(labels: pd.Series, encoder: LabelEncoderBundle) -> list[int]:
    return [encoder.label_to_id[str(label)] for label in labels.astype(str).tolist()]


def compute_balanced_class_weights(label_ids: list[int], num_labels: int) -> torch.Tensor:
    counts = np.bincount(label_ids, minlength=num_labels)
    if (counts == 0).any():
        missing = np.where(counts == 0)[0].tolist()
        raise ValueError(f"Cannot compute class weights; missing class ids in training data: {missing}")

    total = counts.sum()
    weights = total / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float)


def classification_report_table(y_true: list[str], y_pred: list[str]) -> pd.DataFrame:

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


def confusion_matrix_payload(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "labels": labels,
        "matrix": matrix.tolist(),
        "manageable_for_display": len(labels) <= 20,
    }


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


def _is_binary_problem(labels: list[str]) -> bool:
    return sorted(set(labels)) == ["manipulative", "non_manipulative"]


def _trainer_metrics(eval_prediction: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, label_ids = eval_prediction
    predicted_ids = logits.argmax(axis=-1)
    return {
        "accuracy": float(accuracy_score(label_ids, predicted_ids)),
        "macro_f1": float(f1_score(label_ids, predicted_ids, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(label_ids, predicted_ids, average="weighted", zero_division=0)
        ),
    }


def build_training_arguments(
    output_dir: str | Path,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to="none",
        seed=RANDOM_STATE,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        use_cpu=device == "cpu",
        dataloader_pin_memory=device == "cuda",
        optim="adamw_torch",
    )


def run_bert_classifier(
    df: pd.DataFrame,
    stage_name: str = "stage1_binary",
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = DEFAULT_MAX_LENGTH,
    num_train_epochs: int = DEFAULT_NUM_TRAIN_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    per_device_train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    rare_label_min_count: int | None = None,
    use_class_weights: bool = DEFAULT_USE_CLASS_WEIGHTS,
    output_dir: str | Path | None = None,
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

    encoder = build_label_encoder(pd.concat([train_df[LABEL_COLUMN], test_df[LABEL_COLUMN]]))
    y_train = train_df[LABEL_COLUMN].astype(str)
    y_test = test_df[LABEL_COLUMN].astype(str)
    train_label_ids = encode_labels(y_train, encoder)
    test_label_ids = encode_labels(y_test, encoder)
    device = detect_torch_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(
        train_df[text_column].tolist(),
        truncation=True,
        padding=False,
        max_length=max_length,
    )
    test_encodings = tokenizer(
        test_df[text_column].tolist(),
        truncation=True,
        padding=False,
        max_length=max_length,
    )

    train_dataset = TextClassificationDataset(train_encodings, train_label_ids)
    test_dataset = TextClassificationDataset(test_encodings, test_label_ids)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(encoder.label_to_id),
        id2label=encoder.id_to_label,
        label2id=encoder.label_to_id,
    )

    if output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "models" / f"{stage_name}_bert"

    training_args = build_training_arguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
    )

    class_weights = None
    if use_class_weights:
        class_weights = compute_balanced_class_weights(
            train_label_ids,
            num_labels=len(encoder.label_to_id),
        )
        trainer_class = WeightedLossTrainer
        trainer_kwargs = {"class_weights": class_weights}
    else:
        trainer_class = Trainer
        trainer_kwargs = {}

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=_trainer_metrics,
        **trainer_kwargs,
    )

    if device == "cpu":
        print(
            "Warning: torch MPS is unavailable, so BERT will run on CPU. "
            "On Apple Silicon, an arm64 Python 3.11/3.12 environment with a fresh torch install "
            "usually restores MPS acceleration."
        )

    trainer.train()
    prediction_output = trainer.predict(test_dataset)
    predicted_ids = prediction_output.predictions.argmax(axis=-1)
    predicted_labels = [encoder.id_to_label[int(index)] for index in predicted_ids]
    true_labels = y_test.tolist()

    report_df = classification_report_table(true_labels, predicted_labels)
    confusion_payload = confusion_matrix_payload(true_labels, predicted_labels)

    metrics = {
        "model": "bert_sequence_classifier",
        "stage": stage_name,
        "dataset_config": DEFAULT_CONFIG,
        "data_source": "MentalManip with exact text deduplication before split",
        "target_column": LABEL_COLUMN,
        "text_column": text_column,
        "random_seed": RANDOM_STATE,
        "test_size_fraction": TEST_SIZE,
        **split_metadata,
        "number_of_classes": int(pd.concat([y_train, y_test]).nunique()),
        "model_name": model_name,
        "tokenizer_name": model_name,
        "device": device,
        "max_length": int(max_length),
        "num_train_epochs": int(num_train_epochs),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "per_device_train_batch_size": int(per_device_train_batch_size),
        "per_device_eval_batch_size": int(per_device_eval_batch_size),
        "use_class_weights": bool(use_class_weights),
        "class_weights": (
            {
                encoder.id_to_label[index]: float(weight)
                for index, weight in enumerate(class_weights.tolist())
            }
            if class_weights is not None
            else None
        ),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(f1_score(true_labels, predicted_labels, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)
        ),
        "class_metrics": report_df.set_index("class").to_dict(orient="index"),
        "confusion_matrix": confusion_payload,
    }

    if _is_binary_problem(true_labels):
        metrics["precision"] = float(
            precision_score(
                true_labels,
                predicted_labels,
                pos_label="manipulative",
                zero_division=0,
            )
        )
        metrics["recall"] = float(
            recall_score(
                true_labels,
                predicted_labels,
                pos_label="manipulative",
                zero_division=0,
            )
        )
        metrics["f1"] = float(
            f1_score(
                true_labels,
                predicted_labels,
                pos_label="manipulative",
                zero_division=0,
            )
        )

    predictions = pd.DataFrame(
        {
            "true_label": true_labels,
            "predicted_label": predicted_labels,
        }
    )

    return metrics, predictions, report_df


def save_metrics(
    metrics: dict,
    output_path: str | Path = Path("results/metrics/bert_metrics.json"),
) -> Path:
    path = Path(output_path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def save_predictions(
    predictions: pd.DataFrame,
    output_path: str | Path = Path("results/predictions/bert_predictions.csv"),
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


def print_summary(metrics: dict) -> None:
    print("BERT Summary")
    print(f"Stage: {metrics['stage']}")
    print(f"Dataset config: {metrics['dataset_config']}")
    print(f"Model: {metrics['model_name']}")
    print(f"Device: {metrics['device']}")
    print(f"Total examples: {metrics['total_examples_after_cleaning']}")
    print(f"Train size / test size: {metrics['train_size']} / {metrics['test_size']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Class weights: {metrics['use_class_weights']}")
    if "precision" in metrics:
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")


def main() -> None:
    df = load_dataset(
        task=DEFAULT_TASK,
        config=DEFAULT_CONFIG,
        rare_label_min_count=None,
    )
    stage_name = "stage1_binary" if DEFAULT_TASK == "binary" else "stage2_technique"

    metrics, predictions, report_df = run_bert_classifier(
        df,
        stage_name=stage_name,
        model_name=DEFAULT_MODEL_NAME,
        max_length=DEFAULT_MAX_LENGTH,
        num_train_epochs=DEFAULT_NUM_TRAIN_EPOCHS,
        learning_rate=DEFAULT_LEARNING_RATE,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        per_device_train_batch_size=DEFAULT_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=DEFAULT_EVAL_BATCH_SIZE,
        rare_label_min_count=DEFAULT_RARE_LABEL_MIN_COUNT,
    )

    metrics_path = save_metrics(
        metrics,
        output_path=PROJECT_ROOT / "results" / "metrics" / f"{stage_name}_bert_metrics.json",
    )
    predictions_path = save_predictions(
        predictions,
        output_path=PROJECT_ROOT / "results" / "predictions" / f"{stage_name}_bert_predictions.csv",
    )
    report_path = save_classification_report(
        report_df,
        output_path=PROJECT_ROOT / "results" / "metrics" / f"{stage_name}_bert_report.csv",
    )

    print_summary(metrics)
    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved class report to: {report_path}")

    if metrics["confusion_matrix"]["manageable_for_display"]:
        confusion_path = save_confusion_matrix_plot(
            metrics["confusion_matrix"],
            output_path=PROJECT_ROOT / "figures" / f"{stage_name}_bert_confusion_matrix.png",
            title=f"{stage_name} BERT Confusion Matrix",
        )
        print(f"Saved confusion matrix plot to: {confusion_path}")
    else:
        print("Skipped confusion matrix plot because the label count was too large.")


if __name__ == "__main__":
    main()
