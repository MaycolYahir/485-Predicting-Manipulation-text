"""Load and prepare the MentalManip dataset for the project pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from datasets import load_dataset as hf_load_dataset


DATASET_NAME = "audreyeleven/MentalManip"
DEFAULT_CONFIG = "mentalmanip_maj"
LABEL_COLUMN = "manipulation_type"
TEXT_COLUMN = "text"
OUTPUT_COLUMNS = [TEXT_COLUMN, LABEL_COLUMN, "manipulative", "technique", "vulnerability"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _clean_text_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def load_raw_dataframe(config: str = DEFAULT_CONFIG) -> pd.DataFrame:
    """Download or load the requested MentalManip config from Hugging Face."""
    dataset = hf_load_dataset(DATASET_NAME, config)
    if "train" not in dataset:
        raise ValueError(f"Expected a 'train' split in {DATASET_NAME}:{config}.")
    return dataset["train"].to_pandas()


def normalize_raw_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Rename and clean the raw MentalManip columns used by the project."""
    df = raw_df.copy()
    df = df.rename(
        columns={
            "Dialogue": TEXT_COLUMN,
            "dialogue": TEXT_COLUMN,
            "Manipulative": "manipulative",
            "manipulative": "manipulative",
            "Technique": "technique",
            "technique": "technique",
            "Vulnerability": "vulnerability",
            "vulnerability": "vulnerability",
        }
    )

    required_columns = {TEXT_COLUMN, "manipulative", "technique"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required MentalManip columns: {missing_columns}")

    df[TEXT_COLUMN] = _clean_text_series(df[TEXT_COLUMN])
    df["technique"] = _clean_text_series(df["technique"])

    if "vulnerability" not in df.columns:
        df["vulnerability"] = ""
    df["vulnerability"] = _clean_text_series(df["vulnerability"])

    manipulative = pd.to_numeric(df["manipulative"], errors="coerce").fillna(0).astype(int)
    df["manipulative"] = manipulative.clip(lower=0, upper=1)

    return df[[TEXT_COLUMN, "manipulative", "technique", "vulnerability"]].copy()


def build_binary_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Stage 1 dataset: classify manipulative vs non-manipulative."""
    df = normalize_raw_dataframe(raw_df)
    df[LABEL_COLUMN] = df["manipulative"].map({1: "manipulative", 0: "non_manipulative"})
    return df[OUTPUT_COLUMNS].reset_index(drop=True)


def build_technique_dataframe(
    raw_df: pd.DataFrame,
    rare_label_min_count: int | None = None,
) -> pd.DataFrame:
    """
    Stage 2 dataset: classify the full exact Technique string.

    Rare-label collapsing is optional and only used when explicitly requested.
    """
    df = normalize_raw_dataframe(raw_df)
    df = df[df["manipulative"] == 1].copy()
    df = df[df["technique"] != ""].copy()
    df[LABEL_COLUMN] = df["technique"]

    if rare_label_min_count is not None:
        technique_counts = df[LABEL_COLUMN].value_counts()
        rare_labels = technique_counts[technique_counts < rare_label_min_count].index
        if len(rare_labels):
            df[LABEL_COLUMN] = df[LABEL_COLUMN].where(
                ~df[LABEL_COLUMN].isin(rare_labels),
                "OTHER",
            )

    return df[OUTPUT_COLUMNS].reset_index(drop=True)


def build_dataframe(
    raw_df: pd.DataFrame,
    task: Literal["binary", "technique"] = "binary",
    rare_label_min_count: int | None = None,
) -> pd.DataFrame:
    """Build the stage-specific project DataFrame from raw MentalManip data."""
    if task == "binary":
        return build_binary_dataframe(raw_df)
    if task == "technique":
        return build_technique_dataframe(raw_df, rare_label_min_count=rare_label_min_count)
    raise ValueError(f"Unknown task: {task!r}")


def load_dataset(
    task: Literal["binary", "technique"] = "binary",
    config: str = DEFAULT_CONFIG,
    rare_label_min_count: int | None = None,
) -> pd.DataFrame:
    """Main entry point for project scripts."""
    raw_df = load_raw_dataframe(config=config)
    return build_dataframe(
        raw_df,
        task=task,
        rare_label_min_count=rare_label_min_count,
    )


def deduplicate_conversations(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate dialogue texts."""
    return df.drop_duplicates(subset=[TEXT_COLUMN]).reset_index(drop=True)


def dataset_snapshot_path(task: str, config: str = DEFAULT_CONFIG) -> Path:
    return PROJECT_ROOT / "data" / "processed" / f"{config}_{task}_deduped_conversations.csv"


def save_dataset_snapshot(
    task: Literal["binary", "technique"] = "binary",
    config: str = DEFAULT_CONFIG,
    rare_label_min_count: int | None = None,
) -> Path:
    """Save the deduplicated stage-specific dataset used by the pipeline."""
    df = load_dataset(
        task=task,
        config=config,
        rare_label_min_count=rare_label_min_count,
    )
    path = dataset_snapshot_path(task=task, config=config)
    path.parent.mkdir(parents=True, exist_ok=True)
    deduplicate_conversations(df).to_csv(path, index=False)
    return path


def summarize_dataset(raw_df: pd.DataFrame) -> dict:
    """Compute dataset stats for reporting and debugging."""
    df = normalize_raw_dataframe(raw_df)
    manipulative_counts = df["manipulative"].value_counts().sort_index()
    manipulative_rows = int(manipulative_counts.get(1, 0))
    non_manipulative_rows = int(manipulative_counts.get(0, 0))

    stage2_df = df[(df["manipulative"] == 1) & (df["technique"] != "")].copy()
    technique_counts = stage2_df["technique"].value_counts()

    majority_class_size = int(manipulative_counts.max()) if not manipulative_counts.empty else 0
    minority_class_size = int(manipulative_counts.min()) if not manipulative_counts.empty else 0
    imbalance_ratio = (
        float(majority_class_size / minority_class_size)
        if minority_class_size
        else float("inf")
    )

    return {
        "config": DEFAULT_CONFIG,
        "total_rows": int(len(df)),
        "manipulative_rows": manipulative_rows,
        "non_manipulative_rows": non_manipulative_rows,
        "stage2_rows": int(len(stage2_df)),
        "stage2_num_classes": int(technique_counts.shape[0]),
        "stage1_class_counts": {
            "non_manipulative": non_manipulative_rows,
            "manipulative": manipulative_rows,
        },
        "stage1_imbalance_ratio_majority_to_minority": imbalance_ratio,
        "top_20_technique_labels": {
            str(label): int(count)
            for label, count in technique_counts.head(20).items()
        },
        "smallest_stage2_class_size": int(technique_counts.min()) if not technique_counts.empty else 0,
        "largest_stage2_class_size": int(technique_counts.max()) if not technique_counts.empty else 0,
    }


def print_dataset_stats(raw_df: pd.DataFrame) -> dict:
    """Print the required dataset stats and return them as a dict."""
    stats = summarize_dataset(raw_df)

    print(f"MentalManip config: {stats['config']}")
    print(f"Total rows: {stats['total_rows']}")
    print(
        "Manipulative vs non-manipulative rows: "
        f"{stats['manipulative_rows']} / {stats['non_manipulative_rows']}"
    )
    print(f"Stage-2 technique rows: {stats['stage2_rows']}")
    print(
        "Stage-1 class imbalance ratio (majority/minority): "
        f"{stats['stage1_imbalance_ratio_majority_to_minority']:.3f}"
    )
    print(f"Stage-2 number of exact technique labels: {stats['stage2_num_classes']}")
    print(
        "Stage-2 class size range: "
        f"{stats['smallest_stage2_class_size']} to {stats['largest_stage2_class_size']}"
    )
    print("\nTop 20 technique labels:")
    for label, count in stats["top_20_technique_labels"].items():
        print(f"{label}: {count}")

    if stats["smallest_stage2_class_size"] < 2:
        print(
            "\nWarning: some exact technique labels are extremely rare. "
            "If stratified splitting fails, enable the optional rare-label "
            "collapse step to map tiny classes to OTHER."
        )

    return stats


def main() -> None:
    raw_df = load_raw_dataframe(config=DEFAULT_CONFIG)
    print_dataset_stats(raw_df)

    binary_df = build_binary_dataframe(raw_df)
    technique_df = build_technique_dataframe(raw_df)
    binary_path = save_dataset_snapshot(task="binary", config=DEFAULT_CONFIG)
    technique_path = save_dataset_snapshot(task="technique", config=DEFAULT_CONFIG)

    print(f"\nBinary rows after preprocessing: {len(binary_df)}")
    print(f"Technique rows after preprocessing: {len(technique_df)}")
    print(f"Saved stage-1 snapshot to: {binary_path}")
    print(f"Saved stage-2 snapshot to: {technique_path}")


if __name__ == "__main__":
    main()
