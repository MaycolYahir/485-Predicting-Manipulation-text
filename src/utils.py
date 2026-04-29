"""Small shared utilities for project scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


LABEL_COLUMN = "manipulation_type"
TEXT_COLUMN = "text"
RANDOM_STATE = 42
TEST_SIZE = 0.2
OTHER_LABEL = "OTHER"


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def prepare_modeling_dataframe(
    df: pd.DataFrame,
    label_column: str = LABEL_COLUMN,
    text_column: str = TEXT_COLUMN,
) -> tuple[pd.DataFrame, int, int]:
    """Clean labels/text and drop exact duplicate texts before splitting."""
    clean_df = df.dropna(subset=[label_column, text_column]).copy()
    clean_df[label_column] = clean_df[label_column].astype(str).str.strip()
    clean_df[text_column] = clean_df[text_column].astype(str).str.strip()
    clean_df = clean_df[(clean_df[label_column] != "") & (clean_df[text_column] != "")].copy()

    original_count = len(clean_df)
    duplicate_text_rows = int(clean_df.duplicated(subset=[text_column]).sum())
    if duplicate_text_rows:
        clean_df = clean_df.drop_duplicates(subset=[text_column]).copy()

    return clean_df, original_count, duplicate_text_rows


def collapse_rare_labels(
    df: pd.DataFrame,
    min_count: int,
    label_column: str = LABEL_COLUMN,
    replacement_label: str = OTHER_LABEL,
) -> tuple[pd.DataFrame, list[str]]:
    """Collapse labels with fewer than min_count examples into OTHER."""
    if min_count < 2:
        raise ValueError("min_count must be at least 2 when collapsing rare labels.")

    label_counts = df[label_column].value_counts()
    rare_labels = sorted(label_counts[label_counts < min_count].index.astype(str).tolist())
    if not rare_labels:
        return df.copy(), rare_labels

    collapsed_df = df.copy()
    collapsed_df[label_column] = collapsed_df[label_column].where(
        ~collapsed_df[label_column].isin(rare_labels),
        replacement_label,
    )
    return collapsed_df, rare_labels


def split_for_modeling(
    df: pd.DataFrame,
    label_column: str = LABEL_COLUMN,
    text_column: str = TEXT_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    rare_label_min_count: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Prepare a DataFrame for modeling and run the project's stratified split.

    If rare_label_min_count is provided, labels with fewer than that many examples
    are collapsed into OTHER before splitting. This is mainly for stage-2 technique
    labels when exact label combinations are too sparse for a stratified split.
    """
    clean_df, original_count, duplicate_text_rows = prepare_modeling_dataframe(
        df,
        label_column=label_column,
        text_column=text_column,
    )

    rare_labels_collapsed: list[str] = []
    if rare_label_min_count is not None:
        clean_df, rare_labels_collapsed = collapse_rare_labels(
            clean_df,
            min_count=rare_label_min_count,
            label_column=label_column,
        )

    label_counts = clean_df[label_column].value_counts()
    if label_counts.empty:
        raise ValueError("No rows remain after cleaning text and labels.")

    min_label_count = int(label_counts.min())
    if min_label_count < 2:
        raise ValueError(
            "Stratified split requires at least 2 examples per class after cleaning. "
            f"Smallest class has {min_label_count} example(s). "
            "For stage 2, consider enabling rare-label collapse into OTHER."
        )

    train_df, test_df = train_test_split(
        clean_df,
        test_size=test_size,
        random_state=random_state,
        stratify=clean_df[label_column],
    )

    split_metadata = {
        "total_examples_before_deduplication": int(original_count),
        "duplicate_text_rows_removed_before_split": int(duplicate_text_rows),
        "total_examples_after_cleaning": int(len(clean_df)),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "rare_label_min_count": rare_label_min_count,
        "rare_labels_collapsed": rare_labels_collapsed,
        "number_of_collapsed_labels": int(len(rare_labels_collapsed)),
        "number_of_classes_after_cleaning": int(clean_df[label_column].nunique()),
        "smallest_class_size_after_cleaning": min_label_count,
    }

    return train_df, test_df, split_metadata
