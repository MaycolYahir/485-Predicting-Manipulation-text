"""Exploratory data analysis helpers for the manipulation dataset."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

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

try:
    from .load_data import find_default_raw_file, load_dataset
    from .utils import ensure_dir, project_path
except ImportError:
    from load_data import find_default_raw_file, load_dataset
    from utils import ensure_dir, project_path


def total_conversations(df: pd.DataFrame) -> int:

    """Return the total number of conversations."""

    return len(df)


def manipulation_type_counts(df: pd.DataFrame) -> pd.Series:
    """Return counts for each manipulation_type label."""
    return df["manipulation_type"].value_counts(dropna=False)


def manipulation_proportions(df: pd.DataFrame) -> pd.Series:
    """Return proportions for manipulated vs non-manipulated examples."""
    return df["is_manipulation"].value_counts(normalize=True, dropna=False)


def average_conversation_length(df: pd.DataFrame) -> float:
    """Return the average conversation_length."""
    return float(pd.to_numeric(df["conversation_length"], errors="coerce").mean())


def average_word_count_total(df: pd.DataFrame) -> float:
    """Return the average total word count per conversation."""
    return float(pd.to_numeric(df["word_count_total"], errors="coerce").mean())


def average_messages_per_conversation(df: pd.DataFrame) -> float:
    """Estimate the average number of messages per conversation from flattened text."""
    if "text" not in df:
        return average_conversation_length(df)

    message_counts = df["text"].fillna("").apply(
        lambda text: sum(1 for line in str(text).splitlines() if line.strip())
    )
    if message_counts.sum() == 0 and "conversation_length" in df:
        return average_conversation_length(df)

    return float(message_counts.mean())


def average_text_length_chars(df: pd.DataFrame) -> float:
    """Return the average flattened text length in characters."""
    return float(df["text"].fillna("").astype(str).str.len().mean())


def data_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact summary table for report and result files."""
    label_counts = manipulation_type_counts(df)
    proportions = manipulation_proportions(df)

    summary_rows = [
        ("total_conversations", total_conversations(df)),
        ("average_conversation_length", average_conversation_length(df)),
        ("average_word_count_total", average_word_count_total(df)),
        ("average_messages_per_conversation", average_messages_per_conversation(df)),
        ("average_text_length_characters", average_text_length_chars(df)),
    ]

    for label, count in label_counts.items():
        
        summary_rows.append((f"count_manipulation_type_{label}", int(count)))

    for label, proportion in proportions.items():

        summary_rows.append((f"proportion_is_manipulation_{label}", float(proportion)))

    return pd.DataFrame(summary_rows, columns=["metric", "value"])


def save_summary_table(
        
    df: pd.DataFrame,
    output_path: str | Path = project_path("results", "metrics", "data_summary.csv"),
) -> Path:
    """Save the summary table to CSV."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    data_summary_table(df).to_csv(output_path, index=False)
    return output_path


def plot_label_distribution(
    df: pd.DataFrame,
    output_path: str | Path = project_path("figures", "label_distribution.png"),
) -> Path:
    """Create and save a bar chart of manipulation_type counts."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    counts = manipulation_type_counts(df).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="bar", ax=ax, color="#4C78A8", edgecolor="black")
    ax.set_title("Distribution of Manipulation Types")
    ax.set_xlabel("Manipulation Type")
    ax.set_ylabel("Number of Conversations")
    ax.tick_params(axis="x", rotation=45)

    for label in ax.get_xticklabels():
        label.set_ha("right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main() -> None:
    input_path = find_default_raw_file()
    df = load_dataset()

    summary_path = save_summary_table(df)
    figure_path = plot_label_distribution(df)

    print(f"Loaded file: {input_path}")
    print(f"Total conversations: {total_conversations(df)}")
    print("\nManipulation type counts:")
    print(manipulation_type_counts(df).to_string())
    print("\nData summary:")
    print(data_summary_table(df).to_string(index=False))
    print(f"\nSaved summary table to: {summary_path}")
    print(f"Saved label distribution chart to: {figure_path}")


if __name__ == "__main__":
    main()
