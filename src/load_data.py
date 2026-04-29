"""
load_data.py  —  MentalManip (HuggingFace) version
Replaces the old Kaggle JSON loader.

Dataset: audreyeleven/MentalManip
  Subsets available: "mentalmanip_maj"  (4 000 rows, majority-vote labels)  ← recommended
                     "mentalmanip_con"  (2 920 rows, consensus labels)
                     "mentalmanip_detailed" (4 000 rows, raw per-annotator data)

Output columns (matches what majority_baseline.py and naivebayes.py expect):
  text              – the dialogue string (renamed from 'dialogue')
  manipulation_type – the label we classify on (see TASK below)
  manipulative      – raw binary flag kept for reference (1 = manipulative)
  technique         – raw comma-separated technique string kept for reference
  vulnerability     – raw comma-separated vulnerability string kept for reference

TASK options (set TASK constant below):
  "binary"    → manipulation_type = "manipulative" / "non_manipulative"   (2 classes)
  "technique" → manipulation_type = first/primary technique on manipulative
                examples only; non-manipulative rows are dropped            (N classes)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from datasets import load_dataset as hf_load_dataset

# ── Configuration ─────────────────────────────────────────────────────────────

SUBSET   = "mentalmanip_maj"   # or "mentalmanip_con"
TASK: Literal["binary", "technique"] = "binary"

# For TASK="technique": drop techniques with fewer than this many examples
# so the classifier has enough signal per class.
MIN_TECHNIQUE_SUPPORT = 50

OUTPUT_COLUMNS = ["text", "manipulation_type", "manipulative", "technique", "vulnerability"]

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
DEDUPED_DATA_FILE = PROJECT_ROOT / "data" / "processed" / "deduped_conversations.csv"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_raw_dataframe(subset: str = SUBSET) -> pd.DataFrame:
    """Download (or use cached) MentalManip from HuggingFace and return raw DataFrame."""
    hf_ds = hf_load_dataset("audreyeleven/MentalManip", subset)
    # The dataset only ships a 'train' split — we do our own 80/20 split later.
    df = hf_ds["train"].to_pandas()
    return df


def _primary_technique(technique_str: str | None) -> str | None:
    """Return the first listed technique from a comma-separated string."""
    if not technique_str or pd.isna(technique_str):
        return None
    parts = [t.strip() for t in str(technique_str).split(",") if t.strip()]
    return parts[0] if parts else None


def build_dataframe(
    raw_df: pd.DataFrame,
    task: Literal["binary", "technique"] = TASK,
    min_technique_support: int = MIN_TECHNIQUE_SUPPORT,
) -> pd.DataFrame:
    """
    Transform the raw HuggingFace DataFrame into the shape expected by
    majority_baseline.py and naivebayes.py.

    LABEL_COLUMN used downstream: "manipulation_type"
    TEXT_COLUMN  used downstream: "text"
    """
    df = raw_df.copy()

    # Rename to match downstream expectations
    df = df.rename(columns={"Dialogue": "text", "dialogue": "text"})  # handle either casing
    # HuggingFace col names for mentalmanip_maj/con: id, Dialogue, Manipulative, Technique, Vulnerability
    # Lowercase everything for safety
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"dialogue": "text"})

    # Keep raw columns for reference
    df["manipulative"] = df["manipulative"].astype(int)
    df["technique"]    = df["technique"].fillna("").astype(str)
    df["vulnerability"] = df.get("vulnerability", pd.Series([""] * len(df))).fillna("").astype(str)

    if task == "binary":
        df["manipulation_type"] = df["manipulative"].map(
            {1: "manipulative", 0: "non_manipulative"}
        )

    elif task == "technique":
        # Only keep manipulative examples that have at least one technique label
        df = df[df["manipulative"] == 1].copy()
        df = df[df["technique"] != ""].copy()

        df["manipulation_type"] = df["technique"].apply(_primary_technique)
        df = df.dropna(subset=["manipulation_type"])

        # Drop rare techniques so the classifier has enough signal
        counts = df["manipulation_type"].value_counts()
        valid_techniques = counts[counts >= min_technique_support].index
        df = df[df["manipulation_type"].isin(valid_techniques)].copy()

        if df.empty:
            raise ValueError(
                f"No techniques have >= {min_technique_support} examples. "
                "Lower MIN_TECHNIQUE_SUPPORT or use TASK='binary'."
            )

    else:
        raise ValueError(f"Unknown TASK: {task!r}. Choose 'binary' or 'technique'.")

    # Ensure text is a clean string
    df["text"] = df["text"].fillna("").astype(str)

    return df[OUTPUT_COLUMNS].reset_index(drop=True)


def load_dataset() -> pd.DataFrame:
    """Main entry point used by majority_baseline.py and naivebayes.py."""
    raw_df = load_raw_dataframe()
    return build_dataframe(raw_df)


def deduplicate_conversations(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["text"]).reset_index(drop=True)


def load_deduped_dataset() -> pd.DataFrame:
    return deduplicate_conversations(load_dataset())


def save_deduped_dataset() -> Path:
    df = load_deduped_dataset()
    DEDUPED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEDUPED_DATA_FILE, index=False)
    return DEDUPED_DATA_FILE


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading MentalManip subset='{SUBSET}', task='{TASK}' ...")
    df = load_dataset()

    print(f"Total rows after processing : {len(df)}")
    print(f"Columns                     : {list(df.columns)}")
    print(f"\nClass distribution (manipulation_type):")
    print(df["manipulation_type"].value_counts().to_string())

    print("\nFirst 3 examples:")
    for _, row in df[["manipulation_type", "text"]].head(3).iterrows():
        print("---")
        print(f"label : {row['manipulation_type']}")
        print(f"text  : {str(row['text'])[:300]}")

    path = save_deduped_dataset()
    print(f"\nSaved deduped dataset to: {path}")


if __name__ == "__main__":
    main()