from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from .load_data import (
        DEFAULT_CONFIG,
        build_technique_dataframe,
        load_raw_dataframe,
        normalize_raw_dataframe,
        print_dataset_stats,
        summarize_dataset,
    )
    from .utils import ensure_dir
except ImportError:
    from load_data import (
        DEFAULT_CONFIG,
        build_technique_dataframe,
        load_raw_dataframe,
        normalize_raw_dataframe,
        print_dataset_stats,
        summarize_dataset,
    )
    from utils import ensure_dir


def data_summary_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    stats = summarize_dataset(raw_df)
    summary_rows = [
        ("config", stats["config"]),
        ("total_rows", stats["total_rows"]),
        ("manipulative_rows", stats["manipulative_rows"]),
        ("non_manipulative_rows", stats["non_manipulative_rows"]),
        ("stage2_rows", stats["stage2_rows"]),
        ("stage2_num_classes", stats["stage2_num_classes"]),
        (
            "stage1_imbalance_ratio_majority_to_minority",
            stats["stage1_imbalance_ratio_majority_to_minority"],
        ),
        ("smallest_stage2_class_size", stats["smallest_stage2_class_size"]),
        ("largest_stage2_class_size", stats["largest_stage2_class_size"]),
    ]

    for label, count in stats["top_20_technique_labels"].items():
        summary_rows.append((f"top_technique_count::{label}", count))

    return pd.DataFrame(summary_rows, columns=["metric", "value"])


def save_summary_table(
    raw_df: pd.DataFrame,
    output_path: str | Path = Path("results/metrics/mentalmanip_data_summary.csv"),
) -> Path:
    path = Path(output_path)
    ensure_dir(path.parent)
    data_summary_table(raw_df).to_csv(path, index=False)
    return path


def save_top_techniques(
    raw_df: pd.DataFrame,
    output_path: str | Path = Path("results/metrics/mentalmanip_top_techniques.csv"),
) -> Path:
    normalized_df = normalize_raw_dataframe(raw_df)
    stage2_df = build_technique_dataframe(normalized_df)
    technique_counts = stage2_df["manipulation_type"].value_counts().rename_axis("technique").reset_index(
        name="count"
    )

    path = Path(output_path)
    ensure_dir(path.parent)
    technique_counts.to_csv(path, index=False)
    return path


def save_stats_json(
    raw_df: pd.DataFrame,
    output_path: str | Path = Path("results/metrics/mentalmanip_dataset_stats.json"),
) -> Path:
    path = Path(output_path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(summarize_dataset(raw_df), indent=2), encoding="utf-8")
    return path


def main() -> None:
    raw_df = load_raw_dataframe(config=DEFAULT_CONFIG)
    print_dataset_stats(raw_df)

    summary_path = save_summary_table(raw_df)
    top_techniques_path = save_top_techniques(raw_df)
    stats_path = save_stats_json(raw_df)

    print(f"\nSaved summary table to: {summary_path}")
    print(f"Saved top-techniques table to: {top_techniques_path}")
    print(f"Saved stats JSON to: {stats_path}")


if __name__ == "__main__":
    main()
