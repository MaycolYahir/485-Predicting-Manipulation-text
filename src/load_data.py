"""Load and flatten the raw manipulation conversation dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .utils import project_path
except ImportError:
    from utils import project_path


OUTPUT_COLUMNS = [
    "conversation_id",
    "text",
    "manipulation_type",
    "is_manipulation",
    "context_type",
    "conversation_length",
    "word_count_total",
    "question_count",
    "denial_count",
]


def load_raw_json(file_path: str | Path) -> list[dict[str, Any]]:
    """Load a raw dataset file stored as JSONL or a JSON array."""
    path = Path(file_path)
    raw_text = path.read_text(encoding="utf-8").strip()

    if not raw_text:
        return []

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(raw_text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
        return records

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]

    raise ValueError(f"Expected a JSON array, JSON object, or JSONL records in {path}")


def flatten_messages(messages: list[dict[str, Any]] | None) -> str:
    """Flatten message dictionaries into one speaker-prefixed text string."""
    if not messages:
        return ""

    lines = []
    for message in messages:
        speaker = str(message.get("speaker", "Unknown")).strip() or "Unknown"
        text = str(message.get("text", "")).strip()
        if text:
            lines.append(f"{speaker}: {text}")

    return "\n".join(lines)


def build_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the project DataFrame with flattened text and selected columns."""
    rows = []
    for record in records:
        row = {column: record.get(column) for column in OUTPUT_COLUMNS if column != "text"}
        row["text"] = flatten_messages(record.get("messages"))
        rows.append(row)

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load the raw dataset file and return a processed pandas DataFrame."""
    records = load_raw_json(file_path)
    return build_dataframe(records)


def find_default_raw_file() -> Path:
    """Find the first JSON or JSONL file in data/raw."""
    raw_dir = project_path("data", "raw")
    candidates = sorted(raw_dir.glob("*.json")) + sorted(raw_dir.glob("*.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"No .json or .jsonl files found in {raw_dir}. "
            "Place the Kaggle file in data/raw/ or pass --input."
        )
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and flatten the raw conversation dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to the raw JSON or JSONL file. Defaults to the first file in data/raw/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input is not None else find_default_raw_file()
    df = load_dataset(input_path)

    print(f"Loaded file: {input_path}")
    print(f"Number of rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 3 processed examples:")

    preview_columns = ["conversation_id", "manipulation_type", "text"]
    with pd.option_context("display.max_colwidth", 500, "display.width", 120):
        print(df[preview_columns].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
