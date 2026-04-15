from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_FILE = PROJECT_ROOT / "data/raw/manipulational_conversation.json"
DEDUPED_DATA_FILE = PROJECT_ROOT / "data/processed/deduped_conversations.csv"


def load_raw_json(file_path: str | Path) -> list[dict[str, Any]]:

    # Load a raw dataset
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

    # flatten message dictionaries into one speaker

    if not messages:
        return "No message enter a message"

    lines = []

    for message in messages:

        speaker = str(message.get("speaker", "Unknown")).strip() or "Unknown"

        text = str(message.get("text", "")).strip()

        if text:

            lines.append(f"{speaker}: {text}")

    return "\n".join(lines)

#building data
def build_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
   

    rows = []
    for record in records:

        row = {column: record.get(column) for column in OUTPUT_COLUMNS if column != "text"}

        row["text"] = flatten_messages(record.get("messages"))

        rows.append(row)

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def load_dataset() -> pd.DataFrame:
    
    records = load_raw_json(RAW_DATA_FILE)
    
    return build_dataframe(records)


def deduplicate_conversations(df: pd.DataFrame) -> pd.DataFrame:

    return df.drop_duplicates(subset=["text"]).reset_index(drop=True)


def load_deduped_dataset() -> pd.DataFrame:

    return deduplicate_conversations(load_dataset())


def save_deduped_dataset() -> Path:

    df = load_deduped_dataset()
    DEDUPED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEDUPED_DATA_FILE, index=False)
    return DEDUPED_DATA_FILE


def find_default_raw_file() -> Path:

    return RAW_DATA_FILE


def main() -> None:

    input_path = find_default_raw_file()
    df = load_dataset()

    print(f"Loaded file: {input_path}")

    print(f"Number of rows: {len(df)}")

    deduped_df = deduplicate_conversations(df)
    deduped_path = save_deduped_dataset()

    print(f"Unique text rows after deduplication: {len(deduped_df)}")
    print(f"Removed exact duplicate text rows: {len(df) - len(deduped_df)}")
    print(f"Saved deduped dataset to: {deduped_path}")

    print(f"Columns: {list(df.columns)}")

    print("\nFirst 3 processed examples:")

    preview_columns = ["conversation_id", "manipulation_type", "text"]

    with pd.option_context("display.max_colwidth", 500, "display.width", 120):

        print(df[preview_columns].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
