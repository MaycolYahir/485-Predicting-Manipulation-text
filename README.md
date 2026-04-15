# Detecting Psychological Manipulation Types in Synthetic Conversations

COMPSCI 485 NLP final project for multiclass classification of `manipulation_type` from synthetic conversation text.


## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data

Download the Kaggle dataset and place the raw file in `data/raw/`.

The loader supports either:

- JSONL: one JSON object per line
- JSON array: one list of JSON objects

## Run The Loader

The loader uses the hardcoded raw dataset path `data/raw/manipulational_conversation.json`:

```bash
python src/load_data.py
```

If your shell does not define `python`, use `python3` for these commands.

The script prints the number of rows, the processed columns, and the first three flattened examples.

## Modeling Plan

The main experiment should use text-only classification from flattened messages. Metadata-only and text-plus-metadata models can be useful comparisons, but some metadata fields may leak the label and make the task artificially easy.

Because the dataset is synthetic, results should be discussed as performance on generated conversations rather than guaranteed real-world manipulation detection.
