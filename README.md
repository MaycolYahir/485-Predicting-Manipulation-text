# Detecting Psychological Manipulation in MentalManip Dialogues

COMPSCI 485 NLP final project using the Hugging Face `audreyeleven/MentalManip`
dataset with a two-stage text classification pipeline.


## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data

The project now loads data directly from Hugging Face with:

```python
from datasets import load_dataset
ds = load_dataset("audreyeleven/MentalManip", "mentalmanip_maj")
```

The active config is `mentalmanip_maj`, which is the majority-vote version.

## Run The Loader

```bash
python3 -m src.load_data
```

The loader prints dataset stats including:

- total rows
- manipulative vs non-manipulative counts
- stage-2 row count
- class imbalance
- top 20 exact technique labels

## Modeling Plan

The project keeps the original text-only baselines and reuses the existing
train/test split structure:

- Stage 1: binary classification
  `Dialogue` -> `Manipulative`
- Stage 2: exact technique-string classification on manipulative rows only
  `Dialogue` -> full `Technique` string

Current implemented baselines:

- majority-class baseline
- TF-IDF + Multinomial Naive Bayes

## Run The Notebooks

```bash
./.venv/bin/jupyter lab
```

Main notebooks:

- `notebooks/01_eda.ipynb`
  MentalManip dataset stats, previews, and top-technique plots
- `notebooks/02_text_only_models.ipynb`
  Stage 1 binary results and stage 2 exact-technique results

The notebooks:

- load `mentalmanip_maj`
- run stage 1 and stage 2 in notebook cells
- save notebook-local metrics and predictions under `notebooks/results/`
- save notebook-local figures under `notebooks/figures/`

Because the dataset is synthetic/annotated dialogue data, results should be
discussed as performance on this benchmark rather than as guaranteed real-world
manipulation detection.
