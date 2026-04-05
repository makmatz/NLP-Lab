# Sentiment Analysis using NLP Models

A lab project exploring sentiment analysis with classical NLP models, word embeddings, and pre-trained Transformers on two datasets: **MR** (Movie Reviews) and **Semeval2017A** (Twitter sentiment).

## Project Structure

```
├── main.py                  # Training/evaluation with word embeddings + custom models
├── run.sh                   # Script to run all word embedding experiments (Q1-Q5)
├── main_prep.py             # Data preprocessing and embedding preparation
├── finetune_pretrained.py   # Fine-tuning pre-trained Transformers (run on GPU/Colab)
├── transfer_pretrained.py   # Zero-shot evaluation with HuggingFace pipelines
├── models.py                # Custom model definitions (BaselineDNN, LSTM)
├── training.py              # Training loop utilities
├── dataloading.py           # DataLoader definitions
├── attention.py             # Attention, MultiHead, and Transformer Encoder models
├── early_stopper.py         # Early stopping callback
├── config.py                # Paths configuration
├── utils/
│   ├── load_datasets.py     # Dataset loading utilities
│   └── load_embeddings.py   # Embedding loading utilities
├── embeddings/              # Pre-trained word embeddings (not tracked)
├── datasets/                # Raw datasets (not tracked)
└── report/                  # LaTeX report and results
```

## Prerequisites

Python 3.8+ is required.

### 1 - Create a Virtual Environment

Using conda (recommended):

```bash
conda create -n nlp-lab python=3.10
conda activate nlp-lab
```

Or using virtualenv:

```bash
python -m venv venv
source venv/bin/activate
```

### 2 - Install PyTorch

Follow the instructions at [pytorch.org](https://pytorch.org/) to install the version matching your system and CUDA setup.

### 3 - Install Requirements

```bash
pip install -r requirements.txt
```

### 4 - Download Pre-trained Word Embeddings

Place embedding files in the `embeddings/` folder. Supported embeddings:

| Embedding | Description | Dimensions |
|-----------|-------------|------------|
| [GloVe 6B](http://nlp.stanford.edu/data/glove.6B.zip) | Generic English | 50d, 100d, 200d, 300d |
| [GloVe Twitter 27B](http://nlp.stanford.edu/data/glove.twitter.27B.zip) | Twitter-specific | 25d, 50d, 100d, 200d |
| [fastText](https://fasttext.cc/docs/en/english-vectors.html) | Generic English | 300d |

## Running the Project

### Word Embedding Models (Q1–Q5)

Run all experiments at once using the provided script:

```bash
bash run.sh
```

Or run a specific model manually:

```bash
python main.py --model <model> --dataset <dataset> [--n_head N] [--n_layer N] [--no_show]
```

Available models: `baseline_mean`, `baseline_mean_max`, `lstm`, `attention`, `multihead`, `transformer`  
Available datasets: `MR`, `Semeval2017A`

Results are appended to `report/results.txt` and loss curves are saved under `curves/`.

### Zero-Shot Evaluation with HuggingFace Pipelines (Q6)

```bash
python transfer_pretrained.py
```

### Fine-Tuning Pre-trained Transformers (Q7)

Intended to run on GPU. Recommended: Google Colab with T4 or A100.

```bash
python finetune_pretrained.py
```

Models: `bert-base-cased`, `roberta-base`, `distilbert-base-uncased`  
Datasets: MR, Semeval2017A

## Results Summary

### Word Embedding Models — MR (Test Set)

| Model | Accuracy | F1 (macro) |
|-------|----------|------------|
| Baseline DNN (mean) | 0.7281 | 0.7279 |
| Baseline DNN (mean+max) | 0.7462 | 0.7454 |
| BiLSTM | 0.7568 | 0.7546 |
| Self-Attention | 0.7492 | 0.7489 |
| MultiHead Attention (4 heads) | 0.7477 | 0.7475 |
| MultiHead Attention (8 heads) | 0.7402 | 0.7401 |
| Transformer Encoder (4h, 3L) | 0.7492 | 0.7462 |
| Transformer Encoder (8h, 6L) | 0.7477 | 0.7456 |
| Transformer Encoder (4h, 6L) | 0.7704 | 0.7699 |

### Zero-Shot Pre-trained Transformers (Q6)

| Model | MR Accuracy | Semeval Accuracy |
|-------|-------------|------------------|
| siebert/sentiment-roberta-large-english | 0.9260 | — |
| textattack/bert-base-uncased-SST-2 | 0.8988 | — |
| distilbert-base-uncased-finetuned-sst-2-english | 0.8912 | — |
| cardiffnlp/twitter-roberta-base-sentiment | — | 0.7238 |
| cardiffnlp/twitter-roberta-base-sentiment-latest | — | 0.7205 |

### Fine-Tuned Transformers (Q7)

| Model | MR Accuracy | Semeval Accuracy |
|-------|-------------|------------------|
| bert-base-cased | 0.8429 | 0.6775 |
| distilbert-base-uncased | 0.8430 | 0.6839 |
| roberta-base | 0.6720 | 0.7040 |
