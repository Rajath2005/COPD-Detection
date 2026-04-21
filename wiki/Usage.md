# 🚀 Usage Guide

## Table of Contents
1. [Workflow Overview](#workflow-overview)
2. [Step 1 — Preprocessing](#step-1--preprocessing)
3. [Step 2 — Feature Extraction](#step-2--feature-extraction)
4. [Step 3 — Exploratory Data Analysis](#step-3--exploratory-data-analysis)
5. [Step 4 — Train the Baseline CNN](#step-4--train-the-baseline-cnn)
6. [Step 5 — Train the Hybrid CNN-LSTM](#step-5--train-the-hybrid-cnn-lstm)
7. [Step 6 — Evaluate Models](#step-6--evaluate-models)
8. [Step 7 — Predict on New Audio](#step-7--predict-on-new-audio)
9. [Using Jupyter Notebooks](#using-jupyter-notebooks)
10. [Command Reference](#command-reference)

---

## Workflow Overview

```
data/raw/         →  preprocess.py   →  data/processed/
data/processed/   →  features.py     →  data/features/
data/features/    →  train.py        →  models/  +  results/
models/           →  evaluate.py     →  results/metrics.json
```

Each step can be run from the command line or through the corresponding Jupyter notebook.

---

## Step 1 — Preprocessing

The preprocessing script loads raw ICBHI WAV files, reads cycle annotations, applies noise reduction, resamples, and saves individual cycle clips.

```bash
python src/preprocess.py \
  --raw-dir data/raw \
  --out-dir data/processed \
  --sample-rate 22050 \
  --min-duration 0.5
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--raw-dir` | `data/raw` | Path to raw ICBHI WAV files |
| `--out-dir` | `data/processed` | Output directory for cycle clips |
| `--sample-rate` | `22050` | Target sample rate (Hz) |
| `--min-duration` | `0.5` | Minimum cycle duration to retain (seconds) |
| `--denoise` | `True` | Apply spectral subtraction noise reduction |

**Output:** Cycle-level WAV files in `data/processed/`, organised by patient:

```
data/processed/
├── 101_1b1_Al_sc_Meditron_cycle_01.wav
├── 101_1b1_Al_sc_Meditron_cycle_02.wav
├── ...
└── metadata.csv         ← cycle labels, patient IDs, diagnoses
```

---

## Step 2 — Feature Extraction

Extracts Mel-spectrograms and MFCCs from preprocessed cycle clips, then saves them as numpy arrays.

```bash
python src/features.py \
  --input-dir data/processed \
  --out-dir data/features \
  --n-mels 128 \
  --n-mfcc 40 \
  --max-frames 128
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-dir` | `data/processed` | Directory of cycle WAV files |
| `--out-dir` | `data/features` | Output directory for feature arrays |
| `--n-mels` | `128` | Number of Mel filter banks |
| `--n-mfcc` | `40` | Number of MFCC coefficients |
| `--max-frames` | `128` | Pad/truncate to this many time frames |
| `--augment` | `False` | Apply augmentation to training set |

**Output:**

```
data/features/
├── X_train.npy      ← shape (N_train, 3, 128, 128)
├── X_val.npy
├── X_test.npy
├── y_train.npy      ← integer class labels
├── y_val.npy
└── y_test.npy
```

---

## Step 3 — Exploratory Data Analysis

Open the EDA notebook for visualisations of the dataset:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

This notebook includes:
- Class distribution bar charts
- Sample waveform plots
- Mel-spectrogram visualisations for each class
- Patient demographic breakdowns
- Cycle duration distributions

---

## Step 4 — Train the Baseline CNN

```bash
python src/train.py \
  --model baseline_cnn \
  --features-dir data/features \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.001 \
  --out-dir models/baseline_cnn
```

Training logs are printed to the console and saved to `models/baseline_cnn/training_log.csv`.

Checkpoints are saved at each epoch if validation loss improves:

```
models/baseline_cnn/
├── best_model.pth        ← best validation checkpoint
├── last_model.pth        ← final epoch checkpoint
└── training_log.csv
```

---

## Step 5 — Train the Hybrid CNN-LSTM

```bash
python src/train.py \
  --model cnn_lstm \
  --features-dir data/features \
  --epochs 150 \
  --batch-size 16 \
  --lr 0.0005 \
  --num-classes 4 \
  --out-dir models/cnn_lstm
```

**Additional CNN-LSTM arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--lstm-hidden` | `128` | LSTM hidden units (per direction) |
| `--lstm-layers` | `2` | Number of stacked LSTM layers |
| `--attention` | `True` | Enable attention mechanism |
| `--bidirectional` | `True` | Use bidirectional LSTM |

---

## Step 6 — Evaluate Models

```bash
# Evaluate baseline CNN
python src/evaluate.py \
  --model-path models/baseline_cnn/best_model.pth \
  --model-type baseline_cnn \
  --features-dir data/features \
  --out-dir results/baseline_cnn

# Evaluate hybrid CNN-LSTM
python src/evaluate.py \
  --model-path models/cnn_lstm/best_model.pth \
  --model-type cnn_lstm \
  --features-dir data/features \
  --out-dir results/cnn_lstm \
  --num-classes 4
```

**Outputs in `results/<model>/`:**

| File | Description |
|------|-------------|
| `metrics.json` | Accuracy, precision, recall, F1, AUC |
| `confusion_matrix.png` | Normalised confusion matrix plot |
| `roc_curves.png` | Per-class ROC curves |
| `loss_curve.png` | Training vs. validation loss |
| `classification_report.txt` | Full scikit-learn classification report |

---

## Step 7 — Predict on New Audio

To run inference on a single new audio file:

```bash
python src/predict.py \
  --audio path/to/recording.wav \
  --model-path models/cnn_lstm/best_model.pth \
  --model-type cnn_lstm
```

**Output:**

```
File: recording.wav
Prediction: Moderate COPD
Confidence: 78.4%
Probabilities:
  Normal   :  5.2%
  Mild     : 11.8%
  Moderate : 78.4%
  Severe   :  4.6%
```

---

## Using Jupyter Notebooks

All steps can also be executed through interactive notebooks:

| Notebook | Description |
|----------|-------------|
| `notebooks/01_eda.ipynb` | Dataset exploration and visualisation |
| `notebooks/02_preprocessing.ipynb` | Step-through of preprocessing pipeline |
| `notebooks/03_baseline_cnn.ipynb` | Baseline CNN — training, evaluation, analysis |
| `notebooks/04_cnn_lstm_hybrid.ipynb` | Hybrid model — training, evaluation, analysis |

Launch all notebooks:

```bash
jupyter notebook notebooks/
```

---

## Command Reference

```
python src/preprocess.py --help
python src/features.py --help
python src/train.py --help
python src/evaluate.py --help
python src/predict.py --help
```

---

*← [Installation](Installation.md) | Next: [Model Training →](Model-Training.md)*
