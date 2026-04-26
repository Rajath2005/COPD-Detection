<!-- Banner -->
<div align="center">

![Banner](https://capsule-render.vercel.app/api?type=waving&color=0:0D1117,50:0a1628,100:0d2137&height=220&section=header&text=COPD%20Detection&fontSize=52&fontColor=00FFD1&animation=fadeIn&fontAlignY=38&desc=Deep%20Learning%20%E2%80%A2%20Respiratory%20Sound%20Analysis%20%E2%80%A2%20CNN-LSTM&descSize=17&descColor=7FDBFF&descAlignY=58)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Librosa](https://img.shields.io/badge/Librosa-Audio_Processing-FF6B6B?style=for-the-badge)](https://librosa.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![VCET](https://img.shields.io/badge/VCET_Puttur-VTU_Affiliated-0D1117?style=for-the-badge&labelColor=0a1628)](https://vcetputtur.ac.in)
[![Status](https://img.shields.io/badge/Status-Phase_I_Active-00FFD1?style=for-the-badge&labelColor=0D1117)](https://github.com/Rajath2005/COPD-Detection)
[![Issues](https://img.shields.io/github/issues/Rajath2005/COPD-Detection?style=for-the-badge&color=7FDBFF)](https://github.com/Rajath2005/COPD-Detection/issues)

<h3>Deep Learning-Based Early COPD Detection and Severity Classification<br>Using Respiratory Sound Analysis</h3>

> VTU Major Project · Department of Computer Science · VCET Puttur · 2025–26

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Motivation & Problem Statement](#-motivation--problem-statement)
- [Team](#-team)
- [Objectives](#-objectives)
- [Dataset — ICBHI 2017](#-dataset--icbhi-2017)
- [Methodology & Architecture](#-methodology--architecture)
  - [Pipeline Overview](#pipeline-overview)
  - [Preprocessing](#1-preprocessing)
  - [Feature Extraction](#2-feature-extraction)
  - [Model 1 — Baseline CNN](#3-model-1--baseline-cnn)
  - [Model 2 — CNN-LSTM Hybrid](#4-model-2--cnn-lstm-hybrid)
  - [Evaluation Strategy](#5-evaluation-strategy)
- [Repository Structure](#️-repository-structure)
- [Tech Stack](#️-tech-stack)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Dataset Setup](#dataset-setup)
  - [Running the Pipeline](#running-the-pipeline)
- [Notebooks Guide](#-notebooks-guide)
- [Objective 1 Completion Report](wiki/Objective-1-Completion-Report.md)
- [Expected Results](#-expected-results)
- [Project Timeline](#-project-timeline)
- [Applications](#-applications)
- [Contributing](#-contributing)
- [License](#-license)
- [References](#-references)
- [Acknowledgements](#-acknowledgements)

---

## 📌 Overview

Chronic Obstructive Pulmonary Disease (COPD) is a progressive, life-threatening lung disease affecting over **300 million people** worldwide and is the **third leading cause of death** globally (WHO, 2023). Despite its prevalence, COPD is frequently diagnosed only in advanced stages, largely because early symptoms — chronic cough, mild breathlessness — are subtle and often dismissed.

This project develops an **AI-powered, non-invasive COPD screening system** that analyzes **respiratory (lung) sounds** recorded via digital stethoscopes to:

1. **Detect** the presence of COPD (binary classification: COPD / Normal)
2. **Classify** its severity into four clinical grades: **Normal → Mild → Moderate → Severe**

The system leverages the **ICBHI 2017 Respiratory Sound Database** and a hybrid **CNN-LSTM** deep learning architecture, which jointly captures spectral (spatial) and temporal patterns from audio signals — making it more robust than traditional CNN or LSTM-only approaches.

---

## 💡 Motivation & Problem Statement

### Why COPD?
- COPD is **irreversible but manageable** — early intervention significantly slows progression and improves quality of life.
- Conventional diagnosis (spirometry, CT scans) is expensive, invasive, requires specialist access, and is largely unavailable in rural or resource-limited settings.
- **Acoustic-based screening** using lung sounds (wheezing, crackles, rhonchi) is cost-effective, portable, and repeatable.

### Why Deep Learning on Audio?
- Lung sounds contain rich acoustic biomarkers that correlate with structural airway changes in COPD.
- Traditional signal processing approaches (hand-crafted features + classical ML) lack generalization.
- CNNs capture local spectral patterns in spectrograms; LSTMs capture long-range temporal dependencies in breathing cycles — making a **CNN-LSTM hybrid ideal** for this domain.

### Research Gap Addressed
- Most existing work focuses only on binary classification (COPD / Non-COPD) or single-disease detection (asthma, pneumonia). This project extends to **4-class severity grading** aligned with GOLD (Global Initiative for Chronic Obstructive Lung Disease) clinical stages.

---

## 👥 Team

| USN | Name | Role |
|---|---|---|
| 4VP23CS084 | Sanath K | Team Leader — Project coordination, model development |
| 4VP23CS070 | Rajath Kiran A | Team Member — Data preprocessing, feature engineering |
| 4VP23CS076 | Rithesh | Team Member — Model training, hyperparameter tuning |
| 4VP23CS093 | Sheethal D Rai | Team Member — Evaluation, reporting, documentation |

**Project Guide:** Prof. Pramod Kumar PM
Department of Computer Science & Engineering, VCET Puttur (VTU Affiliated)

---

## 🎯 Objectives

| # | Objective | Phase |
|---|---|---|
| 1 | Acquire and understand the ICBHI 2017 respiratory sound dataset | Phase I |
| 2 | Implement audio preprocessing: noise reduction, resampling to 22050 Hz, respiratory cycle segmentation | Phase I |
| 3 | Extract audio features: MFCC (40 coefficients), Mel-spectrograms (128 bands), Chroma, Spectral Contrast | Phase I |
| 4 | Build and validate a **Baseline CNN** model for binary COPD classification (COPD / Normal) | Phase I |
| 5 | Extend to a **CNN-LSTM Hybrid** for 4-class severity classification: Normal / Mild / Moderate / Severe | Phase II |
| 6 | Achieve binary classification accuracy **> 90%** and severity F1-score **> 80%** | Phase II |
| 7 | Evaluate models using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix | Phase II |
| 8 | Document findings in a research report and target a journal submission (IEEE / Springer) | Phase III |

---

## 🗃️ Dataset — ICBHI 2017

**ICBHI 2017 Respiratory Sound Database**
> Rocha et al., "An Open Access Database for the Evaluation of Respiratory Sound Classification Algorithms", *Physiological Measurement*, 2019.

| Property | Details |
|---|---|
| Source | [ICBHI 2017 Challenge](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge) |
| Total Recordings | 920 audio files (`.wav` format) |
| Total Duration | ~5.5 hours |
| Subjects | 126 patients |
| Recording Devices | 7 different digital stethoscopes |
| Recording Locations | Trachea, anterior/posterior chest, lateral |
| Breath Cycle Annotations | 6,898 annotated respiratory cycles |
| Sound Labels | Normal, Crackle, Wheeze, Crackle+Wheeze |
| Patient Demographics | Age, sex, weight, height, BMI, diagnosis included |

### Diagnoses Included
The dataset covers patients with: **COPD**, Healthy, URTI (Upper Respiratory Tract Infection), Bronchiectasis, Pneumonia, Bronchiolitis, and LRTI (Lower Respiratory Tract Infection).

For this project, patient diagnosis metadata is used to derive COPD severity labels aligned with **GOLD Staging**:
- **GOLD 0 / Normal** — No obstruction
- **GOLD I / Mild** — FEV1 ≥ 80% predicted
- **GOLD II / Moderate** — 50% ≤ FEV1 < 80%
- **GOLD III–IV / Severe** — FEV1 < 50%

> **Note:** The raw dataset audio files are not tracked in this repository due to size. See [Dataset Setup](#dataset-setup) for download instructions.

---

## 🧠 Methodology & Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  COPD Detection Pipeline                    │
└─────────────────────────────────────────────────────────────┘

  [RAW AUDIO INPUT]  (.wav files, 920 recordings)
          │
          ▼
  ┌───────────────────────────────────┐
  │       1. DATA PREPROCESSING       │
  │  • Noise reduction (noisereduce)  │
  │  • Resampling → 22050 Hz          │
  │  • Channel normalization (mono)   │
  │  • Respiratory cycle segmentation │
  │    using ICBHI annotation files   │
  └───────────────────────────────────┘
          │
          ▼
  ┌───────────────────────────────────┐
  │      2. FEATURE EXTRACTION        │
  │  • Mel-Spectrogram (128 bands)    │
  │  • MFCC (40 coefficients)         │
  │  • Delta & Delta-Delta MFCCs      │
  │  • Chroma Features                │
  │  • Spectral Contrast              │
  │  • Zero Crossing Rate             │
  └───────────────────────────────────┘
          │
          ▼
  ┌───────────────────────────────────┐
  │     3. LABEL ASSIGNMENT           │
  │  • Binary: COPD / Normal          │
  │  • 4-Class: Normal / Mild /       │
  │    Moderate / Severe (GOLD)       │
  └───────────────────────────────────┘
          │
      ┌───┴──────────────┐
      │                  │
      ▼                  ▼
  ┌──────────┐    ┌────────────────────┐
  │ MODEL 1  │    │     MODEL 2        │
  │ Baseline │    │  CNN-LSTM Hybrid   │
  │   CNN    │    │  (Sequential +     │
  │ (Binary) │    │   Temporal)        │
  └──────────┘    └────────────────────┘
      │                  │
      ▼                  ▼
  ┌──────────┐    ┌────────────────────┐
  │  COPD /  │    │ Normal / Mild /    │
  │  Normal  │    │ Moderate / Severe  │
  └──────────┘    └────────────────────┘
          │
          ▼
  ┌───────────────────────────────────┐
  │      4. EVALUATION                │
  │  • Accuracy, Precision, Recall    │
  │  • F1-Score (macro + weighted)    │
  │  • ROC-AUC (OvR)                  │
  │  • Confusion Matrix               │
  │  • Learning Curves                │
  └───────────────────────────────────┘
```

---

### 1. Preprocessing

**File:** `src/preprocess.py`

The preprocessing pipeline transforms raw `.wav` recordings into clean, segmented audio clips ready for feature extraction.

**Steps:**
1. **Load audio** using `librosa.load()` at a target sample rate of 22,050 Hz (mono)
2. **Noise reduction** using `noisereduce` library (spectral gating method)
3. **Normalization** — peak amplitude normalized to [-1, 1]
4. **Segmentation** — ICBHI annotation `.txt` files provide exact start/end times for each respiratory cycle. Each cycle is extracted as an independent sample.
5. **Padding/Truncation** — cycles are zero-padded or truncated to a fixed length (e.g., 5 seconds) for uniform input size.

**Output:** Segmented `.npy` audio arrays stored in `data/processed/`

---

### 2. Feature Extraction

**File:** `src/features.py`

Features are extracted per respiratory cycle and stored as NumPy arrays.

| Feature | Description | Shape (per sample) |
|---|---|---|
| **Mel-Spectrogram** | Frequency-time representation using mel scale | `(128, T)` |
| **MFCC** | 40 Mel-Frequency Cepstral Coefficients + Δ + ΔΔ | `(120, T)` |
| **Chroma** | 12 pitch class energy distributions | `(12, T)` |
| **Spectral Contrast** | Energy difference across 7 frequency bands | `(7, T)` |
| **Zero Crossing Rate** | Rate of signal sign changes | `(1, T)` |

For the CNN models, Mel-Spectrograms (and optionally stacked MFCC features) are used as 2D image-like inputs. For the CNN-LSTM model, the temporal axis `T` forms the sequence dimension fed into the LSTM layers.

**Output:** Feature arrays stored in `data/features/` as `.npy` files with matching label arrays.

---

### 3. Model 1 — Baseline CNN

**File:** `src/models/baseline_cnn.py`

A standard 2D Convolutional Neural Network treating Mel-Spectrograms as images.

```
Input: Mel-Spectrogram (1, 128, T)
  │
  ├── Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
  ├── Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
  ├── Conv2D(128, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
  ├── Conv2D(256, 3×3) → BatchNorm → ReLU → GlobalAvgPool
  │
  ├── Flatten
  ├── Dropout(0.4)
  ├── Dense(256) → ReLU
  ├── Dropout(0.3)
  └── Dense(2) → Softmax (Binary: COPD / Normal)
```

**Training Config:**
- Optimizer: Adam (lr=1e-4)
- Loss: Cross-Entropy
- Batch Size: 32
- Epochs: 50 with EarlyStopping (patience=10)
- Data Augmentation: Time stretching, pitch shifting, additive noise

---

### 4. Model 2 — CNN-LSTM Hybrid

**File:** `src/models/cnn_lstm.py`

The hybrid model first applies convolutional layers to capture local spectral patterns, then feeds the resulting feature maps as a time sequence into LSTM layers to model temporal breathing dynamics.

```
Input: Mel-Spectrogram (1, 128, T)
  │
  ├── [CNN Block — Feature Extraction]
  │     Conv2D(64, 3×3) → BN → ReLU → MaxPool (freq axis only)
  │     Conv2D(128, 3×3) → BN → ReLU → MaxPool (freq axis only)
  │     Conv2D(256, 3×3) → BN → ReLU
  │     → Reshape: (Batch, T', Features)   ← time steps preserved
  │
  ├── [LSTM Block — Temporal Modeling]
  │     BiLSTM(256 units) → Dropout(0.3)
  │     LSTM(128 units) → Dropout(0.3)
  │
  ├── Attention Layer (optional, Phase II)
  │
  ├── Dense(256) → ReLU → Dropout(0.4)
  ├── Dense(128) → ReLU
  └── Dense(4) → Softmax (4-class: Normal/Mild/Moderate/Severe)
```

**Key Design Decisions:**
- Pooling is applied **only along the frequency axis** in CNN blocks so that the time dimension `T` is preserved and passed intact to the LSTM.
- **Bidirectional LSTM** is used so that each time step has context from both past and future respiratory cycle segments.
- An optional **attention mechanism** highlights the most diagnostically relevant time windows.

---

### 5. Evaluation Strategy

**File:** `src/evaluate.py`

| Metric | Why It Matters |
|---|---|
| **Accuracy** | Overall correctness; baseline measure |
| **Precision** | Minimize false COPD positives (avoid unnecessary anxiety) |
| **Recall (Sensitivity)** | Minimize missed COPD cases (critical for screening) |
| **F1-Score** | Harmonic mean; handles class imbalance |
| **ROC-AUC** | Discrimination ability across thresholds |
| **Confusion Matrix** | Per-class error analysis |
| **Learning Curves** | Overfitting / underfitting diagnosis |

For multi-class severity, both **macro-averaged** and **weighted** F1-scores are reported. Given the clinical context (where missing a true COPD case is more costly than a false positive), **Recall** is treated as the primary optimization metric.

---

## 🗂️ Repository Structure

```
COPD-Detection/
│
├── 📁 data/
│   ├── raw/                        # Original ICBHI 2017 .wav files + .txt annotations
│   │   └── (download separately — see Dataset Setup)
│   ├── processed/                  # Segmented respiratory cycle arrays (.npy)
│   └── features/                   # Extracted feature arrays
│       ├── mel_spectrograms.npy
│       ├── mfcc_features.npy
│       └── labels_{binary,4class}.npy
│
├── 📁 notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   │                               #   - Dataset distribution, demographics
│   │                               #   - Audio waveform & spectrogram visualization
│   │                               #   - Class imbalance analysis
│   ├── 02_preprocessing.ipynb      # Preprocessing walkthrough with visualizations
│   ├── 03_baseline_cnn.ipynb       # Baseline CNN training, evaluation, ablation
│   └── 04_cnn_lstm_hybrid.ipynb    # CNN-LSTM training, severity classification
│
├── 📁 src/
│   ├── preprocess.py               # Full preprocessing pipeline (CLI + importable)
│   ├── features.py                 # Feature extraction module
│   ├── dataset.py                  # PyTorch Dataset class for ICBHI data
│   ├── augment.py                  # Audio augmentation utilities
│   ├── train.py                    # Training loop with logging & checkpointing
│   ├── evaluate.py                 # Evaluation script (metrics, plots)
│   ├── utils.py                    # Helper functions (seed setting, plotting, etc.)
│   └── models/
│       ├── __init__.py
│       ├── baseline_cnn.py         # Baseline CNN architecture (PyTorch nn.Module)
│       └── cnn_lstm.py             # CNN-LSTM Hybrid architecture
│
├── 📁 results/
│   ├── plots/
│   │   ├── confusion_matrix_cnn.png
│   │   ├── confusion_matrix_cnn_lstm.png
│   │   ├── roc_curves.png
│   │   └── loss_accuracy_curves.png
│   └── metrics.json                # Serialized evaluation results (all models)
│
├── 📁 reports/
│   ├── synopsis.pdf                # Approved project synopsis
│   └── progress_reports/
│       ├── PR1_introduction.pdf
│       ├── PR2_requirements.pdf
│       ├── PR3_design.pdf          # (upcoming)
│       └── PR4_implementation.pdf  # (upcoming)
│
├── 📁 checkpoints/                 # Saved model weights (.pth)
│   ├── baseline_cnn_best.pth
│   └── cnn_lstm_best.pth
│
├── requirements.txt                # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

> **Note:** `data/raw/`, `data/processed/`, `data/features/`, and `checkpoints/` are listed in `.gitignore` and must be generated locally following the [Getting Started](#-getting-started) instructions.

---

## 🛠️ Tech Stack

| Category | Library / Tool | Version | Purpose |
|---|---|---|---|
| **Language** | Python | 3.10+ | Core development language |
| **Deep Learning** | PyTorch | ≥ 2.0 | Model building, training, GPU acceleration |
| **Audio Processing** | Librosa | ≥ 0.10 | Feature extraction, audio I/O |
| **Audio I/O** | SoundFile | ≥ 0.12 | Reading/writing WAV files |
| **Noise Reduction** | noisereduce | ≥ 3.0 | Spectral gating-based denoising |
| **Numerical** | NumPy | ≥ 1.24 | Array operations |
| **Data Manipulation** | Pandas | ≥ 2.0 | Metadata handling, CSVs |
| **Visualization** | Matplotlib | ≥ 3.7 | Plots, spectrograms, curves |
| **Visualization** | Seaborn | ≥ 0.12 | Statistical visualizations |
| **ML Utilities** | Scikit-learn | ≥ 1.3 | Metrics, splits, class balancing |
| **Notebooks** | Jupyter | ≥ 7.0 | Interactive development & EDA |
| **Experiment Tracking** | TensorBoard | ≥ 2.13 | Loss curves, metric logging |
| **Dataset** | ICBHI 2017 | — | Respiratory sound recordings |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- `pip` package manager
- Git
- A CUDA-capable GPU is **recommended** for training (CPU training is possible but slow)
- ~3 GB free disk space for the dataset + features

Verify your Python version:
```bash
python --version
# Should output: Python 3.10.x or higher
```

---

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Rajath2005/COPD-Detection.git
cd COPD-Detection

# 2. Create a virtual environment (recommended)
python -m venv venv

# Activate on Linux/macOS:
source venv/bin/activate

# Activate on Windows (Command Prompt):
venv\Scripts\activate.bat

# Activate on Windows (PowerShell):
venv\Scripts\Activate.ps1

# 3. Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Dataset Setup

The ICBHI 2017 dataset must be downloaded separately from the official source.

```bash
# Step 1: Visit the official ICBHI 2017 Challenge page
# https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
# Register if required and download the dataset archive.

# Step 2: Extract the archive
unzip ICBHI_2017_Challenge.zip -d data/raw/
# OR
tar -xvf ICBHI_2017_Challenge.tar.gz -C data/raw/

# Expected structure after extraction:
# data/raw/
#   ├── 101_1b1_Al_sc_Meditron.wav
#   ├── 101_1b1_Al_sc_Meditron.txt   ← annotation file
#   ├── 102_1b1_Al_sc_Meditron.wav
#   ├── 102_1b1_Al_sc_Meditron.txt
#   ├── ...
#   └── ICBHI_Challenge_diagnosis.txt ← patient diagnoses
```

> **Tip:** The annotation `.txt` files contain tab-separated rows: `start_time  end_time  crackle_label  wheeze_label`. These are parsed by `src/preprocess.py` to segment recordings into individual respiratory cycles.

---

### Running the Pipeline

Run each step sequentially:

```bash
# Step 1: Preprocess raw audio → segmented cycles
python src/preprocess.py \
  --input_dir data/raw/ \
  --output_dir data/processed/ \
  --target_sr 22050 \
  --cycle_duration 5.0

# Step 2: Extract features from processed cycles
python src/features.py \
  --input_dir data/processed/ \
  --output_dir data/features/ \
  --feature_type mel_mfcc

# Step 3a: Train the Baseline CNN (binary classification)
python src/train.py \
  --model baseline_cnn \
  --feature_dir data/features/ \
  --task binary \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4 \
  --checkpoint_dir checkpoints/

# Step 3b: Train the CNN-LSTM Hybrid (severity classification)
python src/train.py \
  --model cnn_lstm \
  --feature_dir data/features/ \
  --task severity \
  --epochs 100 \
  --batch_size 32 \
  --lr 5e-5 \
  --checkpoint_dir checkpoints/

# Step 4: Evaluate a trained model
python src/evaluate.py \
  --model cnn_lstm \
  --checkpoint checkpoints/cnn_lstm_best.pth \
  --feature_dir data/features/ \
  --task severity \
  --output_dir results/

# Step 5: Launch TensorBoard to monitor training
tensorboard --logdir runs/
```

---

### Running Notebooks

```bash
# Start the Jupyter server
jupyter notebook notebooks/

# Or launch JupyterLab for a better UI
pip install jupyterlab
jupyter lab notebooks/
```

Open notebooks in this order for a complete walkthrough:
1. `01_eda.ipynb` → Understand the dataset
2. `02_preprocessing.ipynb` → See preprocessing steps visually
3. `03_baseline_cnn.ipynb` → Train and evaluate the baseline
4. `04_cnn_lstm_hybrid.ipynb` → Full hybrid model

---

## 📓 Notebooks Guide

### `01_eda.ipynb` — Exploratory Data Analysis
- Dataset overview: number of recordings, patients, total duration
- Audio waveform visualization of Normal vs COPD samples
- Mel-Spectrogram comparison across disease types
- Class distribution and imbalance analysis
- Patient demographics (age, sex, BMI) by diagnosis
- Respiratory cycle duration statistics

### `02_preprocessing.ipynb` — Preprocessing Walkthrough
- Step-by-step noise reduction demonstration (before/after spectrograms)
- Resampling effect on audio quality
- Respiratory cycle segmentation walkthrough using annotations
- Fixed-length padding/truncation strategy comparison

### `03_baseline_cnn.ipynb` — Baseline CNN
- Feature preparation and data loaders
- Model architecture summary and parameter count
- Training loop with live loss/accuracy curves
- Validation performance: accuracy, precision, recall, F1
- Confusion matrix visualization
- Error analysis: common misclassification patterns
- Ablation study: MFCC vs Mel-Spectrogram as input

### `04_cnn_lstm_hybrid.ipynb` — CNN-LSTM Hybrid
- Architecture design rationale
- Training on 4-class severity labels
- Multi-class metrics: macro F1, per-class precision/recall
- ROC curves (One-vs-Rest for each severity class)
- Attention weight visualization (if attention is enabled)
- Comparison against Baseline CNN

---

## 📊 Expected Results

### Binary Classification (COPD / Normal) — Baseline CNN

| Metric | Target | Expected Range |
|---|---|---|
| Accuracy | > 90% | 90–94% |
| Precision (COPD) | > 88% | 87–92% |
| Recall (COPD) | > 88% | 88–93% |
| F1-Score (COPD) | > 88% | 88–92% |
| ROC-AUC | > 0.92 | 0.92–0.96 |

### 4-Class Severity Classification — CNN-LSTM Hybrid

| Metric | Target | Expected Range |
|---|---|---|
| Accuracy | > 80% | 80–87% |
| Macro F1-Score | > 78% | 78–85% |
| Weighted F1-Score | > 82% | 82–88% |
| ROC-AUC (OvR, macro) | > 0.88 | 0.88–0.93 |

> **Note:** These are projected targets based on prior literature on ICBHI 2017 benchmarks. Actual results will be reported and updated here during Phase II.

### Benchmark Comparison (Literature)

| Paper | Model | Task | Accuracy |
|---|---|---|---|
| Perna & Tagarelli (2019) | CNN | Binary | 83.7% |
| Nguyen et al. (2020) | CNN-RNN | Binary | 86.5% |
| Ma et al. (2021) | CNN-LSTM + Attention | Binary | 89.2% |
| **This Work (Target)** | **CNN-LSTM Hybrid** | **Binary + 4-Class** | **> 90%** |

---

## 📅 Project Timeline

| # | Milestone | Deadline | Status |
|---|---|---|---|
| 1 | Synopsis submission | 27 Feb 2026 | ✅ Done |
| 2 | Synopsis presentation | 03–10 Mar 2026 | ✅ Done |
| 3 | Progress Report 1 — Introduction & Background | 14 Apr 2026 | 🔄 In Progress |
| 4 | Progress Report 2 — Requirements & Dataset | 14 Apr 2026 | 🔄 In Progress |
| 5 | Presentation 1 (EDA + Preprocessing + Baseline CNN) | 2nd Week May 2026 | ⏳ Upcoming |
| 6 | Progress Report 3 — System Design & Architecture | 4th Week Aug 2026 | ⏳ Upcoming |
| 7 | Progress Report 4 — Implementation Details | 4th Week Aug 2026 | ⏳ Upcoming |
| 8 | Presentation 2 (CNN-LSTM Results) | 2nd Week Sep 2026 | ⏳ Upcoming |
| 9 | Progress Report 5 — Testing & Validation | 2nd Week Oct 2026 | ⏳ Upcoming |
| 10 | Progress Report 6 — Results & Analysis | 2nd Week Oct 2026 | ⏳ Upcoming |
| 11 | Final Report + Journal Paper Draft | 20 Nov 2026 | ⏳ Upcoming |
| 12 | Internal Viva / Project Exhibition | 4th Week Nov 2026 | ⏳ Upcoming |

---

## 🏥 Applications

### Immediate
- **Early COPD Screening Tool** — deployable as a mobile/web app integrated with a digital stethoscope for point-of-care screening
- **Rural & Remote Healthcare** — works on low-cost devices, no specialist required, no imaging equipment needed
- **Telemedicine Integration** — patients record lung sounds at home; AI flags abnormalities for remote physician review

### Research & Clinical
- **AI-Assisted Diagnosis** — reduces clinician workload in high-volume respiratory clinics
- **Longitudinal Monitoring** — track severity progression over time using repeated recordings
- **Drug Trial Endpoints** — objective acoustic biomarkers for COPD severity in clinical trials

### Educational
- **Medical Training** — visualize and study respiratory acoustic patterns for student training
- **Open-Source Benchmark** — reproducible ICBHI 2017 pipeline for the research community

---

## 🤝 Contributing

We welcome contributions from the community! Whether you're fixing a bug, improving documentation, or proposing a new model variant — all contributions are appreciated.

### How to Contribute

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/<your-username>/COPD-Detection.git
cd COPD-Detection

# 3. Create a feature branch
git checkout -b feature/your-feature-name

# 4. Make your changes and commit
git add .
git commit -m "feat: add attention mechanism to CNN-LSTM"

# 5. Push to your fork
git push origin feature/your-feature-name

# 6. Open a Pull Request on GitHub
```

### Contribution Guidelines
- Follow PEP 8 style for Python code
- Add docstrings to all functions and classes
- Write unit tests for new features in `tests/`
- Update relevant documentation / notebook if the pipeline changes
- Use clear, descriptive commit messages (preferably [Conventional Commits](https://www.conventionalcommits.org/))

### Reporting Issues
Please use [GitHub Issues](https://github.com/Rajath2005/COPD-Detection/issues) to report bugs or suggest improvements. Include:
- A clear description of the problem
- Steps to reproduce (with code snippets if applicable)
- Expected vs. actual behaviour
- Environment details (OS, Python version, PyTorch version)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

You are free to use, modify, and distribute this code for personal, academic, or commercial purposes, provided proper attribution is given.

---

## 📚 References

1. Rocha, B.M., et al. (2019). *An Open Access Database for the Evaluation of Respiratory Sound Classification Algorithms.* Physiological Measurement, 40(3), 035001. [DOI](https://doi.org/10.1088/1361-6579/ab03ea)

2. Perna, D., & Tagarelli, A. (2019). *Deep Auscultation: Predicting Respiratory Anomalies and Diseases via Recurrent Neural Networks.* IEEE CBMS 2019.

3. Nguyen, T., et al. (2020). *Lung Sound Classification Using Co-tuning and Stochastic Normalization.* IEEE TNSRE.

4. Ma, Y., et al. (2021). *LungBRN: A Smart Digital Stethoscope for Detecting Respiratory Disease Using bi-ResNet Deep Learning Algorithm.* IEEE BioCAS 2021.

5. Global Initiative for Chronic Obstructive Lung Disease (GOLD). (2024). *Global Strategy for the Diagnosis, Management, and Prevention of COPD.* [goldcopd.org](https://goldcopd.org)

6. WHO. (2023). *Chronic obstructive pulmonary disease (COPD) Fact Sheet.* [who.int](https://www.who.int/news-room/fact-sheets/detail/chronic-obstructive-pulmonary-disease-(copd))

---

## 🙏 Acknowledgements

- **ICBHI 2017 Challenge** organizers and contributors for making the respiratory sound database publicly available
- **Prof. Pramod Kumar PM** — for continuous guidance, mentorship, and technical feedback throughout the project
- **VCET Puttur & VTU** — for academic infrastructure and support
- **PyTorch**, **Librosa**, and **scikit-learn** open-source communities
- Prior researchers on the ICBHI 2017 benchmark whose published results guide our baselines

---

<div align="center">

![Footer](https://capsule-render.vercel.app/api?type=waving&color=0:0d2137,100:0D1117&height=120&section=footer)

*Built with 🫁 + 💻 by Team COPD-Detection — VCET Puttur, 2025–26*

[![GitHub](https://img.shields.io/badge/GitHub-Rajath2005%2FCOPD--Detection-181717?style=flat-square&logo=github)](https://github.com/Rajath2005/COPD-Detection)

</div>
