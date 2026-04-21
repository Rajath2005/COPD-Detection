# 🧠 System Architecture

## Table of Contents
1. [Pipeline Overview](#pipeline-overview)
2. [Preprocessing Pipeline](#preprocessing-pipeline)
3. [Feature Extraction](#feature-extraction)
4. [Baseline CNN Model](#baseline-cnn-model)
5. [Hybrid CNN-LSTM Model](#hybrid-cnn-lstm-model)
6. [Loss Functions & Optimisers](#loss-functions--optimisers)
7. [Evaluation Framework](#evaluation-framework)

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────┐
│                   FULL SYSTEM PIPELINE                  │
└─────────────────────────────────────────────────────────┘

  [1] Raw Audio Input
      (WAV files from ICBHI 2017 — 44.1 kHz / 22.05 kHz)
           │
           ▼
  [2] Preprocessing
      ├── Resampling → 22,050 Hz
      ├── Noise Reduction (spectral subtraction / bandpass filter 100–8000 Hz)
      ├── Normalisation (amplitude scaling)
      └── Cycle Segmentation (using provided ICBHI annotations)
           │
           ▼
  [3] Feature Extraction
      ├── Mel-Spectrogram  (128 mel bins, 25 ms window, 10 ms hop)
      └── MFCC             (40 coefficients + deltas + delta-deltas)
           │
           ▼
  [4] Data Augmentation (training only)
      ├── Time Stretching  (±10%)
      ├── Pitch Shifting   (±2 semitones)
      ├── Additive Gaussian Noise
      └── Time Masking / Frequency Masking (SpecAugment)
           │
           ▼
  ┌────────────────────────────────────┐
  │  BRANCH A — Baseline CNN           │
  │  (Binary: COPD / Normal)           │
  └────────────────────────────────────┘
           │
  ┌────────────────────────────────────┐
  │  BRANCH B — Hybrid CNN-LSTM        │
  │  (4-class: Normal/Mild/Mod/Severe) │
  └────────────────────────────────────┘
           │
           ▼
  [5] Performance Evaluation
      (Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix)
           │
           ▼
  [6] COPD Risk Result Output
```

---

## Preprocessing Pipeline

### 2.1 Resampling

All audio files are resampled to a unified **22,050 Hz** sample rate using high-quality sinc interpolation (`librosa.resample`). This standardises the frequency resolution across the dataset.

### 2.2 Noise Reduction

A two-stage noise reduction approach is applied:

1. **Bandpass filter** (100–8,000 Hz) — removes sub-bass rumble and ultrasonic artefacts irrelevant to lung sounds
2. **Spectral subtraction** — estimates the noise profile from silent segments and subtracts it from the signal spectrum

### 2.3 Signal Normalisation

Peak amplitude normalisation scales each clip to a maximum absolute amplitude of 1.0, eliminating recording-level differences across devices.

### 2.4 Cycle Segmentation

The ICBHI 2017 dataset includes annotation files marking individual breathing cycle start/end timestamps. Each audio file is split into individual cycles of approximately 1–4 seconds. Cycles shorter than 0.5 s are discarded.

---

## Feature Extraction

### 3.1 Mel-Spectrogram

| Parameter | Value |
|-----------|-------|
| FFT size (n_fft) | 1024 |
| Hop length | 256 samples (~11.6 ms) |
| Window length | 512 samples (~23.2 ms) |
| Mel filters | 128 |
| Frequency range | 60–8,000 Hz |
| Output shape | 128 × T (time frames) |

Mel-spectrograms represent the energy in logarithmically-spaced frequency bands over time, mimicking human auditory perception. They capture crackle and wheeze patterns as distinctive texture patterns.

### 3.2 MFCC

| Parameter | Value |
|-----------|-------|
| Number of coefficients | 40 |
| Delta order | 2 (velocity + acceleration) |
| Output shape | 120 × T (40 × 3 delta orders) |

MFCCs summarise the spectral envelope of each frame, stripping fine pitch detail while retaining timbral characteristics. They have been widely used in respiratory sound classification.

### 3.3 Input Tensor Preparation

Feature arrays are zero-padded or truncated to a fixed length (**128 time frames**, approx. 1.5 s at 22,050 Hz with hop=256). The final model input is a **3-channel tensor** (Mel-spectrogram + MFCC + Delta-MFCC) of shape `(3, 128, 128)`.

---

## Baseline CNN Model

The baseline model is a pure convolutional network, treating the Mel-spectrogram as a 2-D image.

```
Input: (1, 128, 128)  — single-channel Mel-spectrogram
│
├── Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
│
├── Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
│
├── Conv2D(128, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
│
├── GlobalAveragePooling2D
│
├── Dense(256) → ReLU → Dropout(0.5)
│
└── Dense(2) → Softmax        ← binary output: [P(COPD), P(Normal)]
```

**Parameters (approx.):** ~1.2 million  
**Task:** Binary classification — COPD present vs. absent

---

## Hybrid CNN-LSTM Model

The hybrid model processes spectral and temporal information in two stages.

```
Input: (3, 128, 128)  — 3-channel feature stack
│
┌─────────────── CNN ENCODER ───────────────┐
│  Conv2D(64, 3×3)  → BN → ReLU → MaxPool  │
│  Conv2D(128, 3×3) → BN → ReLU → MaxPool  │
│  Conv2D(256, 3×3) → BN → ReLU → MaxPool  │
│  → Output shape: (256, 16, 16)            │
└───────────────────────────────────────────┘
│
│  Reshape → (256, 16×16) = (256, 256)     ← treat each CNN feature map column as a time step
│
┌─────────────── LSTM LAYERS ───────────────┐
│  Bidirectional LSTM(128, return_seq=True) │
│  Bidirectional LSTM(64)                  │
│  Dropout(0.4)                             │
└───────────────────────────────────────────┘
│
│  Attention Layer (scaled dot-product)
│
├── Dense(256) → ReLU → Dropout(0.5)
│
└── Dense(4) → Softmax   ← 4-class: [Normal, Mild, Moderate, Severe]
```

**Parameters (approx.):** ~3.8 million  
**Task:** 4-class COPD severity classification

### Why CNN + LSTM?

| Component | What it captures |
|-----------|-----------------|
| CNN layers | Local spectral patterns — crackle texture, wheeze bands, frequency distribution |
| LSTM layers | Temporal dynamics — how respiratory sounds evolve across a breathing cycle |
| Attention | Weight important time steps more heavily — focuses on pathological sound events |

---

## Loss Functions & Optimisers

| Setting | Baseline CNN | Hybrid CNN-LSTM |
|---------|-------------|-----------------|
| Loss | Binary Cross-Entropy | Categorical Cross-Entropy |
| Optimiser | Adam (lr=1e-3) | Adam (lr=5e-4) |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) | CosineAnnealingLR |
| Class weighting | Balanced | Balanced |
| Batch size | 32 | 16 |
| Max epochs | 100 (early stopping, patience=10) | 150 (early stopping, patience=15) |

**Class imbalance handling:** The ICBHI 2017 dataset has unequal class distribution. Inverse-frequency class weights are computed and passed to the loss function to prevent the model from being biased toward the majority class.

---

## Evaluation Framework

### Metrics

| Metric | Formula | Why used |
|--------|---------|----------|
| Accuracy | (TP+TN) / Total | Overall correctness |
| Precision | TP / (TP+FP) | Penalises false positives |
| Recall (Sensitivity) | TP / (TP+FN) | Penalises false negatives — critical in medical screening |
| F1-Score | 2×(P×R)/(P+R) | Harmonic balance of precision and recall |
| AUC-ROC | Area under ROC curve | Threshold-independent separability |
| Specificity | TN / (TN+FP) | True negative rate |

### Outputs

- Per-class confusion matrix
- Classification report (scikit-learn)
- Learning curves (training vs. validation loss/accuracy)
- ROC curves per class
- Saved predictions in `results/metrics.json`

---

*← [Project Overview](Project-Overview.md) | Next: [Dataset →](Dataset.md)*
