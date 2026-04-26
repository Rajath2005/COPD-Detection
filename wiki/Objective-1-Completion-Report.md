# Objective 1 Completion Report

## Title

COPD Detection Major Project  
Objective 1: Data Preprocessing and Feature Extraction Pipeline  
Status: Completed Successfully

---

## Objective 1 (as approved)

Build a complete preprocessing pipeline for the ICBHI 2017 Respiratory Sound Database to:

1. Load and standardize respiratory audio.
2. Perform robust multi-stage noise reduction.
3. Segment recordings into fixed-length respiratory windows.
4. Extract deep-learning-ready acoustic features.
5. Apply controlled augmentation for COPD training data.
6. Create binary and severity labels.
7. Perform patient-level train/val/test splitting with no leakage.
8. Save reproducible artifacts (`.npy`, metadata, config).

---

## Final Implementation Summary

The Objective 1 workflow is now consolidated in:

- `src/preprocess.py`

This file now handles both preprocessing and feature extraction end-to-end (the old separate `src/features.py` pipeline was removed because it no longer matched the required objective specification).

Notebook support for reproducible analysis and demonstration is available at:

- `notebooks/01_COPD_Preprocessing_Pipeline.ipynb`

---

## Dataset Used

- Source: ICBHI 2017 Respiratory Sound Database
- Local path used for full run: `../ICBHI`
- Total recordings processed: `920` `.wav` files
- Total patients: `126`
- Diagnosis file used: `patient_diagnosis.csv`

---

## Detailed Pipeline Implemented

### 1) Audio loading and standardization

- All audio is loaded as mono and resampled to `22050 Hz` using `librosa`.
- This ensures a fixed temporal and spectral base for all downstream operations.

### 2) Multi-stage denoising

Three denoising/filtering steps are applied in sequence:

1. Stationary spectral gating (`noisereduce`, `stationary=True`)
2. Non-stationary spectral gating (`noisereduce`, `stationary=False`)
3. Butterworth bandpass filtering (`80–2000 Hz`, SOS implementation)

This combination reduces environmental/background noise while preserving breathing-relevant frequency content.

### 3) Respiratory-cycle window segmentation

- Annotation `.txt` files are parsed (`start end crackle wheeze` per line).
- For each cycle, a `5-second` window is extracted around the cycle center.
- If too short near boundaries, reflect-padding is used.
- If no valid annotation exists, fallback segmentation uses sliding windows (`5s` window, `2.5s` hop).

### 4) Feature extraction (fixed target shapes)

- Log-Mel spectrogram:
  - shape: `(1, 128, 431)`
  - normalized to `[0, 1]`
- MFCC stack:
  - channels: MFCC + delta + delta2
  - shape: `(3, 40, 431)`
  - per-channel z-score normalization

### 5) Data augmentation

- Applied only to **training split** and only for **COPD-positive samples**.
- Augmentation factor: `4` copies per eligible sample.
- Transforms used:
  - AddGaussianNoise
  - TimeStretch
  - PitchShift
  - Shift
- `spec_augment()` helper function (time/frequency masking) is included in `src/preprocess.py` for later model training use.

### 6) Labeling strategy

- Binary label:
  - `COPD -> 1`
  - all others -> `0`
- Severity label:
  - Non-COPD -> `0 (Normal)`
  - COPD patients use crackle/wheeze ratio thresholds:
    - `< 0.25 -> 1 (Mild)`
    - `0.25 to 0.60 -> 2 (Moderate)`
    - `> 0.60 -> 3 (Severe)`

### 7) Split strategy and leakage prevention

- Patient-level split target: `60/20/20` (train/val/test).
- Stratified grouped splitting is used where feasible.
- Small-set fallback logic was added for debug runs (`--limit`) to avoid split failures.
- No patient overlap is allowed between splits (explicitly validated).

---

## Files Added/Updated for Objective 1

### Created

- `wiki/Objective-1-Completion-Report.md` (this report)
- `notebooks/01_COPD_Preprocessing_Pipeline.ipynb`
- `data/features/.gitkeep`

### Updated

- `src/preprocess.py` (full unified pipeline)
- `requirements.txt` (added required dependencies)
- `.gitignore` (feature artifact handling)

### Removed

- `src/features.py` (legacy, superseded)
- `notebooks/01_eda.ipynb` (empty placeholder)

---

## Full Run Command and Runtime

### Command used

```bash
python src/preprocess.py --data_dir "..\\ICBHI" --output_dir data/features --n_jobs -1 --seed 42 --copd_aug_factor 4
```

### Final runtime

- `3,027,520 ms` (~`50m 28s`)

---

## Final Output Artifacts Produced

Saved in `data/features`:

- `X_mel_train.npy` -> `(18164, 1, 128, 431)`
- `X_mel_val.npy` -> `(1468, 1, 128, 431)`
- `X_mel_test.npy` -> `(1170, 1, 128, 431)`
- `X_mfcc_train.npy` -> `(18164, 3, 40, 431)`
- `X_mfcc_val.npy` -> `(1468, 3, 40, 431)`
- `X_mfcc_test.npy` -> `(1170, 3, 40, 431)`
- `y_binary_train.npy`, `y_binary_val.npy`, `y_binary_test.npy`
- `y_severity_train.npy`, `y_severity_val.npy`, `y_severity_test.npy`
- `metadata.csv`
- `config.json`

---

## Result Distribution Summary

### Train

- Rows: `18164`
- Patients: `75`
- Binary: `{1: 17380, 0: 784}`
- Severity: `{3: 9345, 1: 4050, 2: 3985, 0: 784}`
- Augmented rows: `13904`

### Validation

- Rows: `1468`
- Patients: `26`
- Binary: `{1: 1242, 0: 226}`
- Severity: `{2: 671, 3: 377, 0: 226, 1: 194}`
- Augmented rows: `0`

### Test

- Rows: `1170`
- Patients: `25`
- Binary: `{1: 1028, 0: 142}`
- Severity: `{1: 420, 2: 384, 3: 224, 0: 142}`
- Augmented rows: `0`

---

## Problems Faced and How They Were Solved

### Problem 1: Split failure on debug subset

**Issue:**  
During smoke-test run with `--limit` (small subset), stratified group split failed:

- `ValueError: n_splits=5 cannot be greater than the number of members in each class.`

**Root cause:**  
The limited sample set had too few patients per class for 5-fold stratified grouping.

**Fix applied:**  
Added conditional fallback logic in `patient_train_val_test()`:

- Use `StratifiedGroupKFold` for sufficient class support.
- Automatically switch to `GroupShuffleSplit` fallback for tiny debug subsets.

**Result:**  
Debug validation runs succeeded while preserving proper split behavior for full dataset runs.

---

### Problem 2: Full-run memory crash near end

**Issue:**  
First full run failed after lengthy processing with:

- `numpy._core._exceptions._ArrayMemoryError`
- Unable to allocate ~`4.28 GiB` when stacking all feature tensors in RAM.

**Root cause:**  
`np.stack()` was building very large in-memory arrays before saving.

**Fix applied:**  
Refactored feature writing in `run_pipeline()` to memory-safe disk-backed writes:

- `np.lib.format.open_memmap(...)` for per-split feature arrays.
- Incremental sample-by-sample write directly to `.npy` on disk.
- Labels kept small and saved normally.

**Result:**  
Second full run completed successfully with same output spec and no RAM crash.

---

## Validation and Success Checks

All key checks passed:

- Required output tensors generated.
- Output shapes match Objective 1 spec.
- Binary and severity labels generated correctly.
- Augmentation applied only to train split.
- No patient leakage:
  - train-val overlap = `0`
  - train-test overlap = `0`
  - val-test overlap = `0`
- `metadata.csv` and `config.json` generated for reproducibility.

---

## Objective 1 Completion Statement

Objective 1 has been fully completed and validated.

The project now has a production-ready preprocessing and feature-generation pipeline that is reproducible, leakage-safe, and directly usable for:

- Objective 2: Baseline binary COPD classifier training
- Objective 3: Enhanced severity classification model training

---

## Recommended Next Step

Proceed to Objective 2 with:

- Inputs: `X_mel_*` and `y_binary_*`
- Baseline model: CNN binary classifier
- Maintain same patient-level split artifacts created in this objective

