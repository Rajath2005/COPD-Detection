# COPD Detection - Code Explanations & Documentation

This folder contains comprehensive PDF explanations generated from the project notebooks and code. Each section below guides you to the relevant documentation.

## 📋 Documentation Structure

### 1. **Data Pipeline & Signal Processing**
   - **Location:** `data-pipeline/`
   - **File:** `ICBHI_Signal_Pipeline.pdf`
   - **Purpose:** Explains the ICBHI dataset preprocessing pipeline, signal handling, and data flow
   - **Relevant Code:** [`src/preprocess.py`](../../src/preprocess.py), [`notebooks/01_COPD_Preprocessing_Pipeline.ipynb`](../../notebooks/01_COPD_Preprocessing_Pipeline.ipynb)

### 2. **Feature Extraction & Acoustic Analysis**
   - **Location:** `feature-extraction/`
   - **Files:**
     - `VCET_Acoustic_Feature_Architecture.pdf` - Technical architecture of acoustic feature extraction
     - `Acoustic_Blueprints_for_Respiratory_Health.pdf` - Detailed explanation of respiratory audio feature engineering
   - **Purpose:** Comprehensive guide to mel-spectrograms, MFCCs, and acoustic feature design
   - **Relevant Code:** [`src/preprocess.py`](../../src/preprocess.py)

### 3. **Model Architecture & Training**
   - **Location:** `model-training/`
   - **File:** `Deep_Learning_Respiratory_Classification.pdf`
   - **Purpose:** Explains deep learning model architecture, training procedures, optimization strategies
   - **Relevant Code:** [`src/train.py`](../../src/train.py)

### 4. **Classification & Severity Grading**
   - **Location:** `classification/`
   - **File:** `Acoustic_COPD_Severity_Grading.pdf`
   - **Purpose:** Details COPD severity classification, grading methodology, and evaluation metrics
   - **Relevant Code:** [`src/evaluate.py`](../../src/evaluate.py)

---

## 🎯 Quick Navigation Guide

**Looking for:**
- 🔊 **How audio preprocessing works?** → See `data-pipeline/ICBHI_Signal_Pipeline.pdf`
- 🎵 **How acoustic features are extracted?** → See `feature-extraction/` PDFs
- 🧠 **How the neural network is designed?** → See `model-training/Deep_Learning_Respiratory_Classification.pdf`
- 📊 **How COPD severity is classified?** → See `classification/Acoustic_COPD_Severity_Grading.pdf`

---

## 📚 Project Overview

For a high-level understanding, refer to:
- [`../../wiki/Project-Overview.md`](../../wiki/Project-Overview.md)
- [`../../wiki/Architecture.md`](../../wiki/Architecture.md)
- [`../../README.md`](../../README.md)

---

## 🔗 Related Sections

| Topic | Location |
|-------|----------|
| Installation & Setup | [`../../wiki/Installation.md`](../../wiki/Installation.md) |
| Model Training | [`../../wiki/Model-Training.md`](../../wiki/Model-Training.md) |
| Results & Performance | [`../../wiki/Results.md`](../../wiki/Results.md) |
| Dataset Information | [`../../wiki/Dataset.md`](../../wiki/Dataset.md) |
| Usage Guide | [`../../wiki/Usage.md`](../../wiki/Usage.md) |

---

## 💡 How to Use These Documents

1. **For Learning:** Start with the overview in `wiki/Project-Overview.md`, then dive into the specific PDF for your topic
2. **For Implementation:** Cross-reference the PDFs with the actual code in `src/` folder
3. **For Understanding Flow:** Follow the data pipeline documentation from preprocessing → feature extraction → training → evaluation
4. **For Troubleshooting:** Check the specific module PDF for implementation details

---

**Last Updated:** April 29, 2026
